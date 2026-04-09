"""
services/article_crawler.py
────────────────────────────────────────────────────────────────────────────────
뉴스 기사 originallink에서 실제 본문을 크롤링하여 추출한다.

[왜 필요한가]
네이버 뉴스 API가 반환하는 description은 2~3줄 요약(80~150자)에 불과하다.
이 짧은 텍스트로는 공식 표현("밝혔다", "보도자료를 통해", "공식 입장") 탐지가
불가능한 경우가 많다.

본문 전체를 가져오면:
  - rule_based_scorer: 본문에서 공식 표현을 직접 찾을 수 있음
  - semantic_similarity: 2줄 요약 대신 본문 전체로 기준 문장과 비교
  - classifier: 더 풍부한 텍스트로 분류 정확도 향상
  - feature_extractor: 기관명, 인용문, 수치 등 추출 정확도 향상

[크롤링 전략]
1순위: <article> 태그 내부 <p> 태그 텍스트 결합
2순위: id/class에 "article", "content", "body" 포함된 div 내부 <p>
3순위: og:description 메타태그 (description보다는 긴 경우가 많음)
4순위: 전체 <p> 태그 중 길이 기준 상위 텍스트 결합
폴백: 기존 description 유지 (크롤링 실패 시)

[v2 변경: 병렬 크롤링]
기존: for문 순차 처리 → 134건 × 1.3초 = 약 3분
변경: ThreadPoolExecutor(max_workers=10) 병렬 처리 → 134건 ÷ 10 = 약 15~20초

왜 ThreadPoolExecutor인가:
  - 크롤링은 I/O 바운드(네트워크 대기) 작업이므로 스레드 풀이 적합
  - CPU 바운드가 아니라 GIL(Global Interpreter Lock) 영향 없음
  - 기존 requests 코드를 그대로 사용 가능 (asyncio+aiohttp는 전면 수정 필요)
  - 서버 분리 불필요 — 같은 프로세스 안에서 동시 요청만 늘리는 구조

max_workers=10 근거:
  - 10개 동시 요청은 대부분의 언론사 서버에 부담 없는 수준
  - 20개 이상은 일부 서버에서 429(Too Many Requests) 위험
  - 10개로도 134건 기준 3분→15초 수준으로 충분히 빠름

[제한]
- 일부 언론사는 크롤링 차단 (403/429 응답) → 폴백 처리
- JavaScript 렌더링 필요한 사이트는 추출 불가 → 폴백 처리
- robots.txt 준수는 하지 않지만, max_workers 제한으로 서버 부하 최소화
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from logger import get_logger

logger = get_logger("article_crawler")

# ─────────────────────────────────────────────────────────────
# 크롤링 설정
# ─────────────────────────────────────────────────────────────

# HTTP 요청 타임아웃 (초)
CRAWL_TIMEOUT = 5

# 병렬 크롤링 동시 워커 수
# 10개 동시 요청 = 대부분의 언론사 서버에 안전한 수준
# 20개 이상은 429 에러 위험 → 10개로 고정
MAX_WORKERS = 10

# 크롤링된 본문 최소 길이 (이보다 짧으면 크롤링 실패로 간주)
MIN_CONTENT_LENGTH = 100

# 크롤링된 본문 최대 길이 (너무 길면 모델 입력에 부적합)
MAX_CONTENT_LENGTH = 3000

# User-Agent 헤더 (차단 방지용)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

# 크롤링 차단이 잦은 도메인 (시도하지 않고 바로 폴백)
SKIP_DOMAINS = [
    "n.news.naver.com",   # 네이버 뉴스 내부 링크 (리다이렉트)
    "news.naver.com",     # 네이버 뉴스
    "blog.",              # 블로그
    "cafe.",              # 카페
    "youtube.",           # 유튜브
    "instagram.",         # 인스타그램
]

# 본문 추출 시 제거할 불필요한 텍스트 패턴
NOISE_PATTERNS = [
    r"기자\s*[=:]\s*\S+",                  # 기자 = 홍길동
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]+",  # 이메일
    r"https?://\S+",                        # URL
    r"Copyright\s*©.*",                     # 저작권
    r"무단\s*전재.*금지",                    # 무단 전재 금지
    r"저작권자.*재배포\s*금지",              # 저작권자 재배포 금지
    r"\[.*?뉴스.*?\]",                      # [○○뉴스]
    r"▶.*",                                 # ▶ 관련 기사
    r"◇.*",                                 # ◇ 관련 기사
    r"☞.*",                                 # ☞ 관련 링크
    r"사진\s*[=:]\s*\S+",                   # 사진 = 제공처
    r"출처\s*[=:]\s*\S+",                   # 출처 = 제공처
]


# ─────────────────────────────────────────────────────────────
# HTML 파싱 유틸리티 (외부 라이브러리 없이 정규표현식 기반)
#
# BeautifulSoup을 쓰면 더 정확하지만,
# requirements.txt에 추가 의존성을 넣지 않기 위해
# 정규표현식으로 처리한다.
# 대부분의 한국 뉴스 사이트는 이 방식으로 충분히 추출 가능하다.
# ─────────────────────────────────────────────────────────────

def _remove_tags(html: str) -> str:
    """HTML 태그를 모두 제거하고 텍스트만 반환한다."""
    if not html:
        return ""
    # script, style 태그 내용 제거
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 나머지 태그 제거
    text = re.sub(r"<[^>]+>", " ", text)
    # HTML 엔티티 변환
    for entity, char in {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
        "&#x27;": "'", "&#x2F;": "/",
    }.items():
        text = text.replace(entity, char)
    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_meta_og_description(html: str) -> str:
    """og:description 메타태그에서 내용을 추출한다."""
    # <meta property="og:description" content="..." />
    match = re.search(
        r'<meta[^>]+property\s*=\s*["\']og:description["\'][^>]+content\s*=\s*["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return _remove_tags(match.group(1)).strip()

    # content가 property보다 앞에 오는 경우
    match = re.search(
        r'<meta[^>]+content\s*=\s*["\']([^"\']+)["\'][^>]+property\s*=\s*["\']og:description["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return _remove_tags(match.group(1)).strip()

    return ""


def _extract_article_tag(html: str) -> str:
    """
    <article> 태그 내부의 <p> 태그 텍스트를 결합하여 반환한다.
    한국 언론사 대부분이 기사 본문을 <article> 안에 넣는다.
    """
    # <article ...> ... </article> 추출
    article_match = re.search(
        r"<article[^>]*>(.*?)</article>",
        html,
        re.DOTALL | re.IGNORECASE,
    )
    if not article_match:
        return ""

    article_html = article_match.group(1)
    # article 내부의 <p> 태그 텍스트 추출
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", article_html, re.DOTALL | re.IGNORECASE)
    if not paragraphs:
        # <p> 태그가 없으면 article 전체 텍스트
        return _remove_tags(article_html)

    texts = [_remove_tags(p) for p in paragraphs]
    # 빈 문자열과 짧은 문자열(광고/캡션) 제거
    texts = [t for t in texts if len(t) > 20]
    return " ".join(texts)


def _extract_content_div(html: str) -> str:
    """
    id 또는 class에 "article", "content", "body", "text" 가 포함된
    div 내부의 <p> 태그 텍스트를 결합하여 반환한다.
    <article> 태그가 없는 언론사 대응용.
    """
    # id/class에 관련 키워드가 있는 div 찾기
    patterns = [
        r'<div[^>]+(?:id|class)\s*=\s*["\'][^"\']*(?:article[_-]?(?:body|content|text)|newsct_article|news_body|article_body)[^"\']*["\'][^>]*>(.*?)</div>',
        r'<div[^>]+(?:id|class)\s*=\s*["\'][^"\']*(?:content[_-]?(?:body|text|area)|main[_-]?content)[^"\']*["\'][^>]*>(.*?)</div>',
    ]

    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if match:
            div_html = match.group(1)
            paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", div_html, re.DOTALL | re.IGNORECASE)
            if paragraphs:
                texts = [_remove_tags(p) for p in paragraphs if len(_remove_tags(p)) > 20]
                if texts:
                    return " ".join(texts)
            # <p> 없으면 div 전체 텍스트
            text = _remove_tags(div_html)
            if len(text) > MIN_CONTENT_LENGTH:
                return text

    return ""


def _extract_all_paragraphs(html: str) -> str:
    """
    전체 HTML에서 <p> 태그를 모두 추출하고,
    길이 기준으로 상위 텍스트를 결합한다.
    최후의 폴백 전략.
    """
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL | re.IGNORECASE)
    if not paragraphs:
        return ""

    texts = [_remove_tags(p) for p in paragraphs]
    # 30자 이상인 것만 (광고/캡션 제거)
    texts = [t for t in texts if len(t) > 30]
    # 길이 순으로 정렬하여 상위 10개 (본문 문단이 보통 가장 길다)
    texts.sort(key=len, reverse=True)
    return " ".join(texts[:10])


def _clean_content(text: str) -> str:
    """크롤링된 본문에서 불필요한 패턴을 제거한다."""
    if not text:
        return ""

    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text)

    # 연속 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    # 최대 길이 제한
    if len(text) > MAX_CONTENT_LENGTH:
        text = text[:MAX_CONTENT_LENGTH]

    return text


def _should_skip(url: str) -> bool:
    """크롤링을 건너뛸 도메인인지 확인한다."""
    if not url:
        return True
    url_lower = url.lower()
    for domain in SKIP_DOMAINS:
        if domain in url_lower:
            return True
    return False


# ─────────────────────────────────────────────────────────────
# 단일 기사 본문 크롤링
# ─────────────────────────────────────────────────────────────

def crawl_article_content(url: str) -> str:
    """
    단일 URL에서 기사 본문을 크롤링하여 반환한다.

    추출 우선순위:
      1. <article> 태그 내부 <p>
      2. id/class에 article/content/body 포함된 div 내부 <p>
      3. og:description 메타태그
      4. 전체 <p> 태그 상위 텍스트

    이 함수는 ThreadPoolExecutor에서 병렬로 호출된다.
    따라서 내부에서 공유 상태를 수정하지 않는다 (thread-safe).

    Args:
        url: 기사 원문 URL (originallink)
    Returns:
        추출된 본문 텍스트. 실패 시 빈 문자열 반환.
    """
    if _should_skip(url):
        return ""

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=CRAWL_TIMEOUT,
            allow_redirects=True,
        )
        response.raise_for_status()

        # 인코딩 자동 감지 (한국 언론사는 euc-kr 또는 utf-8)
        if response.encoding and "euc" in response.encoding.lower():
            html = response.content.decode("euc-kr", errors="replace")
        else:
            html = response.text

        if not html or len(html) < 500:
            return ""

        # 1순위: <article> 태그
        content = _extract_article_tag(html)
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return _clean_content(content)

        # 2순위: content/article div
        content = _extract_content_div(html)
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return _clean_content(content)

        # 3순위: og:description (description보다 긴 경우가 많음)
        content = _extract_meta_og_description(html)
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return _clean_content(content)

        # 4순위: 전체 <p> 태그
        content = _extract_all_paragraphs(html)
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return _clean_content(content)

        return ""

    except requests.exceptions.Timeout:
        # 타임아웃은 흔하므로 debug 레벨로 로그
        logger.debug(f"크롤링 타임아웃: {url}")
        return ""
    except requests.exceptions.HTTPError as e:
        # 403/429 등 차단 응답
        status_code = e.response.status_code if e.response else "unknown"
        logger.debug(f"크롤링 HTTP 에러: {status_code} | {url}")
        return ""
    except Exception as e:
        logger.debug(f"크롤링 실패: {e} | {url}")
        return ""


# ─────────────────────────────────────────────────────────────
# 단일 기사 크롤링 워커 (ThreadPoolExecutor에서 호출)
#
# 왜 별도 함수인가:
# ThreadPoolExecutor.submit()에 넘기려면
# (인덱스, 기사) 튜플을 받아서 (인덱스, 결과)를 반환하는
# 래퍼 함수가 필요하다.
# 인덱스를 함께 반환해야 원래 기사 순서를 유지할 수 있다.
# ─────────────────────────────────────────────────────────────

def _crawl_single_worker(index: int, article: dict) -> tuple:
    """
    ThreadPoolExecutor에서 호출되는 단일 기사 크롤링 워커.

    Args:
        index:   기사 리스트에서의 인덱스 (순서 유지용)
        article: 기사 딕셔너리
    Returns:
        (index, crawled_content) 튜플
        crawled_content가 빈 문자열이면 크롤링 실패
    """
    url = article.get("originallink", "") or article.get("link", "")
    crawled_content = crawl_article_content(url)
    return (index, crawled_content)


# ─────────────────────────────────────────────────────────────
# 배치 크롤링 (병렬 처리)
#
# [v2 변경: 순차 → 병렬]
# 기존: for문으로 1건씩 요청 + 0.3초 딜레이
#   → 134건 × 1.3초 = 약 174초 (3분)
#
# 변경: ThreadPoolExecutor(max_workers=10)로 10건 동시 요청
#   → 134건 ÷ 10워커 × 1초 = 약 15초
#   → 딜레이 제거 (동시 요청이라 서버별로 분산됨)
#
# 왜 딜레이를 제거해도 되는가:
#   순차 처리에서 딜레이는 한 서버에 연속 요청을 방지하기 위함.
#   병렬 처리에서는 10개 요청이 서로 다른 언론사 서버로 분산되므로
#   같은 서버에 동시 10개가 가는 경우는 드물다.
#   만약 같은 서버에 집중되더라도 max_workers=10 수준은 안전하다.
# ─────────────────────────────────────────────────────────────

def crawl_articles_batch(articles: list) -> list:
    """
    기사 목록의 originallink에서 본문을 병렬 크롤링하여
    각 기사의 content(description)를 본문으로 교체한다.

    크롤링 성공 시: content를 크롤링된 본문으로 교체
    크롤링 실패 시: 기존 description(API 반환 요약) 유지

    병렬 처리 방식:
      ThreadPoolExecutor(max_workers=10)로 최대 10건 동시 요청.
      완료되는 순서대로 결과를 수집하고, 인덱스로 원래 기사에 매핑.

    Args:
        articles: 네이버 API에서 수집한 기사 딕셔너리 리스트
                  각 기사에 "originallink"와 "description" 키가 있어야 함
    Returns:
        본문이 보강된 기사 리스트 (원본 리스트를 수정하여 반환)
    """
    if not articles:
        logger.warning("크롤링 입력 데이터가 비어 있음")
        return articles

    total = len(articles)
    success_count = 0
    skip_count = 0
    fail_count = 0

    logger.info(f"본문 크롤링 시작 | {total}건 (병렬 {MAX_WORKERS}워커)")

    # ── 크롤링 대상 / 스킵 대상 분리 ────────────────────────
    # 스킵 도메인 기사는 미리 걸러서 불필요한 스레드 생성 방지
    crawl_targets = []  # (index, article) 튜플 리스트
    for i, article in enumerate(articles):
        url = article.get("originallink", "") or article.get("link", "")
        if _should_skip(url):
            # 스킵 대상: content_source만 표시하고 넘어감
            article["content_source"] = "skipped"
            skip_count += 1
        else:
            crawl_targets.append((i, article))

    # ── 병렬 크롤링 실행 ─────────────────────────────────────
    # ThreadPoolExecutor로 크롤링 대상만 동시 요청
    # as_completed()를 사용하여 완료되는 순서대로 결과 처리
    # (전부 완료될 때까지 블로킹하지 않고 중간 진행률 출력 가능)

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # submit: 각 크롤링 작업을 스레드 풀에 제출
        # future → (index, article) 매핑을 유지하여 결과를 원본에 반영
        future_to_index = {
            executor.submit(_crawl_single_worker, idx, article): (idx, article)
            for idx, article in crawl_targets
        }

        # as_completed: 완료되는 순서대로 결과를 수집
        for future in as_completed(future_to_index):
            idx, article = future_to_index[future]
            completed += 1

            try:
                # _crawl_single_worker의 반환값: (index, crawled_content)
                _, crawled_content = future.result()

                if crawled_content and len(crawled_content) >= MIN_CONTENT_LENGTH:
                    # 크롤링 성공: content를 본문으로 교체
                    # 원본 description은 별도 키로 보존 (나중에 비교용)
                    article["original_description"] = article.get("description", "")
                    article["description"] = crawled_content
                    article["content_source"] = "crawled"
                    success_count += 1
                else:
                    # 크롤링 실패: 기존 description 유지
                    article["content_source"] = "api_description"
                    fail_count += 1

            except Exception as exc:
                # 스레드 내부에서 잡히지 않은 예외 (극히 드묾)
                logger.debug(f"크롤링 워커 예외 | idx={idx} | {exc}")
                article["content_source"] = "api_description"
                fail_count += 1

            # 진행률 로그 (50건마다)
            if completed % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"크롤링 진행 | {completed}/{len(crawl_targets)}건 "
                    f"(성공={success_count} 실패={fail_count}) "
                    f"경과={elapsed:.1f}초"
                )

    # ── 완료 로그 ────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(
        f"본문 크롤링 완료 | 총 {total}건 | "
        f"성공={success_count}건 ({success_count * 100 // max(total, 1)}%) | "
        f"실패={fail_count}건 | 스킵={skip_count}건 | "
        f"소요시간={elapsed:.1f}초"
    )

    return articles


# ─────────────────────────────────────────────────────────────
# 단독 실행 테스트
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python -m services.article_crawler <URL>")
        print("예: python -m services.article_crawler https://www.yna.co.kr/view/...")
        sys.exit(1)

    test_url = sys.argv[1]
    print(f"크롤링 대상: {test_url}")
    print("-" * 60)

    result = crawl_article_content(test_url)

    if result:
        print(f"[성공] 본문 길이: {len(result)}자")
        print("-" * 60)
        # 앞 500자만 출력
        print(result[:500])
        if len(result) > 500:
            print(f"\n... (총 {len(result)}자)")
    else:
        print("[실패] 본문 추출 불가")
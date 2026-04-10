"""
news_search.py
──────────────
네이버 뉴스 검색 API를 호출하여 기사 목록을 수집한다.
환경변수에서 API 키를 읽고, 응답을 정제된 리스트로 반환한다.

[v2 변경사항 - 쿼리 확장]
1. search_news_expanded() 신규 추가
   - 단일 키워드 → 의도 기반 보강 쿼리 다수로 자동 확장
   - 예: "악뮤" → "악뮤", "악뮤 최근", "악뮤 발표", "악뮤 공식", "악뮤 활동"
   - 예: "클로드 유출" → "클로드 유출", "클로드", "유출", "클로드 유출 공식", ...
   - 토큰 수에 따라 확장 전략 분기 (1토큰/2토큰/3토큰+)

2. generate_query_candidates() 신규 추가
   - 수집 없이 확장 쿼리 후보 리스트만 반환
   - 서버/UI가 사용자에게 "이 후보들로 검색 범위를 넓힐까요?" 제안할 때 활용

3. 기존 함수 전부 유지
   - search_news, search_news_combined, _deduplicate, _filter_by_keyword

[사용 예]
  # 기본 수집 (기존)
  articles = search_news_combined("삼성전자")

  # 확장 수집 (v2 신규)
  articles = search_news_expanded("악뮤")
  # → 내부적으로 "악뮤", "악뮤 최근", "악뮤 발표" 등 여러 쿼리 실행

  # 후보만 받아보기
  candidates = generate_query_candidates("악뮤")
  # → ["악뮤", "악뮤 최근", "악뮤 발표", ...]
"""

import sys
import re
import requests
import os
from dotenv import load_dotenv
from logger import get_logger
from config import (
    NEWS_DISPLAY_PER_REQUEST,
    NEWS_DEFAULT_SORT,
    NAVER_API_URL,
    NAVER_API_TIMEOUT,
    NAVER_API_HEADER_ID_KEY,
    NAVER_API_HEADER_SECRET_KEY,
    KEYWORD_FILTER_ENABLED,
    KEYWORD_FILTER_MIN_MATCH,
)

load_dotenv()

logger = get_logger("news_search")


# ═════════════════════════════════════════════════════════════
# v2 신규: 쿼리 확장 설정
# ═════════════════════════════════════════════════════════════

# 의도 기반 보강 템플릿 (단일/2토큰 쿼리에 부착)
# 카테고리별로 묶어서 관리, {q}는 원본 쿼리로 치환됨
_INTENT_TEMPLATES = {
    # 최신성: "악뮤" → "악뮤 최근", "악뮤 근황"
    "recent":  ["{q} 최근", "{q} 근황"],
    # 활동/계획: "악뮤" → "악뮤 활동", "악뮤 일정"
    "action":  ["{q} 활동", "{q} 일정", "{q} 소식"],
    # 공식성 보강: "악뮤" → "악뮤 공식", "악뮤 발표"
    "official": ["{q} 공식", "{q} 발표", "{q} 보도자료"],
}

# 확장 시 원본 쿼리에 병합할 boost 키워드
# (2~3토큰 쿼리에 하나씩 추가로 덧붙이는 용도)
_BOOST_KEYWORDS = ["공식", "발표", "보도자료", "공식 입장"]

# 확장 쿼리 최대 개수 (원본 포함)
_MAX_EXPANDED_QUERIES = 6


# ═════════════════════════════════════════════════════════════
# 기존 함수 (v1 유지)
# ═════════════════════════════════════════════════════════════

def search_news(query: str, display: int = None, sort: str = None) -> list[dict]:
    """
    네이버 뉴스 검색 API를 1회 호출하여 기사 목록을 반환한다.
    """
    display = display or NEWS_DISPLAY_PER_REQUEST
    sort = sort or NEWS_DEFAULT_SORT

    client_id = os.getenv("naver_client_id")
    client_secret = os.getenv("naver_client_secret")

    if not client_id or not client_secret:
        raise ValueError(".env 파일에 naver_client_id / naver_client_secret 가 없음")

    headers = {
        NAVER_API_HEADER_ID_KEY: client_id,
        NAVER_API_HEADER_SECRET_KEY: client_secret,
    }
    params = {"query": query, "display": display, "sort": sort}

    logger.info(f"뉴스 수집 시작 | query={query} display={display} sort={sort}")

    try:
        res = requests.get(
            NAVER_API_URL,
            headers=headers,
            params=params,
            timeout=NAVER_API_TIMEOUT,
        )
        res.raise_for_status()
    except requests.exceptions.Timeout:
        raise ConnectionError(f"네이버 API 요청 타임아웃 ({NAVER_API_TIMEOUT}초 초과)")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"네이버 API 요청 실패: {e}")

    try:
        data = res.json()
    except Exception as e:
        raise ValueError(f"API 응답 JSON 파싱 실패: {e}")

    if "items" not in data:
        raise ValueError(f"API 응답에 'items' 키 없음. 응답 내용: {data}")

    if not data["items"]:
        logger.warning(f"검색 결과 없음 | query={query}")
        return []

    articles = []
    for item in data["items"]:
        articles.append({
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "originallink": item.get("originallink", ""),
            "link": item.get("link", ""),
            "pubDate": item.get("pubDate", ""),
        })

    logger.info(f"뉴스 수집 완료 | {len(articles)}건 수집됨")
    return articles


def _remove_html_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _filter_by_keyword(articles: list[dict], query: str) -> list[dict]:
    if not KEYWORD_FILTER_ENABLED:
        return articles

    tokens = query.strip().split()
    if not tokens:
        return articles

    filtered = []
    for article in articles:
        title = _remove_html_tags(article.get("title", "")).lower()
        desc = _remove_html_tags(article.get("description", "")).lower()
        combined = title + " " + desc
        match_count = sum(1 for token in tokens if token.lower() in combined)
        if match_count >= min(KEYWORD_FILTER_MIN_MATCH, len(tokens)):
            filtered.append(article)

    logger.info(
        f"키워드 필터링 | 원본={len(articles)}건 → 통과={len(filtered)}건 "
        f"(키워드: {tokens})"
    )
    return filtered


def _deduplicate(articles: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for article in articles:
        key = article.get("originallink") or article.get("link", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(article)
        elif not key:
            unique.append(article)

    removed = len(articles) - len(unique)
    if removed > 0:
        logger.info(
            f"중복 제거 | 원본={len(articles)}건 → 고유={len(unique)}건 "
            f"(중복 {removed}건 제거)"
        )
    return unique


def search_news_combined(query: str, display: int = None) -> list[dict]:
    """
    sim(관련도) + date(최신) 이중 수집 후 중복 제거 + 키워드 필터.
    """
    display = display or NEWS_DISPLAY_PER_REQUEST

    logger.info(f"이중 수집 시작 | query={query} display={display} (sim + date)")

    try:
        sim_articles = search_news(query=query, display=display, sort="sim")
    except (ValueError, ConnectionError) as e:
        logger.error(f"sim 수집 실패: {e}")
        sim_articles = []

    try:
        date_articles = search_news(query=query, display=display, sort="date")
    except (ValueError, ConnectionError) as e:
        logger.error(f"date 수집 실패: {e}")
        date_articles = []

    combined = sim_articles + date_articles

    if not combined:
        logger.warning(f"이중 수집 결과 없음 | query={query}")
        return []

    logger.info(
        f"이중 수집 완료 | sim={len(sim_articles)}건 + date={len(date_articles)}건 "
        f"= 총 {len(combined)}건"
    )

    unique = _deduplicate(combined)
    filtered = _filter_by_keyword(unique, query)

    logger.info(
        f"최종 수집 결과 | {len(filtered)}건 "
        f"(원본 {len(combined)}건 → 중복제거 {len(unique)}건 → 필터 {len(filtered)}건)"
    )
    return filtered


# ═════════════════════════════════════════════════════════════
# v2 신규: 쿼리 확장
# ═════════════════════════════════════════════════════════════

def generate_query_candidates(query: str, max_count: int = _MAX_EXPANDED_QUERIES) -> list[str]:
    """
    [v2 신규] 원본 쿼리에서 확장 쿼리 후보 리스트를 생성한다.

    수집을 수행하지 않고 후보만 반환하므로
    서버/UI에서 "이 후보들로 검색할까요?" 사용자 제안에 활용 가능.

    확장 전략 (토큰 수 기반):
      1토큰 (예: "악뮤")
        → 의도 템플릿 위주로 확장
        → [악뮤, 악뮤 최근, 악뮤 발표, 악뮤 공식, 악뮤 활동, 악뮤 근황]

      2토큰 (예: "클로드 유출")
        → 원본 + 개별 토큰 + boost 키워드
        → [클로드 유출, 클로드, 유출, 클로드 유출 공식, 클로드 유출 발표, 클로드 유출 보도자료]

      3토큰 이상 (예: "이재명 경기지사 출마")
        → 원본 + 부분 조합 + boost 키워드
        → [이재명 경기지사 출마, 이재명 경기지사, 경기지사 출마, 이재명 출마, ...]

    Args:
        query:     원본 검색어
        max_count: 반환 후보 최대 개수
    Returns:
        확장 쿼리 리스트 (첫 번째는 항상 원본)
    """
    query = query.strip()
    if not query:
        return []

    tokens = query.split()
    candidates = [query]  # 원본은 항상 첫 번째

    if len(tokens) == 1:
        # ── 1토큰: 의도 템플릿 확장 ─────────────────────
        # 공식성 + 최신성 + 활동 순서로 다양성 확보
        for template in _INTENT_TEMPLATES["official"]:
            candidates.append(template.format(q=query))
        for template in _INTENT_TEMPLATES["recent"]:
            candidates.append(template.format(q=query))
        for template in _INTENT_TEMPLATES["action"]:
            candidates.append(template.format(q=query))

    elif len(tokens) == 2:
        # ── 2토큰: 개별 토큰 + boost 키워드 ────────────
        # 개별 토큰 (각각 단독 검색)
        candidates.append(tokens[0])
        candidates.append(tokens[1])
        # 원본 + boost
        for boost in _BOOST_KEYWORDS:
            candidates.append(f"{query} {boost}")

    else:
        # ── 3토큰 이상: 부분 조합 + boost ──────────────
        # 인접 2토큰 조합
        for i in range(len(tokens) - 1):
            candidates.append(f"{tokens[i]} {tokens[i+1]}")
        # 첫 토큰 + 마지막 토큰
        candidates.append(f"{tokens[0]} {tokens[-1]}")
        # 원본 + boost (2개만)
        for boost in _BOOST_KEYWORDS[:2]:
            candidates.append(f"{query} {boost}")

    # 중복 제거 (순서 유지)
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean not in seen:
            seen.add(c_clean)
            unique_candidates.append(c_clean)

    return unique_candidates[:max_count]


def search_news_expanded(
    query: str,
    display: int = None,
    max_variants: int = _MAX_EXPANDED_QUERIES,
) -> list[dict]:
    """
    [v2 신규] 쿼리 확장 후 네이버 뉴스 API 다중 호출로 수집량을 늘린다.

    처리 흐름:
      1. generate_query_candidates() 로 확장 쿼리 후보 생성
      2. 각 후보에 대해 search_news_combined() 호출 (sim+date 이중)
      3. 전체 결과를 합치고 originallink 기준 중복 제거
      4. 원본 쿼리 기준 키워드 필터링 (옵션)

    API 호출량:
      - 후보 N개 × sim/date 2회 = 최대 2N회
      - 기본값 N=6 → 최대 12회
      - 네이버 free API 일일 한도(25,000건) 대비 무시할 수준

    Args:
        query:        원본 검색어
        display:      후보 쿼리당 수집 건수
        max_variants: 확장 후보 최대 개수 (원본 포함)
    Returns:
        통합 수집된 기사 리스트 (중복 제거됨)
    """
    query = query.strip()
    if not query:
        logger.warning("search_news_expanded: 빈 쿼리")
        return []

    display = display or NEWS_DISPLAY_PER_REQUEST

    # 1단계: 확장 쿼리 후보 생성
    candidates = generate_query_candidates(query, max_count=max_variants)
    logger.info(
        f"[쿼리 확장] 원본='{query}' → {len(candidates)}개 후보 생성: {candidates}"
    )

    # 2단계: 각 후보에 대해 이중 수집
    all_articles = []
    for i, variant in enumerate(candidates, 1):
        logger.info(f"[확장 수집 {i}/{len(candidates)}] query='{variant}'")
        try:
            articles = search_news_combined(query=variant, display=display)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"확장 수집 실패 | query='{variant}' | {e}")
            continue

    if not all_articles:
        logger.warning(f"확장 수집 결과 없음 | query='{query}'")
        return []

    # 3단계: 전체 중복 제거 (originallink 기준)
    unique = _deduplicate(all_articles)

    # 4단계: 원본 쿼리 기준 키워드 필터링 (config에서 켜진 경우)
    filtered = _filter_by_keyword(unique, query)

    logger.info(
        f"[확장 수집 완료] 후보={len(candidates)}개 | "
        f"원본 총합={len(all_articles)}건 → 중복제거={len(unique)}건 → 최종={len(filtered)}건"
    )

    return filtered


if __name__ == "__main__":
    # 사용법: python news_search.py 악뮤
    # 사용법: python news_search.py "클로드 유출" --expand
    test_query = sys.argv[1] if len(sys.argv) > 1 else "뉴스"
    use_expand = "--expand" in sys.argv

    if use_expand:
        print(f"\n=== 확장 수집 테스트: '{test_query}' ===\n")
        print("생성된 확장 후보:")
        for c in generate_query_candidates(test_query):
            print(f"  - {c}")
        print()
        result = search_news_expanded(test_query)
    else:
        print(f"\n=== 이중 수집 테스트: '{test_query}' ===\n")
        result = search_news_combined(test_query)

    for i, item in enumerate(result, 1):
        title = _remove_html_tags(item["title"])
        print(f"[{i}] {title} | {item['originallink']}")
    print(f"\n총 {len(result)}건")
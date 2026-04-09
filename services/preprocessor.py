# 텍스트 정리 담당 : 모델에 넣기 전에 텍스트를 깨끗하게 만드는 단계

"""
services/preprocessor.py
─────────────────────────
수집된 뉴스 기사의 원시 텍스트를 정제한다.
HTML 태그 제거, 특수문자 정리, 공백 정규화를 수행한다.
"""

import re
from urllib.parse import urlparse
from logger import get_logger

logger = get_logger("preprocessor")


def remove_html_tags(text: str) -> str:
    """
    HTML 태그를 제거한다.
    네이버 API 응답의 title/description에는 <b>, </b> 같은 태그가 포함됨.
    
    Args:
        text: 원시 텍스트
    Returns:
        HTML 태그가 제거된 텍스트
    """
    return re.sub(r"<[^>]+>", "", text)


def clean_special_characters(text: str) -> str:
    """
    HTML 엔티티 및 불필요한 특수문자를 정리한다.
    예: &amp; → &, &lt; → <, &quot; → "
    
    Args:
        text: HTML 태그 제거된 텍스트
    Returns:
        특수문자가 정리된 텍스트
    """
    # HTML 엔티티 변환
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")

    # 줄바꿈/탭을 공백으로 통일
    text = re.sub(r"[\r\n\t]+", " ", text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    연속된 공백을 하나로 줄이고 앞뒤 공백을 제거한다.
    
    Args:
        text: 정리된 텍스트
    Returns:
        공백이 정규화된 텍스트
    """
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """
    텍스트 전처리 전체 파이프라인.
    HTML 제거 → 특수문자 정리 → 공백 정규화 순으로 처리한다.
    
    Args:
        text: 원시 텍스트
    Returns:
        정제된 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    text = remove_html_tags(text)
    text = clean_special_characters(text)
    text = normalize_whitespace(text)
    return text


def extract_domain(url: str) -> str:
    """
    URL에서 도메인을 추출한다.
    예: https://mosf.go.kr/news/123 → mosf.go.kr
    
    기관명/출처 분석에 사용된다.
    
    Args:
        url: 기사 원문 URL
    Returns:
        도메인 문자열 (파싱 실패 시 빈 문자열)
    """
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower()
    except Exception:
        return ""


def preprocess_article(article: dict) -> dict:
    """
    단일 기사 딕셔너리 전체를 전처리한다.
    title, description을 정제하고 domain을 추출하여 추가한다.
    
    Args:
        article: news_search.py에서 반환된 기사 딕셔너리
    Returns:
        전처리된 기사 딕셔너리
    """
    try:
        domain = extract_domain(article.get("originallink", ""))
        cleaned = {
            "title":        clean_text(article.get("title", "")),
            "content":      clean_text(article.get("description", "")),
            "source":       extract_source_name(domain),
            "originallink": article.get("originallink", ""),
            "link":         article.get("link", ""),
            "pubDate":      article.get("pubDate", ""),
            "domain":       domain,
        }
        return cleaned
    except Exception as e:
        logger.error(f"기사 전처리 실패: {e} | article={article}")
        return {}


def preprocess_articles(articles: list[dict]) -> list[dict]:
    """
    기사 목록 전체를 전처리한다.
    빈 결과나 실패한 기사는 제외한다.
    
    Args:
        articles: 수집된 기사 딕셔너리 리스트
    Returns:
        전처리된 기사 딕셔너리 리스트
    """
    if not articles:
        logger.warning("전처리 입력 데이터가 비어 있음")
        return []

    logger.info(f"전처리 시작 | {len(articles)}건")

    results = []
    for article in articles:
        cleaned = preprocess_article(article)
        # title 또는 content가 비어있는 기사는 제외
        if cleaned and cleaned.get("title"):
            results.append(cleaned)

    logger.info(f"전처리 완료 | {len(results)}건 유효")
    return results

# ─────────────────────────────────────────────────────────────
# 도메인 → 언론사명 매핑 테이블
#
# 설계 원칙:
# 네이버 뉴스 API 응답에는 source 필드가 없음.
# originallink의 도메인을 파싱하여 언론사명을 역매핑한다.
# 매핑에 없는 도메인은 도메인 자체를 언론사명으로 사용.
# ─────────────────────────────────────────────────────────────
DOMAIN_TO_SOURCE = {
    # 통신사
    "yna.co.kr":           "연합뉴스",
    "yonhapnewstv.co.kr":  "연합뉴스TV",
    "newsis.com":          "뉴시스",
    "news1.kr":            "뉴스1",
    "newspim.com":         "뉴스핌",
    # 종합일간지
    "chosun.com":          "조선일보",
    "joongang.co.kr":      "중앙일보",
    "donga.com":           "동아일보",
    "hani.co.kr":          "한겨레",
    "khan.co.kr":          "경향신문",
    "kmib.co.kr":          "국민일보",
    "seoul.co.kr":         "서울신문",
    "munhwa.com":          "문화일보",
    "hankookilbo.com":     "한국일보",
    "heraldcorp.com":      "헤럴드경제",
    # 경제지
    "hankyung.com":        "한국경제",
    "mk.co.kr":            "매일경제",
    "edaily.co.kr":        "이데일리",
    "mt.co.kr":            "머니투데이",
    "sedaily.com":         "서울경제",
    "etnews.com":          "전자신문",
    "zdnet.co.kr":         "지디넷코리아",
    "bloter.net":          "블로터",
    "financialpost.co.kr": "파이낸셜포스트",
    # 방송
    "kbs.co.kr":           "KBS",
    "mbc.co.kr":           "MBC",
    "sbs.co.kr":           "SBS",
    "ytn.co.kr":           "YTN",
    "mbn.co.kr":           "MBN",
    "jtbc.co.kr":          "JTBC",
    "tvchosun.com":        "TV조선",
    "channela.com":        "채널A",
    # 인터넷/전문
    "pressian.com":        "프레시안",
    "ohmynews.com":        "오마이뉴스",
    "mediatoday.co.kr":    "미디어오늘",
    "sisain.co.kr":        "시사IN",
    "weekly.khan.co.kr":   "주간경향",
    "asiae.co.kr":         "아시아경제",
    "ajunews.com":         "아주경제",
    "inews24.com":         "아이뉴스24",
    "metroseoul.co.kr":    "메트로신문",
    "spotvnews.co.kr":     "스포TV뉴스",
    "sports.chosun.com":   "스포츠조선",
    "mydaily.co.kr":       "마이데일리",
    "isplus.com":          "일간스포츠",
}


def extract_source_name(domain: str) -> str:
    """
    도메인에서 언론사명을 추출한다.

    매핑 테이블에 있으면 한글 언론사명 반환.
    없으면 도메인 자체를 반환 (www. 제거 후).

    Args:
        domain: extract_domain()으로 추출한 도메인 문자열
    Returns:
        언론사명 문자열
    """
    if not domain:
        return ""

    # www. 제거 후 매핑 테이블 조회
    clean_domain = domain.replace("www.", "").replace("m.", "")

    # 정확히 일치하는 도메인 먼저 확인
    for key, name in DOMAIN_TO_SOURCE.items():
        if key in clean_domain:
            return name

    # 매핑 없으면 도메인 자체 반환
    return clean_domain
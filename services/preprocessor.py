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
        cleaned = {
            "title": clean_text(article.get("title", "")),
            "content": clean_text(article.get("description", "")),
            "originallink": article.get("originallink", ""),
            "link": article.get("link", ""),
            "pubDate": article.get("pubDate", ""),
            "domain": extract_domain(article.get("originallink", "")),
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
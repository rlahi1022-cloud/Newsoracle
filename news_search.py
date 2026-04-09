"""
news_search.py
──────────────
네이버 뉴스 검색 API를 호출하여 기사 목록을 수집한다.
환경변수에서 API 키를 읽고, 응답을 정제된 리스트로 반환한다.

[변경 이력]
- NAVER_API dict 제거 → config의 개별 상수로 교체
- URL, 타임아웃, 헤더 키 이름 하드코딩 제거 → config에서 관리
- __main__ 테스트 쿼리 하드코딩 제거 → sys.argv로 CLI 인자 수신
- search_news_combined() 추가: sim + date 이중 수집 → 중복 제거 → 키워드 필터링
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


def search_news(query: str, display: int = None, sort: str = None) -> list[dict]:
    """
    네이버 뉴스 검색 API를 1회 호출하여 기사 목록을 반환한다.

    Args:
        query:   검색 쿼리 문자열
        display: 가져올 기사 수 (기본값: config의 NEWS_DISPLAY_PER_REQUEST)
        sort:    정렬 방식 - "date"(최신순) / "sim"(관련도순)

    Returns:
        기사 정보 딕셔너리 리스트

    Raises:
        ValueError:      API 키 누락 또는 응답 구조 이상
        ConnectionError: 네트워크 요청 실패
    """
    display = display or NEWS_DISPLAY_PER_REQUEST
    sort = sort or NEWS_DEFAULT_SORT

    # API 키 누락 방어
    client_id = os.getenv("naver_client_id")
    client_secret = os.getenv("naver_client_secret")

    if not client_id or not client_secret:
        raise ValueError(".env 파일에 naver_client_id / naver_client_secret 가 없음")

    headers = {
        NAVER_API_HEADER_ID_KEY: client_id,
        NAVER_API_HEADER_SECRET_KEY: client_secret,
    }
    params = {
        "query": query,
        "display": display,
        "sort": sort,
    }

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

    # 프로젝트에서 사용할 필드만 추출하여 반환
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
    """
    HTML 태그를 제거한다.
    네이버 API 응답의 title/description에 <b></b> 등이 포함되어 있어
    키워드 매칭 전에 제거해야 정확한 비교가 가능하다.
    """
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _filter_by_keyword(articles: list[dict], query: str) -> list[dict]:
    """
    검색 키워드가 제목 또는 본문에 포함된 기사만 남긴다.

    쿼리를 공백 기준으로 분리하여 각 토큰이 제목이나 description에
    최소 KEYWORD_FILTER_MIN_MATCH개 이상 포함되어야 통과한다.

    예: "이란 전쟁" → ["이란", "전쟁"]
        제목에 "이란" 포함 → 통과 (MIN_MATCH=1)
        제목에 "재생에너지"만 → 탈락
    """
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
    """
    originallink 기준으로 중복 기사를 제거한다.
    sim과 date 결과를 합칠 때 같은 기사가 양쪽에 포함될 수 있으므로
    먼저 수집된 것(sim 우선)을 유지한다.
    """
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
        logger.info(f"중복 제거 | 원본={len(articles)}건 → 고유={len(unique)}건 (중복 {removed}건 제거)")

    return unique


def search_news_combined(query: str, display: int = None) -> list[dict]:
    """
    sim(관련도순) + date(최신순) 이중 수집 후 중복 제거 + 키워드 필터링을 적용한다.

    왜 이중 수집이 필요한가:
      sort=date만 사용 → 관련 없는 기사가 다수 섞임
      sort=sim만 사용 → 최신 기사를 놓칠 수 있음

    처리 흐름:
      1. sort=sim으로 display건 수집
      2. sort=date로 display건 수집
      3. 합침 (sim 우선)
      4. originallink 기준 중복 제거
      5. 키워드 필터링
    """
    display = display or NEWS_DISPLAY_PER_REQUEST

    logger.info(f"이중 수집 시작 | query={query} display={display} (sim + date)")

    # 1단계: sort=sim
    try:
        sim_articles = search_news(query=query, display=display, sort="sim")
    except (ValueError, ConnectionError) as e:
        logger.error(f"sim 수집 실패: {e}")
        sim_articles = []

    # 2단계: sort=date
    try:
        date_articles = search_news(query=query, display=display, sort="date")
    except (ValueError, ConnectionError) as e:
        logger.error(f"date 수집 실패: {e}")
        date_articles = []

    # 3단계: 합치기 (sim 우선)
    combined = sim_articles + date_articles

    if not combined:
        logger.warning(f"이중 수집 결과 없음 | query={query}")
        return []

    logger.info(f"이중 수집 완료 | sim={len(sim_articles)}건 + date={len(date_articles)}건 = 총 {len(combined)}건")

    # 4단계: 중복 제거
    unique = _deduplicate(combined)

    # 5단계: 키워드 필터링
    filtered = _filter_by_keyword(unique, query)

    logger.info(f"최종 수집 결과 | {len(filtered)}건 (원본 {len(combined)}건 → 중복제거 {len(unique)}건 → 필터 {len(filtered)}건)")

    return filtered


if __name__ == "__main__":
    # 사용법: python news_search.py 이재명
    test_query = sys.argv[1] if len(sys.argv) > 1 else "뉴스"
    print(f"\n=== 이중 수집 테스트: '{test_query}' ===\n")
    result = search_news_combined(test_query)
    for i, item in enumerate(result, 1):
        title = _remove_html_tags(item["title"])
        print(f"[{i}] {title} | {item['originallink']}")
    print(f"\n총 {len(result)}건")
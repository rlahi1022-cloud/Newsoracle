"""
news_search.py
──────────────
네이버 뉴스 검색 API를 호출하여 기사 목록을 수집한다.

[v3 변경사항]
- search_news_by_category(query, category) 신규 추가
  · query_expander와 연동하여 카테고리 기반 확장 수집
  · 사용자가 선택한 카테고리(recent/official/action/issue/analysis)에 맞게
    자동 생성된 확장 쿼리로 다중 API 호출
- v2 기능 전부 유지 (search_news_expanded, generate_query_candidates)
- 유료 API 일절 사용 안 함, 네이버 free API만
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

_MAX_EXPANDED_QUERIES = 6


# ═════════════════════════════════════════════════════════════
# 기본 수집 (단일 쿼리)
# ═════════════════════════════════════════════════════════════

def search_news(query: str, display: int = None, sort: str = None) -> list[dict]:
    """네이버 뉴스 API 1회 호출."""
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
        res = requests.get(NAVER_API_URL, headers=headers, params=params, timeout=NAVER_API_TIMEOUT)
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
    """config의 KEYWORD_FILTER_ENABLED가 True일 때만 필터링."""
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

    logger.info(f"키워드 필터링 | 원본={len(articles)}건 → 통과={len(filtered)}건 (키워드: {tokens})")
    return filtered


def _deduplicate(articles: list[dict]) -> list[dict]:
    """originallink 기준 중복 제거."""
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
    """sim + date 이중 수집 후 중복 제거 + 키워드 필터."""
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

    logger.info(f"이중 수집 완료 | sim={len(sim_articles)}건 + date={len(date_articles)}건 = 총 {len(combined)}건")

    unique = _deduplicate(combined)
    filtered = _filter_by_keyword(unique, query)

    logger.info(f"최종 수집 결과 | {len(filtered)}건")
    return filtered


# ═════════════════════════════════════════════════════════════
# v2: 단순 규칙 기반 확장 수집 (기존 유지)
# ═════════════════════════════════════════════════════════════

def generate_query_candidates(query: str, max_count: int = _MAX_EXPANDED_QUERIES) -> list[str]:
    """
    [v2] 간단 규칙 기반 확장 후보 생성.
    토큰 수 기반 분기 (query_expander보다 단순한 fallback용).
    """
    query = query.strip()
    if not query:
        return []

    tokens = query.split()
    candidates = [query]

    if len(tokens) == 1:
        # 단일 토큰: query_expander 사용 권장, 여기서는 원본만
        pass
    elif len(tokens) == 2:
        candidates.append(tokens[0])
        candidates.append(tokens[1])
    else:
        for i in range(len(tokens) - 1):
            candidates.append(f"{tokens[i]} {tokens[i+1]}")
        candidates.append(f"{tokens[0]} {tokens[-1]}")

    seen = set()
    unique = []
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean not in seen:
            seen.add(c_clean)
            unique.append(c_clean)

    return unique[:max_count]


def search_news_expanded(query: str, display: int = None,
                         max_variants: int = _MAX_EXPANDED_QUERIES) -> list[dict]:
    """
    [v2] generate_query_candidates 기반 확장 수집.
    category 선택 없는 버전 (모든 후보 사용).
    """
    query = query.strip()
    if not query:
        return []

    display = display or NEWS_DISPLAY_PER_REQUEST
    candidates = generate_query_candidates(query, max_count=max_variants)
    logger.info(f"[확장 수집] 원본='{query}' → {len(candidates)}개 후보: {candidates}")

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
        return []

    unique = _deduplicate(all_articles)
    filtered = _filter_by_keyword(unique, query)
    logger.info(f"[확장 수집 완료] 후보={len(candidates)}개 → 최종={len(filtered)}건")
    return filtered


# ═════════════════════════════════════════════════════════════
# v3 신규: 카테고리 기반 확장 수집 (query_expander 연동)
# ═════════════════════════════════════════════════════════════

def search_news_by_category(query: str, category: str,
                             display: int = None,
                             max_variants: int = _MAX_EXPANDED_QUERIES) -> list[dict]:
    """
    [v3 신규] 사용자가 선택한 카테고리에 맞게 쿼리를 자동 확장하여 수집.

    처리 흐름:
      1. query_expander.expand_query(query, category) 호출
         → 카테고리 프로토타입 문장에서 추출된 명사 + 원본 쿼리 조합
         → 예: "악뮤" + "recent" → ["악뮤", "악뮤 최근", "악뮤 소식", ...]
      2. 각 확장 쿼리에 대해 search_news_combined(sim+date) 호출
      3. 전체 결과 통합 + originallink 기준 중복 제거
      4. 키워드 필터링 (config 설정에 따름)

    API 호출량:
      - 확장 쿼리 N개 × 2 (sim+date) = 최대 2N회
      - 기본 N=5 → 최대 10회
      - 네이버 free API 일일 한도(25,000건) 대비 무시 가능

    Args:
        query:        원본 사용자 쿼리
        category:     query_expander가 지원하는 카테고리
                      (recent/official/action/issue/analysis)
        display:      확장 쿼리당 수집 건수
        max_variants: 확장 쿼리 최대 개수
    Returns:
        통합 수집된 기사 리스트 (중복 제거됨)
    """
    query = query.strip()
    if not query:
        logger.warning("search_news_by_category: 빈 쿼리")
        return []

    if not category:
        logger.warning("search_news_by_category: 카테고리 없음. search_news_combined로 폴백")
        return search_news_combined(query, display=display)

    display = display or NEWS_DISPLAY_PER_REQUEST

    # 1단계: query_expander로 카테고리 기반 확장 쿼리 생성
    try:
        from services.query_expander import expand_query
        expanded_queries = expand_query(query, category, max_variants=max_variants)
    except ImportError as e:
        logger.error(f"query_expander import 실패: {e}. search_news_combined로 폴백")
        return search_news_combined(query, display=display)
    except Exception as e:
        logger.error(f"expand_query 실패: {e}. search_news_combined로 폴백")
        return search_news_combined(query, display=display)

    if not expanded_queries:
        logger.warning(f"확장 쿼리 생성 실패 | query='{query}' category='{category}'")
        return search_news_combined(query, display=display)

    logger.info(
        f"[카테고리 수집] query='{query}' category='{category}' → "
        f"{len(expanded_queries)}개 확장: {expanded_queries}"
    )

    # 2단계: 각 확장 쿼리로 수집
    all_articles = []
    for i, variant in enumerate(expanded_queries, 1):
        logger.info(f"[카테고리 수집 {i}/{len(expanded_queries)}] query='{variant}'")
        try:
            articles = search_news_combined(query=variant, display=display)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"카테고리 수집 실패 | query='{variant}' | {e}")
            continue

    if not all_articles:
        logger.warning(f"카테고리 수집 결과 없음 | query='{query}' category='{category}'")
        return []

    # 3단계: 중복 제거
    unique = _deduplicate(all_articles)

    # 4단계: 원본 쿼리 기준 키워드 필터링
    filtered = _filter_by_keyword(unique, query)

    logger.info(
        f"[카테고리 수집 완료] category='{category}' | "
        f"확장 {len(expanded_queries)}개 → 원본 총합 {len(all_articles)}건 → "
        f"중복제거 {len(unique)}건 → 최종 {len(filtered)}건"
    )

    return filtered


# ═════════════════════════════════════════════════════════════
# CLI 테스트
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 사용법:
    #   python news_search.py 악뮤
    #   python news_search.py 악뮤 --expand
    #   python news_search.py 악뮤 --category recent
    test_query = sys.argv[1] if len(sys.argv) > 1 else "뉴스"

    if "--category" in sys.argv:
        idx = sys.argv.index("--category")
        category = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "recent"
        print(f"\n=== 카테고리 수집 테스트: '{test_query}' [{category}] ===\n")
        result = search_news_by_category(test_query, category)
    elif "--expand" in sys.argv:
        print(f"\n=== 확장 수집 테스트: '{test_query}' ===\n")
        result = search_news_expanded(test_query)
    else:
        print(f"\n=== 이중 수집 테스트: '{test_query}' ===\n")
        result = search_news_combined(test_query)

    for i, item in enumerate(result, 1):
        title = _remove_html_tags(item["title"])
        print(f"[{i}] {title} | {item['originallink']}")
    print(f"\n총 {len(result)}건")
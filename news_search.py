"""
news_search.py
──────────────
네이버 뉴스 검색 API를 호출하여 기사 목록을 수집한다.
환경변수에서 API 키를 읽고, 응답을 정제된 리스트로 반환한다.
"""

import requests
import os
from dotenv import load_dotenv
from logger import get_logger
from config import NAVER_API

load_dotenv()

logger = get_logger("news_search")


def search_news(query: str, display: int = None, sort: str = None) -> list[dict]:
    """
    네이버 뉴스 검색 API를 호출하여 기사 목록을 반환한다.

    Args:
        query:   검색 쿼리 문자열
        display: 가져올 기사 수 (기본값: config의 NAVER_API display)
        sort:    정렬 방식 - "date"(최신순) / "sim"(관련도순)

    Returns:
        기사 정보 딕셔너리 리스트
        [{"title": ..., "description": ..., "originallink": ..., "link": ..., "pubDate": ...}, ...]

    Raises:
        ValueError:      API 키 누락 또는 응답 구조 이상
        ConnectionError: 네트워크 요청 실패
    """
    # config 기본값 적용
    display = display or NAVER_API["display"]
    sort = sort or NAVER_API["sort"]

    # API 키 누락 방어
    client_id = os.getenv("naver_client_id")
    client_secret = os.getenv("naver_client_secret")

    if not client_id or not client_secret:
        raise ValueError(".env 파일에 naver_client_id / naver_client_secret 가 없음")

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {
        "query": query,
        "display": display,
        "sort": sort,
    }

    logger.info(f"뉴스 수집 시작 | query={query} display={display} sort={sort}")

    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res.raise_for_status()  # 4xx / 5xx 에러 감지
    except requests.exceptions.Timeout:
        raise ConnectionError("네이버 API 요청 타임아웃 (5초 초과)")
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"네이버 API 요청 실패: {e}")

    try:
        data = res.json()
    except Exception as e:
        raise ValueError(f"API 응답 JSON 파싱 실패: {e}")

    # 응답 구조 이상 방어
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


if __name__ == "__main__":
    result = search_news("이재명")
    for item in result:
        print(item["title"], "|", item["originallink"])
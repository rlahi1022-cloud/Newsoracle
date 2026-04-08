"""
collect_data.py
────────────────
네이버 뉴스 검색 API를 사용하여 학습 데이터를 자동 수집한다.

레이블 태깅 방식 (두 가지 조합):
  1. 도메인 기준: .go.kr / .or.kr / .ac.kr → label=1
  2. 쿼리 기준:   공식 쿼리 그룹 → label=1 / 가십 쿼리 그룹 → label=0
  → 둘 다 일치할 때만 레이블 확정, 충돌하면 도메인 우선

실행:
  python collect_data.py
"""

import requests
import os
import time
import re
import csv
import random
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# 수집 설정
# ─────────────────────────────────────────

TARGET_COUNT = 3000       # 목표 수집 건수
DISPLAY_PER_QUERY = 100   # 쿼리당 최대 수집 건수 (API 최대값)
REQUEST_DELAY = 0.5       # 요청 간 딜레이 (초) - API 과부하 방지
OUTPUT_PATH = "data/train_data.csv"

# ─────────────────────────────────────────
# 공식성 높음 쿼리 목록 (label=1 후보)
# ─────────────────────────────────────────
OFFICIAL_QUERIES = [
    "기획재정부 보도자료",
    "국토교통부 발표",
    "보건복지부 공고",
    "교육부 공식",
    "행정안전부 지침",
    "고용노동부 고시",
    "환경부 정책",
    "과학기술정보통신부 발표",
    "금융위원회 공고",
    "국회 본회의 의결",
    "한국은행 기준금리",
    "질병관리청 브리핑",
    "식품의약품안전처 공고",
    "국세청 공지",
    "경찰청 공식 발표",
    "법무부 보도자료",
    "외교부 공식 입장",
    "통일부 공식 발표",
    "국방부 보도자료",
    "문화체육관광부 발표",
    "농림축산식품부 공고",
    "산업통상자원부 발표",
    "중소벤처기업부 공지",
    "여성가족부 보도자료",
    "해양수산부 공고",
    "공정거래위원회 결정",
    "방송통신위원회 의결",
    "국민권익위원회 발표",
    "감사원 감사결과",
    "헌법재판소 결정",
]

# ─────────────────────────────────────────
# 공식성 낮음 쿼리 목록 (label=0 후보)
# ─────────────────────────────────────────
NON_OFFICIAL_QUERIES = [
    "연예인 열애설",
    "연예인 루머",
    "익명 제보 의혹",
    "카더라 소문",
    "SNS 논란",
    "유튜버 주장",
    "네티즌 반응",
    "온라인 커뮤니티 화제",
    "인터넷 밈 화제",
    "연예인 스캔들",
    "가십 뉴스",
    "익명 소식통",
    "찌라시 논란",
    "블라인드 폭로",
    "팬덤 논란",
    "방송 사고 화제",
    "연예계 충격",
    "인스타그램 논란",
    "틱톡 화제",
    "유명인 사생활",
]

# ─────────────────────────────────────────
# 공식 도메인 목록 (도메인 기준 태깅)
# ─────────────────────────────────────────
OFFICIAL_DOMAINS = [
    ".go.kr",
    ".or.kr",
    ".ac.kr",
    ".re.kr",
    "yonhapnews.co.kr",
    "yna.co.kr",
    "korea.kr",
]

# 비공식 도메인 키워드 (이게 포함되면 label=0 강화)
NON_OFFICIAL_DOMAIN_KEYWORDS = [
    "blog", "cafe", "tistory", "naver.com",
    "instagram", "youtube", "twitter", "tiktok",
]


def get_naver_news(query: str, display: int = 100) -> list[dict]:
    """
    네이버 뉴스 검색 API 호출.

    Args:
        query:   검색 쿼리
        display: 가져올 건수 (최대 100)
    Returns:
        기사 딕셔너리 리스트
    """
    client_id = os.getenv("naver_client_id")
    client_secret = os.getenv("naver_client_secret")

    if not client_id or not client_secret:
        raise ValueError(".env 파일에 naver_client_id / naver_client_secret 없음")

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": display, "sort": "date"}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res.raise_for_status()
        data = res.json()
        return data.get("items", [])
    except requests.exceptions.RequestException as e:
        print(f"  [오류] API 요청 실패: {e}")
        return []
    except Exception as e:
        print(f"  [오류] 응답 파싱 실패: {e}")
        return []


def remove_html_tags(text: str) -> str:
    """HTML 태그 및 엔티티 제거."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_domain(url: str) -> str:
    """URL에서 도메인 추출."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def assign_label(query_group: str, domain: str) -> int:
    """
    도메인 + 쿼리 그룹 두 가지 기준을 조합하여 레이블을 결정한다.

    우선순위:
      1. 도메인이 명확히 공식 → label=1 확정
      2. 도메인이 명확히 비공식 → label=0 확정
      3. 도메인 불분명 → 쿼리 그룹 기준으로 결정

    Args:
        query_group: "official" 또는 "non_official"
        domain:      기사 원문 도메인
    Returns:
        0 또는 1
    """
    # 도메인 기준 공식 여부 확인
    is_official_domain = any(od in domain for od in OFFICIAL_DOMAINS)
    is_non_official_domain = any(nd in domain for nd in NON_OFFICIAL_DOMAIN_KEYWORDS)

    if is_official_domain:
        return 1
    if is_non_official_domain:
        return 0

    # 도메인 불분명 → 쿼리 그룹 기준
    if query_group == "official":
        return 1
    return 0


def collect_articles(queries: list[str], query_group: str, target: int) -> list[dict]:
    """
    쿼리 목록으로 기사를 수집하고 레이블을 태깅한다.

    Args:
        queries:     쿼리 문자열 리스트
        query_group: "official" 또는 "non_official"
        target:      목표 수집 건수
    Returns:
        수집된 기사 딕셔너리 리스트
    """
    collected = []
    seen_links = set()  # 중복 제거용

    # 쿼리 순서를 랜덤하게 섞어서 다양성 확보
    shuffled_queries = queries.copy()
    random.shuffle(shuffled_queries)

    for query in shuffled_queries:
        if len(collected) >= target:
            break

        print(f"  쿼리: '{query}' 수집 중...")
        items = get_naver_news(query, display=DISPLAY_PER_QUERY)

        for item in items:
            if len(collected) >= target:
                break

            original_link = item.get("originallink", "")
            link = item.get("link", "")

            # 중복 기사 제거
            unique_key = original_link or link
            if unique_key in seen_links:
                continue
            seen_links.add(unique_key)

            title = remove_html_tags(item.get("title", ""))
            content = remove_html_tags(item.get("description", ""))
            domain = extract_domain(original_link)

            # 제목 또는 본문이 비어있으면 스킵
            if not title or not content:
                continue

            label = assign_label(query_group, domain)

            collected.append({
                "title": title,
                "content": content,
                "source": domain,
                "originallink": original_link,
                "official_label": label,
            })

        print(f"  → 현재 수집: {len(collected)}건")
        time.sleep(REQUEST_DELAY)  # API 과부하 방지

    return collected


def save_to_csv(articles: list[dict], output_path: str):
    """
    수집된 기사를 CSV로 저장한다.

    Args:
        articles:    기사 딕셔너리 리스트
        output_path: 저장 경로
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["title", "content", "source", "originallink", "official_label"]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(articles)

    print(f"\n저장 완료: {output_path} ({len(articles)}건)")


def main():
    print("=" * 60)
    print("뉴스 학습 데이터 자동 수집 시작")
    print(f"목표: {TARGET_COUNT}건 (공식 1500건 + 비공식 1500건)")
    print("=" * 60)

    half = TARGET_COUNT // 2

    # ── 공식 기사 수집 (label=1) ──────────────────────
    print(f"\n[1/2] 공식 기사 수집 중 (목표: {half}건)")
    official_articles = collect_articles(OFFICIAL_QUERIES, "official", half)
    print(f"공식 기사 수집 완료: {len(official_articles)}건")

    # ── 비공식 기사 수집 (label=0) ────────────────────
    print(f"\n[2/2] 비공식 기사 수집 중 (목표: {half}건)")
    non_official_articles = collect_articles(NON_OFFICIAL_QUERIES, "non_official", half)
    print(f"비공식 기사 수집 완료: {len(non_official_articles)}건")

    # ── 합치고 셔플 ───────────────────────────────────
    all_articles = official_articles + non_official_articles
    random.shuffle(all_articles)

    # ── 레이블 분포 출력 ──────────────────────────────
    label_1 = sum(1 for a in all_articles if a["official_label"] == 1)
    label_0 = sum(1 for a in all_articles if a["official_label"] == 0)
    print(f"\n레이블 분포: 공식(1)={label_1}건 / 비공식(0)={label_0}건")

    # ── CSV 저장 ──────────────────────────────────────
    save_to_csv(all_articles, OUTPUT_PATH)

    print("\n완료! 이제 아래 명령어로 학습하면 됩니다:")
    print(f"  python main.py --mode train --train-path {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
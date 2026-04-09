"""
collect_data.py
────────────────
네이버 뉴스 검색 API를 사용하여 학습 데이터를 수집하고 레이블을 부여한다.

[레이블링 기준 - 기관의 직접 발표 여부]

이 프로젝트의 목적: "기사를 낸 기관이 공식인지" 판별
따라서 레이블 기준은 명확하게 정의해야 한다.

  label=1 (공식): 기관이 직접 발표한 기사
    - .go.kr / .or.kr 등 공식 기관 도메인에서 직접 발행
    - "보도자료", "공식 발표", "브리핑" + 기관명 주어 동시 등장
    - 기관명이 직접 주어로 "발표했다", "밝혔다", "결정했다" 구조

  label=0 (비공식): 기관의 직접 발표가 아닌 기사
    - 익명 제보: "소식통", "관계자에 따르면", "익명"
    - 추측/분석: "것으로 알려져", "전망이다", "추정된다"
    - 연예/가십: 공식 기관 주어 없는 사생활/루머

[실행]
  # 학습 데이터 수집 (4000건)
  python collect_data.py --mode train --output data/train_data.csv --target 4000

  # OOD 테스트셋 수집 (200건, 실제 추론 환경 재현)
  python collect_data.py --mode ood --output data/ood_test.csv --target 200
"""

import os
import re
import csv
import time
import random
import argparse
from datetime import datetime
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 수집 설정
# ─────────────────────────────────────────────────────────────

TARGET_COUNT      = 4000
DISPLAY_PER_QUERY = 100
REQUEST_DELAY     = 0.3
DEFAULT_OUTPUT    = "data/train_data.csv"
OOD_OUTPUT        = "data/ood_test.csv"

# ─────────────────────────────────────────────────────────────
# 도메인 기반 레이블 확정
# ─────────────────────────────────────────────────────────────

OFFICIAL_DOMAIN_KEYWORDS = [
    ".go.kr", ".or.kr", ".re.kr", ".ac.kr",
    "yna.co.kr", "yonhapnews.co.kr", "korea.kr", "bok.or.kr",
    "fsc.go.kr", "fss.or.kr", "nts.go.kr",
    "moef.go.kr", "mohw.go.kr", "moe.go.kr",
    "mois.go.kr", "moj.go.kr", "mofa.go.kr",
    "motie.go.kr", "kostat.go.kr", "kma.go.kr",
]

NON_OFFICIAL_DOMAIN_KEYWORDS = [
    "blog.", "tistory.", "cafe.",
    "instagram.", "youtube.", "twitter.", "tiktok.", "facebook.",
]

# ─────────────────────────────────────────────────────────────
# 공식 기사 수집 쿼리 (label=1)
# ─────────────────────────────────────────────────────────────

OFFICIAL_QUERIES = [
    "기획재정부 보도자료", "국토교통부 발표", "보건복지부 공고",
    "교육부 공식 발표", "행정안전부 지침 발표", "고용노동부 고시",
    "환경부 정책 발표", "과학기술정보통신부 발표", "금융위원회 공고",
    "금융감독원 발표", "국방부 보도자료", "외교부 공식 입장",
    "법무부 보도자료", "통일부 공식 발표", "문화체육관광부 발표",
    "농림축산식품부 공고", "산업통상자원부 발표", "중소벤처기업부 공지",
    "여성가족부 보도자료", "해양수산부 공고",
    "한국은행 기준금리 발표", "한국은행 통화정책 결정",
    "통계청 통계 발표", "기상청 공식 예보", "특허청 공고",
    "국민연금공단 발표", "국민건강보험공단 공고", "한국전력 공시",
    "공정거래위원회 결정", "감사원 감사결과",
    "질병관리청 브리핑", "식품의약품안전처 공고",
    "국세청 공지", "국회 본회의 의결",
    "삼성전자 실적 발표", "현대자동차 공식 발표",
    "협약 체결 공식 서명", "기업 보도자료 공시", "코스피 상장 공시",
]

# ─────────────────────────────────────────────────────────────
# 비공식 기사 수집 쿼리 (label=0)
# ─────────────────────────────────────────────────────────────

NON_OFFICIAL_QUERIES = [
    # 익명/추측
    "소식통에 따르면 단독", "관계자에 따르면 주장", "업계 관계자 익명",
    "것으로 알려져 단독", "전망이다 관측", "추정된다 분석",
    "가능성이 높다 예상", "것으로 보인다 업계", "내부 관계자 주장",
    "복수의 관계자에 따르면",
    # 의혹/논란
    "의혹 제기", "논란 확산", "폭로 충격", "주장이 나왔다",
    "논란이 일고 있다", "알려졌다 단독", "것으로 전해졌다",
    "루머 사실 확인", "카더라 통신", "찌라시 논란",
    # 연예/가십
    "연예인 열애설", "연예인 루머", "익명 제보 의혹",
    "SNS 논란", "연예인 스캔들", "가십 뉴스",
    "블라인드 폭로", "팬덤 논란", "연예계 충격 폭로", "유명인 사생활 논란",
]

# ─────────────────────────────────────────────────────────────
# OOD 테스트셋 수집 쿼리
#
# 학습 쿼리와 완전히 다른 일반 키워드
# 실제 사용자가 검색할 법한 키워드로 구성
# 공식/비공식이 자연스럽게 섞인 환경 재현
# ─────────────────────────────────────────────────────────────

OOD_QUERIES = [
    "기준금리", "코스피", "환율", "부동산 정책", "물가 상승",
    "자동차 산업", "반도체 수출", "배터리 기업", "전기차",
    "국회 통과", "정부 예산", "외교 관계", "의료 개혁",
    "미국 금리", "중동 정세", "무역 협상",
]


# ─────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────

def log(level: str, message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level:5s}] {message}")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    for entity, char in {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    }.items():
        text = text.replace(entity, char)
    return re.sub(r"\s+", " ", text).strip()


def extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


def assign_label(domain: str, query_group: str) -> tuple:
    """
    도메인 + 쿼리 그룹을 조합하여 레이블을 결정한다.

    우선순위:
      1. 공식 기관 도메인 → label=1 확정 (기관이 직접 발행)
      2. 비공식 도메인   → label=0 확정
      3. 일반 언론사     → 쿼리 그룹 기준
      4. OOD 미확정      → label=-1 (수동 검수 권장)
    """
    for kw in OFFICIAL_DOMAIN_KEYWORDS:
        if kw in domain:
            return 1, f"공식 도메인 확정: {domain}"
    for kw in NON_OFFICIAL_DOMAIN_KEYWORDS:
        if kw in domain:
            return 0, f"비공식 도메인 확정: {domain}"
    if query_group == "official":
        return 1, f"공식 쿼리 기반: {domain}"
    if query_group == "non_official":
        return 0, f"비공식 쿼리 기반: {domain}"
    # OOD: 도메인/쿼리로 판단 불가 → 수동 검수 필요
    return -1, f"OOD 미확정: {domain}"


def fetch_naver_news(query: str, display: int = 100, start: int = 1) -> list:
    """네이버 뉴스 검색 API 단일 호출."""
    client_id     = os.getenv("NAVER_CLIENT_ID") or os.getenv("naver_client_id")
    client_secret = os.getenv("NAVER_CLIENT_SECRET") or os.getenv("naver_client_secret")

    if not client_id or not client_secret:
        raise ValueError(".env에 naver_client_id / naver_client_secret이 없습니다.")

    try:
        resp = requests.get(
            "https://openapi.naver.com/v1/search/news.json",
            headers={
                "X-Naver-Client-Id":     client_id,
                "X-Naver-Client-Secret": client_secret,
            },
            params={"query": query, "display": min(display, 100), "start": start, "sort": "date"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("items", [])
    except Exception as e:
        log("WARN", f"API 실패: '{query}' | {e}")
        return []


def collect_group(queries: list, query_group: str, target: int, seen_links: set) -> list:
    """쿼리 목록으로 기사를 수집하고 레이블을 태깅한다."""
    collected = []
    shuffled  = queries.copy()
    random.shuffle(shuffled)

    for query in shuffled:
        if len(collected) >= target:
            break

        items = fetch_naver_news(query, display=DISPLAY_PER_QUERY)
        added = 0

        for item in items:
            if len(collected) >= target:
                break

            original_link = item.get("originallink", "")
            link          = item.get("link", "")
            unique_key    = original_link or link

            if not unique_key or unique_key in seen_links:
                continue
            seen_links.add(unique_key)

            title   = clean_text(item.get("title", ""))
            content = clean_text(item.get("description", ""))
            if not title or not content:
                continue

            domain        = extract_domain(original_link)
            label, reason = assign_label(domain, query_group)

            collected.append({
                "title":          title,
                "content":        content,
                "source":         domain,
                "originallink":   original_link,
                "official_label": label,
                "label_reason":   reason,
            })
            added += 1

        log("INFO", f"  [{query_group}] '{query}': {added}건 추가 (누적 {len(collected)}건)")
        time.sleep(REQUEST_DELAY)

    return collected


def save_to_csv(articles: list, output_path: str):
    if not articles:
        log("WARN", "저장할 데이터가 없습니다.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fieldnames = ["title", "content", "source", "originallink", "official_label", "label_reason"]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(articles)

    log("INFO", f"CSV 저장 완료: {output_path} ({len(articles)}건)")


def print_summary(articles: list, mode: str):
    from collections import Counter

    label_1   = sum(1 for a in articles if a["official_label"] == 1)
    label_0   = sum(1 for a in articles if a["official_label"] == 0)
    label_unk = sum(1 for a in articles if a["official_label"] == -1)
    sources   = Counter(a["source"] for a in articles)

    reason_types = Counter()
    for a in articles:
        r = a.get("label_reason", "")
        if "공식 도메인" in r:   reason_types["도메인 확정(공식)"] += 1
        elif "비공식 도메인" in r: reason_types["도메인 확정(비공식)"] += 1
        elif "공식 쿼리" in r:   reason_types["쿼리 기반(공식)"] += 1
        elif "비공식 쿼리" in r: reason_types["쿼리 기반(비공식)"] += 1
        elif "OOD" in r:         reason_types["OOD 미확정"] += 1

    print()
    print("=" * 65)
    print(f"  수집 완료 요약 [{mode} 모드]")
    print("=" * 65)
    print(f"  총 수집 건수     : {len(articles)}건")
    print(f"  공식(label=1)    : {label_1}건")
    print(f"  비공식(label=0)  : {label_0}건")
    if label_unk > 0:
        print(f"  미확정(label=-1) : {label_unk}건  ← 수동 검수 후 사용 권장")
    print()
    print("  레이블 결정 방식:")
    for rt, cnt in reason_types.most_common():
        print(f"    {rt:30s} {cnt}건")
    print()
    print("  상위 출처 도메인 10개:")
    for domain, cnt in sources.most_common(10):
        print(f"    {domain:40s} {cnt}건")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="네이버 뉴스 수집 + 도메인/쿼리 혼합 레이블링",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "ood"],
        help="train: 학습 데이터 / ood: OOD 테스트셋",
    )
    parser.add_argument("--output", type=str, default=None, help="저장 경로")
    parser.add_argument("--target", type=int, default=TARGET_COUNT, help="목표 수집 건수")
    args = parser.parse_args()

    if args.output is None:
        args.output = OOD_OUTPUT if args.mode == "ood" else DEFAULT_OUTPUT

    naver_id     = os.getenv("NAVER_CLIENT_ID") or os.getenv("naver_client_id")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET") or os.getenv("naver_client_secret")
    if not naver_id or not naver_secret:
        log("ERROR", ".env에 naver_client_id / naver_client_secret이 없습니다.")
        raise SystemExit(1)

    seen_links = set()

    if args.mode == "train":
        half = args.target // 2
        log("INFO", "=" * 65)
        log("INFO", f"학습 데이터 수집 | 목표: {args.target}건 (공식 {half} + 비공식 {half})")
        log("INFO", "=" * 65)

        log("INFO", f"[1/2] 공식 기사 수집 (목표: {half}건)")
        official = collect_group(OFFICIAL_QUERIES, "official", half, seen_links)
        log("INFO", f"[1/2] 완료: {len(official)}건")

        log("INFO", f"[2/2] 비공식 기사 수집 (목표: {half}건)")
        non_official = collect_group(NON_OFFICIAL_QUERIES, "non_official", half, seen_links)
        log("INFO", f"[2/2] 완료: {len(non_official)}건")

        all_articles = official + non_official
        random.shuffle(all_articles)

    else:  # ood
        log("INFO", "=" * 65)
        log("INFO", f"OOD 테스트셋 수집 | 목표: {args.target}건")
        log("INFO", "label=-1 기사는 수동 검수 후 사용 권장")
        log("INFO", "=" * 65)
        all_articles = collect_group(OOD_QUERIES, "ood", args.target, seen_links)

    if not all_articles:
        log("ERROR", "수집된 기사가 없습니다.")
        raise SystemExit(1)

    print_summary(all_articles, args.mode)
    save_to_csv(all_articles, args.output)

    print()
    if args.mode == "train":
        log("INFO", "학습 시작:")
        log("INFO", f"  python main.py --mode train --train-path {args.output}")
        log("INFO", "OOD 테스트셋 수집 (실제 성능 측정용):")
        log("INFO", "  python collect_data.py --mode ood --target 200")
    else:
        log("INFO", "OOD 테스트셋 준비 완료.")
        log("INFO", "학습 완료 후 자동으로 OOD 평가가 실행됩니다.")


if __name__ == "__main__":
    main()
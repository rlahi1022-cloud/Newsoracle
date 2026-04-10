"""
collect_data.py
────────────────
네이버 뉴스 검색 API를 사용하여 학습 데이터를 수집하고 레이블을 부여한다.

[레이블링 기준 - 공식성 판별]

이 프로젝트의 목적: "기사가 공식적 근거를 기반으로 작성되었는가" 판별
공식성은 카테고리(정치/연예/경제)가 아니라 근거 구조로 판단한다.

  label=1 (공식): 공식 기관/기업/단체가 직접 발표한 기사
    - .go.kr / .or.kr 등 공식 기관 도메인에서 직접 발행
    - "보도자료", "공식 발표", "브리핑" + 기관명 주어 동시 등장
    - 기관명이 직접 주어로 "발표했다", "밝혔다", "결정했다" 구조
    - 소속사 공식 입장, 기업 공시, 협회 공지 등 포함

  label=0 (비공식): 공식 기관의 직접 발표가 아닌 기사
    - 익명 제보: "소식통", "관계자에 따르면", "익명"
    - 추측/분석: "것으로 알려져", "전망이다", "추정된다"
    - 루머/가십: 공식 기관 주어 없는 사생활/추측 기사

[쿼리 설계 원칙]
  - 6개 분야(정부부처, 공공기관/경제, 연예/엔터, 스포츠, IT/기술, 의료/학술)를
    균등하게 배분하여 classifier 분야 편향을 방지한다.
  - 같은 연예 기사라도 소속사 공식 입장이면 label=1이 될 수 있다.
  - 같은 정치 기사라도 익명 관계자 발언 위주면 label=0이 될 수 있다.

[실행]
  # 학습 데이터 수집 (6000건)
  python collect_data.py --mode train --output data/train_data.csv --target 6000

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

# .env 파일에서 네이버 API 키를 로드한다
load_dotenv()

# ─────────────────────────────────────────────────────────────
# 수집 설정 상수
# ─────────────────────────────────────────────────────────────

# 목표 수집 건수: 6000건 (공식 3000 + 비공식 3000)
# 6개 분야 균등 배분으로 분야 편향을 방지한다
TARGET_COUNT = 6000

# 네이버 API 1회 요청당 최대 수집 건수 (API 최대값 100)
DISPLAY_PER_QUERY = 100

# API 요청 간 딜레이(초) - 네이버 API 쿼터 보호용
# 너무 빠르게 요청하면 429 에러 발생 가능
REQUEST_DELAY = 0.3

# 기본 저장 경로
DEFAULT_OUTPUT = "data/train_data.csv"
OOD_OUTPUT = "data/ood_test.csv"

# ─────────────────────────────────────────────────────────────
# 도메인 기반 레이블 확정 리스트
#
# 도메인만으로 공식/비공식을 확정할 수 있는 경우에 사용한다.
# 일반 언론사 도메인은 쿼리 그룹 기준으로 결정한다.
# ─────────────────────────────────────────────────────────────

# 공식 기관 도메인 키워드
# .go.kr = 정부기관, .or.kr = 공공기관, .re.kr = 연구기관, .ac.kr = 학교
# 연합뉴스(yna/yonhapnews)는 국가기간뉴스통신사로 공식 분류
OFFICIAL_DOMAIN_KEYWORDS = [
    ".go.kr", ".or.kr", ".re.kr", ".ac.kr",
    "yna.co.kr", "yonhapnews.co.kr", "korea.kr", "bok.or.kr",
    "fsc.go.kr", "fss.or.kr", "nts.go.kr",
    "moef.go.kr", "mohw.go.kr", "moe.go.kr",
    "mois.go.kr", "moj.go.kr", "mofa.go.kr",
    "motie.go.kr", "kostat.go.kr", "kma.go.kr",
]

# 비공식 도메인 키워드
# 블로그, SNS, 카페 등 개인 미디어 플랫폼은 비공식으로 확정
NON_OFFICIAL_DOMAIN_KEYWORDS = [
    "blog.", "tistory.", "cafe.",
    "instagram.", "youtube.", "twitter.", "tiktok.", "facebook.",
]

# ─────────────────────────────────────────────────────────────
# 공식 기사 수집 쿼리 (label=1)
#
# 6개 분야를 균등하게 배분하여 분야 편향을 방지한다.
# 각 분야별 약 14~16개 쿼리로 구성한다.
# 총 85개 이상 쿼리로 6000건 수집 시 다양성을 확보한다.
#
# 설계 원칙:
# - "기관명/단체명 + 발표 행위 동사" 조합으로 구성
# - 쿼리 자체가 공식 발표 기사를 가리키도록
# ─────────────────────────────────────────────────────────────

OFFICIAL_QUERIES = [
    # ── 분야 1: 정부 부처 공식 발표 (16개) ──────────────────
    # 정부 부처가 직접 배포한 보도자료/공고/브리핑 기반 기사
    "기획재정부 보도자료",
    "국토교통부 발표",
    "보건복지부 공고",
    "교육부 공식 발표",
    "행정안전부 지침 발표",
    "고용노동부 고시",
    "환경부 정책 발표",
    "과학기술정보통신부 발표",
    "금융위원회 공고",
    "금융감독원 발표",
    "국방부 보도자료",
    "외교부 공식 입장",
    "법무부 보도자료",
    "통일부 공식 발표",
    "문화체육관광부 발표",
    "농림축산식품부 공고",

    # ── 분야 2: 공공기관 / 경제 공식 발표 (16개) ────────────
    # 공공기관, 중앙은행, 통계청 등의 공식 데이터 발표
    "산업통상자원부 발표",
    "중소벤처기업부 공지",
    "여성가족부 보도자료",
    "해양수산부 공고",
    "한국은행 기준금리 발표",
    "한국은행 통화정책 결정",
    "통계청 통계 발표",
    "기상청 공식 예보",
    "특허청 공고",
    "국민연금공단 발표",
    "국민건강보험공단 공고",
    "한국전력 공시",
    "공정거래위원회 결정",
    "감사원 감사결과",
    "국세청 공지",
    "국회 본회의 의결",

    # ── 분야 3: 연예 / 엔터테인먼트 공식 발표 (14개) ───────
    # 소속사 공식 입장, 공식 컴백 발표, 수상 공식 발표 등
    # 연예 분야도 공식 발표 구조면 공식성이 높다
    "소속사 공식 입장",
    "소속사 공식 발표",
    "소속사 보도자료",
    "컴백 공식 발표",
    "전속계약 체결 공식",
    "결혼 공식 발표",
    "콘서트 공식 일정 발표",
    "음원 발매 공식 발표",
    "영화 제작 공식 발표",
    "출연 확정 공식 발표",
    "시상식 수상자 공식 발표",
    "드라마 캐스팅 공식 확정",
    "엔터테인먼트 공시 발표",
    "방송사 편성 공식 발표",

    # ── 분야 4: 스포츠 공식 발표 (14개) ────────────────────
    # 협회/구단/리그 공식 발표, 선수 계약 공식 체결 등
    "대한축구협회 공식 발표",
    "대한체육회 공식 발표",
    "KBO 공식 발표",
    "FA 계약 체결 공식",
    "감독 선임 공식 발표",
    "선수 영입 공식 발표",
    "올림픽 대표팀 명단 공식",
    "프로야구 드래프트 결과",
    "FIFA 공식 발표",
    "IOC 공식 결정",
    "리그 일정 공식 발표",
    "구단 공식 입장",
    "대한배구협회 공식 발표",
    "대한농구협회 공식 발표",

    # ── 분야 5: IT / 기술 공식 발표 (14개) ─────────────────
    # 기업 공식 출시, 서비스 공지, 약관 변경 등
    "서비스 출시 공식 발표",
    "약관 변경 공지",
    "AI 서비스 공식 출시",
    "삼성전자 실적 발표",
    "현대자동차 공식 발표",
    "SK하이닉스 실적 공시",
    "네이버 공식 발표",
    "카카오 공식 입장",
    "기업 보도자료 공시",
    "코스피 상장 공시",
    "스타트업 투자 유치 공식",
    "플랫폼 정책 변경 공지",
    "앱 업데이트 공식 발표",
    "기술 특허 취득 공식",

    # ── 분야 6: 의료 / 학술 공식 발표 (14개) ────────────────
    # 질병관리청, 식약처, 병원, 학회 등의 공식 발표
    "질병관리청 브리핑",
    "식품의약품안전처 공고",
    "의약품 허가 승인 발표",
    "임상시험 결과 발표",
    "WHO 공식 발표",
    "대한의사협회 공식 입장",
    "대한약사회 공식 발표",
    "의료기관 공식 발표",
    "백신 접종 공식 일정",
    "감염병 공식 통계",
    "대학 연구팀 연구결과 발표",
    "학회 공식 발표",
    "임상시험 승인 공고",
    "의료 가이드라인 공식 발표",
]

# ─────────────────────────────────────────────────────────────
# 비공식 기사 수집 쿼리 (label=0)
#
# 6개 분야에 대응하는 비공식/추측/루머 쿼리를 균등 배분한다.
# 공식 쿼리와 1:1 대응하여 클래스 균형을 맞춘다.
#
# 설계 원칙:
# - 익명 소식통, 추측 표현, 루머, 커뮤니티 반응 중심
# - 발화 주체가 불명확하거나 검증이 어려운 기사를 수집
# ─────────────────────────────────────────────────────────────

NON_OFFICIAL_QUERIES = [
    # ── 분야 1: 정치/정부 관련 비공식 (10개) ────────────────
    # 익명 관계자, 추측, 내부 소식통 기반 기사
    "소식통에 따르면 단독",
    "관계자에 따르면 주장",
    "복수의 관계자에 따르면",
    "내부 관계자 주장",
    "정치권 관측 전망",
    "것으로 알려져 단독",
    "의혹 제기 논란",
    "논란 확산 파문",
    "카더라 통신 루머",
    "찌라시 논란 정치",

    # ── 분야 2: 경제/업계 관련 비공식 (10개) ────────────────
    # 업계 추정, 전망, 관측 기반 기사
    "업계 관계자 익명",
    "전망이다 관측",
    "추정된다 분석",
    "가능성이 높다 예상",
    "것으로 보인다 업계",
    "알려졌다 단독",
    "것으로 전해졌다",
    "루머 사실 확인 경제",
    "주장이 나왔다 업계",
    "논란이 일고 있다 경제",

    # ── 분야 3: 연예/엔터 관련 비공식 (10개) ────────────────
    # 열애설, 루머, SNS 논란, 익명 제보 기반 기사
    "연예인 열애설",
    "연예인 루머",
    "익명 제보 의혹 연예",
    "SNS 논란 연예인",
    "연예인 스캔들",
    "블라인드 폭로 연예",
    "팬덤 논란 아이돌",
    "연예계 충격 폭로",
    "유명인 사생활 논란",
    "이혼설 추정 연예",

    # ── 분야 4: 스포츠 관련 비공식 (10개) ──────────────────
    # 이적설, 의혹, 소식통 기반 기사
    "선수 이적설 소식통",
    "승부조작 의혹 스포츠",
    "도핑 의혹 선수",
    "선수 부상 소식통",
    "감독 경질설 소문",
    "스포츠 폭로 익명",
    "선수 불화설 팀",
    "FA 이적 루머",
    "구단 매각설 추정",
    "은퇴설 추측 스포츠",

    # ── 분야 5: IT/기술 관련 비공식 (10개) ─────────────────
    # 해킹 의혹, 서비스 장애 추정, 인수합병 루머 등
    "서비스 장애 추정 원인",
    "해킹 의혹 개인정보",
    "인수합병 추측 IT",
    "스타트업 위기설",
    "앱 논란 사용자 불만",
    "개인정보 유출 의혹",
    "플랫폼 갑질 논란",
    "기술 특허 분쟁 의혹",
    "AI 윤리 논란 추측",
    "서비스 종료 루머",

    # ── 분야 6: 의료/건강 관련 비공식 (10개) ────────────────
    # 부작용 의혹, 건강 루머, 소식통 기반 기사
    "부작용 의혹 약물",
    "의료사고 소식통",
    "건강 루머 유명인",
    "민간요법 효과 추정",
    "백신 부작용 논란",
    "병원 비리 의혹",
    "의료진 폭로 익명",
    "건강기능식품 과대광고 논란",
    "질병 괴담 온라인",
    "의약품 리콜 루머",
]

# ─────────────────────────────────────────────────────────────
# OOD 테스트셋 수집 쿼리
#
# 학습 쿼리와 완전히 다른 일반 키워드로 구성한다.
# 실제 사용자가 검색할 법한 키워드로 구성한다.
# 공식/비공식이 자연스럽게 섞인 실제 환경을 재현한다.
#
# OOD(Out-of-Distribution): 학습 데이터 분포 밖의 데이터
# → 학습 때 못 본 패턴에 대한 모델 일반화 능력을 측정한다
# ─────────────────────────────────────────────────────────────

OOD_QUERIES = [
    # 경제/금융 일반 키워드
    "기준금리", "코스피", "환율", "부동산 정책", "물가 상승",
    # 산업 일반 키워드
    "자동차 산업", "반도체 수출", "배터리 기업", "전기차",
    # 정치/외교 일반 키워드
    "국회 통과", "정부 예산", "외교 관계", "의료 개혁",
    # 국제 일반 키워드
    "미국 금리", "중동 정세", "무역 협상",
    # 연예/스포츠 일반 키워드 (분야 편향 방지)
    "BTS", "블랙핑크", "손흥민", "대한민국 축구",
    # IT/기술 일반 키워드
    "AI 기술", "챗봇", "자율주행",
    # 의료/건강 일반 키워드
    "독감 유행", "건강검진", "병원 진료",
]


# ═════════════════════════════════════════════════════════════
# 유틸리티 함수
# ═════════════════════════════════════════════════════════════

def log(level: str, message: str):
    """
    타임스탬프 + 로그 레벨 포함 로그 출력.
    logger.py 대신 독립적으로 사용한다.
    이유: collect_data.py는 단독 실행 스크립트이므로
          프로젝트 내 logger.py에 의존하지 않도록 분리한다.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level:5s}] {message}")


def clean_text(text: str) -> str:
    """
    HTML 태그 제거 + HTML 엔티티 변환 + 연속 공백 정리.
    네이버 API 응답의 title/description에는
    <b></b> 태그와 &amp; 등의 HTML 엔티티가 포함되어 있다.
    학습 데이터 품질을 위해 반드시 정제해야 한다.
    """
    if not text:
        return ""
    # HTML 태그 제거 (예: <b>키워드</b> → 키워드)
    text = re.sub(r"<[^>]+>", "", text)
    # HTML 엔티티 변환 (예: &amp; → &, &lt; → <)
    for entity, char in {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    }.items():
        text = text.replace(entity, char)
    # 연속 공백/탭/줄바꿈을 단일 공백으로
    return re.sub(r"\s+", " ", text).strip()


def extract_domain(url: str) -> str:
    """
    URL에서 도메인(netloc)을 추출한다.
    예: "https://www.moef.go.kr/nw/nes/..." → "www.moef.go.kr"
    도메인 기반 레이블 확정에 사용한다.
    """
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


def assign_label(domain: str, query_group: str) -> tuple:
    """
    도메인 + 쿼리 그룹을 조합하여 레이블을 결정한다.

    레이블 결정 우선순위:
      1순위: 공식 기관 도메인 → label=1 확정 (기관이 직접 발행한 것이 확실)
      2순위: 비공식 도메인(블로그/SNS) → label=0 확정
      3순위: 일반 언론사 도메인 → 쿼리 그룹 기준으로 결정
      4순위: OOD 모드에서 판단 불가 → label=-1 (수동 검수 권장)

    Args:
        domain:      기사 원문 링크의 도메인
        query_group: "official" / "non_official" / "ood"

    Returns:
        (label, reason) 튜플
        label: 1(공식) / 0(비공식) / -1(미확정)
        reason: 레이블 결정 근거 문자열
    """
    # 1순위: 공식 기관 도메인 확인
    for kw in OFFICIAL_DOMAIN_KEYWORDS:
        if kw in domain:
            return 1, f"공식 도메인 확정: {domain}"

    # 2순위: 비공식 도메인(블로그/SNS) 확인
    for kw in NON_OFFICIAL_DOMAIN_KEYWORDS:
        if kw in domain:
            return 0, f"비공식 도메인 확정: {domain}"

    # 3순위: 일반 언론사 → 쿼리 그룹 기준
    if query_group == "official":
        return 1, f"공식 쿼리 기반: {domain}"
    if query_group == "non_official":
        return 0, f"비공식 쿼리 기반: {domain}"

    # 4순위: OOD 모드 - 도메인/쿼리로 판단 불가
    # label=-1은 학습에 사용하지 않고 수동 검수 후 사용 권장
    return -1, f"OOD 미확정: {domain}"


def fetch_naver_news(query: str, display: int = 100, start: int = 1, sort: str = "date") -> list:
    """
    네이버 뉴스 검색 API 단일 호출.

    Args:
        query:   검색 키워드
        display: 한 번에 가져올 기사 수 (최대 100)
        start:   검색 시작 위치 (1~1000)
        sort:    정렬 기준 ("date" = 최신순, "sim" = 유사도순)

    Returns:
        기사 아이템 리스트 (빈 리스트일 수 있음)

    Raises:
        ValueError: .env에 API 키가 없을 때
    """
    # 네이버 API 키 로드 (대소문자 양쪽 지원)
    client_id = os.getenv("NAVER_CLIENT_ID") or os.getenv("naver_client_id")
    client_secret = os.getenv("NAVER_CLIENT_SECRET") or os.getenv("naver_client_secret")

    if not client_id or not client_secret:
        raise ValueError(
            ".env에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET이 없습니다. "
            ".env.example을 참고하여 .env 파일을 생성하세요."
        )

    try:
        resp = requests.get(
            "https://openapi.naver.com/v1/search/news.json",
            headers={
                "X-Naver-Client-Id": client_id,
                "X-Naver-Client-Secret": client_secret,
            },
            params={
                "query": query,
                "display": min(display, 100),
                "start": start,
                "sort": sort,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        # 응답 JSON 구조 검증
        if not isinstance(data, dict):
            log("WARN", f"API 응답 형식 이상: '{query}' | type={type(data)}")
            return []

        return data.get("items", [])

    except requests.exceptions.Timeout:
        log("WARN", f"API 타임아웃: '{query}'")
        return []
    except requests.exceptions.HTTPError as e:
        log("WARN", f"API HTTP 에러: '{query}' | {e}")
        return []
    except requests.exceptions.ConnectionError:
        log("WARN", f"API 연결 실패: '{query}' | 네트워크 확인 필요")
        return []
    except requests.exceptions.RequestException as e:
        log("WARN", f"API 요청 실패: '{query}' | {e}")
        return []
    except ValueError as e:
        log("WARN", f"API JSON 파싱 실패: '{query}' | {e}")
        return []


def collect_group(queries: list, query_group: str, target: int, seen_links: set) -> list:
    """
    쿼리 목록으로 기사를 수집하고 레이블을 태깅한다.

    수집 전략:
    - 쿼리 목록을 셔플하여 특정 쿼리 편향을 방지한다
    - sort=date와 sort=sim을 번갈아 사용하여 다양성을 확보한다
    - 중복 기사는 seen_links 세트로 필터링한다
    - 목표 건수에 도달하면 조기 종료한다

    Args:
        queries:     검색 쿼리 리스트
        query_group: "official" / "non_official" / "ood"
        target:      목표 수집 건수
        seen_links:  이미 수집한 기사 링크 세트 (중복 방지용)

    Returns:
        수집된 기사 딕셔너리 리스트
    """
    collected = []
    shuffled = queries.copy()
    random.shuffle(shuffled)

    # sort 옵션을 번갈아 사용하여 다양성 확보
    # date: 최신 기사 위주, sim: 키워드 관련도 높은 기사 위주
    sort_options = ["date", "sim"]

    for sort_idx, query in enumerate(shuffled):
        if len(collected) >= target:
            break

        # 쿼리마다 date/sim 번갈아 사용
        current_sort = sort_options[sort_idx % len(sort_options)]
        items = fetch_naver_news(query, display=DISPLAY_PER_QUERY, sort=current_sort)
        added = 0

        for item in items:
            if len(collected) >= target:
                break

            # 원문 링크 우선, 없으면 네이버 링크 사용
            original_link = item.get("originallink", "")
            link = item.get("link", "")
            unique_key = original_link or link

            # 중복 기사 필터링
            if not unique_key or unique_key in seen_links:
                continue
            seen_links.add(unique_key)

            # 제목과 본문(description) 정제
            title = clean_text(item.get("title", ""))
            content = clean_text(item.get("description", ""))

            # 빈 제목이나 빈 본문은 학습 데이터로 부적합
            if not title or not content:
                continue

            # 도메인 추출 및 레이블 결정
            domain = extract_domain(original_link)
            label, reason = assign_label(domain, query_group)

            collected.append({
                "title": title,
                "content": content,
                "source": domain,
                "originallink": original_link,
                "official_label": label,
                "label_reason": reason,
            })
            added += 1

        log("INFO", f"  [{query_group:12s}] '{query}' ({current_sort}): "
            f"+{added}건 (누적 {len(collected)}건)")
        time.sleep(REQUEST_DELAY)

    return collected


def save_to_csv(articles: list, output_path: str):
    """
    수집한 기사를 CSV 파일로 저장한다.

    CSV 컬럼:
    - title:          정제된 기사 제목
    - content:        정제된 기사 본문 (네이버 API description)
    - source:         원문 도메인
    - originallink:   원문 링크 URL
    - official_label: 공식성 레이블 (1=공식, 0=비공식, -1=미확정)
    - label_reason:   레이블 결정 근거

    encoding은 utf-8-sig를 사용한다.
    이유: 엑셀에서 한글이 깨지는 것을 방지한다.
    """
    if not articles:
        log("WARN", "저장할 데이터가 없습니다.")
        return

    # 저장 경로의 디렉토리가 없으면 자동 생성
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "title", "content", "source", "originallink",
        "official_label", "label_reason",
    ]

    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(articles)

    log("INFO", f"CSV 저장 완료: {output_path} ({len(articles)}건)")


def print_summary(articles: list, mode: str):
    """
    수집 결과 요약을 터미널에 출력한다.
    레이블 분포, 결정 방식별 비율, 상위 출처 도메인을 보여준다.
    """
    from collections import Counter

    # 레이블별 건수 집계
    label_1 = sum(1 for a in articles if a["official_label"] == 1)
    label_0 = sum(1 for a in articles if a["official_label"] == 0)
    label_unk = sum(1 for a in articles if a["official_label"] == -1)

    # 출처 도메인별 건수 집계
    sources = Counter(a["source"] for a in articles)

    # 레이블 결정 방식별 건수 집계
    reason_types = Counter()
    for a in articles:
        r = a.get("label_reason", "")
        if "공식 도메인" in r:
            reason_types["도메인 확정(공식)"] += 1
        elif "비공식 도메인" in r:
            reason_types["도메인 확정(비공식)"] += 1
        elif "공식 쿼리" in r:
            reason_types["쿼리 기반(공식)"] += 1
        elif "비공식 쿼리" in r:
            reason_types["쿼리 기반(비공식)"] += 1
        elif "OOD" in r:
            reason_types["OOD 미확정"] += 1

    print()
    print("=" * 65)
    print(f"  수집 완료 요약 [{mode} 모드]")
    print("=" * 65)
    print(f"  총 수집 건수     : {len(articles)}건")
    print(f"  공식(label=1)    : {label_1}건")
    print(f"  비공식(label=0)  : {label_0}건")
    if label_unk > 0:
        print(f"  미확정(label=-1) : {label_unk}건  ← 수동 검수 후 사용 권장")

    # 클래스 비율 계산 (불균형 경고용)
    total_labeled = label_1 + label_0
    if total_labeled > 0:
        ratio_1 = label_1 / total_labeled * 100
        ratio_0 = label_0 / total_labeled * 100
        print(f"\n  클래스 비율: 공식 {ratio_1:.1f}% / 비공식 {ratio_0:.1f}%")
        if abs(ratio_1 - ratio_0) > 20:
            print("  ⚠ 클래스 불균형 감지! class_weight 조정 권장")

    print()
    print("  레이블 결정 방식:")
    for rt, cnt in reason_types.most_common():
        print(f"    {rt:30s} {cnt}건")

    print()
    print("  상위 출처 도메인 10개:")
    for domain, cnt in sources.most_common(10):
        display_domain = domain if domain else "(도메인 없음)"
        print(f"    {display_domain:40s} {cnt}건")

    print("=" * 65)


# ═════════════════════════════════════════════════════════════
# 메인 실행 함수
# ═════════════════════════════════════════════════════════════

def main():
    """
    CLI 진입점.

    사용법:
      python collect_data.py --mode train --output data/train_data.csv --target 6000
      python collect_data.py --mode ood --output data/ood_test.csv --target 200
    """
    parser = argparse.ArgumentParser(
        description="네이버 뉴스 수집 + 도메인/쿼리 혼합 레이블링",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "ood"],
        help="train: 학습 데이터 수집 / ood: OOD 테스트셋 수집",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="저장 경로 (미지정 시 모드별 기본 경로 사용)",
    )
    parser.add_argument(
        "--target", type=int, default=TARGET_COUNT,
        help="목표 수집 건수",
    )
    args = parser.parse_args()

    # 저장 경로 기본값 설정
    if args.output is None:
        args.output = OOD_OUTPUT if args.mode == "ood" else DEFAULT_OUTPUT

    # ── API 키 사전 검증 ─────────────────────────────────────
    # 수집 시작 전에 키가 있는지 확인하여 불필요한 대기를 방지한다
    naver_id = os.getenv("NAVER_CLIENT_ID") or os.getenv("naver_client_id")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET") or os.getenv("naver_client_secret")
    if not naver_id or not naver_secret:
        log("ERROR", ".env에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET이 없습니다.")
        log("ERROR", ".env.example을 참고하여 .env 파일을 생성하세요.")
        raise SystemExit(1)

    # 중복 방지용 세트 (공식/비공식 그룹 간 중복도 방지)
    seen_links = set()

    if args.mode == "train":
        # ── 학습 데이터 수집 모드 ────────────────────────────
        # 공식/비공식 각각 절반씩 수집하여 클래스 균형을 맞춘다
        half = args.target // 2

        log("INFO", "=" * 65)
        log("INFO", f"학습 데이터 수집 시작")
        log("INFO", f"  목표: {args.target}건 (공식 {half} + 비공식 {half})")
        log("INFO", f"  저장: {args.output}")
        log("INFO", f"  공식 쿼리: {len(OFFICIAL_QUERIES)}개")
        log("INFO", f"  비공식 쿼리: {len(NON_OFFICIAL_QUERIES)}개")
        log("INFO", "=" * 65)

        log("INFO", f"[1/2] 공식 기사 수집 (목표: {half}건)")
        official = collect_group(OFFICIAL_QUERIES, "official", half, seen_links)
        log("INFO", f"[1/2] 완료: {len(official)}건")

        log("INFO", f"[2/2] 비공식 기사 수집 (목표: {half}건)")
        non_official = collect_group(NON_OFFICIAL_QUERIES, "non_official", half, seen_links)
        log("INFO", f"[2/2] 완료: {len(non_official)}건")

        # 합치고 셔플 (학습 시 순서 편향 방지)
        all_articles = official + non_official
        random.shuffle(all_articles)

    else:
        # ── OOD 테스트셋 수집 모드 ───────────────────────────
        # 학습 쿼리와 완전히 다른 일반 키워드로 수집한다
        # label=-1 기사는 수동 검수 후 사용을 권장한다
        log("INFO", "=" * 65)
        log("INFO", f"OOD 테스트셋 수집 시작")
        log("INFO", f"  목표: {args.target}건")
        log("INFO", f"  저장: {args.output}")
        log("INFO", "  label=-1 기사는 수동 검수 후 사용 권장")
        log("INFO", "=" * 65)
        all_articles = collect_group(OOD_QUERIES, "ood", args.target, seen_links)

    # ── 수집 결과 검증 ───────────────────────────────────────
    if not all_articles:
        log("ERROR", "수집된 기사가 0건입니다.")
        log("ERROR", "네트워크 연결 또는 API 키를 확인하세요.")
        raise SystemExit(1)

    # ── 요약 출력 및 저장 ────────────────────────────────────
    print_summary(all_articles, args.mode)
    save_to_csv(all_articles, args.output)

    # ── 다음 단계 안내 ───────────────────────────────────────
    print()
    if args.mode == "train":
        log("INFO", "다음 단계:")
        log("INFO", f"  1. 학습 시작:")
        log("INFO", f"     python main.py --mode train --train-path {args.output}")
        log("INFO", f"  2. OOD 테스트셋 수집 (실제 성능 측정용):")
        log("INFO", f"     python collect_data.py --mode ood --target 200")
    else:
        log("INFO", "OOD 테스트셋 준비 완료.")
        log("INFO", "학습 완료 후 아래 명령으로 OOD 평가를 실행하세요:")
        log("INFO", f"  python main.py --mode train --train-path data/train_data.csv")
        log("INFO", "  (학습 완료 시 OOD 평가가 자동 실행됩니다)")


if __name__ == "__main__":
    main()
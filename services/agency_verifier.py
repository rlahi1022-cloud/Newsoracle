"""
services/agency_verifier.py
────────────────────────────
기사 본문 자체를 분석하여 공식성을 판단한다.
하드코딩 도메인 매핑 없이 본문 내용으로만 판별.

판단 기준 4가지:
1. 공식 인용 표현: "발표했다", "보도자료에 따르면" 등
2. 비공식 표현 페널티: "소식통에 따르면", "알려졌다" 등
3. 공식 기관명이 주어로 직접 등장 여부
4. 교차 보도 분석: 동일 키워드를 여러 언론사가 보도했는지
"""

import os
import re
import requests
from dotenv import load_dotenv
from config import OFFICIAL_DOMAINS, ORGANIZATION_PATTERNS
from logger import get_logger

load_dotenv()
logger = get_logger("agency_verifier")

# 공식성 확정 임계값
AGENCY_VERIFIED_THRESHOLD = 0.5

# ─────────────────────────────────────────
# 1. 공식 인용 표현
# 기관이 직접 발표/입장을 밝힌 경우
# ─────────────────────────────────────────
OFFICIAL_CITATION_PATTERNS = [
    r"발표했다", r"발표했습니다", r"발표에 따르면",
    r"밝혔다", r"밝혔습니다", r"밝힌 바에 따르면",
    r"공식 발표", r"공식 입장", r"공식 확인",
    r"보도자료", r"보도자료에 따르면",
    r"입장문", r"성명을 통해", r"공식 성명",
    r"브리핑", r"기자회견",
    r"공고했다", r"공지했다",
    r"의결했다", r"결정했다",
    r"고시했다", r"시행했다", r"발효됐다",
    r"공식 채널", r"공식 답변",
    r"공식 통계", r"공식 집계",
    r"직접 밝혔다", r"공식 인정",
]

# ─────────────────────────────────────────
# 2. 비공식 표현 (페널티)
# 추측/분석/카더라성 표현
# ─────────────────────────────────────────
NON_OFFICIAL_PATTERNS = [
    r"소식통에 따르면", r"익명의", r"관계자에 따르면",
    r"업계에서는", r"알려졌다", r"것으로 보인다",
    r"가능성이 높다", r"전망이다", r"관측된다",
    r"루머", r"의혹", r"카더라", r"찌라시",
    r"폭로", r"주장했다", r"주장이 나왔다",
    r"제기됐다", r"논란이", r"논란을",
    r"열애설", r"스캔들", r"추정된다", r"추측",
    r"~것으로 알려져", r"~것으로 전해져",
    r"내부 관계자", r"복수의 관계자",
]

# ─────────────────────────────────────────
# 3. 공식 기관명 목록
# 본문에 주어로 등장하는지 확인
# ─────────────────────────────────────────
OFFICIAL_ORG_NAMES = [
    "기획재정부", "국토교통부", "보건복지부", "교육부",
    "행정안전부", "고용노동부", "환경부", "과학기술정보통신부",
    "금융위원회", "금융감독원", "국회", "한국은행",
    "질병관리청", "식품의약품안전처", "국세청", "경찰청",
    "법무부", "외교부", "통일부", "국방부",
    "문화체육관광부", "농림축산식품부", "산업통상자원부",
    "중소벤처기업부", "여성가족부", "해양수산부",
    "공정거래위원회", "방송통신위원회", "국민권익위원회",
    "감사원", "헌법재판소", "대법원", "검찰청",
    "대통령실", "국무조정실", "기상청", "통계청",
    "특허청", "조달청", "병무청", "소방청",
    "해양경찰청", "한국수자원공사", "한국전력",
    "한국도로공사", "한국토지주택공사",
    "국민건강보험공단", "국민연금공단",
    "근로복지공단", "한국산업인력공단",
    "연합뉴스",  # 국가 공식 통신사
]

# 기관명 + 주어 조사 패턴
ORG_SUBJECT_SUFFIXES = ["은", "는", "이", "가", "에서", "측은", "측이", "측에서"]


def extract_org_from_title(title: str) -> str:
    """
    기사 제목에서 기관명을 추출한다.
    대괄호/소괄호 안 기관명 우선 추출.

    예: [한국은행] 기준금리 → 한국은행
        (기획재정부) 발표 → 기획재정부

    Args:
        title: 정제된 기사 제목
    Returns:
        감지된 기관명 또는 빈 문자열
    """
    if not title:
        return ""

    # 대괄호/소괄호 안 추출
    bracket_matches = re.findall(r"[\[（\(]([^\]\）\)]{1,20})[\]）\)]", title)
    for match in bracket_matches:
        for org in OFFICIAL_ORG_NAMES:
            if org in match:
                return org

    # 제목 시작 부분 기관명 매칭
    for org in OFFICIAL_ORG_NAMES:
        if title.startswith(org):
            return org

    # 기관명 패턴 (부/처/청/원/위원회 등)
    for pattern in ORGANIZATION_PATTERNS:
        full_pattern = r"[가-힣]{2,10}" + pattern
        matches = re.findall(full_pattern, title)
        if matches:
            return matches[0]

    return ""


def score_official_citations(text: str) -> float:
    """
    본문에서 공식 인용 표현이 얼마나 등장하는지 점수화한다.
    3개 이상 등장하면 만점 1.0

    Args:
        text: 기사 제목 + 본문
    Returns:
        0.0 ~ 1.0
    """
    if not text:
        return 0.0
    count = sum(1 for p in OFFICIAL_CITATION_PATTERNS if re.search(p, text))
    return round(min(count / 3.0, 1.0), 4)


def score_non_official_expressions(text: str) -> float:
    """
    본문에서 비공식 표현이 얼마나 등장하는지 점수화한다.
    높을수록 공식성 낮음 (페널티).
    2개 이상이면 최대 페널티

    Args:
        text: 기사 제목 + 본문
    Returns:
        0.0 ~ 1.0 (페널티 점수)
    """
    if not text:
        return 0.0
    count = sum(1 for p in NON_OFFICIAL_PATTERNS if re.search(p, text))
    return round(min(count / 2.0, 1.0), 4)


def score_org_as_subject(text: str) -> float:
    """
    공식 기관명이 주어로 직접 등장하는지 점수화한다.

    "한국은행은 발표했다" → 주어 등장 ✅ → 1.0
    "한국은행 관련 논란" → 주어 아님 ❌ → 0.0

    Args:
        text: 기사 제목 + 본문
    Returns:
        0.0 또는 1.0
    """
    if not text:
        return 0.0
    for org in OFFICIAL_ORG_NAMES:
        if org not in text:
            continue
        for suffix in ORG_SUBJECT_SUFFIXES:
            if f"{org}{suffix}" in text:
                return 1.0
    return 0.0


def score_cross_coverage(query: str, current_link: str) -> float:
    """
    교차 보도 분석:
    동일 키워드를 여러 언론사가 보도했는지 확인한다.

    여러 언론사가 동시에 보도 → 공신력 있는 사건일 가능성 높음
    한 매체만 단독 보도 → 공신력 낮을 수 있음

    처리:
    1. 쿼리로 네이버 재검색
    2. 수집된 기사들의 도메인 다양성 측정
    3. 도메인 종류가 많을수록 높은 점수

    Args:
        query:        원래 검색 쿼리
        current_link: 현재 기사 링크 (중복 제외용)
    Returns:
        0.0 ~ 1.0
    """
    client_id = os.getenv("naver_client_id")
    client_secret = os.getenv("naver_client_secret")

    if not client_id or not client_secret:
        return 0.0

    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {"query": query, "display": 20, "sort": "date"}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res.raise_for_status()
        items = res.json().get("items", [])

        if not items:
            return 0.0

        # 현재 기사 제외하고 도메인 종류 수집
        from urllib.parse import urlparse
        domains = set()
        for item in items:
            link = item.get("originallink", "") or item.get("link", "")
            if link == current_link:
                continue
            try:
                domain = urlparse(link).netloc.lower()
                if domain:
                    domains.add(domain)
            except Exception:
                continue

        # 도메인 종류가 5개 이상이면 만점
        cross_score = round(min(len(domains) / 5.0, 1.0), 4)
        logger.debug(f"교차 보도 분석 | query={query} | 도메인 수={len(domains)} | score={cross_score}")
        return cross_score

    except Exception as e:
        logger.error(f"교차 보도 분석 실패: {e}")
        return 0.0


def generate_verification_message(
    org_name: str,
    citation_score: float,
    non_official_score: float,
    org_subject_score: float,
    cross_score: float,
    agency_score: float,
    is_official_domain: bool,
) -> str:
    """
    분석 결과를 바탕으로 검증 메시지를 생성한다.

    Args:
        org_name:           감지된 기관명
        citation_score:     공식 인용 표현 점수
        non_official_score: 비공식 표현 페널티
        org_subject_score:  기관명 주어 등장 점수
        cross_score:        교차 보도 점수
        agency_score:       최종 점수
        is_official_domain: 공식 도메인 여부
    Returns:
        검증 메시지 문자열
    """
    if is_official_domain:
        return f"공식 기관({org_name}) 도메인의 직접 발표 기사입니다."

    if agency_score >= 0.7:
        if org_subject_score > 0 and citation_score > 0:
            return f"'{org_name}'의 공식 발표를 직접 인용하였고 복수 언론사가 교차 보도한 신뢰도 높은 기사입니다."
        elif cross_score >= 0.6:
            return f"복수의 언론사가 동일 주제를 교차 보도한 기사입니다."
        else:
            return f"공식 기관 관련 표현이 다수 포함된 신뢰도 높은 기사입니다."
    elif agency_score >= AGENCY_VERIFIED_THRESHOLD:
        if non_official_score >= 0.5:
            return f"공식성 표현과 추측성 표현이 혼재된 기사입니다. 원문 확인을 권장합니다."
        return f"공식성 표현이 일부 포함된 기사입니다."
    else:
        if non_official_score >= 0.5:
            return f"추측성/분석성 표현이 다수 포함된 기사입니다. 공식 발표 기사가 아닐 수 있습니다."
        return f"공식 기관의 직접 발표 여부를 확인하기 어렵습니다."


def verify_agency(article: dict, query: str) -> dict:
    """
    단일 기사를 4가지 기준으로 분석하여 공식성을 판단한다.

    처리 순서:
    1. 제목에서 기관명 추출
    2. 본문 공식 인용 표현 점수화
    3. 본문 비공식 표현 페널티 점수화
    4. 기관명 주어 등장 여부
    5. 교차 보도 분석
    6. 종합 agency_score 산출

    Args:
        article: 전처리된 기사 딕셔너리
        query:   원래 검색 쿼리
    Returns:
        기관 검증 결과 딕셔너리
    """
    domain       = article.get("domain", "")
    title        = article.get("title", "")
    content      = article.get("content", "")
    originallink = article.get("originallink", "")
    full_text    = f"{title} {content}"

    # 기관명 추출 (제목 기반)
    org_name = extract_org_from_title(title)

    # 공식 도메인 여부 (.go.kr / .or.kr 등)
    is_official_domain = any(od in domain for od in OFFICIAL_DOMAINS)

    # 본문 분석 4가지
    citation_score     = score_official_citations(full_text)
    non_official_score = score_non_official_expressions(full_text)
    org_subject_score  = score_org_as_subject(full_text)
    cross_score        = score_cross_coverage(query, originallink)

    # 최종 agency_score 계산
    # 공식 인용 + 기관 주어 + 교차 보도 → 점수 ↑
    # 비공식 표현 → 페널티
    raw_score = (
        citation_score     * 0.30
        + org_subject_score  * 0.25
        + cross_score        * 0.25
        - non_official_score * 0.30
    )

    # 공식 도메인이면 보너스
    if is_official_domain:
        raw_score += 0.4

    agency_score = round(max(0.0, min(raw_score, 1.0)), 4)

    # 검증 메시지 생성
    message = generate_verification_message(
        org_name, citation_score, non_official_score,
        org_subject_score, cross_score, agency_score, is_official_domain
    )

    logger.debug(
        f"기관 검증 | org={org_name} "
        f"citation={citation_score} non_official={non_official_score} "
        f"org_subject={org_subject_score} cross={cross_score} "
        f"agency_score={agency_score}"
    )

    return {
        "org_name":             org_name,
        "report_count":         0,
        "agency_score":         agency_score,
        "is_official_domain":   is_official_domain,
        "citation_score":       citation_score,
        "non_official_score":   non_official_score,
        "org_subject_score":    org_subject_score,
        "cross_score":          cross_score,
        "verification_message": message,
    }


def verify_agency_batch(articles: list, query: str) -> list:
    """
    기사 목록 전체에 기관 신뢰도 검증을 수행한다.

    교차 보도 분석은 쿼리당 1회만 API 호출하여
    모든 기사에 동일한 cross_score 적용 (API 절약).

    Args:
        articles: 전처리된 기사 딕셔너리 리스트
        query:    원래 검색 쿼리
    Returns:
        기관 검증 결과 딕셔너리 리스트
    """
    if not articles:
        logger.warning("기관 검증 입력이 비어 있음")
        return []

    logger.info(f"기관 신뢰도 검증 시작 | {len(articles)}건 | query={query}")

    # 교차 보도 점수는 쿼리당 1회만 계산 (API 호출 절약)
    shared_cross_score = score_cross_coverage(query, "")
    logger.debug(f"교차 보도 공통 점수: {shared_cross_score}")

    results = []
    for article in articles:
        domain       = article.get("domain", "")
        title        = article.get("title", "")
        content      = article.get("content", "")
        full_text    = f"{title} {content}"

        org_name           = extract_org_from_title(title)
        is_official_domain = any(od in domain for od in OFFICIAL_DOMAINS)
        citation_score     = score_official_citations(full_text)
        non_official_score = score_non_official_expressions(full_text)
        org_subject_score  = score_org_as_subject(full_text)

        raw_score = (
            citation_score     * 0.30
            + org_subject_score  * 0.25
            + shared_cross_score * 0.25
            - non_official_score * 0.30
        )
        if is_official_domain:
            raw_score += 0.4

        agency_score = round(max(0.0, min(raw_score, 1.0)), 4)

        message = generate_verification_message(
            org_name, citation_score, non_official_score,
            org_subject_score, shared_cross_score, agency_score, is_official_domain
        )

        results.append({
            "org_name":             org_name,
            "report_count":         0,
            "agency_score":         agency_score,
            "is_official_domain":   is_official_domain,
            "citation_score":       citation_score,
            "non_official_score":   non_official_score,
            "org_subject_score":    org_subject_score,
            "cross_score":          shared_cross_score,
            "verification_message": message,
        })

    verified_count = sum(
        1 for r in results if r["agency_score"] >= AGENCY_VERIFIED_THRESHOLD
    )
    logger.info(
        f"기관 검증 완료 | 총 {len(results)}건 중 공식성 확인 {verified_count}건"
    )

    return results
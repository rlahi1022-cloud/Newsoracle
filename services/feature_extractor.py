"""
services/feature_extractor.py
──────────────────────────────
전처리된 기사에서 모델 입력에 필요한 특징(feature)을 추출한다.

[v2 재설계 이유]
기존 구조의 근본 문제 3가지:
1. link_score:      네이버 API originallink는 항상 존재 → 전 기사 1.0 고정 → 변별력 0
2. text_length:     API description은 항상 80~100자 발췌 → 전 기사 동일 → 변별력 0
3. domain_score:    OFFICIAL_DOMAINS가 .go.kr 위주라 실제 언론사 도메인 전부 미매칭

[v2 변경 내용]
- link_score, text_length_score 제거
- domain_score → 3등급 신뢰도 체계로 교체
  1등급(1.0): .go.kr/.or.kr 정부/공공기관
  2등급(0.7): yna.co.kr 등 국가 공식 통신사
  3등급(0.4): 주요 종합일간지 (조선/중앙/동아/한겨레/경향)
  4등급(0.2): 경제전문지 (한국경제/매일경제/이데일리 등)
  미매칭(0.0): 나머지
- quote_score 추가:  기사 내 직접인용(" ") 포함 여부
- stat_score 추가:   수치/통계 데이터 포함 여부 (%, 억원, 달러, 건)
- title_format 추가: 제목 구조 분석 ([기관명] 형식, 발표성 동사 등)
"""

import re
from logger import get_logger

logger = get_logger("feature_extractor")

# ─────────────────────────────────────────────────────────────
# 도메인 신뢰도 등급 테이블
# 실제 네이버 뉴스 API가 반환하는 도메인 기준으로 정의
# ─────────────────────────────────────────────────────────────
DOMAIN_GRADE = {
    # 1등급 (1.0): 정부/공공기관 공식 도메인
    "grade_1": [
        ".go.kr", ".or.kr", ".re.kr", ".ac.kr",
    ],
    # 2등급 (0.7): 국가 공식 통신사
    "grade_2": [
        "yna.co.kr", "yonhapnewstv.co.kr",
    ],
    # 3등급 (0.5): 주요 종합일간지 (공신력 높음)
    "grade_3": [
        "chosun.com", "joongang.co.kr", "donga.com",
        "hani.co.kr", "khan.co.kr", "kmib.co.kr",
        "seoul.co.kr", "munhwa.com",
    ],
    # 4등급 (0.35): 경제/IT 전문지
    "grade_4": [
        "hankyung.com", "mk.co.kr", "edaily.co.kr",
        "mt.co.kr", "sedaily.com", "etnews.com",
        "zdnet.co.kr", "bloter.net", "financialpost.co.kr",
    ],
    # 5등급 (0.2): 주요 방송/인터넷 뉴스
    "grade_5": [
        "news1.kr", "newsis.com", "newspim.com",
        "ytn.co.kr", "mbn.co.kr", "sbs.co.kr",
        "kbs.co.kr", "mbc.co.kr", "jtbc.co.kr",
        "pressian.com", "ohmynews.com",
        "hankookilbo.com", "heraldcorp.com",
    ],
}

# 등급별 점수 매핑
GRADE_SCORES = {
    "grade_1": 1.0,
    "grade_2": 0.7,
    "grade_3": 0.5,
    "grade_4": 0.35,
    "grade_5": 0.2,
}

# ─────────────────────────────────────────────────────────────
# 공식성 표현 키워드 (feature_extractor 자체 정의)
# config의 OFFICIAL_EXPRESSIONS와 별도로 더 세밀하게 분류
# ─────────────────────────────────────────────────────────────
OFFICIAL_EXPR_STRONG = [
    # 강한 공식 표현 (기관이 직접 발표하는 동사)
    "발표했다", "발표했습니다", "발표에 따르면",
    "밝혔다", "밝혔습니다",
    "공식 발표", "공식 입장", "보도자료",
    "입장문", "성명", "브리핑", "기자회견",
    "의결했다", "결정했다", "고시했다",
    "공고했다", "공지했다",
    "협약 체결", "업무협약", "MOU 체결",
]

OFFICIAL_EXPR_WEAK = [
    # 약한 공식 표현 (공식성 암시)
    "발표", "공고", "공지", "시행",
    "공식", "공시", "실적",
    "출시", "론칭",
]

# 수치/통계 패턴: 공식 발표에는 구체적 수치가 동반됨
STAT_PATTERNS = [
    r"\d+(\.\d+)?%",         # 퍼센트 (예: 3.5%)
    r"\d+억\s*원",            # 억원
    r"\d+조\s*원",            # 조원
    r"\d+만\s*원",            # 만원
    r"\d+달러",               # 달러
    r"\d+건",                 # 건수
    r"\d+명",                 # 인원수
    r"전년\s*(대비|比)",       # 전년 대비
    r"전분기\s*(대비|比)",     # 전분기 대비
    r"역대\s*(최대|최고|최저)", # 역대 기록
    r"\d+위",                 # 순위
]

# 제목 공식 구조 패턴
TITLE_OFFICIAL_PATTERNS = [
    r"^[\[（\(]",             # [기관명] 으로 시작하는 제목
    r"보도자료",               # 보도자료 명시
    r"공식\s*(발표|입장|확인)",# 공식 발표/입장/확인
    r"업무협약|MOU|파트너십",  # 협약 관련
    r"실적\s*(발표|공시)",     # 실적 발표
]


def score_domain_grade(domain: str) -> float:
    """
    도메인의 신뢰도 등급을 점수로 반환한다.

    3등급 체계:
    - 1등급(.go.kr 등): 1.0
    - 2등급(연합뉴스):  0.7
    - 3등급(종합일간지): 0.5
    - 4등급(경제전문지): 0.35
    - 5등급(주요 인터넷뉴스): 0.2
    - 미매칭: 0.0

    기존 domain_score가 0.0을 남발했던 이유:
    언론사 도메인(news1.kr 등)이 OFFICIAL_DOMAINS에 없어서 전부 미매칭.
    → 실제 뉴스 수집 환경에 맞게 5등급 체계로 세분화.

    Args:
        domain: 기사 원문 도메인 (예: www.yna.co.kr)
    Returns:
        0.0 ~ 1.0 도메인 신뢰도 점수
    """
    if not domain:
        return 0.0

    for grade, domains in DOMAIN_GRADE.items():
        for d in domains:
            if d in domain:
                return GRADE_SCORES[grade]

    return 0.0


def score_official_expression(text: str) -> float:
    """
    공식성 표현의 강도를 점수화한다.

    강한 표현(발표했다, 보도자료 등)과 약한 표현(발표, 공고 등)을
    구분하여 가중치를 다르게 적용한다.

    강한 표현 1개 = 0.5점, 2개 이상 = 1.0점
    약한 표현만 있을 경우 = 0.3점

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        0.0 ~ 1.0
    """
    if not text:
        return 0.0

    strong_count = sum(1 for e in OFFICIAL_EXPR_STRONG if e in text)
    weak_count   = sum(1 for e in OFFICIAL_EXPR_WEAK   if e in text)

    if strong_count >= 2:
        return 1.0
    elif strong_count == 1:
        return 0.5 + min(weak_count * 0.1, 0.5)
    elif weak_count >= 2:
        return 0.3
    elif weak_count == 1:
        return 0.15
    return 0.0


def score_organization_name(text: str) -> float:
    """
    기사에 공식 기관명/대기업명이 등장하는지 점수화한다.

    단순 포함 여부가 아니라 등장 기관의 수에 비례.
    기관이 많이 언급될수록 공식 발표 기사일 가능성 높음.

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        0.0 ~ 1.0
    """
    if not text:
        return 0.0

    # 공공기관 접미사 패턴 (부/처/청/원/위원회/공단/공사)
    gov_pattern = r"[가-힣]{2,8}(부|처|청|원|위원회|공단|공사|은행|청장|장관)"
    gov_matches = re.findall(gov_pattern, text)

    # 주요 대기업명 (직접 포함 여부)
    major_corps = [
        "현대자동차", "현대차", "기아", "삼성전자", "삼성",
        "LG전자", "LG", "SK하이닉스", "SK", "포스코",
        "한국은행", "연합뉴스", "국회", "대법원", "대통령실",
        "BNK부산은행", "BNK", "한온시스템",
    ]
    corp_count = sum(1 for c in major_corps if c in text)

    total_count = len(gov_matches) + corp_count

    if total_count >= 3:
        return 1.0
    elif total_count == 2:
        return 0.7
    elif total_count == 1:
        return 0.4
    return 0.0


def score_direct_quote(text: str) -> float:
    """
    기사에 직접 인용문(" ")이 포함되어 있는지 점수화한다.

    공식 발표 기사는 기관/기업의 공식 발언을 인용하는 경우가 많음.
    예: 기획재정부는 "경기 회복세가 지속되고 있다"고 밝혔다.

    단, 연예 기사도 인용이 많으므로 단독 피처로 쓰지 않고
    다른 피처와 함께 앙상블에 사용.

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        0.0 / 0.5 / 1.0
    """
    if not text:
        return 0.0

    # 큰따옴표/작은따옴표 인용문 패턴
    quote_patterns = [
        r'"[^"]{5,100}"',    # "발언 내용"
        r'"[^"]{5,100}"',    # 유니코드 따옴표
        r"'[^']{5,100}'",    # '발언 내용'
    ]

    count = sum(len(re.findall(p, text)) for p in quote_patterns)

    if count >= 2:
        return 1.0
    elif count == 1:
        return 0.5
    return 0.0


def score_statistics(text: str) -> float:
    """
    기사에 구체적인 수치/통계가 포함되어 있는지 점수화한다.

    공식 발표 기사는 구체적 수치를 동반하는 경우가 많음.
    예: "기준금리를 연 3.5%로 동결", "경상수지 232억 달러 역대 최대"

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        0.0 ~ 1.0
    """
    if not text:
        return 0.0

    count = sum(1 for p in STAT_PATTERNS if re.search(p, text))

    if count >= 3:
        return 1.0
    elif count == 2:
        return 0.7
    elif count == 1:
        return 0.4
    return 0.0


def score_title_format(title: str) -> float:
    """
    제목 구조가 공식 발표 형식인지 점수화한다.

    공식 발표 기사 제목 패턴:
    - [기관명] 내용 (대괄호 기관명으로 시작)
    - '발표', '공고', '공시' 등 공식 동사 포함
    - "수치 + 역대 최대/최고/최저" 구조

    Args:
        title: 정제된 기사 제목
    Returns:
        0.0 ~ 1.0
    """
    if not title:
        return 0.0

    count = sum(1 for p in TITLE_OFFICIAL_PATTERNS if re.search(p, title))

    return min(count * 0.4, 1.0)


def extract_features(article: dict) -> dict:
    """
    전처리된 단일 기사에서 모든 특징을 추출한다.

    [v2 피처 목록]
    - domain_grade_score:   도메인 신뢰도 등급 점수 (0/0.2/0.35/0.5/0.7/1.0)
    - official_expr_score:  공식성 표현 강도 점수
    - org_name_score:       기관/기업명 등장 점수
    - quote_score:          직접 인용문 포함 점수
    - stat_score:           수치/통계 포함 점수
    - title_format_score:   제목 구조 공식성 점수
    - embedding_input:      semantic_similarity 입력용 텍스트
    - classifier_input:     classifier_model 입력용 텍스트

    Args:
        article: 전처리된 기사 딕셔너리
    Returns:
        특징 딕셔너리
    """
    try:
        title        = article.get("title", "")
        content      = article.get("content", "")
        domain       = article.get("domain", "")
        full_text    = f"{title} {content}"

        features = {
            # ── 도메인 신뢰도 (5등급 체계) ────────────────
            "domain_grade_score":  score_domain_grade(domain),

            # ── 공식성 표현 강도 ───────────────────────────
            "official_expr_score": score_official_expression(full_text),

            # ── 기관/기업명 등장 ───────────────────────────
            "org_name_score":      score_organization_name(full_text),

            # ── 직접 인용문 포함 ───────────────────────────
            "quote_score":         score_direct_quote(full_text),

            # ── 수치/통계 데이터 포함 ──────────────────────
            "stat_score":          score_statistics(full_text),

            # ── 제목 구조 공식성 ───────────────────────────
            "title_format_score":  score_title_format(title),

            # ── 모델 입력용 텍스트 ─────────────────────────
            "embedding_input":     full_text,
            "classifier_input":    full_text,
        }
        return features

    except Exception as e:
        logger.error(f"특징 추출 실패: {e} | title={article.get('title', '')}")
        return {}


def extract_features_batch(articles: list) -> list:
    """
    기사 목록 전체에서 특징을 추출한다.

    Args:
        articles: 전처리된 기사 딕셔너리 리스트
    Returns:
        특징 딕셔너리 리스트
    """
    if not articles:
        logger.warning("특징 추출 입력 데이터가 비어 있음")
        return []

    logger.info(f"특징 추출 시작 | {len(articles)}건")
    results = [f for a in articles if (f := extract_features(a))]
    logger.info(f"특징 추출 완료 | {len(results)}건")
    return results
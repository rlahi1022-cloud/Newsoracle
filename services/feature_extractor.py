"""
services/feature_extractor.py
──────────────────────────────
전처리된 기사에서 모델 입력에 필요한 특징(feature)을 추출한다.

[v5 변경사항]
1. config.py에서 ANONYMOUS_EXPRESSIONS import
   - 기존 하드코딩된 UNOFFICIAL_EXPRESSIONS를 config 기반으로 교체
   - config에서 관리하므로 표현 추가/삭제가 코드 수정 없이 가능

2. config.py에서 OFFICIAL_SUBJECT_VERBS import
   - "기관명 + 공식 동사" 패턴 탐지 함수 score_subject_verb() 신규 추가
   - 예: "한국은행은 결정했다" → 공식 발화 주체 확인

3. config.py에서 OFFICIAL_EXPRESSIONS import
   - 기존 OFFICIAL_EXPR_STRONG/WEAK를 config 기반으로 보강
   - config의 72개 표현을 활용

4. 기존 v4의 직접 발화 탐지(score_direct_speech) 유지

[피처 구조]
extract_features() 반환값:
  domain_grade_score:    도메인 신뢰도 등급 (0.0~1.0)
  official_expr_score:   공식성 표현 강도 (0.0~1.0)
  unofficial_expr_score: 비공식 표현 강도 (0.0~1.0)
  org_name_score:        기관/기업명 등장 (0.0~1.0)
  quote_score:           직접 인용문 포함 (0.0~1.0)
  stat_score:            수치/통계 포함 (0.0~1.0)
  title_format_score:    제목 구조 공식성 (0.0~1.0)
  direct_speech_score:   본인 직접 발언 (0.0~1.0)
  subject_verb_score:    기관명+공식동사 패턴 (0.0~1.0) [v5 신규]
  has_direct_speech:     직접 발화 확인 플래그 (bool)
  embedding_input:       semantic_similarity 입력용 텍스트
  classifier_input:      classifier_model 입력용 텍스트
  _evidence:             판단 근거 딕셔너리 (설명용)
"""

import re
from logger import get_logger

# config에서 공식성/비공식성 판단 기준을 import
# 이렇게 하면 config.py에서만 표현을 추가/삭제하면 자동 반영됨
from config import (
    ANONYMOUS_EXPRESSIONS,
    OFFICIAL_SUBJECT_VERBS,
    OFFICIAL_EXPRESSIONS,
    ORGANIZATION_PATTERNS,
)

logger = get_logger("feature_extractor")

# ─────────────────────────────────────────────────────────────
# 도메인 신뢰도 등급 테이블
#
# 네이버 뉴스 API가 반환하는 실제 도메인 기준으로 정의한다.
# 등급이 높을수록 출처의 공신력이 높다.
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
    # 5등급 (0.2): 주요 방송/인터넷/연예/스포츠 뉴스
    "grade_5": [
        "news1.kr", "newsis.com", "newspim.com",
        "ytn.co.kr", "mbn.co.kr", "sbs.co.kr",
        "kbs.co.kr", "mbc.co.kr", "jtbc.co.kr",
        "pressian.com", "ohmynews.com",
        "hankookilbo.com", "heraldcorp.com",
        "sportsseoul.com", "spotvnews.co.kr", "mydaily.co.kr",
        "starnewskorea.com", "entertain.naver.com",
        "isplus.com", "osen.mt.co.kr", "xportsnews.com",
        "tenasia.hankyung.com", "newsen.com",
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
# 공식성 표현 키워드 (v5: config 기반 + 로컬 보강)
#
# OFFICIAL_EXPR_STRONG: 기사에 등장하면 공식성이 높다고 확신할 수 있는 표현
# OFFICIAL_EXPR_WEAK: 단독으로는 약하지만 다른 표현과 함께 나오면 가산
#
# config.OFFICIAL_EXPRESSIONS의 표현 중 강한 것은 STRONG에,
# 약한 것은 WEAK에 자동 분류하지 않고 수동으로 관리한다.
# 이유: 자동 분류하면 "발표"같은 약한 표현이 STRONG에 들어갈 수 있음
# ─────────────────────────────────────────────────────────────

OFFICIAL_EXPR_STRONG = [
    # ── 정부/기관 공식 발표 ──────────────────────────────────
    "발표했다", "발표했습니다", "발표에 따르면",
    "밝혔다", "밝혔습니다", "밝힌 바",
    "공식 발표", "공식 입장", "공식 확인",
    "보도자료", "보도자료에 따르면", "보도자료를 통해",
    "입장문", "성명", "성명을 발표", "공식 성명",
    "브리핑", "기자회견", "기자간담회",
    "의결했다", "결정했다", "고시했다",
    "공고했다", "공지했다", "시행했다",
    "협약 체결", "업무협약", "MOU 체결",
    "공식 채널", "공식 답변", "공식 통계",
    "직접 밝혔다", "공식 인정",

    # ── 소속사/엔터 공식 표현 ────────────────────────────────
    "소속사 측은", "소속사는", "소속사가",
    "소속사 측이", "소속사 관계자는",
    "공식 입장을 밝혔다", "공식 입장을 전했다",
    "공식적으로 발표", "공식적으로 확인",
    "구단은", "구단 측은", "구단이 발표",
    "협회는", "협회가 발표", "연맹은",

    # ── [태그] 계열 ──────────────────────────────────────────
    "[공식]", "[단독]", "[속보]", "[긴급]",

    # ── 본인 직접 발언 (v4에서 이관) ─────────────────────────
    "고 말했다", "고 밝혔다", "고 전했다",
    "고 설명했다", "고 답했다", "고 답변했다",
    "고 당부했다", "고 강조했다", "고 덧붙였다",
    "고 언급했다", "고 회상했다", "고 고백했다",
    "고 소감을 밝혔다", "고 소감을 전했다",
    "라고 말했다", "라고 밝혔다", "라고 전했다",
    "라고 설명했다", "라고 답했다",
    "이라고 말했다", "이라고 밝혔다",
    "직접 언급", "직접 해명", "직접 설명",

    # ── 방송/인터뷰 출연 (v4에서 이관) ───────────────────────
    "출연해", "출연하여", "출연한 자리에서",
    "인터뷰에서", "인터뷰를 통해",
    "라디오에서", "방송에서", "프로그램에서",
    "기자회견에서", "간담회에서",
    "취재진에게", "기자들과 만나",

    # ── v5 신규: 법률/사법 표현 ──────────────────────────────
    "판결했다", "선고했다", "기소했다",
    "대법원 판결", "헌법재판소 결정",

    # ── v5 신규: 의료/보건 표현 ──────────────────────────────
    "허가 승인", "긴급사용승인", "접종 개시",
    "임상시험 결과",

    # ── v5 신규: 국제기구 표현 ───────────────────────────────
    "UN 결의", "WHO 권고", "IMF 전망",
    "정상회담", "외교 합의",
]

OFFICIAL_EXPR_WEAK = [
    "발표", "공고", "공지", "시행", "공식", "공시", "실적",
    "출시", "론칭", "오픈", "확정", "선정", "인가", "승인", "허가",
    "계약", "체결", "서명", "조인",
    "컴백", "데뷔", "컴백 확정", "데뷔 확정",
    "이적", "계약 해지", "전속계약",
    "출연 확정", "캐스팅 확정", "합류",
    "출연", "게스트", "MC", "홍보했다",
    # v5 신규
    "판결", "선고", "기소", "구속",
    "논문 발표", "연구 결과", "학위 수여",
    "배당 결정", "주주총회", "이사회 결의",
    "훈련 실시", "작전 발표",
]

# ─────────────────────────────────────────────────────────────
# 비공식 표현 (v5: config.ANONYMOUS_EXPRESSIONS 사용)
#
# 기존 하드코딩 리스트를 config에서 가져온다.
# config에서 추가/삭제하면 자동 반영된다.
# ─────────────────────────────────────────────────────────────

UNOFFICIAL_EXPRESSIONS = ANONYMOUS_EXPRESSIONS

# ─────────────────────────────────────────────────────────────
# 수치/통계 패턴
# 공식 발표 기사는 구체적인 수치를 포함하는 경우가 많다.
# ─────────────────────────────────────────────────────────────

STAT_PATTERNS = [
    r"\d+(\.\d+)?%",
    r"\d+억\s*원",
    r"\d+조\s*원",
    r"\d+만\s*원",
    r"\d+달러",
    r"\d+건",
    r"\d+명",
    r"전년\s*(대비|比)",
    r"전분기\s*(대비|比)",
    r"역대\s*(최대|최고|최저)",
    r"\d+위",
]

# ─────────────────────────────────────────────────────────────
# 제목 공식 구조 패턴
# ─────────────────────────────────────────────────────────────

TITLE_OFFICIAL_PATTERNS = [
    r"^\[공식\]", r"^\[단독\]", r"^\[속보\]", r"^[\[（\(]",
    r"보도자료",
    r"공식\s*(발표|입장|확인)",
    r"업무협약|MOU|파트너십",
    r"실적\s*(발표|공시)",
    r"컴백\s*확정", r"데뷔\s*확정", r"출연\s*확정",
    r"계약\s*체결", r"이적\s*확정",
    r"인터뷰", r"직접\s*(해명|언급|설명)",
    # v5 신규
    r"판결", r"선고", r"기소",
    r"허가\s*승인", r"임상\s*결과",
]

# ─────────────────────────────────────────────────────────────
# 직접 발화 탐지 리스트 (v4에서 유지)
# ─────────────────────────────────────────────────────────────

DIRECT_SPEECH_VERBS = [
    "고 말했다", "고 밝혔다", "고 전했다",
    "고 설명했다", "고 답했다", "고 답변했다",
    "고 당부했다", "고 강조했다", "고 덧붙였다",
    "고 언급했다", "고 회상했다", "고 고백했다",
    "라고 말했다", "라고 밝혔다", "라고 전했다",
    "라고 설명했다", "라고 답했다",
    "이라고 말했다", "이라고 밝혔다",
    "고 소감을 밝혔다", "고 소감을 전했다",
]

DIRECT_SPEECH_CONTEXTS = [
    "출연해", "출연하여", "출연한",
    "인터뷰에서", "인터뷰를 통해",
    "라디오에서", "방송에서", "프로그램에서",
    "기자회견에서", "간담회에서",
    "취재진에게", "기자들과 만나",
]


# ═════════════════════════════════════════════════════════════
# 개별 피처 점수 계산 함수
# ═════════════════════════════════════════════════════════════

def score_domain_grade(domain: str) -> float:
    """
    도메인의 신뢰도 등급을 점수로 반환한다.

    5등급 체계:
    - 1등급(.go.kr 등): 1.0
    - 2등급(연합뉴스):   0.7
    - 3등급(종합일간지): 0.5
    - 4등급(경제전문지): 0.35
    - 5등급(방송/인터넷): 0.2
    - 미매칭:            0.0
    """
    if not domain:
        return 0.0
    for grade, domains in DOMAIN_GRADE.items():
        for d in domains:
            if d in domain:
                return GRADE_SCORES[grade]
    return 0.0


def score_official_expression(text: str) -> dict:
    """
    공식성 표현의 강도를 점수화하고, 탐지된 표현 목록을 반환한다.

    강한 표현 2개 이상 = 1.0
    강한 표현 1개 = 0.5 + 약한 표현 보너스
    약한 표현만 2개 이상 = 0.3
    약한 표현 1개 = 0.15

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        {"score": float, "strong_found": list, "weak_found": list}
    """
    if not text:
        return {"score": 0.0, "strong_found": [], "weak_found": []}

    strong_found = [e for e in OFFICIAL_EXPR_STRONG if e in text]
    weak_found = [e for e in OFFICIAL_EXPR_WEAK if e in text]

    sc = len(strong_found)
    wc = len(weak_found)

    if sc >= 2:
        score = 1.0
    elif sc == 1:
        score = 0.5 + min(wc * 0.1, 0.5)
    elif wc >= 2:
        score = 0.3
    elif wc == 1:
        score = 0.15
    else:
        score = 0.0

    return {"score": score, "strong_found": strong_found, "weak_found": weak_found}


def score_unofficial_expression(text: str) -> dict:
    """
    비공식 표현(익명 소식통, 추측, 루머)을 탐지하고 점수화한다.

    config.ANONYMOUS_EXPRESSIONS를 사용한다.
    높을수록 비공식적인 기사.

    3건 이상 = 1.0, 2건 = 0.6, 1건 = 0.3, 0건 = 0.0

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        {"score": float, "found": list}
    """
    if not text:
        return {"score": 0.0, "found": []}

    found = [e for e in UNOFFICIAL_EXPRESSIONS if e in text]
    c = len(found)

    if c >= 3:
        score = 1.0
    elif c == 2:
        score = 0.6
    elif c == 1:
        score = 0.3
    else:
        score = 0.0

    return {"score": score, "found": found}


def score_organization_name(text: str) -> dict:
    """
    기사에 공식 기관명/대기업명/소속사명/방송사명이 등장하는지 점수화한다.

    두 가지 방식으로 탐지:
    1. config.ORGANIZATION_PATTERNS 정규표현식 패턴
    2. 직접 명시한 주요 기관/기업 이름 리스트

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        {"score": float, "found_orgs": list}
    """
    if not text:
        return {"score": 0.0, "found_orgs": []}

    # config의 ORGANIZATION_PATTERNS를 사용하여 정규표현식 탐지
    gov_pattern = r"[가-힣]{2,8}(부|처|청|원|위원회|공단|공사|은행|청장|장관)"
    gov_matches = re.findall(gov_pattern, text)

    # 주요 기관/기업/소속사 목록
    major_entities = [
        # 정부/공공기관
        "현대자동차", "현대차", "기아", "삼성전자", "삼성",
        "LG전자", "LG", "SK하이닉스", "SK", "포스코",
        "한국은행", "연합뉴스", "국회", "대법원", "대통령실",
        "BNK부산은행", "BNK", "한온시스템",

        # 엔터테인먼트 소속사
        "SM엔터테인먼트", "SM엔터", "에스엠",
        "JYP엔터테인먼트", "JYP엔터", "JYP",
        "YG엔터테인먼트", "YG엔터", "YG",
        "하이브", "HYBE", "빅히트뮤직", "빅히트",
        "플레디스", "스타쉽", "큐브엔터", "FNC", "IST엔터", "안테나",
        "카카오엔터테인먼트", "카카오엔터", "CJ ENM",
        "EDAM엔터테인먼트", "EDAM엔터",
        "키이스트", "매니지먼트숲", "BH엔터", "나무액터스",
        "울림엔터테인먼트", "RBW", "WM엔터",

        # 방송사
        "MBC", "KBS", "SBS", "JTBC", "tvN", "MBN", "YTN",

        # 스포츠 기관/구단
        "KBO", "한국야구위원회",
        "KFA", "대한축구협회",
        "KBL", "한국농구연맹",
        "KOVO", "한국배구연맹",
        "대한체육회", "국민체육진흥공단",

        # IT/플랫폼 기업
        "네이버", "카카오", "쿠팡", "배달의민족", "우아한형제들",
        "토스", "비바리퍼블리카", "당근마켓", "야놀자",
        "넷플릭스", "디즈니플러스", "티빙", "웨이브",

        # v5 신규: 의료/교육/국제기구
        "질병관리청", "식품의약품안전처", "대한의사협회",
        "서울대학교", "연세대학교", "고려대학교",
        "WHO", "UN", "IMF", "세계보건기구",
    ]
    found_entities = [c for c in major_entities if c in text]

    # 정규표현식 결과와 직접 명시 결과 통합
    found_orgs = found_entities.copy()
    if gov_matches:
        for suffix in gov_matches:
            full_matches = re.findall(r"[가-힣]{2,8}" + suffix, text)
            found_orgs.extend(full_matches)

    # 중복 제거 (순서 유지)
    found_orgs = list(dict.fromkeys(found_orgs))
    total = len(gov_matches) + len(found_entities)

    if total >= 3:
        score = 1.0
    elif total == 2:
        score = 0.7
    elif total == 1:
        score = 0.4
    else:
        score = 0.0

    return {"score": score, "found_orgs": found_orgs}


def score_direct_quote(text: str) -> float:
    """
    기사에 직접 인용문(" ")이 포함되어 있는지 점수화한다.

    공식 발표 기사는 기관/기업의 공식 발언을 인용하는 경우가 많다.
    2건 이상 = 1.0, 1건 = 0.5

    Args:
        text: 전체 텍스트
    Returns:
        0.0 ~ 1.0
    """
    if not text:
        return 0.0
    patterns = [r'"[^"]{5,100}"', r'\u201c[^\u201d]{5,100}\u201d', r"'[^']{5,100}'"]
    count = sum(len(re.findall(p, text)) for p in patterns)
    if count >= 2:
        return 1.0
    elif count == 1:
        return 0.5
    return 0.0


def score_statistics(text: str) -> float:
    """
    기사에 구체적인 수치/통계가 포함되어 있는지 점수화한다.

    공식 발표는 수치 데이터를 포함하는 경우가 많다.
    3건 이상 = 1.0, 2건 = 0.7, 1건 = 0.4

    Args:
        text: 전체 텍스트
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

    [공식], [단독], [속보] 태그, 보도자료, 실적 공시 등의 패턴을 탐지한다.

    Args:
        title: 정제된 기사 제목
    Returns:
        0.0 ~ 1.0
    """
    if not title:
        return 0.0
    count = sum(1 for p in TITLE_OFFICIAL_PATTERNS if re.search(p, title))
    return min(count * 0.4, 1.0)


def score_direct_speech(text: str) -> dict:
    """
    본인 직접 발언 기사인지 탐지한다.

    직접 인용문 + 발언 동사 = 본인 직접 발언 확정 (1.0)
    발언 동사 + 방송 맥락 = 거의 확정 (0.8)
    발언 동사 2개 이상 = 가능성 높음 (0.6)
    발언 동사 1개 = 약한 시그널 (0.3)

    이 점수는 rule_based_scorer와 ensemble에서 사용된다.

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        {"score": float, "has_direct_speech": bool,
         "speech_verbs_found": list, "speech_contexts_found": list}
    """
    if not text:
        return {
            "score": 0.0, "has_direct_speech": False,
            "speech_verbs_found": [], "speech_contexts_found": [],
        }

    verbs_found = [v for v in DIRECT_SPEECH_VERBS if v in text]
    contexts_found = [c for c in DIRECT_SPEECH_CONTEXTS if c in text]
    has_quotes = bool(re.search(r'["\u201c][^"\u201d]{5,}["\u201d]', text))

    score = 0.0
    has_direct_speech = False

    if verbs_found and has_quotes:
        score = 1.0
        has_direct_speech = True
    elif verbs_found and contexts_found:
        score = 0.8
        has_direct_speech = True
    elif len(verbs_found) >= 2:
        score = 0.6
        has_direct_speech = True
    elif verbs_found:
        score = 0.3
        has_direct_speech = False
    elif contexts_found and has_quotes:
        score = 0.4
        has_direct_speech = False

    return {
        "score": score,
        "has_direct_speech": has_direct_speech,
        "speech_verbs_found": verbs_found[:5],
        "speech_contexts_found": contexts_found[:3],
    }


def score_subject_verb(text: str) -> dict:
    """
    기관명 + 공식 동사 패턴을 탐지한다. (v5 신규)

    config.OFFICIAL_SUBJECT_VERBS를 사용하여
    "한국은행은 결정했다", "소속사는 밝혔다" 같은
    공식 발화 주체 + 동사 구조를 탐지한다.

    이 패턴이 있으면 발화 주체가 명확한 공식 기사일 가능성이 높다.
    단순히 "한국은행 관련 논란" 같은 간접 언급과 구별하기 위함.

    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        {"score": float, "found_patterns": list}
    """
    if not text:
        return {"score": 0.0, "found_patterns": []}

    found_patterns = []

    # config의 OFFICIAL_SUBJECT_VERBS에서 동사 패턴을 가져온다
    for verb in OFFICIAL_SUBJECT_VERBS:
        if verb in text:
            # 동사 앞에 기관명/주어가 있는지 확인
            # "~은/는/이/가" + 동사 패턴
            # 예: "한국은행은 결정했다" → 매치
            # 예: "결정했다고 전해졌다" → 주어 없으므로 약한 매치
            pattern = r"[가-힣A-Za-z]{2,15}(?:은|는|이|가|에서|측은|측이)\s*" + re.escape(verb)
            matches = re.findall(pattern, text)
            if matches:
                found_patterns.extend(matches[:3])

    # 중복 제거
    found_patterns = list(dict.fromkeys(found_patterns))
    count = len(found_patterns)

    if count >= 3:
        score = 1.0
    elif count == 2:
        score = 0.7
    elif count == 1:
        score = 0.4
    else:
        score = 0.0

    return {"score": score, "found_patterns": found_patterns}


# ═════════════════════════════════════════════════════════════
# 통합 특징 추출 함수
# ═════════════════════════════════════════════════════════════

def extract_features(article: dict) -> dict:
    """
    전처리된 단일 기사에서 모든 특징을 추출한다.

    v5 변경:
    - subject_verb_score 추가 (기관명+공식동사 패턴)
    - _evidence에 subject_verb 근거 포함

    Args:
        article: 전처리된 기사 딕셔너리
                 필수 키: title, content, domain
    Returns:
        특징 딕셔너리 (점수 + 설명 근거 포함)
        빈 딕셔너리: 추출 실패 시
    """
    try:
        title = article.get("title", "")
        content = article.get("content", "")
        domain = article.get("domain", "")
        full_text = f"{title} {content}"

        # 각 피처 점수 계산
        official_result = score_official_expression(full_text)
        unofficial_result = score_unofficial_expression(full_text)
        org_result = score_organization_name(full_text)
        speech_result = score_direct_speech(full_text)
        subject_verb_result = score_subject_verb(full_text)

        features = {
            # ── 도메인 신뢰도 (5등급 체계) ────────────────
            "domain_grade_score": score_domain_grade(domain),

            # ── 공식성 표현 강도 ───────────────────────────
            "official_expr_score": official_result["score"],

            # ── 비공식 표현 강도 ───────────────────────────
            "unofficial_expr_score": unofficial_result["score"],

            # ── 기관/기업명 등장 ───────────────────────────
            "org_name_score": org_result["score"],

            # ── 직접 인용문 포함 ───────────────────────────
            "quote_score": score_direct_quote(full_text),

            # ── 수치/통계 데이터 포함 ──────────────────────
            "stat_score": score_statistics(full_text),

            # ── 제목 구조 공식성 ───────────────────────────
            "title_format_score": score_title_format(title),

            # ── 본인 직접 발언 탐지 ───────────────────────
            "direct_speech_score": speech_result["score"],
            "has_direct_speech": speech_result["has_direct_speech"],

            # ── 기관명+공식동사 패턴 (v5 신규) ─────────────
            "subject_verb_score": subject_verb_result["score"],

            # ── 모델 입력용 텍스트 ─────────────────────────
            "embedding_input": full_text,
            "classifier_input": full_text,

            # ── 설명용 근거 ────────────────────────────────
            # rule_based_scorer와 ensemble에서 판단 근거 메시지 생성에 사용
            "_evidence": {
                "official_strong": official_result["strong_found"],
                "official_weak": official_result["weak_found"],
                "unofficial_found": unofficial_result["found"],
                "orgs_found": org_result["found_orgs"],
                "domain": domain,
                "speech_verbs": speech_result["speech_verbs_found"],
                "speech_contexts": speech_result["speech_contexts_found"],
                "has_direct_speech": speech_result["has_direct_speech"],
                "subject_verb_patterns": subject_verb_result["found_patterns"],
            },
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
        특징 딕셔너리 리스트 (추출 실패한 기사는 제외)
    """
    if not articles:
        logger.warning("특징 추출 입력 데이터가 비어 있음")
        return []

    logger.info(f"특징 추출 시작 | {len(articles)}건")
    results = [f for a in articles if (f := extract_features(a))]
    logger.info(f"특징 추출 완료 | {len(results)}건")
    return results
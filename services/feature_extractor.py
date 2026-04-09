"""
services/feature_extractor.py
──────────────────────────────
전처리된 기사에서 모델 입력에 필요한 특징(feature)을 추출한다.

[v4 변경 — 본인 직접 발언 인식]
  아이유 라디오 출연 인터뷰 기사처럼 "본인이 직접 말한 것"을
  공식 표현으로 잡지 못하는 문제 수정.

  추가:
  1. 본인 직접 발언 동사: "~고 말했다", "~고 설명했다" 등
  2. 방송/인터뷰 출연 표현: "출연해", "라디오에서" 등
  3. score_direct_speech() 신규 함수 — 직접 발화 탐지
  4. has_direct_speech 플래그 — ensemble에서 신뢰성 보정용
"""

import re
from logger import get_logger

logger = get_logger("feature_extractor")

# ── 도메인 신뢰도 등급 테이블 ──────────────────────────────
DOMAIN_GRADE = {
    "grade_1": [".go.kr", ".or.kr", ".re.kr", ".ac.kr"],
    "grade_2": ["yna.co.kr", "yonhapnewstv.co.kr"],
    "grade_3": [
        "chosun.com", "joongang.co.kr", "donga.com",
        "hani.co.kr", "khan.co.kr", "kmib.co.kr",
        "seoul.co.kr", "munhwa.com",
    ],
    "grade_4": [
        "hankyung.com", "mk.co.kr", "edaily.co.kr",
        "mt.co.kr", "sedaily.com", "etnews.com",
        "zdnet.co.kr", "bloter.net", "financialpost.co.kr",
    ],
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
GRADE_SCORES = {"grade_1": 1.0, "grade_2": 0.7, "grade_3": 0.5, "grade_4": 0.35, "grade_5": 0.2}

# ── 공식성 표현 (v4 확장) ──────────────────────────────────
OFFICIAL_EXPR_STRONG = [
    # 정부/기관
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
    # 소속사/엔터
    "소속사 측은", "소속사는", "소속사가",
    "소속사 측이", "소속사 관계자는",
    "공식 입장을 밝혔다", "공식 입장을 전했다",
    "공식적으로 발표", "공식적으로 확인",
    "구단은", "구단 측은", "구단이 발표",
    "협회는", "협회가 발표", "연맹은",
    # [태그]
    "[공식]", "[단독]", "[속보]", "[긴급]",
    # 본인 직접 발언 (v4 핵심)
    "고 말했다", "고 밝혔다", "고 전했다",
    "고 설명했다", "고 답했다", "고 답변했다",
    "고 당부했다", "고 강조했다", "고 덧붙였다",
    "고 언급했다", "고 회상했다", "고 고백했다",
    "고 소감을 밝혔다", "고 소감을 전했다",
    "라고 말했다", "라고 밝혔다", "라고 전했다",
    "라고 설명했다", "라고 답했다",
    "이라고 말했다", "이라고 밝혔다",
    "직접 언급", "직접 해명", "직접 설명",
    # 방송/인터뷰 출연 (v4)
    "출연해", "출연하여", "출연한 자리에서",
    "인터뷰에서", "인터뷰를 통해",
    "라디오에서", "방송에서", "프로그램에서",
    "기자회견에서", "간담회에서",
    "취재진에게", "기자들과 만나",
]

OFFICIAL_EXPR_WEAK = [
    "발표", "공고", "공지", "시행", "공식", "공시", "실적",
    "출시", "론칭", "오픈", "확정", "선정", "인가", "승인", "허가",
    "계약", "체결", "서명", "조인",
    "컴백", "데뷔", "컴백 확정", "데뷔 확정",
    "이적", "계약 해지", "전속계약",
    "출연 확정", "캐스팅 확정", "합류",
    "출연", "게스트", "MC", "홍보했다",
]

# ── 비공식 표현 (v4 수정) ──────────────────────────────────
UNOFFICIAL_EXPRESSIONS = [
    "관계자에 따르면", "관계자는", "내부 관계자",
    "복수의 관계자", "업계 관계자", "측근에 따르면",
    "소식통에 따르면", "익명의", "이름을 밝히지 않은",
    "것으로 알려졌다", "것으로 전해졌다", "것으로 보인다",
    "것으로 파악됐다",
    "알려졌다", "추정된다", "추측", "관측된다", "전망이다",
    "가능성이 높다", "가능성이 제기",
    "루머", "의혹", "카더라", "찌라시",
    "열애설", "스캔들", "불화설", "갈등설",
    "누리꾼", "네티즌", "온라인에서 화제",
    "커뮤니티에서", "SNS에서 확산",
    "업계에서는", "업계에선",
    "논란이", "논란을", "논란에 휩싸여",
    "폭로", "주장했다", "주장이 나왔다",
    "제기됐다", "비판이",
]

# ── 수치/통계 패턴 ─────────────────────────────────────────
STAT_PATTERNS = [
    r"\d+(\.\d+)?%", r"\d+억\s*원", r"\d+조\s*원", r"\d+만\s*원",
    r"\d+달러", r"\d+건", r"\d+명",
    r"전년\s*(대비|比)", r"전분기\s*(대비|比)",
    r"역대\s*(최대|최고|최저)", r"\d+위",
]

# ── 제목 공식 구조 패턴 ────────────────────────────────────
TITLE_OFFICIAL_PATTERNS = [
    r"^\[공식\]", r"^\[단독\]", r"^\[속보\]", r"^[\[（\(]",
    r"보도자료", r"공식\s*(발표|입장|확인)",
    r"업무협약|MOU|파트너십", r"실적\s*(발표|공시)",
    r"컴백\s*확정", r"데뷔\s*확정", r"출연\s*확정",
    r"계약\s*체결", r"이적\s*확정",
    r"인터뷰", r"직접\s*(해명|언급|설명)",
]

# ── 직접 발화 탐지 리스트 (v4 핵심) ────────────────────────
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


def score_domain_grade(domain: str) -> float:
    """도메인의 신뢰도 등급을 점수로 반환한다."""
    if not domain:
        return 0.0
    for grade, domains in DOMAIN_GRADE.items():
        for d in domains:
            if d in domain:
                return GRADE_SCORES[grade]
    return 0.0


def score_official_expression(text: str) -> dict:
    """공식성 표현의 강도를 점수화하고, 탐지된 표현 목록을 반환한다."""
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
    """비공식 표현을 탐지하고 점수화한다."""
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
    """기관명/기업명/소속사명/방송사명 등장을 점수화한다."""
    if not text:
        return {"score": 0.0, "found_orgs": []}
    gov_pattern = r"[가-힣]{2,8}(부|처|청|원|위원회|공단|공사|은행|청장|장관)"
    gov_matches = re.findall(gov_pattern, text)
    major_entities = [
        "현대자동차", "현대차", "기아", "삼성전자", "삼성",
        "LG전자", "LG", "SK하이닉스", "SK", "포스코",
        "한국은행", "연합뉴스", "국회", "대법원", "대통령실",
        "SM엔터테인먼트", "SM엔터", "JYP엔터테인먼트", "JYP엔터", "JYP",
        "YG엔터테인먼트", "YG엔터", "YG",
        "하이브", "HYBE", "빅히트뮤직", "빅히트",
        "플레디스", "스타쉽", "큐브엔터", "FNC", "IST엔터", "안테나",
        "카카오엔터테인먼트", "카카오엔터", "CJ ENM",
        "EDAM엔터테인먼트", "EDAM엔터",
        "키이스트", "매니지먼트숲", "BH엔터", "나무액터스",
        "울림엔터테인먼트", "RBW", "WM엔터",
        "MBC", "KBS", "SBS", "JTBC", "tvN", "MBN", "YTN",
        "KBO", "KFA", "KBL", "KOVO", "대한체육회",
        "네이버", "카카오", "쿠팡", "배달의민족", "토스", "당근마켓",
        "넷플릭스", "디즈니플러스", "티빙", "웨이브",
    ]
    found_entities = [c for c in major_entities if c in text]
    found_orgs = found_entities.copy()
    if gov_matches:
        for suffix in gov_matches:
            full_matches = re.findall(r"[가-힣]{2,8}" + suffix, text)
            found_orgs.extend(full_matches)
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
    """직접 인용문 포함 여부를 점수화한다."""
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
    """수치/통계 포함 여부를 점수화한다."""
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
    """제목 구조의 공식성을 점수화한다."""
    if not title:
        return 0.0
    count = sum(1 for p in TITLE_OFFICIAL_PATTERNS if re.search(p, title))
    return min(count * 0.4, 1.0)


def score_direct_speech(text: str) -> dict:
    """
    본인 직접 발언 기사인지 탐지한다. (v4 핵심 신규)

    직접 인용문 + 발언 동사 → 본인 직접 발언 확정
    방송 맥락 + 발언 동사 → 거의 확정
    발언 동사 2개 이상 → 가능성 높음

    Returns:
        {"score": float, "has_direct_speech": bool,
         "speech_verbs_found": list, "speech_contexts_found": list}
    """
    if not text:
        return {"score": 0.0, "has_direct_speech": False,
                "speech_verbs_found": [], "speech_contexts_found": []}

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
        "score": score, "has_direct_speech": has_direct_speech,
        "speech_verbs_found": verbs_found[:5],
        "speech_contexts_found": contexts_found[:3],
    }


def extract_features(article: dict) -> dict:
    """전처리된 단일 기사에서 모든 특징을 추출한다."""
    try:
        title = article.get("title", "")
        content = article.get("content", "")
        domain = article.get("domain", "")
        full_text = f"{title} {content}"

        official_result = score_official_expression(full_text)
        unofficial_result = score_unofficial_expression(full_text)
        org_result = score_organization_name(full_text)
        speech_result = score_direct_speech(full_text)

        features = {
            "domain_grade_score": score_domain_grade(domain),
            "official_expr_score": official_result["score"],
            "unofficial_expr_score": unofficial_result["score"],
            "org_name_score": org_result["score"],
            "quote_score": score_direct_quote(full_text),
            "stat_score": score_statistics(full_text),
            "title_format_score": score_title_format(title),
            "direct_speech_score": speech_result["score"],
            "has_direct_speech": speech_result["has_direct_speech"],
            "embedding_input": full_text,
            "classifier_input": full_text,
            "_evidence": {
                "official_strong": official_result["strong_found"],
                "official_weak": official_result["weak_found"],
                "unofficial_found": unofficial_result["found"],
                "orgs_found": org_result["found_orgs"],
                "domain": domain,
                "speech_verbs": speech_result["speech_verbs_found"],
                "speech_contexts": speech_result["speech_contexts_found"],
                "has_direct_speech": speech_result["has_direct_speech"],
            },
        }
        return features
    except Exception as e:
        logger.error(f"특징 추출 실패: {e} | title={article.get('title', '')}")
        return {}


def extract_features_batch(articles: list) -> list:
    """기사 목록 전체에서 특징을 추출한다."""
    if not articles:
        logger.warning("특징 추출 입력 데이터가 비어 있음")
        return []
    logger.info(f"특징 추출 시작 | {len(articles)}건")
    results = [f for a in articles if (f := extract_features(a))]
    logger.info(f"특징 추출 완료 | {len(results)}건")
    return results
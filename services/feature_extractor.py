"""
services/feature_extractor.py
──────────────────────────────
전처리된 기사에서 모델 입력에 필요한 특징(feature)을 추출한다.

[v7 변경사항 - 자동 매핑 강화]
1. score_organization_name() 5층 탐지 구조로 확장
   - v6의 3층(한글접미사+영문약어+config) → v7의 5층
   - [신규] 1층: 엄격 공기 검사 (조사+공식동사 co-occurrence)
     · "삼성전자가 실적 발표" → 삼성전자 자동 추출
     · "네이버는 공개했다" → 네이버 자동 추출
   - [신규] 2층: 기업 접미사 확장 (전자/화학/건설/뱅크/엔터 등)
     · LG화학/현대건설/카카오뱅크 자동 탐지
   - 3~5층: 기존 v6 유지 (한글기관접미사/영문약어/config)

2. 하드코딩 화이트리스트 완전 없음
   - major_entities 리스트 사용 안 함
   - 최소한의 블랙리스트(일반명사 제외용)만 존재

3. 중복 제거 로직 추가
   - "SK하이닉스" 탐지 시 "SK" 단독 제외 (긴 것 우선)
"""

import re
from logger import get_logger

from config import (
    ANONYMOUS_EXPRESSIONS,
    OFFICIAL_SUBJECT_VERBS,
    OFFICIAL_EXPRESSIONS,
)

try:
    from config import ORGANIZATION_PATTERNS
except ImportError:
    ORGANIZATION_PATTERNS = []

logger = get_logger("feature_extractor")


# ─────────────────────────────────────────────────────────────
# 도메인 신뢰도 등급 테이블
# ─────────────────────────────────────────────────────────────
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

GRADE_SCORES = {
    "grade_1": 1.0, "grade_2": 0.7, "grade_3": 0.5,
    "grade_4": 0.35, "grade_5": 0.2,
}

# ─────────────────────────────────────────────────────────────
# 공식성 표현 (v6 유지)
# ─────────────────────────────────────────────────────────────
OFFICIAL_EXPR_STRONG = [
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
    "소속사 측은", "소속사는", "소속사가",
    "소속사 측이", "소속사 관계자는",
    "공식 입장을 밝혔다", "공식 입장을 전했다",
    "공식적으로 발표", "공식적으로 확인",
    "구단은", "구단 측은", "구단이 발표",
    "협회는", "협회가 발표", "연맹은",
    "[공식]", "[단독]", "[속보]", "[긴급]",
    "고 말했다", "고 밝혔다", "고 전했다",
    "고 설명했다", "고 답했다", "고 답변했다",
    "고 당부했다", "고 강조했다", "고 덧붙였다",
    "고 언급했다", "고 회상했다", "고 고백했다",
    "고 소감을 밝혔다", "고 소감을 전했다",
    "라고 말했다", "라고 밝혔다", "라고 전했다",
    "라고 설명했다", "라고 답했다",
    "이라고 말했다", "이라고 밝혔다",
    "직접 언급", "직접 해명", "직접 설명",
    "출연해", "출연하여", "출연한 자리에서",
    "인터뷰에서", "인터뷰를 통해",
    "라디오에서", "방송에서", "프로그램에서",
    "기자회견에서", "간담회에서",
    "취재진에게", "기자들과 만나",
    "판결했다", "선고했다", "기소했다",
    "대법원 판결", "헌법재판소 결정",
    "허가 승인", "긴급사용승인", "접종 개시",
    "임상시험 결과",
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
    "판결", "선고", "기소", "구속",
    "논문 발표", "연구 결과", "학위 수여",
    "배당 결정", "주주총회", "이사회 결의",
    "훈련 실시", "작전 발표",
]

UNOFFICIAL_EXPRESSIONS = ANONYMOUS_EXPRESSIONS

STAT_PATTERNS = [
    r"\d+(\.\d+)?%", r"\d+억\s*원", r"\d+조\s*원", r"\d+만\s*원",
    r"\d+달러", r"\d+건", r"\d+명",
    r"전년\s*(대비|比)", r"전분기\s*(대비|比)",
    r"역대\s*(최대|최고|최저)", r"\d+위",
]

TITLE_OFFICIAL_PATTERNS = [
    r"^\[공식\]", r"^\[단독\]", r"^\[속보\]", r"^[\[（\(]",
    r"보도자료", r"공식\s*(발표|입장|확인)",
    r"업무협약|MOU|파트너십", r"실적\s*(발표|공시)",
    r"컴백\s*확정", r"데뷔\s*확정", r"출연\s*확정",
    r"계약\s*체결", r"이적\s*확정",
    r"인터뷰", r"직접\s*(해명|언급|설명)",
    r"판결", r"선고", r"기소",
    r"허가\s*승인", r"임상\s*결과",
]

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
# v7 신규: 기관명 탐지용 정규식 5층
# ═════════════════════════════════════════════════════════════

# ─── 1층: 엄격 공기 검사용 ─────────────────────────────────
# 공기 검사용 핵심 공식 동사 (어간만 - 활용형 자동 커버)
# "발표" → 발표/발표했다/발표한/발표됐다/발표하며 전부 매칭
_CO_OCCURRENCE_VERBS = [
    "발표", "밝혔", "공지", "공고", "결정", "의결", "시행",
    "출시", "공개", "선언", "공시", "확정", "전했", "말했",
    "설명했", "답했", "강조했", "당부했", "고시", "인정",
    "해명", "언급", "체결", "서명", "승인", "허가",
]

# 조사 + 공기 검사 주어 추출 패턴
# [2~14자 한글/영문] + (은|는|이|가|에서|측은|측이|에 따르면)
_SUBJECT_EXTRACT_PATTERN = re.compile(
    r"([가-힣A-Za-z][가-힣A-Za-z0-9]{1,13})"
    r"(?:은|는|이|가|에서|측은|측이|에 따르면)"
)

# 주어 추출 시 제외할 일반 명사 (기관이 아닌 흔한 단어)
# 시간/대명사/일반명사/비공식 표현 겹치는 단어 위주
_SUBJECT_BLACKLIST = {
    # 시간
    "오늘", "어제", "내일", "모레", "이번", "지난", "최근",
    "현재", "과거", "미래", "작년", "올해", "금년", "향후",
    # 대명사
    "그", "그녀", "그들", "이것", "저것", "나", "우리", "이들",
    "본인", "당사자", "자신",
    # 일반명사
    "기자", "사람", "시민", "누리꾼", "일부", "상당수", "다수",
    "모두", "전부", "일각", "업계", "관계자", "전문가",
    "측근", "지인", "주변", "친구", "가족",
    # 기타
    "어느", "모든", "각각",
}


# ─── 2층: 기업 접미사 확장 ─────────────────────────────────
# 한국 대기업/중견기업의 상호 접미사
# 삼성전자, LG화학, 현대건설, SK하이닉스, 카카오뱅크 등 자동 커버
_COMPANY_SUFFIX_PATTERN = re.compile(
    r"[가-힣A-Za-z]{1,8}"
    r"(?:전자|전기|중공업|건설|제약|바이오|생명|화학|"
    r"에너지|그룹|홀딩스|증권|카드|캐피탈|자동차|"
    r"텔레콤|반도체|디스플레이|모빌리티|뱅크|"
    r"파이낸셜|자산운용|헬스케어|코퍼레이션|"
    r"인터내셔널|물산|상사|제일제당|하이닉스|"
    r"케미칼|머티리얼즈|테크놀로지|시스템즈|"
    r"네트웍스|컴퍼니|푸드|푸즈|엔터|뮤직)"
)


# ─── 3층: 한글 기관 접미사 (v6 유지) ─────────────────────
_KOREAN_ORG_SUFFIX_PATTERN = re.compile(
    r"[가-힣]{2,10}"
    r"(?:부|처|청|원|위원회|공단|공사|은행|"
    r"협회|연맹|협의회|엔터테인먼트|"
    r"대학교|대학|병원|구단|재단|연구소|연구원|"
    r"학회|조합|노조|노동조합)"
)


# ─── 4층: 영문 대문자 약어 (v6 유지) ─────────────────────
_ENGLISH_ABBR_PATTERN = re.compile(r"\b[A-Z]{2,5}\b")

_ENGLISH_ABBR_BLACKLIST = {
    "AI", "IT", "CEO", "CFO", "COO", "CTO", "CMO",
    "GDP", "GNP", "CPI", "PPI", "PR", "HR", "VIP",
    "DIY", "DIV", "ETC", "FAQ", "FYI",
    "THE", "AND", "FOR", "NOR", "BUT", "YET",
    "NEW", "OLD", "NOT", "ALL", "ANY", "OUT", "OFF",
    "YOU", "HIM", "HER", "HIS", "OUR", "WHY",
    "HOW", "NOW", "CAN", "MAY", "WAY", "DAY", "SEE",
    "GOT", "GET", "PUT", "LET", "TOO", "TWO", "ONE",
    "PDF", "JPG", "PNG", "GIF", "CSV", "TXT", "XML",
    "API", "URL", "URI", "SDK", "IDE", "OS", "UI", "UX",
    "HTML", "CSS", "JSON", "YAML", "HTTP", "HTTPS",
    "KM", "CM", "MM", "KG", "GB", "MB", "TB", "KW",
    "USA", "UK", "EU", "TV", "PC", "SMS", "MMS",
    "ATM", "PIN", "SNS", "NFT", "IPO", "MOU",
}


# ═════════════════════════════════════════════════════════════
# 기관명 탐지 헬퍼 함수 (5층)
# ═════════════════════════════════════════════════════════════

def _detect_subject_with_co_occurrence(text: str) -> list:
    """
    [1층] 엄격 공기 검사 기반 주어 추출.

    조사(은/는/이/가/에서/측은/측이) 앞의 명사를 후보로 잡고,
    뒤 30자 이내에 공식 동사가 있어야 최종 인정.

    "삼성전자가 실적 발표" → 삼성전자 ✅
    "네이버는 서비스를 공개했다" → 네이버 ✅
    "오늘은 맑다" → ❌ (블랙리스트)
    "기자는 물었다" → ❌ (블랙리스트)
    """
    found = []
    for match in _SUBJECT_EXTRACT_PATTERN.finditer(text):
        subject = match.group(1)

        # 블랙리스트 필터
        if subject in _SUBJECT_BLACKLIST:
            continue

        # 길이 2자 미만은 제외
        if len(subject) < 2:
            continue

        # 공기 검사: 매치 끝 위치 이후 30자 이내에 공식 동사 있는지
        end_pos = match.end()
        window = text[end_pos: end_pos + 30]

        if any(verb in window for verb in _CO_OCCURRENCE_VERBS):
            found.append(subject)

    return list(dict.fromkeys(found))


def _detect_company_suffix(text: str) -> list:
    """[2층] 기업 접미사 패턴 탐지."""
    full_matches = [m.group(0) for m in _COMPANY_SUFFIX_PATTERN.finditer(text)]
    return list(dict.fromkeys(full_matches))


def _detect_korean_orgs(text: str) -> list:
    """[3층] 한글 기관 접미사 패턴 탐지."""
    full_matches = [m.group(0) for m in _KOREAN_ORG_SUFFIX_PATTERN.finditer(text)]
    return list(dict.fromkeys(full_matches))


def _detect_english_abbr(text: str) -> list:
    """[4층] 영문 대문자 약어 탐지. 블랙리스트 제외."""
    candidates = _ENGLISH_ABBR_PATTERN.findall(text)
    filtered = [c for c in candidates if c not in _ENGLISH_ABBR_BLACKLIST]
    return list(dict.fromkeys(filtered))


def _detect_config_patterns(text: str) -> list:
    """[5층] config.ORGANIZATION_PATTERNS 커스텀 정규식."""
    if not ORGANIZATION_PATTERNS:
        return []
    found = []
    for pattern in ORGANIZATION_PATTERNS:
        try:
            matches = re.findall(pattern, text)
            for m in matches:
                if isinstance(m, tuple):
                    m = next((x for x in m if x), "")
                if m and isinstance(m, str):
                    found.append(m)
        except re.error as e:
            logger.warning(f"ORGANIZATION_PATTERNS 오류 스킵: {pattern} | {e}")
            continue
    return list(dict.fromkeys(found))


def _dedupe_with_longest_priority(items: list) -> list:
    """
    중복 제거 + 긴 것 우선.

    "SK하이닉스"가 있으면 "SK" 단독은 제외.
    "삼성전자"가 있으면 "삼성" 단독은 제외.
    """
    if not items:
        return []

    # 중복 제거 + 길이 내림차순
    unique_items = list(dict.fromkeys(items))
    sorted_items = sorted(unique_items, key=len, reverse=True)
    result = []

    for item in sorted_items:
        # 이미 result에 있는 더 긴 항목에 포함되면 스킵
        if any(item != longer and item in longer for longer in result):
            continue
        result.append(item)

    # 원래 등장 순서대로 재정렬
    original_order = {}
    for idx, item in enumerate(items):
        if item not in original_order:
            original_order[item] = idx
    result.sort(key=lambda x: original_order.get(x, 999))

    return result


# ═════════════════════════════════════════════════════════════
# 개별 피처 점수 계산 함수
# ═════════════════════════════════════════════════════════════

def score_domain_grade(domain: str) -> float:
    """도메인 신뢰도 5등급 점수."""
    if not domain:
        return 0.0
    for grade, domains in DOMAIN_GRADE.items():
        for d in domains:
            if d in domain:
                return GRADE_SCORES[grade]
    return 0.0


def score_official_expression(text: str) -> dict:
    """공식성 표현 강도 점수화."""
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
    """비공식 표현 점수화."""
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
    [v7] 기관명/기업명 5층 자동 매핑 탐지.

    탐지 5층:
      1. 엄격 공기 검사 (조사+공식동사 co-occurrence)
         → "삼성전자가 실적 발표", "네이버는 공개했다"
      2. 기업 접미사 확장 (전자/화학/뱅크/엔터 등)
         → "LG화학", "카카오뱅크", "현대건설"
      3. 한글 기관 접미사 (부/처/청/위원회/협회 등)
         → "금융위원회", "대한축구협회"
      4. 영문 대문자 약어 (2~5자, 블랙리스트 제외)
         → "KBS", "HYBE", "JYP"
      5. config.ORGANIZATION_PATTERNS (커스텀)

    중복 제거: 긴 것 우선 (SK하이닉스 있으면 SK 단독 제외)

    점수: 3건↑=1.0, 2건=0.7, 1건=0.4, 0건=0.0
    """
    if not text:
        return {"score": 0.0, "found_orgs": []}

    # 5층 탐지 실행
    layer_1 = _detect_subject_with_co_occurrence(text)
    layer_2 = _detect_company_suffix(text)
    layer_3 = _detect_korean_orgs(text)
    layer_4 = _detect_english_abbr(text)
    layer_5 = _detect_config_patterns(text)

    # 통합 + 중복 제거 (긴 것 우선)
    all_found = layer_1 + layer_2 + layer_3 + layer_4 + layer_5
    found_orgs = _dedupe_with_longest_priority(all_found)

    total = len(found_orgs)

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
    """직접 인용문 포함 점수화."""
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
    """수치/통계 포함 점수화."""
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
    """제목 공식 구조 점수화."""
    if not title:
        return 0.0
    count = sum(1 for p in TITLE_OFFICIAL_PATTERNS if re.search(p, title))
    return min(count * 0.4, 1.0)


def score_direct_speech(text: str) -> dict:
    """본인 직접 발언 탐지."""
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
    elif contexts_found and has_quotes:
        score = 0.4
    return {
        "score": score,
        "has_direct_speech": has_direct_speech,
        "speech_verbs_found": verbs_found[:5],
        "speech_contexts_found": contexts_found[:3],
    }


def score_subject_verb(text: str) -> dict:
    """기관명 + 공식 동사 패턴 탐지 (config.OFFICIAL_SUBJECT_VERBS)."""
    if not text:
        return {"score": 0.0, "found_patterns": []}
    found_patterns = []
    for verb in OFFICIAL_SUBJECT_VERBS:
        if verb in text:
            pattern = r"[가-힣A-Za-z]{2,15}(?:은|는|이|가|에서|측은|측이)\s*" + re.escape(verb)
            matches = re.findall(pattern, text)
            if matches:
                found_patterns.extend(matches[:3])
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
# 통합 특징 추출
# ═════════════════════════════════════════════════════════════

def extract_features(article: dict) -> dict:
    """단일 기사에서 모든 특징 추출."""
    try:
        title = article.get("title", "")
        content = article.get("content", "")
        domain = article.get("domain", "")
        full_text = f"{title} {content}"

        official_result = score_official_expression(full_text)
        unofficial_result = score_unofficial_expression(full_text)
        org_result = score_organization_name(full_text)
        speech_result = score_direct_speech(full_text)
        subject_verb_result = score_subject_verb(full_text)

        return {
            "domain_grade_score": score_domain_grade(domain),
            "official_expr_score": official_result["score"],
            "unofficial_expr_score": unofficial_result["score"],
            "org_name_score": org_result["score"],
            "quote_score": score_direct_quote(full_text),
            "stat_score": score_statistics(full_text),
            "title_format_score": score_title_format(title),
            "direct_speech_score": speech_result["score"],
            "has_direct_speech": speech_result["has_direct_speech"],
            "subject_verb_score": subject_verb_result["score"],
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
                "subject_verb_patterns": subject_verb_result["found_patterns"],
            },
        }
    except Exception as e:
        logger.error(f"특징 추출 실패: {e} | title={article.get('title', '')}")
        return {}


def extract_features_batch(articles: list) -> list:
    """기사 목록 전체 특징 추출."""
    if not articles:
        logger.warning("특징 추출 입력 데이터가 비어 있음")
        return []
    logger.info(f"특징 추출 시작 | {len(articles)}건")
    results = [f for a in articles if (f := extract_features(a))]
    logger.info(f"특징 추출 완료 | {len(results)}건")
    return results
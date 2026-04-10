"""
services/rule_based_scorer.py
──────────────────────────────
특징 벡터를 기반으로 규칙 기반 공식성 점수를 계산한다.

[v5 변경사항]
1. subject_verb_score 가산점 추가 (기관명+공식동사 패턴 탐지 시)
   - "한국은행은 결정했다" → 발화 주체가 명확 → 공식성 가산
   - feature_extractor v5에서 추출한 피처

2. direct_speech_score 반영 (v4에서 유지)
   - 본인 직접 발언 탐지 시 가산

3. 판단 근거 메시지에 subject_verb_patterns, direct_speech 포함

4. 가중치 재배분
   - 기존: domain(0.25) + official(0.30) + org(0.20) + quote(0.10) + stat(0.05) + title(0.10) = 1.0
   - 변경: domain(0.20) + official(0.25) + org(0.15) + quote(0.08) + stat(0.05) + title(0.07)
           + direct_speech(0.10) + subject_verb(0.10) = 1.0
   - 이유: 직접 발화와 기관+동사 패턴이 공식성의 가장 직접적인 증거
"""

from logger import get_logger

logger = get_logger("rule_based_scorer")

# ─────────────────────────────────────────────────────────────
# 피처별 가중치 (v5 재배분)
#
# 원칙: "누가 직접 말했는가" > "어떤 표현인가" > "어디서 나왔는가"
#
# subject_verb_score + direct_speech_score를 합쳐서 0.20
# → "기관이 직접 발표했다" + "본인이 직접 말했다"가
#   공식성 판단의 핵심이라는 설계 의도 반영
# ─────────────────────────────────────────────────────────────

FEATURE_WEIGHTS = {
    # 출처 자체의 공신력 (정부도메인 > 연합 > 종합일간지)
    "domain_grade_score": 0.20,

    # 공식 발표 표현 강도 (가장 직접적인 공식성 키워드)
    "official_expr_score": 0.25,

    # 기관/기업명 등장 여부
    "org_name_score": 0.15,

    # 직접 인용문(" ") 포함 여부
    "quote_score": 0.08,

    # 수치/통계 데이터 포함 여부
    "stat_score": 0.05,

    # 제목 구조 공식성 ([기관명] 형식, [공식] 태그 등)
    "title_format_score": 0.07,

    # 본인 직접 발언 탐지 (v4에서 추가)
    # "고 말했다", "인터뷰에서" 등 본인이 직접 말한 기사
    "direct_speech_score": 0.10,

    # 기관명+공식동사 패턴 (v5 신규)
    # "한국은행은 결정했다", "소속사는 밝혔다" 등
    # 발화 주체가 명확한 공식 기사의 핵심 증거
    "subject_verb_score": 0.10,
}

# 비공식 표현 감점 가중치
# 비공식 표현이 1.0이면 최종 점수에서 0.20 감점
# config.ANONYMOUS_EXPRESSIONS에 정의된 표현이 탐지될 때 적용
UNOFFICIAL_PENALTY_WEIGHT = 0.20

# 가중치 합 검증 (양수 가중치만 1.0이어야 함)
_total_weight = sum(FEATURE_WEIGHTS.values())
assert abs(_total_weight - 1.0) < 1e-9, (
    f"FEATURE_WEIGHTS 합이 1.0이 아님: {_total_weight:.4f}. "
    "가중치를 수정하세요."
)

# 도메인 등급 이름 매핑 (설명 메시지용)
_GRADE_NAMES = {
    1.0: "1등급(정부/공공)",
    0.7: "2등급(국가통신사)",
    0.5: "3등급(종합일간지)",
    0.35: "4등급(경제전문지)",
    0.2: "5등급(방송/인터넷)",
    0.0: "미등록",
}


def _get_grade_name(domain_score: float) -> str:
    """도메인 점수에 해당하는 등급 이름을 반환한다."""
    for score, name in _GRADE_NAMES.items():
        if abs(domain_score - score) < 0.01:
            return name
    return "미등록"


def compute_rule_score(features: dict) -> dict:
    """
    특징 딕셔너리에서 규칙 기반 공식성 점수와 판단 근거를 계산한다.

    계산 방식:
    1. 양수 피처 8개를 가중 합산 (합계 1.0)
    2. 비공식 표현 감점 적용 (최대 -0.20)
    3. 0.0 ~ 1.0 범위로 클램핑

    Args:
        features: feature_extractor.py v5에서 추출된 특징 딕셔너리
    Returns:
        {"rule_score": float, "rule_reason": str}
    """
    if not features:
        return {"rule_score": 0.0, "rule_reason": "특징 데이터 없음"}

    try:
        # ── 양수 피처 가중 합산 ───────────────────────────────
        score = 0.0
        for feature_name, weight in FEATURE_WEIGHTS.items():
            value = float(features.get(feature_name, 0.0))
            # 0~1 범위로 클램핑 (비정상 값 방어)
            value = max(0.0, min(value, 1.0))
            score += value * weight

        # ── 비공식 표현 감점 ──────────────────────────────────
        # unofficial_expr_score가 높을수록 공식성 점수를 감점
        unofficial_score = float(features.get("unofficial_expr_score", 0.0))
        unofficial_score = max(0.0, min(unofficial_score, 1.0))
        penalty = unofficial_score * UNOFFICIAL_PENALTY_WEIGHT
        score = max(0.0, score - penalty)

        rule_score = round(score, 4)

        # ── 판단 근거 메시지 생성 ─────────────────────────────
        evidence = features.get("_evidence", {})
        reason_parts = []

        # 공식 표현 탐지 내역
        strong = evidence.get("official_strong", [])
        weak = evidence.get("official_weak", [])
        if strong:
            display = strong[:3]
            reason_parts.append(f"공식 표현 {len(strong)}건: {', '.join(display)}")
        elif weak:
            display = weak[:3]
            reason_parts.append(f"약한 공식 표현 {len(weak)}건: {', '.join(display)}")
        else:
            reason_parts.append("공식 표현 미탐지")

        # 기관명 탐지 내역
        orgs = evidence.get("orgs_found", [])
        if orgs:
            display = orgs[:3]
            reason_parts.append(f"기관명 {len(orgs)}건: {', '.join(display)}")
        else:
            reason_parts.append("기관명 미탐지")

        # 비공식 표현 내역
        unofficial = evidence.get("unofficial_found", [])
        if unofficial:
            display = unofficial[:3]
            reason_parts.append(f"비공식 표현 {len(unofficial)}건: {', '.join(display)}")
        else:
            reason_parts.append("비공식 표현 없음")

        # 기관명+공식동사 패턴 (v5 신규)
        subject_verb = evidence.get("subject_verb_patterns", [])
        if subject_verb:
            display = subject_verb[:2]
            reason_parts.append(f"공식발화 패턴 {len(subject_verb)}건: {', '.join(display)}")

        # 직접 발화 탐지 (v4에서 유지)
        has_speech = evidence.get("has_direct_speech", False)
        speech_verbs = evidence.get("speech_verbs", [])
        if has_speech and speech_verbs:
            display = speech_verbs[:2]
            reason_parts.append(f"직접 발화 확인: {', '.join(display)}")

        # 도메인 등급
        domain_score = float(features.get("domain_grade_score", 0.0))
        grade_name = _get_grade_name(domain_score)
        domain_str = evidence.get("domain", "")
        if domain_str:
            reason_parts.append(f"도메인: {domain_str} ({grade_name})")
        else:
            reason_parts.append(f"도메인: {grade_name}")

        rule_reason = " | ".join(reason_parts)

        return {"rule_score": rule_score, "rule_reason": rule_reason}

    except Exception as e:
        logger.error(f"규칙 기반 점수 계산 실패: {e}")
        return {"rule_score": 0.0, "rule_reason": f"계산 오류: {e}"}


def compute_rule_scores_batch(features_list: list) -> list:
    """
    기사 특징 목록 전체에 대해 규칙 기반 점수를 계산한다.

    Args:
        features_list: 특징 딕셔너리 리스트
    Returns:
        각 기사의 {"rule_score": float, "rule_reason": str} 리스트
    """
    if not features_list:
        logger.warning("규칙 기반 스코어러 입력이 비어 있음")
        return []

    logger.info(f"규칙 기반 점수 계산 시작 | {len(features_list)}건")

    results = [compute_rule_score(f) for f in features_list]

    scores = [r["rule_score"] for r in results]
    if scores:
        avg = sum(scores) / len(scores)
        min_s = min(scores)
        max_s = max(scores)
        logger.info(
            f"규칙 기반 점수 계산 완료 | "
            f"평균={avg:.4f} 최솟값={min_s:.4f} 최댓값={max_s:.4f}"
        )
    else:
        logger.warning("규칙 기반 점수 계산 결과가 비어 있음")

    return results
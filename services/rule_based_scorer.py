"""
services/rule_based_scorer.py
──────────────────────────────
특징 벡터를 기반으로 규칙 기반 공식성 점수를 계산한다.

[v4 변경]
  1. direct_speech_score 가산점 추가 (본인 직접 발언 탐지 시)
  2. 판단 근거 메시지에 직접 발화 내역 포함
  3. unofficial 감점 시 직접 발화가 확인되면 감점 완화
     (본인이 직접 말했는데 "~고 전했다"가 비공식으로 잡히는 문제 방지)
"""

from logger import get_logger

logger = get_logger("rule_based_scorer")

# ── 피처별 가중치 (합계 1.0) ───────────────────────────────
FEATURE_WEIGHTS = {
    "domain_grade_score": 0.20,
    "official_expr_score": 0.25,
    "org_name_score": 0.15,
    "quote_score": 0.10,
    "stat_score": 0.05,
    "title_format_score": 0.10,
    # v4 신규: 직접 발화 점수
    "direct_speech_score": 0.15,
}

# 비공식 표현 감점 가중치
UNOFFICIAL_PENALTY_WEIGHT = 0.20

# 가중치 합 검증
_total_weight = sum(FEATURE_WEIGHTS.values())
assert abs(_total_weight - 1.0) < 1e-9, (
    f"FEATURE_WEIGHTS 합이 1.0이 아님: {_total_weight:.4f}"
)

# 도메인 등급 이름 매핑
_GRADE_NAMES = {
    1.0: "1등급(정부/공공)", 0.7: "2등급(국가통신사)",
    0.5: "3등급(종합일간지)", 0.35: "4등급(경제전문지)",
    0.2: "5등급(방송/인터넷)", 0.0: "미등록",
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

    v4 변경:
    - direct_speech_score 반영
    - 직접 발화 확인 시 비공식 표현 감점을 절반으로 완화
    - 판단 근거에 직접 발화 내역 포함

    Returns:
        {"rule_score": float, "rule_reason": str}
    """
    if not features:
        return {"rule_score": 0.0, "rule_reason": "특징 데이터 없음"}

    try:
        # ── 양수 피처 가중 합산 ───────────────────────
        score = 0.0
        for feature_name, weight in FEATURE_WEIGHTS.items():
            value = float(features.get(feature_name, 0.0))
            value = max(0.0, min(value, 1.0))
            score += value * weight

        # ── 비공식 표현 감점 ──────────────────────────
        unofficial_score = float(features.get("unofficial_expr_score", 0.0))
        unofficial_score = max(0.0, min(unofficial_score, 1.0))
        has_direct_speech = features.get("has_direct_speech", False)

        # v4: 직접 발화가 확인되면 비공식 감점을 절반으로 완화
        # 이유: "~고 전했다"가 비공식으로 잡히지만 본인이 직접 말한 거면 공식임
        if has_direct_speech:
            penalty = unofficial_score * UNOFFICIAL_PENALTY_WEIGHT * 0.5
        else:
            penalty = unofficial_score * UNOFFICIAL_PENALTY_WEIGHT

        score = max(0.0, score - penalty)
        rule_score = round(score, 4)

        # ── 판단 근거 메시지 생성 ─────────────────────
        evidence = features.get("_evidence", {})
        reason_parts = []

        # 직접 발화 (v4 신규 — 가장 먼저 표시)
        if has_direct_speech:
            speech_verbs = evidence.get("speech_verbs", [])
            speech_ctx = evidence.get("speech_contexts", [])
            parts_str = ""
            if speech_ctx:
                parts_str += ", ".join(speech_ctx[:2])
            if speech_verbs:
                if parts_str:
                    parts_str += " + "
                parts_str += ", ".join(speech_verbs[:2])
            reason_parts.append(f"🎙️ 본인 직접 발언 확인: {parts_str}")

        # 공식 표현
        strong = evidence.get("official_strong", [])
        weak = evidence.get("official_weak", [])
        if strong:
            reason_parts.append(f"공식 표현 {len(strong)}건: {', '.join(strong[:3])}")
        elif weak:
            reason_parts.append(f"약한 공식 표현 {len(weak)}건: {', '.join(weak[:3])}")
        else:
            reason_parts.append("공식 표현 미탐지")

        # 기관명
        orgs = evidence.get("orgs_found", [])
        if orgs:
            reason_parts.append(f"기관명 {len(orgs)}건: {', '.join(orgs[:3])}")
        else:
            reason_parts.append("기관명 미탐지")

        # 비공식 표현
        unofficial = evidence.get("unofficial_found", [])
        if unofficial:
            if has_direct_speech:
                reason_parts.append(
                    f"비공식 표현 {len(unofficial)}건 (직접 발화 확인 → 감점 완화): "
                    f"{', '.join(unofficial[:2])}"
                )
            else:
                reason_parts.append(f"비공식 표현 {len(unofficial)}건: {', '.join(unofficial[:3])}")
        else:
            reason_parts.append("비공식 표현 없음")

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
    """기사 특징 목록 전체에 대해 규칙 기반 점수를 계산한다."""
    if not features_list:
        logger.warning("규칙 기반 스코어러 입력이 비어 있음")
        return []
    logger.info(f"규칙 기반 점수 계산 시작 | {len(features_list)}건")
    results = [compute_rule_score(f) for f in features_list]
    scores = [r["rule_score"] for r in results]
    avg = sum(scores) / len(scores)
    logger.info(
        f"규칙 기반 점수 계산 완료 | "
        f"평균={avg:.4f} 최솟값={min(scores):.4f} 최댓값={max(scores):.4f}"
    )
    return results
"""
services/ensemble.py
────────────────────────────────────────────────────────────────────────────────
공식성 점수 + 신뢰성 점수를 통합하여 최종 판정을 내리는 앙상블 모듈

[v7 — conditional weighting + 폴백 경로 제거]

  v6까지의 문제:
    classifier < 0.5이면 완전 배제 (폴백 경로)
    → 55건 중 51건이 폴백 → 사실상 3중 앙상블
    → "4중 앙상블"이라고 말할 수 없는 구조

  v7 해결:
    폴백 경로를 완전 제거하고, 항상 4개 모델이 참여한다.
    classifier 신뢰도에 따라 가중치를 3구간으로 동적 조절한다.

    구간 1 (LOW):    classifier < 0.15
      → rule 주도, classifier 최소 참여 (0.10)
    구간 2 (NORMAL): 0.15 <= classifier <= 0.85
      → 4개 균등 참여
    구간 3 (HIGH):   classifier > 0.85
      → classifier 과신 방지, agency 견제 강화

    추가:
    - classifier 입력값을 0.05~0.95로 클리핑 (극단값 방지)
    - score_method를 "low_confidence" / "normal" / "high_confidence"로 표시
    - v6의 직접 발화 신뢰성 보정 유지
"""

from logger import get_logger
from config import ENSEMBLE_WEIGHTS, OFFICIAL_THRESHOLD, EnsembleConfig
from services.cross_validator import determine_final_verdict

logger = get_logger(__name__)

# ── 점수 클리핑 범위 ──────────────────────────────────────────
# rule, semantic 등 일반 점수의 클리핑 범위
SCORE_CLIP_MIN = 0.02
SCORE_CLIP_MAX = 0.98

# 신뢰성 점수의 출력 범위 제한
RELIABILITY_CLAMP_MIN = 0.10
RELIABILITY_CLAMP_MAX = 0.95

# v6 유지: 직접 발화 확인 시 신뢰성 최소 보장값
DIRECT_SPEECH_RELIABILITY_FLOOR = 0.50

# v7 신규: 신뢰성 → 공식성 보정
# 교차 보도가 많은 기사는 공식 발표에 기반했을 가능성이 높으므로
# 신뢰성이 이 값 이상이면 공식성에 가산점을 준다
# 예: 44개 독립 언론사가 같은 사실을 보도 → 추측이 아니라 공식 발표 기반
RELIABILITY_TO_OFFICIAL_THRESHOLD = 0.6
# 보정 최대값: 신뢰성 1.0일 때 공식성에 최대 이만큼 가산
# (1.0 - 0.6) * 0.625 = 0.25 최대 가산
RELIABILITY_TO_OFFICIAL_MAX_BOOST = 0.625


def _clip_score(value: float) -> float:
    """일반 점수를 SCORE_CLIP_MIN~SCORE_CLIP_MAX 범위로 클리핑한다."""
    return max(SCORE_CLIP_MIN, min(float(value), SCORE_CLIP_MAX))


def _clip_classifier(value: float) -> float:
    """
    classifier 점수를 별도 범위로 클리핑한다.

    v7 핵심:
      classifier가 0.001이면 0.05로 올리고, 0.999이면 0.95로 깎는다.
      이렇게 해야 극단값이 전체 앙상블을 왜곡하지 않는다.

    클리핑 범위: config.EnsembleConfig.CLASSIFIER_CLIP_MIN ~ CLASSIFIER_CLIP_MAX
    """
    return max(
        EnsembleConfig.CLASSIFIER_CLIP_MIN,
        min(float(value), EnsembleConfig.CLASSIFIER_CLIP_MAX)
    )


def _clamp_reliability(value: float) -> float:
    """신뢰도를 RELIABILITY_CLAMP_MIN~RELIABILITY_CLAMP_MAX 범위로 제한한다."""
    return max(RELIABILITY_CLAMP_MIN, min(float(value), RELIABILITY_CLAMP_MAX))


def _select_weights(classifier_score: float) -> tuple:
    """
    classifier 점수에 따라 가중치 구간을 선택한다.

    v7 핵심 로직:
      폴백 경로 없이, classifier 신뢰도에 따라 3구간의 가중치를 반환한다.
      어떤 구간이든 4개 모델이 모두 참여한다.

    Args:
        classifier_score: 클리핑 완료된 classifier 점수

    Returns:
        (rule_w, semantic_w, classifier_w, agency_w, method_name)
    """
    if classifier_score < EnsembleConfig.CLASSIFIER_LOW_BOUNDARY:
        # 구간 1: classifier 거의 확신 없음 → rule/semantic 주도
        return (
            EnsembleConfig.LOW_RULE_WEIGHT,
            EnsembleConfig.LOW_SEMANTIC_WEIGHT,
            EnsembleConfig.LOW_CLASSIFIER_WEIGHT,
            EnsembleConfig.LOW_AGENCY_WEIGHT,
            "low_confidence",
        )
    elif classifier_score > EnsembleConfig.CLASSIFIER_HIGH_BOUNDARY:
        # 구간 3: classifier 과도하게 확신 → agency 견제 강화
        return (
            EnsembleConfig.HIGH_RULE_WEIGHT,
            EnsembleConfig.HIGH_SEMANTIC_WEIGHT,
            EnsembleConfig.HIGH_CLASSIFIER_WEIGHT,
            EnsembleConfig.HIGH_AGENCY_WEIGHT,
            "high_confidence",
        )
    else:
        # 구간 2: 정상 범위 → 4개 균등 참여
        return (
            EnsembleConfig.NORMAL_RULE_WEIGHT,
            EnsembleConfig.NORMAL_SEMANTIC_WEIGHT,
            EnsembleConfig.NORMAL_CLASSIFIER_WEIGHT,
            EnsembleConfig.NORMAL_AGENCY_WEIGHT,
            "normal",
        )


def _build_explanation(
    rule_score, rule_reason, semantic_score, classifier_score,
    agency_score, official_score, score_method,
    reliability_score, reliability_reason,
    verification_message, verdict, direct_speech_boosted,
    reliability_boosted=False, reliability_boost_amount=0.0,
):
    """사용자에게 보여줄 종합 설명 메시지를 생성한다."""
    parts = []

    # 공식성 판정
    if official_score >= EnsembleConfig.OFFICIAL_SCORE_THRESHOLD:
        parts.append(f"✅ 공식성 높음 ({official_score:.2f})")
    else:
        parts.append(f"❌ 공식성 낮음 ({official_score:.2f})")

    # 규칙 기반 근거
    if rule_reason:
        parts.append(f"[규칙] {rule_reason}")

    # 기관 검증 메시지
    if verification_message:
        parts.append(f"[기관] {verification_message}")

    # 교차 보도 근거
    if reliability_reason:
        parts.append(f"[교차보도] {reliability_reason}")

    # v6 유지: 직접 발화 보정 표시
    if direct_speech_boosted:
        parts.append(
            f"[신뢰성 보정] 본인 직접 발언 확인 → "
            f"신뢰성 {DIRECT_SPEECH_RELIABILITY_FLOOR:.0%} 이상 보장"
        )

    # v7 신규: 신뢰성 → 공식성 보정 표시
    if reliability_boosted:
        parts.append(
            f"[교차보도 보정] 다수 독립 언론 교차 보도 확인 → "
            f"공식성 +{reliability_boost_amount:.2f} 가산"
        )

    # v7: classifier 구간별 표시
    if score_method == "low_confidence":
        parts.append(
            f"[모델] 분류기 저신뢰({classifier_score:.2f}) → "
            f"규칙+유사도 주도 판정"
        )
    elif score_method == "high_confidence":
        parts.append(
            f"[모델] 분류기 고신뢰({classifier_score:.2f}) → "
            f"과신 방지 가중치 적용"
        )
    else:
        if classifier_score >= 0.7:
            parts.append(f"[모델] 분류기 공식 판정({classifier_score:.2f})")
        elif classifier_score <= 0.3:
            parts.append(f"[모델] 분류기 비공식 판정({classifier_score:.2f})")
        else:
            parts.append(f"[모델] 분류기 중립({classifier_score:.2f})")

    # 의미 유사도 판정
    if semantic_score >= 0.5:
        parts.append(f"[유사도] 공식 문체 유사({semantic_score:.2f})")
    elif semantic_score <= 0.2:
        parts.append(f"[유사도] 공식 문체 비유사({semantic_score:.2f})")

    return " | ".join(parts)


def ensemble_single(
    article: dict,
    rule_result: dict,
    semantic_score: float,
    classifier_score: float,
    agency_score: float = 0.0,
    reliability_score: float = 0.0,
    reliability_reason: str = "",
    verification_message: str = "",
    has_direct_speech: bool = False,
) -> dict:
    """
    단일 기사에 대해 공식성 점수와 설명 메시지를 포함한 최종 판정을 반환한다.

    v7 변경:
    - 폴백 경로 제거, conditional weighting 3구간 적용
    - classifier를 별도 범위(0.05~0.95)로 클리핑
    - 항상 4개 모델이 참여하여 가중 합산
    - v6의 직접 발화 신뢰성 보정 유지
    """
    rule_score = float(rule_result.get("rule_score", 0.0))
    rule_reason = rule_result.get("rule_reason", "")

    # ── 점수 클리핑 ───────────────────────────────────────────
    # rule, semantic: 일반 범위(0.02~0.98)
    # classifier: 별도 범위(0.05~0.95) — 극단값 방지 강화
    rule_sc = _clip_score(rule_score)
    semantic_sc = _clip_score(semantic_score)
    classifier_sc = _clip_classifier(classifier_score)
    agency_sc = max(0.0, min(float(agency_score), 1.0))

    # ── conditional weighting으로 가중치 선택 ──────────────────
    rule_w, semantic_w, classifier_w, agency_w, score_method = (
        _select_weights(classifier_sc)
    )

    # ── 공식성 점수 계산 (항상 4개 모델 참여) ──────────────────
    official_score = (
        rule_sc * rule_w
        + semantic_sc * semantic_w
        + classifier_sc * classifier_w
        + agency_sc * agency_w
    )

    official_score = round(max(0.0, min(float(official_score), 1.0)), 4)

    # ── 신뢰성 → 공식성 보정 (v7 신규) ───────────────────────
    # 교차 보도가 많으면 공식 발표에 기반했을 가능성이 높다.
    # 44개 독립 언론사가 같은 사실을 보도했는데 공식성 0.08이면 말이 안 됨.
    # 그건 추측이 아니라 보도자료 배포에 의한 것이므로 공식성을 보정한다.
    #
    # 보정 공식: boost = (reliability - 0.6) * 0.625
    #   reliability 0.6 → boost 0.00 (보정 없음)
    #   reliability 0.7 → boost 0.0625
    #   reliability 0.8 → boost 0.125
    #   reliability 0.9 → boost 0.1875
    #   reliability 1.0 → boost 0.25 (최대)
    #
    # 안전장치: 원래 공식성이 극단적으로 낮은 기사(0.08)는
    # 보정해도 0.33 정도라 임계값(0.45)을 넘지 못함 → 오탐 방지
    reliability_boosted = False
    reliability_boost_amount = 0.0
    if reliability_score >= RELIABILITY_TO_OFFICIAL_THRESHOLD:
        reliability_boost_amount = round(
            (reliability_score - RELIABILITY_TO_OFFICIAL_THRESHOLD)
            * RELIABILITY_TO_OFFICIAL_MAX_BOOST,
            4,
        )
        official_score = round(
            min(official_score + reliability_boost_amount, 1.0), 4
        )
        reliability_boosted = True
        logger.debug(
            f"신뢰성→공식성 보정 | reliability={reliability_score:.2f} → "
            f"boost=+{reliability_boost_amount:.4f} → "
            f"official={official_score:.4f} | "
            f"title={article.get('title', '')[:30]}"
        )

    # ── 신뢰성 보정 (v6 유지) ─────────────────────────────────
    # 직접 발화가 확인되면 교차 보도 없어도 신뢰성을 최소 보장
    direct_speech_boosted = False
    if has_direct_speech and reliability_score < DIRECT_SPEECH_RELIABILITY_FLOOR:
        reliability_score = DIRECT_SPEECH_RELIABILITY_FLOOR
        direct_speech_boosted = True
        logger.debug(
            f"직접 발화 확인 → 신뢰성 보정 {DIRECT_SPEECH_RELIABILITY_FLOOR} | "
            f"title={article.get('title', '')[:30]}"
        )

    reliability_score = _clamp_reliability(reliability_score)

    # ── 최종 판정 ─────────────────────────────────────────────
    verdict_result = determine_final_verdict(
        official_score=official_score,
        reliability_score=reliability_score
    )

    # ── 설명 메시지 ───────────────────────────────────────────
    explanation = _build_explanation(
        rule_score=rule_score, rule_reason=rule_reason,
        semantic_score=semantic_score, classifier_score=classifier_score,
        agency_score=agency_score, official_score=official_score,
        score_method=score_method,
        reliability_score=reliability_score, reliability_reason=reliability_reason,
        verification_message=verification_message,
        verdict=verdict_result["verdict"],
        direct_speech_boosted=direct_speech_boosted,
        reliability_boosted=reliability_boosted,
        reliability_boost_amount=reliability_boost_amount,
    )

    result = {
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "originallink": article.get("originallink", ""),
        "pubDate": article.get("pubDate", ""),
        "domain": article.get("domain", ""),
        "rule_score": round(float(rule_score), 4),
        "rule_reason": rule_reason,
        "semantic_score": round(float(semantic_score), 4),
        "classifier_score": round(float(classifier_score), 4),
        "agency_score": round(float(agency_score), 4),
        "official_score": official_score,
        "score_method": score_method,
        "reliability_score": round(float(reliability_score), 4),
        "reliability_reason": reliability_reason,
        "cluster_id": article.get("cluster_id", -1),
        "cluster_size": article.get("cluster_size", 1),
        "unique_sources": article.get("unique_sources", 1),
        "has_official_domain": article.get("has_official_domain", False),
        "verdict": verdict_result["verdict"],
        "verdict_emoji": verdict_result["verdict_emoji"],
        "verdict_reason": verdict_result["verdict_reason"],
        "is_verified": verdict_result["is_verified"],
        "explanation": explanation,
        "verification_message": verification_message,
        "has_direct_speech": has_direct_speech,
        "direct_speech_boosted": direct_speech_boosted,
        "reliability_boosted": reliability_boosted,
        "reliability_boost_amount": reliability_boost_amount,
        "final_official_score": official_score,
        "predicted_label": verdict_result["verdict"],
    }

    logger.debug(
        f"앙상블 완료 | title={article.get('title', '')[:20]} | "
        f"official={official_score:.4f} ({score_method}) | "
        f"weights=[r={rule_w} s={semantic_w} c={classifier_w} a={agency_w}] | "
        f"reliability={reliability_score:.4f} | "
        f"direct_speech={'Y' if has_direct_speech else 'N'} | "
        f"verdict={verdict_result['verdict']}"
    )

    return result


def ensemble_batch(
    articles: list[dict],
    rule_scores: list,
    semantic_scores: list[float],
    classifier_results: list[dict],
    agency_results: list[dict] = None,
    features_list: list[dict] = None,
) -> list[dict]:
    """
    기사 목록 전체에 앙상블을 적용하여 최종 판정 결과 목록을 반환한다.

    v7 변경:
    - score_method가 "low_confidence" / "normal" / "high_confidence"로 변경
    - 로그에 구간별 분포 표시
    - v6의 features_list, 직접 발화 기능 유지
    """
    if not articles:
        logger.warning("앙상블 입력 기사 목록이 비어있음")
        return []

    n = len(articles)
    logger.info(f"[앙상블] 시작 | {n}건")

    rule_scores = _pad_list(rule_scores, n, {"rule_score": 0.0, "rule_reason": ""})
    semantic_scores = _pad_list(semantic_scores, n, 0.0)
    classifier_results = _pad_list(classifier_results, n, {"score": 0.0})
    agency_results = _pad_list(agency_results or [], n, {"agency_score": 0.0})
    features_list = _pad_list(features_list or [], n, {})

    results = []
    # v7: 구간별 카운터
    method_counts = {"low_confidence": 0, "normal": 0, "high_confidence": 0}
    direct_speech_count = 0
    reliability_boost_count = 0

    for i, article in enumerate(articles):
        try:
            # rule 처리 (dict 또는 float 호환)
            raw_rule = rule_scores[i]
            if isinstance(raw_rule, dict):
                rule_result = raw_rule
            else:
                rule_result = {
                    "rule_score": float(raw_rule) if raw_rule else 0.0,
                    "rule_reason": "",
                }

            semantic_score = (
                float(semantic_scores[i])
                if semantic_scores[i] is not None
                else 0.0
            )

            clf_result = classifier_results[i]
            if isinstance(clf_result, dict):
                classifier_score = float(
                    clf_result.get(
                        "classifier_score", clf_result.get("score", 0.0)
                    )
                )
            else:
                classifier_score = (
                    float(clf_result) if clf_result is not None else 0.0
                )

            agn_result = agency_results[i]
            if isinstance(agn_result, dict):
                agency_score = float(agn_result.get("agency_score", 0.0))
                verification_message = agn_result.get("verification_message", "")
            else:
                agency_score = (
                    float(agn_result) if agn_result is not None else 0.0
                )
                verification_message = ""

            reliability_score = float(article.get("reliability_score", 0.0))
            reliability_reason = article.get("reliability_reason", "")

            # v6 유지: features_list에서 직접 발화 정보 가져오기
            feat = features_list[i] if i < len(features_list) else {}
            has_direct_speech = feat.get("has_direct_speech", False)

            result = ensemble_single(
                article=article,
                rule_result=rule_result,
                semantic_score=semantic_score,
                classifier_score=classifier_score,
                agency_score=agency_score,
                reliability_score=reliability_score,
                reliability_reason=reliability_reason,
                verification_message=verification_message,
                has_direct_speech=has_direct_speech,
            )
            results.append(result)

            # 구간별 카운터 증가
            method = result.get("score_method", "normal")
            if method in method_counts:
                method_counts[method] += 1
            if has_direct_speech:
                direct_speech_count += 1
            if result.get("reliability_boosted", False):
                reliability_boost_count += 1

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(f"앙상블 처리 실패 | idx={i} | error={exc}")
            results.append(_default_result(article))

    # ── 요약 로그 ─────────────────────────────────────────────
    verified_count = sum(1 for r in results if r.get("is_verified", False))
    avg_official = sum(r.get("official_score", 0) for r in results) / max(n, 1)
    avg_reliability = sum(r.get("reliability_score", 0) for r in results) / max(n, 1)

    logger.info(
        f"[앙상블] 완료 | "
        f"오피셜={verified_count}/{n}건 | "
        f"평균 공식성={avg_official:.4f} | "
        f"평균 신뢰성={avg_reliability:.4f} | "
        f"구간분포=[low={method_counts['low_confidence']} "
        f"normal={method_counts['normal']} "
        f"high={method_counts['high_confidence']}] | "
        f"직접 발화={direct_speech_count}건 | "
        f"교차보도 보정={reliability_boost_count}건"
    )

    return results


def _pad_list(lst, target_len, default):
    """리스트 길이가 target_len보다 짧으면 default로 채워서 반환한다."""
    if lst is None:
        return [default] * target_len
    result = list(lst)
    while len(result) < target_len:
        result.append(default)
    return result


def _default_result(article: dict) -> dict:
    """앙상블 처리 실패 시 반환하는 기본 결과 딕셔너리."""
    return {
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "originallink": article.get("originallink", ""),
        "pubDate": article.get("pubDate", ""),
        "domain": article.get("domain", ""),
        "rule_score": 0.0, "rule_reason": "처리 실패",
        "semantic_score": 0.0, "classifier_score": 0.0,
        "agency_score": 0.0, "official_score": 0.0,
        "score_method": "error",
        "reliability_score": 0.0, "reliability_reason": "처리 실패",
        "cluster_id": -1, "cluster_size": 1,
        "unique_sources": 1, "has_official_domain": False,
        "verdict": "검증 불가", "verdict_emoji": "❌",
        "verdict_reason": "앙상블 처리 중 오류 발생",
        "is_verified": False,
        "explanation": "❌ 판정 실패 — 앙상블 처리 중 오류 발생",
        "verification_message": "",
        "has_direct_speech": False, "direct_speech_boosted": False,
        "reliability_boosted": False, "reliability_boost_amount": 0.0,
        "final_official_score": 0.0, "predicted_label": "검증 불가",
    }
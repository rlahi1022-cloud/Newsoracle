"""
services/ensemble.py
────────────────────────────────────────────────────────────────────────────────
공식성 점수 + 신뢰성 점수를 통합하여 최종 판정을 내리는 앙상블 모듈

[v6 — 직접 발화 신뢰성 보정]
  v5의 설명 가능한 AI 기반 위에 추가:
  1. has_direct_speech가 True이면 reliability_score를 보정
     → 본인이 직접 말한 기사는 교차 보도 없어도 신뢰성이 높아야 함
     → 최소 0.50으로 끌어올림 (RELIABILITY_SCORE_THRESHOLD 충족)
  2. explanation에 직접 발화 보정 여부 표시
"""

from logger import get_logger
from config import ENSEMBLE_WEIGHTS, OFFICIAL_THRESHOLD, EnsembleConfig
from services.cross_validator import determine_final_verdict

logger = get_logger(__name__)

SCORE_CLIP_MIN = 0.02
SCORE_CLIP_MAX = 0.98
RELIABILITY_CLAMP_MIN = 0.10
RELIABILITY_CLAMP_MAX = 0.95

# v6 신규: 직접 발화 확인 시 신뢰성 최소 보장값
# 본인이 직접 말한 기사는 교차 보도 없어도 이 값 이상으로 보정
DIRECT_SPEECH_RELIABILITY_FLOOR = 0.50


def _clip_score(value: float) -> float:
    """점수를 SCORE_CLIP_MIN~SCORE_CLIP_MAX 범위로 클리핑한다."""
    return max(SCORE_CLIP_MIN, min(float(value), SCORE_CLIP_MAX))


def _clamp_reliability(value: float) -> float:
    """신뢰도를 RELIABILITY_CLAMP_MIN~RELIABILITY_CLAMP_MAX 범위로 제한한다."""
    return max(RELIABILITY_CLAMP_MIN, min(float(value), RELIABILITY_CLAMP_MAX))


def _build_explanation(
    rule_score, rule_reason, semantic_score, classifier_score,
    agency_score, official_score, score_method,
    reliability_score, reliability_reason,
    verification_message, verdict, direct_speech_boosted,
):
    """사용자에게 보여줄 종합 설명 메시지를 생성한다."""
    parts = []

    if official_score >= EnsembleConfig.OFFICIAL_SCORE_THRESHOLD:
        parts.append(f"✅ 공식성 높음 ({official_score:.2f})")
    else:
        parts.append(f"❌ 공식성 낮음 ({official_score:.2f})")

    if rule_reason:
        parts.append(f"[규칙] {rule_reason}")

    if verification_message:
        parts.append(f"[기관] {verification_message}")

    if reliability_reason:
        parts.append(f"[교차보도] {reliability_reason}")

    # v6: 직접 발화 보정 표시
    if direct_speech_boosted:
        parts.append(f"[신뢰성 보정] 본인 직접 발언 확인 → 신뢰성 {DIRECT_SPEECH_RELIABILITY_FLOOR:.0%} 이상 보장")

    if score_method == "classifier_excluded":
        parts.append(f"[모델] 분류기 저신뢰({classifier_score:.2f}) → 규칙+기관 기반 판정")
    else:
        if classifier_score >= 0.7:
            parts.append(f"[모델] 분류기 공식 판정({classifier_score:.2f})")
        elif classifier_score <= 0.3:
            parts.append(f"[모델] 분류기 비공식 판정({classifier_score:.2f})")
        else:
            parts.append(f"[모델] 분류기 불확실({classifier_score:.2f})")

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

    v6 변경:
    - has_direct_speech 파라미터 추가
    - 직접 발화 확인 시 reliability_score를 DIRECT_SPEECH_RELIABILITY_FLOOR 이상으로 보정
    """
    rule_score = float(rule_result.get("rule_score", 0.0))
    rule_reason = rule_result.get("rule_reason", "")

    # 점수 클리핑
    rule_sc = _clip_score(rule_score)
    semantic_sc = _clip_score(semantic_score)
    classifier_sc = _clip_score(classifier_score)

    # 공식성 점수 계산 (conditional weighting)
    if classifier_sc >= EnsembleConfig.CLASSIFIER_LOW_CONFIDENCE:
        official_score = (
            rule_sc * EnsembleConfig.RULE_WEIGHT
            + semantic_sc * EnsembleConfig.SEMANTIC_WEIGHT
            + classifier_sc * EnsembleConfig.CLASSIFIER_WEIGHT
        )
        score_method = "classifier_included"
    else:
        official_score = (
            rule_sc * EnsembleConfig.FALLBACK_RULE_WEIGHT
            + semantic_sc * EnsembleConfig.FALLBACK_SEMANTIC_WEIGHT
            + agency_score * EnsembleConfig.FALLBACK_AGENCY_WEIGHT
        )
        score_method = "classifier_excluded"

    # agency 보너스
    if score_method == "classifier_included" and agency_score > 0:
        agency_bonus = agency_score * EnsembleConfig.AGENCY_BONUS_MAX
        official_score = min(official_score + agency_bonus, 1.0)

    official_score = round(float(official_score), 4)

    # ── 신뢰성 보정 (v6 핵심) ────────────────────────────────
    # 직접 발화가 확인되면 교차 보도 없어도 신뢰성을 최소 보장
    # 이유: 아이유가 라디오에서 직접 말한 기사인데 교차 보도 1건이라고
    #       신뢰성 25%로 나오면 안 됨
    direct_speech_boosted = False
    if has_direct_speech and reliability_score < DIRECT_SPEECH_RELIABILITY_FLOOR:
        reliability_score = DIRECT_SPEECH_RELIABILITY_FLOOR
        direct_speech_boosted = True
        logger.debug(
            f"직접 발화 확인 → 신뢰성 보정 {DIRECT_SPEECH_RELIABILITY_FLOOR} | "
            f"title={article.get('title', '')[:30]}"
        )

    reliability_score = _clamp_reliability(reliability_score)

    # 최종 판정
    verdict_result = determine_final_verdict(
        official_score=official_score,
        reliability_score=reliability_score
    )

    # 설명 메시지
    explanation = _build_explanation(
        rule_score=rule_score, rule_reason=rule_reason,
        semantic_score=semantic_score, classifier_score=classifier_score,
        agency_score=agency_score, official_score=official_score,
        score_method=score_method,
        reliability_score=reliability_score, reliability_reason=reliability_reason,
        verification_message=verification_message,
        verdict=verdict_result["verdict"],
        direct_speech_boosted=direct_speech_boosted,
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
        "final_official_score": official_score,
        "predicted_label": verdict_result["verdict"],
    }

    logger.debug(
        f"앙상블 완료 | title={article.get('title', '')[:20]} | "
        f"official={official_score:.4f} ({score_method}) | "
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

    v6 변경:
    - features_list 파라미터 추가 (직접 발화 정보 전달용)
    - features_list가 없어도 동작 (하위 호환)
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
    classifier_excluded_count = 0
    direct_speech_count = 0

    for i, article in enumerate(articles):
        try:
            # rule 처리 (dict 또는 float 호환)
            raw_rule = rule_scores[i]
            if isinstance(raw_rule, dict):
                rule_result = raw_rule
            else:
                rule_result = {"rule_score": float(raw_rule) if raw_rule else 0.0, "rule_reason": ""}

            semantic_score = float(semantic_scores[i]) if semantic_scores[i] is not None else 0.0

            clf_result = classifier_results[i]
            if isinstance(clf_result, dict):
                classifier_score = float(clf_result.get("classifier_score", clf_result.get("score", 0.0)))
            else:
                classifier_score = float(clf_result) if clf_result is not None else 0.0

            agn_result = agency_results[i]
            if isinstance(agn_result, dict):
                agency_score = float(agn_result.get("agency_score", 0.0))
                verification_message = agn_result.get("verification_message", "")
            else:
                agency_score = float(agn_result) if agn_result is not None else 0.0
                verification_message = ""

            reliability_score = float(article.get("reliability_score", 0.0))
            reliability_reason = article.get("reliability_reason", "")

            # v6: features_list에서 직접 발화 정보 가져오기
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

            if result.get("score_method") == "classifier_excluded":
                classifier_excluded_count += 1
            if has_direct_speech:
                direct_speech_count += 1

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(f"앙상블 처리 실패 | idx={i} | error={exc}")
            results.append(_default_result(article))

    verified_count = sum(1 for r in results if r.get("is_verified", False))
    avg_official = sum(r.get("official_score", 0) for r in results) / max(n, 1)
    avg_reliability = sum(r.get("reliability_score", 0) for r in results) / max(n, 1)

    logger.info(
        f"[앙상블] 완료 | "
        f"오피셜={verified_count}/{n}건 | "
        f"평균 공식성={avg_official:.4f} | "
        f"평균 신뢰성={avg_reliability:.4f} | "
        f"classifier 제외={classifier_excluded_count}건 | "
        f"직접 발화={direct_speech_count}건"
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
        "final_official_score": 0.0, "predicted_label": "검증 불가",
    }
"""
services/ensemble.py
────────────────────────────────────────────────────────────────────────────────
공식성 + 신뢰성 통합 앙상블 모듈

[v8 — 하이브리드 신뢰성]
  v7 conditional weighting 로직(공식성 산출)은 한 글자도 바꾸지 않고 보존.
  신규 추가:
    - compute_external_reliability(cross_info)
    - compute_reliability(title, content, originallink, cross_info)
    - build_final_result() 에서 내부/외부/breakdown/weights 포함

[v7 — conditional weighting + 폴백 경로 제거]  (아래는 v7 원문 주석 그대로)
  v6까지의 문제: classifier < 0.5 시 완전 배제 → 사실상 3중 앙상블
  v7 해결: 항상 4개 모델 참여 + 3구간 동적 가중치 (LOW/NORMAL/HIGH)
  classifier 클리핑 0.05~0.95, 신뢰성→공식성 보정, 직접 발화 신뢰성 보정 유지
"""

from logger import get_logger
from config import ENSEMBLE_WEIGHTS, OFFICIAL_THRESHOLD, EnsembleConfig
from services.cross_validator import determine_final_verdict
from services.reliability_scorer import compute_internal_reliability

logger = get_logger(__name__)

# ── 점수 클리핑 범위 (v7 유지) ────────────────────────────────
SCORE_CLIP_MIN = 0.02
SCORE_CLIP_MAX = 0.98
RELIABILITY_CLAMP_MIN = 0.10
RELIABILITY_CLAMP_MAX = 0.95
DIRECT_SPEECH_RELIABILITY_FLOOR = 0.50
RELIABILITY_TO_OFFICIAL_THRESHOLD = 0.6
RELIABILITY_TO_OFFICIAL_MAX_BOOST = 0.625


def _clip_score(value: float) -> float:
    """일반 점수 클리핑 (v7 유지)."""
    return max(SCORE_CLIP_MIN, min(float(value), SCORE_CLIP_MAX))


def _clip_classifier(value: float) -> float:
    """classifier 별도 클리핑 0.05~0.95 (v7 유지)."""
    return max(
        EnsembleConfig.CLASSIFIER_CLIP_MIN,
        min(float(value), EnsembleConfig.CLASSIFIER_CLIP_MAX),
    )


def _clamp_reliability(value: float) -> float:
    """신뢰도 범위 제한 (v7 유지)."""
    return max(RELIABILITY_CLAMP_MIN, min(float(value), RELIABILITY_CLAMP_MAX))


def _select_weights(classifier_score: float) -> tuple:
    """
    classifier 점수에 따라 가중치 구간 선택 (v7 원본 로직 완전 보존).

    구간 1: classifier < 0.15         → LOW (rule 주도)
    구간 2: 0.15 <= score <= 0.85     → NORMAL (균등)
    구간 3: classifier > 0.85         → HIGH (과신 방지)
    """
    if classifier_score < EnsembleConfig.CLASSIFIER_LOW_BOUNDARY:
        return (
            EnsembleConfig.LOW_RULE_WEIGHT,
            EnsembleConfig.LOW_SEMANTIC_WEIGHT,
            EnsembleConfig.LOW_CLASSIFIER_WEIGHT,
            EnsembleConfig.LOW_AGENCY_WEIGHT,
            "low_confidence",
        )
    elif classifier_score > EnsembleConfig.CLASSIFIER_HIGH_BOUNDARY:
        return (
            EnsembleConfig.HIGH_RULE_WEIGHT,
            EnsembleConfig.HIGH_SEMANTIC_WEIGHT,
            EnsembleConfig.HIGH_CLASSIFIER_WEIGHT,
            EnsembleConfig.HIGH_AGENCY_WEIGHT,
            "high_confidence",
        )
    else:
        return (
            EnsembleConfig.NORMAL_RULE_WEIGHT,
            EnsembleConfig.NORMAL_SEMANTIC_WEIGHT,
            EnsembleConfig.NORMAL_CLASSIFIER_WEIGHT,
            EnsembleConfig.NORMAL_AGENCY_WEIGHT,
            "normal",
        )


# ─────────────────────────────────────────────────────────────
# [v8 신규] 외부 신뢰성 계산
# ─────────────────────────────────────────────────────────────

def compute_external_reliability(cross_info: dict) -> dict:
    """
    외부 신뢰성 점수 계산 — 교차보도/독립출처/공식도메인/내용일치도 종합.

    Args:
        cross_info: {
            "cluster_size": int,
            "unique_sources": int,
            "has_official_domain": bool,
            "avg_similarity": float,
        }

    Returns:
        {"external_reliability": float, "details": {...}}
    """
    try:
        ci = cross_info or {}
        cluster_size = int(ci.get("cluster_size", 1) or 1)
        unique_sources = int(ci.get("unique_sources", 1) or 1)
        has_official = bool(ci.get("has_official_domain", False))
        avg_sim = float(ci.get("avg_similarity", 0.0) or 0.0)

        # 독립 출처 수 기반 점수 (출처 1개당 0.2, 최대 0.8)
        source_score = min(unique_sources * 0.2, 0.8)
        # 공식 도메인 포함 시 가산
        domain_bonus = 0.15 if has_official else 0.0
        # 내용 일치도 가산 (단독 보도 시 0)
        sim_bonus = min(max(avg_sim, 0.0), 1.0) * 0.15

        # 단독 보도(cluster_size <= 1)는 외부 검증 불가 → 0.3 캡
        if cluster_size <= 1:
            external = min(source_score + domain_bonus, 0.3)
        else:
            external = source_score + domain_bonus + sim_bonus

        external = round(max(0.0, min(external, 1.0)), 4)

        return {
            "external_reliability": external,
            "details": {
                "cluster_size": cluster_size,
                "unique_sources": unique_sources,
                "has_official_domain": has_official,
                "avg_similarity": round(avg_sim, 4),
            },
        }
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning(f"외부 신뢰성 계산 실패 | error={exc}")
        return {
            "external_reliability": 0.3,
            "details": {
                "cluster_size": 1, "unique_sources": 1,
                "has_official_domain": False, "avg_similarity": 0.0,
            },
        }


# ─────────────────────────────────────────────────────────────
# [v8 신규] 하이브리드 신뢰성 (내부 × 0.5 + 외부 × 0.5)
# ─────────────────────────────────────────────────────────────

def compute_reliability(
    title: str, content: str, originallink: str, cross_info: dict
) -> dict:
    """
    내부(텍스트 3축) + 외부(교차보도) 하이브리드 신뢰성.

    가중치: EnsembleConfig.INTERNAL_RELIABILITY_WEIGHT (기본 0.5)
           + EnsembleConfig.EXTERNAL_RELIABILITY_WEIGHT (기본 0.5)
    """
    try:
        internal_res = compute_internal_reliability(
            title=title, content=content, originallink=originallink
        )
        external_res = compute_external_reliability(cross_info or {})

        w_in = EnsembleConfig.INTERNAL_RELIABILITY_WEIGHT
        w_ex = EnsembleConfig.EXTERNAL_RELIABILITY_WEIGHT

        internal = internal_res["internal_reliability"]
        external = external_res["external_reliability"]
        final = round(internal * w_in + external * w_ex, 4)

        return {
            "reliability_score": final,
            "internal_reliability": internal,
            "external_reliability": external,
            "reliability_breakdown": internal_res["breakdown"],
            "internal_details": internal_res["details"],
            "external_details": external_res["details"],
            "weights": {"internal": w_in, "external": w_ex},
        }
    except Exception as exc:
        logger.error(f"하이브리드 신뢰성 계산 실패 | error={exc}", exc_info=True)
        return {
            "reliability_score": 0.5, "internal_reliability": 0.5,
            "external_reliability": 0.5,
            "reliability_breakdown": {
                "source_accountability": 0.5,
                "verifiability": 0.5, "neutrality": 0.5,
            },
            "internal_details": {}, "external_details": {},
            "weights": {"internal": 0.5, "external": 0.5},
        }


def _build_explanation(
    rule_score, rule_reason, semantic_score, classifier_score,
    agency_score, official_score, score_method,
    reliability_score, reliability_reason,
    verification_message, verdict, direct_speech_boosted,
    reliability_boosted=False, reliability_boost_amount=0.0,
):
    """v7 원본 설명 빌더 완전 보존."""
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
    if direct_speech_boosted:
        parts.append(
            f"[신뢰성 보정] 본인 직접 발언 확인 → "
            f"신뢰성 {DIRECT_SPEECH_RELIABILITY_FLOOR:.0%} 이상 보장"
        )
    if reliability_boosted:
        parts.append(
            f"[교차보도 보정] 다수 독립 언론 교차 보도 확인 → "
            f"공식성 +{reliability_boost_amount:.2f} 가산"
        )
    if score_method == "low_confidence":
        parts.append(f"[모델] 분류기 저신뢰({classifier_score:.2f}) → 규칙+유사도 주도 판정")
    elif score_method == "high_confidence":
        parts.append(f"[모델] 분류기 고신뢰({classifier_score:.2f}) → 과신 방지 가중치 적용")
    else:
        if classifier_score >= 0.7:
            parts.append(f"[모델] 분류기 공식 판정({classifier_score:.2f})")
        elif classifier_score <= 0.3:
            parts.append(f"[모델] 분류기 비공식 판정({classifier_score:.2f})")
        else:
            parts.append(f"[모델] 분류기 중립({classifier_score:.2f})")
    if semantic_score >= 0.5:
        parts.append(f"[유사도] 공식 문체 유사({semantic_score:.2f})")
    elif semantic_score <= 0.2:
        parts.append(f"[유사도] 공식 문체 비유사({semantic_score:.2f})")
    return " | ".join(parts)


def ensemble_single(
    article: dict, rule_result: dict, semantic_score: float,
    classifier_score: float, agency_score: float = 0.0,
    reliability_score: float = 0.0, reliability_reason: str = "",
    verification_message: str = "", has_direct_speech: bool = False,
) -> dict:
    """
    단일 기사 앙상블 (v7 로직 완전 보존 + v8 reliability 필드 추가).
    """
    rule_score = float(rule_result.get("rule_score", 0.0))
    rule_reason = rule_result.get("rule_reason", "")

    # ── v7 공식성 계산 (완전 보존) ───────────────────────────
    rule_sc = _clip_score(rule_score)
    semantic_sc = _clip_score(semantic_score)
    classifier_sc = _clip_classifier(classifier_score)
    agency_sc = max(0.0, min(float(agency_score), 1.0))

    rule_w, semantic_w, classifier_w, agency_w, score_method = (
        _select_weights(classifier_sc)
    )

    official_score = (
        rule_sc * rule_w + semantic_sc * semantic_w
        + classifier_sc * classifier_w + agency_sc * agency_w
    )
    official_score = round(max(0.0, min(float(official_score), 1.0)), 4)

    # ── v8 하이브리드 신뢰성 계산 ────────────────────────────
    cross_info = {
        "cluster_size": article.get("cluster_size", 1),
        "unique_sources": article.get("unique_sources", 1),
        "has_official_domain": article.get("has_official_domain", False),
        "avg_similarity": article.get("avg_similarity", 0.0),
    }
    hybrid = compute_reliability(
        title=article.get("title", ""),
        content=article.get("content", ""),
        originallink=article.get("originallink", ""),
        cross_info=cross_info,
    )
    # 외부 pipeline이 전달한 reliability_score가 있으면 max로 반영
    combined_reliability = max(
        float(reliability_score or 0.0), hybrid["reliability_score"]
    )

    # ── v7 신뢰성→공식성 보정 (완전 보존) ───────────────────
    reliability_boosted = False
    reliability_boost_amount = 0.0
    if combined_reliability >= RELIABILITY_TO_OFFICIAL_THRESHOLD:
        reliability_boost_amount = round(
            (combined_reliability - RELIABILITY_TO_OFFICIAL_THRESHOLD)
            * RELIABILITY_TO_OFFICIAL_MAX_BOOST, 4,
        )
        official_score = round(
            min(official_score + reliability_boost_amount, 1.0), 4
        )
        reliability_boosted = True

    # ── v7 직접 발화 보정 (완전 보존) ───────────────────────
    direct_speech_boosted = False
    if has_direct_speech and combined_reliability < DIRECT_SPEECH_RELIABILITY_FLOOR:
        combined_reliability = DIRECT_SPEECH_RELIABILITY_FLOOR
        direct_speech_boosted = True

    combined_reliability = _clamp_reliability(combined_reliability)

    verdict_result = determine_final_verdict(
        official_score=official_score,
        reliability_score=combined_reliability,
    )

    explanation = _build_explanation(
        rule_score=rule_score, rule_reason=rule_reason,
        semantic_score=semantic_score, classifier_score=classifier_score,
        agency_score=agency_score, official_score=official_score,
        score_method=score_method,
        reliability_score=combined_reliability,
        reliability_reason=reliability_reason,
        verification_message=verification_message,
        verdict=verdict_result["verdict"],
        direct_speech_boosted=direct_speech_boosted,
        reliability_boosted=reliability_boosted,
        reliability_boost_amount=reliability_boost_amount,
    )

    # ── build_final_result (v7 필드 + v8 신규 필드) ─────────
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
        "reliability_score": round(float(combined_reliability), 4),
        "reliability_reason": reliability_reason,
        # [v8 신규]
        "internal_reliability": hybrid["internal_reliability"],
        "external_reliability": hybrid["external_reliability"],
        "reliability_breakdown": hybrid["reliability_breakdown"],
        "internal_details": hybrid["internal_details"],
        "external_details": hybrid["external_details"],
        "weights": hybrid["weights"],
        "cluster_id": article.get("cluster_id", -1),
        "cluster_size": article.get("cluster_size", 1),
        "unique_sources": article.get("unique_sources", 1),
        "has_official_domain": article.get("has_official_domain", False),
        "avg_similarity": article.get("avg_similarity", 0.0),
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
        f"앙상블 완료 | title={article.get('title','')[:20]} | "
        f"official={official_score:.4f} ({score_method}) | "
        f"reliability={combined_reliability:.4f} "
        f"(in={hybrid['internal_reliability']:.2f} "
        f"ex={hybrid['external_reliability']:.2f}) | "
        f"verdict={verdict_result['verdict']}"
    )
    return result


def ensemble_batch(
    articles: list, rule_scores: list, semantic_scores: list,
    classifier_results: list, agency_results: list = None,
    features_list: list = None,
) -> list:
    """배치 앙상블 (v7 원본 로직 완전 보존)."""
    if not articles:
        logger.warning("앙상블 입력 기사 목록이 비어있음")
        return []

    n = len(articles)
    logger.info(f"[앙상블] 시작 | {n}건")

    rule_scores = _pad(rule_scores, n, {"rule_score": 0.0, "rule_reason": ""})
    semantic_scores = _pad(semantic_scores, n, 0.0)
    classifier_results = _pad(classifier_results, n, {"score": 0.0})
    agency_results = _pad(agency_results or [], n, {"agency_score": 0.0})
    features_list = _pad(features_list or [], n, {})

    results = []
    method_counts = {"low_confidence": 0, "normal": 0, "high_confidence": 0}
    direct_count = 0
    boost_count = 0

    for i, article in enumerate(articles):
        try:
            raw = rule_scores[i]
            rule_result = raw if isinstance(raw, dict) else {
                "rule_score": float(raw) if raw else 0.0, "rule_reason": "",
            }
            sem = float(semantic_scores[i]) if semantic_scores[i] is not None else 0.0
            clf = classifier_results[i]
            if isinstance(clf, dict):
                clf_score = float(clf.get("classifier_score", clf.get("score", 0.0)))
            else:
                clf_score = float(clf) if clf is not None else 0.0
            agn = agency_results[i]
            if isinstance(agn, dict):
                agn_score = float(agn.get("agency_score", 0.0))
                vmsg = agn.get("verification_message", "")
            else:
                agn_score = float(agn) if agn is not None else 0.0
                vmsg = ""
            rel_score = float(article.get("reliability_score", 0.0))
            rel_reason = article.get("reliability_reason", "")
            feat = features_list[i] if i < len(features_list) else {}
            hds = feat.get("has_direct_speech", False)

            r = ensemble_single(
                article=article, rule_result=rule_result,
                semantic_score=sem, classifier_score=clf_score,
                agency_score=agn_score, reliability_score=rel_score,
                reliability_reason=rel_reason,
                verification_message=vmsg, has_direct_speech=hds,
            )
            results.append(r)
            m = r.get("score_method", "normal")
            if m in method_counts:
                method_counts[m] += 1
            if hds:
                direct_count += 1
            if r.get("reliability_boosted", False):
                boost_count += 1
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(f"앙상블 처리 실패 | idx={i} | error={exc}")
            results.append(_default(article))

    verified = sum(1 for r in results if r.get("is_verified", False))
    avg_off = sum(r.get("official_score", 0) for r in results) / max(n, 1)
    avg_rel = sum(r.get("reliability_score", 0) for r in results) / max(n, 1)
    logger.info(
        f"[앙상블] 완료 | 오피셜={verified}/{n} | "
        f"평균공식성={avg_off:.4f} 평균신뢰성={avg_rel:.4f} | "
        f"구간=[low={method_counts['low_confidence']} "
        f"norm={method_counts['normal']} "
        f"high={method_counts['high_confidence']}] | "
        f"직접발화={direct_count} 교차보정={boost_count}"
    )
    return results


def _pad(lst, n, default):
    if lst is None:
        return [default] * n
    r = list(lst)
    while len(r) < n:
        r.append(default)
    return r


def _default(article: dict) -> dict:
    return {
        "title": article.get("title", ""), "source": article.get("source", ""),
        "originallink": article.get("originallink", ""),
        "pubDate": article.get("pubDate", ""), "domain": article.get("domain", ""),
        "rule_score": 0.0, "rule_reason": "처리 실패",
        "semantic_score": 0.0, "classifier_score": 0.0,
        "agency_score": 0.0, "official_score": 0.0, "score_method": "error",
        "reliability_score": 0.0, "reliability_reason": "처리 실패",
        "internal_reliability": 0.0, "external_reliability": 0.0,
        "reliability_breakdown": {
            "source_accountability": 0.0, "verifiability": 0.0, "neutrality": 0.0,
        },
        "internal_details": {}, "external_details": {},
        "weights": {"internal": 0.5, "external": 0.5},
        "cluster_id": -1, "cluster_size": 1,
        "unique_sources": 1, "has_official_domain": False, "avg_similarity": 0.0,
        "verdict": "검증 불가", "verdict_emoji": "❌",
        "verdict_reason": "앙상블 처리 중 오류 발생", "is_verified": False,
        "explanation": "❌ 판정 실패 — 앙상블 처리 중 오류 발생",
        "verification_message": "", "has_direct_speech": False,
        "direct_speech_boosted": False, "reliability_boosted": False,
        "reliability_boost_amount": 0.0,
        "final_official_score": 0.0, "predicted_label": "검증 불가",
    }
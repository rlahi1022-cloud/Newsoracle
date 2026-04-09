"""
services/ensemble.py
────────────────────────────────────────────────────────────────────────────────
공식성 점수 + 신뢰성 점수를 통합하여 최종 판정을 내리는 앙상블 모듈

[변경 이력]
  v1: rule + semantic + classifier + agency 가중 합산
  v2: official_score + reliability_score 분리 판정
  v3: classifier_score 키 호환성 수정
  v4: classifier 저신뢰 분기 로직 추가
      - classifier < 0.5 → rule+semantic+agency 기반 재계산
      - classifier >= 0.5 → 기존 가중 합산
      이유: classifier가 연예 기사를 전부 비공식(0.001)으로 분류하여
            [공식] 태그, 소속사 입장문 등을 잡아도 공식성이 낮게 나옴
      해결: classifier가 확신 없으면 제외하고 다른 피처로 판단
"""

from logger import get_logger
from config import ENSEMBLE_WEIGHTS, OFFICIAL_THRESHOLD, EnsembleConfig
from services.cross_validator import determine_final_verdict

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────────────────────
# 단일 기사 앙상블
# ────────────────────────────────────────────────────────────────────────────────

def ensemble_single(
    article: dict,
    rule_score: float,
    semantic_score: float,
    classifier_score: float,
    agency_score: float = 0.0,
    reliability_score: float = 0.0,
    reliability_reason: str = "",
) -> dict:
    """
    단일 기사에 대해 공식성 점수와 신뢰성 점수를 통합하여 최종 판정을 반환한다.

    [v4 분기 로직]
    classifier_score >= 0.5 (확신 있음):
      → 기존 가중 합산: rule*0.30 + semantic*0.10 + classifier*0.40
      → agency 보너스 가산

    classifier_score < 0.5 (확신 없음):
      → classifier를 제외하고 rule+semantic+agency로 재계산
      → rule*0.60 + semantic*0.15 + agency*0.25
      → 연예/스포츠 [공식] 기사가 classifier에 의해 눌리는 문제 해결
    """
    # ── 공식성 점수 계산 ─────────────────────────────────────────────────────

    if classifier_score >= EnsembleConfig.CLASSIFIER_LOW_CONFIDENCE:
        # classifier가 확신 있음 → 기존 가중 합산
        official_score = (
            rule_score       * EnsembleConfig.RULE_WEIGHT
            + semantic_score * EnsembleConfig.SEMANTIC_WEIGHT
            + classifier_score * EnsembleConfig.CLASSIFIER_WEIGHT
        )
        score_method = "classifier_included"
    else:
        # classifier가 확신 없음 → classifier 제외하고 재계산
        # rule + semantic + agency로 공식성 판단
        # 이 경우 [공식] 태그(rule), 기관 검증(agency)이 주도
        official_score = (
            rule_score       * EnsembleConfig.FALLBACK_RULE_WEIGHT
            + semantic_score * EnsembleConfig.FALLBACK_SEMANTIC_WEIGHT
            + agency_score   * EnsembleConfig.FALLBACK_AGENCY_WEIGHT
        )
        score_method = "classifier_excluded"
        logger.debug(
            f"classifier 저신뢰({classifier_score:.4f}) → "
            f"rule+semantic+agency 기반 재계산 | "
            f"title={article.get('title', '')[:30]}"
        )

    # agency 보너스 (classifier 포함 경로에서만 적용)
    if score_method == "classifier_included" and agency_score > 0:
        agency_bonus = agency_score * EnsembleConfig.AGENCY_BONUS_MAX
        official_score = min(official_score + agency_bonus, 1.0)

    official_score = round(float(official_score), 4)

    # ── 최종 판정 ────────────────────────────────────────────────────────────
    verdict_result = determine_final_verdict(
        official_score=official_score,
        reliability_score=reliability_score
    )

    # ── 결과 딕셔너리 구성 ───────────────────────────────────────────────────
    result = {
        # 기사 기본 정보
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "originallink": article.get("originallink", ""),
        "pubDate": article.get("pubDate", ""),
        "domain": article.get("domain", ""),

        # 공식성 세부 점수
        "rule_score": round(float(rule_score), 4),
        "semantic_score": round(float(semantic_score), 4),
        "classifier_score": round(float(classifier_score), 4),
        "agency_score": round(float(agency_score), 4),
        "official_score": official_score,
        "score_method": score_method,  # v4: 어떤 경로로 계산됐는지 추적

        # 신뢰성 세부 정보
        "reliability_score": round(float(reliability_score), 4),
        "reliability_reason": reliability_reason,
        "cluster_id": article.get("cluster_id", -1),
        "cluster_size": article.get("cluster_size", 1),
        "unique_sources": article.get("unique_sources", 1),
        "has_official_domain": article.get("has_official_domain", False),

        # 최종 판정
        "verdict": verdict_result["verdict"],
        "verdict_emoji": verdict_result["verdict_emoji"],
        "verdict_reason": verdict_result["verdict_reason"],
        "is_verified": verdict_result["is_verified"],

        # 하위 호환성
        "final_official_score": official_score,
        "predicted_label": verdict_result["verdict"],
    }

    logger.debug(
        f"앙상블 완료 | "
        f"title={article.get('title', '')[:20]} | "
        f"official={official_score:.4f} ({score_method}) | "
        f"reliability={reliability_score:.4f} | "
        f"verdict={verdict_result['verdict']}"
    )

    return result


# ────────────────────────────────────────────────────────────────────────────────
# 배치 앙상블
# ────────────────────────────────────────────────────────────────────────────────

def ensemble_batch(
    articles: list[dict],
    rule_scores: list[float],
    semantic_scores: list[float],
    classifier_results: list[dict],
    agency_results: list[dict] = None,
) -> list[dict]:
    """
    기사 목록 전체에 앙상블을 적용하여 최종 판정 결과 목록을 반환한다.
    """
    if not articles:
        logger.warning("앙상블 입력 기사 목록이 비어있음")
        return []

    n = len(articles)
    logger.info(f"[앙상블] 시작 | {n}건")

    # 길이 불일치 방어
    rule_scores = _pad_list(rule_scores, n, 0.0)
    semantic_scores = _pad_list(semantic_scores, n, 0.0)
    classifier_results = _pad_list(classifier_results, n, {"score": 0.0})
    agency_results = _pad_list(agency_results or [], n, {"agency_score": 0.0})

    results = []
    classifier_excluded_count = 0  # v4: classifier 제외된 기사 수 추적

    for i, article in enumerate(articles):
        try:
            rule_score = float(rule_scores[i]) if rule_scores[i] is not None else 0.0
            semantic_score = float(semantic_scores[i]) if semantic_scores[i] is not None else 0.0

            clf_result = classifier_results[i]
            if isinstance(clf_result, dict):
                classifier_score = float(clf_result.get("classifier_score", clf_result.get("score", 0.0)))
            else:
                classifier_score = float(clf_result) if clf_result is not None else 0.0

            agn_result = agency_results[i]
            if isinstance(agn_result, dict):
                agency_score = float(agn_result.get("agency_score", 0.0))
            else:
                agency_score = float(agn_result) if agn_result is not None else 0.0

            reliability_score = float(article.get("reliability_score", 0.0))
            reliability_reason = article.get("reliability_reason", "")

            result = ensemble_single(
                article=article,
                rule_score=rule_score,
                semantic_score=semantic_score,
                classifier_score=classifier_score,
                agency_score=agency_score,
                reliability_score=reliability_score,
                reliability_reason=reliability_reason,
            )
            results.append(result)

            # classifier 제외 카운트
            if result.get("score_method") == "classifier_excluded":
                classifier_excluded_count += 1

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(f"앙상블 처리 실패 | idx={i} | error={exc}")
            results.append(_default_result(article))

    # 최종 요약 로그
    verified_count = sum(1 for r in results if r.get("is_verified", False))
    avg_official = sum(r.get("official_score", 0) for r in results) / max(n, 1)
    avg_reliability = sum(r.get("reliability_score", 0) for r in results) / max(n, 1)

    logger.info(
        f"[앙상블] 완료 | "
        f"오피셜 검증됨={verified_count}/{n}건 | "
        f"평균 공식성={avg_official:.4f} | "
        f"평균 신뢰성={avg_reliability:.4f} | "
        f"classifier 제외={classifier_excluded_count}건"
    )

    return results


def _pad_list(lst: list, target_len: int, default) -> list:
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
        "rule_score": 0.0,
        "semantic_score": 0.0,
        "classifier_score": 0.0,
        "agency_score": 0.0,
        "official_score": 0.0,
        "score_method": "error",
        "reliability_score": 0.0,
        "reliability_reason": "처리 실패",
        "cluster_id": -1,
        "cluster_size": 1,
        "unique_sources": 1,
        "has_official_domain": False,
        "verdict": "검증 불가",
        "verdict_emoji": "❌",
        "verdict_reason": "앙상블 처리 중 오류 발생",
        "is_verified": False,
        "final_official_score": 0.0,
        "predicted_label": "검증 불가",
    }
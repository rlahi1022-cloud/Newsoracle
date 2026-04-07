# 최종 점수 계산기
"""
services/ensemble.py
─────────────────────
규칙 기반 점수 / 의미 유사도 점수 / 분류 모델 점수를
가중 평균으로 합산하여 최종 공식성 점수를 산출한다.
"""

from config import ENSEMBLE_WEIGHTS, OFFICIAL_THRESHOLD
from logger import get_logger

logger = get_logger("ensemble")


def compute_final_score(
    rule_score: float,
    semantic_score: float,
    classifier_score: float,
) -> dict:
    """
    세 모델의 점수를 앙상블하여 최종 공식성 점수를 계산한다.
    
    공식:
      final_score = w_rule * rule_score
                  + w_semantic * semantic_score
                  + w_classifier * classifier_score
    
    Args:
        rule_score:       규칙 기반 점수 (0.0 ~ 1.0)
        semantic_score:   의미 유사도 점수 (0.0 ~ 1.0)
        classifier_score: 분류 모델 점수 (0.0 ~ 1.0)
    Returns:
        {
            "final_official_score": float,
            "predicted_label": str  ("공식성 높음" / "공식성 낮음")
        }
    """
    try:
        w_rule = ENSEMBLE_WEIGHTS["rule"]
        w_sem = ENSEMBLE_WEIGHTS["semantic"]
        w_cls = ENSEMBLE_WEIGHTS["classifier"]

        # 가중치 합이 1.0인지 검증
        total_weight = w_rule + w_sem + w_cls
        if abs(total_weight - 1.0) > 1e-6:
            logger.warning(f"앙상블 가중치 합이 1.0이 아님: {total_weight:.4f} → 자동 정규화")
            w_rule /= total_weight
            w_sem /= total_weight
            w_cls /= total_weight

        final_score = (
            w_rule * rule_score
            + w_sem * semantic_score
            + w_cls * classifier_score
        )
        final_score = round(max(0.0, min(final_score, 1.0)), 4)

        label = "공식성 높음" if final_score >= OFFICIAL_THRESHOLD else "공식성 낮음"

        return {
            "final_official_score": final_score,
            "predicted_label": label,
        }

    except Exception as e:
        logger.error(f"앙상블 점수 계산 실패: {e}")
        return {"final_official_score": 0.0, "predicted_label": "공식성 낮음"}


def ensemble_batch(
    articles: list[dict],
    rule_scores: list[float],
    semantic_scores: list[float],
    classifier_results: list[dict],
) -> list[dict]:
    """
    기사 목록 전체에 앙상블을 적용하고 결과를 통합한다.
    
    Args:
        articles:           전처리된 기사 딕셔너리 리스트
        rule_scores:        규칙 기반 점수 리스트
        semantic_scores:    의미 유사도 점수 리스트
        classifier_results: 분류 모델 결과 리스트 ({"classifier_score": ..., ...})
    Returns:
        최종 결과 딕셔너리 리스트
    """
    if not articles:
        logger.warning("앙상블 입력 데이터가 비어 있음")
        return []

    logger.info(f"앙상블 시작 | {len(articles)}건")

    results = []
    for i, article in enumerate(articles):
        try:
            rule_score = rule_scores[i] if i < len(rule_scores) else 0.0
            semantic_score = semantic_scores[i] if i < len(semantic_scores) else 0.0
            cls_result = classifier_results[i] if i < len(classifier_results) else {}
            classifier_score = cls_result.get("classifier_score", 0.0)

            ensemble_result = compute_final_score(rule_score, semantic_score, classifier_score)

            # 기사 메타정보 + 각 모델 점수 + 최종 결과 통합
            output = {
                "title": article.get("title", ""),
                "source": article.get("domain", ""),
                "originallink": article.get("originallink", ""),
                "pubDate": article.get("pubDate", ""),
                "rule_score": rule_score,
                "semantic_score": semantic_score,
                "classifier_score": classifier_score,
                "final_official_score": ensemble_result["final_official_score"],
                "predicted_label": ensemble_result["predicted_label"],
            }
            results.append(output)

        except Exception as e:
            logger.error(f"앙상블 처리 실패 | index={i} | {e}")
            continue

    logger.info(f"앙상블 완료 | {len(results)}건")
    return results
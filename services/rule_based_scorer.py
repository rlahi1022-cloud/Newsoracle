# 규칙 기반 점수 계산

"""
services/rule_based_scorer.py
──────────────────────────────
특징 벡터를 기반으로 규칙 기반 공식성 점수를 계산한다.
딥러닝 모델 없이도 빠르게 베이스라인 점수를 산출한다.
"""

from logger import get_logger

logger = get_logger("rule_based_scorer")

# 각 특징의 가중치 정의
# 합산 후 정규화하므로 합이 1.0일 필요는 없음
FEATURE_WEIGHTS = {
    "domain_score": 0.35,
    "official_expr_score": 0.25,
    "org_name_score": 0.20,
    "text_length_score": 0.10,
    "link_score": 0.10,
}


def compute_rule_score(features: dict) -> float:
    """
    특징 딕셔너리에서 규칙 기반 공식성 점수를 계산한다.
    
    각 특징에 가중치를 곱한 뒤 합산하여 0.0 ~ 1.0 점수 반환.
    
    Args:
        features: feature_extractor.py에서 추출된 특징 딕셔너리
    Returns:
        0.0 ~ 1.0 사이 rule_score
    """
    if not features:
        return 0.0

    try:
        score = 0.0
        total_weight = sum(FEATURE_WEIGHTS.values())

        for feature_name, weight in FEATURE_WEIGHTS.items():
            value = features.get(feature_name, 0.0)
            score += value * weight

        # 가중치 합으로 나눠서 0~1 범위로 정규화
        normalized_score = score / total_weight
        return round(normalized_score, 4)

    except Exception as e:
        logger.error(f"규칙 기반 점수 계산 실패: {e}")
        return 0.0


def compute_rule_scores_batch(features_list: list[dict]) -> list[float]:
    """
    기사 특징 목록 전체에 대해 규칙 기반 점수를 계산한다.
    
    Args:
        features_list: 특징 딕셔너리 리스트
    Returns:
        각 기사의 rule_score 리스트
    """
    if not features_list:
        logger.warning("규칙 기반 스코어러 입력이 비어 있음")
        return []

    logger.info(f"규칙 기반 점수 계산 시작 | {len(features_list)}건")

    scores = [compute_rule_score(f) for f in features_list]

    logger.info(f"규칙 기반 점수 계산 완료 | 평균 점수: {sum(scores)/len(scores):.4f}")
    return scores
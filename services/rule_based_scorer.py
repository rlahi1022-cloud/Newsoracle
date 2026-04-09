"""
services/rule_based_scorer.py
──────────────────────────────
특징 벡터를 기반으로 규칙 기반 공식성 점수를 계산한다.

[v2 재설계 이유]
기존 구조의 정규화 버그:
  total_weight = sum(weights) = 1.0 이므로
  score / total_weight = score / 1.0 = score 그대로
  → 정규화가 전혀 이루어지지 않았음

더 큰 문제: 피처 자체의 값 분포가 고르지 않음
  link_score:        모든 기사 1.0 고정 (변별력 0)
  text_length_score: 모든 기사 ~0.20 고정 (변별력 0)
  domain_score:      언론사 도메인 전부 미매칭 → 0.0 (변별력 0)

[v2 변경 내용]
1. 피처 재구성
   - link_score, text_length_score 제거 (변별력 없음)
   - domain_score → domain_grade_score (5등급 체계, 실제 언론사 포함)
   - quote_score, stat_score, title_format_score 신규 추가

2. 정규화 방식 수정
   - 기존: score / sum(weights) → sum(weights)=1.0이라 무의미
   - 수정: 각 피처의 이론적 최대값 대비 실제 값 비율로 계산
   - 즉, "모든 피처가 최대값일 때 = 1.0" 보장

3. 피처별 가중치 재설계
   - domain_grade_score:  0.30 (출처 자체의 공신력)
   - official_expr_score: 0.30 (공식 발표 표현 → 가장 직접적 지표)
   - org_name_score:      0.20 (기관/기업명 등장)
   - quote_score:         0.10 (직접 인용문)
   - stat_score:          0.05 (수치/통계)
   - title_format_score:  0.05 (제목 구조)
"""

from logger import get_logger

logger = get_logger("rule_based_scorer")

# ─────────────────────────────────────────────────────────────
# 피처별 가중치
# 합계 = 1.0 이어야 함 (아래에서 검증)
# ─────────────────────────────────────────────────────────────
FEATURE_WEIGHTS = {
    # 출처 자체의 공신력 (정부도메인 > 연합 > 종합일간지 > 경제지 > 인터넷뉴스)
    "domain_grade_score":  0.30,

    # 공식 발표 표현 강도 (가장 직접적인 공식성 지표)
    "official_expr_score": 0.30,

    # 기관/기업명 등장 여부
    "org_name_score":      0.20,

    # 직접 인용문(" ") 포함 여부
    "quote_score":         0.10,

    # 수치/통계 데이터 포함 여부
    "stat_score":          0.05,

    # 제목 구조 공식성 ([기관명] 형식 등)
    "title_format_score":  0.05,
}

# 가중치 합 검증 (합이 1.0이 아니면 시작 시 경고)
_total_weight = sum(FEATURE_WEIGHTS.values())
assert abs(_total_weight - 1.0) < 1e-9, (
    f"FEATURE_WEIGHTS 합이 1.0이 아님: {_total_weight:.4f}. "
    "가중치를 수정하세요."
)


def compute_rule_score(features: dict) -> float:
    """
    특징 딕셔너리에서 규칙 기반 공식성 점수를 계산한다.

    [정규화 방식]
    각 피처값은 이미 0.0~1.0 범위로 정규화되어 feature_extractor에서 반환됨.
    여기서는 피처값 * 가중치의 합산만 수행.
    모든 피처가 1.0일 때 최종 score = 1.0 보장.

    Args:
        features: feature_extractor.py v2에서 추출된 특징 딕셔너리
    Returns:
        0.0 ~ 1.0 사이 rule_score
    """
    if not features:
        return 0.0

    try:
        score = 0.0
        for feature_name, weight in FEATURE_WEIGHTS.items():
            value = float(features.get(feature_name, 0.0))
            # 피처값이 0~1 범위를 벗어나는 경우 클램핑
            value = max(0.0, min(value, 1.0))
            score += value * weight

        return round(score, 4)

    except Exception as e:
        logger.error(f"규칙 기반 점수 계산 실패: {e}")
        return 0.0


def compute_rule_scores_batch(features_list: list) -> list:
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

    avg = sum(scores) / len(scores)
    min_s = min(scores)
    max_s = max(scores)
    logger.info(
        f"규칙 기반 점수 계산 완료 | "
        f"평균={avg:.4f} 최솟값={min_s:.4f} 최댓값={max_s:.4f}"
    )
    return scores
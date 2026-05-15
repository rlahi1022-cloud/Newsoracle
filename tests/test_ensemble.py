"""
tests/test_ensemble.py
─────────────────────────────────────
services/ensemble 의 보조 함수(_clip_score, _clip_classifier, _select_weights,
compute_external_reliability) 검증. 임베딩이나 분류기 로딩이 필요한
ensemble_batch / ensemble_single 통합 흐름은 다른 통합 테스트에서 다룬다.

[왜 보조 함수만 따로 테스트하는가]
  v7 conditional weighting 의 핵심은 _select_weights 가 classifier 점수에 따라
  3구간 가중치를 정확히 골라주는 것. 가중치 합 ≠ 1.0 이거나 분기 임계값이
  잘못되면 모든 기사 점수가 동시에 어긋난다.
  같은 이유로 외부 신뢰성(compute_external_reliability) 도 단독/다중 출처,
  공식 도메인 보너스, 입력 이상치(문자열 등) 같은 경계 케이스를 잡아야 한다.

[테스트 전략]
  1. 클리핑: 점수가 [SCORE_CLIP_MIN, SCORE_CLIP_MAX] 안에 들어오는지
  2. 가중치 선택: 3구간(low/normal/high) 라벨과 합 == 1.0
  3. 외부 신뢰성: 단독 보도 시 0.3 캡, 다중 출처가 더 높음, 공식 도메인 보너스
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.ensemble import (
    _clip_score,
    _clip_classifier,
    _select_weights,
    compute_external_reliability,
    SCORE_CLIP_MIN,
    SCORE_CLIP_MAX,
)
from config import EnsembleConfig


class TestScoreClipping(unittest.TestCase):
    # 점수 클리핑은 앙상블 입력의 안전벨트.
    # 클리핑이 빠지면 외부 모델 버그(2.0, -1.0 등)가 그대로 verdict 까지 흘러간다.

    def test_clip_score_within_bounds(self):
        # 일반 점수: [SCORE_CLIP_MIN, SCORE_CLIP_MAX] 안에 강제 정렬
        self.assertEqual(_clip_score(-1.0), SCORE_CLIP_MIN)
        self.assertEqual(_clip_score(2.0), SCORE_CLIP_MAX)
        self.assertEqual(_clip_score(0.5), 0.5)

    def test_clip_classifier_uses_separate_bounds(self):
        # classifier 는 일반 점수보다 좁은 범위(0.05~0.95)로 별도 클리핑.
        # 0/1 극단값이 conditional weighting 의 분기 임계값을 왜곡하는 것을 막기 위함.
        self.assertEqual(_clip_classifier(0.0), EnsembleConfig.CLASSIFIER_CLIP_MIN)
        self.assertEqual(_clip_classifier(1.0), EnsembleConfig.CLASSIFIER_CLIP_MAX)
        self.assertEqual(_clip_classifier(0.5), 0.5)


class TestSelectWeights(unittest.TestCase):
    # 3구간 가중치 분기가 v7 conditional weighting 의 핵심.
    # 라벨이 잘못 나오거나 임계값이 어긋나면 앙상블 전체가 빗나간다.

    def test_low_confidence_branch(self):
        # classifier < CLASSIFIER_LOW_BOUNDARY → LOW 구간(rule 주도)
        weights = _select_weights(EnsembleConfig.CLASSIFIER_LOW_BOUNDARY - 0.01)
        # 반환 튜플의 마지막 원소가 구간 라벨
        self.assertEqual(weights[-1], "low_confidence")

    def test_normal_branch(self):
        # LOW < classifier < HIGH → NORMAL(균등)
        mid = (EnsembleConfig.CLASSIFIER_LOW_BOUNDARY + EnsembleConfig.CLASSIFIER_HIGH_BOUNDARY) / 2
        weights = _select_weights(mid)
        self.assertEqual(weights[-1], "normal")

    def test_high_confidence_branch(self):
        # classifier > CLASSIFIER_HIGH_BOUNDARY → HIGH(과신 방지)
        weights = _select_weights(EnsembleConfig.CLASSIFIER_HIGH_BOUNDARY + 0.01)
        self.assertEqual(weights[-1], "high_confidence")

    def test_weights_sum_to_one(self):
        # 3구간 어디서든 가중치 합 == 1.0 이어야 점수 스케일이 보존된다.
        # 한 구간이라도 합이 어긋나면 그 구간 기사들만 점수 분포가 기울어진다.
        for score in [0.0, 0.5, 0.99]:
            rule_w, sem_w, cls_w, agency_w, _ = _select_weights(score)
            total = rule_w + sem_w + cls_w + agency_w
            self.assertAlmostEqual(total, 1.0, places=3,
                                   msg=f"가중치 합 != 1.0 at classifier={score}")


class TestComputeExternalReliability(unittest.TestCase):
    # 외부 신뢰성 = 교차 보도/독립 출처/공식 도메인/내용 일치도 종합.
    # 단독 보도 캡(0.3)이 없으면 1개 출처 짜리 가짜 뉴스도 점수가 부풀어 오른다.

    def test_single_source_capped(self):
        # 단독 보도(cluster_size=1)는 외부 검증이 사실상 불가 → 0.3 캡
        result = compute_external_reliability({
            "cluster_size": 1, "unique_sources": 1,
            "has_official_domain": False, "avg_similarity": 0.0,
        })
        self.assertLessEqual(result["external_reliability"], 0.3)

    def test_multi_source_higher(self):
        # 출처가 많을수록 점수가 단조 증가해야 한다(monotonic).
        # 그렇지 않으면 "여러 언론사 교차 보도 = 더 신뢰" 라는 설계 의도가 무너짐.
        result_solo = compute_external_reliability({
            "cluster_size": 1, "unique_sources": 1,
            "has_official_domain": False, "avg_similarity": 0.0,
        })
        result_multi = compute_external_reliability({
            "cluster_size": 5, "unique_sources": 4,
            "has_official_domain": False, "avg_similarity": 0.8,
        })
        self.assertGreater(
            result_multi["external_reliability"],
            result_solo["external_reliability"],
        )

    def test_official_domain_bonus(self):
        # 다른 조건이 같을 때 공식 도메인(.go.kr 등) 포함이면 점수가 더 높아야 함
        base = compute_external_reliability({
            "cluster_size": 3, "unique_sources": 3,
            "has_official_domain": False, "avg_similarity": 0.5,
        })
        bonus = compute_external_reliability({
            "cluster_size": 3, "unique_sources": 3,
            "has_official_domain": True, "avg_similarity": 0.5,
        })
        self.assertGreater(
            bonus["external_reliability"],
            base["external_reliability"],
        )

    def test_result_within_unit_interval(self):
        # 어떤 입력이 들어와도 점수는 [0, 1] 범위 — 앙상블 합산 시 스케일 보존
        for ci in [
            {},
            {"cluster_size": 100, "unique_sources": 100, "has_official_domain": True, "avg_similarity": 1.0},
            {"cluster_size": 0, "unique_sources": 0, "has_official_domain": False, "avg_similarity": -1.0},
        ]:
            r = compute_external_reliability(ci)
            self.assertGreaterEqual(r["external_reliability"], 0.0)
            self.assertLessEqual(r["external_reliability"], 1.0)

    def test_malformed_input_safe(self):
        # cluster_size 가 문자열로 들어오는 등 잘못된 입력에도 폴백 결과 반환.
        # try/except 가 빠지면 한 기사 때문에 배치 전체가 죽을 수 있다.
        result = compute_external_reliability({"cluster_size": "bad"})
        self.assertIn("external_reliability", result)
        self.assertGreaterEqual(result["external_reliability"], 0.0)
        self.assertLessEqual(result["external_reliability"], 1.0)


if __name__ == "__main__":
    unittest.main()

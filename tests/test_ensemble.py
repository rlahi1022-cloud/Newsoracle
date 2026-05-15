"""
tests/test_ensemble.py
─────────────────────────────────────
services/ensemble 의 보조 함수(_clip_score, _clip_classifier, _select_weights,
compute_external_reliability) 검증. 임베딩이나 분류기 로딩이 필요한
ensemble_batch / ensemble_single 통합 흐름은 다른 통합 테스트에서 다룬다.
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
    def test_clip_score_within_bounds(self):
        self.assertEqual(_clip_score(-1.0), SCORE_CLIP_MIN)
        self.assertEqual(_clip_score(2.0), SCORE_CLIP_MAX)
        self.assertEqual(_clip_score(0.5), 0.5)

    def test_clip_classifier_uses_separate_bounds(self):
        # classifier 는 0.05~0.95 클립 (config 값 기준)
        self.assertEqual(_clip_classifier(0.0), EnsembleConfig.CLASSIFIER_CLIP_MIN)
        self.assertEqual(_clip_classifier(1.0), EnsembleConfig.CLASSIFIER_CLIP_MAX)
        self.assertEqual(_clip_classifier(0.5), 0.5)


class TestSelectWeights(unittest.TestCase):
    def test_low_confidence_branch(self):
        weights = _select_weights(EnsembleConfig.CLASSIFIER_LOW_BOUNDARY - 0.01)
        # 마지막 원소가 라벨
        self.assertEqual(weights[-1], "low_confidence")

    def test_normal_branch(self):
        mid = (EnsembleConfig.CLASSIFIER_LOW_BOUNDARY + EnsembleConfig.CLASSIFIER_HIGH_BOUNDARY) / 2
        weights = _select_weights(mid)
        self.assertEqual(weights[-1], "normal")

    def test_high_confidence_branch(self):
        weights = _select_weights(EnsembleConfig.CLASSIFIER_HIGH_BOUNDARY + 0.01)
        self.assertEqual(weights[-1], "high_confidence")

    def test_weights_sum_to_one(self):
        for score in [0.0, 0.5, 0.99]:
            rule_w, sem_w, cls_w, agency_w, _ = _select_weights(score)
            total = rule_w + sem_w + cls_w + agency_w
            self.assertAlmostEqual(total, 1.0, places=3,
                                   msg=f"가중치 합 != 1.0 at classifier={score}")


class TestComputeExternalReliability(unittest.TestCase):
    def test_single_source_capped(self):
        # 단독 보도(cluster_size=1)는 외부 검증 불가 → 0.3 캡
        result = compute_external_reliability({
            "cluster_size": 1, "unique_sources": 1,
            "has_official_domain": False, "avg_similarity": 0.0,
        })
        self.assertLessEqual(result["external_reliability"], 0.3)

    def test_multi_source_higher(self):
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
        # 어떤 입력이라도 0~1 범위
        for ci in [
            {},
            {"cluster_size": 100, "unique_sources": 100, "has_official_domain": True, "avg_similarity": 1.0},
            {"cluster_size": 0, "unique_sources": 0, "has_official_domain": False, "avg_similarity": -1.0},
        ]:
            r = compute_external_reliability(ci)
            self.assertGreaterEqual(r["external_reliability"], 0.0)
            self.assertLessEqual(r["external_reliability"], 1.0)

    def test_malformed_input_safe(self):
        # 이상한 타입이 와도 폴백 결과를 반환해야 함
        result = compute_external_reliability({"cluster_size": "bad"})
        self.assertIn("external_reliability", result)
        self.assertGreaterEqual(result["external_reliability"], 0.0)
        self.assertLessEqual(result["external_reliability"], 1.0)


if __name__ == "__main__":
    unittest.main()

"""
tests/test_rule_based_scorer.py
─────────────────────────────────────
services/rule_based_scorer.py 의 규칙 기반 점수 계산을 검증한다.
가중치 합/범위/감점/배치 처리/근거 메시지 생성을 다룬다.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.rule_based_scorer import (
    FEATURE_WEIGHTS,
    UNOFFICIAL_PENALTY_WEIGHT,
    compute_rule_score,
    compute_rule_scores_batch,
)


class TestFeatureWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        total = sum(FEATURE_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_all_weights_positive(self):
        for name, weight in FEATURE_WEIGHTS.items():
            self.assertGreater(weight, 0.0, f"{name} 가중치가 0 이하: {weight}")


class TestComputeRuleScore(unittest.TestCase):
    def test_empty_features_returns_zero(self):
        result = compute_rule_score({})
        self.assertEqual(result["rule_score"], 0.0)
        self.assertEqual(result["rule_reason"], "특징 데이터 없음")

    def test_all_zero_features(self):
        features = {name: 0.0 for name in FEATURE_WEIGHTS}
        features["unofficial_expr_score"] = 0.0
        result = compute_rule_score(features)
        self.assertEqual(result["rule_score"], 0.0)

    def test_all_max_features(self):
        features = {name: 1.0 for name in FEATURE_WEIGHTS}
        features["unofficial_expr_score"] = 0.0
        result = compute_rule_score(features)
        # 모든 양수 가중치 합은 1.0
        self.assertAlmostEqual(result["rule_score"], 1.0, places=3)

    def test_unofficial_penalty_subtracts(self):
        features = {name: 1.0 for name in FEATURE_WEIGHTS}
        features["unofficial_expr_score"] = 1.0
        result = compute_rule_score(features)
        # 1.0 - 0.20 패널티 = 0.80
        expected = round(1.0 - UNOFFICIAL_PENALTY_WEIGHT, 4)
        self.assertAlmostEqual(result["rule_score"], expected, places=3)

    def test_clamps_to_zero_minimum(self):
        # 양수 피처 0 + 비공식 1.0 → 음수가 되면 0으로 클램프
        features = {"unofficial_expr_score": 1.0}
        result = compute_rule_score(features)
        self.assertGreaterEqual(result["rule_score"], 0.0)

    def test_out_of_range_values_clamped(self):
        # 정상 입력 범위(0~1)를 벗어난 값도 안전하게 처리되어야 함
        features = {name: 5.0 for name in FEATURE_WEIGHTS}  # 모두 5.0
        features["unofficial_expr_score"] = -1.0  # 음수
        result = compute_rule_score(features)
        # 1.0으로 클램프되어 최대 1.0
        self.assertLessEqual(result["rule_score"], 1.0)
        self.assertGreaterEqual(result["rule_score"], 0.0)

    def test_evidence_included_in_reason(self):
        features = {
            "domain_grade_score": 1.0,
            "official_expr_score": 0.5,
            "_evidence": {
                "official_strong": ["발표했다", "밝혔다"],
                "orgs_found": ["한국은행"],
                "unofficial_found": [],
                "domain": "bok.or.kr",
                "subject_verb_patterns": ["한국은행은 결정했다"],
                "has_direct_speech": True,
                "speech_verbs": ["고 말했다"],
            },
        }
        result = compute_rule_score(features)
        self.assertIn("발표했다", result["rule_reason"])
        self.assertIn("한국은행", result["rule_reason"])
        self.assertIn("bok.or.kr", result["rule_reason"])


class TestComputeRuleScoresBatch(unittest.TestCase):
    def test_batch_returns_same_length(self):
        feats = [
            {"domain_grade_score": 1.0},
            {"official_expr_score": 0.5},
            {},
        ]
        results = compute_rule_scores_batch(feats)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("rule_score", r)
            self.assertIn("rule_reason", r)

    def test_empty_input_returns_empty(self):
        self.assertEqual(compute_rule_scores_batch([]), [])


if __name__ == "__main__":
    unittest.main()

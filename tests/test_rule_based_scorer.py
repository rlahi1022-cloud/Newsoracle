"""
tests/test_rule_based_scorer.py
─────────────────────────────────────
services/rule_based_scorer.py 의 규칙 기반 점수 계산을 검증한다.
가중치 합/범위/감점/배치 처리/근거 메시지 생성을 다룬다.

[왜 이 모듈을 따로 테스트하는가]
  앙상블 최종 점수의 약 20~30% 비중을 차지하는 규칙 기반 점수가
  잘못 설계되면(예: 가중치 합 ≠ 1.0) 다른 모델 점수와 비교가 불가능해진다.
  특히 v5 에서 가중치가 8개로 재배분된 뒤로는 "합이 정확히 1.0인지"
  스모크 테스트로 항상 보장해야 회귀가 생기지 않는다.

[테스트 전략]
  1. 메타 검증: 가중치 합/부호
  2. 경계값: 모든 피처 0, 모든 피처 1, 비공식 패널티 최대치
  3. 입력 방어: 범위 밖 값 / 빈 입력 / 음수 입력
  4. 근거 메시지: rule_reason 에 evidence 가 사람이 읽을 수 있게 포함되는지
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
    # 가중치 dict 자체가 잘못 정의되면 점수 자체가 의미를 잃는다.
    # 코드를 만지지 않아도 회귀로 잡히도록 메타 검증을 둔다.

    def test_weights_sum_to_one(self):
        # 합이 1.0 이어야 다른 모델(0~1 범위)과 같은 스케일로 앙상블 가능
        total = sum(FEATURE_WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_all_weights_positive(self):
        # 양수 가중치는 8개 모두 0보다 커야 함 (0이면 해당 피처가 사실상 비활성)
        for name, weight in FEATURE_WEIGHTS.items():
            self.assertGreater(weight, 0.0, f"{name} 가중치가 0 이하: {weight}")


class TestComputeRuleScore(unittest.TestCase):
    # 핵심 계산 함수 — 입력 dict 에서 합산/감점/클램핑까지 한 번에 처리한다.

    def test_empty_features_returns_zero(self):
        # 빈 입력 → 0.0 (특징 추출이 실패한 케이스)
        result = compute_rule_score({})
        self.assertEqual(result["rule_score"], 0.0)
        self.assertEqual(result["rule_reason"], "특징 데이터 없음")

    def test_all_zero_features(self):
        # 모든 피처가 0 이면 점수도 0 (sanity check)
        features = {name: 0.0 for name in FEATURE_WEIGHTS}
        features["unofficial_expr_score"] = 0.0
        result = compute_rule_score(features)
        self.assertEqual(result["rule_score"], 0.0)

    def test_all_max_features(self):
        # 모든 양수 피처가 1 이면 가중치 합 == 1.0 이므로 점수도 1.0 이어야 함.
        # 이 케이스가 깨지면 가중치 합 검증과 연결된 회귀.
        features = {name: 1.0 for name in FEATURE_WEIGHTS}
        features["unofficial_expr_score"] = 0.0
        result = compute_rule_score(features)
        self.assertAlmostEqual(result["rule_score"], 1.0, places=3)

    def test_unofficial_penalty_subtracts(self):
        # 비공식 표현 1.0 일 때 패널티 -0.20 가 정확히 적용되는지
        features = {name: 1.0 for name in FEATURE_WEIGHTS}
        features["unofficial_expr_score"] = 1.0
        result = compute_rule_score(features)
        expected = round(1.0 - UNOFFICIAL_PENALTY_WEIGHT, 4)
        self.assertAlmostEqual(result["rule_score"], expected, places=3)

    def test_clamps_to_zero_minimum(self):
        # 양수 피처 0 + 비공식 1.0 → 이론상 -0.20 이지만 0 으로 클램프되어야 한다.
        # 클램프가 빠지면 앙상블 단계에서 음수 점수가 흘러 들어가 verdict 가 망가진다.
        features = {"unofficial_expr_score": 1.0}
        result = compute_rule_score(features)
        self.assertGreaterEqual(result["rule_score"], 0.0)

    def test_out_of_range_values_clamped(self):
        # 입력 검증 — feature_extractor 의 버그로 5.0 같은 값이 들어와도 점수가
        # 폭주하지 않아야 한다(0~1 범위 안에 머무름).
        features = {name: 5.0 for name in FEATURE_WEIGHTS}
        features["unofficial_expr_score"] = -1.0
        result = compute_rule_score(features)
        self.assertLessEqual(result["rule_score"], 1.0)
        self.assertGreaterEqual(result["rule_score"], 0.0)

    def test_evidence_included_in_reason(self):
        # 점수가 왜 그 값인지 사람이 읽을 수 있어야 한다.
        # _evidence 의 official_strong/orgs/domain 등이 rule_reason 에 포함되는지 확인.
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
    # 배치 입력 시 출력 개수/구조가 입력과 1:1 매칭되어야
    # 이후 앙상블 단계에서 zip() 으로 짝짓기가 깨지지 않는다.

    def test_batch_returns_same_length(self):
        feats = [
            {"domain_grade_score": 1.0},
            {"official_expr_score": 0.5},
            {},
        ]
        results = compute_rule_scores_batch(feats)
        self.assertEqual(len(results), 3)
        # 각 결과에 rule_score / rule_reason 가 모두 들어 있어야 함
        for r in results:
            self.assertIn("rule_score", r)
            self.assertIn("rule_reason", r)

    def test_empty_input_returns_empty(self):
        # 빈 입력은 빈 출력 — 예외 없이 통과해야 다른 모듈 점수와 zip 길이 안 어긋남
        self.assertEqual(compute_rule_scores_batch([]), [])


if __name__ == "__main__":
    unittest.main()

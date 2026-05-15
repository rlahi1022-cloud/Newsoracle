"""
tests/test_cross_validator.py
─────────────────────────────────────
services/cross_validator.determine_final_verdict 의 4가지 판정 분기를
검증한다. 임베딩 모델을 로드하는 함수는 테스트하지 않는다(통합 테스트 영역).
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.cross_validator import (
    determine_final_verdict,
    OFFICIAL_VERIFIED_THRESHOLD,
    RELIABILITY_FINAL_THRESHOLD,
)


class TestDetermineFinalVerdict(unittest.TestCase):
    def test_both_high_verified(self):
        result = determine_final_verdict(0.9, 0.9)
        self.assertEqual(result["verdict"], "오피셜 검증됨")
        self.assertTrue(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "✅")

    def test_official_only(self):
        # 공식성만 충족 (신뢰성 미달)
        result = determine_final_verdict(0.9, RELIABILITY_FINAL_THRESHOLD - 0.1)
        self.assertIn("공식 표현", result["verdict"])
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "⚠️")

    def test_reliability_only(self):
        # 신뢰성만 충족 (공식성 미달)
        result = determine_final_verdict(OFFICIAL_VERIFIED_THRESHOLD - 0.1, 0.9)
        self.assertIn("교차 보도", result["verdict"])
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "⚠️")

    def test_both_low_not_verified(self):
        result = determine_final_verdict(0.1, 0.1)
        self.assertEqual(result["verdict"], "검증 불가")
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "❌")

    def test_boundary_both_at_threshold(self):
        # 경계값은 >= 비교이므로 검증됨이어야 함
        result = determine_final_verdict(
            OFFICIAL_VERIFIED_THRESHOLD, RELIABILITY_FINAL_THRESHOLD
        )
        self.assertTrue(result["is_verified"])


if __name__ == "__main__":
    unittest.main()

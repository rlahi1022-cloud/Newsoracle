"""
tests/test_cross_validator.py
─────────────────────────────────────
services/cross_validator.determine_final_verdict 의 4가지 판정 분기를
검증한다. 임베딩 모델을 로드하는 함수는 테스트하지 않는다(통합 테스트 영역).

[왜 이 함수만 따로 테스트하는가]
  determine_final_verdict 는 파이프라인의 "결론"이다.
  공식성/신뢰성 점수가 아무리 정확해도 이 함수의 분기 조건이 잘못되면
  사용자에게 표시되는 verdict_emoji/verdict 문구가 어긋난다.
  반대로 이 함수가 깔끔하면 임계값을 config 에서 조정해도 분기 안정성 보장.

[테스트 전략]
  - 4가지 분기(both high / official only / reliability only / both low)를
    모두 직접 호출하여 검증.
  - 경계값(>= 임계값) 동작도 별도로 검증 — off-by-one 버그 방지.
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
        # 공식성 ↑ 신뢰성 ↑ → 가장 강한 판정(✅ 오피셜 검증됨)
        # 사용자에게 "이 뉴스는 믿을 만하다"고 보여주는 유일한 케이스
        result = determine_final_verdict(0.9, 0.9)
        self.assertEqual(result["verdict"], "오피셜 검증됨")
        self.assertTrue(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "✅")

    def test_official_only(self):
        # 공식 표현은 강하지만 교차 보도 부족 — 단독 보도 가능성
        # is_verified=False, 경고 이모지(⚠️)로 사용자에게 주의 환기
        result = determine_final_verdict(0.9, RELIABILITY_FINAL_THRESHOLD - 0.1)
        self.assertIn("공식 표현", result["verdict"])
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "⚠️")

    def test_reliability_only(self):
        # 여러 언론이 보도했지만 공식 표현이 약함 — 추측성/비공식 보도 가능성
        result = determine_final_verdict(OFFICIAL_VERIFIED_THRESHOLD - 0.1, 0.9)
        self.assertIn("교차 보도", result["verdict"])
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "⚠️")

    def test_both_low_not_verified(self):
        # 둘 다 미달 → 검증 불가(❌) — 신뢰할 근거 없음
        result = determine_final_verdict(0.1, 0.1)
        self.assertEqual(result["verdict"], "검증 불가")
        self.assertFalse(result["is_verified"])
        self.assertEqual(result["verdict_emoji"], "❌")

    def test_boundary_both_at_threshold(self):
        # 임계값과 정확히 같을 때 — 코드가 > 비교를 쓰면 여기서 깨진다.
        # config 의 임계값을 미세 조정해도 이 케이스가 회귀를 잡아준다.
        result = determine_final_verdict(
            OFFICIAL_VERIFIED_THRESHOLD, RELIABILITY_FINAL_THRESHOLD
        )
        self.assertTrue(result["is_verified"])


if __name__ == "__main__":
    unittest.main()

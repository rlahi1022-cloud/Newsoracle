"""
tests/test_helpers.py
─────────────────────────────────────
utils/helpers.py 의 ensure_dir, save_json, load_json, get_timestamp 검증.
임시 디렉토리에서 동작하므로 프로젝트 파일을 건드리지 않는다.

[왜 이 모듈을 따로 테스트하는가]
  helpers 는 결과 저장/로깅/디렉토리 관리에 두루 쓰여서 작은 회귀가
  파이프라인 전체에 파급된다. 특히 save_json 의 ensure_ascii=False 동작이
  깨지면 결과 파일이 \\uXXXX 형태로 깨져 가독성이 사라진다.

[테스트 환경]
  파일 시스템에 부수효과가 있으므로 모든 케이스를 tempfile.TemporaryDirectory()
  안에서 실행한다. 테스트가 끝나면 자동으로 정리된다.
"""

import os
import re
import sys
import json
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.helpers import ensure_dir, save_json, load_json, get_timestamp


class TestEnsureDir(unittest.TestCase):
    # 결과 저장/로그/모델 캐시 등에서 두루 쓰이므로
    # "중첩 경로 자동 생성"과 "이미 존재해도 에러 없음"을 보장해야 한다.

    def test_creates_nested_directory(self):
        # mkdir -p 같이 다단계 경로 한 번에 만들 수 있어야 함
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "a", "b", "c")
            ensure_dir(target)
            self.assertTrue(os.path.isdir(target))

    def test_idempotent(self):
        # 같은 경로로 두 번 호출해도 FileExistsError 없이 통과해야 함
        # 매 요청마다 결과 디렉토리를 ensure 하는 코드 패턴이 흔하기 때문
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "exists")
            ensure_dir(target)
            ensure_dir(target)
            self.assertTrue(os.path.isdir(target))


class TestSaveAndLoadJson(unittest.TestCase):
    # 검색 결과/캐시/카테고리 라벨 로딩 등에 광범위하게 쓰임.
    # 저장-로드 라운드트립이 깨지면 사이트 UI 부터 학습 데이터까지 줄줄이 영향.

    def test_roundtrip_dict(self):
        # 가장 흔한 형태(dict)가 정확히 복원되는지 + 중간 디렉토리 자동 생성도
        data = {"query": "기준금리", "count": 12, "verified": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "out.json")
            save_json(data, path)
            self.assertTrue(os.path.exists(path))
            loaded = load_json(path)
            self.assertEqual(loaded, data)

    def test_roundtrip_list(self):
        # 검색 결과(list[dict])도 그대로 복원되어야 함
        data = [{"title": "기사1"}, {"title": "기사2"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            save_json(data, path)
            loaded = load_json(path)
            self.assertEqual(loaded, data)

    def test_unicode_preserved(self):
        # ensure_ascii=False 가 깨지면 결과 JSON 이 \\uXXXX 로 도배되어
        # 사람이 읽을 수 없게 된다. 한글이 그대로 들어가는지 디스크에서 확인.
        data = {"한글키": "한글 값 ✅"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u.json")
            save_json(data, path)
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            self.assertIn("한글", raw)

    def test_load_missing_file_raises(self):
        # 파일 없으면 명시적으로 FileNotFoundError —
        # 빈 dict 폴백을 쓰면 호출자가 실패를 못 알아챔
        with self.assertRaises(FileNotFoundError):
            load_json("/no/such/path/file.json")


class TestGetTimestamp(unittest.TestCase):
    # 결과 파일명에 들어가므로 포맷이 깨지면 파일 정렬/구분이 흐트러진다.

    def test_format(self):
        # "YYYY-MM-DD_HHMMSS" — result_writer/run_tests 모두 이 포맷에 맞춰서 파싱
        ts = get_timestamp()
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}_\d{6}$")


if __name__ == "__main__":
    unittest.main()

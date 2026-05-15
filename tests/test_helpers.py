"""
tests/test_helpers.py
─────────────────────────────────────
utils/helpers.py 의 ensure_dir, save_json, load_json, get_timestamp 검증.
임시 디렉토리에서 동작하므로 프로젝트 파일을 건드리지 않는다.
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
    def test_creates_nested_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "a", "b", "c")
            ensure_dir(target)
            self.assertTrue(os.path.isdir(target))

    def test_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "exists")
            ensure_dir(target)
            ensure_dir(target)  # 두 번 호출해도 에러 없어야 함
            self.assertTrue(os.path.isdir(target))


class TestSaveAndLoadJson(unittest.TestCase):
    def test_roundtrip_dict(self):
        data = {"query": "기준금리", "count": 12, "verified": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "out.json")
            save_json(data, path)
            self.assertTrue(os.path.exists(path))
            loaded = load_json(path)
            self.assertEqual(loaded, data)

    def test_roundtrip_list(self):
        data = [{"title": "기사1"}, {"title": "기사2"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            save_json(data, path)
            loaded = load_json(path)
            self.assertEqual(loaded, data)

    def test_unicode_preserved(self):
        data = {"한글키": "한글 값 ✅"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u.json")
            save_json(data, path)
            with open(path, encoding="utf-8") as f:
                raw = f.read()
            # ensure_ascii=False 로 저장되므로 한글이 그대로 들어가야 함
            self.assertIn("한글", raw)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_json("/no/such/path/file.json")


class TestGetTimestamp(unittest.TestCase):
    def test_format(self):
        ts = get_timestamp()
        # "YYYY-MM-DD_HHMMSS" 형식
        self.assertRegex(ts, r"^\d{4}-\d{2}-\d{2}_\d{6}$")


if __name__ == "__main__":
    unittest.main()

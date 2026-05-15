"""
tests/test_server_api.py
─────────────────────────────────────
FastAPI 서버 (server.py) 의 HTTP 엔드포인트 통합 테스트.

[전제]
  서버가 NEWSORACLE_BASE_URL (기본: http://localhost:8000) 에서 떠 있어야 한다.
  서버 미기동 시 setUpClass 에서 전체 SKIP 처리한다.

[빠른 테스트 vs 느린 테스트]
  - 빠른 테스트: job_id 발급, 즉시 상태 조회, 입력 검증 등 외부 API 호출 없음.
  - 느린 테스트: /api/search 가 실제 네이버 API + 모델 추론을 마칠 때까지 폴링.
    네이버 API 키와 시간(수십 초)이 필요하므로 RUN_SLOW_TESTS=1 일 때만 실행한다.

[실행]
  # 터미널 1
  python3 server.py

  # 터미널 2
  python3 tests/run_tests.py                   # 빠른 통합 테스트만
  RUN_SLOW_TESTS=1 python3 tests/run_tests.py  # 느린 폴링 테스트까지
"""

import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
BASE_URL = os.environ.get("NEWSORACLE_BASE_URL", "http://localhost:8000").rstrip("/")
RUN_SLOW = os.environ.get("RUN_SLOW_TESTS", "").lower() in ("1", "true", "yes")

# 헬스체크 / 일반 요청 타임아웃 (초)
HEALTH_TIMEOUT = 2.0
REQ_TIMEOUT = 10.0

# 느린 테스트 폴링 설정
POLL_INTERVAL = 2.0
POLL_MAX_WAIT = 60.0


def _server_alive() -> tuple[bool, str]:
    """
    서버가 떠 있는지 GET / 로 헬스체크.
    Returns: (alive, reason)
    """
    try:
        r = requests.get(BASE_URL + "/", timeout=HEALTH_TIMEOUT)
        if r.status_code < 500:
            return True, ""
        return False, f"GET / 상태코드 {r.status_code}"
    except requests.RequestException as exc:
        return False, f"연결 실패: {exc.__class__.__name__}"


# ─────────────────────────────────────────────────────────────
# 베이스 클래스: 서버가 안 떠 있으면 전체 SKIP
# ─────────────────────────────────────────────────────────────
class ServerIntegrationBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alive, reason = _server_alive()
        if not alive:
            raise unittest.SkipTest(
                f"서버 미기동 ({BASE_URL}) — {reason}. "
                f"`python3 server.py` 로 먼저 띄우세요."
            )


# ─────────────────────────────────────────────────────────────
# 빠른 테스트 (외부 API 호출 없음)
# ─────────────────────────────────────────────────────────────
class TestIndexAndDocs(ServerIntegrationBase):
    def test_root_returns_200(self):
        r = requests.get(BASE_URL + "/", timeout=REQ_TIMEOUT)
        self.assertEqual(r.status_code, 200)

    def test_openapi_docs_available(self):
        # FastAPI 가 자동 제공하는 OpenAPI 스펙 — 서버 헬스 검증용
        r = requests.get(BASE_URL + "/openapi.json", timeout=REQ_TIMEOUT)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("paths", data)
        # 핵심 엔드포인트가 스펙에 포함되어 있는지 확인
        self.assertIn("/api/search", data["paths"])
        self.assertIn("/api/suggest", data["paths"])


class TestApiSuggest(ServerIntegrationBase):
    def test_empty_query_returns_error_status(self):
        r = requests.post(
            BASE_URL + "/api/suggest",
            json={"query": ""},
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("status"), "error")
        self.assertEqual(body.get("suggestions"), [])

    def test_valid_query_returns_suggestions(self):
        # 의도 분류는 로컬 임베딩만 사용 — 외부 API 호출 없음
        r = requests.post(
            BASE_URL + "/api/suggest",
            json={"query": "삼성전자"},
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("status"), "ok")
        self.assertEqual(body.get("query"), "삼성전자")
        self.assertIsInstance(body.get("suggestions"), list)
        # 각 suggestion 항목 구조 확인
        if body["suggestions"]:
            first = body["suggestions"][0]
            for key in ("category", "label", "score"):
                self.assertIn(key, first)

    def test_long_query_sets_skip_selection(self):
        # 토큰 3개 이상이면 skip_selection=True
        r = requests.post(
            BASE_URL + "/api/suggest",
            json={"query": "삼성전자 갤럭시 신제품 공식 발표"},
            timeout=REQ_TIMEOUT,
        )
        body = r.json()
        self.assertTrue(body.get("skip_selection"))

    def test_invalid_body_returns_422(self):
        # query 필드 누락 → FastAPI 가 422 반환
        r = requests.post(
            BASE_URL + "/api/suggest",
            json={"not_query": "x"},
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 422)


class TestApiSearchKickoff(ServerIntegrationBase):
    """
    /api/search 즉시 응답 검증 — job_id 발급과 직후 상태만 본다.
    실제 파이프라인 완료까지 기다리지 않는다.
    """

    def test_empty_query_returns_error(self):
        r = requests.post(
            BASE_URL + "/api/search",
            json={"query": "   "},
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("status"), "error")
        self.assertEqual(body.get("job_id"), "")

    def test_valid_query_returns_job_id(self):
        r = requests.post(
            BASE_URL + "/api/search",
            json={"query": "기준금리"},
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("status"), "processing")
        job_id = body.get("job_id")
        self.assertTrue(job_id)
        self.assertEqual(len(job_id), 8)  # server.py 에서 uuid4()[:8]

    def test_invalid_body_returns_422(self):
        r = requests.post(
            BASE_URL + "/api/search",
            json={},  # query 필수
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 422)


class TestApiResult(ServerIntegrationBase):
    def test_unknown_job_id_returns_error(self):
        r = requests.get(
            BASE_URL + "/api/result/this-id-does-not-exist",
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body.get("status"), "error")
        self.assertIn("존재하지 않는", body.get("message", ""))

    def test_just_created_job_is_processing(self):
        # job 만들고 → 곧바로 조회 → processing 상태여야 함
        kick = requests.post(
            BASE_URL + "/api/search",
            json={"query": "테스트쿼리"},
            timeout=REQ_TIMEOUT,
        ).json()
        job_id = kick.get("job_id")
        self.assertTrue(job_id)

        r = requests.get(
            BASE_URL + f"/api/result/{job_id}",
            timeout=REQ_TIMEOUT,
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        # 매우 빨리 처리되어 done 이 됐을 가능성도 있으므로 둘 다 허용
        self.assertIn(body.get("status"), ("processing", "done", "error"))
        self.assertEqual(body.get("query", ""), "테스트쿼리")


# ─────────────────────────────────────────────────────────────
# 느린 테스트 (실제 파이프라인 완료까지 폴링)
#   - 네이버 API 키, 모델 로드, 수십 초 대기 필요
#   - RUN_SLOW_TESTS=1 일 때만 활성화
# ─────────────────────────────────────────────────────────────
@unittest.skipUnless(
    RUN_SLOW,
    "RUN_SLOW_TESTS=1 환경변수가 있을 때만 실행 (실제 네이버 API 호출 + 수십 초 대기)",
)
class TestApiSearchEndToEnd(ServerIntegrationBase):
    def test_pipeline_completes_with_results(self):
        # 1) 검색 요청
        kick = requests.post(
            BASE_URL + "/api/search",
            json={"query": "기준금리"},
            timeout=REQ_TIMEOUT,
        ).json()
        job_id = kick["job_id"]
        self.assertTrue(job_id, "job_id 발급 실패")

        # 2) done 까지 폴링
        deadline = time.time() + POLL_MAX_WAIT
        final = None
        while time.time() < deadline:
            r = requests.get(
                BASE_URL + f"/api/result/{job_id}",
                timeout=REQ_TIMEOUT,
            ).json()
            if r.get("status") in ("done", "error"):
                final = r
                break
            time.sleep(POLL_INTERVAL)

        self.assertIsNotNone(final, f"{POLL_MAX_WAIT}초 안에 파이프라인 미완료")
        self.assertEqual(final.get("status"), "done",
                         f"파이프라인 에러: {final.get('message')}")
        # data 는 리스트, total 은 정수
        self.assertIsInstance(final.get("data"), list)
        self.assertIsInstance(final.get("total"), int)


if __name__ == "__main__":
    unittest.main()

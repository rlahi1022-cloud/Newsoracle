"""
tests/run_tests.py
─────────────────────────────────────
tests/ 디렉토리 아래의 test_*.py 를 모두 발견(discover)하여 실행하고,
결과를 표 형식으로 다음 두 파일에 저장한다.

  tests/results/test_results_<timestamp>.csv
  tests/results/test_results_<timestamp>.md

추가로 콘솔에는 ASCII 표를 함께 출력한다.

표 컬럼:
  module       — 테스트가 정의된 파이썬 모듈
  test_class   — TestCase 클래스명
  test_name    — 개별 테스트 메서드명
  status       — PASS / FAIL / ERROR / SKIP
  duration_ms  — 실행 시간(밀리초)
  message      — 실패/에러 시 첫 줄, 통과 시 공란

실행:
  python3 tests/run_tests.py
  python3 -m tests.run_tests
"""

import os
import sys
import csv
import time
import unittest
from datetime import datetime
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TESTS_DIR)
RESULTS_DIR = os.path.join(TESTS_DIR, "results")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 결과 수집용 커스텀 TestResult
#
# 왜 직접 만드는가:
#   기본 TextTestRunner는 통과/실패 카운트만 주고
#   각 테스트 단위의 (이름, 상태, 시간, 메시지)를 표 형태로 꺼내기 어렵다.
#   startTest/stopTest 후크에서 시작 시각을 기록해 duration을 계산한다.
# ─────────────────────────────────────────────────────────────
class TableTestResult(unittest.TestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: List[dict] = []
        self._current_start: float = 0.0

    def startTest(self, test):
        super().startTest(test)
        self._current_start = time.perf_counter()

    def _push(self, test, status: str, message: str = ""):
        duration_ms = (time.perf_counter() - self._current_start) * 1000.0
        cls = test.__class__
        # setUpClass / setUpModule 단계에서 발생한 예외는 _ErrorHolder 로 들어옴.
        # 이 객체는 _testMethodName 이 없고 description 만 있어서 분기 처리한다.
        test_name = getattr(test, "_testMethodName", None)
        if test_name is None:
            # 예: "setUpClass (tests.test_server_api.TestApiSearchKickoff)"
            test_name = getattr(test, "description", str(test))
        module_name = getattr(cls, "__module__", "")
        class_name = getattr(cls, "__name__", "")
        self.records.append({
            "module": module_name,
            "test_class": class_name,
            "test_name": test_name,
            "status": status,
            "duration_ms": round(duration_ms, 2),
            "message": message.splitlines()[0][:200] if message else "",
        })

    def addSuccess(self, test):
        super().addSuccess(test)
        self._push(test, "PASS")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._push(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._push(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._push(test, "SKIP", reason)


# ─────────────────────────────────────────────────────────────
# 출력 유틸
# ─────────────────────────────────────────────────────────────
def _display_width(text: str) -> int:
    """
    동아시아 폭(East Asian Width) 기반의 표시 폭을 계산한다.
    한글/한자 등 와이드 문자는 2칸, 그 외는 1칸으로 센다.
    표 정렬 시 한글 컬럼이 어긋나지 않도록 한다.
    """
    import unicodedata
    width = 0
    for ch in str(text):
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            width += 2
        else:
            width += 1
    return width


def _pad(text: str, width: int) -> str:
    """표시 폭 기준으로 오른쪽에 공백을 채운다."""
    return str(text) + " " * max(0, width - _display_width(text))


def print_ascii_table(records: List[dict]) -> None:
    """콘솔에 ASCII 표를 출력한다."""
    if not records:
        print("(테스트 결과가 없습니다)")
        return

    columns = ["module", "test_class", "test_name", "status", "duration_ms", "message"]
    headers = {c: c for c in columns}

    # 컬럼별 최대 폭 계산
    widths = {c: _display_width(headers[c]) for c in columns}
    for r in records:
        for c in columns:
            widths[c] = max(widths[c], _display_width(r.get(c, "")))

    # message는 표 폭 폭주를 막기 위해 60칸으로 제한
    widths["message"] = min(widths["message"], 60)

    sep = "+" + "+".join("-" * (widths[c] + 2) for c in columns) + "+"

    print(sep)
    print("| " + " | ".join(_pad(headers[c], widths[c]) for c in columns) + " |")
    print(sep)
    for r in records:
        row_vals = []
        for c in columns:
            val = str(r.get(c, ""))
            if c == "message" and _display_width(val) > widths[c]:
                # 폭 초과 시 잘라내고 말줄임표 표시
                while _display_width(val + "…") > widths[c] and val:
                    val = val[:-1]
                val = val + "…"
            row_vals.append(_pad(val, widths[c]))
        print("| " + " | ".join(row_vals) + " |")
    print(sep)


def save_csv(records: List[dict], path: str) -> None:
    """결과를 CSV 표로 저장한다 (Excel 호환을 위해 utf-8-sig 사용)."""
    fields = ["module", "test_class", "test_name", "status", "duration_ms", "message"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})


def save_markdown(records: List[dict], path: str, summary: dict) -> None:
    """결과를 마크다운 표로 저장한다."""
    fields = ["module", "test_class", "test_name", "status", "duration_ms", "message"]
    status_emoji = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "⏭️"}

    lines = []
    lines.append(f"# Newsoracle 테스트 결과")
    lines.append("")
    lines.append(f"- 실행 시각: `{summary['timestamp']}`")
    lines.append(f"- 전체: **{summary['total']}건** "
                 f"(PASS {summary['pass']} / FAIL {summary['fail']} / "
                 f"ERROR {summary['error']} / SKIP {summary['skip']})")
    lines.append(f"- 성공률: **{summary['pass_rate']:.1f}%**")
    lines.append(f"- 총 소요 시간: {summary['elapsed_sec']:.3f}초")
    lines.append("")
    lines.append("## 상세 결과")
    lines.append("")
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for r in records:
        status = r["status"]
        emoji = status_emoji.get(status, "")
        # 파이프 문자가 셀 안에 들어가면 표가 깨지므로 이스케이프
        msg = (r.get("message") or "").replace("|", "\\|")
        row = [
            r.get("module", ""),
            r.get("test_class", ""),
            r.get("test_name", ""),
            f"{emoji} {status}",
            f"{r.get('duration_ms', 0):.2f}",
            msg,
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────────────────────────
def main() -> int:
    print("─" * 72)
    print("  Newsoracle 테스트 실행")
    print("─" * 72)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=TESTS_DIR, pattern="test_*.py", top_level_dir=PROJECT_ROOT)

    result = TableTestResult()
    started = time.perf_counter()
    suite.run(result)
    elapsed = time.perf_counter() - started

    records = result.records

    # ── 요약 통계 ───────────────────────────────────────────
    total = len(records)
    n_pass = sum(1 for r in records if r["status"] == "PASS")
    n_fail = sum(1 for r in records if r["status"] == "FAIL")
    n_error = sum(1 for r in records if r["status"] == "ERROR")
    n_skip = sum(1 for r in records if r["status"] == "SKIP")
    pass_rate = (n_pass / total * 100.0) if total else 0.0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "total": total,
        "pass": n_pass,
        "fail": n_fail,
        "error": n_error,
        "skip": n_skip,
        "pass_rate": pass_rate,
        "elapsed_sec": elapsed,
    }

    # ── 콘솔 출력 ───────────────────────────────────────────
    print_ascii_table(records)
    print()
    print(f"  총 {total}건 | "
          f"✅ PASS {n_pass} | ❌ FAIL {n_fail} | "
          f"💥 ERROR {n_error} | ⏭️ SKIP {n_skip}")
    print(f"  성공률 {pass_rate:.1f}%  |  총 소요 {elapsed:.3f}초")

    # ── 파일 저장 ───────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, f"test_results_{timestamp}.csv")
    md_path = os.path.join(RESULTS_DIR, f"test_results_{timestamp}.md")
    save_csv(records, csv_path)
    save_markdown(records, md_path, summary)

    print()
    print(f"  📄 CSV  : {csv_path}")
    print(f"  📄 MD   : {md_path}")
    print("─" * 72)

    # 종료 코드: 실패/에러가 있으면 1
    return 0 if (n_fail == 0 and n_error == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

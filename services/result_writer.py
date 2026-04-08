"""
services/result_writer.py
──────────────────────────
최종 앙상블 결과를 JSON 또는 CSV 파일로 저장한다.
기관 검증 메시지 포함한 터미널 출력 포함.
"""

import os
import json
import csv
from datetime import datetime
from config import RESULT_OUTPUT_DIR
from logger import get_logger

logger = get_logger("result_writer")

os.makedirs(RESULT_OUTPUT_DIR, exist_ok=True)


def _generate_filename(query: str, ext: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:20]
    filename = f"{safe_query}_{timestamp}.{ext}"
    return os.path.join(RESULT_OUTPUT_DIR, filename)


def save_as_json(results: list, query: str) -> str:
    filepath = _generate_filename(query, "json")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON 저장 완료: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        raise


def save_as_csv(results: list, query: str) -> str:
    if not results:
        logger.warning("저장할 결과 데이터가 없음")
        return ""

    filepath = _generate_filename(query, "csv")
    fieldnames = [
        "title", "source", "originallink", "pubDate",
        "rule_score", "semantic_score", "classifier_score",
        "agency_score", "org_name", "report_count",
        "is_official_domain", "verification_message",
        "final_official_score", "predicted_label",
    ]
    try:
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        logger.info(f"CSV 저장 완료: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"CSV 저장 실패: {e}")
        raise


def print_results_summary(results: list):
    """
    터미널에 결과 요약을 출력한다.
    기관 검증 메시지를 포함하여 출력.
    """
    if not results:
        print("결과 없음")
        return

    print("\n" + "=" * 70)
    print(f"{'공식성 판별 결과':^70}")
    print("=" * 70)

    sorted_results = sorted(
        results,
        key=lambda x: x.get("final_official_score", 0),
        reverse=True
    )

    for i, r in enumerate(sorted_results[:10], 1):
        print(f"\n[{i}] {r.get('title', '')[:50]}")
        print(f"    출처:        {r.get('source', '-')}")
        print(f"    규칙 점수:   {r.get('rule_score', 0):.4f}")
        print(f"    의미 유사도: {r.get('semantic_score', 0):.4f}")
        print(f"    분류 점수:   {r.get('classifier_score', 0):.4f}")
        print(f"    기관 점수:   {r.get('agency_score', 0):.4f}")
        print(f"    최종 점수:   {r.get('final_official_score', 0):.4f}")
        print(f"    판별 결과:   {r.get('predicted_label', '-')}")
        if r.get("verification_message"):
            print(f"    검증 메시지: {r.get('verification_message')}")

    print("\n" + "=" * 70)
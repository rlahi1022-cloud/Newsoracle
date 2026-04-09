"""
services/result_writer.py
──────────────────────────
최종 앙상블 결과를 JSON 또는 CSV 파일로 저장한다.

[v3 — 설명 가능한 AI]
  터미널 출력에 explanation 필드를 포함하여
  각 기사가 왜 공식/비공식 판정을 받았는지 사람이 읽을 수 있게 출력한다.

  출력 형식:
  ══════════════════════════════════════════
    [1] ✅ SM엔터, 아이돌 그룹 컴백 공식 발표
        출처: starnewskorea.com
        공식성: 0.7821 | 신뢰성: 0.6200
        판정: 오피셜 검증됨
        ────────────────────────────────────
        📋 판단 근거:
        ✅ 공식성 높음 (0.78) |
        [규칙] 공식 표현 3건: 공식 입장, 밝혔다, [공식] |
        기관명 2건: SM엔터테인먼트 |
        비공식 표현 없음 |
        [교차보도] 3개 독립 언론사 교차 보도 확인
  ══════════════════════════════════════════
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
    """쿼리와 타임스탬프 기반 파일명을 생성한다."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    safe_query = query.replace(" ", "_")[:20]
    filename = f"{safe_query}_{timestamp}.{ext}"
    return os.path.join(RESULT_OUTPUT_DIR, filename)


def save_as_json(results: list, query: str) -> str:
    """결과를 JSON 파일로 저장한다."""
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
    """결과를 CSV 파일로 저장한다. v3: explanation 컬럼 추가."""
    if not results:
        logger.warning("저장할 결과 데이터가 없음")
        return ""

    filepath = _generate_filename(query, "csv")
    fieldnames = [
        "title", "source", "originallink", "pubDate", "domain",
        "rule_score", "rule_reason",
        "semantic_score", "classifier_score",
        "agency_score", "verification_message",
        "official_score", "score_method",
        "reliability_score", "reliability_reason",
        "verdict", "verdict_emoji", "is_verified",
        "explanation",
        "final_official_score", "predicted_label",
    ]
    try:
        with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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

    v3: explanation 필드를 포함하여
    각 기사가 왜 공식/비공식인지 판단 근거를 함께 출력한다.
    """
    if not results:
        print("\n  결과 없음")
        return

    print("\n" + "═" * 72)
    print(f"{'공식성 판별 결과':^72}")
    print("═" * 72)

    # 공식성 점수 높은 순으로 정렬
    sorted_results = sorted(
        results,
        key=lambda x: x.get("official_score", 0),
        reverse=True
    )

    for i, r in enumerate(sorted_results[:15], 1):
        verdict_emoji = r.get("verdict_emoji", "")
        title = r.get("title", "")[:50]
        source = r.get("source", "-")
        official_score = r.get("official_score", 0)
        reliability_score = r.get("reliability_score", 0)
        verdict = r.get("verdict", "-")
        explanation = r.get("explanation", "")

        print(f"\n  [{i}] {verdict_emoji} {title}")
        print(f"      출처: {source}")
        print(f"      공식성: {official_score:.4f} | 신뢰성: {reliability_score:.4f}")
        print(f"      판정: {verdict}")

        # 세부 점수 (선택적 표시)
        rule_score = r.get("rule_score", 0)
        semantic_score = r.get("semantic_score", 0)
        classifier_score = r.get("classifier_score", 0)
        agency_score = r.get("agency_score", 0)
        score_method = r.get("score_method", "")

        print(f"      [세부] rule={rule_score:.2f} semantic={semantic_score:.2f} "
              f"classifier={classifier_score:.2f} agency={agency_score:.2f} "
              f"({score_method})")

        # 설명 메시지 (핵심)
        if explanation:
            print(f"      {'─' * 50}")
            print(f"      📋 판단 근거:")
            # explanation이 길 수 있으므로 | 기준으로 줄바꿈
            parts = explanation.split(" | ")
            for part in parts:
                print(f"         {part}")

    # 전체 요약
    total = len(results)
    verified = sum(1 for r in results if r.get("is_verified", False))
    avg_official = sum(r.get("official_score", 0) for r in results) / max(total, 1)

    print("\n" + "─" * 72)
    print(f"  총 {total}건 | 오피셜 검증: {verified}건 | 평균 공식성: {avg_official:.4f}")
    print("═" * 72)


def save_results(results: list, query: str, output_format: str = "csv") -> str:
    """
    출력 형식에 따라 결과를 저장하는 통합 래퍼 함수.

    Args:
        results:       앙상블 최종 결과 리스트
        query:         검색 키워드
        output_format: "csv" | "json" | "both"
    Returns:
        저장된 파일 경로 (both일 경우 csv 경로 반환)
    """
    if output_format == "json":
        return save_as_json(results, query)
    elif output_format == "both":
        save_as_json(results, query)
        return save_as_csv(results, query)
    else:
        # 기본값: csv
        return save_as_csv(results, query)
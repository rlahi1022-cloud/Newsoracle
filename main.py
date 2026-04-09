"""
main.py
────────────────────────────────────────────────────────────────────────────────
프로젝트 진입점. CLI 인자를 받아 train / infer 모드를 실행한다.

실행 예시:
  python main.py --mode infer --query "기준금리"
  python main.py --mode infer --query "국토교통부" --output json
  python main.py --mode train --train-path data/train_data.csv

[v2 변경사항]
  추론 파이프라인에 cross_validator 추가
  공식성 + 신뢰성 둘 다 판정하는 최종 출력 구조로 변경

[v3 변경사항]
  search_news → search_news_combined로 교체
  sim + date 이중 수집 → 중복 제거 → 키워드 필터링 적용

[v4 변경사항]
  run_infer 내 sys.exit(0)을 return으로 교체
  이유: 인터랙티브 모드에서 검색 결과 없음/오타 시 프로그램이 종료되는 문제
  해결: return으로 바꿔서 "결과 없음" 안내 후 다시 입력 대기로 복귀
  참고: sys.exit(1)은 치명적 에러(모듈 로드 실패 등)이므로 유지
"""

import argparse
import sys
from logger import get_logger

logger = get_logger("main")


# ────────────────────────────────────────────────────────────────────────────────
# 추론 모드
# ────────────────────────────────────────────────────────────────────────────────

def run_infer(query: str, output_format: str = "csv") -> None:
    """
    추론 모드: 키워드로 뉴스를 수집하고 공식성 + 신뢰성을 판별하여 출력한다.

    파이프라인 순서:
      1. 뉴스 수집        (news_search — sim+date 이중 수집)
      2. 본문 크롤링      (article_crawler — description→실제 본문 교체)
      3. 전처리           (preprocessor)
      4. 특징 추출        (feature_extractor)
      5. 교차 보도 검증   (cross_validator)
      6. 규칙 기반 점수   (rule_based_scorer)
      7. 의미 유사도      (semantic_similarity)
      8. 분류 모델 추론   (classifier_model)
      9. 기관 신뢰도 검증 (agency_verifier)
      10. 앙상블          (ensemble)
      11. 결과 출력/저장  (result_writer + 터미널 출력)

    Args:
        query (str): 검색 키워드
        output_format (str): 결과 저장 형식 ("csv" | "json" | "both")
    """
    logger.info(f"=== 추론 모드 시작 | query={query} ===")

    # ── 1. 뉴스 수집 (sim + date 이중 수집) ─────────────────────────────────
    logger.info("[1/10] 뉴스 수집 시작 (sim + date 이중 수집)")
    try:
        from news_search import search_news_combined
        articles = search_news_combined(query)
    except ImportError:
        # search_news_combined가 없으면 기존 search_news로 폴백
        try:
            from news_search import search_news
            articles = search_news(query)
            logger.warning("search_news_combined 없음 → search_news fallback 사용")
        except Exception as exc:
            logger.error(f"뉴스 수집 모듈 로드 실패: {exc}", exc_info=True)
            sys.exit(1)
    except Exception as exc:
        logger.error(f"뉴스 수집 실패: {exc}", exc_info=True)
        # 수집 실패는 네트워크/API 문제일 수 있으므로 종료하지 않고 복귀
        print(f"\n  ⚠️ 뉴스 수집에 실패했습니다: {exc}")
        print("  다른 검색어로 다시 시도해 주세요.\n")
        return

    if not articles:
        # 수집 결과 0건 → 종료하지 않고 안내 후 복귀
        logger.warning("수집된 기사가 없음.")
        print(f"\n  ⚠️ '{query}'에 대한 검색 결과가 없습니다.")
        print("  검색어를 확인하고 다시 시도해 주세요.\n")
        return

    logger.info(f"[1/10] 뉴스 수집 완료 | {len(articles)}건")

    # ── 2. 본문 크롤링 (description → 실제 본문 교체) ────────────────────────
    # 왜 필요한가:
    #   네이버 API의 description은 2~3줄 요약(80~150자)에 불과하다.
    #   본문 전체를 가져오면 공식 표현("밝혔다", "보도자료를 통해") 탐지가
    #   가능해지고, semantic similarity와 classifier 정확도가 향상된다.
    # 크롤링 실패 시:
    #   기존 description을 유지하므로 파이프라인이 중단되지 않는다.
    logger.info(f"[2/10] 본문 크롤링 시작 | {len(articles)}건")
    try:
        from services.article_crawler import crawl_articles_batch
        articles = crawl_articles_batch(articles)
    except ImportError:
        logger.warning("article_crawler 모듈 없음 → 크롤링 건너뜀 (API description 사용)")
    except Exception as exc:
        logger.warning(f"본문 크롤링 실패 → API description으로 진행: {exc}")

    logger.info(f"[2/10] 본문 크롤링 완료")

    # ── 3. 전처리 ─────────────────────────────────────────────────────────────
    logger.info("[3/10] 전처리 시작")
    try:
        from services.preprocessor import preprocess_articles
        cleaned_articles = preprocess_articles(articles)
    except Exception as exc:
        logger.error(f"전처리 실패: {exc}", exc_info=True)
        print(f"\n  ⚠️ 전처리 중 오류가 발생했습니다: {exc}\n")
        return

    if not cleaned_articles:
        logger.warning("전처리 후 유효한 기사가 없음.")
        print(f"\n  ⚠️ '{query}' 검색 결과에서 유효한 기사를 찾지 못했습니다.\n")
        return

    logger.info(f"[3/10] 전처리 완료 | {len(cleaned_articles)}건")

    # ── 4. 교차 보도 검증 + 중복 필터링 ──────────────────────────────────────
    # 왜 특징 추출보다 먼저 하는가:
    #   교차 검증에서 클러스터 기반 중복 필터링이 수행되어 기사 수가 줄어든다.
    #   (예: 134건 → 55건)
    #   이후 특징 추출, rule, semantic, classifier를 필터링된 기사 기준으로
    #   수행해야 기사 수 불일치가 발생하지 않는다.
    logger.info("[4/10] 교차 보도 검증 시작 (기사 대 기사 클러스터링 + 중복 필터링)")
    try:
        from services.cross_validator import cross_validate_articles
        validated_articles = cross_validate_articles(cleaned_articles)
    except Exception as exc:
        logger.warning(f"교차 보도 검증 실패 → 신뢰성 점수 0으로 진행: {exc}")
        # 실패해도 파이프라인 계속 진행 (reliability_score=0.0으로 기본값)
        validated_articles = []
        for article in cleaned_articles:
            enriched = {**article}
            enriched["reliability_score"] = 0.0
            enriched["reliability_reason"] = "교차 검증 실패"
            enriched["cluster_id"] = 0
            enriched["cluster_size"] = 1
            enriched["unique_sources"] = 1
            enriched["has_official_domain"] = False
            validated_articles.append(enriched)

    logger.info(f"[4/10] 교차 보도 검증 완료 | {len(validated_articles)}건")

    # ── 5. 특징 추출 (필터링된 기사 기준) ─────────────────────────────────────
    # 중복 필터링 후 남은 기사에 대해서만 특징을 추출한다.
    # 이렇게 해야 이후 rule/semantic/classifier 입력 건수와 일치한다.
    logger.info("[5/10] 특징 추출 시작")
    try:
        from services.feature_extractor import extract_features_batch
        features_list = extract_features_batch(validated_articles)
    except Exception as exc:
        logger.error(f"특징 추출 실패: {exc}", exc_info=True)
        features_list = [{} for _ in validated_articles]

    logger.info(f"[5/10] 특징 추출 완료 | {len(features_list)}건")

    # ── 6. 규칙 기반 점수 ─────────────────────────────────────────────────────
    logger.info("[6/10] 규칙 기반 점수 계산 시작")
    try:
        from services.rule_based_scorer import compute_rule_scores_batch
        rule_scores = compute_rule_scores_batch(features_list)
    except Exception as exc:
        logger.warning(f"규칙 점수 계산 실패 → 0.0으로 대체: {exc}")
        rule_scores = [0.0] * len(validated_articles)

    logger.info("[6/10] 규칙 점수 계산 완료")

    # ── 7. 의미 유사도 점수 ───────────────────────────────────────────────────
    logger.info("[7/10] 의미 유사도 계산 시작")
    try:
        from services.semantic_similarity import compute_semantic_scores_batch
        embedding_texts = [f.get("embedding_input", "") for f in features_list]
        semantic_scores = compute_semantic_scores_batch(embedding_texts)
    except Exception as exc:
        logger.warning(f"의미 유사도 계산 실패 → 0.0으로 대체: {exc}")
        semantic_scores = [0.0] * len(validated_articles)

    logger.info("[7/10] 의미 유사도 계산 완료")

    # ── 8. 분류 모델 추론 ─────────────────────────────────────────────────────
    logger.info("[8/10] 분류 모델 추론 시작")
    try:
        from services.classifier_model import predict_batch
        classifier_texts = [f.get("classifier_input", "") for f in features_list]
        classifier_results = predict_batch(classifier_texts)
    except Exception as exc:
        logger.warning(f"분류 모델 추론 실패 → 0.0으로 대체: {exc}")
        classifier_results = [{"score": 0.0, "label": "비공식"}] * len(validated_articles)

    logger.info("[8/10] 분류 모델 추론 완료")

    # ── 9. 기관 신뢰도 검증 ───────────────────────────────────────────────────
    logger.info("[9/10] 기관 신뢰도 검증 시작")
    try:
        from services.agency_verifier import verify_agency_batch
        agency_results = verify_agency_batch(validated_articles, query)
    except Exception as exc:
        logger.warning(f"기관 검증 실패 → 0.0으로 대체: {exc}")
        agency_results = [{"agency_score": 0.0}] * len(validated_articles)

    logger.info("[9/10] 기관 검증 완료")

    # ── 10. 앙상블 ─────────────────────────────────────────────────────────────
    logger.info("[10/10] 앙상블 및 최종 판정 시작")
    try:
        from services.ensemble import ensemble_batch
        final_results = ensemble_batch(
            articles=validated_articles,
            rule_scores=rule_scores,
            semantic_scores=semantic_scores,
            classifier_results=classifier_results,
            agency_results=agency_results,
            features_list=features_list,
        )
    except Exception as exc:
        logger.error(f"앙상블 실패: {exc}", exc_info=True)
        print(f"\n  ⚠️ 앙상블 처리 중 오류가 발생했습니다: {exc}\n")
        return

    logger.info(f"[10/10] 앙상블 완료 | {len(final_results)}건 판정 완료")

    # ── 결과 출력 (터미널) ────────────────────────────────────────────────────
    _print_results(final_results, query)

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    try:
        from services.result_writer import save_results
        saved_path = save_results(final_results, query, output_format)
        logger.info(f"=== 추론 완료 | 결과 저장: {saved_path} ===")
    except Exception as exc:
        logger.warning(f"결과 저장 실패: {exc}")


# ────────────────────────────────────────────────────────────────────────────────
# 터미널 출력 함수
# ────────────────────────────────────────────────────────────────────────────────

def _print_results(results: list[dict], query: str) -> None:
    """
    최종 판정 결과를 터미널에 구조화하여 출력한다.

    공식성 + 신뢰성 + 최종 판정을 분리하여 보여주어
    클라이언트에게 "왜 오피셜인지" 근거를 명확히 전달한다.
    """
    if not results:
        print("결과 없음")
        return

    # 최종 점수 기준 내림차순 정렬
    sorted_results = sorted(
        results,
        key=lambda x: (x.get("is_verified", False), x.get("official_score", 0)),
        reverse=True
    )

    print("\n" + "═" * 70)
    print(f"  검색어: {query}")
    print(f"  총 {len(results)}건 분석 완료")

    verified_count = sum(1 for r in results if r.get("is_verified", False))
    print(f"  오피셜 검증됨: {verified_count}건 / 전체 {len(results)}건")
    print("═" * 70)

    for i, r in enumerate(sorted_results[:10], 1):
        title = r.get("title", "")[:45]
        verdict_emoji = r.get("verdict_emoji", "❓")
        verdict = r.get("verdict", "알 수 없음")

        print(f"\n[{i}] {title}")
        print(f"    출처: {r.get('source', '-')} | {r.get('domain', '-')}")

        print(f"\n    [공식성 판별]")
        print(f"      규칙 점수     : {r.get('rule_score', 0):.4f}")
        print(f"      의미 유사도   : {r.get('semantic_score', 0):.4f}")
        print(f"      분류 점수     : {r.get('classifier_score', 0):.4f}")
        print(f"      기관 점수     : {r.get('agency_score', 0):.4f}")
        print(f"      공식성 점수   : {r.get('official_score', 0):.4f}")

        print(f"\n    [신뢰성 검증]")
        print(f"      교차 보도 수  : {r.get('cluster_size', 1)}건 "
              f"({r.get('unique_sources', 1)}개 출처)")
        print(f"      공식 도메인   : {'포함 ✅' if r.get('has_official_domain') else '없음'}")
        print(f"      신뢰성 점수   : {r.get('reliability_score', 0):.4f}")
        print(f"      검증 근거     : {r.get('reliability_reason', '-')}")

        print(f"\n    [최종 판정] {verdict_emoji} {verdict}")
        print(f"      근거: {r.get('verdict_reason', '-')}")
        print(f"    {'─' * 60}")

    print("\n" + "═" * 70 + "\n")


# ────────────────────────────────────────────────────────────────────────────────
# 인터랙티브 검색창 모드
# ────────────────────────────────────────────────────────────────────────────────

def run_interactive(output_format: str = "csv") -> None:
    """
    인터랙티브 검색창 모드.
    키워드를 반복 입력받아 결과를 출력한다.
    빈 입력 또는 'q' 입력 시 종료.
    """
    print("\n" + "═" * 70)
    print("  Newsoracle — 뉴스 공식성 판별 시스템")
    print("  종료하려면 엔터 또는 q 입력")
    print("═" * 70 + "\n")

    while True:
        try:
            query = input("🔍 검색어를 입력하세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not query or query.lower() == "q":
            print("종료합니다.")
            break

        # run_infer는 이제 어떤 상황에서도 return으로 복귀한다
        # 검색 결과 없음, 오타, 전처리 실패 등 모두 안내 후 여기로 돌아옴
        run_infer(query=query, output_format=output_format)

        print()


# ────────────────────────────────────────────────────────────────────────────────
# 학습 모드
# ────────────────────────────────────────────────────────────────────────────────

def run_train(train_path: str) -> None:
    """
    학습 모드: CSV 데이터로 분류 모델을 파인튜닝한다.
    """
    import os
    logger.info(f"=== 학습 모드 시작 | train_path={train_path} ===")

    if not os.path.exists(train_path):
        logger.error(f"학습 데이터 파일을 찾을 수 없음: {train_path}")
        sys.exit(1)

    try:
        from training.trainer import train
        from training.dataset import load_dataset_from_csv

        train_dataset, val_dataset, tokenizer = load_dataset_from_csv(train_path)
        train(train_dataset, val_dataset, tokenizer)
        logger.info("=== 학습 완료 ===")
    except Exception as exc:
        logger.error(f"학습 실패: {exc}", exc_info=True)
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────────────────
# CLI 진입점
# ────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI 인자를 파싱하여 train 또는 infer 모드를 실행한다."""
    parser = argparse.ArgumentParser(
        description="뉴스 공식성 판별 파이프라인 (Newsoracle)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --mode infer --query "기준금리"
  python main.py --mode infer --query "국토교통부 발표" --output json
  python main.py --mode train --train-path data/train_data.csv
        """
    )

    parser.add_argument("--mode", type=str, required=True, choices=["infer", "train"],
                        help="실행 모드: infer(추론) 또는 train(학습)")
    parser.add_argument("--query", type=str, default=None,
                        help="[infer 모드] 검색 키워드")
    parser.add_argument("--output", type=str, default="csv", choices=["csv", "json", "both"],
                        help="[infer 모드] 결과 저장 형식 (기본: csv)")
    parser.add_argument("--train-path", type=str, default="data/train_data.csv",
                        help="[train 모드] 학습 데이터 CSV 경로")

    args = parser.parse_args()

    if args.mode == "infer":
        if args.query:
            run_infer(query=args.query, output_format=args.output)
        else:
            run_interactive(output_format=args.output)
    elif args.mode == "train":
        run_train(train_path=args.train_path)


if __name__ == "__main__":
    main()
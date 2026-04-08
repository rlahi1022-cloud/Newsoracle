"""
main.py
────────
프로젝트 진입점. CLI 인자를 받아 train / infer / interactive 모드를 실행한다.

실행 예시:
  python main.py --mode infer --query "기준금리"
  python main.py --mode train --train-path data/train_data.csv
  python main.py --mode interactive
"""

import argparse
import sys
from logger import get_logger
from config import DEFAULT_TRAIN_PATH

logger = get_logger("main")


def run_pipeline(query: str, output_format: str = "csv") -> list:
    """
    단일 쿼리에 대해 전체 파이프라인을 실행한다.
    infer 모드와 interactive 모드 둘 다 이 함수를 공유한다.

    Args:
        query:         검색 쿼리
        output_format: 결과 저장 형식 ("csv" 또는 "json")
    Returns:
        최종 결과 딕셔너리 리스트
    """
    logger.info(f"파이프라인 시작 | query={query}")

    # ── 1. 뉴스 수집 ──────────────────────────────
    from news_search import search_news
    try:
        articles = search_news(query)
    except (ValueError, ConnectionError) as e:
        logger.error(f"뉴스 수집 실패: {e}")
        return []

    if not articles:
        logger.warning("수집된 기사가 없음")
        return []

    # ── 2. 전처리 ─────────────────────────────────
    from services.preprocessor import preprocess_articles
    cleaned_articles = preprocess_articles(articles)

    if not cleaned_articles:
        logger.warning("전처리 후 유효한 기사가 없음")
        return []

    # ── 3. 특징 추출 ──────────────────────────────
    from services.feature_extractor import extract_features_batch
    features_list = extract_features_batch(cleaned_articles)

    # ── 4. 기관 신뢰도 검증 ───────────────────────
    from services.agency_verifier import verify_agency_batch
    agency_results = verify_agency_batch(cleaned_articles, query)

    # ── 5. 규칙 기반 점수 ─────────────────────────
    from services.rule_based_scorer import compute_rule_scores_batch
    rule_scores = compute_rule_scores_batch(features_list)

    # ── 6. 의미 유사도 점수 ───────────────────────
    from services.semantic_similarity import compute_semantic_scores_batch
    embedding_texts = [f.get("embedding_input", "") for f in features_list]
    semantic_scores = compute_semantic_scores_batch(embedding_texts)

    # ── 7. 분류 모델 추론 ─────────────────────────
    from services.classifier_model import predict_batch
    classifier_texts = [f.get("classifier_input", "") for f in features_list]
    classifier_results = predict_batch(classifier_texts)

    # ── 8. 앙상블 ─────────────────────────────────
    from services.ensemble import ensemble_batch
    final_results = ensemble_batch(
        articles=cleaned_articles,
        rule_scores=rule_scores,
        semantic_scores=semantic_scores,
        classifier_results=classifier_results,
        agency_results=agency_results,
    )

    # ── 9. 결과 저장 ──────────────────────────────
    from services.result_writer import save_as_csv, save_as_json, print_results_summary

    print_results_summary(final_results)

    if output_format == "json":
        filepath = save_as_json(final_results, query)
    else:
        filepath = save_as_csv(final_results, query)

    logger.info(f"파이프라인 완료 | 결과 저장: {filepath}")
    return final_results


def run_infer(query: str, output_format: str = "csv"):
    """
    단일 쿼리 추론 모드.

    Args:
        query:         검색 쿼리
        output_format: 결과 저장 형식
    """
    logger.info(f"=== 추론 모드 시작 | query={query} ===")
    run_pipeline(query, output_format)
    logger.info("=== 추론 완료 ===")


def run_interactive(output_format: str = "csv"):
    """
    인터랙티브 모드.
    키워드를 반복 입력받아 파이프라인을 실행한다.
    "종료" 또는 "exit" 입력 시 프로그램 종료.

    Args:
        output_format: 결과 저장 형식
    """
    print("\n" + "=" * 60)
    print("뉴스 공식성 판별기")
    print("종료하려면 '종료' 또는 'exit' 입력")
    print("=" * 60)

    while True:
        try:
            query = input("\n검색할 키워드를 입력하세요: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n프로그램을 종료합니다.")
            break

        # 종료 명령어 처리
        if query.lower() in ["종료", "exit", "quit", "q"]:
            print("프로그램을 종료합니다.")
            break

        # 빈 입력 처리
        if not query:
            print("키워드를 입력해주세요.")
            continue

        print(f"\n'{query}' 검색 중...\n")
        logger.info(f"=== 인터랙티브 모드 | query={query} ===")

        results = run_pipeline(query, output_format)

        if not results:
            print("결과를 가져오지 못했습니다. 다시 시도해주세요.")
            continue

        # 다음 검색 안내
        print(f"\n총 {len(results)}건 판별 완료.")
        print("다른 키워드를 입력하거나 '종료'를 입력하세요.")


def run_train(train_path: str):
    """
    학습 모드: CSV 파일로 분류 모델을 파인튜닝한다.

    Args:
        train_path: 학습 데이터 CSV 경로
    """
    logger.info(f"=== 학습 모드 시작 | train_path={train_path} ===")

    from training.dataset import load_dataset_from_csv
    try:
        train_dataset, val_dataset, tokenizer = load_dataset_from_csv(train_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        sys.exit(1)

    from training.trainer import train
    train(train_dataset, val_dataset, tokenizer)

    logger.info("=== 학습 완료 ===")


def parse_args():
    parser = argparse.ArgumentParser(
        description="뉴스 공식성 판별 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["infer", "train", "interactive"],
        help=(
            "실행 모드:\n"
            "  infer:       단일 쿼리 추론\n"
            "  train:       모델 학습\n"
            "  interactive: 키워드 반복 입력 모드"
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="[infer 모드] 검색 쿼리 (예: '기준금리')",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
        help=f"[train 모드] 학습 CSV 경로 (기본값: {DEFAULT_TRAIN_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="csv",
        choices=["csv", "json"],
        help="결과 저장 형식 (기본값: csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "infer":
        if not args.query:
            logger.error("infer 모드에서는 --query 인자가 필요합니다.")
            sys.exit(1)
        run_infer(query=args.query, output_format=args.output)

    elif args.mode == "interactive":
        run_interactive(output_format=args.output)

    elif args.mode == "train":
        run_train(train_path=args.train_path)
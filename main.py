"""
main.py
────────
프로젝트 진입점. CLI 인자를 받아 train / infer 모드를 실행한다.

실행 예시:
  python main.py --mode infer --query "이재명"
  python main.py --mode train --train-path data/sample_train.csv
"""

import argparse
import sys
from logger import get_logger
from config import DEFAULT_TRAIN_PATH

logger = get_logger("main")


def run_infer(query: str, output_format: str = "csv"):
    """
    추론 모드: 쿼리로 뉴스를 수집하고 공식성 점수를 판별한다.
    
    파이프라인:
    1. 뉴스 수집 (news_search.py)
    2. 전처리 (preprocessor.py)
    3. 특징 추출 (feature_extractor.py)
    4. 규칙 기반 점수 (rule_based_scorer.py)
    5. 의미 유사도 점수 (semantic_similarity.py)
    6. 분류 모델 점수 (classifier_model.py)
    7. 앙상블 (ensemble.py)
    8. 결과 저장 (result_writer.py)
    
    Args:
        query:         검색 쿼리
        output_format: 결과 저장 형식 ("csv" 또는 "json")
    """
    logger.info(f"=== 추론 모드 시작 | query={query} ===")

    # ── 1. 뉴스 수집 ────────────────────────────────
    from news_search import search_news
    try:
        articles = search_news(query)
    except (ValueError, ConnectionError) as e:
        logger.error(f"뉴스 수집 실패: {e}")
        sys.exit(1)

    if not articles:
        logger.warning("수집된 기사가 없음. 종료합니다.")
        sys.exit(0)

    # ── 2. 전처리 ────────────────────────────────────
    from services.preprocessor import preprocess_articles
    cleaned_articles = preprocess_articles(articles)

    if not cleaned_articles:
        logger.warning("전처리 후 유효한 기사가 없음. 종료합니다.")
        sys.exit(0)

    # ── 3. 특징 추출 ─────────────────────────────────
    from services.feature_extractor import extract_features_batch
    features_list = extract_features_batch(cleaned_articles)

    # ── 4. 규칙 기반 점수 계산 ───────────────────────
    from services.rule_based_scorer import compute_rule_scores_batch
    rule_scores = compute_rule_scores_batch(features_list)

    # ── 5. 의미 유사도 점수 계산 ─────────────────────
    from services.semantic_similarity import compute_semantic_scores_batch
    embedding_texts = [f.get("embedding_input", "") for f in features_list]
    semantic_scores = compute_semantic_scores_batch(embedding_texts)

    # ── 6. 분류 모델 추론 ────────────────────────────
    from services.classifier_model import predict_batch
    classifier_texts = [f.get("classifier_input", "") for f in features_list]
    classifier_results = predict_batch(classifier_texts)

    # ── 7. 앙상블 ────────────────────────────────────
    from services.ensemble import ensemble_batch
    final_results = ensemble_batch(
        articles=cleaned_articles,
        rule_scores=rule_scores,
        semantic_scores=semantic_scores,
        classifier_results=classifier_results,
    )

    # ── 8. 결과 저장 및 출력 ─────────────────────────
    from services.result_writer import save_as_csv, save_as_json, print_results_summary

    print_results_summary(final_results)

    if output_format == "json":
        filepath = save_as_json(final_results, query)
    else:
        filepath = save_as_csv(final_results, query)

    logger.info(f"=== 추론 완료 | 결과 저장: {filepath} ===")


def run_train(train_path: str):
    """
    학습 모드: CSV 파일로 분류 모델을 파인튜닝한다.
    
    파이프라인:
    1. 데이터셋 로드 및 train/val 분리
    2. 모델 학습
    3. 최적 모델 저장
    
    Args:
        train_path: 학습 데이터 CSV 경로
    """
    logger.info(f"=== 학습 모드 시작 | train_path={train_path} ===")

    # ── 1. 데이터셋 로드 ─────────────────────────────
    from training.dataset import load_dataset_from_csv
    try:
        train_dataset, val_dataset, tokenizer = load_dataset_from_csv(train_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        sys.exit(1)

    # ── 2. 모델 학습 ─────────────────────────────────
    from training.trainer import train
    train(train_dataset, val_dataset, tokenizer)

    logger.info("=== 학습 완료 ===")


def parse_args():
    """
    CLI 인자를 파싱한다.
    
    Returns:
        파싱된 argparse.Namespace 객체
    """
    parser = argparse.ArgumentParser(
        description="뉴스 공식성 판별 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["infer", "train"],
        help="실행 모드:\n  infer: 뉴스 수집 후 공식성 판별\n  train: 분류 모델 학습",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="[infer 모드] 검색 쿼리 (예: '이재명')",
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
        help="[infer 모드] 결과 저장 형식 (기본값: csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "infer":
        if not args.query:
            logger.error("infer 모드에서는 --query 인자가 필요합니다.")
            sys.exit(1)
        run_infer(query=args.query, output_format=args.output)

    elif args.mode == "train":
        run_train(train_path=args.train_path)
# 의미 유사도 계산

"""
services/semantic_similarity.py
─────────────────────────────────
SentenceTransformer를 사용하여 기사 텍스트와
공식 기관 레퍼런스 문장 사이의 의미 유사도를 계산한다.
"""

import numpy as np
from logger import get_logger
from config import EMBEDDING_MODEL_NAME, REFERENCE_SENTENCES

logger = get_logger("semantic_similarity")

# SentenceTransformer 모델 (최초 1회만 로드)
_embedding_model = None


def _load_embedding_model():
    """
    SentenceTransformer 모델을 지연 로드한다.
    최초 호출 시 1회만 로드하고 이후 캐싱된 모델을 반환.
    
    왜 지연 로드인가:
    - 모델 로드는 수 초가 걸리므로 실제 사용 시점에 로드
    - 학습 모드에서는 임베딩 모델이 필요 없을 수 있음
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"임베딩 모델 로드 중: {EMBEDDING_MODEL_NAME}")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info("임베딩 모델 로드 완료")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise
    return _embedding_model


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    두 벡터 사이의 코사인 유사도를 계산한다.
    
    공식: sim(A, B) = (A · B) / (|A| * |B|)
    
    결과: -1.0 ~ 1.0 (1에 가까울수록 의미가 유사)
    
    Args:
        vec_a: 첫 번째 벡터
        vec_b: 두 번째 벡터
    Returns:
        코사인 유사도 값
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_semantic_score(text: str) -> float:
    """
    기사 텍스트와 레퍼런스 문장들 사이의 평균 코사인 유사도를 반환한다.
    
    레퍼런스 문장: config.REFERENCE_SENTENCES (공식 기관 문체 예시)
    
    처리 순서:
    1. 기사 텍스트 임베딩 생성
    2. 레퍼런스 문장들 임베딩 생성
    3. 각 레퍼런스 문장과의 코사인 유사도 계산
    4. 평균값 반환
    
    Args:
        text: 기사 제목 + 본문 통합 텍스트
    Returns:
        0.0 ~ 1.0 사이 semantic_score
    """
    if not text or not text.strip():
        return 0.0

    try:
        model = _load_embedding_model()

        # 기사 텍스트 임베딩 (최대 512 토큰, 초과분은 자동 잘림)
        article_embedding = model.encode(text, convert_to_numpy=True)

        # 레퍼런스 문장 임베딩
        ref_embeddings = model.encode(REFERENCE_SENTENCES, convert_to_numpy=True)

        # 각 레퍼런스 문장과 유사도 계산
        similarities = [
            cosine_similarity(article_embedding, ref_emb)
            for ref_emb in ref_embeddings
        ]

        # 평균 유사도 반환 (0~1 범위로 클리핑)
        avg_similarity = float(np.mean(similarities))
        return round(max(0.0, min(avg_similarity, 1.0)), 4)

    except Exception as e:
        logger.error(f"의미 유사도 계산 실패: {e}")
        return 0.0


def compute_semantic_scores_batch(texts: list[str]) -> list[float]:
    """
    기사 텍스트 목록 전체의 의미 유사도 점수를 계산한다.
    
    배치 처리로 임베딩 계산 효율을 높인다.
    
    Args:
        texts: 기사 텍스트 리스트
    Returns:
        각 기사의 semantic_score 리스트
    """
    if not texts:
        logger.warning("의미 유사도 계산 입력이 비어 있음")
        return []

    logger.info(f"의미 유사도 계산 시작 | {len(texts)}건")

    try:
        model = _load_embedding_model()

        # 배치 임베딩 계산 (개별 처리보다 훨씬 빠름)
        article_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        ref_embeddings = model.encode(REFERENCE_SENTENCES, convert_to_numpy=True)

        scores = []
        for article_emb in article_embeddings:
            sims = [cosine_similarity(article_emb, ref_emb) for ref_emb in ref_embeddings]
            avg = float(np.mean(sims))
            scores.append(round(max(0.0, min(avg, 1.0)), 4))

        logger.info(f"의미 유사도 계산 완료 | 평균 점수: {sum(scores)/len(scores):.4f}")
        return scores

    except Exception as e:
        logger.error(f"배치 의미 유사도 계산 실패: {e}")
        return [0.0] * len(texts)
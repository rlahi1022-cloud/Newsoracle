"""
services/query_expander.py
──────────────────────────
사용자 쿼리의 의도를 자동 분류하고, 의도에 맞는 확장 쿼리를 생성한다.

[v2 수정사항]
- _JSON_PATH를 모듈 최상단 상수로 승격 (디버깅 편의)
- 프로토타입 로드 로직을 _load_prototypes() 함수로 분리
- 로드 성공/실패 로그 명확화
- 경로 계산을 __file__ 기반 절대 경로로 통일

[설계 원칙 - 하드코딩 최소화]
1. 카테고리 분류: SentenceTransformer 의미 유사도 (기존 모델 재사용)
2. 쿼리 확장: kiwipiepy 명사 추출 (하드코딩 보강어 없음)

[프로토타입 로드 우선순위]
1. config.INTENT_PROTOTYPES
2. data/intent_prototypes.json
3. 빈 dict (fallback)
"""

import os
import json
import numpy as np
from logger import get_logger

logger = get_logger("query_expander")

# ─────────────────────────────────────────────────────────────
# 경로 상수 (모듈 최상단에서 계산)
# ─────────────────────────────────────────────────────────────
# 이 파일: /mnt/hdd/Newsoracle/services/query_expander.py
# 프로젝트 루트: /mnt/hdd/Newsoracle/
_THIS_FILE = os.path.abspath(__file__)
_SERVICES_DIR = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.dirname(_SERVICES_DIR)
_JSON_PATH = os.path.join(_PROJECT_ROOT, "data", "intent_prototypes.json")


def _load_prototypes() -> dict:
    """
    프로토타입을 config → JSON → 빈 dict 순으로 로드한다.

    Returns:
        {category: [sentence, ...]} dict
    """
    # 1순위: config.py
    try:
        from config import INTENT_PROTOTYPES as _CONFIG_PROTOS
        if _CONFIG_PROTOS:
            logger.info(f"INTENT_PROTOTYPES 로드 성공: config.py ({len(_CONFIG_PROTOS)}개 카테고리)")
            return _CONFIG_PROTOS
        else:
            logger.info("config.INTENT_PROTOTYPES가 비어 있음. JSON으로 폴백")
    except ImportError:
        logger.info(f"config에 INTENT_PROTOTYPES 없음. JSON 파일 시도: {_JSON_PATH}")

    # 2순위: JSON 파일
    if not os.path.exists(_JSON_PATH):
        logger.warning(f"INTENT_PROTOTYPES JSON 파일 없음: {_JSON_PATH}")
        return {}

    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"INTENT_PROTOTYPES 로드 성공: {_JSON_PATH} ({len(data)}개 카테고리)")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 실패: {_JSON_PATH} | {e}")
        return {}
    except Exception as e:
        logger.error(f"JSON 로드 실패: {_JSON_PATH} | {type(e).__name__}: {e}")
        return {}


# ─────────────────────────────────────────────────────────────
# 프로토타입 로드 (모듈 import 시 1회)
# ─────────────────────────────────────────────────────────────
INTENT_PROTOTYPES = _load_prototypes()

# ─────────────────────────────────────────────────────────────
# 모델 캐시 (지연 로드)
# ─────────────────────────────────────────────────────────────
_embedding_model = None
_prototype_embeddings = None
_kiwi = None
_category_keywords = None


def _load_embedding_model():
    """SentenceTransformer 지연 로드."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL_NAME
        logger.info(f"쿼리 확장용 임베딩 모델 로드: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _load_kiwi():
    """kiwipiepy 형태소 분석기 지연 로드."""
    global _kiwi
    if _kiwi is None:
        try:
            from kiwipiepy import Kiwi
            logger.info("kiwipiepy 형태소 분석기 로드")
            _kiwi = Kiwi()
        except ImportError:
            logger.error("kiwipiepy 미설치. pip install kiwipiepy 필요")
            raise
    return _kiwi


def _precompute_prototype_embeddings():
    """카테고리별 프로토타입 평균 임베딩 미리 계산."""
    global _prototype_embeddings
    if _prototype_embeddings is not None:
        return _prototype_embeddings

    if not INTENT_PROTOTYPES:
        logger.warning("INTENT_PROTOTYPES가 비어 있음")
        _prototype_embeddings = {}
        return _prototype_embeddings

    model = _load_embedding_model()
    _prototype_embeddings = {}

    for category, sentences in INTENT_PROTOTYPES.items():
        if not sentences:
            continue
        embs = model.encode(sentences, convert_to_numpy=True)
        mean_emb = np.mean(embs, axis=0)
        _prototype_embeddings[category] = mean_emb

    logger.info(f"프로토타입 임베딩 캐시 생성 | {len(_prototype_embeddings)}개 카테고리")
    return _prototype_embeddings


def _precompute_category_keywords():
    """프로토타입 문장에서 kiwipiepy로 명사 추출."""
    global _category_keywords
    if _category_keywords is not None:
        return _category_keywords

    if not INTENT_PROTOTYPES:
        _category_keywords = {}
        return _category_keywords

    kiwi = _load_kiwi()
    _category_keywords = {}

    for category, sentences in INTENT_PROTOTYPES.items():
        nouns = []
        for sent in sentences:
            try:
                tokens = kiwi.tokenize(sent)
                for tok in tokens:
                    if tok.tag in ("NNG", "NNP", "SL") and len(tok.form) >= 2:
                        nouns.append(tok.form)
            except Exception as e:
                logger.warning(f"형태소 분석 실패: '{sent}' | {e}")
                continue

        unique_nouns = list(dict.fromkeys(nouns))
        _category_keywords[category] = unique_nouns
        logger.info(f"카테고리 '{category}' 키워드: {unique_nouns}")

    return _category_keywords


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ═════════════════════════════════════════════════════════════
# 공개 함수
# ═════════════════════════════════════════════════════════════

def classify_intent(query: str, top_k: int = 4, min_similarity: float = 0.0) -> list:
    """
    사용자 쿼리의 의도를 카테고리별로 분류.

    Returns:
        [(category_name, similarity_score), ...] 유사도 내림차순
    """
    query = query.strip()
    if not query:
        return []

    proto_embs = _precompute_prototype_embeddings()
    if not proto_embs:
        logger.warning("프로토타입 없음. 전체 카테고리 기본 반환")
        return [(cat, 0.0) for cat in INTENT_PROTOTYPES.keys()][:top_k]

    try:
        model = _load_embedding_model()
        query_emb = model.encode(query, convert_to_numpy=True)

        scores = []
        for category, proto_emb in proto_embs.items():
            sim = _cosine_similarity(query_emb, proto_emb)
            if sim >= min_similarity:
                scores.append((category, round(sim, 4)))

        scores.sort(key=lambda x: x[1], reverse=True)
        result = scores[:top_k]
        logger.info(f"의도 분류 | query='{query}' → {result}")
        return result

    except Exception as e:
        logger.error(f"의도 분류 실패: {e}")
        return []


def expand_query(query: str, category: str, max_variants: int = 5) -> list:
    """
    선택된 카테고리에 맞게 쿼리 확장.

    Returns:
        확장 쿼리 리스트 (첫 번째는 원본)
    """
    query = query.strip()
    if not query:
        return []

    category_kw = _precompute_category_keywords()
    keywords = category_kw.get(category, [])

    if not keywords:
        logger.warning(f"카테고리 '{category}' 키워드 없음. 원본만 반환")
        return [query]

    expanded = [query]
    for kw in keywords:
        if kw in query:
            continue
        expanded.append(f"{query} {kw}")

    seen = set()
    unique = []
    for q in expanded:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    result = unique[:max_variants]
    logger.info(f"쿼리 확장 | query='{query}' category='{category}' → {result}")
    return result


def get_available_categories() -> list:
    """전체 카테고리 이름 리스트 반환."""
    return list(INTENT_PROTOTYPES.keys())


def warmup():
    """서버 기동 시 호출. 모델/캐시 미리 로드."""
    logger.info("query_expander warmup 시작")
    try:
        _load_embedding_model()
        _precompute_prototype_embeddings()
        _load_kiwi()
        _precompute_category_keywords()
        logger.info("query_expander warmup 완료")
    except Exception as e:
        logger.error(f"query_expander warmup 실패: {e}")
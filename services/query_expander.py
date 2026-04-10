"""
services/query_expander.py
──────────────────────────
사용자 쿼리의 의도를 자동 분류하고, 의도에 맞는 확장 쿼리를 생성한다.

[설계 원칙 - 하드코딩 최소화]
1. 카테고리 분류: SentenceTransformer 의미 유사도
   - 이미 프로젝트에 로드된 jhgan/ko-sroberta-multitask 재사용
   - 추가 모델 로드 없음, 메모리 부담 0
   - 카테고리별 "프로토타입 문장" 3~5개만 정의 (의도 설명문)
   - 템플릿 "{q} 최근" 같은 하드코딩 없음

2. 쿼리 확장: kiwipiepy 명사 추출
   - 프로토타입 문장에서 핵심 명사를 형태소 분석으로 자동 추출
   - 사용자 쿼리 + 추출 명사로 확장 쿼리 조합
   - 하드코딩된 보강어 리스트 없음

[작동 흐름]
  사용자 쿼리 "악뮤" 입력
    ↓
  classify_intent("악뮤")
    → SentenceTransformer로 프로토타입 임베딩과 코사인 유사도 계산
    → 상위 N개 카테고리 반환: [("recent", 0.62), ("action", 0.55), ...]
    ↓
  [UI] 사용자에게 상위 카테고리 제시
    ↓
  사용자가 "recent" 선택
    ↓
  expand_query("악뮤", "recent")
    → 프로토타입 문장에서 kiwipiepy로 명사 추출: ["최근", "근황", "소식"]
    → 쿼리 조합: ["악뮤", "악뮤 최근", "악뮤 근황", "악뮤 소식"]
    ↓
  news_search.search_news_by_category() 가 각 쿼리로 API 호출

[config 연동]
  config.py에 INTENT_PROTOTYPES (dict[str, list[str]]) 정의 필요
  예:
    INTENT_PROTOTYPES = {
        "recent":   ["최근 소식이 궁금하다", "근황을 알고 싶다", ...],
        "official": ["공식 발표 내용", "보도자료 확인", ...],
        ...
    }
"""

import numpy as np
from logger import get_logger

logger = get_logger("query_expander")

# ─────────────────────────────────────────────────────────────
# config에서 프로토타입 import (없으면 빈 dict로 fallback)
# ─────────────────────────────────────────────────────────────
try:
    from config import INTENT_PROTOTYPES
except ImportError:
    logger.warning("config.INTENT_PROTOTYPES 없음. 빈 dict로 fallback")
    INTENT_PROTOTYPES = {}

# ─────────────────────────────────────────────────────────────
# 모델 캐시 (지연 로드)
# ─────────────────────────────────────────────────────────────
_embedding_model = None
_prototype_embeddings = None   # {category: np.ndarray(mean_embedding)}
_kiwi = None
_category_keywords = None      # {category: [추출된 명사 리스트]}


def _load_embedding_model():
    """
    semantic_similarity.py와 동일한 SentenceTransformer를 로드한다.
    이미 해당 모듈에서 로드된 경우 재사용되지 않고 별도 인스턴스 생성되지만,
    실제로는 HuggingFace 캐시에서 즉시 로드되므로 비용 무시할 수준.

    재사용 최적화를 원한다면 semantic_similarity._load_embedding_model()을
    직접 호출해도 되지만, 모듈 간 강결합을 피하기 위해 분리.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        from config import EMBEDDING_MODEL_NAME
        logger.info(f"쿼리 확장용 임베딩 모델 로드: {EMBEDDING_MODEL_NAME}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _load_kiwi():
    """
    kiwipiepy 형태소 분석기를 지연 로드한다.
    kiwipiepy는 경량(10MB)이지만 초기화에 0.5초 정도 걸리므로 지연 로드.
    """
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
    """
    카테고리별 프로토타입 문장들의 평균 임베딩을 미리 계산한다.
    서버 기동 시 1회만 수행되며 이후 캐시 사용.

    왜 평균인가:
    - 카테고리당 프로토타입이 3~5개 (recent = 최근/근황/소식)
    - 각 문장을 개별 비교하면 노이즈 많음
    - 평균 임베딩 = 카테고리의 "의미 중심" → 더 안정적
    """
    global _prototype_embeddings
    if _prototype_embeddings is not None:
        return _prototype_embeddings

    if not INTENT_PROTOTYPES:
        logger.warning("INTENT_PROTOTYPES가 비어 있음. 의도 분류 불가")
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
    """
    각 카테고리 프로토타입 문장에서 kiwipiepy로 명사를 추출하여 캐시.
    쿼리 확장 시 이 명사들을 사용자 쿼리 뒤에 붙여 확장 쿼리를 만든다.

    추출 대상 품사:
    - NNG (일반 명사): 소식, 발표, 일정, 계획, 활동 등
    - NNP (고유 명사): 거의 없지만 포함
    - SL (외국어): 영문 약어 등

    중복 제거 + 길이 2자 이상 조건으로 노이즈 제거.
    """
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
                    # NNG(일반명사), NNP(고유명사), SL(외국어) 추출
                    if tok.tag in ("NNG", "NNP", "SL") and len(tok.form) >= 2:
                        nouns.append(tok.form)
            except Exception as e:
                logger.warning(f"형태소 분석 실패: '{sent}' | {e}")
                continue

        # 중복 제거 (순서 유지)
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
    사용자 쿼리의 의도를 카테고리별로 분류한다.

    처리 순서:
    1. 쿼리 임베딩 계산
    2. 각 카테고리 프로토타입 평균 임베딩과 코사인 유사도 계산
    3. 유사도 내림차순 정렬 후 상위 top_k 반환
    4. min_similarity 미만은 제외

    Args:
        query:          사용자 입력 쿼리
        top_k:          반환할 카테고리 최대 개수
        min_similarity: 최소 유사도 임계값 (이 미만 제외)
    Returns:
        [(category_name, similarity_score), ...] 유사도 내림차순
        예: [("recent", 0.62), ("action", 0.55), ("official", 0.48)]
    """
    query = query.strip()
    if not query:
        return []

    proto_embs = _precompute_prototype_embeddings()
    if not proto_embs:
        logger.warning("프로토타입 없음. 전체 카테고리를 기본 반환")
        return [(cat, 0.0) for cat in INTENT_PROTOTYPES.keys()][:top_k]

    try:
        model = _load_embedding_model()
        query_emb = model.encode(query, convert_to_numpy=True)

        # 카테고리별 유사도 계산
        scores = []
        for category, proto_emb in proto_embs.items():
            sim = _cosine_similarity(query_emb, proto_emb)
            if sim >= min_similarity:
                scores.append((category, round(sim, 4)))

        # 유사도 내림차순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        result = scores[:top_k]
        logger.info(f"의도 분류 | query='{query}' → {result}")
        return result

    except Exception as e:
        logger.error(f"의도 분류 실패: {e}")
        return []


def expand_query(query: str, category: str, max_variants: int = 5) -> list:
    """
    선택된 카테고리에 맞게 쿼리를 확장한다.

    처리 순서:
    1. 카테고리 프로토타입 문장에서 미리 추출된 명사 목록 가져옴
    2. 각 명사를 사용자 쿼리 뒤에 붙여 확장 쿼리 생성
    3. 원본 쿼리 + 확장 쿼리 반환 (중복 제거, 최대 max_variants개)

    예:
      query="악뮤", category="recent"
      → 프로토타입 명사: ["소식", "근황", "최근"]
      → 확장 쿼리: ["악뮤", "악뮤 소식", "악뮤 근황", "악뮤 최근"]

    Args:
        query:        원본 사용자 쿼리
        category:     classify_intent()가 반환한 카테고리 이름
        max_variants: 반환할 쿼리 최대 개수 (원본 포함)
    Returns:
        확장 쿼리 리스트 (첫 번째는 항상 원본)
    """
    query = query.strip()
    if not query:
        return []

    category_kw = _precompute_category_keywords()
    keywords = category_kw.get(category, [])

    if not keywords:
        logger.warning(f"카테고리 '{category}' 키워드 없음. 원본만 반환")
        return [query]

    # 원본 + 키워드 조합
    expanded = [query]
    for kw in keywords:
        # 이미 쿼리에 포함된 키워드는 스킵
        if kw in query:
            continue
        expanded.append(f"{query} {kw}")

    # 중복 제거 (순서 유지)
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
    """
    config에 정의된 전체 카테고리 이름 리스트를 반환한다.
    UI에서 "전체" 또는 "기타" 선택지를 구성할 때 사용.
    """
    return list(INTENT_PROTOTYPES.keys())


def warmup():
    """
    서버 기동 시 호출하여 모델/캐시를 미리 로드한다.
    첫 요청의 지연을 줄이기 위함.

    server.py startup 훅에서 호출 권장:
        from services.query_expander import warmup
        warmup()
    """
    logger.info("query_expander warmup 시작")
    try:
        _load_embedding_model()
        _precompute_prototype_embeddings()
        _load_kiwi()
        _precompute_category_keywords()
        logger.info("query_expander warmup 완료")
    except Exception as e:
        logger.error(f"query_expander warmup 실패: {e}")
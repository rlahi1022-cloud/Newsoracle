"""
services/cross_validator.py
────────────────────────────────────────────────────────────────────────────────
기사 대 기사 클러스터링 기반 교차 보도 검증 모듈

[역할]
  수집된 기사들을 임베딩 후 서로 비교하여
  "같은 사실을 몇 개 독립 언론이 보도했는가"를 측정한다.

[핵심 설계 원칙]
  - 기사 ↔ 가상 문장 비교 (기존) → 신뢰 근거 없음
  - 기사 ↔ 기사 직접 비교 (신규) → 실제 데이터 기반 신뢰 근거

[신뢰성 측정 기준]
  1. 클러스터 내 기사 수    : 같은 사실을 보도한 기사가 많을수록 신뢰도 ↑
  2. 출처 다양성            : 같은 클러스터 내 언론사가 다양할수록 신뢰도 ↑
  3. 공식 도메인 포함 여부  : .go.kr 등 공식 도메인 출처 포함 시 신뢰도 ↑
  4. 내용 일치도            : 클러스터 내 기사 간 평균 유사도 ↑

[최종 판정 기준]
  공식성 점수 (official_score) + 신뢰성 점수 (reliability_score)
  → 둘 다 임계값 이상 → "오피셜 검증됨"
  → 하나만 이상       → 조건부 판정
  → 둘 다 미달        → "검증 불가"
"""

import numpy as np
from typing import Optional
from logger import get_logger
from config import EMBEDDING_MODEL_NAME, EnsembleConfig

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# 클러스터링 설정 상수
# ────────────────────────────────────────────────────────────────────────────────

# 같은 클러스터로 묶기 위한 코사인 유사도 임계값
# 0.75 이상이면 "같은 사실을 보도한 기사"로 간주
# 너무 낮으면 관련 없는 기사가 묶이고, 너무 높으면 같은 기사도 다른 클러스터로 분리됨
CLUSTER_SIMILARITY_THRESHOLD = 0.75

# 신뢰성 점수 계산 시 기준값
# 이 수 이상의 독립 출처가 같은 사실을 보도하면 신뢰도 만점
MAX_SOURCE_COUNT_FOR_SCORE = 5

# 공식 도메인 키워드 (이 도메인이 클러스터에 포함되면 가산점)
OFFICIAL_DOMAIN_BONUS_KEYWORDS = [".go.kr", ".or.kr", "yna.co.kr", "korea.kr"]

# 신뢰성 판정 임계값
RELIABILITY_VERIFIED_THRESHOLD = 0.5

# ── 최종 판정 임계값 (config.EnsembleConfig에서 참조) ─────────────────────
# [v5 변경] 기존에 하드코딩(0.5, 0.4)되어 있어서
# config에서 0.45, 0.50으로 바꿔도 여기서 적용이 안 됐음
# 이제 config 값을 직접 참조하여 불일치 제거
OFFICIAL_VERIFIED_THRESHOLD = EnsembleConfig.OFFICIAL_SCORE_THRESHOLD    # 0.45
RELIABILITY_FINAL_THRESHOLD = EnsembleConfig.RELIABILITY_SCORE_THRESHOLD  # 0.50


# ────────────────────────────────────────────────────────────────────────────────
# 핵심 함수: 기사 임베딩 생성
# ────────────────────────────────────────────────────────────────────────────────

def get_article_embeddings(articles: list[dict]) -> Optional[np.ndarray]:
    """
    기사 목록을 SentenceTransformer로 임베딩하여 numpy 배열로 반환한다.

    Args:
        articles (list[dict]): 전처리된 기사 딕셔너리 목록
                               각 기사에 'title', 'content' 키 필요

    Returns:
        np.ndarray: shape (N, embedding_dim) 의 임베딩 배열
                    임베딩 실패 시 None 반환

    설계 이유:
        SentenceTransformer를 여기서 직접 로드하지 않고
        semantic_similarity.py에서 이미 로드한 모델을 재사용하면
        메모리 중복 로드를 방지할 수 있다.
        단, cross_validator는 독립 모듈이므로 자체 로드 구조를 유지하고
        모델 인스턴스를 파라미터로 받는 구조로 설계한다.
    """
    if not articles:
        logger.warning("임베딩 입력 기사 목록이 비어있음")
        return None

    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"교차 검증용 임베딩 모델 로드: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # 기사 텍스트 구성: 제목 + 본문 (제목에 더 가중치를 두기 위해 2번 포함)
        # 제목이 같으면 같은 사실을 보도한 기사일 가능성이 높음
        texts = []
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "")
            # 제목을 앞에 두 번 넣어 제목 유사도에 더 가중치 부여
            combined = f"{title} {title} {content}".strip()
            texts.append(combined[:512])  # 최대 512자로 제한

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True  # 코사인 유사도 계산을 위해 정규화
        )

        logger.info(f"임베딩 완료 | {len(articles)}건 | shape={embeddings.shape}")
        return embeddings

    except ImportError:
        logger.error("sentence-transformers 미설치. pip install sentence-transformers 실행 필요")
        return None
    except Exception as exc:
        logger.error(f"임베딩 생성 실패: {exc}", exc_info=True)
        return None


# ────────────────────────────────────────────────────────────────────────────────
# 핵심 함수: 코사인 유사도 행렬 계산
# ────────────────────────────────────────────────────────────────────────────────

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    임베딩 배열에서 모든 기사 쌍의 코사인 유사도 행렬을 계산한다.

    Args:
        embeddings (np.ndarray): 정규화된 임베딩 배열 (N, dim)

    Returns:
        np.ndarray: (N, N) 유사도 행렬
                    matrix[i][j] = 기사 i와 기사 j의 코사인 유사도

    설계 이유:
        정규화된 벡터끼리의 내적 = 코사인 유사도
        행렬 곱 한 번으로 모든 쌍의 유사도를 한꺼번에 계산 (효율적)
    """
    if embeddings is None or len(embeddings) == 0:
        return np.array([])

    # 정규화된 벡터의 내적 = 코사인 유사도
    similarity_matrix = np.dot(embeddings, embeddings.T)

    # 부동소수점 오차로 1.0을 초과하는 경우 클리핑
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    return similarity_matrix


# ────────────────────────────────────────────────────────────────────────────────
# 핵심 함수: 클러스터링 (단순 임계값 기반)
# ────────────────────────────────────────────────────────────────────────────────

def cluster_articles_by_similarity(
    similarity_matrix: np.ndarray,
    threshold: float = CLUSTER_SIMILARITY_THRESHOLD
) -> list[list[int]]:
    """
    유사도 행렬을 기반으로 기사들을 클러스터로 묶는다.

    알고리즘: 탐욕적 클러스터링 (Greedy Clustering)
      - 아직 클러스터에 할당되지 않은 첫 번째 기사를 새 클러스터의 seed로 설정
      - seed와 유사도가 threshold 이상인 기사들을 같은 클러스터로 묶음
      - 모든 기사가 클러스터에 할당될 때까지 반복

    Args:
        similarity_matrix (np.ndarray): (N, N) 유사도 행렬
        threshold (float): 같은 클러스터로 묶기 위한 최소 유사도

    Returns:
        list[list[int]]: 클러스터별 기사 인덱스 목록
                         예: [[0, 3, 7], [1, 4], [2], [5, 6]]

    설계 이유:
        K-Means나 DBSCAN 대신 탐욕적 방법을 사용하는 이유:
        - 기사 수가 20~100건 수준으로 적어서 단순 방법으로 충분
        - 클러스터 수를 미리 정하지 않아도 됨 (K-Means는 K를 정해야 함)
        - 유사도 임계값 하나로 직관적으로 제어 가능
    """
    if similarity_matrix is None or len(similarity_matrix) == 0:
        return []

    n = len(similarity_matrix)
    assigned = [False] * n  # 각 기사의 클러스터 할당 여부
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue  # 이미 클러스터에 할당된 기사는 스킵

        # 새 클러스터 시작 (i번 기사가 seed)
        cluster = [i]
        assigned[i] = True

        # i번 기사와 유사도가 threshold 이상인 기사들을 같은 클러스터로
        for j in range(i + 1, n):
            if not assigned[j] and similarity_matrix[i][j] >= threshold:
                cluster.append(j)
                assigned[j] = True

        clusters.append(cluster)

    logger.debug(f"클러스터링 완료 | {n}건 → {len(clusters)}개 클러스터 | threshold={threshold}")
    return clusters


# ────────────────────────────────────────────────────────────────────────────────
# 핵심 함수: 클러스터별 신뢰성 점수 계산
# ────────────────────────────────────────────────────────────────────────────────

def compute_cluster_reliability(
    cluster_indices: list[int],
    articles: list[dict],
    similarity_matrix: np.ndarray
) -> dict:
    """
    단일 클러스터에 대한 신뢰성 점수와 세부 정보를 계산한다.

    Args:
        cluster_indices (list[int]): 클러스터에 속한 기사 인덱스 목록
        articles (list[dict]): 전체 기사 목록 (도메인, 출처 정보 포함)
        similarity_matrix (np.ndarray): 전체 유사도 행렬

    Returns:
        dict: 클러스터 신뢰성 정보
            - cluster_size: 클러스터 내 기사 수
            - unique_sources: 독립 출처 수
            - source_list: 출처 목록
            - has_official_domain: 공식 도메인 포함 여부
            - avg_similarity: 클러스터 내 평균 유사도
            - reliability_score: 0.0 ~ 1.0 최종 신뢰성 점수
            - reliability_reason: 신뢰성 판정 근거 설명
    """
    if not cluster_indices:
        return _empty_reliability_result()

    cluster_articles = [articles[i] for i in cluster_indices if i < len(articles)]
    cluster_size = len(cluster_indices)

    # ── 1. 출처 다양성 계산 ──────────────────────────────────────────────────
    # 도메인 또는 source 필드를 기준으로 독립 출처 추출
    sources = []
    domains = []
    for article in cluster_articles:
        source = article.get("source", "")
        domain = article.get("domain", article.get("originallink", ""))
        if source and source not in sources:
            sources.append(source)
        if domain and domain not in domains:
            domains.append(domain)

    unique_sources = len(sources)

    # ── 2. 공식 도메인 포함 여부 확인 ────────────────────────────────────────
    has_official_domain = False
    official_sources = []
    for domain in domains:
        for official_keyword in OFFICIAL_DOMAIN_BONUS_KEYWORDS:
            if official_keyword in domain:
                has_official_domain = True
                official_sources.append(domain)
                break

    # ── 3. 클러스터 내 평균 유사도 계산 ──────────────────────────────────────
    # 클러스터 내 기사가 1개면 유사도 비교 불가 → 0.0
    avg_similarity = 0.0
    if cluster_size > 1:
        sim_values = []
        for i_idx in range(len(cluster_indices)):
            for j_idx in range(i_idx + 1, len(cluster_indices)):
                i = cluster_indices[i_idx]
                j = cluster_indices[j_idx]
                if i < len(similarity_matrix) and j < len(similarity_matrix):
                    sim_values.append(float(similarity_matrix[i][j]))
        avg_similarity = float(np.mean(sim_values)) if sim_values else 0.0

    # ── 4. 신뢰성 점수 계산 ──────────────────────────────────────────────────
    # 구성 요소:
    #   source_score   : 독립 출처 수 기반 (가장 중요)
    #   official_bonus : 공식 도메인 포함 시 가산점
    #   similarity_score: 클러스터 내 내용 일치도

    # 독립 출처 수 기반 점수 (MAX_SOURCE_COUNT_FOR_SCORE개 이상이면 1.0)
    source_score = min(unique_sources / MAX_SOURCE_COUNT_FOR_SCORE, 1.0)

    # 공식 도메인 포함 시 가산점 (+0.2, 최대 1.0으로 클리핑)
    official_bonus = 0.2 if has_official_domain else 0.0

    # 내용 일치도 점수 (단일 기사면 0.5로 기본값 설정)
    similarity_score = avg_similarity if cluster_size > 1 else 0.5

    # 가중 합산
    # source_score에 높은 가중치를 두는 이유:
    #   여러 독립 언론이 같은 사실을 보도했다는 것이 가장 강력한 신뢰 근거
    reliability_score = (
        source_score * 0.5          # 출처 다양성 (50%)
        + similarity_score * 0.3    # 내용 일치도 (30%)
        + official_bonus * 0.2      # 공식 도메인 가산점 (20%)
    )
    reliability_score = reliability_score + official_bonus

    # [v5 변경] 신뢰성 상한 0.999 (100%는 불가능)
    # 아무리 교차 보도가 많아도 "완벽히 신뢰"는 불가능하다.
    # 공식성은 기관 직접 발표면 100%가 될 수 있지만,
    # 신뢰성은 언론 보도 기반이므로 항상 불확실성이 존재한다.
    reliability_score = min(reliability_score, 0.999)
    reliability_score = round(float(reliability_score), 4)

    # ── 5. 신뢰성 판정 근거 메시지 생성 ─────────────────────────────────────
    reliability_reason = _build_reliability_reason(
        cluster_size=cluster_size,
        unique_sources=unique_sources,
        has_official_domain=has_official_domain,
        official_sources=official_sources,
        avg_similarity=avg_similarity,
        reliability_score=reliability_score,
        sources=sources
    )

    return {
        "cluster_size": cluster_size,
        "unique_sources": unique_sources,
        "source_list": sources,
        "has_official_domain": has_official_domain,
        "official_domain_sources": official_sources,
        "avg_similarity": round(avg_similarity, 4),
        "reliability_score": reliability_score,
        "reliability_reason": reliability_reason,
    }


def _empty_reliability_result() -> dict:
    """
    클러스터가 비어있거나 처리 불가할 때 반환하는 기본 결과.
    """
    return {
        "cluster_size": 0,
        "unique_sources": 0,
        "source_list": [],
        "has_official_domain": False,
        "official_domain_sources": [],
        "avg_similarity": 0.0,
        "reliability_score": 0.0,
        "reliability_reason": "교차 검증 불가 (기사 부족)",
    }


def _build_reliability_reason(
    cluster_size: int,
    unique_sources: int,
    has_official_domain: bool,
    official_sources: list,
    avg_similarity: float,
    reliability_score: float,
    sources: list
) -> str:
    """
    클라이언트에게 보여줄 신뢰성 판정 근거 메시지를 생성한다.

    Args:
        모든 파라미터는 compute_cluster_reliability에서 계산된 값

    Returns:
        str: 사람이 읽을 수 있는 신뢰성 판정 근거 문자열
    """
    parts = []

    # 교차 보도 수
    if cluster_size >= 3:
        parts.append(f"{unique_sources}개 독립 언론사 교차 보도 확인")
    elif cluster_size == 2:
        parts.append(f"{unique_sources}개 언론사 보도 확인")
    else:
        parts.append("단독 보도 (교차 검증 미흡)")

    # 출처 목록 (최대 3개까지만 표시)
    if sources:
        source_display = ", ".join(sources[:3])
        if len(sources) > 3:
            source_display += f" 외 {len(sources) - 3}개"
        parts.append(f"출처: {source_display}")

    # 공식 도메인 포함 여부
    if has_official_domain:
        parts.append(f"공식 도메인 출처 포함 ({', '.join(official_sources[:2])})")

    # 내용 일치도
    if cluster_size > 1:
        if avg_similarity >= 0.85:
            parts.append(f"내용 일치도 매우 높음 ({avg_similarity:.2f})")
        elif avg_similarity >= 0.75:
            parts.append(f"내용 일치도 높음 ({avg_similarity:.2f})")
        else:
            parts.append(f"내용 일치도 보통 ({avg_similarity:.2f})")

    return " | ".join(parts)


# ────────────────────────────────────────────────────────────────────────────────
# 메인 공개 함수: 전체 교차 검증 파이프라인
# ────────────────────────────────────────────────────────────────────────────────

def cross_validate_articles(articles: list[dict]) -> list[dict]:
    """
    수집된 전체 기사에 대해 교차 보도 검증을 수행하고
    각 기사에 신뢰성 점수(reliability_score)와 판정 근거를 추가하여 반환한다.

    처리 흐름:
      1. 전체 기사 임베딩 생성
      2. 기사 대 기사 유사도 행렬 계산
      3. 클러스터링 (같은 사실을 보도한 기사끼리 묶기)
      4. 클러스터별 신뢰성 점수 계산
      5. 각 기사에 소속 클러스터의 신뢰성 점수 부여

    Args:
        articles (list[dict]): 전처리된 기사 딕셔너리 목록

    Returns:
        list[dict]: reliability_score, reliability_reason,
                    cluster_id, cluster_size 필드가 추가된 기사 목록

    설계 이유:
        임베딩 실패나 기사 수 부족 시에도 파이프라인이 중단되지 않도록
        fallback 값(reliability_score=0.0)을 설정하고 계속 진행한다.
    """
    logger.info(f"[교차 검증] 시작 | 총 {len(articles)}건")

    # 기사가 1건 이하면 교차 검증 불가
    if len(articles) <= 1:
        logger.warning("기사가 1건 이하 → 교차 검증 불가, 기본값 반환")
        for article in articles:
            article.update(_empty_reliability_result())
            article["cluster_id"] = 0
        return articles

    # ── Step 1: 임베딩 생성 ─────────────────────────────────────────────────
    embeddings = get_article_embeddings(articles)

    if embeddings is None:
        # 임베딩 실패 시 fallback: 모든 기사에 기본 신뢰성 점수 부여
        logger.warning("임베딩 실패 → 규칙 기반 fallback으로 신뢰성 점수 계산")
        return _fallback_reliability(articles)

    # ── Step 2: 유사도 행렬 계산 ────────────────────────────────────────────
    similarity_matrix = compute_similarity_matrix(embeddings)

    # ── Step 3: 클러스터링 ──────────────────────────────────────────────────
    clusters = cluster_articles_by_similarity(similarity_matrix)

    logger.info(
        f"[교차 검증] 클러스터링 완료 | "
        f"{len(articles)}건 → {len(clusters)}개 클러스터"
    )

    # ── Step 4 & 5: 클러스터별 신뢰성 계산 후 기사에 부여 ──────────────────
    # 기사 인덱스 → 클러스터 ID 역매핑 딕셔너리 생성
    article_to_cluster = {}
    for cluster_id, cluster_indices in enumerate(clusters):
        for article_idx in cluster_indices:
            article_to_cluster[article_idx] = cluster_id

    # 클러스터별 신뢰성 정보 사전 계산
    cluster_reliability_cache = {}
    for cluster_id, cluster_indices in enumerate(clusters):
        reliability_info = compute_cluster_reliability(
            cluster_indices=cluster_indices,
            articles=articles,
            similarity_matrix=similarity_matrix
        )
        cluster_reliability_cache[cluster_id] = reliability_info

        logger.debug(
            f"클러스터 {cluster_id} | "
            f"size={reliability_info['cluster_size']} | "
            f"sources={reliability_info['unique_sources']} | "
            f"reliability={reliability_info['reliability_score']:.4f}"
        )

    # 각 기사에 소속 클러스터의 신뢰성 정보 추가
    result_articles = []
    for article_idx, article in enumerate(articles):
        cluster_id = article_to_cluster.get(article_idx, 0)
        reliability_info = cluster_reliability_cache.get(cluster_id, _empty_reliability_result())

        # 기사 딕셔너리에 신뢰성 정보 병합
        enriched_article = {**article}
        enriched_article["cluster_id"] = cluster_id
        enriched_article["cluster_size"] = reliability_info["cluster_size"]
        enriched_article["unique_sources"] = reliability_info["unique_sources"]
        enriched_article["source_list"] = reliability_info["source_list"]
        enriched_article["has_official_domain"] = reliability_info["has_official_domain"]
        enriched_article["avg_similarity"] = reliability_info["avg_similarity"]
        enriched_article["reliability_score"] = reliability_info["reliability_score"]
        enriched_article["reliability_reason"] = reliability_info["reliability_reason"]

        result_articles.append(enriched_article)

    # 전체 요약 로그
    avg_reliability = float(np.mean([
        a.get("reliability_score", 0.0) for a in result_articles
    ]))
    verified_count = sum(
        1 for a in result_articles
        if a.get("reliability_score", 0.0) >= RELIABILITY_VERIFIED_THRESHOLD
    )

    logger.info(
        f"[교차 검증] 완료 | "
        f"평균 신뢰성={avg_reliability:.4f} | "
        f"신뢰성 검증 통과={verified_count}/{len(result_articles)}건"
    )

    # ── Step 6: 클러스터 기반 중복 필터링 (v5 신규) ────────────────────────────
    # 같은 cluster_id를 가진 기사들 중 대표 1건만 남기고 나머지 제거
    # 대표 기사 선정 기준:
    #   1. 공식 도메인 포함 기사 우선
    #   2. 도메인 등급이 높은 기사 우선
    #   3. 제목이 긴 기사 우선 (정보량이 더 많은 경향)
    #
    # 왜 필요한가:
    #   SK증권 업무협약 기사가 14개 출처로 반복 노출되면
    #   검색 결과 상위를 같은 사건이 독점하여 다양성이 사라짐
    #   대표 1건만 남기면 다양한 사건을 골고루 볼 수 있음
    deduplicated = _deduplicate_by_cluster(result_articles)

    logger.info(
        f"[교차 검증] 중복 필터링 | "
        f"원본={len(result_articles)}건 → 대표={len(deduplicated)}건 "
        f"(중복 제거={len(result_articles) - len(deduplicated)}건)"
    )

    return deduplicated


def _fallback_reliability(articles: list[dict]) -> list[dict]:
    """
    임베딩 모델 로드 실패 시 규칙 기반으로 최소한의 신뢰성 점수를 부여한다.

    임베딩 없이도 다음 기준으로 기본 신뢰성 판단:
      - .go.kr 도메인이면 reliability_score = 0.5
      - 그 외 언론사 도메인이면 reliability_score = 0.3
      - 그 외면 reliability_score = 0.1

    Args:
        articles (list[dict]): 전처리된 기사 목록

    Returns:
        list[dict]: 기본 신뢰성 점수가 부여된 기사 목록
    """
    logger.info("Fallback 신뢰성 점수 계산 (임베딩 없이 도메인 기반)")
    result = []
    for idx, article in enumerate(articles):
        domain = article.get("domain", article.get("originallink", ""))
        originallink = article.get("originallink", "")

        # 공식 도메인 여부 확인
        is_official = any(
            kw in domain or kw in originallink
            for kw in OFFICIAL_DOMAIN_BONUS_KEYWORDS
        )

        # 신뢰 언론사 도메인 여부 (간단 체크)
        trusted_media_keywords = [
            "yonhap", "yna", "kbs", "mbc", "sbs", "ytn",
            "chosun", "joongang", "donga", "hani", "mk", "hankyung"
        ]
        is_trusted_media = any(
            kw in domain or kw in originallink
            for kw in trusted_media_keywords
        )

        if is_official:
            reliability_score = 0.6
            reason = "공식 도메인 출처 확인 (임베딩 없이 도메인 기반 판정)"
        elif is_trusted_media:
            reliability_score = 0.4
            reason = "신뢰 언론사 도메인 확인 (임베딩 없이 도메인 기반 판정)"
        else:
            reliability_score = 0.2
            reason = "도메인 기반 신뢰성 판정 불가 (임베딩 모델 미로드)"

        enriched = {**article}
        enriched["cluster_id"] = idx  # 단독 클러스터
        enriched["cluster_size"] = 1
        enriched["unique_sources"] = 1
        enriched["source_list"] = [article.get("source", "unknown")]
        enriched["has_official_domain"] = is_official
        enriched["avg_similarity"] = 0.0
        enriched["reliability_score"] = reliability_score
        enriched["reliability_reason"] = reason
        result.append(enriched)

    return result


# ────────────────────────────────────────────────────────────────────────────────
# 최종 판정 함수
# ────────────────────────────────────────────────────────────────────────────────

def determine_final_verdict(
    official_score: float,
    reliability_score: float
) -> dict:
    """
    공식성 점수와 신뢰성 점수를 종합하여 최종 판정 결과를 반환한다.

    판정 기준:
      ✅ 오피셜 검증됨    : official_score >= 0.5 AND reliability_score >= 0.4
      ⚠️ 공식성 있음     : official_score >= 0.5 AND reliability_score < 0.4
      ⚠️ 교차 검증됨     : official_score < 0.5  AND reliability_score >= 0.4
      ❌ 검증 불가       : official_score < 0.5  AND reliability_score < 0.4

    Args:
        official_score (float): 앙상블 공식성 점수 (0.0 ~ 1.0)
        reliability_score (float): 교차 검증 신뢰성 점수 (0.0 ~ 1.0)

    Returns:
        dict:
            - verdict: 판정 결과 문자열
            - verdict_emoji: 이모지
            - verdict_reason: 판정 근거
            - is_verified: 최종 오피셜 여부 (bool)
    """
    is_official = official_score >= OFFICIAL_VERIFIED_THRESHOLD
    is_reliable = reliability_score >= RELIABILITY_FINAL_THRESHOLD

    if is_official and is_reliable:
        return {
            "verdict": "오피셜 검증됨",
            "verdict_emoji": "✅",
            "verdict_reason": (
                f"공식성 점수({official_score:.2f}) + "
                f"교차 검증 신뢰성({reliability_score:.2f}) 모두 기준 충족"
            ),
            "is_verified": True,
        }
    elif is_official and not is_reliable:
        return {
            "verdict": "공식 표현 확인 (교차 검증 미흡)",
            "verdict_emoji": "⚠️",
            "verdict_reason": (
                f"공식성 점수({official_score:.2f}) 기준 충족 | "
                f"교차 검증 부족 (신뢰성={reliability_score:.2f})"
            ),
            "is_verified": False,
        }
    elif not is_official and is_reliable:
        return {
            "verdict": "교차 보도 확인 (공식 표현 미흡)",
            "verdict_emoji": "⚠️",
            "verdict_reason": (
                f"교차 검증 신뢰성({reliability_score:.2f}) 기준 충족 | "
                f"공식 표현 부족 (공식성={official_score:.2f})"
            ),
            "is_verified": False,
        }
    else:
        return {
            "verdict": "검증 불가",
            "verdict_emoji": "❌",
            "verdict_reason": (
                f"공식성({official_score:.2f})과 "
                f"신뢰성({reliability_score:.2f}) 모두 기준 미달"
            ),
            "is_verified": False,
        }


# ────────────────────────────────────────────────────────────────────────────────
# 클러스터 기반 중복 필터링 (v5 신규)
#
# [왜 필요한가]
# SK증권 업무협약이 14개 출처로 반복 노출되면
# 검색 결과 상위를 같은 사건이 독점하여 다양성이 사라짐.
# 같은 cluster_id 내에서 대표 1건만 남기면 다양한 사건을 골고루 볼 수 있음.
#
# [대표 기사 선정 기준]
# 1. 공식 도메인 포함 기사 (.go.kr, yna.co.kr 등) 우선
# 2. 도메인 등급이 높은 기사 우선
# 3. 제목이 긴 기사 우선 (정보량이 더 많은 경향)
# ────────────────────────────────────────────────────────────────────────────────

# 도메인 등급 점수 (대표 기사 선정 시 우선순위 판단용)
_DOMAIN_PRIORITY = {
    ".go.kr": 10, ".or.kr": 10, ".re.kr": 10, ".ac.kr": 10,
    "yna.co.kr": 8, "yonhapnewstv.co.kr": 8,
    "chosun.com": 6, "joongang.co.kr": 6, "donga.com": 6,
    "hani.co.kr": 6, "khan.co.kr": 6,
    "hankyung.com": 5, "mk.co.kr": 5, "edaily.co.kr": 5,
    "kbs.co.kr": 4, "mbc.co.kr": 4, "sbs.co.kr": 4,
    "ytn.co.kr": 4, "jtbc.co.kr": 4,
}


def _get_domain_priority(article: dict) -> int:
    """기사의 도메인 우선순위 점수를 반환한다. 높을수록 대표 기사로 선정."""
    domain = article.get("domain", "")
    if not domain:
        return 0
    for keyword, priority in _DOMAIN_PRIORITY.items():
        if keyword in domain:
            return priority
    return 1


def _deduplicate_by_cluster(articles: list[dict]) -> list[dict]:
    """
    같은 cluster_id를 가진 기사들 중 대표 1건만 남기고 나머지를 제거한다.
    제거된 기사들은 대표 기사의 related_articles 필드에 보존한다.

    [v5 변경] related_articles 추가
      교차 보도 클릭 시 같은 사건을 다룬 다른 뉴스의 제목+출처+링크를
      UI에서 보여주기 위해, 제거 대상 기사 정보를 대표 기사에 저장한다.

    단독 기사(cluster_size=1)는 무조건 유지한다.

    Args:
        articles: cluster_id가 부여된 기사 리스트
    Returns:
        중복이 제거된 기사 리스트 (각 대표 기사에 related_articles 포함)
    """
    if not articles:
        return articles

    # cluster_id별로 기사 그룹핑
    cluster_groups = {}
    for article in articles:
        cluster_id = article.get("cluster_id", -1)
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(article)

    deduplicated = []

    for cluster_id, group in cluster_groups.items():
        if len(group) == 1:
            # 단독 기사: related_articles 빈 리스트
            group[0]["related_articles"] = []
            deduplicated.append(group[0])
            continue

        # 대표 기사 선정: 공식 도메인 우선 → 도메인 등급 우선 → 제목 길이 우선
        group.sort(
            key=lambda a: (
                1 if a.get("has_official_domain", False) else 0,
                _get_domain_priority(a),
                len(a.get("title", "")),
            ),
            reverse=True,
        )

        # 대표 기사 1건만 유지
        representative = group[0]

        # 나머지 기사들을 related_articles에 보존
        # 제목, 출처, 링크, 도메인만 저장 (전체 데이터는 불필요)
        related = []
        for other in group[1:]:
            related.append({
                "title": other.get("title", ""),
                "source": other.get("source", ""),
                "originallink": other.get("originallink", ""),
                "domain": other.get("domain", ""),
            })

        representative["related_articles"] = related
        deduplicated.append(representative)

        logger.debug(
            f"클러스터 {cluster_id} | "
            f"{len(group)}건 → 대표: {representative.get('source', '')} | "
            f"관련 기사 {len(related)}건 보존 | "
            f"'{representative.get('title', '')[:30]}'"
        )

    return deduplicated
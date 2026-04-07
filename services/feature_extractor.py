# 특징 추출 담당 : model이 판단할 근거를 만드는 단계

"""
services/feature_extractor.py
──────────────────────────────
전처리된 기사에서 모델 입력에 필요한 특징(feature)을 추출한다.
규칙 기반 / 임베딩 / 분류 모델 각각에 필요한 특징을 한 곳에서 생성한다.
"""

import re
from config import OFFICIAL_DOMAINS, OFFICIAL_EXPRESSIONS, ORGANIZATION_PATTERNS
from logger import get_logger

logger = get_logger("feature_extractor")


def extract_domain_feature(domain: str) -> float:
    """
    도메인이 공식 도메인 목록에 포함되는지 점수화한다.
    
    .go.kr / .or.kr 같은 공공 도메인은 높은 점수 부여.
    
    Args:
        domain: 전처리된 도메인 문자열 (예: mosf.go.kr)
    Returns:
        0.0 ~ 1.0 사이 점수
    """
    if not domain:
        return 0.0
    for official in OFFICIAL_DOMAINS:
        if official in domain:
            return 1.0
    return 0.0


def extract_official_expression_feature(text: str) -> float:
    """
    텍스트에 공식성 표현 키워드가 포함되어 있는지 점수화한다.
    포함된 키워드 수에 비례하여 점수를 높인다. (최대 1.0)
    
    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        0.0 ~ 1.0 사이 점수
    """
    if not text:
        return 0.0
    count = sum(1 for expr in OFFICIAL_EXPRESSIONS if expr in text)
    # 키워드 3개 이상이면 만점
    return min(count / 3.0, 1.0)


def extract_organization_feature(text: str) -> float:
    """
    텍스트에 공공기관명 패턴이 포함되어 있는지 감지한다.
    예: "기획재정부", "국토교통부", "한국은행" 등
    
    Args:
        text: 제목 + 본문 통합 텍스트
    Returns:
        1.0 (감지됨) / 0.0 (감지 안 됨)
    """
    if not text:
        return 0.0
    for pattern in ORGANIZATION_PATTERNS:
        if re.search(pattern, text):
            return 1.0
    return 0.0


def extract_text_length_feature(content: str) -> float:
    """
    본문 길이를 정규화된 점수로 반환한다.
    공식 기사는 일반적으로 길이가 긴 경향이 있음.
    
    기준: 500자 이상이면 만점 (0.0 ~ 1.0)
    
    Args:
        content: 본문 텍스트
    Returns:
        0.0 ~ 1.0 사이 점수
    """
    if not content:
        return 0.0
    length = len(content)
    return min(length / 500.0, 1.0)


def extract_link_feature(originallink: str, link: str) -> float:
    """
    기사 원문 링크와 네이버 링크의 도메인이 다른지 확인한다.
    
    원문 링크가 네이버가 아닌 외부 도메인이면 직접 취재 기사일 가능성이 높음.
    
    Args:
        originallink: 기사 원문 URL
        link:         네이버 뉴스 링크
    Returns:
        1.0 (외부 원문 있음) / 0.0 (네이버 링크만 있음)
    """
    if not originallink:
        return 0.0
    if "naver.com" not in originallink:
        return 1.0
    return 0.0


def extract_features(article: dict) -> dict:
    """
    전처리된 단일 기사에서 모든 특징을 추출한다.
    규칙 기반 스코어러와 모델 입력에 모두 사용된다.
    
    Args:
        article: 전처리된 기사 딕셔너리
    Returns:
        특징 딕셔너리 (각 특징값 포함)
    """
    try:
        title = article.get("title", "")
        content = article.get("content", "")
        domain = article.get("domain", "")
        originallink = article.get("originallink", "")
        link = article.get("link", "")

        # 제목 + 본문 통합 텍스트 (여러 특징 추출에 사용)
        full_text = f"{title} {content}"

        features = {
            # 도메인 공식성 점수
            "domain_score": extract_domain_feature(domain),

            # 공식 표현 키워드 점수
            "official_expr_score": extract_official_expression_feature(full_text),

            # 기관명 포함 여부 점수
            "org_name_score": extract_organization_feature(full_text),

            # 본문 길이 점수
            "text_length_score": extract_text_length_feature(content),

            # 외부 원문 링크 존재 여부
            "link_score": extract_link_feature(originallink, link),

            # 임베딩 모델 입력용 텍스트 (semantic_similarity에서 사용)
            "embedding_input": full_text,

            # 분류 모델 입력용 텍스트 (classifier_model에서 사용)
            "classifier_input": full_text,
        }
        return features

    except Exception as e:
        logger.error(f"특징 추출 실패: {e} | article={article.get('title', '')}")
        return {}


def extract_features_batch(articles: list[dict]) -> list[dict]:
    """
    기사 목록 전체에서 특징을 추출한다.
    
    Args:
        articles: 전처리된 기사 리스트
    Returns:
        특징 딕셔너리 리스트
    """
    if not articles:
        logger.warning("특징 추출 입력 데이터가 비어 있음")
        return []

    logger.info(f"특징 추출 시작 | {len(articles)}건")

    results = []
    for article in articles:
        features = extract_features(article)
        if features:
            results.append(features)

    logger.info(f"특징 추출 완료 | {len(results)}건")
    return results
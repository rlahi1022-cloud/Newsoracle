"""
services/ensemble.py
────────────────────────────────────────────────────────────────────────────────
공식성 점수 + 신뢰성 점수를 통합하여 최종 판정을 내리는 앙상블 모듈

[v5 — 설명 가능한 AI]
  1. 각 기사에 explanation 필드 추가
     → "왜 공식인지 / 왜 비공식인지 / 어떤 요소 때문인지" 한글 메시지
  2. score clipping: 각 점수를 0.02~0.98 범위로 클리핑
     → 극단값이 최종 점수를 왜곡하는 것 방지
  3. conditional weighting: classifier 확신도에 따라 가중치 동적 조정
  4. reliability clamp: 0.1~0.95 범위로 제한
     → 너무 낮거나 너무 높은 신뢰도 표시 방지
  5. rule_score가 dict로 변경됨에 따른 인터페이스 호환

  explanation 예시:
    "✅ 공식성 높음 (0.78) — 공식 표현 3건(공식 입장, 밝혔다, 보도자료),
     기관명 탐지(SM엔터테인먼트), 비공식 표현 없음,
     3개 언론사 교차 보도 확인"

    "❌ 공식성 낮음 (0.23) — 공식 표현 미탐지,
     비공식 표현 2건(관계자에 따르면, 것으로 알려졌다),
     단독 보도"
"""

from logger import get_logger
from config import ENSEMBLE_WEIGHTS, OFFICIAL_THRESHOLD, EnsembleConfig
from services.cross_validator import determine_final_verdict

logger = get_logger(__name__)

# ── 점수 클리핑 범위 ──────────────────────────────────────────
# 극단값(0.0, 1.0)이 최종 점수를 지배하는 것을 방지
SCORE_CLIP_MIN = 0.02
SCORE_CLIP_MAX = 0.98

# ── 신뢰도 클램프 범위 ───────────────────────────────────────
# 사용자에게 보여주는 reliability를 현실적 범위로 제한
RELIABILITY_CLAMP_MIN = 0.10
RELIABILITY_CLAMP_MAX = 0.95


def _clip_score(value: float) -> float:
    """점수를 SCORE_CLIP_MIN~SCORE_CLIP_MAX 범위로 클리핑한다."""
    return max(SCORE_CLIP_MIN, min(float(value), SCORE_CLIP_MAX))


def _clamp_reliability(value: float) -> float:
    """신뢰도를 RELIABILITY_CLAMP_MIN~RELIABILITY_CLAMP_MAX 범위로 제한한다."""
    return max(RELIABILITY_CLAMP_MIN, min(float(value), RELIABILITY_CLAMP_MAX))


def _build_explanation(
    rule_score: float,
    rule_reason: str,
    semantic_score: float,
    classifier_score: float,
    agency_score: float,
    official_score: float,
    score_method: str,
    reliability_score: float,
    reliability_reason: str,
    verification_message: str,
    verdict: str,
) -> str:
    """
    사용자에게 보여줄 종합 설명 메시지를 생성한다.

    이 함수가 "설명 가능한 AI"의 핵심이다.
    각 모듈의 판단 근거를 하나의 읽기 쉬운 문장으로 조합한다.

    Args:
        rule_score:            규칙 기반 점수
        rule_reason:           규칙 기반 판단 근거 (rule_based_scorer에서 생성)
        semantic_score:        의미 유사도 점수
        classifier_score:      분류 모델 점수
        agency_score:          기관 검증 점수
        official_score:        최종 공식성 점수
        score_method:          계산 경로 (classifier_included/excluded)
        reliability_score:     교차 보도 신뢰성 점수
        reliability_reason:    교차 보도 판정 근거
        verification_message:  기관 검증 메시지
        verdict:               최종 판정 문자열
    Returns:
        종합 설명 메시지 문자열
    """
    parts = []

    # ── 1. 최종 판정 + 점수 ──────────────────────────
    if official_score >= EnsembleConfig.OFFICIAL_SCORE_THRESHOLD:
        parts.append(f"✅ 공식성 높음 ({official_score:.2f})")
    else:
        parts.append(f"❌ 공식성 낮음 ({official_score:.2f})")

    # ── 2. 규칙 기반 근거 (가장 핵심) ────────────────
    if rule_reason:
        parts.append(f"[규칙] {rule_reason}")

    # ── 3. 기관 검증 결과 ────────────────────────────
    if verification_message:
        parts.append(f"[기관] {verification_message}")

    # ── 4. 교차 보도 결과 ────────────────────────────
    if reliability_reason:
        parts.append(f"[교차보도] {reliability_reason}")

    # ── 5. 모델 판단 ────────────────────────────────
    if score_method == "classifier_excluded":
        parts.append(f"[모델] 분류기 저신뢰({classifier_score:.2f}) → 규칙+기관 기반 판정")
    else:
        if classifier_score >= 0.7:
            parts.append(f"[모델] 분류기 공식 판정({classifier_score:.2f})")
        elif classifier_score <= 0.3:
            parts.append(f"[모델] 분류기 비공식 판정({classifier_score:.2f})")
        else:
            parts.append(f"[모델] 분류기 불확실({classifier_score:.2f})")

    # ── 6. 의미 유사도 (보조) ────────────────────────
    if semantic_score >= 0.5:
        parts.append(f"[유사도] 공식 문체 유사({semantic_score:.2f})")
    elif semantic_score <= 0.2:
        parts.append(f"[유사도] 공식 문체 비유사({semantic_score:.2f})")

    return " | ".join(parts)


# ────────────────────────────────────────────────────────────────────────────────
# 단일 기사 앙상블
# ────────────────────────────────────────────────────────────────────────────────

def ensemble_single(
    article: dict,
    rule_result: dict,
    semantic_score: float,
    classifier_score: float,
    agency_score: float = 0.0,
    reliability_score: float = 0.0,
    reliability_reason: str = "",
    verification_message: str = "",
) -> dict:
    """
    단일 기사에 대해 공식성 점수와 설명 메시지를 포함한 최종 판정을 반환한다.

    v5 변경:
    - rule_result가 dict ({"rule_score": float, "rule_reason": str})
    - score clipping 적용
    - conditional weighting 적용
    - explanation 필드 추가
    - reliability clamp 적용

    Args:
        article:              전처리된 기사 딕셔너리
        rule_result:          규칙 기반 결과 {"rule_score": float, "rule_reason": str}
        semantic_score:       의미 유사도 점수
        classifier_score:     분류 모델 점수
        agency_score:         기관 검증 점수
        reliability_score:    교차 보도 신뢰성 점수
        reliability_reason:   교차 보도 판정 근거
        verification_message: 기관 검증 메시지
    Returns:
        최종 결과 딕셔너리 (explanation 포함)
    """
    # rule_result에서 점수와 근거 분리
    rule_score = float(rule_result.get("rule_score", 0.0))
    rule_reason = rule_result.get("rule_reason", "")

    # ── 점수 클리핑 ──────────────────────────────────────────
    rule_score_clipped = _clip_score(rule_score)
    semantic_score_clipped = _clip_score(semantic_score)
    classifier_score_clipped = _clip_score(classifier_score)

    # ── 공식성 점수 계산 (conditional weighting) ──────────────
    if classifier_score_clipped >= EnsembleConfig.CLASSIFIER_LOW_CONFIDENCE:
        # classifier가 확신 있음 → 기존 가중 합산
        official_score = (
            rule_score_clipped * EnsembleConfig.RULE_WEIGHT
            + semantic_score_clipped * EnsembleConfig.SEMANTIC_WEIGHT
            + classifier_score_clipped * EnsembleConfig.CLASSIFIER_WEIGHT
        )
        score_method = "classifier_included"
    else:
        # classifier가 확신 없음 → classifier 제외하고 재계산
        official_score = (
            rule_score_clipped * EnsembleConfig.FALLBACK_RULE_WEIGHT
            + semantic_score_clipped * EnsembleConfig.FALLBACK_SEMANTIC_WEIGHT
            + agency_score * EnsembleConfig.FALLBACK_AGENCY_WEIGHT
        )
        score_method = "classifier_excluded"
        logger.debug(
            f"classifier 저신뢰({classifier_score:.4f}) → "
            f"rule+semantic+agency 기반 재계산 | "
            f"title={article.get('title', '')[:30]}"
        )

    # agency 보너스 (classifier 포함 경로에서만 적용)
    if score_method == "classifier_included" and agency_score > 0:
        agency_bonus = agency_score * EnsembleConfig.AGENCY_BONUS_MAX
        official_score = min(official_score + agency_bonus, 1.0)

    official_score = round(float(official_score), 4)

    # ── 신뢰도 클램프 ────────────────────────────────────────
    reliability_score = _clamp_reliability(reliability_score)

    # ── 최종 판정 ────────────────────────────────────────────
    verdict_result = determine_final_verdict(
        official_score=official_score,
        reliability_score=reliability_score
    )

    # ── 설명 메시지 생성 (v5 핵심) ───────────────────────────
    explanation = _build_explanation(
        rule_score=rule_score,
        rule_reason=rule_reason,
        semantic_score=semantic_score,
        classifier_score=classifier_score,
        agency_score=agency_score,
        official_score=official_score,
        score_method=score_method,
        reliability_score=reliability_score,
        reliability_reason=reliability_reason,
        verification_message=verification_message,
        verdict=verdict_result["verdict"],
    )

    # ── 결과 딕셔너리 구성 ───────────────────────────────────
    result = {
        # 기사 기본 정보
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "originallink": article.get("originallink", ""),
        "pubDate": article.get("pubDate", ""),
        "domain": article.get("domain", ""),

        # 공식성 세부 점수
        "rule_score": round(float(rule_score), 4),
        "rule_reason": rule_reason,
        "semantic_score": round(float(semantic_score), 4),
        "classifier_score": round(float(classifier_score), 4),
        "agency_score": round(float(agency_score), 4),
        "official_score": official_score,
        "score_method": score_method,

        # 신뢰성 세부 정보
        "reliability_score": round(float(reliability_score), 4),
        "reliability_reason": reliability_reason,
        "cluster_id": article.get("cluster_id", -1),
        "cluster_size": article.get("cluster_size", 1),
        "unique_sources": article.get("unique_sources", 1),
        "has_official_domain": article.get("has_official_domain", False),

        # 최종 판정
        "verdict": verdict_result["verdict"],
        "verdict_emoji": verdict_result["verdict_emoji"],
        "verdict_reason": verdict_result["verdict_reason"],
        "is_verified": verdict_result["is_verified"],

        # ── 설명 가능한 AI 핵심 필드 (v5 신규) ────────────────
        "explanation": explanation,
        "verification_message": verification_message,

        # 하위 호환성
        "final_official_score": official_score,
        "predicted_label": verdict_result["verdict"],
    }

    logger.debug(
        f"앙상블 완료 | "
        f"title={article.get('title', '')[:20]} | "
        f"official={official_score:.4f} ({score_method}) | "
        f"reliability={reliability_score:.4f} | "
        f"verdict={verdict_result['verdict']}"
    )

    return result


# ────────────────────────────────────────────────────────────────────────────────
# 배치 앙상블
# ────────────────────────────────────────────────────────────────────────────────

def ensemble_batch(
    articles: list[dict],
    rule_scores: list,
    semantic_scores: list[float],
    classifier_results: list[dict],
    agency_results: list[dict] = None,
) -> list[dict]:
    """
    기사 목록 전체에 앙상블을 적용하여 최종 판정 결과 목록을 반환한다.

    v5 변경:
    - rule_scores가 dict 리스트 (rule_score + rule_reason) 또는 float 리스트
    - 둘 다 하위 호환 지원

    Args:
        articles:           전처리된 기사 딕셔너리 리스트
        rule_scores:        규칙 기반 결과 리스트 (dict 또는 float)
        semantic_scores:    의미 유사도 점수 리스트
        classifier_results: 분류 모델 결과 리스트
        agency_results:     기관 검증 결과 리스트
    Returns:
        최종 결과 딕셔너리 리스트 (explanation 포함)
    """
    if not articles:
        logger.warning("앙상블 입력 기사 목록이 비어있음")
        return []

    n = len(articles)
    logger.info(f"[앙상블] 시작 | {n}건")

    # 길이 불일치 방어
    rule_scores = _pad_list(rule_scores, n, {"rule_score": 0.0, "rule_reason": ""})
    semantic_scores = _pad_list(semantic_scores, n, 0.0)
    classifier_results = _pad_list(classifier_results, n, {"score": 0.0})
    agency_results = _pad_list(agency_results or [], n, {"agency_score": 0.0})

    results = []
    classifier_excluded_count = 0

    for i, article in enumerate(articles):
        try:
            # rule_result 처리 (dict 또는 float 호환)
            raw_rule = rule_scores[i]
            if isinstance(raw_rule, dict):
                rule_result = raw_rule
            else:
                # 하위 호환: float이 들어온 경우
                rule_result = {"rule_score": float(raw_rule) if raw_rule else 0.0, "rule_reason": ""}

            semantic_score = float(semantic_scores[i]) if semantic_scores[i] is not None else 0.0

            clf_result = classifier_results[i]
            if isinstance(clf_result, dict):
                classifier_score = float(clf_result.get("classifier_score", clf_result.get("score", 0.0)))
            else:
                classifier_score = float(clf_result) if clf_result is not None else 0.0

            agn_result = agency_results[i]
            if isinstance(agn_result, dict):
                agency_score = float(agn_result.get("agency_score", 0.0))
                verification_message = agn_result.get("verification_message", "")
            else:
                agency_score = float(agn_result) if agn_result is not None else 0.0
                verification_message = ""

            reliability_score = float(article.get("reliability_score", 0.0))
            reliability_reason = article.get("reliability_reason", "")

            result = ensemble_single(
                article=article,
                rule_result=rule_result,
                semantic_score=semantic_score,
                classifier_score=classifier_score,
                agency_score=agency_score,
                reliability_score=reliability_score,
                reliability_reason=reliability_reason,
                verification_message=verification_message,
            )
            results.append(result)

            if result.get("score_method") == "classifier_excluded":
                classifier_excluded_count += 1

        except (TypeError, ValueError, KeyError) as exc:
            logger.warning(f"앙상블 처리 실패 | idx={i} | error={exc}")
            results.append(_default_result(article))

    # 최종 요약 로그
    verified_count = sum(1 for r in results if r.get("is_verified", False))
    avg_official = sum(r.get("official_score", 0) for r in results) / max(n, 1)
    avg_reliability = sum(r.get("reliability_score", 0) for r in results) / max(n, 1)

    logger.info(
        f"[앙상블] 완료 | "
        f"오피셜 검증됨={verified_count}/{n}건 | "
        f"평균 공식성={avg_official:.4f} | "
        f"평균 신뢰성={avg_reliability:.4f} | "
        f"classifier 제외={classifier_excluded_count}건"
    )

    return results


def _pad_list(lst: list, target_len: int, default) -> list:
    """리스트 길이가 target_len보다 짧으면 default로 채워서 반환한다."""
    if lst is None:
        return [default] * target_len
    result = list(lst)
    while len(result) < target_len:
        result.append(default)
    return result


def _default_result(article: dict) -> dict:
    """앙상블 처리 실패 시 반환하는 기본 결과 딕셔너리."""
    return {
        "title": article.get("title", ""),
        "source": article.get("source", ""),
        "originallink": article.get("originallink", ""),
        "pubDate": article.get("pubDate", ""),
        "domain": article.get("domain", ""),
        "rule_score": 0.0,
        "rule_reason": "처리 실패",
        "semantic_score": 0.0,
        "classifier_score": 0.0,
        "agency_score": 0.0,
        "official_score": 0.0,
        "score_method": "error",
        "reliability_score": 0.0,
        "reliability_reason": "처리 실패",
        "cluster_id": -1,
        "cluster_size": 1,
        "unique_sources": 1,
        "has_official_domain": False,
        "verdict": "검증 불가",
        "verdict_emoji": "❌",
        "verdict_reason": "앙상블 처리 중 오류 발생",
        "is_verified": False,
        "explanation": "❌ 판정 실패 — 앙상블 처리 중 오류 발생",
        "verification_message": "",
        "final_official_score": 0.0,
        "predicted_label": "검증 불가",
    }
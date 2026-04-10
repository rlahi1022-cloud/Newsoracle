"""
services/reliability_scorer.py
────────────────────────────────────────────────────────────────────────────────
내부 신뢰성 3축 스코어러 (v8 신규)

하이브리드 신뢰성 시스템의 내부(Internal) 쪽을 담당한다.
기사 자체의 텍스트만으로 산출 가능한 3개 축으로 구성된다.

[3축 구성]
  1. 출처 책임성 (source_accountability)
     - 익명 표현 카운트 (config.ANONYMOUS_EXPRESSIONS)
     - 실명 인용 정규식 카운트 ('"..." OOO OO' 패턴)
     - 실명 많고 익명 적을수록 높음

  2. 검증 가능성 (verifiability)
     - originallink 존재 여부
     - 본문 내 문서 근거 키워드 탐지
       (보도자료/공시/통계청/한국은행/보고서/백서/원문/발표문/입장문/공고/고시)
     - 원문 링크 있고 문서 근거 많을수록 높음

  3. 의도 중립성 (neutrality)
     - 감정·선정 표현 카운트
       (충격/경악/폭발/발칵/난리/분노/황당/어이없/망신/굴욕/참패)
     - 감정 표현 적을수록 높음

[반환 구조]
  {
    "internal_reliability": float (0.0~1.0),
    "breakdown": {
      "source_accountability": float,
      "verifiability": float,
      "neutrality": float,
    },
    "details": {
      "anonymous_count": int,
      "named_quote_count": int,
      "has_originallink": bool,
      "evidence_count": int,
      "emotion_count": int,
      "evidence_hits": list[str],
      "emotion_hits": list[str],
    }
  }

가중치는 config.EnsembleConfig에서 관리한다:
  - SOURCE_ACCOUNTABILITY_WEIGHT (기본 1/3)
  - VERIFIABILITY_WEIGHT          (기본 1/3)
  - NEUTRALITY_WEIGHT             (기본 1/3)
"""

import re
from logger import get_logger
from config import ANONYMOUS_EXPRESSIONS, EnsembleConfig

# 모듈 전용 로거 (timestamp + level 포함)
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────
# 문서 근거 키워드 (검증 가능성 축)
#
# 본문에 이 키워드가 등장하면 기사가 실제 문서/공식 자료에 근거한다고 판단한다.
# 공식 발표 기반 기사일수록 이 키워드가 많이 등장한다.
# ─────────────────────────────────────────────────────────────

EVIDENCE_KEYWORDS = [
    "보도자료", "공시", "통계청", "한국은행",
    "보고서", "백서", "원문", "발표문",
    "입장문", "공고", "고시",
]

# ─────────────────────────────────────────────────────────────
# 감정·선정 표현 키워드 (의도 중립성 축)
#
# 본문에 이 키워드가 많을수록 자극적 보도로 판단하여 중립성을 감점한다.
# 공식 발표 기반 기사는 일반적으로 이런 표현을 쓰지 않는다.
# ─────────────────────────────────────────────────────────────

EMOTION_KEYWORDS = [
    "충격", "경악", "폭발", "발칵", "난리",
    "분노", "황당", "어이없", "망신", "굴욕", "참패",
]

# ─────────────────────────────────────────────────────────────
# 실명 인용 정규식 패턴
#
# 공식 발표 기사는 "..." OOO 장관, "..." OOO 대표 처럼
# 따옴표 뒤에 실명 + 직함이 따라오는 패턴이 많다.
# 이 패턴 매치 수가 많을수록 출처 책임성이 높다.
#
# 패턴 설명:
#   " 또는 " (일반/전각 쌍따옴표) 로 닫힌 인용
#   → 공백 → 2~4자 한글 이름 → 공백 → 1~6자 한글 직함
#
# 직함 예: 장관, 차관, 대표, 사장, 부장, 팀장, 대변인, 위원장, 회장 등
# ─────────────────────────────────────────────────────────────

NAMED_QUOTE_PATTERN = re.compile(
    r'["”』」]\s*[가-힣]{2,4}\s+[가-힣]{1,6}(?:장관|차관|대표|사장|'
    r'부장|팀장|대변인|위원장|회장|청장|처장|실장|본부장|'
    r'의원|총리|대통령|감독|코치|교수|박사|연구원|변호사)'
)


def _count_anonymous(text: str) -> int:
    """
    본문 내 익명 표현 등장 횟수를 센다.

    config.ANONYMOUS_EXPRESSIONS 목록을 순회하면서
    각 표현이 본문에 몇 번 등장하는지 누적한다.
    """
    if not text:
        return 0
    count = 0
    for expr in ANONYMOUS_EXPRESSIONS:
        count += text.count(expr)
    return count


def _count_named_quotes(text: str) -> int:
    """
    본문 내 실명 인용 패턴 매치 수를 센다.

    NAMED_QUOTE_PATTERN을 사용하여 '"..." 홍길동 장관' 형태를 탐지한다.
    """
    if not text:
        return 0
    return len(NAMED_QUOTE_PATTERN.findall(text))


def _detect_keywords(text: str, keywords: list) -> list:
    """
    본문에서 주어진 키워드 목록 중 실제로 등장한 것만 반환한다.

    카운트가 아닌 "어떤 키워드가 맞았는지" 목록을 반환하여
    UI에서 근거를 표시할 수 있도록 한다.
    """
    if not text:
        return []
    hits = []
    for kw in keywords:
        if kw in text:
            hits.append(kw)
    return hits


def _compute_source_accountability(
    anonymous_count: int, named_quote_count: int
) -> float:
    """
    출처 책임성 점수를 계산한다 (0.0 ~ 1.0).

    로직:
      - 실명 인용이 많을수록 +
      - 익명 표현이 많을수록 -
      - 기본값 0.5에서 시작하여 각 요소가 가감

    공식:
      score = 0.5 + (named * 0.10) - (anonymous * 0.08)
      → 0.0 ~ 1.0 범위로 클리핑
    """
    score = 0.5 + (named_quote_count * 0.10) - (anonymous_count * 0.08)
    return max(0.0, min(score, 1.0))


def _compute_verifiability(
    has_originallink: bool, evidence_count: int
) -> float:
    """
    검증 가능성 점수를 계산한다 (0.0 ~ 1.0).

    로직:
      - originallink 있으면 기본 0.5, 없으면 0.2
      - 문서 근거 키워드 1건당 +0.12 (최대 0.5 가산)

    이유:
      원문 링크가 없으면 독자가 직접 확인할 수 없어 검증 불가능하다.
      문서 근거 키워드가 많을수록 근거 문서가 실존할 가능성이 높다.
    """
    base = 0.5 if has_originallink else 0.2
    bonus = min(evidence_count * 0.12, 0.5)
    return max(0.0, min(base + bonus, 1.0))


def _compute_neutrality(emotion_count: int) -> float:
    """
    의도 중립성 점수를 계산한다 (0.0 ~ 1.0).

    로직:
      - 기본값 1.0에서 시작
      - 감정 표현 1건당 -0.15 감점

    이유:
      공식 발표 기반 기사는 감정 표현을 거의 쓰지 않는다.
      자극적 표현이 많을수록 의도 중립성이 낮다고 본다.
    """
    score = 1.0 - (emotion_count * 0.15)
    return max(0.0, min(score, 1.0))


def compute_internal_reliability(
    title: str, content: str, originallink: str = ""
) -> dict:
    """
    내부 신뢰성 3축 점수를 종합하여 반환한다.

    Args:
        title: 기사 제목
        content: 기사 본문
        originallink: 원문 링크 URL (빈 문자열이면 없는 것으로 간주)

    Returns:
        {
            "internal_reliability": float,
            "breakdown": {
                "source_accountability": float,
                "verifiability": float,
                "neutrality": float,
            },
            "details": {...}
        }

    예외 처리:
        입력이 None이거나 분석 실패 시 기본값(0.5)과 빈 details를 반환한다.
    """
    try:
        # ── 입력 방어 ─────────────────────────────────────────
        safe_title = title if isinstance(title, str) else ""
        safe_content = content if isinstance(content, str) else ""
        safe_link = originallink if isinstance(originallink, str) else ""
        # 제목 + 본문을 합쳐서 분석 (제목에만 있는 감정표현도 탐지하기 위함)
        combined_text = safe_title + " " + safe_content

        # ── 축 1: 출처 책임성 ─────────────────────────────────
        anonymous_count = _count_anonymous(combined_text)
        named_quote_count = _count_named_quotes(combined_text)
        source_accountability = _compute_source_accountability(
            anonymous_count, named_quote_count
        )

        # ── 축 2: 검증 가능성 ─────────────────────────────────
        has_originallink = bool(safe_link and safe_link.strip())
        evidence_hits = _detect_keywords(combined_text, EVIDENCE_KEYWORDS)
        evidence_count = len(evidence_hits)
        verifiability = _compute_verifiability(has_originallink, evidence_count)

        # ── 축 3: 의도 중립성 ─────────────────────────────────
        emotion_hits = _detect_keywords(combined_text, EMOTION_KEYWORDS)
        emotion_count = len(emotion_hits)
        neutrality = _compute_neutrality(emotion_count)

        # ── 가중 평균 ─────────────────────────────────────────
        # config.EnsembleConfig의 3축 가중치 사용 (기본 각 1/3)
        w_sa = EnsembleConfig.SOURCE_ACCOUNTABILITY_WEIGHT
        w_vr = EnsembleConfig.VERIFIABILITY_WEIGHT
        w_nt = EnsembleConfig.NEUTRALITY_WEIGHT

        internal_reliability = (
            source_accountability * w_sa
            + verifiability * w_vr
            + neutrality * w_nt
        )
        internal_reliability = round(
            max(0.0, min(internal_reliability, 1.0)), 4
        )

        result = {
            "internal_reliability": internal_reliability,
            "breakdown": {
                "source_accountability": round(source_accountability, 4),
                "verifiability": round(verifiability, 4),
                "neutrality": round(neutrality, 4),
            },
            "details": {
                "anonymous_count": anonymous_count,
                "named_quote_count": named_quote_count,
                "has_originallink": has_originallink,
                "evidence_count": evidence_count,
                "emotion_count": emotion_count,
                "evidence_hits": evidence_hits,
                "emotion_hits": emotion_hits,
            },
        }

        logger.debug(
            f"내부 신뢰성 | title={safe_title[:20]} | "
            f"internal={internal_reliability:.4f} | "
            f"sa={source_accountability:.2f} "
            f"vr={verifiability:.2f} nt={neutrality:.2f} | "
            f"anon={anonymous_count} named={named_quote_count} "
            f"evid={evidence_count} emo={emotion_count}"
        )

        return result

    except (TypeError, ValueError, AttributeError) as exc:
        # 입력 타입 이상 / 속성 없음 등 예측 가능한 예외
        logger.warning(
            f"내부 신뢰성 계산 실패 (입력 오류) | error={exc} | "
            f"title={str(title)[:30]}"
        )
        return _default_result()
    except Exception as exc:
        # 예상치 못한 예외 (stack trace 확인 가능하도록 exc_info=True)
        logger.error(
            f"내부 신뢰성 계산 중 예상치 못한 오류 | error={exc}",
            exc_info=True,
        )
        return _default_result()


def _default_result() -> dict:
    """예외 발생 시 반환하는 기본 결과 (중립값 0.5)."""
    return {
        "internal_reliability": 0.5,
        "breakdown": {
            "source_accountability": 0.5,
            "verifiability": 0.5,
            "neutrality": 0.5,
        },
        "details": {
            "anonymous_count": 0,
            "named_quote_count": 0,
            "has_originallink": False,
            "evidence_count": 0,
            "emotion_count": 0,
            "evidence_hits": [],
            "emotion_hits": [],
        },
    }


def compute_internal_reliability_batch(articles: list) -> list:
    """
    기사 목록 전체에 내부 신뢰성 계산을 적용한다.

    Args:
        articles: 각 기사 dict 리스트 (title, content, originallink 포함)

    Returns:
        각 기사에 대한 내부 신뢰성 결과 dict 리스트
    """
    if not articles:
        logger.warning("내부 신뢰성 배치 입력이 비어있음")
        return []

    logger.info(f"[내부 신뢰성] 시작 | {len(articles)}건")
    results = []
    for i, article in enumerate(articles):
        try:
            res = compute_internal_reliability(
                title=article.get("title", ""),
                content=article.get("content", ""),
                originallink=article.get("originallink", ""),
            )
            results.append(res)
        except (KeyError, TypeError) as exc:
            logger.warning(
                f"배치 처리 실패 | idx={i} | error={exc}"
            )
            results.append(_default_result())

    avg = sum(r["internal_reliability"] for r in results) / max(len(results), 1)
    logger.info(
        f"[내부 신뢰성] 완료 | {len(results)}건 | 평균={avg:.4f}"
    )
    return results
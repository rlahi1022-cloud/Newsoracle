"""
config.py
─────────
프로젝트 전체 설정값을 한 곳에서 관리한다.

[변경 이력]
- MAX_TOKEN_LENGTH: 256 → 128
- batch_size: 16 → 8
- EnsembleConfig, FeatureConfig 클래스 추가
- NAVER_API 상수 추가
- REFERENCE_SENTENCES를 JSON에서 로드
- KEYWORD_FILTER 설정 추가
- OFFICIAL_EXPRESSIONS 보강
- OFFICIAL_DOMAINS에 주요 언론사 추가
- EnsembleConfig 가중치 재조정 (v4)
  기존: classifier=0.70, rule=0.20, semantic=0.10
  변경: classifier=0.40, rule=0.30, semantic=0.10, agency_bonus=0.20
  이유: classifier가 연예 분야를 전부 비공식 처리하여 다른 피처를 압도
  해결: classifier 비중을 낮추고 rule+agency로 보정 가능하게 함
- CLASSIFIER_LOW_CONFIDENCE 임계값 추가
  이유: classifier가 0.5 미만이면 확신이 없다는 뜻이므로
        rule+agency 기반으로 공식성을 재계산하는 분기 로직에 사용
- RELIABILITY_SCORE_THRESHOLD 0.30 → 0.50 상향
  이유: 단독 보도(신뢰성 0.49)가 오피셜로 통과하는 문제
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────

BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR     = os.path.join(BASE_DIR, "models", "saved")
RESULT_OUTPUT_DIR  = os.path.join(BASE_DIR, "results")
DEFAULT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_data.csv")
OOD_TEST_PATH = os.path.join(BASE_DIR, "data", "ood_test.csv")
REFERENCE_SENTENCES_PATH = os.path.join(BASE_DIR, "data", "reference_sentences.json")

# ─────────────────────────────────────────────────────────────
# 네이버 API 설정
# ─────────────────────────────────────────────────────────────

NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"
NAVER_API_TIMEOUT = 5
NAVER_API_HEADER_ID_KEY     = "X-Naver-Client-Id"
NAVER_API_HEADER_SECRET_KEY = "X-Naver-Client-Secret"

NEWS_DISPLAY_PER_REQUEST = 100
NEWS_MAX_PAGES            = 3
NEWS_MAX_TOTAL            = 300
NEWS_DEFAULT_SORT         = "date"

# ─────────────────────────────────────────────────────────────
# 키워드 필터링 설정
# ─────────────────────────────────────────────────────────────

KEYWORD_FILTER_ENABLED = True
KEYWORD_FILTER_MIN_MATCH = 1

# ─────────────────────────────────────────────────────────────
# 분류 모델 설정
# ─────────────────────────────────────────────────────────────

CLASSIFIER_MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"
EMBEDDING_MODEL_NAME  = "jhgan/ko-sroberta-multitask"
NUM_LABELS            = 2
MAX_TOKEN_LENGTH = 128

# ─────────────────────────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────────────────────────

TRAIN_CONFIG = {
    "epochs":        10,
    "batch_size":    8,
    "learning_rate": 2e-5,
    "val_ratio":     0.2,
    "seed":          42,
    "hidden_dropout_prob":          0.2,
    "attention_probs_dropout_prob": 0.2,
    "early_stopping_patience": 3,
    "early_stopping_metric":   "val_loss",
}

# ─────────────────────────────────────────────────────────────
# 앙상블 가중치 (하위 호환성 유지용)
# ─────────────────────────────────────────────────────────────

ENSEMBLE_WEIGHTS = {
    "rule":       0.12,
    "semantic":   0.08,
    "classifier": 0.50,
    "agency":     0.30,
}

OFFICIAL_THRESHOLD = 0.45


# ─────────────────────────────────────────────────────────────
# EnsembleConfig 클래스 (v4)
#
# [v4 변경사항]
# classifier 가중치를 0.70 → 0.40으로 낮춤
# rule 가중치를 0.20 → 0.30으로 올림
# agency 보너스를 0.10 → 0.20으로 올림
#
# 왜 이렇게 바꾸는가:
#   classifier가 연예/스포츠 기사를 전부 비공식(0.001)으로 분류
#   → classifier 0.70이면 다른 피처가 아무리 좋아도 공식성 0.3 미만
#   → [공식] 태그, 소속사 입장문 등을 rule/agency에서 잡아도 무의미
#   → classifier 비중을 낮추고 rule+agency로 보정 가능한 구조로 변경
#
# CLASSIFIER_LOW_CONFIDENCE:
#   classifier 점수가 이 값 미만이면 "확신 없음"으로 간주
#   → 앙상블에서 classifier를 제외하고 rule+semantic+agency로 재계산
#   → 연예 분야처럼 학습 데이터가 부족한 경우에 대한 폴백
# ─────────────────────────────────────────────────────────────

class EnsembleConfig:
    """앙상블 가중치 및 임계값을 클래스 속성으로 관리한다."""

    # ── 기본 가중치 (classifier 확신 있을 때) ─────────────────
    # 합계: 0.30 + 0.10 + 0.40 = 0.80
    # 나머지 0.20은 agency 보너스로 가산
    RULE_WEIGHT       = 0.30
    SEMANTIC_WEIGHT   = 0.10
    CLASSIFIER_WEIGHT = 0.40

    # agency 보너스 최대값 (기존 0.10 → 0.20)
    AGENCY_BONUS_MAX  = 0.20

    # ── classifier 저신뢰 분기 (v4 신규) ─────────────────────
    # classifier 점수가 이 값 미만이면 classifier를 제외하고 재계산
    CLASSIFIER_LOW_CONFIDENCE = 0.5

    # classifier 제외 시 가중치 (합계 1.0)
    # rule과 semantic만으로 공식성을 판단
    FALLBACK_RULE_WEIGHT     = 0.60
    FALLBACK_SEMANTIC_WEIGHT = 0.15
    FALLBACK_AGENCY_WEIGHT   = 0.25

    # ── 최종 판정 임계값 ─────────────────────────────────────
    OFFICIAL_SCORE_THRESHOLD    = 0.45
    RELIABILITY_SCORE_THRESHOLD = 0.50   # 0.30 → 0.50 상향 (단독 보도 통과 방지)


# ─────────────────────────────────────────────────────────────
# FeatureConfig 클래스
# ─────────────────────────────────────────────────────────────

class FeatureConfig:
    """특징 추출 관련 설정값을 클래스 속성으로 관리한다."""
    CLUSTER_SIMILARITY_THRESHOLD = 0.25
    MIN_CLUSTER_SIZE = 2
    RELIABILITY_SCORE_PER_SOURCE = 0.2


# ─────────────────────────────────────────────────────────────
# 공식 도메인 목록 (grade 1~3)
# ─────────────────────────────────────────────────────────────

OFFICIAL_DOMAINS = [
    ".go.kr", ".or.kr", ".ac.kr", ".re.kr",
    "yonhapnews.co.kr", "yna.co.kr",
    "newsis.com", "news1.kr",
    "chosun.com", "joongang.co.kr", "donga.com",
    "hani.co.kr", "khan.co.kr", "kmib.co.kr",
]

# ─────────────────────────────────────────────────────────────
# 언론사 도메인 등급별 분류
# ─────────────────────────────────────────────────────────────

MEDIA_DOMAIN_GRADES = {
    "grade_1": {
        "score": 0.7,
        "domains": [".go.kr", ".or.kr", ".ac.kr", ".re.kr"],
        "desc": "정부/공공/학술/연구 기관",
    },
    "grade_2": {
        "score": 0.5,
        "domains": ["yonhapnews.co.kr", "yna.co.kr", "newsis.com", "news1.kr"],
        "desc": "통신사",
    },
    "grade_3": {
        "score": 0.4,
        "domains": [
            "chosun.com", "joongang.co.kr", "donga.com",
            "hani.co.kr", "khan.co.kr", "kmib.co.kr",
        ],
        "desc": "주요 종합일간지",
    },
    "grade_4": {
        "score": 0.35,
        "domains": [
            "hankyung.com", "mk.co.kr", "edaily.co.kr",
            "mt.co.kr", "sedaily.com", "etnews.com",
            "zdnet.co.kr", "bloter.net",
        ],
        "desc": "경제/IT 전문지",
    },
    "grade_5": {
        "score": 0.2,
        "domains": [
            "ytn.co.kr", "mbn.co.kr", "sbs.co.kr",
            "kbs.co.kr", "mbc.co.kr", "jtbc.co.kr",
            "hankookilbo.com", "heraldcorp.com",
            "pressian.com", "ohmynews.com", "newspim.com",
        ],
        "desc": "주요 인터넷/방송 뉴스",
    },
}

# ─────────────────────────────────────────────────────────────
# 공식성 표현 키워드
# ─────────────────────────────────────────────────────────────

OFFICIAL_EXPRESSIONS = [
    "발표", "공식 발표", "공식 입장", "보도자료",
    "공고", "공지", "고시", "성명", "입장문",
    "브리핑", "기자회견",
    "의결", "결정", "승인", "채택", "시행", "발효",
    "업무협약", "MOU", "협약 체결", "파트너십 체결",
    "공시", "실적 발표",
    "공개", "선보여", "출시", "론칭",
    "개최", "주최", "참가", "참여",
    "수상", "선정", "인증",
    "계약 체결", "투자 유치", "인수", "상장", "IPO",
    "측이 밝혔다", "소속사 공식",
    "컴백", "데뷔", "공식 활동",
    "이적", "트레이드", "FA 계약",
    "감독 선임", "코치 선임", "출전 명단", "엔트리 확정",
]

ORGANIZATION_PATTERNS = [
    r"부$", r"처$", r"청$", r"원$", r"위원회$",
    r"공사$", r"공단$", r"은행$", r"연구원$",
    r"자동차$", r"전자$", r"화학$", r"건설$",
]

# ─────────────────────────────────────────────────────────────
# 의미 유사도 레퍼런스 문장 (JSON에서 로드)
# ─────────────────────────────────────────────────────────────


def _load_reference_sentences() -> list:
    """data/reference_sentences.json에서 기준 문장을 로드한다."""
    fallback = [
        "본 건과 관련하여 아래와 같이 알려드립니다.",
        "당사는 이사회 결의를 거쳐 다음과 같이 공시합니다.",
        "소속사입니다. 공식 입장을 전합니다.",
        "대한체육회는 다음과 같이 공지합니다.",
        "당사는 서비스 변경 사항을 다음과 같이 안내드립니다.",
    ]
    try:
        with open(REFERENCE_SENTENCES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"[WARNING] 기준 문장 로드 실패: {e} → 폴백 문장 사용")
        return fallback

    sentences = []
    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list):
            sentences.extend(value)
    return sentences if sentences else fallback


REFERENCE_SENTENCES = _load_reference_sentences()

# ─────────────────────────────────────────────────────────────
# 로그 설정
# ─────────────────────────────────────────────────────────────

LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT      = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
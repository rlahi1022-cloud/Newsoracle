"""
config.py
─────────
프로젝트 전체 설정값을 한 곳에서 관리한다.
모델명, 가중치, 경로, API 파라미터 등을 여기서 수정하면
다른 파일을 건드리지 않아도 된다.

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
- CLASSIFIER_LOW_CONFIDENCE 임계값 추가
- RELIABILITY_SCORE_THRESHOLD 0.30 → 0.50 상향
- [v5] epochs 10 → 5 (과적합 방지, early stopping과 조합)
- [v5] OFFICIAL_EXPRESSIONS 대폭 확대 (34개 → 72개)
  추가 분야: 법률/사법, 의료/보건, 교육/학술, 국제기구, 군/안보
- [v5] ANONYMOUS_EXPRESSIONS 신규 추가 (30개)
  비공식 판별 기준 강화: 익명 소식통, 추측, 루머 표현 패턴
- [v5] ORGANIZATION_PATTERNS 확대 (12개 → 25개)
  소속사, 협회, 재단, 학회, 의료기관 등 추가
- [v5] OFFICIAL_SUBJECT_VERBS 신규 추가
  "~은/는 발표했다" 구조의 공식 발화 패턴 동사 목록
- [v7] EnsembleConfig 전면 재설계
  폴백 경로 제거 → conditional weighting (3구간)
  항상 4개 모델이 참여하되 classifier 신뢰도에 따라 가중치 동적 조절
- [v8] 하이브리드 신뢰성 도입 (내부 3축 + 외부)
  EnsembleConfig에 INTERNAL/EXTERNAL_RELIABILITY_WEIGHT 추가 (각 0.5)
  내부 3축 가중치: SOURCE_ACCOUNTABILITY/VERIFIABILITY/NEUTRALITY (각 1/3)
  v7 conditional weighting 구조는 완전 보존
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────

# 프로젝트 루트 디렉토리 (이 파일이 위치한 디렉토리)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 학습된 모델 저장 디렉토리
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")

# 추론 결과 저장 디렉토리
RESULT_OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# 기본 학습 데이터 경로
DEFAULT_TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_data.csv")

# OOD(Out-of-Distribution) 테스트셋 경로
# 학습/검증 데이터와 완전히 다른 쿼리로 수집한 별도 평가 데이터
# 이 데이터로만 실제 추론 환경에서의 성능을 측정할 수 있음
OOD_TEST_PATH = os.path.join(BASE_DIR, "data", "ood_test.csv")

# 의미 유사도 기준 문장 JSON 파일 경로
REFERENCE_SENTENCES_PATH = os.path.join(BASE_DIR, "data", "reference_sentences.json")

# ─────────────────────────────────────────────────────────────
# 네이버 API 설정
# ─────────────────────────────────────────────────────────────

# API 엔드포인트 URL
NAVER_API_URL = "https://openapi.naver.com/v1/search/news.json"

# 요청 타임아웃 (초)
NAVER_API_TIMEOUT = 5

# HTTP 헤더 키 이름
NAVER_API_HEADER_ID_KEY = "X-Naver-Client-Id"
NAVER_API_HEADER_SECRET_KEY = "X-Naver-Client-Secret"

# 1회 요청당 수집 건수 (네이버 API 최대값 100)
NEWS_DISPLAY_PER_REQUEST = 100

# 페이지네이션 최대 페이지 수
NEWS_MAX_PAGES = 3

# 총 최대 수집 건수
NEWS_MAX_TOTAL = 300

# 기본 정렬 기준
NEWS_DEFAULT_SORT = "date"

# ─────────────────────────────────────────────────────────────
# 키워드 필터링 설정
# ─────────────────────────────────────────────────────────────

# 키워드 필터링 활성화 여부
KEYWORD_FILTER_ENABLED = True

# 최소 키워드 매치 수
KEYWORD_FILTER_MIN_MATCH = 1

# ─────────────────────────────────────────────────────────────
# 분류 모델 설정
# ─────────────────────────────────────────────────────────────

# KR-ELECTRA: 한국어 특화 사전학습 모델
# BERT보다 적은 데이터로도 높은 성능을 내는 discriminator 방식
CLASSIFIER_MODEL_NAME = "snunlp/KR-ELECTRA-discriminator"

# ko-sroberta: 한국어 문장 임베딩 모델
# SentenceTransformers 기반, cosine similarity 계산용
EMBEDDING_MODEL_NAME = "jhgan/ko-sroberta-multitask"

# 이진 분류 (공식=1, 비공식=0)
NUM_LABELS = 2

# 토큰 최대 길이: 256 → 128로 변경
# 이유: CPU 환경에서 토큰 길이가 속도에 직접 영향을 미침
# 뉴스 기사는 앞부분에 핵심 정보가 집중되므로 128토큰으로 충분
MAX_TOKEN_LENGTH = 128

# ─────────────────────────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────────────────────────

TRAIN_CONFIG = {
    # [v5] epochs: 10 → 5
    # 이유: 이전 학습에서 epoch 2~3에서 이미 과적합 징후 발생
    # early stopping(patience=3)과 조합하면 최대 5까지만 허용해도 충분
    # 실질적으로 epoch 2~4 사이에서 중단될 것으로 예상
    "epochs": 5,

    # batch_size: 16 → 8
    # 이유: CPU 환경에서 배치 크기가 메모리 및 속도에 직접 영향
    "batch_size": 8,

    # 학습률: BERT 계열 권장값
    "learning_rate": 2e-5,

    # 검증 데이터 비율 (전체의 20%)
    "val_ratio": 0.2,

    # 재현성을 위한 랜덤 시드
    "seed": 42,

    # Dropout 정규화: ELECTRA 기본값 0.1 → 0.2로 올려 과적합 억제
    # 학습 시 뉴런을 20% 확률로 꺼서 특정 패턴 암기를 방지
    "hidden_dropout_prob": 0.2,
    "attention_probs_dropout_prob": 0.2,

    # Early Stopping: val_loss가 3 epoch 연속 개선 없으면 중단
    # val_f1이 아닌 val_loss 기준인 이유:
    #   val_f1은 threshold에 따라 왜곡 가능
    #   val_loss는 모델의 확신도를 직접 반영하는 더 순수한 지표
    "early_stopping_patience": 3,
    "early_stopping_metric": "val_loss",
}

# ─────────────────────────────────────────────────────────────
# 앙상블 가중치 (하위 호환성 유지용)
# ─────────────────────────────────────────────────────────────

ENSEMBLE_WEIGHTS = {
    "rule": 0.35,
    "semantic": 0.30,
    "classifier": 0.20,
    "agency": 0.15,
}

OFFICIAL_THRESHOLD = 0.45


# ─────────────────────────────────────────────────────────────
# EnsembleConfig 클래스 (v7)
#
# [v7 전면 재설계]
# 폴백 경로 제거 → conditional weighting (3구간)
#
# 기존 문제:
#   classifier < 0.5이면 classifier를 완전 배제 (폴백 경로)
#   → 55건 중 51건이 폴백 → "4중 앙상블"이라 부를 수 없음
#
# 해결:
#   항상 4개 모델이 참여하되, classifier 신뢰도에 따라
#   가중치를 3구간으로 동적 조절
#
#   구간 1 (LOW): classifier < 0.15 → classifier 거의 무시
#     rule 0.40 + semantic 0.35 + classifier 0.10 + agency 0.15
#     이유: classifier가 거의 확신 없는 상태, 규칙 기반이 주도
#
#   구간 2 (NORMAL): 0.15 <= classifier <= 0.85 → 정상 배분
#     rule 0.35 + semantic 0.30 + classifier 0.20 + agency 0.15
#     이유: classifier가 어느 정도 판단 가능, 4개 균등 참여
#
#   구간 3 (HIGH): classifier > 0.85 → classifier 과신 방지
#     rule 0.30 + semantic 0.25 + classifier 0.25 + agency 0.20
#     이유: classifier가 과도하게 확신하면 다른 모델로 견제
#
# 클리핑:
#   classifier 입력값을 0.05~0.95로 제한하여 극단값 방지
#   0.001이 들어와도 0.05로, 0.999가 들어와도 0.95로 처리
# ─────────────────────────────────────────────────────────────

class EnsembleConfig:
    """앙상블 가중치 및 임계값을 클래스 속성으로 관리한다."""

    # ── classifier 클리핑 범위 ──────────────────────────────
    # 극단값(0.001, 0.999)이 전체 점수를 왜곡하는 것을 방지
    CLASSIFIER_CLIP_MIN = 0.05
    CLASSIFIER_CLIP_MAX = 0.95

    # ── conditional weighting 구간 경계 ─────────────────────
    # classifier_score가 이 범위 밖이면 해당 구간의 가중치 사용
    CLASSIFIER_LOW_BOUNDARY = 0.15
    CLASSIFIER_HIGH_BOUNDARY = 0.85

    # ── 구간 1: LOW (classifier 거의 무시) ──────────────────
    # classifier < 0.15일 때 사용
    # classifier가 거의 확신 없으므로 rule/semantic이 주도
    LOW_RULE_WEIGHT = 0.40
    LOW_SEMANTIC_WEIGHT = 0.35
    LOW_CLASSIFIER_WEIGHT = 0.10
    LOW_AGENCY_WEIGHT = 0.15

    # ── 구간 2: NORMAL (정상 배분) ──────────────────────────
    # 0.15 <= classifier <= 0.85일 때 사용
    # 4개 모델이 균형 있게 참여
    NORMAL_RULE_WEIGHT = 0.35
    NORMAL_SEMANTIC_WEIGHT = 0.30
    NORMAL_CLASSIFIER_WEIGHT = 0.20
    NORMAL_AGENCY_WEIGHT = 0.15

    # ── 구간 3: HIGH (classifier 과신 방지) ─────────────────
    # classifier > 0.85일 때 사용
    # classifier를 약간 올리되, 과신을 막기 위해 agency도 올림
    HIGH_RULE_WEIGHT = 0.30
    HIGH_SEMANTIC_WEIGHT = 0.25
    HIGH_CLASSIFIER_WEIGHT = 0.25
    HIGH_AGENCY_WEIGHT = 0.20

    # ── 최종 판정 임계값 ─────────────────────────────────────
    OFFICIAL_SCORE_THRESHOLD = 0.45
    RELIABILITY_SCORE_THRESHOLD = 0.50

    # ── 하위 호환성 유지용 (ensemble.py v6 이전 코드 호환) ───
    # v7에서는 직접 사용하지 않지만, 다른 파일에서 참조할 수 있으므로 유지
    RULE_WEIGHT = 0.35
    SEMANTIC_WEIGHT = 0.30
    CLASSIFIER_WEIGHT = 0.20
    AGENCY_BONUS_MAX = 0.15
    CLASSIFIER_LOW_CONFIDENCE = 0.15
    FALLBACK_RULE_WEIGHT = 0.40
    FALLBACK_SEMANTIC_WEIGHT = 0.35
    FALLBACK_AGENCY_WEIGHT = 0.15

    # ── [v8] 하이브리드 신뢰성 가중치 ────────────────────────
    # 내부 신뢰성(3축) vs 외부 신뢰성(교차보도) 비율
    INTERNAL_RELIABILITY_WEIGHT = 0.5
    EXTERNAL_RELIABILITY_WEIGHT = 0.5

    # 내부 신뢰성 3축 세부 가중치 (합 = 1.0)
    SOURCE_ACCOUNTABILITY_WEIGHT = 1/3
    VERIFIABILITY_WEIGHT = 1/3
    NEUTRALITY_WEIGHT = 1/3


# ─────────────────────────────────────────────────────────────
# FeatureConfig 클래스
# ─────────────────────────────────────────────────────────────

class FeatureConfig:
    """특징 추출 관련 설정값을 클래스 속성으로 관리한다."""

    # 교차 보도 클러스터링 유사도 임계값
    CLUSTER_SIMILARITY_THRESHOLD = 0.25

    # 최소 클러스터 크기 (이 수 이상이면 교차 보도로 인정)
    MIN_CLUSTER_SIZE = 2

    # 출처 1개당 신뢰도 가산 점수
    RELIABILITY_SCORE_PER_SOURCE = 0.2


# ─────────────────────────────────────────────────────────────
# 공식 도메인 목록 (grade 1~3)
#
# agency_verifier.py, rule_based_scorer.py에서
# 기사 출처 도메인의 공식성을 판단할 때 사용
# ─────────────────────────────────────────────────────────────

OFFICIAL_DOMAINS = [
    # 정부/공공기관
    ".go.kr", ".or.kr", ".ac.kr", ".re.kr",
    # 국가기간뉴스통신사
    "yonhapnews.co.kr", "yna.co.kr",
    # 통신사
    "newsis.com", "news1.kr",
    # 주요 종합일간지
    "chosun.com", "joongang.co.kr", "donga.com",
    "hani.co.kr", "khan.co.kr", "kmib.co.kr",
]

# ─────────────────────────────────────────────────────────────
# 언론사 도메인 등급별 분류
#
# 등급이 높을수록 공식성 base score가 높다.
# rule_based_scorer.py에서 도메인 점수 산출에 사용.
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
# 공식성 표현 키워드 (v5 대폭 확대)
#
# rule_based_scorer.py에서 기사 본문에 이 표현이 포함되어 있으면
# 공식성 점수를 가산한다.
#
# [v5 변경사항]
# 기존 34개 → 72개로 확대
# 추가 분야:
#   - 법률/사법: 판결, 선고, 기소, 구속영장 등
#   - 의료/보건: 허가, 임상, 승인, 접종 등
#   - 교육/학술: 논문 발표, 연구 결과, 학위 수여 등
#   - 국제기구: UN 결의, WHO 권고, IMF 전망 등
#   - 군/안보: 작전 발표, 훈련 실시, 군사 합의 등
#   - 기업 IR: 실적 공시, 배당 결정, 주주총회 의결 등
# ─────────────────────────────────────────────────────────────

OFFICIAL_EXPRESSIONS = [
    # ── 기본 공식 발표 표현 ──────────────────────────────────
    "발표", "공식 발표", "공식 입장", "보도자료",
    "공고", "공지", "고시", "성명", "입장문",
    "브리핑", "기자회견",

    # ── 의사결정/승인 표현 ───────────────────────────────────
    "의결", "결정", "승인", "채택", "시행", "발효",
    "인가", "허가", "비준", "재가",

    # ── 협약/계약 표현 ───────────────────────────────────────
    "업무협약", "MOU", "협약 체결", "파트너십 체결",
    "계약 체결", "투자 유치", "인수", "상장", "IPO",
    "전속계약", "전속 계약",

    # ── 기업 IR / 공시 표현 ──────────────────────────────────
    "공시", "실적 발표", "분기 실적", "영업이익 발표",
    "배당 결정", "주주총회 의결", "이사회 결의",
    "감사보고서", "사업보고서",

    # ── 출시/이벤트 표현 ─────────────────────────────────────
    "공개", "선보여", "출시", "론칭",
    "개최", "주최", "참가", "참여",
    "수상", "선정", "인증",

    # ── 연예/엔터 공식 표현 ──────────────────────────────────
    "측이 밝혔다", "소속사 공식", "소속사 측",
    "컴백", "데뷔", "공식 활동",
    "공식 포스터", "티저 공개", "뮤직비디오 공개",

    # ── 스포츠 공식 표현 ─────────────────────────────────────
    "이적", "트레이드", "FA 계약",
    "감독 선임", "코치 선임", "출전 명단", "엔트리 확정",
    "우승", "준우승", "메달 획득",

    # ── 법률/사법 표현 (v5 신규) ─────────────────────────────
    # 법원/검찰/경찰의 공식 결정은 공식성이 매우 높다
    "판결", "선고", "기소", "구속영장",
    "구속", "불구속", "약식기소", "정식재판",
    "대법원 판결", "헌법재판소 결정",

    # ── 의료/보건 표현 (v5 신규) ─────────────────────────────
    # 식약처/질병관리청/WHO 등의 공식 승인/발표
    "임상시험 결과", "허가 승인", "긴급사용승인",
    "접종 개시", "방역 지침", "격리 해제",
    "진료 지침", "가이드라인 발표",

    # ── 교육/학술 표현 (v5 신규) ─────────────────────────────
    # 대학/학회/연구기관의 공식 발표
    "논문 발표", "연구 결과 발표", "학위 수여",
    "입학 발표", "합격자 발표", "등록금 인상",

    # ── 국제기구 표현 (v5 신규) ──────────────────────────────
    # UN/WHO/IMF 등 국제기구의 공식 결정
    "UN 결의", "WHO 권고", "IMF 전망",
    "정상회담", "외교 합의", "무역 협정",

    # ── 군/안보 표현 (v5 신규) ───────────────────────────────
    # 국방부/합참/군의 공식 발표
    "작전 발표", "훈련 실시", "군사 합의",
    "동원령", "경계 강화",

    # ── [공식] 태그 표현 (v7 신규) ───────────────────────────
    # 뉴스 제목에 [공식], (공식) 태그가 붙은 경우
    # 소속사/기관이 직접 확인한 내용일 가능성이 높음
    "[공식]", "(공식)", "공식 확인",
]

# ─────────────────────────────────────────────────────────────
# 비공식(익명/추측/루머) 표현 키워드 (v5 신규)
#
# rule_based_scorer.py에서 기사 본문에 이 표현이 포함되어 있으면
# 공식성 점수를 감점한다.
#
# 공식성 판단 기준 4번:
#   "익명 소식통 / 추정 / 루머 / 커뮤니티 재가공 중심인가"
#   발화 주체가 불명확하거나 검증이 어려운 표현을 탐지한다.
#
# 카테고리:
#   1. 익명 소식통 (발화 주체 불명확)
#   2. 추측/관측 (검증 불가능한 전망)
#   3. 루머/가십 (커뮤니티 재가공)
#   4. 전언/간접 인용 (출처 불명확)
# ─────────────────────────────────────────────────────────────

ANONYMOUS_EXPRESSIONS = [
    # ── 1. 익명 소식통 (발화 주체 불명확) ────────────────────
    "관계자에 따르면",
    "소식통에 따르면",
    "업계에 따르면",
    "내부 관계자",
    "익명을 요구한",
    "익명의 관계자",
    "복수의 관계자",
    "소식통",
    "측근에 따르면",

    # ── 2. 추측/관측 (검증 불가능한 전망) ────────────────────
    "것으로 알려졌다",
    "것으로 전해졌다",
    "것으로 보인다",
    "것으로 추정된다",
    "것으로 관측된다",
    "가능성이 높다",
    "가능성이 제기",
    "전망이다",
    "추정된다",
    "관측이 나온다",

    # ── 3. 루머/가십 (커뮤니티 재가공) ───────────────────────
    "온라인에서 화제",
    "누리꾼 반응",
    "커뮤니티에서 논란",
    "SNS에서 확산",
    "카더라",
    "찌라시",
    "루머",

    # ── 4. 전언/간접 인용 (출처 불명확) ──────────────────────
    "전해졌다",
    "알려졌다",
    "밝혀졌다",
    "주장했다",
    "주장이 나왔다",
    "주장이 제기",
    "의혹이 제기",
    "논란이 일고",
]

# ─────────────────────────────────────────────────────────────
# 기관명 패턴 (정규표현식) (v5 확대)
#
# feature_extractor.py에서 기사 본문 내 기관명을 탐지할 때 사용.
# 패턴이 매치되면 공식 기관이 언급된 것으로 판단한다.
#
# [v5 변경사항]
# 기존 12개 → 25개로 확대
# 추가: 소속사, 협회, 재단, 학회, 의료기관, 군 조직, 국제기구 등
# ─────────────────────────────────────────────────────────────

ORGANIZATION_PATTERNS = [
    # ── 정부/공공 기관 ───────────────────────────────────────
    r"부$",          # ~부 (기획재정부, 국토교통부)
    r"처$",          # ~처 (식품의약품안전처, 질병관리청→처)
    r"청$",          # ~청 (국세청, 경찰청, 질병관리청)
    r"원$",          # ~원 (감사원, 연구원)
    r"위원회$",      # ~위원회 (공정거래위원회)
    r"공사$",        # ~공사 (한국전력공사, 한국도로공사)
    r"공단$",        # ~공단 (국민연금공단, 국민건강보험공단)

    # ── 금융/경제 기관 ───────────────────────────────────────
    r"은행$",        # ~은행 (한국은행, 산업은행)
    r"보험$",        # ~보험 (국민건강보험)
    r"거래소$",      # ~거래소 (한국거래소)
    r"증권$",        # ~증권 (한국투자증권)

    # ── 기업 ─────────────────────────────────────────────────
    r"자동차$",      # ~자동차 (현대자동차)
    r"전자$",        # ~전자 (삼성전자, LG전자)
    r"화학$",        # ~화학 (LG화학)
    r"건설$",        # ~건설 (현대건설)
    r"그룹$",        # ~그룹 (삼성그룹, SK그룹)

    # ── 연예/엔터 (v5 신규) ──────────────────────────────────
    r"엔터테인먼트$",  # ~엔터테인먼트 (하이브, SM, JYP)
    r"소속사$",       # ~소속사
    r"기획사$",       # ~기획사

    # ── 단체/협회/학회 (v5 신규) ─────────────────────────────
    r"협회$",        # ~협회 (대한축구협회, 대한의사협회)
    r"연구원$",      # ~연구원 (한국개발연구원, 한국과학기술연구원)
    r"재단$",        # ~재단 (삼성미래기술육성재단)
    r"학회$",        # ~학회 (대한내과학회)

    # ── 의료 기관 (v5 신규) ──────────────────────────────────
    r"병원$",        # ~병원 (서울대학교병원, 세브란스병원)
    r"의료원$",      # ~의료원 (고려대학교의료원)
]

# ─────────────────────────────────────────────────────────────
# 공식 발화 주체 + 동사 패턴 (v5 신규)
#
# "~은/는 발표했다", "~이/가 밝혔다" 구조에서
# 공식 기관이 직접 주어로 등장하는 동사 목록.
#
# agency_verifier.py에서 본문 분석 시 사용.
# 기관명 + 이 동사가 조합되면 공식 발화로 판단한다.
#
# 예:
#   "한국은행은 기준금리를 동결하기로 결정했다" → 공식
#   "한국은행에 대한 비판이 제기됐다" → 비공식 (동사 불일치)
# ─────────────────────────────────────────────────────────────

OFFICIAL_SUBJECT_VERBS = [
    # ── 발표/공개 계열 ───────────────────────────────────────
    "발표했다", "발표하였다", "발표한다",
    "밝혔다", "밝혀", "밝힌",
    "공개했다", "공개하였다",
    "알렸다", "알린",

    # ── 결정/의결 계열 ───────────────────────────────────────
    "결정했다", "결정하였다",
    "의결했다", "의결하였다",
    "승인했다", "승인하였다",
    "채택했다", "채택하였다",

    # ── 시행/실시 계열 ───────────────────────────────────────
    "시행한다", "시행했다",
    "실시한다", "실시했다",
    "추진한다", "추진했다",
    "개시했다", "개시한다",

    # ── 입장/성명 계열 ───────────────────────────────────────
    "전했다", "전하였다",
    "설명했다", "설명하였다",
    "강조했다", "강조하였다",
    "당부했다",

    # ── 계약/합의 계열 ───────────────────────────────────────
    "체결했다", "체결하였다",
    "합의했다", "합의하였다",
    "서명했다", "서명하였다",

    # ── 선고/판결 계열 (사법) ────────────────────────────────
    "선고했다", "선고하였다",
    "판결했다", "판결하였다",
    "기소했다",
]

# ─────────────────────────────────────────────────────────────
# 의미 유사도 레퍼런스 문장 (JSON에서 로드)
#
# semantic_similarity.py에서 기사 텍스트와 이 기준 문장들 사이의
# cosine similarity를 계산하여 의미 유사도 점수를 산출한다.
#
# 기준 문장은 공식 발표문, 보도자료, 기관 공지 등에서
# 추출한 대표 문장으로 구성한다.
# ─────────────────────────────────────────────────────────────


def _load_reference_sentences() -> list:
    """
    data/reference_sentences.json에서 기준 문장을 로드한다.

    JSON 파일이 없거나 읽기 실패 시 폴백 문장을 반환한다.
    "_"로 시작하는 키는 주석으로 간주하여 무시한다.

    Returns:
        기준 문장 리스트
    """
    # JSON 파일 로드 실패 시 사용할 폴백 문장
    # 최소한의 공식 발표 패턴을 커버한다
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
    except FileNotFoundError:
        print(f"[WARNING] 기준 문장 파일 없음: {REFERENCE_SENTENCES_PATH} → 폴백 문장 사용")
        return fallback
    except json.JSONDecodeError as e:
        print(f"[WARNING] 기준 문장 JSON 파싱 실패: {e} → 폴백 문장 사용")
        return fallback
    except Exception as e:
        print(f"[WARNING] 기준 문장 로드 실패: {e} → 폴백 문장 사용")
        return fallback

    # "_"로 시작하는 키는 주석이므로 무시
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

# 로그 레벨: 환경변수로 조정 가능 (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 로그 출력 형식
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

# 로그 날짜 형식
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
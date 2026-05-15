# Newsoracle — 뉴스 공식성 판별 딥러닝 파이프라인

## 프로젝트 개요

네이버 뉴스 검색 API로 수집한 기사에 대해
**공식 기관 연관성·출처 신뢰성·본문 의미 유사도·문장 표현 특징**을 종합하여
해당 기사가 *공식적 근거를 기반으로 작성되었는가*를 판별하는
End-to-End 딥러닝 파이프라인이다.

단순 출처 기반 판단이 아닌, **공식성(Officiality)**과 **신뢰성(Credibility)**을
명시적으로 분리하여 평가한다는 점이 핵심 설계 의도이다.

## 프로젝트 정보

- 개발 기간: 2026.04.06 ~ 2026.04.10 (5일)
- 개발 인원: 1인 (전체 설계·구현)
- 장소: 광주인력개발원
- 분류: 개인 프로젝트 (포트폴리오)

---

## 시스템 구성

```
사용자 검색어
     │
     ▼
[FastAPI Server] ── BackgroundTasks ──▶ [9단계 파이프라인]
     │                                       │
     │                                       ▼
     │                              Naver API → 전처리 → 특징 추출
     │                                       │
     │                                       ▼
     │                              4중 앙상블 (rule/semantic/clf/agency)
     │                                       │
     │                                       ▼
     │                              교차 보도 검증 + 최종 판정
     │                                       │
     ▼                                       ▼
[SSE Stream] ◀────────── job_store ◀──── 결과 저장
     │
     ▼
HTML/CSS 프론트엔드 (실시간 결과 표시)
```

### 구성 요소

- **FastAPI Server (server.py)**
  - 서버 시작 시 KR-ELECTRA + SentenceTransformer 1회 사전 로드
  - BackgroundTasks 기반 비동기 잡큐 (job_id → in-memory job_store)
  - SSE(Server-Sent Events) 기반 실시간 결과 스트리밍
  - REST API: `/api/suggest`, `/api/search`, `/api/result/{job_id}`, `/api/stream/{job_id}`

- **services/ — 파이프라인 단계별 모듈**
  - `news_search.py` — Naver Search API 연동, 카테고리 편향 방지용 다중 쿼리
  - `article_crawler.py` — 본문 크롤링 (ThreadPoolExecutor 병렬 처리)
  - `preprocessor.py` — HTML 태그 제거, 특수문자 정리, 공백 정규화
  - `feature_extractor.py` — 출처 도메인, 기관명, 공식 표현 탐지
  - `cross_validator.py` — 유사도 기반 클러스터링 + 중복 기사 필터링
  - `rule_based_scorer.py` — 규칙 기반 공식성 점수 (자동 레이블링 겸용)
  - `semantic_similarity.py` — SentenceTransformer 기반 의미 유사도
  - `classifier_model.py` — KR-ELECTRA fine-tuning 분류기
  - `agency_verifier.py` — 본문 기반 공식 인용 검증
  - `query_expander.py` — 쿼리 의도 분류 (카테고리 선택지 반환)
  - `ensemble.py` — 4중 앙상블 + 최종 판정 로직

- **training/ — 학습 파이프라인**
  - KR-ELECTRA fine-tuning (snunlp/KR-ELECTRA-discriminator)
  - train/val/test split (70/15/15)
  - Early Stopping (val_loss 기준, patience=3)
  - Dropout 0.2, AdamW, Linear Warmup

- **frontend (static/)**
  - 검색창 + 카테고리 선택 모달
  - 결과 카드 (공식성/신뢰성 게이지, 상세 점수 펼침)
  - 교차 보도 클릭 시 관련 기사 목록 표시
  - 캐릭터 로딩 애니메이션 (CSS 물결 효과)

---

## 핵심 설계 특징

### 4중 앙상블 구조

서로 다른 관점의 4개 모델을 가중합하여 단일 모델 의존도를 낮춘다.

| 모델 | 역할 | 출력 |
|------|------|------|
| **Rule-based Scorer** | 도메인·기관명·공식 표현 기반 규칙 점수 | rule_score |
| **Semantic Similarity** | 공식 발표 레퍼런스와의 의미 유사도 | semantic_score |
| **Transformer Classifier** | KR-ELECTRA fine-tuning 분류기 | classifier_score |
| **Agency Verifier** | 본문 기반 공식 인용 검증 | agency_score |

가중치는 `config.EnsembleConfig`로 분리하여 실험·튜닝 시 코드 수정 없이 조정 가능.
Classifier 신뢰도가 낮을 경우 fallback 가중치(rule + semantic + agency)로 자동 전환한다.

### 공식성과 신뢰성의 분리

초기 1차 버전은 *공식 기관 발표 여부* 단일 기준으로 판별했으나,
환율·유가 등 공식 발표 없는 신뢰 가능 정보까지 *공식성 낮음*으로 분류되는 문제가 발생했다.

이를 해결하기 위해 두 지표를 명시적으로 분리:

- **공식성(Officiality)**: 공식 표현·기관 인용·출처 도메인 기반
- **신뢰성(Credibility)**: 교차 보도 수·독립 출처 수·내용 일치도 기반

### 4단계 판정 체계

```
조건 A: final_official_score ≥ 0.50
조건 B: credibility_score    ≥ 0.45

A + B 모두 충족  →  ✅ 오피셜 검증됨
B만 충족         →  ⚠️ 교차 보도 확인 (공식 표현 미흡)
A만 충족         →  ⚠️ 공식 표현 확인 (교차 검증 미흡)
A + B 모두 미달  →  ❌ 검증 불가
```

### 9단계 파이프라인 (확장 구조)

```
1. 뉴스 수집 (Naver API + 다중 쿼리)
2. 본문 크롤링 (ThreadPoolExecutor 병렬)
3. 전처리 (HTML 제거, 정규화)
4. 교차 보도 검증 + 중복 클러스터링
5. 특징 추출
6. 규칙 기반 점수
7. 의미 유사도 (SentenceTransformer)
8. 분류 모델 추론 (KR-ELECTRA)
9. 기관 신뢰도 검증 → 앙상블 → 최종 판정
```

---

## 성능 및 평가

- **KR-ELECTRA 분류기 F1 Score**: 0.9869
- 학습 데이터: rule-based 자동 레이블 약 4,000건
- train/val/test split: 70/15/15
- 검색어 1건당 평균 처리 시간: 25~40초 (134건 기준, 병렬 크롤링 적용 후)

처리 시간 분해:
```
모델 로드        →  0초 (서버 기동 시 사전 로드)
뉴스 수집        →  1초 (Naver API)
본문 크롤링      →  15~20초 (병렬 HTTP 요청)
교차 검증 임베딩 →  5~10초 (텍스트 벡터화)
분류 모델 추론   →  3~5초 (KR-ELECTRA)
의미 유사도      →  3~5초 (Cosine Similarity)
```

---

## API 명세

### POST `/api/suggest`
쿼리 의도 분류 → 카테고리 선택지 반환.
토큰 3개 이상이면 `skip_selection=true`로 모달 스킵.

### POST `/api/search`
검색 요청 접수. `job_id`를 즉시 반환하고 백그라운드에서 파이프라인 실행.

### GET `/api/result/{job_id}`
페이지네이션 기반 결과 조회 (`?page=1&page_size=10`).

### GET `/api/stream/{job_id}`
SSE 기반 실시간 스트리밍. 폴링 없이 서버가 완료 시 push.
최대 5분 타임아웃.

---

## 디렉터리 구조

```
Newsoracle/
├── server.py              # FastAPI 엔트리포인트
├── main.py                # CLI 진입점 (디버깅용)
├── news_search.py         # Naver API 클라이언트
├── collect_data.py        # 학습 데이터 수집 스크립트
├── config.py              # 설정값 (가중치·임계값·경로)
├── logger.py              # 로그 시스템
├── requirements.txt
├── services/              # 파이프라인 단계별 모듈
├── training/              # KR-ELECTRA fine-tuning
├── data/                  # 데이터 / 카테고리 라벨
├── static/                # 프론트엔드 (HTML/CSS/JS)
├── utils/                 # 공통 유틸
└── logs/                  # 일별 로그 (자동 로테이션)
```

---

## 실행 방법

### 환경 설정

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 환경 변수

`.env` 파일에 Naver Search API 키 설정:

```
NAVER_CLIENT_ID=your_client_id
NAVER_CLIENT_SECRET=your_client_secret
```

### 서버 실행

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

브라우저에서 `http://localhost:8000` 접속.

### CLI 실행 (디버깅용)

```bash
python main.py
```

---

## 🧪 Tests

총 **61 케이스** · ✅ 56 PASS · ⏭️ 5 SKIP · 성공률 **91.8%**

### 실행
```bash
python tests/run_tests.py
```
- 실행 결과는 `tests/results/` 에 CSV/MD 형식으로 자동 저장됩니다.

### 모듈별 커버리지
| 모듈 | 케이스 수 | 설명 |
|------|----------|------|
| `test_preprocessor` | 21 | 전처리 파이프라인 (HTML, URL, 도메인 정규화) |
| `test_rule_based_scorer` | 11 | 룰베이스 신뢰도 스코어링 |
| `test_ensemble` | 11 | 앙상블 가중치 / 외부 신뢰도 계산 |
| `test_cross_validator` | 5 | 최종 검증 로직 |
| `test_helpers` | 7 | 공통 유틸 (JSON I/O, 디렉토리) |
| `test_server_api` | 5 (SKIP) | API 엔드포인트 *(서버 기동 시 활성화)* |

### 결과 샘플
<details>
<summary>전체 결과 펼쳐보기</summary>

[`tests/results/test_results_2026-05-15.md`](./tests/results/test_results_2026-05-15.md) 참고

</details>

---

## 기술 스택

- **Language**: Python 3
- **Framework**: FastAPI, Uvicorn
- **ML/DL**: PyTorch, HuggingFace Transformers, Sentence-Transformers
- **Model**: snunlp/KR-ELECTRA-discriminator
- **Data**: Naver Search API
- **OS**: Ubuntu 24.04
- **Tool**: Git + GitHub

---

## 알려진 한계

요구사항에 명시된 모든 기능은 구현 완료되었으나,
다음과 같은 한계가 존재한다.

### 학습 단계의 한계

- Rule-based 자동 레이블 패턴이 단순하여 모델이 3 epoch 만에 F1 0.97에 도달.
  인간 검수 레이블 부재로 인한 약한 과적합 징후 관찰.
- 데이터 수집 쿼리가 코드에 하드코딩되어 주제 편향 가능성 존재.
- OOD(Out-of-Distribution) 테스트셋 규모 부족으로 일반화 성능 검증 미흡.
- 앙상블 가중치가 이론적 최적이 아닌 경험적 설정.

### 추론 단계의 한계

- Classifier가 연예 도메인에 대해 극단적으로 낮은 점수(0.001~0.002)를 출력.
  학습 데이터의 *연예 = 비공식* 편향 학습이 원인.
- 동일 보도자료 기반 반복 기사가 다수 클러스터로 잡힐 경우
  루머성 기사도 높은 신뢰성으로 평가될 가능성 존재.
- 본문 크롤링 성공률 약 90% (일부 사이트 차단·실패).

---

## 향후 개선 방향

- Rule-based 레이블을 초기 부트스트래핑용으로만 사용하고,
  일정 규모 이상은 인간 검수 레이블로 보완하는 2단계 학습 전략 도입.
- 데이터 수집 쿼리를 config 기반으로 분리하여 도메인 편향 완화.
- F1-Score 단독 평가 대신 validation loss 기반 과적합 판정 기준 정립.
- 단위 테스트 도입 (ensemble 키 불일치 같은 사일런트 버그 방지).
- 신뢰성 점수 산출에 *원문 링크 존재 여부*, *주요 언론사 가중치* 등을 추가하여
  saturation 방지 (현재 cap 0.999 적용 중).

---

## 회고

이번 프로젝트에서 가장 크게 배운 점은
**모델 성능은 알고리즘이나 구조가 아니라 데이터 품질에 의해 결정된다**는 것이었다.

특히 초기 학습에서 F1 0.97이 도출되었을 때 이를 *성능 향상*으로 잘못 해석했고,
이후 데이터 규모를 3,000건 → 4,000건으로 확장해도 같은 패턴이 반복되면서
비로소 원인이 *데이터 부족이 아닌 rule-based 레이블의 과도한 일관성*임을 인지했다.

또한 추론 단계에서 발견된 ensemble 키 불일치(`classifier_score` vs `score`) 버그는
단위 테스트 부재가 모델 결과에 직접적인 영향을 줄 수 있음을 보여준 사례였다.

기능 구현 자체보다, 초기에 `config` / `logger` / `services` / `training` 구조를
명확히 분리하여 설계한 점이 5일이라는 짧은 기간 내 안정적 구현을 가능하게 했다.
설계의 작은 선택(모듈화, 로그 구조, config 분리)이 전체 개발 속도와 안정성에
큰 영향을 준다는 점을 체감한 프로젝트였다.

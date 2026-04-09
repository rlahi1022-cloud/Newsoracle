"""
server.py
────────────────────────────────────────────────────────────────────────────────
Newsoracle FastAPI 서버

[역할]
  기존 CLI 파이프라인(main.py)을 HTTP API로 감싸서
  브라우저(HTML UI)에서 검색 요청을 받고 JSON 결과를 반환한다.

[구조]
  POST /api/search    → 검색 요청 → job_id 즉시 반환
  GET  /api/result/{job_id} → 결과 조회 (폴링)
  GET  /               → HTML UI 서빙

[모델 로드 전략]
  서버 시작 시 KR-ELECTRA + SentenceTransformer를 1회 로드하고 메모리에 상주.
  이후 요청마다 추론만 수행하므로 모델 로드 시간(~10초)이 제거됨.

[비동기 처리]
  BackgroundTasks로 파이프라인을 백그라운드에서 실행.
  클라이언트는 즉시 job_id를 받고, 폴링으로 결과가 나오면 화면에 표시.

[실행]
  pip install fastapi uvicorn
  python server.py
  → http://localhost:8000 에서 UI 접속

[Flowscope 연결]
  이 서버가 떠있으면 Flowscope에서 HTTP POST /api/search로 호출 가능.
  Redis Pub/Sub 리스너도 나중에 추가 가능.
"""

import os
import sys
import uuid
import time
import threading
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가 (services/, training/ import 가능하게)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from logger import get_logger

logger = get_logger("server")

# ─────────────────────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Newsoracle API",
    description="뉴스 공식성 판별 딥러닝 파이프라인 API",
    version="1.0.0",
)

# ─────────────────────────────────────────────────────────────
# 잡 스토어 (in-memory)
#
# 왜 Redis가 아닌 딕셔너리인가:
#   사용자 1명(시연용), 동시 요청 적음, 서버 재시작 시 결과 소실돼도 무관.
#   Redis는 대규모 서비스에서 사용하고, 지금은 과한 구조.
#   나중에 Redis로 교체하려면 이 딕셔너리를 Redis client로 바꾸면 끝.
# ─────────────────────────────────────────────────────────────

# job_id → {"status": "processing" | "done" | "error", "data": [...], "query": "...", "created_at": "..."}
job_store = {}

# 동시 접근 방어용 락
# BackgroundTasks가 다른 스레드에서 job_store를 수정할 수 있으므로
job_store_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# 요청/응답 모델 (Pydantic)
#
# 왜 Pydantic 모델을 쓰는가:
#   FastAPI가 자동으로 요청 검증 + API 문서 생성을 해줌.
#   타입이 안 맞으면 422 에러를 자동 반환.
# ─────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """검색 요청 바디"""
    query: str
    page: int = 1
    page_size: int = 10


class SearchResponse(BaseModel):
    """검색 응답 (즉시 반환)"""
    job_id: str
    status: str
    message: str


# ─────────────────────────────────────────────────────────────
# 모델 사전 로드 (서버 시작 시 1회)
#
# 왜 startup 이벤트에서 로드하는가:
#   KR-ELECTRA (~416MB) + SentenceTransformer 로드에 약 10초 소요.
#   요청마다 로드하면 10초 대기 → 서버 시작 시 1회 로드하면 제거됨.
#   메모리에 상주하므로 이후 추론은 순수 계산 시간만 소요.
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_load_models():
    """서버 시작 시 AI 모델들을 미리 로드한다."""
    logger.info("서버 시작 — 모델 사전 로드 시작")
    start = time.time()

    try:
        # KR-ELECTRA 분류 모델 로드
        from services.classifier_model import _load_model_and_tokenizer
        _load_model_and_tokenizer()
        logger.info("KR-ELECTRA 분류 모델 로드 완료")
    except Exception as e:
        logger.warning(f"분류 모델 사전 로드 실패 (추론 시 재시도): {e}")

    try:
        # SentenceTransformer 임베딩 모델 로드
        from services.semantic_similarity import _load_embedding_model
        _load_embedding_model()
        logger.info("SentenceTransformer 임베딩 모델 로드 완료")
    except Exception as e:
        logger.warning(f"임베딩 모델 사전 로드 실패 (추론 시 재시도): {e}")

    elapsed = time.time() - start
    logger.info(f"모델 사전 로드 완료 | 소요시간={elapsed:.1f}초")


# ─────────────────────────────────────────────────────────────
# 파이프라인 실행 함수 (BackgroundTasks에서 호출)
#
# 기존 main.py의 run_infer 로직을 서버용으로 재구성.
# 차이점:
#   - 터미널 출력 제거 (JSON 반환용)
#   - 결과를 job_store에 저장
#   - 에러 발생 시 job_store에 에러 상태 기록
# ─────────────────────────────────────────────────────────────

def run_pipeline_background(job_id: str, query: str):
    """
    백그라운드에서 전체 파이프라인을 실행하고
    결과를 job_store에 저장한다.

    BackgroundTasks.add_task()에 의해 별도 스레드에서 실행됨.
    """
    logger.info(f"[JOB {job_id}] 파이프라인 시작 | query={query}")
    start_time = time.time()

    try:
        # ── 1. 뉴스 수집 ─────────────────────────────────────
        from news_search import search_news_combined
        articles = search_news_combined(query)

        if not articles:
            with job_store_lock:
                job_store[job_id] = {
                    "status": "done",
                    "query": query,
                    "data": [],
                    "total": 0,
                    "message": f"'{query}'에 대한 검색 결과가 없습니다.",
                }
            return

        # ── 2. 본문 크롤링 ────────────────────────────────────
        try:
            from services.article_crawler import crawl_articles_batch
            articles = crawl_articles_batch(articles)
        except Exception:
            pass  # 크롤링 실패해도 description으로 진행

        # ── 3. 전처리 ─────────────────────────────────────────
        from services.preprocessor import preprocess_articles
        cleaned_articles = preprocess_articles(articles)

        if not cleaned_articles:
            with job_store_lock:
                job_store[job_id] = {
                    "status": "done",
                    "query": query,
                    "data": [],
                    "total": 0,
                    "message": "유효한 기사를 찾지 못했습니다.",
                }
            return

        # ── 4. 교차 보도 검증 + 중복 필터링 ───────────────────
        try:
            from services.cross_validator import cross_validate_articles
            validated_articles = cross_validate_articles(cleaned_articles)
        except Exception:
            validated_articles = cleaned_articles
            for a in validated_articles:
                a.setdefault("reliability_score", 0.0)
                a.setdefault("reliability_reason", "교차 검증 실패")
                a.setdefault("cluster_id", 0)
                a.setdefault("cluster_size", 1)
                a.setdefault("unique_sources", 1)
                a.setdefault("has_official_domain", False)

        # ── 5. 특징 추출 ─────────────────────────────────────
        from services.feature_extractor import extract_features_batch
        features_list = extract_features_batch(validated_articles)

        # ── 6. 규칙 기반 점수 ─────────────────────────────────
        from services.rule_based_scorer import compute_rule_scores_batch
        rule_scores = compute_rule_scores_batch(features_list)

        # ── 7. 의미 유사도 ────────────────────────────────────
        from services.semantic_similarity import compute_semantic_scores_batch
        embedding_texts = [f.get("embedding_input", "") for f in features_list]
        semantic_scores = compute_semantic_scores_batch(embedding_texts)

        # ── 8. 분류 모델 추론 ─────────────────────────────────
        from services.classifier_model import predict_batch
        classifier_texts = [f.get("classifier_input", "") for f in features_list]
        classifier_results = predict_batch(classifier_texts)

        # ── 9. 기관 신뢰도 검증 ───────────────────────────────
        try:
            from services.agency_verifier import verify_agency_batch
            agency_results = verify_agency_batch(validated_articles, query)
        except Exception:
            agency_results = [{"agency_score": 0.0}] * len(validated_articles)

        # ── 10. 앙상블 ────────────────────────────────────────
        from services.ensemble import ensemble_batch
        final_results = ensemble_batch(
            articles=validated_articles,
            rule_scores=rule_scores,
            semantic_scores=semantic_scores,
            classifier_results=classifier_results,
            agency_results=agency_results,
        )

        # ── 정렬 (오피셜 우선 → 공식성 높은 순) ───────────────
        final_results.sort(
            key=lambda x: (
                x.get("is_verified", False),
                x.get("official_score", 0),
            ),
            reverse=True,
        )

        # ── related_articles를 앙상블 결과에 병합 ──────────────
        # cross_validator의 중복 필터링에서 보존한 관련 기사 목록을
        # 앙상블 결과에 포함시켜 UI에서 교차 보도 클릭 시 표시할 수 있게 한다.
        for result in final_results:
            title = result.get("title", "")
            for va in validated_articles:
                if va.get("title", "") == title:
                    result["related_articles"] = va.get("related_articles", [])
                    break

        elapsed = time.time() - start_time
        verified_count = sum(1 for r in final_results if r.get("is_verified", False))

        # ── 결과 상세 로그 (분석용) ─────────────────────────────
        # 로그만 보고도 가중합 계산 과정, 판정 근거를 완전히 재현할 수 있도록
        # config 설정값 + 각 기사별 가중합 분해 + cap 적용 여부를 기록한다.
        from config import EnsembleConfig

        logger.info(f"[JOB {job_id}] {'═'*60}")
        logger.info(f"[JOB {job_id}] 결과 요약 | 검색어: {query}")
        logger.info(f"[JOB {job_id}] {'─'*60}")

        # config 설정값 로그 (어떤 가중치로 계산했는지 기록)
        logger.info(
            f"[JOB {job_id}] [CONFIG] "
            f"가중치: rule={EnsembleConfig.RULE_WEIGHT} "
            f"sem={EnsembleConfig.SEMANTIC_WEIGHT} "
            f"clf={EnsembleConfig.CLASSIFIER_WEIGHT} | "
            f"폴백: rule={EnsembleConfig.FALLBACK_RULE_WEIGHT} "
            f"sem={EnsembleConfig.FALLBACK_SEMANTIC_WEIGHT} "
            f"agn={EnsembleConfig.FALLBACK_AGENCY_WEIGHT} | "
            f"clf_threshold={EnsembleConfig.CLASSIFIER_LOW_CONFIDENCE}"
        )
        logger.info(
            f"[JOB {job_id}] [CONFIG] "
            f"임계값: 공식성={EnsembleConfig.OFFICIAL_SCORE_THRESHOLD} "
            f"신뢰성={EnsembleConfig.RELIABILITY_SCORE_THRESHOLD} | "
            f"agency_bonus_max={EnsembleConfig.AGENCY_BONUS_MAX} | "
            f"신뢰성 cap=0.999"
        )
        logger.info(
            f"[JOB {job_id}] [STATS] "
            f"총 {len(final_results)}건 | 오피셜 {verified_count}건 | "
            f"소요 {elapsed:.1f}초"
        )
        logger.info(f"[JOB {job_id}] {'─'*60}")

        # 상위 10건 상세 로그
        for i, r in enumerate(final_results[:10], 1):
            rule = r.get('rule_score', 0)
            sem = r.get('semantic_score', 0)
            clf = r.get('classifier_score', 0)
            agn = r.get('agency_score', 0)
            off = r.get('official_score', 0)
            rel = r.get('reliability_score', 0)
            method = r.get('score_method', 'unknown')

            # 가중합 분해 로그
            if method == 'classifier_included':
                weighted = (
                    f"rule*{EnsembleConfig.RULE_WEIGHT}={rule*EnsembleConfig.RULE_WEIGHT:.4f} + "
                    f"sem*{EnsembleConfig.SEMANTIC_WEIGHT}={sem*EnsembleConfig.SEMANTIC_WEIGHT:.4f} + "
                    f"clf*{EnsembleConfig.CLASSIFIER_WEIGHT}={clf*EnsembleConfig.CLASSIFIER_WEIGHT:.4f}"
                )
                if agn > 0:
                    bonus = agn * EnsembleConfig.AGENCY_BONUS_MAX
                    weighted += f" + agn_bonus={bonus:.4f}"
            else:
                weighted = (
                    f"rule*{EnsembleConfig.FALLBACK_RULE_WEIGHT}={rule*EnsembleConfig.FALLBACK_RULE_WEIGHT:.4f} + "
                    f"sem*{EnsembleConfig.FALLBACK_SEMANTIC_WEIGHT}={sem*EnsembleConfig.FALLBACK_SEMANTIC_WEIGHT:.4f} + "
                    f"agn*{EnsembleConfig.FALLBACK_AGENCY_WEIGHT}={agn*EnsembleConfig.FALLBACK_AGENCY_WEIGHT:.4f}"
                )

            # 판정 조건 로그
            off_pass = "PASS" if off >= EnsembleConfig.OFFICIAL_SCORE_THRESHOLD else "FAIL"
            rel_pass = "PASS" if rel >= EnsembleConfig.RELIABILITY_SCORE_THRESHOLD else "FAIL"
            rel_capped = " (capped)" if rel >= 0.999 else ""

            logger.info(
                f"[JOB {job_id}] [{i:2d}] {r.get('verdict_emoji','')} "
                f"{r.get('title','')[:40]}"
            )
            logger.info(
                f"[JOB {job_id}]      점수: rule={rule:.4f} sem={sem:.4f} "
                f"clf={clf:.4f} agn={agn:.4f}"
            )
            logger.info(
                f"[JOB {job_id}]      가중합({method}): {weighted} = {off:.4f}"
            )
            logger.info(
                f"[JOB {job_id}]      판정: 공식성={off:.4f}({off_pass}) "
                f"신뢰성={rel:.4f}{rel_capped}({rel_pass}) → {r.get('verdict','')}"
            )
            if r.get("verdict_reason"):
                logger.info(
                    f"[JOB {job_id}]      근거: {r.get('verdict_reason','')}"
                )

        logger.info(f"[JOB {job_id}] {'═'*60}")

        # ── 결과 저장 ─────────────────────────────────────────
        with job_store_lock:
            job_store[job_id] = {
                "status": "done",
                "query": query,
                "data": final_results,
                "total": len(final_results),
                "verified_count": verified_count,
                "elapsed": round(elapsed, 1),
                "message": f"{len(final_results)}건 분석 완료 ({elapsed:.1f}초)",
            }

        logger.info(
            f"[JOB {job_id}] 파이프라인 완료 | "
            f"{len(final_results)}건 | 오피셜={verified_count}건 | "
            f"{elapsed:.1f}초"
        )

    except Exception as exc:
        logger.error(f"[JOB {job_id}] 파이프라인 실패: {exc}", exc_info=True)
        with job_store_lock:
            job_store[job_id] = {
                "status": "error",
                "query": query,
                "data": [],
                "total": 0,
                "message": f"분석 중 오류가 발생했습니다: {str(exc)}",
            }


# ─────────────────────────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────────────────────────

@app.post("/api/search", response_model=SearchResponse)
async def api_search(req: SearchRequest, background_tasks: BackgroundTasks):
    """
    검색 요청을 받고 job_id를 즉시 반환한다.
    파이프라인은 백그라운드에서 실행된다.

    요청: POST /api/search {"query": "악뮤"}
    응답: {"job_id": "abc-123", "status": "processing", "message": "분석 시작"}
    """
    if not req.query or not req.query.strip():
        return SearchResponse(
            job_id="",
            status="error",
            message="검색어를 입력해주세요.",
        )

    # job_id 생성
    job_id = str(uuid.uuid4())[:8]

    # job_store에 초기 상태 등록
    with job_store_lock:
        job_store[job_id] = {
            "status": "processing",
            "query": req.query.strip(),
            "data": [],
            "total": 0,
            "message": "뉴스를 수집하고 분석 중입니다...",
        }

    # 백그라운드에서 파이프라인 실행
    background_tasks.add_task(run_pipeline_background, job_id, req.query.strip())

    logger.info(f"검색 요청 접수 | job_id={job_id} query={req.query}")

    return SearchResponse(
        job_id=job_id,
        status="processing",
        message="분석을 시작했습니다. 잠시만 기다려주세요.",
    )


@app.get("/api/result/{job_id}")
async def api_result(job_id: str, page: int = 1, page_size: int = 10):
    """
    job_id로 분석 결과를 조회한다.
    페이지네이션 지원. SSE가 아닌 직접 조회 시 사용.

    요청: GET /api/result/abc-123?page=1&page_size=10
    """
    with job_store_lock:
        job = job_store.get(job_id)

    if job is None:
        return {"status": "error", "message": "존재하지 않는 작업입니다."}

    if job["status"] == "processing":
        return {
            "status": "processing",
            "query": job.get("query", ""),
            "message": job.get("message", "분석 중..."),
        }

    # 페이지네이션 적용
    all_data = job.get("data", [])
    total = len(all_data)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_data = all_data[start_idx:end_idx]
    total_pages = (total + page_size - 1) // page_size

    return {
        "status": job["status"],
        "query": job.get("query", ""),
        "data": page_data,
        "total": total,
        "verified_count": job.get("verified_count", 0),
        "elapsed": job.get("elapsed", 0),
        "message": job.get("message", ""),
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
    }


@app.get("/api/stream/{job_id}")
async def api_stream(job_id: str):
    """
    SSE (Server-Sent Events) 스트리밍 엔드포인트.

    클라이언트가 한 번 연결하면 서버가 작업 완료 시 직접 알려준다.
    폴링(주기적 조회)과 달리 불필요한 요청이 없어 서버 부하가 거의 없음.

    [왜 SSE인가]
      - 폴링: 2초마다 GET 요청 → 30초 대기 시 15회 요청 (낭비)
      - SSE: 연결 1회 → 서버가 완료 시 push → 요청 0회
      - WebSocket보다 구현 간단 (서버→클라이언트 단방향이면 충분)
      - FastAPI가 StreamingResponse로 기본 지원

    [흐름]
      1. 클라이언트: EventSource('/api/stream/abc-123') 연결
      2. 서버: 1초마다 상태 확인 → "processing" 이벤트 전송
      3. 파이프라인 완료 시 → "done" 이벤트 + 전체 결과 전송
      4. 연결 자동 종료

    요청: GET /api/stream/abc-123
    응답: text/event-stream (SSE)
    """
    import asyncio
    import json
    from starlette.responses import StreamingResponse

    async def event_generator():
        """
        SSE 이벤트를 생성하는 비동기 제너레이터.
        작업이 완료되거나 타임아웃(5분)될 때까지 반복한다.
        """
        max_wait = 300  # 최대 대기 시간 (초) — 5분
        waited = 0

        while waited < max_wait:
            with job_store_lock:
                job = job_store.get(job_id)

            if job is None:
                # 존재하지 않는 job_id
                yield f"data: {json.dumps({'status': 'error', 'message': '존재하지 않는 작업입니다.'})}\n\n"
                return

            if job["status"] == "done" or job["status"] == "error":
                # 작업 완료 — 전체 결과를 한 번에 전송
                # 첫 페이지 데이터만 전송 (클라이언트에서 페이지네이션은 별도 GET 요청)
                all_data = job.get("data", [])
                page_data = all_data[:10]  # 첫 10건
                total = len(all_data)
                total_pages = (total + 9) // 10

                result = {
                    "status": job["status"],
                    "query": job.get("query", ""),
                    "data": page_data,
                    "total": total,
                    "verified_count": job.get("verified_count", 0),
                    "elapsed": job.get("elapsed", 0),
                    "message": job.get("message", ""),
                    "page": 1,
                    "page_size": 10,
                    "total_pages": total_pages,
                }
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                return

            # 아직 처리 중 — 진행 상태 전송
            yield f"data: {json.dumps({'status': 'processing', 'message': job.get('message', '분석 중...')}, ensure_ascii=False)}\n\n"

            await asyncio.sleep(1)
            waited += 1

        # 타임아웃
        yield f"data: {json.dumps({'status': 'error', 'message': '분석 시간이 초과되었습니다.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
        },
    )


# ─────────────────────────────────────────────────────────────
# 정적 파일 서빙 (HTML UI)
#
# static/ 폴더에 index.html, style.css 등을 넣으면
# http://localhost:8000/ 에서 접속 가능.
# ─────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# static 폴더의 파일을 /static/ 경로로 서빙
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def serve_index():
    """루트 경로로 접속하면 index.html을 반환한다."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Newsoracle API 서버 실행 중. /docs 에서 API 문서를 확인하세요."}


# ─────────────────────────────────────────────────────────────
# 서버 실행
#
# python server.py로 직접 실행 가능.
# 개발 모드: reload=True로 코드 변경 시 자동 재시작.
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Newsoracle 서버 시작 | http://localhost:8000")
    logger.info("API 문서: http://localhost:8000/docs")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # 프로덕션에서는 False. 개발 중 True로 변경 가능.
        log_level="info",
    )
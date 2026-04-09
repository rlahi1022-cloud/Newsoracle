"""
training/trainer.py
────────────────────
분류 모델을 학습하는 전체 루프.

[v2 변경 사항]
1. Dropout 정규화 적용
   - config의 hidden_dropout_prob, attention_probs_dropout_prob를 모델에 주입
   - 기본값 0.1 → 0.2로 올려 과적합 억제

2. Early Stopping (val_loss 기준)
   - val_f1이 아닌 val_loss 기준으로 멈춤
   - 이유: val_f1은 threshold에 따라 왜곡 가능, val_loss가 더 순수한 지표
   - patience=3: val_loss가 3 epoch 연속 개선 없으면 중단
   - 과적합 진입 시점: train_loss 계속 하락 / val_loss 상승 교차점

3. OOD 평가 추가
   - 학습/검증과 다른 쿼리로 수집한 ood_test.csv로 최종 평가
   - 실제 추론 환경과 유사한 조건에서 성능 측정
   - ood_test.csv 없으면 건너뜀
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from training.evaluator import evaluate
from services.classifier_model import save_model
from config import (
    CLASSIFIER_MODEL_NAME, NUM_LABELS, TRAIN_CONFIG,
    MODEL_SAVE_DIR, OOD_TEST_PATH,
)
from logger import get_logger

logger = get_logger("trainer")

DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "classifier")


class EarlyStopping:
    """
    val_loss 기준 Early Stopping 구현.

    과적합 진입 시점 감지 원리:
    - 정상 학습: train_loss 하락 + val_loss 하락 (같이 내려감)
    - 과적합 진입: train_loss 계속 하락 + val_loss 상승 (역전)
    - patience epoch 동안 val_loss가 개선 없으면 중단

    왜 val_f1이 아닌 val_loss인가:
    - val_f1은 threshold(0.5 기준)에서 계산 → threshold 선택에 따라 왜곡 가능
    - val_loss는 모델 내부의 확신도(softmax 이전 logit)를 직접 반영
    - 모델이 애매해지기 시작하면 loss가 먼저 올라감
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        """
        Args:
            patience:  val_loss 개선 없이 허용할 최대 epoch 수
            min_delta: 개선으로 인정할 최소 감소량
        """
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        현재 val_loss를 받아 중단 여부를 판단한다.

        Args:
            val_loss: 현재 epoch의 검증 손실값
        Returns:
            True: 학습 중단 필요 / False: 학습 계속
        """
        if val_loss < self.best_loss - self.min_delta:
            # val_loss 개선됨 → 카운터 리셋
            self.best_loss = val_loss
            self.counter   = 0
            logger.info(f"val_loss 개선: {val_loss:.4f} (best)")
        else:
            # val_loss 개선 없음 → 카운터 증가
            self.counter += 1
            logger.info(
                f"val_loss 개선 없음: {val_loss:.4f} "
                f"(best={self.best_loss:.4f}, patience={self.counter}/{self.patience})"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early Stopping 발동: "
                    f"{self.patience} epoch 연속 val_loss 개선 없음"
                )
        return self.should_stop


def compute_val_loss(model, dataset, batch_size: int) -> float:
    """
    검증 데이터셋에 대한 평균 loss를 계산한다.

    Early Stopping의 판단 기준으로 사용.
    val_f1과 달리 threshold 영향을 받지 않음.

    Args:
        model:      평가할 모델
        dataset:    검증용 Dataset
        batch_size: 배치 크기
    Returns:
        평균 validation loss
    """
    model.eval()
    total_loss = 0.0
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def train(train_dataset, val_dataset, tokenizer):
    """
    분류 모델 전체 학습 루프.

    처리 순서:
    1. Dropout 설정을 반영한 모델 초기화
    2. epoch 반복 → 배치 학습
    3. val_loss 계산 → Early Stopping 판단
    4. val_f1 계산 → 최적 모델 저장
    5. OOD 테스트셋 평가 (있을 경우)

    Args:
        train_dataset: 학습용 Dataset
        val_dataset:   검증용 Dataset
        tokenizer:     토크나이저 (모델 저장 시 함께 저장)
    """
    epochs         = TRAIN_CONFIG["epochs"]
    batch_size     = TRAIN_CONFIG["batch_size"]
    learning_rate  = TRAIN_CONFIG["learning_rate"]
    patience       = TRAIN_CONFIG["early_stopping_patience"]
    dropout_hidden = TRAIN_CONFIG["hidden_dropout_prob"]
    dropout_attn   = TRAIN_CONFIG["attention_probs_dropout_prob"]

    logger.info(
        f"학습 시작 | epochs(max)={epochs} batch={batch_size} "
        f"lr={learning_rate} device={DEVICE} "
        f"dropout={dropout_hidden} early_stopping_patience={patience}"
    )

    # ── 모델 초기화 (Dropout 주입) ────────────────────────────
    # hidden_dropout_prob, attention_probs_dropout_prob를 직접 주입
    # ELECTRA는 이 두 파라미터로 dropout을 제어함
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=dropout_hidden,
        attention_probs_dropout_prob=dropout_attn,
        ignore_mismatched_sizes=True,
    )
    model.to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # AdamW: BERT 계열 학습에 권장 (weight decay로 L2 정규화 효과)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 학습률 스케줄러: warmup 후 선형 감소
    # warmup: 초반 급격한 가중치 변화 방지
    total_steps  = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    early_stopping = EarlyStopping(patience=patience)
    best_f1        = 0.0
    best_epoch     = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # gradient clipping: 그래디언트 폭발 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:
                # val_loss를 매 10 step마다 계산하여 train_loss 바로 옆에 출력
                # gap = train_loss - val_loss
                #   양수(+): 정상 학습 중
                #   음수(-): train이 val보다 낮음 → 과적합 진입 신호
                current_val_loss = compute_val_loss(model, val_dataset, batch_size)
                model.train()  # val_loss 계산 후 반드시 train 모드로 복귀
                current_train_loss = loss.item()
                gap = current_train_loss - current_val_loss
                gap_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
                overfit_flag = "  ⚠" if gap < -0.05 else ""
                logger.info(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_loss={current_train_loss:.4f}  "
                    f"val_loss={current_val_loss:.4f}  "
                    f"gap={gap_str}"
                    f"{overfit_flag}"
                )

        avg_train_loss = total_loss / len(train_loader)

        # ── val_loss 계산 (Early Stopping 판단용) ────────────────
        val_loss = compute_val_loss(model, val_dataset, batch_size)

        # ── val 지표 계산 (모델 저장 판단용) ────────────────────
        metrics = evaluate(model, val_dataset, batch_size=batch_size)

        # ── epoch 완료 요약: train_loss / val_loss / gap 나란히 출력 ──
        # gap > 0: train이 val보다 높음 (정상, 아직 학습 중)
        # gap < 0: train이 val보다 낮음 → 과적합 진입 신호
        gap = avg_train_loss - val_loss
        gap_str = f"+{gap:.4f}" if gap >= 0 else f"{gap:.4f}"
        overfit_flag = "  ← ⚠ 과적합 주의" if gap < -0.05 else ""

        logger.info(
            f"epoch={epoch} 완료 | "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"gap={gap_str}"
            f"{overfit_flag}"
        )
        logger.info(f"epoch={epoch} | val_metrics={metrics}")

        # ── f1 기준 최적 모델 저장 ───────────────────────────────
        if metrics["f1"] > best_f1:
            best_f1    = metrics["f1"]
            best_epoch = epoch
            save_model(model, tokenizer, FINETUNED_MODEL_PATH)
            logger.info(f"최적 모델 저장 | epoch={epoch} f1={best_f1:.4f}")

        # ── Early Stopping 판단 ───────────────────────────────────
        if early_stopping.step(val_loss):
            logger.info(
                f"학습 조기 종료 | epoch={epoch} | "
                f"최적 epoch={best_epoch} f1={best_f1:.4f}"
            )
            break

    logger.info(f"학습 완료 | best_f1={best_f1:.4f} (epoch={best_epoch})")

    # ── OOD 테스트셋 평가 ─────────────────────────────────────────
    # 학습/검증과 다른 일반 키워드로 수집한 데이터로 최종 평가
    # 이 점수가 실제 서비스 환경에서의 성능에 가장 가까움
    _evaluate_ood(model, tokenizer, batch_size)


def _evaluate_ood(model, tokenizer, batch_size: int):
    """
    OOD 테스트셋으로 최종 성능을 평가한다.

    OOD(Out-of-Distribution) 데이터:
    - 학습/검증 데이터와 다른 분포의 데이터
    - 예: "자동차", "기준금리", "코스피" 등 일반 키워드 기사
    - 이 평가에서도 높은 f1이 나와야 실제 서비스에서 믿을 수 있음

    Args:
        model:      학습 완료된 모델
        tokenizer:  토크나이저
        batch_size: 배치 크기
    """
    if not os.path.exists(OOD_TEST_PATH):
        logger.info(
            f"OOD 테스트셋 없음: {OOD_TEST_PATH} "
            f"(collect_data.py로 일반 키워드 기사를 수집하여 생성 가능)"
        )
        return

    logger.info(f"OOD 평가 시작: {OOD_TEST_PATH}")

    try:
        import pandas as pd
        from training.dataset import NewsOfficialityDataset

        df = pd.read_csv(OOD_TEST_PATH, encoding="utf-8-sig")
        df = df.dropna(subset=["title", "content", "official_label"])

        if len(df) == 0:
            logger.warning("OOD 테스트셋이 비어 있음")
            return

        texts  = (df["title"].fillna("") + " " + df["content"].fillna("")).tolist()
        labels = df["official_label"].astype(int).tolist()

        from config import MAX_TOKEN_LENGTH
        ood_dataset = NewsOfficialityDataset(texts, labels, tokenizer, MAX_TOKEN_LENGTH)
        ood_metrics = evaluate(model, ood_dataset, batch_size=batch_size)

        logger.info(f"OOD 평가 완료 | metrics={ood_metrics}")
        logger.info(
            f"[참고] val_f1과 OOD_f1의 차이가 클수록 과적합이 심한 것\n"
            f"  val_f1과 OOD_f1이 비슷하면 실제 환경에서도 안정적으로 동작"
        )

    except Exception as e:
        logger.error(f"OOD 평가 실패: {e}")
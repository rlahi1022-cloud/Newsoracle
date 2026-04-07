# 모델 실제로 학습시키는 부분
"""
training/trainer.py
────────────────────
분류 모델을 학습하는 전체 루프를 구현한다.
epoch마다 loss를 출력하고 validation 평가를 수행한다.
최적 모델을 자동 저장한다.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from training.evaluator import evaluate
from services.classifier_model import save_model
from config import CLASSIFIER_MODEL_NAME, NUM_LABELS, TRAIN_CONFIG, MODEL_SAVE_DIR
from logger import get_logger
import os

logger = get_logger("trainer")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "classifier")


def train(train_dataset, val_dataset, tokenizer):
    """
    분류 모델 전체 학습 루프.
    
    처리 순서:
    1. 모델 및 옵티마이저 초기화
    2. epoch 반복 → 배치 학습
    3. validation 평가
    4. 최적 f1 기준 모델 저장
    
    Args:
        train_dataset: 학습용 Dataset
        val_dataset:   검증용 Dataset
        tokenizer:     토크나이저 (모델 저장 시 함께 저장)
    """
    epochs = TRAIN_CONFIG["epochs"]
    batch_size = TRAIN_CONFIG["batch_size"]
    learning_rate = TRAIN_CONFIG["learning_rate"]

    logger.info(f"학습 시작 | epochs={epochs} batch={batch_size} lr={learning_rate} device={DEVICE}")

    # 사전학습 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    model.to(DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # AdamW: BERT 계열 학습에 권장되는 옵티마이저 (weight decay 지원)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # 학습률 스케줄러: warmup 후 선형 감소
    total_steps = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            # 그래디언트 폭발 방지 (gradient clipping)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 10 == 0:
                logger.info(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"epoch={epoch} 완료 | avg_loss={avg_loss:.4f}")

        # validation 평가
        metrics = evaluate(model, val_dataset, batch_size=batch_size)
        logger.info(f"epoch={epoch} | val_metrics={metrics}")

        # f1 기준 최적 모델 저장
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_model(model, tokenizer, FINETUNED_MODEL_PATH)
            logger.info(f"최적 모델 저장 | f1={best_f1:.4f}")

    logger.info(f"학습 완료 | best_f1={best_f1:.4f}")
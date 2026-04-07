# 모델 성능 측정 지표 계산 모듈
"""
training/evaluator.py
──────────────────────
모델 평가 지표(accuracy, precision, recall, f1)를 계산한다.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from logger import get_logger

logger = get_logger("evaluator")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, dataset, batch_size: int = 16) -> dict:
    """
    검증 데이터셋에 대해 모델을 평가하고 지표를 반환한다.
    
    Args:
        model:      평가할 모델
        dataset:    검증용 Dataset 객체
        batch_size: 배치 크기
    Returns:
        {"accuracy": ..., "precision": ..., "recall": ..., "f1": ...}
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    logger.info(f"평가 완료 | acc={accuracy:.4f} prec={precision:.4f} rec={recall:.4f} f1={f1:.4f}")
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=["비공식", "공식"]))

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
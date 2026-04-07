# 텍스트 분류 모델 담당
"""
services/classifier_model.py
──────────────────────────────
HuggingFace Transformers 기반 한국어 이진 분류 모델.
공식성 높음(1) / 낮음(0) 이진 분류를 수행한다.
모델명은 config에서 교체 가능하다.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from logger import get_logger
from config import CLASSIFIER_MODEL_NAME, MAX_TOKEN_LENGTH, NUM_LABELS, MODEL_SAVE_DIR

logger = get_logger("classifier_model")

# 저장된 파인튜닝 모델 경로
FINETUNED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "classifier")

# 디바이스 설정 (GPU 있으면 GPU, 없으면 CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델/토크나이저 캐시 (지연 로드)
_tokenizer = None
_model = None


def _load_model_and_tokenizer():
    """
    분류 모델과 토크나이저를 로드한다.
    파인튜닝된 모델이 있으면 그것을 우선 로드하고,
    없으면 HuggingFace 사전학습 모델을 로드한다.
    
    왜 지연 로드인가:
    - 모델 로드는 수 GB 메모리를 사용하므로 필요 시에만 로드
    - infer 모드에서만 필요한 모델이 train 모드에서 로드되지 않도록 설계
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    # 파인튜닝된 모델이 있으면 로드
    if os.path.exists(FINETUNED_MODEL_PATH):
        model_path = FINETUNED_MODEL_PATH
        logger.info(f"파인튜닝 모델 로드: {model_path}")
    else:
        model_path = CLASSIFIER_MODEL_NAME
        logger.info(f"사전학습 모델 로드: {model_path}")

    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=NUM_LABELS,
        )
        _model.to(DEVICE)
        _model.eval()  # 추론 모드로 설정
        logger.info(f"모델 로드 완료 | device={DEVICE}")
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        raise

    return _tokenizer, _model


def predict_single(text: str) -> dict:
    """
    단일 텍스트에 대한 공식성 분류 결과를 반환한다.
    
    처리 순서:
    1. 텍스트 토크나이징
    2. 모델 추론 (softmax 확률 계산)
    3. 공식성 확률(레이블 1) 반환
    
    Args:
        text: 분류할 텍스트 (제목 + 본문)
    Returns:
        {"classifier_score": float, "predicted_label": int}
    """
    if not text or not text.strip():
        return {"classifier_score": 0.0, "predicted_label": 0}

    try:
        tokenizer, model = _load_model_and_tokenizer()

        # 텍스트 토크나이징
        inputs = tokenizer(
            text,
            max_length=MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # 추론 (그래디언트 계산 불필요)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # 공식성 높음(레이블 1)의 확률
        official_prob = float(probs[0][1].item())
        predicted_label = int(torch.argmax(probs, dim=-1).item())

        return {
            "classifier_score": round(official_prob, 4),
            "predicted_label": predicted_label,
        }

    except Exception as e:
        logger.error(f"분류 모델 추론 실패: {e}")
        return {"classifier_score": 0.0, "predicted_label": 0}


def predict_batch(texts: list[str], batch_size: int = 8) -> list[dict]:
    """
    텍스트 목록 전체에 대해 배치 추론을 수행한다.
    
    한 번에 batch_size만큼 처리하여 메모리 효율을 유지한다.
    
    Args:
        texts:      분류할 텍스트 리스트
        batch_size: 한 번에 처리할 샘플 수
    Returns:
        각 텍스트에 대한 {"classifier_score": ..., "predicted_label": ...} 리스트
    """
    if not texts:
        logger.warning("분류 모델 입력이 비어 있음")
        return []

    logger.info(f"분류 모델 추론 시작 | {len(texts)}건 | batch_size={batch_size}")

    try:
        tokenizer, model = _load_model_and_tokenizer()

        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]

            inputs = tokenizer(
                batch_texts,
                max_length=MAX_TOKEN_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for prob_row in probs:
                official_prob = float(prob_row[1].item())
                pred_label = int(torch.argmax(prob_row).item())
                results.append({
                    "classifier_score": round(official_prob, 4),
                    "predicted_label": pred_label,
                })

        logger.info(f"분류 모델 추론 완료 | {len(results)}건")
        return results

    except Exception as e:
        logger.error(f"배치 분류 추론 실패: {e}")
        return [{"classifier_score": 0.0, "predicted_label": 0}] * len(texts)


def save_model(model, tokenizer, save_path: str = FINETUNED_MODEL_PATH):
    """
    파인튜닝된 모델과 토크나이저를 저장한다.
    학습 완료 후 trainer.py에서 호출한다.
    
    Args:
        model:     학습된 모델 객체
        tokenizer: 토크나이저 객체
        save_path: 저장 경로
    """
    try:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        logger.info(f"모델 저장 완료: {save_path}")
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")
        raise
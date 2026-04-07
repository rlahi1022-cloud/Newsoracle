# 학습 데이터 로더 : 학습용 데이터 준비

"""
training/dataset.py
────────────────────
CSV 파일에서 학습 데이터를 읽어 PyTorch Dataset 형태로 변환한다.
train/val 분리 기능을 포함한다.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from config import CLASSIFIER_MODEL_NAME, MAX_TOKEN_LENGTH, TRAIN_CONFIG
from logger import get_logger

logger = get_logger("dataset")

# 학습 데이터 필수 컬럼
REQUIRED_COLUMNS = ["title", "content", "official_label"]


class NewsOfficialityDataset(Dataset):
    """
    뉴스 공식성 분류를 위한 PyTorch Dataset.
    
    HuggingFace Trainer / DataLoader와 호환되는 형태로 데이터를 제공한다.
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        """
        Args:
            texts:      기사 텍스트 리스트 (제목 + 본문)
            labels:     정답 레이블 리스트 (0 또는 1)
            tokenizer:  HuggingFace 토크나이저
            max_length: 최대 토큰 길이
        """
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        단일 샘플을 반환한다.
        input_ids, attention_mask, labels를 포함하는 딕셔너리 반환.
        """
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_dataset_from_csv(csv_path: str):
    """
    CSV 파일에서 학습 데이터를 로드하고 train/val로 분리한다.
    
    CSV 필수 컬럼: title, content, official_label
    
    Args:
        csv_path: 학습 데이터 CSV 경로
    Returns:
        (train_dataset, val_dataset) 튜플
    """
    # 파일 존재 확인
    import os
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"학습 데이터 파일 없음: {csv_path}")

    logger.info(f"학습 데이터 로드 시작: {csv_path}")

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except Exception as e:
        raise ValueError(f"CSV 파일 로드 실패: {e}")

    # 필수 컬럼 검증
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필수 컬럼 없음: {missing}")

    # 결측값 제거
    df = df.dropna(subset=REQUIRED_COLUMNS)

    if len(df) == 0:
        raise ValueError("유효한 학습 데이터가 없음 (모든 행이 결측값)")

    logger.info(f"총 {len(df)}건 로드 | 레이블 분포: {df['official_label'].value_counts().to_dict()}")

    # 제목 + 본문 결합 텍스트 생성
    texts = (df["title"].fillna("") + " " + df["content"].fillna("")).tolist()
    labels = df["official_label"].astype(int).tolist()

    # 레이블 값 검증 (0 또는 1만 허용)
    invalid_labels = [l for l in labels if l not in [0, 1]]
    if invalid_labels:
        raise ValueError(f"잘못된 레이블 값 포함: {set(invalid_labels)} (0 또는 1만 허용)")

    # 토크나이저 로드
    logger.info(f"토크나이저 로드: {CLASSIFIER_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME)

    # train / val 분리
    val_ratio = TRAIN_CONFIG["val_ratio"]
    seed = TRAIN_CONFIG["seed"]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,  # 레이블 비율 유지
    )

    logger.info(f"train: {len(train_texts)}건 | val: {len(val_texts)}건")

    train_dataset = NewsOfficialityDataset(train_texts, train_labels, tokenizer, MAX_TOKEN_LENGTH)
    val_dataset = NewsOfficialityDataset(val_texts, val_labels, tokenizer, MAX_TOKEN_LENGTH)

    return train_dataset, val_dataset, tokenizer
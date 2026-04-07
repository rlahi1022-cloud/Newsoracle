"""
utils/helpers.py
─────────────────
프로젝트 전반에서 재사용되는 유틸리티 함수 모음.
"""

import os
import json
from datetime import datetime
from logger import get_logger

logger = get_logger("helpers")


def ensure_dir(path: str):
    """
    디렉토리가 없으면 생성한다.
    
    Args:
        path: 생성할 디렉토리 경로
    """
    os.makedirs(path, exist_ok=True)


def save_json(data: dict | list, filepath: str):
    """
    데이터를 JSON 파일로 저장한다.
    
    Args:
        data:     저장할 데이터
        filepath: 저장 경로
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"JSON 저장: {filepath}")
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        raise


def load_json(filepath: str) -> dict | list:
    """
    JSON 파일을 로드한다.
    
    Args:
        filepath: 로드할 파일 경로
    Returns:
        로드된 데이터
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"파일 없음: {filepath}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON 로드 실패: {e}")
        raise


def get_timestamp() -> str:
    """
    현재 시각을 문자열로 반환한다.
    파일명, 로그에 사용.
    
    Returns:
        예: "2026-04-07_141011"
    """
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")
"""
logger.py
─────────
프로젝트 전체에서 공통으로 사용하는 로거를 정의한다.
timestamp, log level, 단계 정보를 포함한 형태로 출력한다.
"""

import logging
import os
from datetime import datetime

# 로그 파일 저장 디렉토리
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 로그 파일명: logs/2026-04-07.log 형태
LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거를 반환한다.
    
    왜 모듈별로 분리하는가:
    - 어느 파일에서 출력된 로그인지 name으로 구분 가능
    - 예: [news_search] [INFO] 기사 수집 시작
    
    Args:
        name: 로거 이름 (보통 __name__ 전달)
    
    Returns:
        설정된 Logger 객체
    """
    logger = logging.getLogger(name)

    # 이미 핸들러가 등록된 경우 중복 방지
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 로그 포맷: [시간] [레벨] [모듈명] 메시지
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 콘솔 출력 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # 파일 저장 핸들러 (DEBUG 이상 전부 저장)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
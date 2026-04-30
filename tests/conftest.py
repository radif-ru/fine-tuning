"""Pytest фикстуры для тестов.

Глобальные фикстуры, доступные во всех тестах.
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.core.config import Settings


@pytest.fixture
def temp_dir():
    """Временная директория для тестов.
    
    Yields:
        Path: Путь к временной директории
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_settings(temp_dir):
    """Настройки для тестов с временными путями.
    
    Args:
        temp_dir: Фикстура временной директории
    
    Returns:
        Settings: Конфигурация для тестов
    """
    return Settings(
        BASE_MODEL_NAME="gpt2",
        TRAIN_DATA_PATH=str(temp_dir / "train.jsonl"),
        OUTPUT_DIR=str(temp_dir / "outputs"),
        CHECKPOINT_DIR=str(temp_dir / "checkpoints"),
        LOG_LEVEL="DEBUG",
        LOG_FILE=str(temp_dir / "test.log"),
        NUM_EPOCHS=1,
        PER_DEVICE_BATCH_SIZE=2,
        MAX_SEQ_LENGTH=128,
    )


@pytest.fixture
def sample_train_data():
    """Пример тренировочных данных для тестов.
    
    Returns:
        list: Список примеров в формате Alpaca
    """
    return [
        {
            "instruction": "Напиши приветствие",
            "input": "",
            "output": "Привет! Как дела?"
        },
        {
            "instruction": "Объясни Python",
            "input": "для начинающих",
            "output": "Python — язык программирования..."
        },
        {
            "instruction": "Переведи на английский",
            "input": "Привет мир",
            "output": "Hello world"
        }
    ]


@pytest.fixture
def sample_raw_data():
    """Пример данных в формате raw text.
    
    Returns:
        list: Список текстов
    """
    return [
        {"text": "Первый пример текста для обучения."},
        {"text": "Второй пример текста для обучения."},
        {"text": "Третий пример текста для обучения."}
    ]


@pytest.fixture
def sample_alpaca_file(temp_dir, sample_train_data):
    """Файл с данными в формате Alpaca.
    
    Args:
        temp_dir: Фикстура временной директории
        sample_train_data: Фикстура с данными
    
    Returns:
        Path: Путь к созданному файлу
    """
    file_path = temp_dir / "train_alpaca.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return file_path


@pytest.fixture
def sample_raw_file(temp_dir, sample_raw_data):
    """Файл с данными в формате raw text.
    
    Args:
        temp_dir: Фикстура временной директории
        sample_raw_data: Фикстура с данными
    
    Returns:
        Path: Путь к созданному файлу
    """
    file_path = temp_dir / "train_raw.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_raw_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return file_path


@pytest.fixture
def mock_env_file(temp_dir):
    """Создать mock .env файл для тестов.
    
    Args:
        temp_dir: Фикстура временной директории
    
    Returns:
        Path: Путь к созданному .env файлу
    """
    env_content = """
BASE_MODEL_NAME=gpt2
TRAIN_DATA_PATH=./data/train.jsonl
OUTPUT_DIR=./outputs
CHECKPOINT_DIR=./checkpoints
NUM_EPOCHS=3
LORA_R=8
LORA_ALPHA=16
LOG_LEVEL=INFO
"""
    env_path = temp_dir / ".env"
    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    return env_path


@pytest.fixture
def empty_logger():
    """Мок логгер для тестов.
    
    Returns:
        MagicMock: Мок-объект логгера
    """
    from unittest.mock import MagicMock
    return MagicMock()

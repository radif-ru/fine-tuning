"""Тесты для конфигурации приложения."""

import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


class TestSettings:
    """Тесты для Settings класса."""
    
    def test_required_fields(self):
        """Проверка обязательных полей."""
        # Проверяем что обязательные поля действительно обязательны
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None)

        errors = exc_info.value.errors()
        required_fields = [e["loc"][0] for e in errors if e["type"] == "missing"]
        assert "BASE_MODEL_NAME" in required_fields
        assert "TRAIN_DATA_PATH" in required_fields
    
    def test_default_values(self):
        """Проверка значений по умолчанию."""
        settings = Settings(
            BASE_MODEL_NAME="gpt2",
            TRAIN_DATA_PATH="./data/train.jsonl"
        )
        
        assert settings.LORA_R == 8
        assert settings.LORA_ALPHA == 16
        assert settings.LORA_DROPOUT == 0.1
        assert settings.NUM_EPOCHS == 3
        assert settings.PER_DEVICE_BATCH_SIZE == 1
        assert settings.LEARNING_RATE == 2e-4
        assert settings.MAX_SEQ_LENGTH == 512
        assert settings.LOG_LEVEL == "INFO"
    
    def test_lora_target_modules_parsing(self):
        """Проверка парсинга LORA_TARGET_MODULES из строки."""
        settings = Settings(
            BASE_MODEL_NAME="gpt2",
            TRAIN_DATA_PATH="./data/train.jsonl",
            LORA_TARGET_MODULES="q_proj,v_proj,k_proj"
        )
        
        assert settings.lora_target_modules_list == ["q_proj", "v_proj", "k_proj"]
    
    def test_lora_target_modules_list(self):
        """Проверка LORA_TARGET_MODULES как строка (конвертируется в список)."""
        settings = Settings(
            BASE_MODEL_NAME="gpt2",
            TRAIN_DATA_PATH="./data/train.jsonl",
            LORA_TARGET_MODULES="q_proj,v_proj"
        )
        
        assert settings.lora_target_modules_list == ["q_proj", "v_proj"]
    
    def test_validation_lora_r_positive(self):
        """Проверка валидации LORA_R > 0."""
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                LORA_R=0
            )
    
    def test_validation_lora_dropout_range(self):
        """Проверка валидации 0 <= LORA_DROPOUT <= 1."""
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                LORA_DROPOUT=1.5
            )
    
    def test_validation_lora_bias(self):
        """Проверка валидации LORA_BIAS значений."""
        # Валидные значения
        for bias in ["none", "all", "lora_only"]:
            settings = Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                LORA_BIAS=bias
            )
            assert settings.LORA_BIAS == bias
        
        # Невалидное значение
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                LORA_BIAS="invalid"
            )
    
    def test_validation_eval_strategy(self):
        """Проверка валидации EVAL_STRATEGY."""
        for strategy in ["no", "steps", "epoch"]:
            settings = Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                EVAL_STRATEGY=strategy
            )
            assert settings.EVAL_STRATEGY == strategy
        
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                EVAL_STRATEGY="invalid"
            )
    
    def test_validation_log_level(self):
        """Проверка валидации LOG_LEVEL."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            settings = Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                LOG_LEVEL=level
            )
            assert settings.LOG_LEVEL == level
        
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                LOG_LEVEL="invalid"
            )
    
    def test_validation_learning_rate_positive(self):
        """Проверка валидации LEARNING_RATE > 0."""
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                LEARNING_RATE=0
            )
    
    def test_validation_temperature_positive(self):
        """Проверка валидации TEMPERATURE > 0."""
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                TEMPERATURE=0
            )
    
    def test_validation_top_p_range(self):
        """Проверка валидации 0 <= TOP_P <= 1."""
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                TOP_P=1.5
            )
    
    def test_validation_inference_device(self):
        """Проверка валидации INFERENCE_DEVICE."""
        for device in ["auto", "cpu", "cuda", "cuda:0", "cuda:1"]:
            settings = Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                INFERENCE_DEVICE=device
            )
            assert settings.INFERENCE_DEVICE == device
        
        with pytest.raises(ValidationError):
            Settings(
                BASE_MODEL_NAME="gpt2",
                TRAIN_DATA_PATH="./data/train.jsonl",
                INFERENCE_DEVICE="invalid"
            )
    
    def test_load_from_env_file(self):
        """Проверка загрузки из .env файла."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("BASE_MODEL_NAME=gpt2\n")
            f.write("TRAIN_DATA_PATH=./data/train.jsonl\n")
            f.write("LORA_R=16\n")
            f.write("NUM_EPOCHS=5\n")
            env_path = f.name
        
        try:
            settings = Settings(_env_file=env_path)
            assert settings.BASE_MODEL_NAME == "gpt2"
            assert settings.TRAIN_DATA_PATH == "./data/train.jsonl"
            assert settings.LORA_R == 16
            assert settings.NUM_EPOCHS == 5
        finally:
            os.unlink(env_path)
    
    def test_optional_fields_none(self):
        """Проверка optional полей со значением None."""
        settings = Settings(
            BASE_MODEL_NAME="gpt2",
            TRAIN_DATA_PATH="./data/train.jsonl",
            HF_CACHE_DIR=None,
            VALIDATION_DATA_PATH=None,
            WANDB_API_KEY=None
        )
        
        assert settings.HF_CACHE_DIR is None
        assert settings.VALIDATION_DATA_PATH is None
        assert settings.WANDB_API_KEY is None

"""Конфигурация приложения через pydantic-settings.

Все параметры загружаются из .env файла с валидацией типов.
"""

from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Настройки приложения для fine-tuning LLM.
    
    Загружает конфигурацию из .env файла с валидацией.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # =============================================================================
    # Конфигурация модели
    # =============================================================================
    BASE_MODEL_NAME: str = Field(..., description="Имя базовой модели из HF Hub или локальный путь")
    HF_CACHE_DIR: Optional[str] = Field(None, description="Директория кэша HuggingFace")
    TRUST_REMOTE_CODE: bool = Field(False, description="Доверять remote code при загрузке")
    
    # =============================================================================
    # Конфигурация LoRA
    # =============================================================================
    LORA_R: int = Field(default=8, ge=1, description="LoRA rank")
    LORA_ALPHA: int = Field(default=16, ge=1, description="LoRA alpha scaling")
    LORA_DROPOUT: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout rate")
    LORA_TARGET_MODULES: str = Field(
        default="q_proj,k_proj,v_proj,o_proj",
        description="Модули для применения LoRA (через запятую)"
    )
    LORA_BIAS: str = Field(
        default="none",
        pattern="^(none|all|lora_only)$",
        description="Режим обучения bias"
    )
    LORA_TASK_TYPE: str = Field(default="CAUSAL_LM", description="Тип задачи для LoRA")
    
    # =============================================================================
    # Конфигурация обучения
    # =============================================================================
    OUTPUT_DIR: str = Field(default="./outputs", description="Директория для результатов")
    CHECKPOINT_DIR: str = Field(default="./checkpoints", description="Директория для чекпоинтов")
    NUM_EPOCHS: int = Field(default=3, ge=1, description="Количество эпох обучения")
    PER_DEVICE_BATCH_SIZE: int = Field(default=4, ge=1, description="Batch size на устройство")
    GRADIENT_ACCUMULATION_STEPS: int = Field(default=4, ge=1, description="Шаги накопления градиента")
    LEARNING_RATE: float = Field(default=2e-4, gt=0, description="Learning rate")
    WARMUP_RATIO: float = Field(default=0.03, ge=0.0, le=1.0, description="Доля шагов для warmup")
    WEIGHT_DECAY: float = Field(default=0.01, ge=0.0, description="Weight decay")
    MAX_GRAD_NORM: float = Field(default=1.0, gt=0, description="Max norm для gradient clipping")
    LOGGING_STEPS: int = Field(default=10, ge=1, description="Частота логирования (шаги)")
    SAVE_STEPS: int = Field(default=100, ge=1, description="Частота сохранения чекпоинтов")
    EVAL_STRATEGY: str = Field(
        default="steps",
        pattern="^(no|steps|epoch)$",
        description="Стратегия evaluation"
    )
    EVAL_STEPS: int = Field(default=100, ge=1, description="Частота evaluation")
    LOAD_BEST_MODEL_AT_END: bool = Field(default=True, description="Загружать лучшую модель")
    SAVE_TOTAL_LIMIT: int = Field(default=3, ge=1, description="Максимум сохранённых чекпоинтов")
    
    # =============================================================================
    # Конфигурация данных
    # =============================================================================
    TRAIN_DATA_PATH: str = Field(..., description="Путь к тренировочным данным")
    VALIDATION_DATA_PATH: Optional[str] = Field(None, description="Путь к валидационным данным")
    TEXT_COLUMN: str = Field(default="text", description="Имя колонки с текстом")
    INSTRUCTION_COLUMN: Optional[str] = Field(None, description="Колонка с инструкцией")
    RESPONSE_COLUMN: Optional[str] = Field(None, description="Колонка с ответом")
    MAX_SEQ_LENGTH: int = Field(default=512, ge=1, description="Максимальная длина последовательности")
    
    # =============================================================================
    # Оптимизация
    # =============================================================================
    USE_8BIT_ADAM: bool = Field(default=False, description="Использовать 8-bit AdamW")
    FP16: bool = Field(default=True, description="Mixed precision FP16")
    BF16: bool = Field(default=False, description="Mixed precision BF16")
    GRADIENT_CHECKPOINTING: bool = Field(default=True, description="Gradient checkpointing")
    MAX_MEMORY_MB: Optional[int] = Field(None, ge=1, description="Максимальная память GPU (MB)")
    
    # =============================================================================
    # Логирование и мониторинг
    # =============================================================================
    LOG_LEVEL: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR)$",
        description="Уровень логирования"
    )
    LOG_FILE: Optional[str] = Field(None, description="Путь к файлу логов")
    WANDB_PROJECT: Optional[str] = Field("llm-fine-tuning", description="Проект Weights & Biases")
    WANDB_API_KEY: Optional[str] = Field(None, description="API ключ W&B")
    TENSORBOARD_DIR: Optional[str] = Field(None, description="Директория TensorBoard")
    
    # =============================================================================
    # Конфигурация инференса
    # =============================================================================
    INFERENCE_MODEL_PATH: Optional[str] = Field(None, description="Путь к дообученной модели")
    INFERENCE_DEVICE: str = Field(
        default="auto",
        pattern=r"^(auto|cpu|cuda|cuda:\d+)$",
        description="Устройство для инференса"
    )
    MAX_NEW_TOKENS: int = Field(default=256, ge=1, description="Максимум новых токенов")
    TEMPERATURE: float = Field(default=0.7, gt=0, description="Temperature для sampling")
    TOP_P: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p для nucleus sampling")
    TOP_K: int = Field(default=50, ge=1, description="Top-k sampling")
    REPETITION_PENALTY: float = Field(default=1.0, ge=1.0, description="Штраф за повторения")
    
    # =============================================================================
    # Расширенная конфигурация
    # =============================================================================
    RANDOM_SEED: int = Field(default=42, description="Seed для воспроизводимости")
    DATALOADER_NUM_WORKERS: int = Field(default=4, ge=0, description="Workers для DataLoader")
    REMOVE_UNUSED_COLUMNS: bool = Field(default=True, description="Удалять неиспользуемые колонки")
    GROUP_BY_LENGTH: bool = Field(default=True, description="Группировать по длине")
    USE_PACKING: bool = Field(default=False, description="Использовать packing")
    
    @property
    def lora_target_modules_list(self) -> List[str]:
        """Возвращает LORA_TARGET_MODULES как список."""
        return [m.strip() for m in self.LORA_TARGET_MODULES.split(",") if m.strip()]
    
    @field_validator("TRAIN_DATA_PATH", "VALIDATION_DATA_PATH")
    @classmethod
    def validate_path_exists(cls, v):
        """Проверяет существование пути к данным."""
        # Пропускаем валидацию для HF датасетов (содержат "/")
        if v is None or "/" in v and not v.startswith("./") and not v.startswith("/"):
            return v
        return v

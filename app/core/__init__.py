"""Базовая инфраструктура приложения.

Модуль предоставляет:
- Settings: конфигурация приложения через pydantic-settings
- Исключения: иерархия ошибок для обработки ошибок
- Логирование: настройка логирования с ротацией файлов
"""

from app.core.config import Settings
from app.core.exceptions import (
    FineTuningError,
    ConfigurationError,
    ModelLoadError,
    LoRAError,
    DataLoadError,
    DataFormatError,
    TrainingError,
    InferenceError,
    CheckpointError,
    ExportError,
)
from app.core.logging_config import setup_logging, get_logger

__all__ = [
    "Settings",
    "FineTuningError",
    "ConfigurationError",
    "ModelLoadError",
    "LoRAError",
    "DataLoadError",
    "DataFormatError",
    "TrainingError",
    "InferenceError",
    "CheckpointError",
    "ExportError",
    "setup_logging",
    "get_logger",
]

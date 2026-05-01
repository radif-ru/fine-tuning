"""Модуль для обучения моделей.

Включает конфигурацию, тренер и callbacks.
"""

from .callbacks import LoggingCallback, WandBCallback
from .config import LoRATrainingConfig
from .trainer import LoRATrainer

__all__ = [
    "LoRATrainingConfig",
    "LoRATrainer",
    "LoggingCallback",
    "WandBCallback",
]

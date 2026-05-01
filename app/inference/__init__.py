"""Модуль инференса.

Модуль предоставляет инструменты для:
- Генерации текста с использованием обученных моделей
- Загрузки моделей с LoRA адаптерами
- Batch и streaming генерации
- Экспорта объединённых моделей
- Интерактивного CLI режима
"""

from app.inference.config import GenerationConfig, get_default_config
from app.inference.engine import InferenceEngine
from app.inference.export import ModelExporter, export_model
from app.inference.prompt import PromptBuilder

__all__ = [
    "GenerationConfig",
    "get_default_config",
    "InferenceEngine",
    "ModelExporter",
    "export_model",
    "PromptBuilder",
]

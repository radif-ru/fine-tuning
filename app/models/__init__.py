"""Модуль работы с моделями.

Модуль предоставляет:
- BaseModelLoader: загрузка моделей из HuggingFace Hub
- LoRAManager: управление LoRA адаптерами
- ModelRegistry: реестр предустановленных конфигураций моделей
- LoRAConfig: конфигурация LoRA параметров
"""

from app.models.base import BaseModelLoader, ModelInfo
from app.models.lora import LoRAManager
from app.models.lora_config import LoRAConfig
from app.models.registry import ModelRegistry, get_registry

__all__ = [
    "BaseModelLoader",
    "ModelInfo",
    "LoRAManager",
    "LoRAConfig",
    "ModelRegistry",
    "get_registry",
]

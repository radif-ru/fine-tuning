"""Реестр поддерживаемых моделей.

Централизованное хранение конфигураций и особенностей моделей.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.exceptions import ModelLoadError


@dataclass
class ModelConfig:
    """Конфигурация модели.
    
    Attributes:
        name: Имя модели в HuggingFace Hub
        family: Семейство моделей ('llama', 'gpt', 'phi', etc)
        context_length: Максимальная длина контекста
        target_modules: Модули для LoRA (специфичные для архитектуры)
        special_tokens: Специальные токены
        default_lora_r: Рекомендуемый LoRA rank
        default_lora_alpha: Рекомендуемый LoRA alpha
        supports_8bit: Поддерживает ли 8-bit quantization
        supports_bf16: Поддерживает ли bfloat16
    """
    name: str
    family: str
    context_length: int = 2048
    target_modules: List[str] = field(default_factory=list)
    special_tokens: Dict[str, str] = field(default_factory=dict)
    default_lora_r: int = 8
    default_lora_alpha: int = 16
    supports_8bit: bool = True
    supports_bf16: bool = True


class ModelRegistry:
    """Реестр моделей с предустановленными конфигурациями.
    
    Регистрирует известные модели и их особенности для корректной
    настройки LoRA и других параметров.
    """
    
    def __init__(self):
        """Инициализация реестра с предустановленными моделями."""
        self._models: Dict[str, ModelConfig] = {}
        self._register_defaults()
    
    def register(self, name: str, config: ModelConfig) -> None:
        """Зарегистрировать модель в реестре.
        
        Args:
            name: Краткое имя модели (например, 'tinyllama-1.1b')
            config: Конфигурация модели
        """
        self._models[name] = config
    
    def get_config(self, name: str) -> ModelConfig:
        """Получить конфигурацию модели.
        
        Args:
            name: Имя модели (краткое или полное из HF Hub)
        
        Returns:
            Конфигурация модели
        
        Raises:
            ModelLoadError: Если модель не найдена
        """
        # Поиск по краткому имени
        if name in self._models:
            return self._models[name]
        
        # Поиск по полному имени
        for config in self._models.values():
            if config.name == name:
                return config
        
        # Если не найдена, возвращаем generic конфиг
        return ModelConfig(
            name=name,
            family="unknown",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
    
    def list_models(self) -> List[str]:
        """Получить список зарегистрированных моделей.
        
        Returns:
            Список кратких имён моделей
        """
        return list(self._models.keys())
    
    def is_registered(self, name: str) -> bool:
        """Проверить зарегистрирована ли модель.
        
        Args:
            name: Имя модели
        
        Returns:
            True если модель зарегистрирована
        """
        return name in self._models or any(
            config.name == name for config in self._models.values()
        )
    
    def get_target_modules(self, model_name: str) -> List[str]:
        """Получить target modules для модели.
        
        Args:
            model_name: Имя модели
        
        Returns:
            Список модулей для LoRA
        """
        config = self.get_config(model_name)
        return config.target_modules if config.target_modules else [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]
    
    def _register_defaults(self) -> None:
        """Регистрация предустановленных моделей."""
        # TinyLlama
        self.register("tinyllama-1.1b", ModelConfig(
            name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            family="llama",
            context_length=2048,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            default_lora_r=8,
            default_lora_alpha=16,
            supports_bf16=True,
        ))
        
        # Phi-3 Mini
        self.register("phi-3-mini", ModelConfig(
            name="microsoft/Phi-3-mini-4k-instruct",
            family="phi",
            context_length=4096,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            default_lora_r=16,
            default_lora_alpha=32,
            supports_bf16=True,
        ))
        
        # GPT-2
        self.register("gpt2", ModelConfig(
            name="gpt2",
            family="gpt",
            context_length=1024,
            target_modules=["c_attn", "c_proj"],
            default_lora_r=4,
            default_lora_alpha=8,
            supports_bf16=False,
        ))
        
        # GPT-2 Medium
        self.register("gpt2-medium", ModelConfig(
            name="gpt2-medium",
            family="gpt",
            context_length=1024,
            target_modules=["c_attn", "c_proj"],
            default_lora_r=4,
            default_lora_alpha=8,
            supports_bf16=False,
        ))
        
        # Mistral 7B
        self.register("mistral-7b", ModelConfig(
            name="mistralai/Mistral-7B-v0.1",
            family="llama",
            context_length=8192,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            default_lora_r=16,
            default_lora_alpha=32,
            supports_bf16=True,
        ))
        
        # Llama 2 7B
        self.register("llama2-7b", ModelConfig(
            name="meta-llama/Llama-2-7b-hf",
            family="llama",
            context_length=4096,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            default_lora_r=16,
            default_lora_alpha=32,
            supports_bf16=True,
        ))


# Глобальный экземпляр реестра
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Получить глобальный экземпляр реестра.
    
    Returns:
        ModelRegistry
    """
    return _registry

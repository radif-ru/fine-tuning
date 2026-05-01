"""Конфигурация для генерации текста.

Dataclass для параметров генерации с валидацией значений.
"""

from dataclasses import dataclass, field
from typing import Optional

from app.core.config import Settings
from app.core.logging_config import get_logger

logger = get_logger("inference.config")


@dataclass
class GenerationConfig:
    """Конфигурация параметров генерации текста.
    
    Attributes:
        max_new_tokens: Максимальное количество новых токенов
        temperature: Температура для sampling (> 0)
        top_p: Nucleus sampling probability [0, 1]
        top_k: Top-k sampling
        repetition_penalty: Штраф за повторения (1.0 = без штрафа)
        do_sample: Использовать sampling или greedy decoding
    """
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    def __post_init__(self):
        """Валидация значений после инициализации."""
        self._validate()
    
    def _validate(self) -> None:
        """Валидация параметров генерации.
        
        Raises:
            ValueError: При невалидных значениях параметров
        """
        errors = []
        
        if self.max_new_tokens < 1:
            errors.append(f"max_new_tokens должен быть >= 1, получено {self.max_new_tokens}")
        
        if self.temperature <= 0:
            errors.append(f"temperature должен быть > 0, получено {self.temperature}")
        
        if not 0 <= self.top_p <= 1:
            errors.append(f"top_p должен быть в [0, 1], получено {self.top_p}")
        
        if self.top_k < 1:
            errors.append(f"top_k должен быть >= 1, получено {self.top_k}")
        
        if self.repetition_penalty < 1.0:
            errors.append(f"repetition_penalty должен быть >= 1.0, получено {self.repetition_penalty}")
        
        if errors:
            raise ValueError("; ".join(errors))
    
    def to_dict(self) -> dict:
        """Преобразовать конфигурацию в словарь.
        
        Returns:
            Словарь с параметрами генерации
        """
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }
    
    @classmethod
    def from_env(cls, settings: Optional[Settings] = None) -> "GenerationConfig":
        """Создать конфигурацию из Settings.
        
        Args:
            settings: Экземпляр Settings (если None, создаётся новый)
        
        Returns:
            GenerationConfig с параметрами из окружения
        """
        if settings is None:
            settings = Settings()
        
        # Получаем значения из Settings с fallback на значения по умолчанию
        return cls(
            max_new_tokens=getattr(settings, "MAX_NEW_TOKENS", 256),
            temperature=getattr(settings, "TEMPERATURE", 0.7),
            top_p=getattr(settings, "TOP_P", 0.9),
            top_k=getattr(settings, "TOP_K", 50),
            repetition_penalty=getattr(settings, "REPETITION_PENALTY", 1.0),
            do_sample=getattr(settings, "DO_SAMPLE", True),
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "GenerationConfig":
        """Создать конфигурацию из словаря.
        
        Args:
            config_dict: Словарь с параметрами
        
        Returns:
            GenerationConfig
        """
        return cls(
            max_new_tokens=config_dict.get("max_new_tokens", 256),
            temperature=config_dict.get("temperature", 0.7),
            top_p=config_dict.get("top_p", 0.9),
            top_k=config_dict.get("top_k", 50),
            repetition_penalty=config_dict.get("repetition_penalty", 1.0),
            do_sample=config_dict.get("do_sample", True),
        )


def get_default_config() -> GenerationConfig:
    """Получить конфигурацию по умолчанию.
    
    Returns:
        GenerationConfig со значениями по умолчанию
    """
    return GenerationConfig()

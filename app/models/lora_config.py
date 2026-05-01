"""Конфигурация LoRA для дообучения.

LoRA (Low-Rank Adaptation) - параметро-эффективное дообучение.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from peft import LoraConfig as PEFTLoraConfig

from app.core.config import Settings


@dataclass
class LoRAConfig:
    """Конфигурация LoRA адаптера.
    
    Attributes:
        r: LoRA rank (размер low-rank матриц)
        lora_alpha: Scaling factor для LoRA
        lora_dropout: Dropout rate для LoRA слоёв
        target_modules: Модули для применения LoRA
        bias: Режим обучения bias ('none', 'all', 'lora_only')
        task_type: Тип задачи ('CAUSAL_LM', 'SEQ_CLS', etc)
    """
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        """Валидация параметров после инициализации."""
        if self.r < 1:
            raise ValueError(f"r должен быть >= 1, получено: {self.r}")
        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha должен быть >= 1, получено: {self.lora_alpha}")
        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout должен быть в [0, 1], получено: {self.lora_dropout}")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"bias должен быть 'none', 'all' или 'lora_only', получено: {self.bias}")
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "LoRAConfig":
        """Создать LoRAConfig из Settings.
        
        Args:
            settings: Конфигурация приложения
        
        Returns:
            LoRAConfig с параметрами из Settings
        """
        return cls(
            r=settings.LORA_R,
            lora_alpha=settings.LORA_ALPHA,
            lora_dropout=settings.LORA_DROPOUT,
            target_modules=settings.lora_target_modules_list,
            bias=settings.LORA_BIAS,
            task_type=settings.LORA_TASK_TYPE,
        )
    
    def to_peft_config(self) -> PEFTLoraConfig:
        """Конвертировать в PEFT LoraConfig.
        
        Returns:
            PEFT LoraConfig для использования с peft
        """
        return PEFTLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=self.task_type,
        )
    
    def get_scaling(self) -> float:
        """Получить scaling factor для LoRA.
        
        Returns:
            alpha / r
        """
        return self.lora_alpha / self.r
    
    def to_dict(self) -> dict:
        """Конвертировать в словарь.
        
        Returns:
            Словарь с параметрами конфигурации
        """
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }

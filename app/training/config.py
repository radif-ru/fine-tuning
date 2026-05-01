"""Конфигурация для обучения.

LoRATrainingConfig для настройки тренировочного процесса.
"""

from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments

from app.core.config import Settings


@dataclass
class LoRATrainingConfig:
    """Конфигурация для обучения LoRA.
    
    Attributes:
        num_train_epochs: Количество эпох обучения
        per_device_train_batch_size: Batch size на устройство
        per_device_eval_batch_size: Batch size для evaluation
        learning_rate: Learning rate
        warmup_steps: Шаги warmup
        max_steps: Максимальное количество шагов (-1 = по эпохам)
        optim: Оптимизатор ('adamw_torch', 'adamw_bnb_8bit')
        weight_decay: Weight decay
        max_grad_norm: Max norm для gradient clipping
        lr_scheduler_type: Тип scheduler ('linear', 'cosine', etc)
        gradient_accumulation_steps: Шаги накопления градиента
        fp16: Использовать FP16
        bf16: Использовать BF16
        logging_steps: Частота логирования
        save_steps: Частота сохранения чекпоинтов
        eval_steps: Частота evaluation
        save_total_limit: Максимум сохранённых чекпоинтов
        resume_from_checkpoint: Путь для resume
    """
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = -1
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        """Валидация параметров."""
        if self.num_train_epochs < 1:
            raise ValueError(f"num_train_epochs должен быть >= 1, получено: {self.num_train_epochs}")
        if self.per_device_train_batch_size < 1:
            raise ValueError(f"per_device_train_batch_size должен быть >= 1")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate должен быть > 0")
        if self.optim not in ("adamw_torch", "adamw_bnb_8bit", "adamw_8bit"):
            raise ValueError(f"Неподдерживаемый оптимизатор: {self.optim}")
    
    @classmethod
    def from_settings(cls, settings: Settings) -> "LoRATrainingConfig":
        """Создать конфиг из Settings.
        
        Args:
            settings: Настройки приложения
        
        Returns:
            LoRATrainingConfig
        """
        return cls(
            num_train_epochs=settings.NUM_EPOCHS,
            per_device_train_batch_size=settings.PER_DEVICE_BATCH_SIZE,
            learning_rate=settings.LEARNING_RATE,
            warmup_steps=int(settings.WARMUP_RATIO * 1000),  # Примерная конвертация
            optim="adamw_8bit" if settings.USE_8BIT_ADAM else "adamw_torch",
            weight_decay=settings.WEIGHT_DECAY,
            max_grad_norm=settings.MAX_GRAD_NORM,
            gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
            fp16=settings.FP16,
            bf16=settings.BF16,
            logging_steps=settings.LOGGING_STEPS,
            save_steps=settings.SAVE_STEPS,
            eval_steps=settings.EVAL_STEPS,
            save_total_limit=settings.SAVE_TOTAL_LIMIT,
        )
    
    def to_training_arguments(
        self,
        output_dir: str,
        evaluation_strategy: str = "no"
    ) -> TrainingArguments:
        """Конвертировать в HF TrainingArguments.
        
        Args:
            output_dir: Директория для вывода
            evaluation_strategy: Стратегия evaluation
        
        Returns:
            TrainingArguments
        """
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps if self.max_steps > 0 else None,
            optim=self.optim,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type=self.lr_scheduler_type,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.fp16,
            bf16=self.bf16,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps if evaluation_strategy != "no" else None,
            evaluation_strategy=evaluation_strategy,
            save_total_limit=self.save_total_limit,
            logging_first_step=True,
            logging_nan_inf_filter=True,
            load_best_model_at_end=False,
            report_to="none",  # Мы управляем логированием через callbacks
            remove_unused_columns=False,
        )
    
    def to_dict(self) -> dict:
        """Конвертировать в словарь.
        
        Returns:
            Словарь с параметрами
        """
        return {
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "learning_rate": self.learning_rate,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "fp16": self.fp16,
            "bf16": self.bf16,
        }

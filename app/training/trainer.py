"""Тренер LoRA для обучения моделей.

Обёртка над transformers.Trainer с поддержкой LoRA.
"""

from pathlib import Path
from typing import Optional

from datasets import Dataset
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer, Trainer

from app.core.exceptions import TrainingError
from app.core.logging_config import get_logger

from .callbacks import LoggingCallback, ProgressCallback
from .config import LoRATrainingConfig

logger = get_logger("training.trainer")


class LoRATrainer:
    """Тренер для LoRA обучения.
    
    Интегрирует:
    - Модель с LoRA (PeftModel)
    - Токенизатор
    - Конфигурацию обучения
    - Callbacks для логирования
    """
    
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        config: LoRATrainingConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None
    ):
        """Инициализация тренера.
        
        Args:
            model: Модель с LoRA адаптером
            tokenizer: Токенизатор
            config: Конфигурация обучения
            train_dataset: Тренировочный датасет
            eval_dataset: Валидационный датасет
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Создаём внутренний HF Trainer
        self._trainer: Optional[Trainer] = None
        
        # Логирование информации
        self._log_model_info()
    
    def train(
        self,
        output_dir: str,
        resume_from_checkpoint: Optional[str] = None
    ) -> None:
        """Запустить обучение.
        
        Args:
            output_dir: Директория для сохранения результатов
            resume_from_checkpoint: Путь для продолжения обучения
        
        Raises:
            TrainingError: При ошибке обучения
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Создаём TrainingArguments
            evaluation_strategy = "steps" if self.eval_dataset else "no"
            training_args = self.config.to_training_arguments(
                output_dir=str(output_path),
                evaluation_strategy=evaluation_strategy
            )
            
            # Колбэки
            callbacks = [
                LoggingCallback(),
                ProgressCallback(),
            ]
            
            # Создаём Trainer
            self._trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                processing_class=self.tokenizer,
                callbacks=callbacks,
            )
            
            logger.info(
                f"Начало обучения | "
                f"epochs={self.config.num_train_epochs}, "
                f"batch_size={self.config.per_device_train_batch_size}, "
                f"lr={self.config.learning_rate}"
            )
            
            # Запуск обучения
            checkpoint = resume_from_checkpoint or self.config.resume_from_checkpoint
            self._trainer.train(resume_from_checkpoint=checkpoint)
            
            logger.info("Обучение завершено успешно")
            
        except Exception as e:
            raise TrainingError(
                "Ошибка во время обучения",
                error=str(e)
            )
    
    def save_model(self, output_dir: str) -> None:
        """Сохранить модель.

        Args:
            output_dir: Директория для сохранения
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self._trainer:
            self._trainer.save_model(str(output_path))
        else:
            self.model.save_pretrained(str(output_path))

        self.tokenizer.save_pretrained(str(output_path))

        logger.info(f"Модель сохранена | path={output_dir}")
    
    def save_adapter(self, output_dir: str) -> None:
        """Сохранить только LoRA адаптер.
        
        Args:
            output_dir: Директория для сохранения
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(str(output_path))
            logger.info(f"LoRA адаптер сохранён | path={output_dir}")
        else:
            logger.warning("Модель не имеет LoRA адаптера, сохраняем полную модель")
            self.save_model(output_dir)
    
    def get_trainable_params_count(self) -> int:
        """Получить количество обучаемых параметров.
        
        Returns:
            Количество обучаемых параметров
        """
        trainable = 0
        for param in self.model.parameters():
            if param.requires_grad:
                trainable += param.numel()
        return trainable
    
    def get_total_params_count(self) -> int:
        """Получить общее количество параметров.
        
        Returns:
            Общее количество параметров
        """
        total = 0
        for param in self.model.parameters():
            total += param.numel()
        return total
    
    def print_trainable_parameters(self) -> None:
        """Вывести информацию об обучаемых параметрах."""
        trainable = self.get_trainable_params_count()
        total = self.get_total_params_count()
        percentage = 100 * trainable / total if total > 0 else 0
        
        logger.info(
            f"Параметры модели | "
            f"trainable={trainable:,} ({percentage:.2f}%), "
            f"total={total:,}"
        )
    
    def _log_model_info(self) -> None:
        """Логирование информации о модели."""
        self.print_trainable_parameters()

"""Callbacks для тренировки.

Логирование метрик, интеграция с W&B.
"""

import time
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from app.core.logging_config import get_logger

logger = get_logger("training.callbacks")


class LoggingCallback(TrainerCallback):
    """Callback для логирования метрик тренировки.
    
    Логирует loss, learning rate, прогресс обучения.
    """
    
    def __init__(self):
        """Инициализация callback."""
        super().__init__()
        self.start_time = None
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Вызывается в начале тренировки."""
        self.start_time = time.time()
        logger.info("Начало тренировки")
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """Вызывается при логировании."""
        if logs is None:
            return
        
        step = state.global_step
        
        # Формируем сообщение
        msg_parts = [f"Шаг {step}"]
        
        if "loss" in logs:
            msg_parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            msg_parts.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            msg_parts.append(f"epoch={logs['epoch']:.2f}")
        
        logger.info(" | ".join(msg_parts))
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Вызывается в конце эпохи."""
        epoch = int(state.epoch)
        logger.info(f"Эпоха {epoch} завершена")
    
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Вызывается при сохранении чекпоинта."""
        logger.info(f"Чекпоинт сохранён | step={state.global_step}")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Вызывается в конце тренировки."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        logger.info(f"Тренировка завершена | время={elapsed:.1f}s")


class WandBCallback(TrainerCallback):
    """Callback для интеграции с Weights & Biases.
    
    Опциональный callback, активируется только если wandb установлен.
    """
    
    def __init__(
        self,
        project: str = "llm-fine-tuning",
        name: Optional[str] = None,
        config: Optional[dict] = None
    ):
        """Инициализация callback.
        
        Args:
            project: Имя проекта в W&B
            name: Имя run
            config: Конфигурация для логирования
        """
        super().__init__()
        self.project = project
        self.name = name
        self.config = config or {}
        self._wandb = None
        self._initialized = False
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Инициализация W&B при начале тренировки."""
        try:
            import wandb
            self._wandb = wandb
            
            if not self._wandb.run:
                self._wandb.init(
                    project=self.project,
                    name=self.name,
                    config=self.config
                )
            
            self._initialized = True
            logger.info(f"W&B инициализирован | project={self.project}")
        
        except ImportError:
            logger.warning("W&B не установлен, логирование отключено")
        except Exception as e:
            logger.warning(f"Не удалось инициализировать W&B: {e}")
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """Логирование метрик в W&B."""
        if not self._initialized or logs is None:
            return
        
        try:
            self._wandb.log(logs, step=state.global_step)
        except Exception as e:
            logger.debug(f"Ошибка логирования в W&B: {e}")
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Завершение W&B при окончании тренировки."""
        if self._initialized and self._wandb:
            try:
                self._wandb.finish()
                logger.info("W&B завершён")
            except Exception as e:
                logger.debug(f"Ошибка завершения W&B: {e}")


class ProgressCallback(TrainerCallback):
    """Callback для отображения прогресса через tqdm."""
    
    def __init__(self):
        """Инициализация callback."""
        super().__init__()
        self._progress_bar = None
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Создание progress bar."""
        try:
            from tqdm.auto import tqdm
            
            total_steps = state.max_steps or args.num_train_epochs * 1000
            self._progress_bar = tqdm(
                total=total_steps,
                desc="Обучение",
                unit="step"
            )
        except ImportError:
            pass
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Обновление progress bar."""
        if self._progress_bar:
            self._progress_bar.update(1)
            self._progress_bar.set_postfix({
                "loss": state.log_history[-1].get("loss", 0) if state.log_history else 0
            })
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Закрытие progress bar."""
        if self._progress_bar:
            self._progress_bar.close()

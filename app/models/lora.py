"""Управление LoRA адаптерами.

Классы для применения, сохранения и загрузки LoRA адаптеров.
"""

from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel, get_peft_model
from transformers import PreTrainedModel

from app.core.exceptions import LoRAError
from app.core.logging_config import get_logger

from .lora_config import LoRAConfig

logger = get_logger("models.lora")


class LoRAManager:
    """Менеджер для работы с LoRA адаптерами.
    
    Отвечает за:
    - Применение LoRA к базовой модели
    - Сохранение адаптеров
    - Загрузку адаптеров
    - Merge адаптера с базовой моделью
    """
    
    def __init__(self):
        """Инициализация менеджера."""
        self._peft_model: Optional[PeftModel] = None
    
    def apply_lora(
        self,
        model: PreTrainedModel,
        config: LoRAConfig
    ) -> PeftModel:
        """Применить LoRA к базовой модели.
        
        Args:
            model: Базовая модель
            config: Конфигурация LoRA
        
        Returns:
            Модель с LoRA адаптером
        
        Raises:
            LoRAError: При ошибке применения LoRA
        """
        try:
            peft_config = config.to_peft_config()
            peft_model = get_peft_model(model, peft_config)
            
            self._peft_model = peft_model
            
            # Логирование информации
            trainable_params, total_params = self._get_trainable_params(peft_model)
            logger.info(
                f"LoRA применена | rank={config.r}, alpha={config.lora_alpha}, "
                f"trainable={trainable_params:,} ({100 * trainable_params / total_params:.2f}%), "
                f"total={total_params:,}"
            )
            
            return peft_model
            
        except Exception as e:
            raise LoRAError(
                "Не удалось применить LoRA к модели",
                error=str(e)
            )
    
    def save_adapter(
        self,
        model: PeftModel,
        path: str,
        save_pretrained_kwargs: Optional[dict] = None
    ) -> None:
        """Сохранить LoRA адаптер.
        
        Args:
            model: Модель с LoRA адаптером
            path: Путь для сохранения
            save_pretrained_kwargs: Дополнительные аргументы для save_pretrained
        
        Raises:
            LoRAError: При ошибке сохранения
        """
        try:
            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            kwargs = save_pretrained_kwargs or {}
            model.save_pretrained(output_path, **kwargs)
            
            logger.info(f"LoRA адаптер сохранён | path={path}")
            
        except Exception as e:
            raise LoRAError(
                f"Не удалось сохранить LoRA адаптер: {path}",
                error=str(e)
            )
    
    def load_adapter(
        self,
        model: PreTrainedModel,
        path: str,
        adapter_name: str = "default"
    ) -> PeftModel:
        """Загрузить LoRA адаптер к базовой модели.
        
        Args:
            model: Базовая модель
            path: Путь к сохранённому адаптеру
            adapter_name: Имя адаптера
        
        Returns:
            Модель с загруженным адаптером
        
        Raises:
            LoRAError: При ошибке загрузки
        """
        try:
            peft_model = PeftModel.from_pretrained(
                model,
                path,
                adapter_name=adapter_name
            )
            
            self._peft_model = peft_model
            
            logger.info(f"LoRA адаптер загружен | path={path}, name={adapter_name}")
            
            return peft_model
            
        except Exception as e:
            raise LoRAError(
                f"Не удалось загрузить LoRA адаптер: {path}",
                error=str(e)
            )
    
    def merge_and_unload(self, model: PeftModel) -> PreTrainedModel:
        """Объединить LoRA адаптер с базовой моделью.
        
        Оптимизация для inference: объединяет веса LoRA
        в базовые веса модели для более быстрой генерации.
        
        Args:
            model: Модель с LoRA адаптером
        
        Returns:
            Базовая модель с объединёнными весами
        
        Raises:
            LoRAError: При ошибке объединения
        """
        try:
            merged_model = model.merge_and_unload()
            
            logger.info("LoRA адаптер объединён с базовой моделью")
            
            return merged_model
            
        except Exception as e:
            raise LoRAError(
                "Не удалось объединить LoRA адаптер с моделью",
                error=str(e)
            )
    
    def get_trainable_params_count(self, model: Optional[PeftModel] = None) -> int:
        """Получить количество обучаемых параметров.
        
        Args:
            model: Модель с LoRA (если None, используется сохранённая)
        
        Returns:
            Количество обучаемых параметров
        """
        if model is None:
            model = self._peft_model
        
        if model is None:
            return 0
        
        trainable_params, _ = self._get_trainable_params(model)
        return trainable_params
    
    @staticmethod
    def _get_trainable_params(model: torch.nn.Module) -> tuple:
        """Подсчёт обучаемых и общих параметров.
        
        Args:
            model: PyTorch модель
        
        Returns:
            Кортеж (обучаемые параметры, все параметры)
        """
        trainable_params = 0
        all_params = 0
        
        for _, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return trainable_params, all_params
    
    def print_trainable_parameters(self, model: Optional[PeftModel] = None) -> None:
        """Вывести информацию об обучаемых параметрах.
        
        Args:
            model: Модель с LoRA (если None, используется сохранённая)
        """
        if model is None:
            model = self._peft_model
        
        if model is None:
            logger.warning("Модель не загружена")
            return
        
        trainable_params, total_params = self._get_trainable_params(model)
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        logger.info(
            f"Параметры модели | "
            f"trainable={trainable_params:,} ({percentage:.2f}%), "
            f"all={total_params:,}"
        )

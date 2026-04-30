"""Export моделей: merge LoRA адаптера с базовой моделью.

Класс ModelExporter для объединения и сохранения полной модели.
"""

from pathlib import Path
from typing import Optional

from transformers import PreTrainedModel, PreTrainedTokenizer

from app.core.exceptions import ExportError
from app.core.logging_config import get_logger
from app.models.base import BaseModelLoader
from app.models.lora import LoRAManager

logger = get_logger("inference.export")


class ModelExporter:
    """Экспортёр моделей для merge и сохранения.
    
    Выполняет:
    - Загрузку базовой модели и адаптера
    - Merge LoRA в базовую модель
    - Сохранение полной модели и токенизатора
    """
    
    def __init__(self):
        """Инициализация экспортёра."""
        self._lora_manager = LoRAManager()
    
    def merge_and_save(
        self,
        base_model_name: str,
        adapter_path: str,
        output_path: str,
        save_tokenizer: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
    ) -> None:
        """Объединить LoRA адаптер с базовой моделью и сохранить.
        
        Args:
            base_model_name: Имя базовой модели (HF Hub или локальный путь)
            adapter_path: Путь к LoRA адаптеру
            output_path: Путь для сохранения объединённой модели
            save_tokenizer: Сохранить токенизатор
            load_in_8bit: Загрузить в 8-bit (не рекомендуется для export)
            device_map: Стратегия размещения на устройствах
        
        Raises:
            ExportError: При ошибке экспорта
        """
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(
                f"Начало экспорта | base={base_model_name}, "
                f"adapter={adapter_path}, output={output_path}"
            )
            
            # Загрузка базовой модели
            loader = BaseModelLoader()
            model, tokenizer = loader.load(
                model_name=base_model_name,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
            )
            
            logger.info(f"Базовая модель загружена | model={base_model_name}")
            
            # Загрузка LoRA адаптера
            model = self._lora_manager.load_adapter(
                model=model,
                path=adapter_path,
            )
            
            logger.info(f"LoRA адаптер загружен | path={adapter_path}")
            
            # Merge адаптера
            merged_model = self._lora_manager.merge_and_unload(model)
            
            logger.info("LoRA адаптер объединён с базовой моделью")
            
            # Сохранение модели
            merged_model.save_pretrained(output_dir)
            logger.info(f"Модель сохранена | path={output_dir}")
            
            # Сохранение токенизатора
            if save_tokenizer and tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
                logger.info(f"Токенизатор сохранён | path={output_dir}")
            
            logger.info(f"Экспорт завершён успешно | output={output_path}")
            
        except Exception as e:
            raise ExportError(
                f"Ошибка экспорта модели: {e}",
                error=str(e)
            )
    
    def validate_merged_model(
        self,
        model_path: str,
        test_prompt: str = "Hello, this is a test."
    ) -> bool:
        """Проверить, что объединённая модель работает.
        
        Args:
            model_path: Путь к сохранённой модели
            test_prompt: Тестовый промпт
        
        Returns:
            True если модель загружается и генерирует текст
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Валидация модели | path={model_path}")
            
            # Загрузка модели и токенизатора
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Тестовая генерация
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Валидация успешна | generated_length={len(result)}")
            return True
            
        except Exception as e:
            logger.error(f"Валидация не удалась: {e}")
            return False


def export_model(
    base_model: str,
    adapter_path: str,
    output_path: str,
    save_tokenizer: bool = True,
) -> None:
    """Функция для удобного экспорта модели.
    
    Args:
        base_model: Имя базовой модели
        adapter_path: Путь к адаптеру
        output_path: Путь для сохранения
        save_tokenizer: Сохранить токенизатор
    """
    exporter = ModelExporter()
    exporter.merge_and_save(
        base_model_name=base_model,
        adapter_path=adapter_path,
        output_path=output_path,
        save_tokenizer=save_tokenizer,
    )

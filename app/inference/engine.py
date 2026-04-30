"""Inference Engine для генерации текста.

Поддерживает загрузку моделей, LoRA адаптеров, генерацию текста
в различных режимах: single, batch, streaming.
"""

from typing import Iterator, List, Optional, Union

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from app.core.exceptions import InferenceError
from app.core.logging_config import get_logger
from app.models.base import BaseModelLoader
from app.models.lora import LoRAManager

from .config import GenerationConfig

logger = get_logger("inference.engine")


class InferenceEngine:
    """Движок для инференса языковых моделей.
    
    Поддерживает:
    - Загрузку базовых моделей через ModelRegistry
    - Загрузку LoRA адаптеров
    - Generation с параметрами (temperature, top_p, max_new_tokens)
    - Batch generation
    - Streaming generation через yield
    - Merge & unload для оптимизации
    """
    
    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[str] = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        torch_dtype: Optional[str] = None,
    ):
        """Инициализация Inference Engine.
        
        Args:
            base_model_name: Имя базовой модели (HF Hub или локальный путь)
            adapter_path: Путь к LoRA адаптеру (опционально)
            device: Устройство ('auto', 'cpu', 'cuda', 'cuda:0', etc)
            load_in_8bit: Загрузить в 8-bit режиме
            torch_dtype: Тип данных ('float16', 'bfloat16', 'float32')
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.torch_dtype = torch_dtype
        
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._lora_manager = LoRAManager()
        self._is_merged = False
        
        logger.info(
            f"InferenceEngine инициализирован | model={base_model_name}, "
            f"adapter={adapter_path}, device={device}"
        )
    
    def load(self) -> "InferenceEngine":
        """Загрузить модель и адаптер.
        
        Returns:
            self для chaining
        
        Raises:
            InferenceError: При ошибке загрузки
        """
        try:
            # Загрузка базовой модели
            loader = BaseModelLoader()
            device_map = self.device if self.device != "auto" else "auto"
            
            self._model, self._tokenizer = loader.load(
                model_name=self.base_model_name,
                load_in_8bit=self.load_in_8bit,
                device_map=device_map,
                torch_dtype=self.torch_dtype,
            )
            
            logger.info(f"Базовая модель загружена | model={self.base_model_name}")
            
            # Загрузка LoRA адаптера если указан
            if self.adapter_path:
                self._model = self._lora_manager.load_adapter(
                    model=self._model,
                    path=self.adapter_path,
                )
                logger.info(f"LoRA адаптер загружен | path={self.adapter_path}")
            
            return self
            
        except Exception as e:
            raise InferenceError(
                f"Ошибка загрузки модели: {e}",
                error=str(e)
            )
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """Сгенерировать текст по промпту.
        
        Args:
            prompt: Текст промпта
            config: Конфигурация генерации (по умолчанию GenerationConfig())
        
        Returns:
            Сгенерированный текст
        
        Raises:
            InferenceError: Если модель не загружена
        """
        if self._model is None or self._tokenizer is None:
            raise InferenceError("Модель не загружена. Вызовите load() сначала.")
        
        config = config or GenerationConfig()
        
        try:
            # Токенизация
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            # Перенос на устройство модели
            device = self._model.device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generation
            generation_kwargs = config.to_dict()
            generation_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
            generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Декодирование (убираем prompt из вывода)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            
            result = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            logger.debug(
                f"Генерация завершена | prompt_len={input_length}, "
                f"generated_len={len(generated_tokens)}"
            )
            
            return result
            
        except Exception as e:
            raise InferenceError(
                f"Ошибка генерации: {e}",
                error=str(e)
            )
    
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """Сгенерировать текст для нескольких промптов.
        
        Args:
            prompts: Список промптов
            config: Конфигурация генерации
        
        Returns:
            Список сгенерированных текстов
        """
        if self._model is None or self._tokenizer is None:
            raise InferenceError("Модель не загружена. Вызовите load() сначала.")
        
        config = config or GenerationConfig()
        
        results = []
        for i, prompt in enumerate(prompts):
            try:
                result = self.generate(prompt, config)
                results.append(result)
                logger.debug(f"Batch progress | {i+1}/{len(prompts)}")
            except Exception as e:
                logger.error(f"Ошибка генерации для prompt {i}: {e}")
                results.append("")
        
        logger.info(f"Batch generation завершена | count={len(prompts)}")
        return results
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """Сгенерировать текст с потоковой передачей токенов.
        
        Args:
            prompt: Текст промпта
            config: Конфигурация генерации
        
        Yields:
            Токены по мере генерации
        """
        if self._model is None or self._tokenizer is None:
            raise InferenceError("Модель не загружена. Вызовите load() сначала.")
        
        config = config or GenerationConfig()
        
        try:
            # Токенизация
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            device = self._model.device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Streaming generation
            input_length = inputs["input_ids"].shape[1]
            
            generation_kwargs = config.to_dict()
            generation_kwargs["pad_token_id"] = self._tokenizer.pad_token_id
            generation_kwargs["eos_token_id"] = self._tokenizer.eos_token_id
            
            # Используем streamer для потоковой передачи
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            generation_kwargs["streamer"] = streamer
            
            # Запуск generation в отдельном потоке
            generation_kwargs = {**inputs, **generation_kwargs}
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Yield токены по мере генерации
            for text in streamer:
                if text:
                    yield text
            
            thread.join()
            
        except Exception as e:
            raise InferenceError(
                f"Ошибка streaming generation: {e}",
                error=str(e)
            )
    
    def merge_and_unload(self) -> "InferenceEngine":
        """Объединить LoRA адаптер с базовой моделью.
        
        Оптимизация для inference: объединяет веса LoRA
        в базовые веса модели для более быстрой генерации.
        
        Returns:
            self для chaining
        """
        if self._model is None:
            raise InferenceError("Модель не загружена")
        
        if self._is_merged:
            logger.warning("Модель уже объединена")
            return self
        
        try:
            self._model = self._lora_manager.merge_and_unload(self._model)
            self._is_merged = True
            logger.info("LoRA адаптер объединён с базовой моделью")
            return self
            
        except Exception as e:
            raise InferenceError(
                f"Ошибка merge: {e}",
                error=str(e)
            )
    
    @property
    def is_loaded(self) -> bool:
        """Проверить загружена ли модель."""
        return self._model is not None and self._tokenizer is not None
    
    @property
    def is_merged(self) -> bool:
        """Проверить объединён ли адаптер."""
        return self._is_merged
    
    def get_model_info(self) -> dict:
        """Получить информацию о модели.
        
        Returns:
            Словарь с информацией о модели
        """
        if self._model is None:
            return {"status": "not_loaded"}
        
        config = self._model.config
        return {
            "status": "loaded",
            "base_model": self.base_model_name,
            "adapter": self.adapter_path,
            "merged": self._is_merged,
            "architecture": getattr(config, "architectures", ["unknown"])[0] if hasattr(config, "architectures") else "unknown",
            "vocab_size": getattr(config, "vocab_size", 0),
        }

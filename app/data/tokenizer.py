"""Обёртка над токенизатором для обучения.

Токенизация с поддержкой truncation, padding, labels для causal LM.
"""

from typing import Dict, Optional, Union

from transformers import PreTrainedTokenizer

from app.core.exceptions import DataFormatError
from app.core.logging_config import get_logger

logger = get_logger("data.tokenizer")


class TokenizerWrapper:
    """Обёртка над токенизатором для удобного использования.
    
    Поддерживает:
    - Tokenization с truncation
    - Padding (max_length или batch)
    - Подготовку labels для causal LM
    - Chat templates для instruct моделей
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        truncation: bool = True,
        padding: Union[bool, str] = "max_length"
    ):
        """Инициализация обёртки.
        
        Args:
            tokenizer: Токенизатор модели
            max_length: Максимальная длина последовательности
            truncation: Включить truncation
            padding: Стратегия padding ('max_length', True, False)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding
        
        # Установка pad_token если не задан
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Установлен pad_token = eos_token ({tokenizer.pad_token})")
    
    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict:
        """Токенизировать текст.
        
        Args:
            text: Текст для токенизации
            add_special_tokens: Добавлять специальные токены
            return_tensors: Формат возврата ('pt', 'np', None)
        
        Returns:
            Словарь с input_ids, attention_mask
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=add_special_tokens,
            return_tensors=return_tensors
        )
    
    def tokenize_dataset(
        self,
        dataset,
        text_column: str = "text"
    ):
        """Токенизировать датасет.
        
        Args:
            dataset: Датасет для токенизации
            text_column: Имя колонки с текстом
        
        Returns:
            Токенизированный датасет
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                add_special_tokens=True
            )
        
        return dataset.map(
            tokenize_function,
            batched=True,
            desc="Токенизация",
            remove_columns=[text_column]
        )
    
    def prepare_for_training(
        self,
        dataset,
        text_column: str = "text"
    ):
        """Подготовить датасет для обучения causal LM.
        
        Добавляет labels = input_ids для предсказания следующего токена.
        
        Args:
            dataset: Токенизированный датасет
            text_column: Колонка с текстом (будет удалена)
        
        Returns:
            Датасет готовый для обучения
        """
        def prepare_function(examples):
            # Токенизация
            result = self.tokenizer(
                examples[text_column],
                max_length=self.max_length,
                truncation=self.truncation,
                padding=self.padding,
                add_special_tokens=True
            )
            
            # Для causal LM: labels = input_ids
            result["labels"] = result["input_ids"].copy()
            
            return result
        
        return dataset.map(
            prepare_function,
            batched=True,
            desc="Подготовка для обучения",
            remove_columns=[text_column]
        )
    
    def apply_chat_template(
        self,
        messages: list,
        add_generation_prompt: bool = False
    ) -> str:
        """Применить chat template если доступен.
        
        Args:
            messages: Список сообщений
            add_generation_prompt: Добавить промпт для генерации
        
        Returns:
            Отформатированная строка
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False
                )
            except Exception as e:
                logger.warning(f"Не удалось применить chat_template: {e}")
        
        # Fallback: ручное форматирование
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                lines.append(f"System: {content}")
            elif role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
        
        if add_generation_prompt:
            lines.append("Assistant:")
        
        return "\n".join(lines)
    
    def encode_conversation(
        self,
        messages: list,
        max_length: Optional[int] = None
    ) -> Dict:
        """Закодировать разговор в токены.
        
        Args:
            messages: Список сообщений
            max_length: Максимальная длина (переопределяет настройки)
        
        Returns:
            Токенизированный разговор
        """
        text = self.apply_chat_template(messages)
        max_len = max_length or self.max_length
        
        return self.tokenizer(
            text,
            max_length=max_len,
            truncation=self.truncation,
            padding=self.padding,
            add_special_tokens=True
        )
    
    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = True
    ) -> str:
        """Декодировать токены в текст.
        
        Args:
            token_ids: Список токенов
            skip_special_tokens: Пропускать специальные токены
        
        Returns:
            Декодированный текст
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def get_vocab_size(self) -> int:
        """Получить размер словаря.
        
        Returns:
            Размер словаря
        """
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """ID токена паддинга."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """ID токена конца последовательности."""
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """ID токена начала последовательности."""
        return self.tokenizer.bos_token_id

"""Форматирование данных для обучения.

Конвертация различных форматов данных в единый формат для обучения.
"""

from typing import Callable, Optional

from datasets import Dataset

from app.core.exceptions import DataFormatError
from app.core.logging_config import get_logger

from .templates import (
    ALPACA_TEMPLATE,
    ALPACA_NO_INPUT_TEMPLATE,
    RAW_TEMPLATE,
    SHAREGPT_TEMPLATE,
    PromptTemplate,
    format_alpaca,
    format_sharegpt,
)

logger = get_logger("data.formatter")


class DataFormatter:
    """Форматирование данных для обучения.
    
    Поддерживает форматы:
    - Alpaca (instruction/input/output)
    - ShareGPT (conversations)
    - Raw (просто text)
    - Custom templates
    """
    
    def __init__(
        self,
        instruction_column: str = "instruction",
        input_column: str = "input",
        response_column: str = "output",
        text_column: str = "text"
    ):
        """Инициализация форматтера.
        
        Args:
            instruction_column: Имя колонки с инструкцией
            input_column: Имя колонки с входными данными
            response_column: Имя колонки с ответом
            text_column: Имя колонки с текстом (для raw формата)
        """
        self.instruction_column = instruction_column
        self.input_column = input_column
        self.response_column = response_column
        self.text_column = text_column
    
    def format_alpaca(
        self,
        dataset: Dataset,
        system_prompt: Optional[str] = None
    ) -> Dataset:
        """Форматировать датасет в Alpaca формат.
        
        Args:
            dataset: Исходный датасет
            system_prompt: Системный промпт (игнорируется)
        
        Returns:
            Отформатированный датасет с колонкой 'text'
        """
        def format_example(example):
            instruction = example.get(self.instruction_column, "")
            input_text = example.get(self.input_column, "")
            output = example.get(self.response_column, "")
            
            if input_text:
                text = ALPACA_TEMPLATE.format(
                    instruction=instruction,
                    input=input_text,
                    output=output
                )
            else:
                text = ALPACA_NO_INPUT_TEMPLATE.format(
                    instruction=instruction,
                    output=output
                )
            
            return {"text": text}
        
        return dataset.map(format_example, desc="Форматирование Alpaca")
    
    def format_sharegpt(
        self,
        dataset: Dataset,
        conversations_column: str = "conversations",
        system_prompt: str = ""
    ) -> Dataset:
        """Форматировать датасет в ShareGPT формат.
        
        Args:
            dataset: Исходный датасет
            conversations_column: Имя колонки с conversations
            system_prompt: Системный промпт
        
        Returns:
            Отформатированный датасет с колонкой 'text'
        """
        def format_example(example):
            conversations = example.get(conversations_column, [])
            text = format_sharegpt(conversations, system_prompt)
            return {"text": text}
        
        return dataset.map(format_example, desc="Форматирование ShareGPT")
    
    def format_raw(
        self,
        dataset: Dataset,
        text_column: Optional[str] = None
    ) -> Dataset:
        """Оставить датасет как есть (raw формат).
        
        Args:
            dataset: Исходный датасет
            text_column: Имя колонки с текстом (если отличается)
        
        Returns:
            Датасет с колонкой 'text'
        """
        col = text_column or self.text_column
        
        if col == "text":
            # Уже в правильном формате
            return dataset
        
        # Переименовываем колонку
        return dataset.rename_column(col, "text")
    
    def format_custom(
        self,
        dataset: Dataset,
        template: PromptTemplate,
        column_mapping: Optional[dict] = None
    ) -> Dataset:
        """Форматировать датасет с кастомным шаблоном.
        
        Args:
            dataset: Исходный датасет
            template: Кастомный шаблон
            column_mapping: Маппинг колонок датасета на плейсхолдеры
        
        Returns:
            Отформатированный датасет
        """
        placeholders = template.get_placeholders()
        mapping = column_mapping or {}
        
        def format_example(example):
            kwargs = {}
            for placeholder in placeholders:
                # Используем маппинг если есть, иначе ищем колонку с таким же именем
                column = mapping.get(placeholder, placeholder)
                kwargs[placeholder] = example.get(column, "")
            
            try:
                text = template.format(**kwargs)
            except DataFormatError as e:
                logger.warning(f"Ошибка форматирования: {e}")
                text = ""
            
            return {"text": text}
        
        return dataset.map(format_example, desc=f"Форматирование {template.name}")
    
    def format_conversation(
        self,
        messages: list,
        system_prompt: str = "",
        add_generation_prompt: bool = False
    ) -> str:
        """Форматировать список сообщений в текст.
        
        Args:
            messages: Список сообщений [{"role": "...", "content": "..."}]
            system_prompt: Системный промпт
            add_generation_prompt: Добавить промпт для генерации
        
        Returns:
            Отформатированная строка
        """
        lines = []
        
        if system_prompt:
            lines.append(f"System: {system_prompt}")
        
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
    
    def auto_format(
        self,
        dataset: Dataset,
        format_type: str = "auto"
    ) -> Dataset:
        """Автоматически определить и применить форматирование.
        
        Args:
            dataset: Исходный датасет
            format_type: Тип формата ('alpaca', 'sharegpt', 'raw', 'auto')
        
        Returns:
            Отформатированный датасет
        """
        if format_type == "auto":
            format_type = self._detect_format(dataset)
            logger.info(f"Автоматически определён формат: {format_type}")
        
        if format_type == "alpaca":
            return self.format_alpaca(dataset)
        elif format_type == "sharegpt":
            return self.format_sharegpt(dataset)
        elif format_type == "raw":
            return self.format_raw(dataset)
        else:
            raise DataFormatError(f"Неподдерживаемый формат: {format_type}")
    
    def _detect_format(self, dataset: Dataset) -> str:
        """Автоматически определить формат датасета.
        
        Args:
            dataset: Датасет для анализа
        
        Returns:
            Определённый формат
        """
        columns = set(dataset.column_names)
        
        # Проверяем наличие специфичных колонок
        if "conversations" in columns:
            return "sharegpt"
        
        if self.instruction_column in columns and self.response_column in columns:
            return "alpaca"
        
        if self.text_column in columns:
            return "raw"
        
        # По умолчанию raw
        return "raw"

"""Полный пайплайн обработки данных.

Интеграция: загрузка → форматирование → токенизация.
"""

from typing import Optional

from datasets import Dataset

from app.core.exceptions import DataLoadError, DataFormatError
from app.core.logging_config import get_logger
from app.data.formatter import DataFormatter
from app.data.loader import DataLoader
from app.data.tokenizer import TokenizerWrapper

logger = get_logger("data.pipeline")


class DataPipeline:
    """Полный пайплайн обработки данных.
    
    Объединяет:
    - DataLoader: загрузка из разных источников
    - DataFormatter: форматирование в нужный формат
    - TokenizerWrapper: токенизация
    """
    
    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        formatter: Optional[DataFormatter] = None
    ):
        """Инициализация пайплайна.
        
        Args:
            loader: Загрузчик данных
            formatter: Форматтер данных
        """
        self.loader = loader or DataLoader()
        self.formatter = formatter or DataFormatter()
    
    def process(
        self,
        source: str,
        format_type: str = "auto",
        tokenizer_wrapper: Optional[TokenizerWrapper] = None,
        text_column: str = "text",
        split: str = "train",
        validate_columns: Optional[list] = None
    ) -> Dataset:
        """Обработать данные через полный пайплайн.
        
        Args:
            source: Путь к данным или имя HF датасета
            format_type: Тип формата ('alpaca', 'sharegpt', 'raw', 'auto')
            tokenizer_wrapper: Обёртка токенизатора (опционально)
            text_column: Имя колонки с текстом
            split: Сплит для HF датасетов
            validate_columns: Колонки для валидации
        
        Returns:
            Обработанный датасет
        
        Raises:
            DataLoadError: При ошибке загрузки
            DataFormatError: При ошибке форматирования
        """
        logger.info(f"Начало обработки данных | source={source}, format={format_type}")
        
        # 1. Загрузка
        try:
            dataset = self.loader.load(source, split=split)
            logger.info(f"Данные загружены | samples={len(dataset)}")
        except Exception as e:
            raise DataLoadError(f"Ошибка загрузки: {source}", error=str(e))
        
        # 2. Валидация
        if validate_columns:
            self.loader.validate(dataset, required_columns=validate_columns)
        
        # 3. Форматирование
        try:
            dataset = self.formatter.auto_format(dataset, format_type=format_type)
            logger.info(f"Данные отформатированы | format={format_type}")
        except Exception as e:
            raise DataFormatError(f"Ошибка форматирования: {format_type}", error=str(e))
        
        # 4. Токенизация (если указан токенизатор)
        if tokenizer_wrapper:
            dataset = tokenizer_wrapper.prepare_for_training(dataset, text_column=text_column)
            logger.info(f"Данные токенизированы | max_length={tokenizer_wrapper.max_length}")
        
        logger.info(f"Обработка завершена | final_samples={len(dataset)}")
        
        return dataset
    
    def process_with_config(
        self,
        config: dict
    ) -> Dataset:
        """Обработать данные с использованием конфигурации.
        
        Args:
            config: Конфигурация обработки
                - data_path: путь к данным
                - format_type: тип формата
                - text_column: колонка с текстом
                - instruction_column: колонка с инструкцией
                - response_column: колонка с ответом
        
        Returns:
            Обработанный датасет
        """
        data_path = config.get("data_path")
        format_type = config.get("format_type", "auto")
        text_column = config.get("text_column", "text")
        
        # Создаём форматтер с правильными колонками
        formatter = DataFormatter(
            instruction_column=config.get("instruction_column", "instruction"),
            input_column=config.get("input_column", "input"),
            response_column=config.get("response_column", "output"),
            text_column=text_column
        )
        
        # Создаём пайплайн с кастомным форматтером
        pipeline = DataPipeline(loader=self.loader, formatter=formatter)
        
        return pipeline.process(
            source=data_path,
            format_type=format_type,
            text_column=text_column
        )

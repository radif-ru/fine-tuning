"""Загрузка данных из различных источников.

Поддерживаемые форматы: JSONL, JSON, CSV, HuggingFace Datasets.
"""

import json
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset, load_dataset

from app.core.exceptions import DataLoadError
from app.core.logging_config import get_logger

logger = get_logger("data.loader")


class DataLoader:
    """Загрузчик данных из различных источников.
    
    Поддерживает:
    - JSONL файлы
    - JSON файлы
    - CSV файлы
    - HuggingFace Datasets
    """
    
    def __init__(self):
        """Инициализация загрузчика."""
        pass
    
    def load(
        self,
        path: str,
        format: Optional[str] = None,
        text_column: str = "text",
        split: str = "train"
    ) -> Dataset:
        """Загрузить данные из указанного источника.
        
        Автоматически определяет формат если не указан явно.
        
        Args:
            path: Путь к файлу или имя HF датасета
            format: Формат данных ('jsonl', 'json', 'csv', 'hf')
            text_column: Имя колонки с текстом
            split: Сплит для HF датасетов
        
        Returns:
            Загруженный датасет
        
        Raises:
            DataLoadError: При ошибке загрузки
        """
        if format is None:
            format = self._detect_format(path)
        
        logger.info(f"Загрузка данных | path={path}, format={format}")
        
        try:
            if format == "jsonl":
                return self.load_jsonl(path)
            elif format == "json":
                return self.load_json(path)
            elif format == "csv":
                return self.load_csv(path, text_column=text_column)
            elif format == "hf":
                return self.load_hf_dataset(path, split=split)
            else:
                raise DataLoadError(f"Неподдерживаемый формат: {format}")
        
        except Exception as e:
            raise DataLoadError(
                f"Не удалось загрузить данные: {path}",
                format=format,
                error=str(e)
            )
    
    def load_jsonl(self, path: Union[str, Path]) -> Dataset:
        """Загрузить JSONL файл.
        
        Args:
            path: Путь к JSONL файлу
        
        Returns:
            Датасет
        """
        path = Path(path)
        if not path.exists():
            raise DataLoadError(f"Файл не найден: {path}")
        
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Невалидный JSON в строке {line_num}: {e}")
        
        if not data:
            raise DataLoadError(f"Нет данных в файле: {path}")
        
        return Dataset.from_list(data)
    
    def load_json(self, path: Union[str, Path]) -> Dataset:
        """Загрузить JSON файл.
        
        Поддерживает:
        - Список объектов: [{...}, {...}]
        - Один объект: {...}
        
        Args:
            path: Путь к JSON файлу
        
        Returns:
            Датасет
        """
        path = Path(path)
        if not path.exists():
            raise DataLoadError(f"Файл не найден: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Если это один объект, оборачиваем в список
        if isinstance(data, dict):
            data = [data]
        
        if not isinstance(data, list):
            raise DataLoadError(f"Неподдерживаемая структура JSON: {type(data)}")
        
        return Dataset.from_list(data)
    
    def load_csv(
        self,
        path: Union[str, Path],
        text_column: str = "text",
        delimiter: str = ","
    ) -> Dataset:
        """Загрузить CSV файл.
        
        Args:
            path: Путь к CSV файлу
            text_column: Имя колонки с текстом
            delimiter: Разделитель
        
        Returns:
            Датасет
        """
        path = Path(path)
        if not path.exists():
            raise DataLoadError(f"Файл не найден: {path}")
        
        return load_dataset(
            "csv",
            data_files=str(path),
            delimiter=delimiter,
            split="train"
        )
    
    def load_hf_dataset(
        self,
        name: str,
        split: str = "train",
        config: Optional[str] = None
    ) -> Dataset:
        """Загрузить датасет из HuggingFace Hub.
        
        Args:
            name: Имя датасета
            split: Сплит (train, test, validation)
            config: Конфигурация датасета
        
        Returns:
            Датасет
        """
        try:
            kwargs = {"split": split}
            if config:
                kwargs["name"] = config
            
            return load_dataset(name, **kwargs)
        
        except Exception as e:
            raise DataLoadError(
                f"Не удалось загрузить HF датасет: {name}",
                split=split,
                error=str(e)
            )
    
    def validate(self, dataset: Dataset, required_columns: Optional[list] = None) -> bool:
        """Валидировать датасет.
        
        Args:
            dataset: Датасет для проверки
            required_columns: Обязательные колонки
        
        Returns:
            True если валидация прошла
        
        Raises:
            DataLoadError: Если валидация не прошла
        """
        if len(dataset) == 0:
            raise DataLoadError("Датасет пуст")
        
        if required_columns:
            missing = set(required_columns) - set(dataset.column_names)
            if missing:
                raise DataLoadError(
                    f"Отсутствуют обязательные колонки: {missing}",
                    available=dataset.column_names
                )
        
        logger.info(f"Валидация пройдена | samples={len(dataset)}, columns={dataset.column_names}")
        
        return True
    
    def _detect_format(self, path: str) -> str:
        """Автоматически определить формат данных.
        
        Args:
            path: Путь к файлу или имя датасета
        
        Returns:
            Определённый формат
        """
        path_lower = path.lower()
        
        # Если путь содержит / - это скорее всего HF датасет
        if "/" in path and not path.startswith("./") and not path.startswith("/"):
            return "hf"
        
        # Определение по расширению
        if path_lower.endswith(".jsonl"):
            return "jsonl"
        elif path_lower.endswith(".json"):
            return "json"
        elif path_lower.endswith(".csv"):
            return "csv"
        
        # По умолчанию пробуем JSONL
        return "jsonl"

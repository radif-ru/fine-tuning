"""Базовый класс для генераторов датасетов"""

from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any
import json
import csv
from pathlib import Path


class BaseGenerator(ABC):
    """Базовый класс для генераторов датасетов"""

    @abstractmethod
    def generate(self, count: int) -> Iterator[Dict[str, Any]]:
        """Генерирует указанное количество примеров

        Args:
            count: Количество примеров для генерации

        Yields:
            Dict[str, Any]: Словарь с данными примера
        """
        pass

    def save(self, path: str, count: int, format: str = "jsonl") -> None:
        """Сохраняет сгенерированные данные в файл

        Args:
            path: Путь к выходному файлу
            count: Количество примеров для генерации
            format: Формат файла (jsonl или csv)
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            self._save_jsonl(output_path, count)
        elif format == "csv":
            self._save_csv(output_path, count)
        else:
            raise ValueError(f"Неподдерживаемый формат: {format}")

    def _save_jsonl(self, path: Path, count: int) -> None:
        """Сохраняет данные в JSONL формат"""
        with path.open("w", encoding="utf-8") as f:
            for item in self.generate(count):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _save_csv(self, path: Path, count: int) -> None:
        """Сохраняет данные в CSV формат"""
        items = list(self.generate(count))
        if not items:
            return

        fieldnames = list(items[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(items)

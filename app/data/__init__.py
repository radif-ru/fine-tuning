"""Модуль обработки данных.

Модуль предоставляет инструменты для:
- Загрузки данных из различных источников (JSONL, JSON, CSV, HF Datasets)
- Форматирования данных в Alpaca, ShareGPT и другие форматы
- Токенизации для обучения causal language models
- Генерации синтетических датасетов
- Полного пайплайна обработки данных
"""

from app.data.formatter import DataFormatter
from app.data.loader import DataLoader
from app.data.pipeline import DataPipeline
from app.data.tokenizer import TokenizerWrapper
from app.data.generators import BaseGenerator, SyntheticGenerator, QAGenerator

__all__ = [
    "DataFormatter",
    "DataLoader",
    "DataPipeline",
    "TokenizerWrapper",
    "BaseGenerator",
    "SyntheticGenerator",
    "QAGenerator",
]

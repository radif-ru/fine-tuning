"""Конфигурация логирования приложения.

Настройка структурированного логирования в консоль и файл.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Настройка логирования приложения.
    
    Args:
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_file: Путь к файлу логов (опционально)
        format_string: Кастомный формат (опционально)
    
    Returns:
        Настроенный корневой логгер приложения
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    
    # Получаем или создаём логгер
    logger = logging.getLogger("llm_fine_tuner")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Удаляем существующие обработчики (для повторной инициализации)
    logger.handlers.clear()
    
    # Форматтер для всех обработчиков
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Обработчик консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    console_handler.set_name("console")
    logger.addHandler(console_handler)
    
    # Файловый обработчик (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.set_name("file")
        logger.addHandler(file_handler)
    
    # Не пропагируем в root logger (избегаем дублирования)
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Получить логгер с указанным именем.
    
    Args:
        name: Имя модуля/компонента для логирования
    
    Returns:
        Логгер с именем llm_fine_tuner.{name}
    """
    return logging.getLogger(f"llm_fine_tuner.{name}")

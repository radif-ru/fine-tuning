"""Тесты для конфигурации логирования."""

import logging
import tempfile
from pathlib import Path

import pytest

from app.core.logging_config import get_logger, setup_logging


class TestSetupLogging:
    """Тесты для setup_logging функции."""
    
    def test_console_logging(self):
        """Проверка логирования в консоль."""
        logger = setup_logging(level="INFO")
        
        assert logger.name == "llm_fine_tuner"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1
        
        # Проверяем наличие console handler
        console_handlers = [h for h in logger.handlers if h.name == "console"]
        assert len(console_handlers) == 1
    
    def test_file_logging(self):
        """Проверка логирования в файл."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test.log"
            logger = setup_logging(level="DEBUG", log_file=str(log_file))
            
            # Проверяем создание файла
            assert log_file.exists()
            
            # Проверяем наличие file handler
            file_handlers = [h for h in logger.handlers if h.name == "file"]
            assert len(file_handlers) == 1
    
    def test_log_levels(self):
        """Проверка разных уровней логирования."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            logger = setup_logging(level=level)
            assert logger.level == getattr(logging, level)
    
    def test_custom_format(self):
        """Проверка кастомного формата логов."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logging(format_string=custom_format)
        
        # Проверяем что формат применён
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, logging.Formatter)
    
    def test_directory_creation(self):
        """Проверка автоматического создания директории для логов."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_dir = Path(tmp_dir) / "nested" / "logs"
            log_file = nested_dir / "app.log"
            
            assert not nested_dir.exists()
            
            setup_logging(log_file=str(log_file))
            
            assert nested_dir.exists()
    
    def test_no_propagation(self):
        """Проверка что логгер не пропагирует в root."""
        logger = setup_logging()
        assert not logger.propagate


class TestGetLogger:
    """Тесты для get_logger функции."""
    
    def test_logger_name(self):
        """Проверка имени логгера."""
        logger = get_logger("test_module")
        assert logger.name == "llm_fine_tuner.test_module"
    
    def test_logger_hierarchy(self):
        """Проверка иерархии логгеров."""
        parent = setup_logging()
        child = get_logger("child")
        
        assert child.parent == parent

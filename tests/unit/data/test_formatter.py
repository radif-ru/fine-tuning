"""Тесты для DataFormatter."""

import pytest
from datasets import Dataset

from app.data.formatter import DataFormatter


class TestDataFormatter:
    """Тесты для DataFormatter."""
    
    @pytest.fixture
    def formatter(self):
        """Фикстура для DataFormatter."""
        return DataFormatter()
    
    @pytest.fixture
    def alpaca_dataset(self):
        """Датасет в формате Alpaca."""
        return Dataset.from_list([
            {"instruction": "Напиши привет", "input": "", "output": "Привет!"},
            {"instruction": "Объясни Python", "input": "для начинающих", "output": "Python - язык..."}
        ])
    
    @pytest.fixture
    def raw_dataset(self):
        """Датасет в raw формате."""
        return Dataset.from_list([
            {"text": "Пример текста 1"},
            {"text": "Пример текста 2"}
        ])
    
    def test_init(self, formatter):
        """Проверка инициализации."""
        assert formatter.instruction_column == "instruction"
        assert formatter.response_column == "output"
    
    def test_format_alpaca_no_input(self, formatter, alpaca_dataset):
        """Проверка форматирования Alpaca без input."""
        dataset = Dataset.from_list([
            {"instruction": "Напиши привет", "input": "", "output": "Привет!"}
        ])
        
        result = formatter.format_alpaca(dataset)
        
        assert "text" in result.column_names
        assert "### Инструкция:" in result[0]["text"]
        assert "Привет!" in result[0]["text"]
    
    def test_format_alpaca_with_input(self, formatter, alpaca_dataset):
        """Проверка форматирования Alpaca с input."""
        result = formatter.format_alpaca(alpaca_dataset)
        
        # Вторая запись должна иметь ### Входные данные:
        assert "### Входные данные:" in result[1]["text"]
    
    def test_format_raw(self, formatter, raw_dataset):
        """Проверка форматирования raw."""
        result = formatter.format_raw(raw_dataset)
        
        assert "text" in result.column_names
        assert result[0]["text"] == "Пример текста 1"
    
    def test_format_conversation(self, formatter):
        """Проверка форматирования разговора."""
        messages = [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Здравствуй!"}
        ]
        
        result = formatter.format_conversation(messages)
        
        assert "User: Привет" in result
        assert "Assistant: Здравствуй!" in result
    
    def test_auto_format_alpaca(self, formatter, alpaca_dataset):
        """Проверка авто-форматирования Alpaca."""
        result = formatter.auto_format(alpaca_dataset, format_type="alpaca")
        
        assert "text" in result.column_names
    
    def test_auto_format_raw(self, formatter, raw_dataset):
        """Проверка авто-форматирования raw."""
        result = formatter.auto_format(raw_dataset, format_type="raw")
        
        assert "text" in result.column_names
    
    def test_detect_format_alpaca(self, formatter, alpaca_dataset):
        """Проверка авто-определения Alpaca формата."""
        fmt = formatter._detect_format(alpaca_dataset)
        
        assert fmt == "alpaca"
    
    def test_detect_format_raw(self, formatter, raw_dataset):
        """Проверка авто-определения raw формата."""
        fmt = formatter._detect_format(raw_dataset)
        
        assert fmt == "raw"

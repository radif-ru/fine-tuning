"""Тесты для DataLoader."""

import json
import pytest
from pathlib import Path

from app.core.exceptions import DataLoadError
from app.data.loader import DataLoader


class TestDataLoader:
    """Тесты для DataLoader."""
    
    @pytest.fixture
    def loader(self):
        """Фикстура для DataLoader."""
        return DataLoader()
    
    @pytest.fixture
    def jsonl_file(self, temp_dir):
        """Создать JSONL файл для тестов."""
        file_path = temp_dir / "test.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"text": "пример 1"}) + "\n")
            f.write(json.dumps({"text": "пример 2"}) + "\n")
        return file_path
    
    @pytest.fixture
    def json_file(self, temp_dir):
        """Создать JSON файл для тестов."""
        file_path = temp_dir / "test.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([{"text": "пример 1"}, {"text": "пример 2"}], f)
        return file_path
    
    def test_init(self, loader):
        """Проверка инициализации."""
        assert loader is not None
    
    def test_detect_format_jsonl(self, loader):
        """Проверка определения JSONL формата."""
        fmt = loader._detect_format("data.jsonl")
        assert fmt == "jsonl"
    
    def test_detect_format_json(self, loader):
        """Проверка определения JSON формата."""
        fmt = loader._detect_format("data.json")
        assert fmt == "json"
    
    def test_detect_format_csv(self, loader):
        """Проверка определения CSV формата."""
        fmt = loader._detect_format("data.csv")
        assert fmt == "csv"
    
    def test_detect_format_hf(self, loader):
        """Проверка определения HF датасета."""
        fmt = loader._detect_format("username/dataset-name")
        assert fmt == "hf"
    
    def test_load_jsonl(self, loader, jsonl_file):
        """Проверка загрузки JSONL."""
        dataset = loader.load_jsonl(jsonl_file)
        
        assert len(dataset) == 2
        assert dataset[0]["text"] == "пример 1"
    
    def test_load_jsonl_not_exists(self, loader):
        """Проверка загрузки несуществующего JSONL."""
        with pytest.raises(DataLoadError):
            loader.load_jsonl("/nonexistent/file.jsonl")
    
    def test_load_json(self, loader, json_file):
        """Проверка загрузки JSON."""
        dataset = loader.load_json(json_file)
        
        assert len(dataset) == 2
    
    def test_validate_success(self, loader, jsonl_file):
        """Проверка успешной валидации."""
        dataset = loader.load_jsonl(jsonl_file)
        
        result = loader.validate(dataset, required_columns=["text"])
        assert result is True
    
    def test_validate_empty(self, loader):
        """Проверка валидации пустого датасета."""
        from datasets import Dataset
        empty_dataset = Dataset.from_list([])
        
        with pytest.raises(DataLoadError):
            loader.validate(empty_dataset)
    
    def test_validate_missing_columns(self, loader, jsonl_file):
        """Проверка валидации с отсутствующими колонками."""
        dataset = loader.load_jsonl(jsonl_file)
        
        with pytest.raises(DataLoadError):
            loader.validate(dataset, required_columns=["missing_column"])

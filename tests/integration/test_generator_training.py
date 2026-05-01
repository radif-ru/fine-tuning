"""Integration тесты для генератора датасетов с обучением"""

import pytest
import json
from pathlib import Path

from app.data.generators.synthetic import SyntheticGenerator


@pytest.mark.integration
@pytest.mark.slow
class TestGeneratorWithTraining:
    """Тесты генератора с полным циклом обучения"""

    def test_generate_and_train_on_synthetic_data(self, temp_dir):
        """Генерация данных и обучение модели на них"""
        # 1. Генерация синтетического датасета
        generator = SyntheticGenerator(topics=["programming"], seed=42)
        data_path = temp_dir / "synthetic_train.jsonl"
        generator.save(str(data_path), count=20, format="jsonl")
        
        # Проверка, что файл создан
        assert data_path.exists()
        
        # Проверка формата данных
        with data_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 20
            for line in lines:
                data = json.loads(line)
                assert "instruction" in data
                assert "input" in data
                assert "output" in data

    def test_generate_multiple_topics_and_train(self, temp_dir):
        """Генерация данных с несколькими темами"""
        # 1. Генерация с несколькими темами
        generator = SyntheticGenerator(topics=["programming", "science"], seed=42)
        data_path = temp_dir / "multi_topic_train.jsonl"
        generator.save(str(data_path), count=30, format="jsonl")
        
        # Проверка
        assert data_path.exists()
        
        with data_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 30
            
            # Проверка, что данные валидны
            for line in lines:
                data = json.loads(line)
                assert len(data["instruction"]) > 0
                assert len(data["output"]) > 0

    def test_generate_csv_format(self, temp_dir):
        """Генерация данных в CSV формате"""
        generator = SyntheticGenerator(topics=["general"], seed=42)
        data_path = temp_dir / "synthetic_train.csv"
        generator.save(str(data_path), count=15, format="csv")
        
        # Проверка
        assert data_path.exists()
        
        import csv
        with data_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 15
            assert "instruction" in rows[0]
            assert "output" in rows[0]

    def test_reproducibility_with_seed(self, temp_dir):
        """Проверка воспроизводимости с seed"""
        # Генерация с одинаковым seed
        gen1 = SyntheticGenerator(topics=["programming"], seed=123)
        path1 = temp_dir / "data1.jsonl"
        gen1.save(str(path1), count=10, format="jsonl")
        
        gen2 = SyntheticGenerator(topics=["programming"], seed=123)
        path2 = temp_dir / "data2.jsonl"
        gen2.save(str(path2), count=10, format="jsonl")
        
        # Чтение и сравнение
        with path1.open("r", encoding="utf-8") as f:
            lines1 = f.readlines()
        with path2.open("r", encoding="utf-8") as f:
            lines2 = f.readlines()
        
        assert lines1 == lines2

    def test_end_to_end_pipeline(self, temp_dir):
        """Полный цикл: генерация → проверка формата → готовность к обучению"""
        # 1. Генерация
        generator = SyntheticGenerator(topics=["programming", "science", "general"], seed=42)
        train_path = temp_dir / "train.jsonl"
        generator.save(str(train_path), count=50, format="jsonl")
        
        # 2. Проверка, что данные подходят для обучения
        with train_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # Проверка количества
        assert len(lines) == 50
        
        # Проверка формата каждого примера
        for line in lines:
            data = json.loads(line)
            # Alpaca формат
            assert isinstance(data["instruction"], str)
            assert isinstance(data["input"], str)
            assert isinstance(data["output"], str)
            assert len(data["instruction"]) > 0
            assert len(data["output"]) > 0

    def test_large_dataset_generation(self, temp_dir):
        """Генерация большого датасета"""
        generator = SyntheticGenerator(seed=42)
        data_path = temp_dir / "large_dataset.jsonl"
        
        # Генерация 500 примеров
        generator.save(str(data_path), count=500, format="jsonl")
        
        # Проверка
        assert data_path.exists()
        
        line_count = 0
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                data = json.loads(line)
                assert "instruction" in data
                assert "output" in data
        
        assert line_count == 500

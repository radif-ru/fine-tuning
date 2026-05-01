"""Unit тесты для генераторов датасетов"""

import pytest
import json
import csv
from pathlib import Path
from app.data.generators.base import BaseGenerator
from app.data.generators.synthetic import SyntheticGenerator
from app.data.generators.qa import QAGenerator


class MockGenerator(BaseGenerator):
    """Мок генератора для тестирования базового класса"""

    def generate(self, count: int):
        for i in range(count):
            yield {"id": i, "text": f"Example {i}"}


class TestBaseGenerator:
    """Тесты базового класса генератора"""

    def test_generate_yields_correct_count(self):
        """Генерация возвращает правильное количество примеров"""
        generator = MockGenerator()
        items = list(generator.generate(5))
        assert len(items) == 5

    def test_save_jsonl(self, temp_dir):
        """Сохранение в JSONL формат"""
        generator = MockGenerator()
        output_path = temp_dir / "output.jsonl"

        generator.save(str(output_path), count=3, format="jsonl")

        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 3
            for line in lines:
                data = json.loads(line)
                assert "id" in data
                assert "text" in data

    def test_save_csv(self, temp_dir):
        """Сохранение в CSV формат"""
        generator = MockGenerator()
        output_path = temp_dir / "output.csv"

        generator.save(str(output_path), count=3, format="csv")

        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 3
            assert "id" in rows[0]
            assert "text" in rows[0]

    def test_save_creates_parent_directory(self, temp_dir):
        """Сохранение создаёт родительскую директорию"""
        generator = MockGenerator()
        output_path = temp_dir / "subdir" / "output.jsonl"

        generator.save(str(output_path), count=1, format="jsonl")

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_unsupported_format(self, temp_dir):
        """Неподдерживаемый формат вызывает ошибку"""
        generator = MockGenerator()
        output_path = temp_dir / "output.txt"

        with pytest.raises(ValueError, match="Неподдерживаемый формат"):
            generator.save(str(output_path), count=1, format="txt")


class TestSyntheticGenerator:
    """Тесты синтетического генератора"""

    def test_init_default_topics(self):
        """Инициализация с темами по умолчанию"""
        generator = SyntheticGenerator()
        assert generator.topics == ["programming", "science", "general"]

    def test_init_custom_topics(self):
        """Инициализация с пользовательскими темами"""
        generator = SyntheticGenerator(topics=["programming"])
        assert generator.topics == ["programming"]

    def test_init_with_seed(self):
        """Инициализация с seed для воспроизводимости"""
        gen1 = SyntheticGenerator(seed=42)
        gen2 = SyntheticGenerator(seed=42)

        items1 = list(gen1.generate(5))
        items2 = list(gen2.generate(5))

        assert items1 == items2

    def test_init_invalid_topics(self):
        """Неподдерживаемые темы вызывают ошибку"""
        with pytest.raises(ValueError, match="Неподдерживаемые темы"):
            SyntheticGenerator(topics=["invalid_topic"])

    def test_generate_returns_correct_count(self):
        """Генерация возвращает правильное количество примеров"""
        generator = SyntheticGenerator(seed=42)
        items = list(generator.generate(10))
        assert len(items) == 10

    def test_generate_alpaca_format(self):
        """Генерация в формате Alpaca"""
        generator = SyntheticGenerator(seed=42)
        item = next(generator.generate(1))

        assert "instruction" in item
        assert "input" in item
        assert "output" in item
        assert item["input"] == ""  # input всегда пустой

    def test_generate_non_empty_fields(self):
        """Поля не пустые"""
        generator = SyntheticGenerator(seed=42)
        items = list(generator.generate(10))

        for item in items:
            assert len(item["instruction"]) > 0
            assert len(item["output"]) > 0

    def test_save_jsonl(self, temp_dir):
        """Сохранение в JSONL"""
        generator = SyntheticGenerator(seed=42)
        output_path = temp_dir / "synthetic.jsonl"

        generator.save(str(output_path), count=5, format="jsonl")

        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 5
            for line in lines:
                data = json.loads(line)
                assert "instruction" in data
                assert "output" in data

    def test_save_csv(self, temp_dir):
        """Сохранение в CSV"""
        generator = SyntheticGenerator(seed=42)
        output_path = temp_dir / "synthetic.csv"

        generator.save(str(output_path), count=5, format="csv")

        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert "instruction" in rows[0].keys()
            assert "output" in rows[0].keys()

    def test_single_topic_generation(self):
        """Генерация с одной темой"""
        generator = SyntheticGenerator(topics=["programming"], seed=42)
        items = list(generator.generate(10))

        for item in items:
            assert len(item["instruction"]) > 0
            assert len(item["output"]) > 0

    def test_multiple_topics_generation(self):
        """Генерация с несколькими темами"""
        generator = SyntheticGenerator(topics=["programming", "science"], seed=42)
        items = list(generator.generate(20))

        assert len(items) == 20
        for item in items:
            assert "instruction" in item
            assert "output" in item


class TestQAGenerator:
    """Тесты QA генератора"""

    def test_init_default_categories(self):
        """Инициализация с категориями по умолчанию"""
        generator = QAGenerator()
        assert generator.categories == ["general", "technical", "practical"]

    def test_init_custom_categories(self):
        """Инициализация с пользовательскими категориями"""
        generator = QAGenerator(categories=["technical"])
        assert generator.categories == ["technical"]

    def test_init_with_seed(self):
        """Инициализация с seed для воспроизводимости"""
        gen1 = QAGenerator(seed=42)
        gen2 = QAGenerator(seed=42)

        items1 = list(gen1.generate(5))
        items2 = list(gen2.generate(5))

        assert items1 == items2

    def test_init_invalid_categories(self):
        """Неподдерживаемые категории вызывают ошибку"""
        with pytest.raises(ValueError, match="Неподдерживаемые категории"):
            QAGenerator(categories=["invalid_category"])

    def test_generate_returns_correct_count(self):
        """Генерация возвращает правильное количество примеров"""
        generator = QAGenerator(seed=42)
        items = list(generator.generate(10))
        assert len(items) == 10

    def test_generate_alpaca_format(self):
        """Генерация в формате Alpaca"""
        generator = QAGenerator(seed=42)
        item = next(generator.generate(1))

        assert "instruction" in item
        assert "input" in item
        assert "output" in item
        assert item["input"] == ""

    def test_generate_non_empty_fields(self):
        """Поля не пустые"""
        generator = QAGenerator(seed=42)
        items = list(generator.generate(10))

        for item in items:
            assert len(item["instruction"]) > 0
            assert len(item["output"]) > 0

    def test_save_jsonl(self, temp_dir):
        """Сохранение в JSONL"""
        generator = QAGenerator(seed=42)
        output_path = temp_dir / "qa.jsonl"

        generator.save(str(output_path), count=5, format="jsonl")

        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 5
            for line in lines:
                data = json.loads(line)
                assert "instruction" in data
                assert "output" in data

    def test_save_csv(self, temp_dir):
        """Сохранение в CSV"""
        generator = QAGenerator(seed=42)
        output_path = temp_dir / "qa.csv"

        generator.save(str(output_path), count=5, format="csv")

        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert "instruction" in rows[0].keys()
            assert "output" in rows[0].keys()

    def test_single_category_generation(self):
        """Генерация с одной категорией"""
        generator = QAGenerator(categories=["technical"], seed=42)
        items = list(generator.generate(10))

        for item in items:
            assert len(item["instruction"]) > 0
            assert len(item["output"]) > 0

    def test_multiple_categories_generation(self):
        """Генерация с несколькими категориями"""
        generator = QAGenerator(categories=["general", "technical"], seed=42)
        items = list(generator.generate(20))

        assert len(items) == 20
        for item in items:
            assert "instruction" in item
            assert "output" in item


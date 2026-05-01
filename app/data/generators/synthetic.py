"""Синтетический генератор датасетов"""

import random
from typing import Iterator, Dict, Any, Optional
from app.data.generators.base import BaseGenerator


class SyntheticGenerator(BaseGenerator):
    """Генерирует синтетические инструкции и ответы для fine-tuning

    Генерирует данные в формате Alpaca (instruction, input, output).
    """

    # Шаблоны инструкций по темам
    TOPIC_TEMPLATES = {
        "programming": [
            "Напиши функцию на Python для {task}",
            "Объясни концепцию {concept} в программировании",
            "Как исправить ошибку {error}?",
            "Приведи пример использования {pattern}",
            "Опиши разницу между {concept1} и {concept2}",
        ],
        "science": [
            "Объясни {concept} простыми словами",
            "Как работает {phenomenon}?",
            "Какие преимущества имеет {method}?",
            "Опиши процесс {process}",
            "В чём разница между {concept1} и {concept2}?",
        ],
        "general": [
            "Напиши {text_type} о {topic}",
            "Как {action}?",
            "Перечисли {count} способов {task}",
            "Опиши преимущества {topic}",
            "Объясни, почему {phenomenon}",
        ],
    }

    # Словари для заполнения шаблонов
    VOCABULARY = {
        "programming": {
            "task": ["сортировки списка", "поиска подстроки", "парсинга JSON", "работы с файлами"],
            "concept": ["рекурсия", "замыкание", "декоратор", "генератор", "контекстный менеджер"],
            "error": ["IndexError", "KeyError", "TypeError", "AttributeError"],
            "pattern": ["singleton", "factory", "observer", "strategy"],
            "concept1": ["список", "кортеж"],
            "concept2": ["множество", "словарь"],
        },
        "science": {
            "concept": ["фотосинтез", "гравитация", "эволюция", "квантовая механика", "относительность"],
            "phenomenon": ["электрический ток", "магнитное поле", "химическая реакция"],
            "method": ["научный метод", "эксперимент", "наблюдение"],
            "process": ["клеточное деление", "пищеварение", "дыхание"],
            "concept1": ["физика", "химия"],
            "concept2": ["биология", "геология"],
        },
        "general": {
            "text_type": ["эссе", "статью", "резюме", "обзор"],
            "topic": ["искусственном интеллекте", "климатических изменениях", "истории интернета"],
            "action": ["улучшить продуктивность", "выучить язык", "начать бизнес"],
            "count": ["пять", "три", "десять"],
            "task": ["решения проблем", "сохранения энергии", "здорового питания"],
            "phenomenon": ["небо синее", "вода кипит", "листья падают"],
        },
    }

    def __init__(self, topics: Optional[list[str]] = None, seed: Optional[int] = None):
        """Инициализирует генератор

        Args:
            topics: Список тем для генерации. Если None, используются все темы.
            seed: Seed для воспроизводимости
        """
        self._random = random.Random(seed) if seed is not None else random

        self.topics = topics or list(self.TOPIC_TEMPLATES.keys())
        self._validate_topics()

    def _validate_topics(self) -> None:
        """Проверяет, что все темы поддерживаются"""
        invalid_topics = set(self.topics) - set(self.TOPIC_TEMPLATES.keys())
        if invalid_topics:
            raise ValueError(f"Неподдерживаемые темы: {invalid_topics}")

    def generate(self, count: int) -> Iterator[Dict[str, Any]]:
        """Генерирует инструкции в Alpaca формате

        Args:
            count: Количество примеров для генерации

        Yields:
            Dict[str, Any]: Словарь с ключами instruction, input, output
        """
        for _ in range(count):
            yield self._generate_example()

    def _generate_example(self) -> Dict[str, Any]:
        """Генерирует один пример"""
        topic = self._random.choice(self.topics)
        template = self._random.choice(self.TOPIC_TEMPLATES[topic])
        vocab = self.VOCABULARY[topic]

        # Заполняем шаблон случайными значениями
        instruction = template.format(**{
            k: self._random.choice(v) if isinstance(v, list) else v
            for k, v in vocab.items()
        })

        # Генерируем ответ
        output = self._generate_response(instruction, topic)

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
        }

    def _generate_response(self, instruction: str, topic: str) -> str:
        """Генерирует ответ на инструкцию

        Args:
            instruction: Инструкция
            topic: Тема инструкции

        Returns:
            Сгенерированный ответ
        """
        responses = {
            "programming": [
                "Вот решение с использованием стандартной библиотеки Python.",
                "Для этой задачи можно использовать следующий подход.",
                "Рассмотрим пошаговое решение этой проблемы.",
                "Вот пример кода с комментариями.",
                "Этот паттерн часто используется в реальных проектах.",
            ],
            "science": [
                "Это явление объясняется следующими законами природы.",
                "Давайте рассмотрим этот процесс с научной точки зрения.",
                "Исследования показывают, что этот метод эффективен.",
                "В основе этого процесса лежат фундаментальные принципы.",
                "Это явление можно объяснить через теорию...",
            ],
            "general": [
                "Вот подробный ответ на ваш вопрос.",
                "Рассмотрим несколько аспектов этой темы.",
                "Это важный вопрос, который требует детального рассмотрения.",
                "Приведу примеры и объяснения.",
                "Давайте разберём это пошагово.",
            ],
        }

        base_response = self._random.choice(responses[topic])

        # Добавляем детали в зависимости от типа инструкции
        if "функция" in instruction or "код" in instruction:
            return f"{base_response}\n\nПример:\n```python\ndef example():\n    pass\n```"
        elif "объясни" in instruction or "опиши" in instruction:
            return f"{base_response}\n\nКлючевые моменты:\n1. Первый аспект\n2. Второй аспект\n3. Третий аспект"
        else:
            return f"{base_response}\n\nНадеюсь, это поможет!"

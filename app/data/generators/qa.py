"""QA генератор датасетов"""

import random
from typing import Iterator, Dict, Any, Optional
from app.data.generators.base import BaseGenerator


class QAGenerator(BaseGenerator):
    """Генерирует вопросы и ответы для обучения модели

    Генерирует данные в формате Alpaca (instruction, input, output)
    с фокусом на вопросно-ответные пары.
    """

    # Категории вопросов
    CATEGORIES = {
        "general": [
            "Что такое {concept}?",
            "Как работает {phenomenon}?",
            "В чём разница между {concept1} и {concept2}?",
            "Объясни {concept} простыми словами",
            "Почему {phenomenon} происходит?",
        ],
        "technical": [
            "Как реализовать {task} на {language}?",
            "Как исправить ошибку {error}?",
            "Опиши алгоритм {algorithm}",
            "Как оптимизировать {process}?",
            "В чём преимущества {technology}?",
        ],
        "practical": [
            "Как {action}?",
            "Перечисли {count} способов {task}",
            "Какие есть альтернативы для {tool}?",
            "Как выбрать {item}?",
            "Что делать если {situation}?",
        ],
    }

    # Словари для заполнения шаблонов
    VOCABULARY = {
        "general": {
            "concept": ["машинное обучение", "нейросеть", "блокчейн", "квантовая физика", "генетика"],
            "phenomenon": ["гравитация", "электричество", "магнетизм", "теплопередача"],
            "concept1": ["AI", "блокчейн", "нейросеть"],
            "concept2": ["ML", "квантовые вычисления", "глубокое обучение"],
        },
        "technical": {
            "task": ["сортировку", "поиск", "парсинг", "авторизацию"],
            "language": ["Python", "JavaScript", "Java", "C++"],
            "error": ["404", "500", "timeout", "memory leak"],
            "algorithm": ["быстрой сортировки", "поиска в глубину", "динамического программирования"],
            "process": ["запросы к базе данных", "обработку изображений", "обучение модели"],
            "technology": ["Docker", "Kubernetes", "React", "TensorFlow"],
        },
        "practical": {
            "action": ["изучить Python", "настроить сервер", "оптимизировать код", "отладить программу"],
            "count": ["пять", "три", "десять"],
            "task": ["решения проблем", "сохранения энергии", "повышения продуктивности"],
            "tool": ["Git", "Docker", "Linux", "VS Code"],
            "item": ["базу данных", "фреймворк", "язык программирования"],
            "situation": ["программа зависает", "тесты не проходят", "сервер не отвечает"],
        },
    }

    # Шаблоны ответов
    RESPONSE_TEMPLATES = {
        "general": [
            "{concept} — это {definition}. Ключевые особенности:\n1. {feature1}\n2. {feature2}\n3. {feature3}",
            "{concept} представляет собой {definition}. Важно понимать, что {explanation}.",
            "Простыми словами, {concept} — это {definition}. Например, {example}.",
        ],
        "technical": [
            "Для реализации {task} на {language} можно использовать следующий подход:\n\n```{language}\n# код\n```\n\nКлючевые моменты: {points}.",
            "Ошибка {error} обычно возникает, когда {cause}. Решение: {solution}.",
            "Алгоритм {algorithm} работает следующим образом:\n1. {step1}\n2. {step2}\n3. {step3}\nСложность: {complexity}.",
        ],
        "practical": [
            "Чтобы {action}, выполните следующие шаги:\n1. {step1}\n2. {step2}\n3. {step3}\n\nДополнительные советы: {tips}.",
            "Вот {count} способов {task}:\n1. Способ 1: {method1}\n2. Способ 2: {method2}\n3. Способ 3: {method3}",
            "Для выбора {item} учитывайте:\n- {criterion1}\n- {criterion2}\n- {criterion3}",
        ],
    }

    def __init__(self, categories: Optional[list[str]] = None, seed: Optional[int] = None):
        """Инициализирует генератор

        Args:
            categories: Список категорий для генерации. Если None, используются все категории.
            seed: Seed для воспроизводимости
        """
        self._random = random.Random(seed) if seed is not None else random

        self.categories = categories or list(self.CATEGORIES.keys())
        self._validate_categories()

    def _validate_categories(self) -> None:
        """Проверяет, что все категории поддерживаются"""
        invalid_categories = set(self.categories) - set(self.CATEGORIES.keys())
        if invalid_categories:
            raise ValueError(f"Неподдерживаемые категории: {invalid_categories}")

    def generate(self, count: int) -> Iterator[Dict[str, Any]]:
        """Генерирует вопросы и ответы в Alpaca формате

        Args:
            count: Количество примеров для генерации

        Yields:
            Dict[str, Any]: Словарь с ключами instruction, input, output
        """
        for _ in range(count):
            yield self._generate_example()

    def _generate_example(self) -> Dict[str, Any]:
        """Генерирует один пример"""
        category = self._random.choice(self.categories)
        template = self._random.choice(self.CATEGORIES[category])
        vocab = self.VOCABULARY[category]

        # Заполняем шаблон вопроса
        instruction = template.format(**{
            k: self._random.choice(v) if isinstance(v, list) else v
            for k, v in vocab.items()
        })

        # Генерируем ответ
        output = self._generate_response(instruction, category)

        return {
            "instruction": instruction,
            "input": "",
            "output": output,
        }

    def _generate_response(self, instruction: str, category: str) -> str:
        """Генерирует ответ на вопрос

        Args:
            instruction: Вопрос
            category: Категория вопроса

        Returns:
            Сгенерированный ответ
        """
        response_template = self._random.choice(self.RESPONSE_TEMPLATES[category])

        # Заполняем шаблон ответа
        response = response_template.format(
            concept="данная технология",
            definition="инструмент или метод для решения определённых задач",
            feature1="высокая эффективность",
            feature2="простота использования",
            feature3="широкое применение",
            explanation="это важный аспект, который нужно учитывать",
            example="в веб-разработке или анализе данных",
            task="задачу",
            language="Python",
            error="ошибка",
            cause="нарушены условия выполнения",
            solution="проверить входные данные и логику",
            algorithm="алгоритм",
            step1="первый шаг выполнения",
            step2="второй шаг выполнения",
            step3="третий шаг выполнения",
            complexity="O(n log n)",
            action="действие",
            count="несколько",
            tips="используйте документацию и тестирование",
            method1="первый метод",
            method2="второй метод",
            method3="третий метод",
            item="инструмент",
            criterion1="функциональность",
            criterion2="производительность",
            criterion3="сообщество",
            points="важно следовать лучшим практикам",
        )

        return response

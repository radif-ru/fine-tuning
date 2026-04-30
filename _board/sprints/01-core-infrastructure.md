# Спринт 01. Core Infrastructure

- **Источник:** Требования из `_docs/requirements.md` и архитектура из `_docs/architecture.md`
- **Ветка:** `feature/01-core-infra`
- **Открыт:** 2026-05-01
- **Закрыт:** —

## 1. Цель спринта

Реализовать **базовую инфраструктуру** фреймворка: конфигурация через `pydantic-settings`, структурированное логирование, иерархия исключений, CLI-скелет с поддержкой команд `train` и `inference`. Подготовить фундамент для последующих спринтов по загрузке моделей и данных.

## 2. Скоуп и non-goals

### В скоупе

- Реализация `app/core/config.py` — `Settings` на pydantic-settings
- Реализация `app/core/logging_config.py` — настройка логирования
- Реализация `app/core/exceptions.py` — иерархия исключений
- Создание `app/__main__.py` — точка входа для `python -m app`
- CLI с argparse: команды `train` и `inference` (скелет)
- Unit-тесты для core (`tests/unit/core/`)
- Примеры конфигураций в `configs/`

### Вне скоупа (non-goals)

- Реальная загрузка моделей — в Спринте 02
- Реальная загрузка данных — в Спринте 03
- Реальный тренировочный цикл — в Спринте 04
- Реальный инференс — в Спринте 05
- Интеграционные тесты с реальными моделями — позже

## 3. Acceptance Criteria спринта

- [ ] `python -m app --help` выводит справку по CLI
- [ ] `python -m app train --help` выводит справку по train
- [ ] `python -m app inference --help` выводит справку по inference
- [ ] Конфигурация загружается из `.env` без ошибок
- [ ] Невалидная конфигурация выдаёт понятную ошибку
- [ ] Логирование работает (файл и консоль)
- [ ] Все unit-тесты проходят (`pytest tests/unit/core/`)
- [ ] Покрытие кода core ≥ 85%
- [ ] Все задачи спринта — `Done`, сводная таблица актуальна

## 4. Решения по архитектуре

| Решение | Ссылка | Обоснование |
|---------|--------|-------------|
| **pydantic-settings** | `_docs/architecture.md` §3.2 | Валидация типов, значения по умолчанию, понятные ошибки |
| **argparse** | `_docs/architecture.md` §3.1 | Стандартная библиотека, достаточно для MVP |
| **logging** | `_docs/architecture.md` §3.2 | Стандартная библиотека, гибкая конфигурация |
| **Иерархия исключений** | `_docs/architecture.md` §4 | Разделение ошибок по типам для разной обработки |

## 5. Этап 1. Core Components

Реализация базовых компонентов.

### Задача 1.1. Configuration (`app/core/config.py`)

- **Статус:** Done
- **Приоритет:** critical
- **Объём:** M
- **Зависит от:** —
- **Связанные документы:** `_docs/configuration.md`, `_docs/architecture.md` §3.2
- **Затрагиваемые файлы:** `app/core/config.py`, `tests/unit/core/test_config.py`

#### Описание

Реализовать класс `Settings` на `pydantic-settings`:

```python
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from typing import Optional, List

class Settings(BaseSettings):
    # Model Configuration
    BASE_MODEL_NAME: str
    HF_CACHE_DIR: Optional[str] = None
    TRUST_REMOTE_CODE: bool = False
    
    # LoRA Configuration
    LORA_R: int = Field(default=8, ge=1)
    LORA_ALPHA: int = Field(default=16, ge=1)
    LORA_DROPOUT: float = Field(default=0.1, ge=0, le=1)
    LORA_TARGET_MODULES: List[str] = Field(default=["q_proj", "k_proj", "v_proj", "o_proj"])
    LORA_BIAS: str = Field(default="none", pattern="^(none|all|lora_only)$")
    LORA_TASK_TYPE: str = Field(default="CAUSAL_LM")
    
    # Training Configuration
    OUTPUT_DIR: str = "./outputs"
    CHECKPOINT_DIR: str = "./checkpoints"
    NUM_EPOCHS: int = Field(default=3, ge=1)
    PER_DEVICE_BATCH_SIZE: int = Field(default=4, ge=1)
    # ... и так далее по `_docs/configuration.md`
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

Требования:
- Все поля из `.env.example` должны быть в Settings
- Валидация диапазонов (ge, le, gt, lt)
- Валидация enum-значений через pattern или Literal
- Значения по умолчанию
- Поддержка списков (comma-separated в env)

#### Definition of Done

- [ ] `Settings` содержит все поля из `.env.example`
- [ ] Валидация типов работает (строки, числа, списки)
- [ ] Валидация диапазонов работает (например, LORA_R >= 1)
- [ ] Невалидные значения выдают понятные ошибки
- [ ] Загрузка из `.env` работает
- [ ] Тесты: полное покрытие полей, валидация, значения по умолчанию

---

### Задача 1.2. Logging (`app/core/logging_config.py`)

- **Статус:** Done
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** —
- **Связанные документы:** `_docs/architecture.md` §3.2, `_docs/instructions.md` §5
- **Затрагиваемые файлы:** `app/core/logging_config.py`, `tests/unit/core/test_logging.py`

#### Описание

Реализовать настройку логирования:

```python
import logging
import sys
from typing import Optional
from pathlib import Path

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
        Настроенный логгер
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    
    logger = logging.getLogger("llm_fine_tuner")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler (если указан)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger
```

#### Definition of Done

- [ ] Логирование в консоль работает
- [ ] Логирование в файл работает (если указан)
- [ ] Уровни логирования переключаются
- [ ] Формат логов настраивается
- [ ] Директория для логов создаётся автоматически
- [ ] Тесты: проверка вывода, уровней, файлового логирования

---

### Задача 1.3. Exceptions (`app/core/exceptions.py`)

- **Статус:** Done
- **Приоритет:** medium
- **Объём:** XS
- **Зависит от:** —
- **Связанные документы:** `_docs/architecture.md` §4
- **Затрагиваемые файлы:** `app/core/exceptions.py`

#### Описание

Реализовать иерархию исключений:

```python
class FineTuningError(Exception):
    """Базовый класс для всех ошибок фреймворка."""
    pass

class ConfigurationError(FineTuningError):
    """Ошибка в конфигурации."""
    pass

class ModelLoadError(FineTuningError):
    """Ошибка загрузки модели."""
    pass

class DataLoadError(FineTuningError):
    """Ошибка загрузки данных."""
    pass

class TrainingError(FineTuningError):
    """Ошибка во время обучения."""
    pass

class InferenceError(FineTuningError):
    """Ошибка во время инференса."""
    pass
```

#### Definition of Done

- [ ] Все классы исключений созданы
- [ ] Наследование от FineTuningError
- [ ] Докстринги на русском языке
- [ ] Тесты: n/a (простые классы)

---

## 6. Этап 2. CLI Interface

Реализация CLI.

### Задача 2.1. CLI Skeleton (`app/__main__.py`)

- **Статус:** Progress
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задача 1.1, 1.2
- **Связанные документы:** `_docs/architecture.md` §3.1
- **Затрагиваемые файлы:** `app/__main__.py`, `tests/unit/test_cli.py`

#### Описание

Реализовать точку входа с argparse:

```python
import argparse
import sys
from app.core.config import Settings
from app.core.logging_config import setup_logging

def main():
    parser = argparse.ArgumentParser(
        prog="llm-fine-tuner",
        description="Фреймворк для дообучения LLM с LoRA"
    )
    
    parser.add_argument(
        "--config", "-c",
        default=".env",
        help="Путь к файлу конфигурации (.env)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Запустить обучение")
    train_parser.add_argument("--base-model", help="Базовая модель")
    train_parser.add_argument("--data-path", help="Путь к данным")
    train_parser.add_argument("--output-dir", help="Директория вывода")
    train_parser.add_argument("--epochs", type=int, help="Количество эпох")
    
    # Inference command
    inference_parser = subparsers.add_parser("inference", help="Запустить инференс")
    inference_parser.add_argument("--model-path", required=True, help="Путь к модели")
    inference_parser.add_argument("--prompt", help="Промпт для генерации")
    inference_parser.add_argument("--interactive", action="store_true", help="Интерактивный режим")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load settings
    settings = Settings(_env_file=args.config)
    
    # Setup logging
    logger = setup_logging(
        level=settings.LOG_LEVEL,
        log_file=settings.LOG_FILE
    )
    
    # Route to command
    if args.command == "train":
        logger.info("Запуск обучения...")
        # TODO: Implement in Sprint 04
        print("Команда train будет реализована в Sprint 04")
    
    elif args.command == "inference":
        logger.info("Запуск инференса...")
        # TODO: Implement in Sprint 05
        print("Команда inference будет реализована в Sprint 05")

if __name__ == "__main__":
    main()
```

#### Definition of Done

- [ ] `python -m app --help` работает
- [ ] `python -m app train --help` работает
- [ ] `python -m app inference --help` работает
- [ ] Аргумент `--config` переключает файл конфигурации
- [ ] CLI использует Settings для загрузки конфигурации
- [ ] CLI использует setup_logging для настройки логирования
- [ ] Тесты: проверка парсинга аргументов, help сообщений

---

## 7. Этап 3. Utils

Базовые утилиты для работы с устройствами и памятью (понадобятся в Sprint 02).

### Задача 3.1. Device Utils (`app/utils/device.py`)

- **Статус:** ToDo
- **Приоритет:** medium
- **Объём:** S
- **Зависит от:** —
- **Связанные документы:** `_docs/architecture.md` §2.5
- **Затрагиваемые файлы:** `app/utils/device.py`, `tests/unit/utils/test_device.py`

#### Описание

Реализовать определение доступных устройств:

```python
import torch
from typing import Optional

def get_device(preferred: Optional[str] = None) -> str:
    """Определить оптимальное устройство для вычислений.
    
    Args:
        preferred: Предпочтительное устройство ('cuda', 'cpu', 'auto')
    
    Returns:
        Строка с устройством ('cuda', 'cuda:0', 'cpu')
    """
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda" or preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"

def get_device_info() -> dict:
    """Получить информацию о доступных устройствах.
    
    Returns:
        Словарь с информацией: cuda_available, device_count, device_name
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return info
```

#### Definition of Done

- [ ] Функция `get_device()` возвращает корректное устройство
- [ ] Функция `get_device_info()` возвращает информацию о CUDA
- [ ] Работает на CPU-only системах без ошибок
- [ ] Тесты: проверка логики выбора устройства

---

### Задача 3.2. Memory Utils (`app/utils/memory.py`)

- **Статус:** ToDo
- **Приоритет:** medium
- **Объём:** S
- **Зависит от:** —
- **Связанные документы:** `_docs/architecture.md` §2.5
- **Затрагиваемые файлы:** `app/utils/memory.py`, `tests/unit/utils/test_memory.py`

#### Описание

Реализовать мониторинг памяти:

```python
import torch
from typing import Optional, Dict

def get_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
    """Получить статистику использования памяти.
    
    Args:
        device: Устройство ('cuda', 'cuda:0', 'cpu')
    
    Returns:
        Словарь с allocated, reserved, free (в MB)
    """
    if device and device.startswith("cuda"):
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        return {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "free_mb": round(total - allocated, 2),
            "total_mb": round(total, 2),
        }
    return {"allocated_mb": 0, "reserved_mb": 0, "free_mb": 0, "total_mb": 0}

def clear_memory_cache():
    """Очистить кэш CUDA памяти."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

#### Definition of Done

- [ ] `get_memory_stats()` возвращает корректные значения для CUDA
- [ ] `get_memory_stats()` работает на CPU без ошибок
- [ ] `clear_memory_cache()` вызывает empty_cache()
- [ ] Тесты: проверка структуры возвращаемых данных

---

## 8. Этап 4. Test Infrastructure

Фикстуры и вспомогательные функции для тестов.

### Задача 4.1. Test Fixtures (`tests/conftest.py`)

- **Статус:** ToDo
- **Приоритет:** medium
- **Объём:** S
- **Зависит от:** Задача 1.1
- **Связанные документы:** `_docs/testing.md`
- **Затрагиваемые файлы:** `tests/conftest.py`

#### Описание

Создать базовые фикстуры pytest:

```python
import pytest
import tempfile
from pathlib import Path
from app.core.config import Settings

@pytest.fixture
def temp_dir():
    """Временная директория для тестов."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)

@pytest.fixture
def mock_settings(temp_dir):
    """Настройки для тестов с временными путями."""
    return Settings(
        BASE_MODEL_NAME="gpt2",
        OUTPUT_DIR=str(temp_dir / "outputs"),
        CHECKPOINT_DIR=str(temp_dir / "checkpoints"),
        LOG_LEVEL="DEBUG",
    )

@pytest.fixture
def sample_train_data():
    """Пример тренировочных данных для тестов."""
    return [
        {"instruction": "Привет", "input": "", "output": "Здравствуй!"},
        {"instruction": "Как дела?", "input": "", "output": "Хорошо, спасибо!"},
    ]
```

#### Definition of Done

- [ ] Фикстура `temp_dir` создаёт временную директорию
- [ ] Фикстура `mock_settings` возвращает валидный Settings
- [ ] Фикстура `sample_train_data` возвращает пример данных
- [ ] Все фикстуры работают в тестах

---

## 9. Этап 5. Examples

Примеры конфигураций.

### Задача 5.1. Example Configs (`configs/`)

- **Статус:** ToDo
- **Приоритет:** low
- **Объём:** S
- **Зависит от:** Задача 1.1
- **Связанные документы:** `_docs/configuration.md` § «Примеры конфигураций»
- **Затрагиваемые файлы:** `configs/tinyllama_lora.env`, `configs/phi3_lora.env`

#### Описание

Создать примеры конфигураций:

**configs/tinyllama_lora.env:**
```bash
# TinyLlama для CPU/GPU с 8GB
BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LORA_R=8
LORA_ALPHA=16
TRAIN_DATA_PATH=./data/train.jsonl
OUTPUT_DIR=./outputs/tinyllama
NUM_EPOCHS=3
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
MAX_SEQ_LENGTH=512
FP16=true
```

**configs/phi3_lora.env:**
```bash
# Phi-3 для GPU с 16GB+
BASE_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
LORA_R=16
LORA_ALPHA=32
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
TRAIN_DATA_PATH=./data/train.jsonl
OUTPUT_DIR=./outputs/phi3
NUM_EPOCHS=5
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
MAX_SEQ_LENGTH=2048
BF16=true
```

#### Definition of Done

- [ ] Созданы 2+ примера конфигураций
- [ ] Примеры покрывают разные сценарии (маленькая/большая модель, CPU/GPU)
- [ ] Комментарии на русском языке
- [ ] Тесты: n/a (конфигурационные файлы)

---

## 10. Риски и смягчение

| # | Риск | Смягчение |
|---|------|-----------|
| 1 | pydantic-settings сложнее ожидаемого | Использовать документацию и примеры |
| 2 | Неполнота списка полей Settings | При реализации следующих спринтов добавлять недостающие поля |
| 3 | CLI требует рефакторинга при добавлении команд | Использовать subparsers, легко расширять |

## 11. Сводная таблица задач спринта

| #   | Задача                                   | Приоритет | Объём | Статус | Зависит от  |
|-----|------------------------------------------|:---------:|:-----:|:------:|:-----------:|
| 1.1 | Configuration (`app/core/config.py`)    | critical  | M     | Done | —           |
| 1.2 | Logging (`app/core/logging_config.py`)   | high      | S     | Done   | —           |
| 1.3 | Exceptions (`app/core/exceptions.py`)    | medium    | XS    | Done   | —           |
| 2.1 | CLI Skeleton (`app/__main__.py`)        | high      | M     | Progress   | 1.1, 1.2    |
| 3.1 | Device Utils (`app/utils/device.py`)     | medium    | S     | ToDo   | —           |
| 3.2 | Memory Utils (`app/utils/memory.py`)     | medium    | S     | ToDo   | —           |
| 4.1 | Test Fixtures (`tests/conftest.py`)      | medium    | S     | ToDo   | 1.1         |
| 5.1 | Example Configs (`configs/`)             | low       | S     | ToDo   | 1.1         |

## 12. История изменений спринта

- **2026-04-30** — спринт открыт
- **2026-04-30** — добавлены задачи 3.1, 3.2 (Utils), 4.1 (Test Fixtures)

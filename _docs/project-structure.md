# Структура проекта

## Целевое дерево

```
fine-tuning/
├── _docs/                      # Проектная документация
│   ├── README.md               # Индекс документации
│   ├── architecture.md         # Архитектура системы
│   ├── requirements.md         # Требования (FR/NFR/CON)
│   ├── configuration.md        # Конфигурация через .env
│   ├── training.md             # Процесс обучения
│   ├── inference.md            # Инференс и генерация
│   ├── testing.md              # Стратегия тестирования
│   ├── instructions.md         # Правила разработки
│   ├── project-structure.md    # Этот файл
│   ├── roadmap.md              # Дорожная карта
│   └── current-state.md        # Текущее состояние
│
├── _board/                     # Доска задач и спринты
│   ├── README.md               # Описание процесса
│   ├── plan.md                 # План спринтов
│   ├── process.md              # Пошаговый процесс задачи
│   ├── progress.txt            # Текущие заметки
│   └── sprints/
│       ├── 00-bootstrap.md     # Спринт 00 (инфраструктура)
│       └── 01-core-infra.md    # Спринт 01 (реализация)
│
├── app/                        # Исходный код приложения
│   ├── __init__.py
│   ├── __main__.py             # Точка входа: python -m app
│   ├── core/                   # Ядро системы
│   │   ├── __init__.py
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   ├── logging_config.py   # Конфигурация логирования
│   │   └── exceptions.py       # Базовые исключения
│   │
│   ├── models/                 # Работа с моделями
│   │   ├── __init__.py
│   │   ├── base.py             # Загрузка базовой модели
│   │   ├── lora.py             # LoRA конфигурация и применение
│   │   └── registry.py         # Реестр поддерживаемых моделей
│   │
│   ├── data/                   # Работа с данными
│   │   ├── __init__.py
│   │   ├── loader.py           # Загрузка данных из разных источников
│   │   ├── formatter.py        # Форматирование данных
│   │   └── tokenizer.py        # Токенизация датасетов
│   │
│   ├── training/               # Обучение
│   │   ├── __init__.py
│   │   ├── trainer.py          # LoRATrainer (обёртка над Trainer)
│   │   ├── config.py           # TrainingConfig
│   │   └── callbacks.py        # Callbacks (logging, wandb, etc.)
│   │
│   ├── inference/              # Инференс
│   │   ├── __init__.py
│   │   ├── engine.py           # InferenceEngine
│   │   └── cli.py              # CLI для инференса
│   │
│   └── utils/                  # Утилиты
│       ├── __init__.py
│       ├── device.py           # Определение устройств (cuda/cpu)
│       ├── memory.py           # Мониторинг памяти
│       └── file.py             # Работа с файлами
│
├── tests/                      # Тесты
│   ├── __init__.py
│   ├── conftest.py             # Фикстуры pytest
│   ├── unit/                   # Unit тесты
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── models/
│   │   ├── data/
│   │   └── training/
│   └── integration/            # Integration тесты
│       └── __init__.py
│
├── configs/                    # Примеры конфигураций
│   ├── tinyllama_lora.yaml
│   └── phi3_lora.yaml
│
├── data/                       # Данные (в .gitignore)
│   └── .gitkeep
│
├── logs/                       # Логи (в .gitignore)
│   └── .gitkeep
│
├── checkpoints/                # Чекпоинты (в .gitignore)
│   └── .gitkeep
│
├── outputs/                    # Результаты (в .gitignore)
│   └── .gitkeep
│
├── .env                        # Локальная конфигурация (в .gitignore)
├── .env.example                # Шаблон конфигурации
├── .gitignore                  # Игнорируемые файлы
├── pyproject.toml              # Конфигурация Python проекта
├── requirements.txt            # Зависимости
└── README.md                   # Главный README
```

## Описание директорий

### `_docs/`

Проектная документация — исчерпывающее описание архитектуры, требований, процессов. Читается перед работой с кодом.

### `_board/`

Управление проектом: спринты, задачи, процессы. Содержит:
- `plan.md` — индекс спринтов
- `process.md` — как выполнять задачу
- `sprints/*.md` — детали каждого спринта

### `app/`

Исходный код приложения. Структура отражает архитектурные слои:

| Директория | Назначение |
|------------|------------|
| `core/` | Конфигурация, логирование, базовые исключения |
| `models/` | Загрузка моделей, LoRA, реестр моделей |
| `data/` | Загрузка, форматирование, токенизация данных |
| `training/` | Тренировочный цикл, конфигурация, callbacks |
| `inference/` | Генерация текста, интерактивный режим |
| `utils/` | Вспомогательные функции |

### `tests/`

Тесты, зеркалят структуру `app/`:

| Директория | Назначение |
|------------|------------|
| `unit/` | Изолированные тесты с моками |
| `integration/` | Сквозные тесты компонентов |
| `conftest.py` | Общие фикстуры pytest |

### `configs/`

Примеры конфигураций для различных сценариев.

### `data/`, `logs/`, `checkpoints/`, `outputs/`

Runtime-директории, содержимое в `.gitignore`. Используются для:
- `data/` — входные данные
- `logs/` — файлы логов
- `checkpoints/` — промежуточные чекпоинты
- `outputs/` — финальные результаты

## Что должно попасть в `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/

# Virtual environments
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp

# Project specific
data/*
!data/.gitkeep
logs/*
!logs/.gitkeep
checkpoints/*
!checkpoints/.gitkeep
outputs/*
!outputs/.gitkeep

# Model artifacts
*.bin
*.safetensors
*.pt
*.pth
*.ckpt

# Test artifacts
.pytest_cache/
.coverage
htmlcov/

# Jupyter
.ipynb_checkpoints
```

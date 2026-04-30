# LLM Fine-Tuning Framework

Расширяемый фреймворк для параметро-эффективного дообучения больших языковых моделей (LLM) с использованием LoRA (Low-Rank Adaptation). Поддерживает любые модели из HuggingFace Hub, гибкую конфигурацию через environment variables и расширяемую архитектуру для работы с различными форматами данных.

## Возможности

- **LoRA Fine-Tuning** — эффективное дообучение с минимальными вычислительными затратами
- **Поддержка любых моделей** — работа с любыми causal language models из HuggingFace Hub
- **Гибкая конфигурация** — все параметры через `.env` файл, без изменения кода
- **Инструкционное дообучение** — поддержка форматов instruction/response для chat-моделей
- **Мониторинг** — интеграция с Weights & Biases и TensorBoard
- **Логирование** — структурированное логирование всех этапов
- **CLI-интерфейс** — удобный запуск тренировки и инференса
- **Покрытие тестами** — unit и integration тесты

## Требования

- **Python** 3.10+
- **CUDA** 11.8+ (для GPU-ускорения) или CPU-only режим
- **ОЗУ** — минимум 8GB (16GB+ рекомендуется)
- **Диск** — 10GB+ для кэша моделей

## Установка

```bash
git clone <repo-url>
cd fine-tuning

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Настройка

1. Скопировать шаблон конфигурации:

```bash
cp .env.example .env
```

2. Отредактировать `.env` под вашу задачу:

```bash
# Пример базовой настройки
BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
TRAIN_DATA_PATH=./data/train.jsonl
OUTPUT_DIR=./outputs
NUM_EPOCHS=3
LEARNING_RATE=2e-4
```

3. Подготовить данные в формате JSONL:

```jsonl
{"text": "Инструкция: Напиши приветствие\nОтвет: Привет! Как дела?"}
{"text": "Инструкция: Объясни Python\nОтвет: Python — язык программирования..."}
```

## Использование

### Обучение

```bash
python -m app train --config .env
```

Или с переопределением параметров:

```bash
python -m app train \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data-path ./data/custom.jsonl \
  --output-dir ./outputs/experiment-1 \
  --epochs 5
```

### Инференс

```bash
python -m app inference \
  --model-path ./checkpoints/final \
  --prompt "Напиши код на Python для сортировки списка"
```

Или интерактивный режим:

```bash
python -m app inference --interactive
```

## Структура проекта

```
fine-tuning/
├── _docs/              # Проектная документация
├── _board/             # Доска задач и спринты
├── app/                # Исходный код приложения
│   ├── core/           # Ядро: конфигурация, логирование
│   ├── models/         # Работа с моделями и адаптерами
│   ├── data/           # Загрузка и обработка данных
│   ├── training/       # Тренировочный цикл
│   └── inference/      # Инференс и генерация
├── configs/            # Примеры конфигураций
├── tests/              # Тесты
├── data/               # Данные (в .gitignore)
├── logs/               # Логи (в .gitignore)
├── checkpoints/        # Чекпоинты (в .gitignore)
└── outputs/            # Результаты (в .gitignore)
```

## Документация

- 📘 [`_docs/README.md`](./_docs/README.md) — индекс документации
- 🏗️ [`_docs/architecture.md`](./_docs/architecture.md) — архитектура компонентов
- 📋 [`_docs/requirements.md`](./_docs/requirements.md) — требования и ограничения
- ⚙️ [`_docs/configuration.md`](./_docs/configuration.md) — полный гайд по конфигурации
- 🎯 [`_docs/training.md`](./_docs/training.md) — процесс обучения
- 🤖 [`_docs/inference.md`](./_docs/inference.md) — инференс и генерация
- 🧪 [`_docs/testing.md`](./_docs/testing.md) — стратегия тестирования

## Тесты

```bash
# Запуск всех тестов
pytest -q

# С покрытием
pytest --cov=app --cov-report=term-missing

# Только unit-тесты
pytest -m unit

# Только integration-тесты
pytest -m integration
```

## Конфигурация через .env

Ключевые переменные:

| Переменная | Описание | Пример |
|------------|----------|--------|
| `BASE_MODEL_NAME` | Базовая модель | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `LORA_R` | LoRA rank | `8` |
| `LORA_ALPHA` | LoRA alpha | `16` |
| `TRAIN_DATA_PATH` | Путь к данным | `./data/train.jsonl` |
| `NUM_EPOCHS` | Количество эпох | `3` |
| `LEARNING_RATE` | Learning rate | `2e-4` |
| `OUTPUT_DIR` | Директория вывода | `./outputs` |

Полный список — в `.env.example`.

## Лицензия

MIT

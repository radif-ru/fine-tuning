# LLM Fine-Tuning Framework

Расширяемый фреймворк для параметро-эффективного дообучения больших языковых моделей (LLM) с использованием LoRA (Low-Rank Adaptation). Поддерживает любые модели из HuggingFace Hub, гибкую конфигурацию через environment variables и расширяемую архитектуру для работы с различными форматами данных.

## Возможности

- **🧠 LoRA Fine-Tuning** — эффективное дообучение с минимальными вычислительными затратами через PEFT
- **🤗 Поддержка HuggingFace Hub** — работа с любыми causal language models
- **⚙️ Гибкая конфигурация** — все параметры через `.env` файл, без изменения кода
- **📊 Множественные форматы данных** — JSONL, JSON, CSV, HF Datasets с шаблонами Alpaca и ShareGPT
- **🎲 Генератор датасетов** — синтетическая генерация данных для обучения с расширяемой архитектурой
- **💾 Checkpointing** — сохранение и возобновление обучения с любого шага
- **🎯 Инференс** — single prompt, интерактивный REPL, batch processing из файла
- **📦 Export** — merge LoRA адаптера в полную модель для деплоя
- **🔢 8-bit Quantization** — обучение и инференс с квантизацией для экономии памяти
- **📈 Мониторинг** — интеграция с Weights & Biases и TensorBoard
- **📝 Структурированное логирование** — все этапы с ротацией логов
- **🖥️ CLI-интерфейс** — удобный запуск всех операций
- **🧪 Покрытие тестами** — 70%+ покрытие, unit и integration тесты

## Требования

- **Python** 3.10+
- **CUDA** 11.8+ (для GPU-ускорения) или CPU-only режим
- **ОЗУ** — минимум 8GB (16GB+ рекомендуется)
- **Диск** — 10GB+ для кэша моделей

## Quick Start

Полный workflow для дообучения TinyLlama с LoRA:

```bash
# 1. Клонировать и установить
git clone <repo-url>
cd fine-tuning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Настроить окружение
cp configs/tinyllama_lora.env .env
# Или отредактируйте .env под вашу задачу

# 3. Генерация синтетического датасета (Alpaca формат)
python -m app generate-dataset \
  --type synthetic \
  --output data/synthetic_train.jsonl \
  --count 100 \
  --topics programming science \
  --seed 42

# 4. Обучение с LoRA
python -m app --config configs/tinyllama_lora.env train \
  --data-path data/synthetic_train.jsonl

# 5. Тестирование дообученной модели (интерактивный режим)
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --interactive

# 6. Сборка финальной модели (merge LoRA адаптера)
python -m app export \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --output-path ./merged-model
```

### Параметры конфигурации (из configs/tinyllama_lora.env)

```bash
# LoRA параметры (оптимальные для TinyLlama)
LORA_R=8              # Rank адаптера
LORA_ALPHA=32         # Alpha параметр масштабирования
LORA_TARGET_MODULES=q_proj,v_proj  # Модули для адаптации

# Параметры обучения
NUM_EPOCHS=3          # Количество эпох
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=512

# Оптимизации
FP16=true             # Смешанная точность
GRADIENT_CHECKPOINTING=true  # Экономия памяти
```

## Полный workflow

Проект поддерживает полный цикл fine-tuning LLM с LoRA:

### 1. Генерация датасета

Синтетическая генерация данных для обучения в формате Alpaca:

```bash
# Генерация 100 примеров по теме программирования
python -m app generate-dataset \
  --type synthetic \
  --output data/programming_train.jsonl \
  --count 100 \
  --topics programming \
  --seed 42

# Генерация QA пар (вопросы-ответы)
python -m app generate-dataset \
  --type qa \
  --output data/qa_train.jsonl \
  --count 100 \
  --categories technical practical

# Генерация в CSV формате
python -m app generate-dataset \
  --type synthetic \
  --output data/train.csv \
  --count 200 \
  --format csv
```

Формат данных: JSONL с полями `instruction`, `input`, `output` (Alpaca формат).

### 2. Обучение с LoRA

Обучение базовой модели с применением LoRA адаптеров:

```bash
# Использование готовой конфигурации для TinyLlama
python -m app --config configs/tinyllama_lora.env train \
  --data-path data/synthetic_train.jsonl

# Переопределение параметров из командной строки
python -m app --config configs/tinyllama_lora.env train \
  --data-path data/synthetic_train.jsonl \
  --epochs 5 \
  --batch-size 2 \
  --output-dir ./outputs/experiment-1

# Возобновление обучения из чекпоинта
python -m app --config configs/tinyllama_lora.env train \
  --resume-from-checkpoint ./outputs/tinyllama/checkpoint-500
```

### 3. Тестирование модели

Запуск инференса с дообученной моделью:

```bash
# Одиночный промпт
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --prompt "Напиши функцию на Python для сортировки списка"

# Интерактивный режим (REPL)
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --interactive

# Batch обработка из файла
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --input-file data/prompts.txt \
  --output-file data/results.txt
```

### 4. Сборка финальной модели

Объединение LoRA адаптера с базовой моделью для деплоя:

```bash
# Merge и сохранение полной модели
python -m app export \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --output-path ./merged-model

# Использование объединённой модели для инференса
python -m app inference \
  --base-model ./merged-model \
  --prompt "Пример текста"
```

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

Используйте готовую конфигурацию или создайте свою:

```bash
# Использование готовой конфигурации для TinyLlama
cp configs/tinyllama_lora.env .env

# Или создание своей конфигурации
cp .env.example .env
# Отредактируйте .env с нужными параметрами
```

Пример базовой настройки в `.env`:

```bash
# Модель
BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# LoRA параметры
LORA_R=8
LORA_ALPHA=32
LORA_TARGET_MODULES=q_proj,v_proj

# Параметры обучения
TRAIN_DATA_PATH=./data/train.jsonl
OUTPUT_DIR=./outputs
NUM_EPOCHS=3
LEARNING_RATE=2e-4
PER_DEVICE_BATCH_SIZE=4
```

### Подготовка данных

Данные должны быть в формате JSONL с полями `instruction`, `input`, `output`:

```jsonl
{"instruction": "Напиши приветствие", "input": "", "output": "Привет! Как дела?"}
{"instruction": "Объясни Python", "input": "", "output": "Python — язык программирования..."}
```

Или используйте синтетический генератор (см. Quick Start).

## Использование

### Обучение

```bash
python -m app train
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
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Напиши код на Python для сортировки списка"
```

Или интерактивный режим:

```bash
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --interactive
```

### Генерация датасетов

Синтетическая генерация данных для обучения:

```bash
# Базовая генерация (все темы)
python -m app generate-dataset \
  --type synthetic \
  --output data/synthetic.jsonl \
  --count 100

# С выбором тем
python -m app generate-dataset \
  --type synthetic \
  --output data/programming.jsonl \
  --count 50 \
  --topics programming

# QA генератор (вопросы-ответы)
python -m app generate-dataset \
  --type qa \
  --output data/qa.jsonl \
  --count 100 \
  --categories technical practical

# В CSV формате с seed для воспроизводимости
python -m app generate-dataset \
  --type synthetic \
  --output data/synthetic.csv \
  --count 200 \
  --format csv \
  --seed 42
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

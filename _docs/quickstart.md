# Quick Start Guide

Полный пошаговый гайд для дообучения TinyLlama с LoRA.

## Предварительные требования

- Python 3.10+
- CUDA 11.8+ (для GPU) или CPU-only режим
- 8GB+ RAM (16GB+ рекомендуется)
- 10GB+ дискового пространства для кэша моделей

## Шаг 1: Установка

```bash
# Клонирование репозитория
git clone <repo-url>
cd fine-tuning

# Создание виртуального окружения
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# или .venv\Scripts\activate  # Windows

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
```

## Шаг 2: Настройка конфигурации

```bash
# Использование готовой конфигурации для TinyLlama
cp configs/tinyllama_lora.env .env
```

Конфигурация `configs/tinyllama_lora.env` уже содержит оптимальные параметры:
- Базовая модель: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- LoRA rank: 8
- LoRA alpha: 32
- Target modules: q_proj, v_proj
- Epochs: 3
- Batch size: 4
- Learning rate: 2e-4
- FP16: включено
- Gradient checkpointing: включено

## Шаг 3: Генерация датасета

Генерация синтетических данных в формате Alpaca:

```bash
python -m app generate-dataset \
  --type synthetic \
  --output data/synthetic_train.jsonl \
  --count 100 \
  --topics programming science \
  --seed 42
```

Формат данных (JSONL):
```jsonl
{"instruction": "Напиши функцию на Python для сортировки списка", "input": "", "output": "Вот пример функции..."}
{"instruction": "Объясни концепцию рекурсии", "input": "", "output": "Рекурсия — это..."}
```

Опции генератора:
- `--type`: synthetic или qa
- `--output`: путь к выходному файлу
- `--count`: количество примеров
- `--topics`: темы (programming, science, general)
- `--seed`: seed для воспроизводимости
- `--format`: jsonl или csv

## Шаг 4: Обучение с LoRA

```bash
python -m app --config configs/tinyllama_lora.env train \
  --data-path data/synthetic_train.jsonl
```

Опции обучения:
- `--config`: путь к конфигурационному файлу (должен быть перед subcommand)
- `--data-path`: путь к данным
- `--base-model`: базовая модель (переопределяет конфиг)
- `--output-dir`: директория вывода
- `--epochs`: количество эпох
- `--batch-size`: batch size
- `--resume-from-checkpoint`: продолжение из чекпоинта

Процесс обучения:
1. Загрузка базовой модели из HuggingFace Hub
2. Применение LoRA адаптеров
3. Загрузка и форматирование данных
4. Токенизация
5. Обучение с логированием
6. Сохранение LoRA адаптера в `outputs/tinyllama/final`

## Шаг 5: Тестирование модели

Интерактивный режим (REPL):

```bash
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --interactive
```

Одиночный промпт:

```bash
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --prompt "Напиши функцию на Python для сортировки списка"
```

Опции инференса:
- `--base-model`: базовая модель или путь к объединённой модели
- `--adapter-path`: путь к LoRA адаптеру
- `--prompt`: одиночный промпт
- `--interactive`: интерактивный режим
- `--input-file`: файл с промптами (batch режим)
- `--output-file`: файл для сохранения результатов
- `--max-new-tokens`: максимум новых токенов (default: 256)
- `--temperature`: temperature для sampling (default: 0.7)
- `--top-p`: top-p sampling (default: 0.9)
- `--top-k`: top-k sampling (default: 50)
- `--prompt-template`: шаблон промптов (alpaca, raw, chat)

## Шаг 6: Сборка финальной модели

Объединение LoRA адаптера с базовой моделью:

```bash
python -m app export \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --output-path ./merged-model
```

Опции экспорта:
- `--base-model`: базовая модель
- `--adapter-path`: путь к LoRA адаптеру
- `--output-path`: путь для сохранения объединённой модели

После объединения модель можно использовать без адаптера:

```bash
python -m app inference \
  --base-model ./merged-model \
  --prompt "Пример текста"
```

## Полный скрипт (one-liner)

```bash
# Установка
pip install -r requirements.txt

# Настройка
cp configs/tinyllama_lora.env .env

# Данные
python -m app generate-dataset --type synthetic --output data/synthetic_train.jsonl --count 100 --topics programming science --seed 42

# Обучение
python -m app --config configs/tinyllama_lora.env train --data-path data/synthetic_train.jsonl

# Тестирование
python -m app inference --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-path outputs/tinyllama/final --interactive

# Экспорт
python -m app export --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-path outputs/tinyllama/final --output-path ./merged-model
```

## Troubleshooting

### Ошибка: ModuleNotFoundError
Убедитесь, что вы активировали виртуальное окружение:
```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate  # Windows
```

### Ошибка: CUDA out of memory
Уменьшите `PER_DEVICE_BATCH_SIZE` в конфигурации или используйте CPU:
```bash
PER_DEVICE_BATCH_SIZE=2
# или
FP16=false
GRADIENT_CHECKPOINTING=true
```

### Медленная загрузка модели
Используйте кэш HuggingFace:
```bash
HF_CACHE_DIR=/path/to/cache
```

## Дополнительные ресурсы

- [README.md](../README.md) — полное описание проекта
- [training.md](./training.md) — детальный процесс обучения
- [architecture.md](./architecture.md) — архитектура компонентов
- [configuration.md](./configuration.md) — полная конфигурация

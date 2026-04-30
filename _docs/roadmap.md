# Дорожная карта

## Этап 0: Bootstrap (Спринт 00) — ✅

**Цель:** Подготовить инфраструктуру проекта

**Задачи:**
- [x] Структура каталогов
- [x] Корневые файлы (.gitignore, README.md, requirements.txt)
- [x] Полный набор документации в `_docs/`
- [x] Доска задач в `_board/`
- [x] Скелет пакетов `app/` и `tests/`
- [x] Шаблон `.env.example`

**Статус:** Завершён

---

## Этап 1: Core Infrastructure (Спринт 01)

**Цель:** Реализовать базовую инфраструктуру проекта

**Длительность:** 1-2 недели

**Задачи:**
1. Конфигурация (`app/core/`)
   - `config.py` — Settings на pydantic-settings
   - `logging_config.py` — настройка логирования
   - `exceptions.py` — иерархия исключений

2. CLI скелет (`app/__main__.py`)
   - Аргументы командной строки
   - Базовая структура команд train/inference

3. Utils (`app/utils/`)
   - `device.py` — определение устройств (CUDA/CPU)
   - `memory.py` — мониторинг памяти GPU

4. Test Infrastructure (`tests/`)
   - `conftest.py` — фикстуры pytest
   - Unit-тесты для core и utils

5. Examples (`configs/`)
   - Примеры конфигураций TinyLlama, Phi-3

**Acceptance Criteria:**
- `python -m app --help` работает
- `pytest tests/unit/core/` проходит
- Конфигурация загружается из `.env`
- Утилиты device/memory работают на CPU и CUDA

---

## Этап 2: Model Management (Спринт 02)

**Цель:** Загрузка моделей и применение LoRA

**Длительность:** 1-2 недели

**Задачи:**
1. Base Model Loader (`app/models/base.py`)
   - Загрузка из HuggingFace Hub
   - Загрузка из локального пути
   - Auto device map

2. LoRA Manager (`app/models/lora.py`)
   - LoRAConfig dataclass
   - Применение LoRA через PEFT
   - Сохранение/загрузка адаптеров

3. Model Registry (`app/models/registry.py`)
   - Реестр поддерживаемых моделей
   - Специфичные настройки для моделей

4. Тесты (`tests/unit/models/`)

**Acceptance Criteria:**
- Загрузка GPT2/TinyLlama работает
- LoRA применяется корректно
- Адаптер сохраняется и загружается

---

## Этап 3: Data Pipeline (Спринт 03)

**Цель:** Загрузка и обработка данных

**Длительность:** 1-2 недели

**Задачи:**
1. Data Loader (`app/data/loader.py`)
   - JSONL format
   - JSON format
   - CSV format
   - HuggingFace datasets

2. Data Formatter (`app/data/formatter.py`)
   - Instruction formatting (Alpaca-style)
   - Plain text formatting
   - Custom templates

3. Tokenizer (`app/data/tokenizer.py`)
   - Dataset tokenization
   - Batch encoding
   - Truncation/Padding

4. Тесты (`tests/unit/data/`)

**Acceptance Criteria:**
- Загрузка всех форматов работает
- Форматирование instruction корректно
- Токенизация без ошибок

---

## Этап 4: Training (Спринт 04)

**Цель:** Тренировочный цикл

**Длительность:** 2-3 недели

**Задачи:**
1. Training Config (`app/training/config.py`)
   - TrainingArguments wrapper
   - Валидация параметров

2. LoRA Trainer (`app/training/trainer.py`)
   - Интеграция с transformers.Trainer
   - Поддержка resume from checkpoint

3. Callbacks (`app/training/callbacks.py`)
   - LoggingCallback
   - WandbCallback
   - CheckpointCallback

4. CLI train command (`app/training/cli.py`)

5. Тесты (`tests/unit/training/`, `tests/integration/`)

**Acceptance Criteria:**
- Тренировка на GPT2 проходит
- Чекпоинты сохраняются
- Логирование работает
- Resume from checkpoint работает

---

## Этап 5: Inference (Спринт 05)

**Цель:** Генерация текста

**Длительность:** 1 неделя

**Задачи:**
1. Inference Engine (`app/inference/engine.py`)
   - Загрузка базовой модели + LoRA
   - Generation с параметрами
   - Batch inference

2. CLI inference (`app/inference/cli.py`)
   - Single prompt
   - Interactive mode
   - File input/output

3. Тесты (`tests/unit/inference/`)

**Acceptance Criteria:**
- Генерация работает
- Interactive mode функционирует
- Batch processing работает

---

## Этап 6: Polish & Documentation (Спринт 06)

**Цель:** Доработка и документация

**Длительность:** 1 неделя

**Задачи:**
1. Примеры конфигураций (`configs/`)
2. Примеры данных (`examples/`)
3. Полное покрытие тестами (>70%)
4. Обновление README.md
5. Финальная проверка документации

**Acceptance Criteria:**
- Примеры работают из коробки
- Покрытие ≥ 70%
- Документация актуальна

---

## Future Enhancements (вне скоупа MVP)

### QLoRA (4-bit)

Поддержка 4-bit квантизации для обучения на GPU с меньшей памятью.

### Другие PEFT методы

- IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- Prompt Tuning
- Prefix Tuning

### Multi-GPU Training

- DistributedDataParallel
- DeepSpeed integration

### Другие типы моделей

- Encoder-only (BERT-style) для классификации
- Encoder-decoder (T5-style) для seq2seq

### Оптимизации

- Flash Attention 2
- Unsloth (быстрое обучение)

### Инструменты

- Gradio UI для инференса
- REST API сервис
- Автоматический подбор гиперпараметров

## Текущий статус

| Этап | Статус | Прогресс |
|------|--------|----------|
| 0. Bootstrap | ✅ Завершён | 100% |
| 1. Core Infrastructure | 📝 Запланирован | 0% |
| 2. Model Management | 📋 Backlog | 0% |
| 3. Data Pipeline | 📋 Backlog | 0% |
| 4. Training | 📋 Backlog | 0% |
| 5. Inference | 📋 Backlog | 0% |
| 6. Polish | 📋 Backlog | 0% |

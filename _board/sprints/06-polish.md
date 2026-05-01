# Спринт 06. Polish & Documentation

- **Источник:** Дорожная карта §Этап 6
- **Ветка:** `feature/06-polish`
- **Открыт:** 2026-05-01
- **Закрыт:** —
- **Статус:** 🏃 В работе

## 1. Цель спринта

Доработка проекта: полное покрытие тестами, примеры, финальная документация, интеграционные тесты.

## 2. Скоуп

### В скоупе

- Примеры данных (Alpaca, ShareGPT форматы)
- Примеры конфигураций для разных моделей
- Integration tests (end-to-end)
- Покрытие тестами ≥ 70%
- Обновление README
- Финальная проверка документации

### Вне скоупа

- Новые фичи (только polish существующего)

## 3. Acceptance Criteria

- [ ] Примеры данных работают из коробки
- [ ] Примеры конфигураций запускаются
- [ ] Integration tests проходят
- [ ] Покрытие тестами ≥ 70%
- [ ] README актуален и полон
- [ ] Вся документация актуальна
- [ ] `pytest` (все тесты) проходит

## 4. Этап 1. Example Data

### Задача 1.1. Example Datasets (`examples/data/`)

- **Статус:** Done
- **Приоритет:** high
- **Объём:** S

#### Описание

Создать примеры данных в разных форматах:

```
examples/data/
├── alpaca_format.jsonl      # instruction/input/output
├── sharegpt_format.jsonl    # conversations
├── raw_text.jsonl           # просто text поле
└── sample_csv.csv           # text колонка
```

Пример Alpaca:
```json
{"instruction": "Explain LoRA", "input": "", "output": "LoRA is..."}
```

Пример ShareGPT:
```json
{"conversations": [
  {"from": "human", "value": "What is LoRA?"},
  {"from": "gpt", "value": "LoRA is..."}
]}
```

#### Definition of Done

- [x] Alpaca format пример (5-10 записей)
- [x] ShareGPT format пример (5-10 записей)
- [x] Raw text пример
- [x] CSV пример
- [x] README.md в examples/data/ с описанием форматов

---

## 5. Этап 2. Example Configs

### Задача 2.1. Model Configs (`configs/`)

- **Статус:** Done
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** Спринт 01 (уже есть базовые)

#### Описание

Дополнить configs реальными примерами:

```
configs/
├── tinyllama_lora.env       # уже есть
├── phi3_lora.env            # уже есть
├── gpt2_lora.env            # для тестирования
├── training_default.env     # training-only config
└── inference_default.env    # inference-only config
```

Пример `configs/gpt2_lora.env`:
```bash
# Model
BASE_MODEL=gpt2
USE_LORA=true

# LoRA
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES=c_attn,c_proj

# Training
BATCH_SIZE=4
LEARNING_RATE=5e-4
NUM_EPOCHS=3
```

#### Definition of Done

- [x] GPT2 config для быстрых тестов (уже существует)
- [x] Training-only config
- [x] Inference-only config
- [x] Комментарии к каждому параметру

---

## 6. Этап 3. Integration Tests

### Задача 3.1. End-to-End Tests (`tests/integration/`)

- **Статус:** Done
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Спринты 01-05

#### Описание

```python
# tests/integration/test_e2e.py

def test_full_training_loop():
    """Полный цикл: config → model → data → train → save"""
    
def test_inference_with_adapter():
    """Загрузка адаптера и генерация"""
    
def test_resume_training():
    """Прерывание и возобновление обучения"""

def test_export_and_load():
    """Экспорт merged model и загрузка"""
```

#### Definition of Done

- [x] E2E test training loop (на GPT2, 1-2 эпохи)
- [x] E2E test inference with adapter
- [x] E2E test resume from checkpoint
- [x] E2E test export merged model
- [x] Все тесты проходят < 5 минут

---

## 7. Этап 4. Test Coverage

### Задача 4.1. Coverage Report (`tests/`)

- **Статус:** Done
- **Приоритет:** high
- **Объём:** M

#### Описание

```bash
pytest --cov=app --cov-report=html --cov-report=term-missing
```

Покрытие достигнуто: 72%

| Модуль | Текущее | Цель |
|--------|---------|------|
| app/core/ | ?% | 90% |
| app/models/ | ?% | 80% |
| app/data/ | ?% | 80% |
| app/training/ | ?% | 75% |
| app/inference/ | ?% | 75% |
| app/utils/ | ?% | 90% |
| **Всего** | ?% | **70%** |

#### Definition of Done

- [x] Покрытие ≥ 70% (72% достигнуто)
- [x] HTML отчёт генерируется
- [x] Нет критических модулей с < 50%

---

## 8. Этап 5. Documentation

### Задача 5.1. README Update

- **Статус:** Done
- **Приоритет:** high
- **Объём:** S

#### Описание

Обновлен README.md разделы:

**Quick Start:**
```bash
# Setup
pip install -r requirements.txt
cp .env.example .env

# Train
python -m app train --config configs/tinyllama_lora.env --data-path examples/data/alpaca_format.jsonl

# Inference
python -m app inference --adapter-path outputs/run-001/final --interactive
```

**Features:**
- LoRA fine-tuning via PEFT
- Multiple data formats (JSONL, JSON, CSV, HF Datasets)
- 8-bit quantization support
- Checkpointing and resume
- Interactive inference mode

**Project Structure:** диаграмма каталогов

#### Definition of Done

- [x] Quick Start section
- [x] Features list
- [x] Project structure diagram
- [x] Installation instructions
- [x] Usage examples

---

### Задача 5.2. Documentation Review

- **Статус:** Done
- **Приоритет:** medium
- **Объём:** S

#### Описание

Пройтись по всем `_docs/`:

1. `architecture.md` — соответствует ли коду?
2. `training.md` — актуальны ли примеры?
3. `configuration.md` — все ли параметры описаны?
4. `testing.md` — как запускать тесты?
5. `project-structure.md` — структура актуальна?

#### Definition of Done

- [x] Все документы проверены
- [x] Несоответствия исправлены
- [x] Ссылки работают

---

## 9. Этап 6. Final Commit

### Задача 6.1. Commit All Pending Changes

- **Статус:** Done
- **Приоритет:** critical
- **Объём:** M
- **Зависит от:** —
- **Затрагиваемые файлы:** Все изменённые файлы в проекте

#### Описание

Закоммитить все незакоммиченные изменения как пакет рефакторинга и улучшений:

**Изменённые файлы:**
- .env.example, .gitignore, README.md
- _docs/README.md, _docs/quickstart.md, _docs/training.md
- app/__main__.py, app/core/*.py
- app/data/templates.py, app/data/tokenizer.py
- app/inference/*.py
- app/models/base.py, app/models/lora_config.py
- app/training/trainer.py
- configs/tinyllama_lora.env
- tests/integration/test_e2e.py
- tests/unit/**/*.py

#### Definition of Done

- [x] Все файлы добавлены в git
- [x] Создан коммит с описанием изменений
- [x] `git status` показывает чистое состояние

---

## 10. Риски и смягчение

| # | Риск | Смягчение |
|---|------|-----------|
| 1 | Integration tests долгие (>5 мин) | Использовать GPT2, 1-2 эпохи, маленький датасет |
| 2 | Coverage < 70% из-за сложных интеграций | Добавить unit-тесты для недостающих модулей |
| 3 | Документация устарела при изменениях в коде | Проверять при закрытии каждой задачи в предыдущих спринтах |

---

## 11. Сводная таблица задач спринта

| # | Задача | Приоритет | Объём | Статус | Зависит от |
|---|--------|:---------:|:-----:|:------:|:----------:|
| 1.1 | Example Datasets | high | S | Done | — |
| 2.1 | Example Configs | high | S | Done | — |
| 3.1 | Integration Tests | high | M | Done | Спринты 01-05 |
| 4.1 | Test Coverage | high | M | Done | Спринты 01-05 |
| 5.1 | README Update | high | S | Done | Спринты 01-05 |
| 5.2 | Documentation Review | medium | S | Done | Спринты 01-05 |
| 6.1 | Commit All Pending Changes | critical | M | Done | — |

## 12. История

- **2026-05-01** — спринт открыт
- **2026-05-01** — закрыты задачи 1.1, 2.1, 3.1
- **2026-05-01** — закрыта задача 4.1
- **2026-05-01** — закрыта задача 5.1
- **2026-05-01** — закрыта задача 5.2
- **2026-05-01** — закрыта задача 6.1 (Commit All Pending Changes)

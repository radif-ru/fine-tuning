# Спринт 06. Polish & Documentation

- **Источник:** Дорожная карта §Этап 6
- **Ветка:** `feature/06-polish`
- **Открыт:** —
- **Закрыт:** —
- **Статус:** Backlog

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

- **Статус:** ToDo
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

- [ ] Alpaca format пример (5-10 записей)
- [ ] ShareGPT format пример (5-10 записей)
- [ ] Raw text пример
- [ ] CSV пример
- [ ] README.md в examples/data/ с описанием форматов

---

## 5. Этап 2. Example Configs

### Задача 2.1. Model Configs (`configs/`)

- **Статус:** ToDo
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

- [ ] GPT2 config для быстрых тестов
- [ ] Training-only config
- [ ] Inference-only config
- [ ] Комментарии к каждому параметру

---

## 6. Этап 3. Integration Tests

### Задача 3.1. End-to-End Tests (`tests/integration/`)

- **Статус:** ToDo
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

- [ ] E2E test training loop (на GPT2, 1-2 эпохи)
- [ ] E2E test inference with adapter
- [ ] E2E test resume from checkpoint
- [ ] E2E test export merged model
- [ ] Все тесты проходят < 5 минут

---

## 7. Этап 4. Test Coverage

### Задача 4.1. Coverage Report (`tests/`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** M

#### Описание

```bash
pytest --cov=app --cov-report=html --cov-report=term-missing
```

Добить покрытие до 70%:

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

- [ ] Покрытие ≥ 70%
- [ ] HTML отчёт генерируется
- [ ] Нет критических модулей с < 50%

---

## 8. Этап 5. Documentation

### Задача 5.1. README Update

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S

#### Описание

Обновить README.md разделы:

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

- [ ] Quick Start section
- [ ] Features list
- [ ] Project structure diagram
- [ ] Installation instructions
- [ ] Usage examples

---

### Задача 5.2. Documentation Review

- **Статус:** ToDo
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

- [ ] Все документы проверены
- [ ] Несоответствия исправлены
- [ ] Ссылки работают

---

## 9. Риски и смягчение

| # | Риск | Смягчение |
|---|------|-----------|
| 1 | Integration tests долгие (>5 мин) | Использовать GPT2, 1-2 эпохи, маленький датасет |
| 2 | Coverage < 70% из-за сложных интеграций | Добавить unit-тесты для недостающих модулей |
| 3 | Документация устарела при изменениях в коде | Проверять при закрытии каждой задачи в предыдущих спринтах |

---

## 10. Сводная таблица задач спринта

| # | Задача | Приоритет | Объём | Статус | Зависит от |
|---|--------|:---------:|:-----:|:------:|:----------:|
| 1.1 | Example Datasets | high | S | ToDo | — |
| 2.1 | Example Configs | high | S | ToDo | — |
| 3.1 | Integration Tests | high | M | ToDo | Спринты 01-05 |
| 4.1 | Test Coverage | high | M | ToDo | Спринты 01-05 |
| 5.1 | README Update | high | S | ToDo | Спринты 01-05 |
| 5.2 | Documentation Review | medium | S | ToDo | Спринты 01-05 |

## 11. История

- **YYYY-MM-DD** — спринт открыт

# Спринт 03. Data Pipeline

- **Источник:** Дорожная карта §Этап 3
- **Ветка:** `feature/03-data-pipeline`
- **Открыт:** —
- **Закрыт:** —
- **Статус:** Backlog

## 1. Цель спринта

Реализовать полный пайплайн обработки данных: загрузка из разных форматов, форматирование инструкций, токенизация.

## 2. Скоуп

### В скоупе

- Загрузка: JSONL, JSON, CSV, HuggingFace Datasets
- Форматирование: Alpaca-style, ShareGPT, Custom templates
- Токенизация с truncation/padding
- Датасеты для Trainer (Dataset / DataLoader)

### Вне скоупа

- Streaming для больших датасетов (будущее улучшение)
- Data augmentation

## 3. Acceptance Criteria

- [ ] Все форматы загружаются без ошибок
- [ ] Форматирование инструкций работает
- [ ] Токенизация с truncation/padding корректна
- [ ] Dataset совместим с HF Trainer
- [ ] `pytest tests/unit/data/` проходит

## 4. Этап 1. Data Loader

### Задача 1.1. Data Loader (`app/data/loader.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** M

#### Описание

```python
from datasets import load_dataset

class DataLoader:
    def load_jsonl(self, path: str) -> Dataset
    def load_json(self, path: str) -> Dataset
    def load_csv(self, path: str, text_column: str = "text") -> Dataset
    def load_hf_dataset(self, name: str, split: str = "train") -> Dataset
```

#### Definition of Done

- [ ] JSONL загрузка (список объектов с полями)
- [ ] JSON загрузка (единый объект или список)
- [ ] CSV загрузка с указанием text_column
- [ ] HF Datasets интеграция
- [ ] Валидация наличия обязательных полей
- [ ] Тесты: все форматы, ошибки при невалидных файлах

---

## 5. Этап 2. Data Formatter

### Задача 2.1. Prompt Templates (`app/data/templates.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S

#### Описание

```python
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    name: str
    template: str  # "{instruction}\n{input}\n{output}"
    
ALPACA_TEMPLATE = """Below is an instruction that describes a task...
### Instruction:
{instruction}
### Input:
{input}
### Response:
{output}"""

class TemplateRegistry:
    def get(self, name: str) -> PromptTemplate
    def register(self, name: str, template: str)
```

#### Definition of Done

- [ ] Alpaca template предустановлен
- [ ] ShareGPT template предустановлен
- [ ] Поддержка custom templates
- [ ] Валидация placeholder'ов в шаблоне
- [ ] Тесты: форматирование, регистрация

---

### Задача 2.2. Data Formatter (`app/data/formatter.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задача 2.1

#### Описание

```python
class DataFormatter:
    def format_alpaca(self, dataset: Dataset) -> Dataset
    def format_sharegpt(self, dataset: Dataset) -> Dataset
    def format_custom(self, dataset: Dataset, template: PromptTemplate) -> Dataset
    def format_conversation(self, messages: list[dict]) -> str  # для ShareGPT
```

#### Definition of Done

- [ ] Alpaca форматирование (instruction/input/output)
- [ ] ShareGPT форматирование (conversations)
- [ ] Обработка пустых полей (input может отсутствовать)
- [ ] Dataset.map() для batch processing
- [ ] Тесты: форматирование разных структур

---

## 6. Этап 3. Tokenizer

### Задача 3.1. Tokenizer Wrapper (`app/data/tokenizer.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** M
- **Зависит от:** Задача 2.2

#### Описание

```python
class TokenizerWrapper:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 2048)
    def tokenize(self, text: str) -> dict  # input_ids, attention_mask, labels
    def tokenize_dataset(self, dataset: Dataset, text_column: str = "text") -> Dataset
    def prepare_for_training(self, dataset: Dataset) -> Dataset  # sets labels = input_ids
```

#### Definition of Done

- [ ] Tokenization с truncation
- [ ] Padding (max_length или batch)
- [ ] Labels = input_ids для causal LM
- [ ] Dataset совместим с HF Trainer
- [ ] Поддержка chat_template для instruct моделей
- [ ] Тесты: tokenization, padding, truncation

---

## 7. Этап 4. Data Pipeline Integration

### Задача 4.1. Pipeline (`app/data/pipeline.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** Задачи 1.1, 2.2, 3.1

#### Описание

```python
class DataPipeline:
    def process(
        self,
        source: str,           # путь или имя HF датасета
        format_type: str,     # "alpaca", "sharegpt", "raw"
        tokenizer: TokenizerWrapper,
        split: str = "train"
    ) -> Dataset
```

#### Definition of Done

- [ ] Единый метод load → format → tokenize
- [ ] Поддержка всех format_type
- [ ] Проверка целостности данных на каждом этапе
- [ ] Тесты: полный pipeline end-to-end

---

## 8. Сводная таблица

| # | Задача | Приоритет | Статус |
|---|--------|:---------:|:------:|
| 1.1 | Data Loader | critical | ToDo |
| 2.1 | Prompt Templates | high | ToDo |
| 2.2 | Data Formatter | high | ToDo |
| 3.1 | Tokenizer Wrapper | critical | ToDo |
| 4.1 | Pipeline Integration | high | ToDo |

## 9. История

- **YYYY-MM-DD** — спринт открыт

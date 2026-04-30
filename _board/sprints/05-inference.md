# Спринт 05. Inference

- **Источник:** Дорожная карта §Этап 5
- **Ветка:** `feature/05-inference`
- **Открыт:** —
- **Закрыт:** —
- **Статус:** Backlog

## 1. Цель спринта

Реализовать инференс: генерацию текста с LoRA адаптерами, interactive mode, batch processing, export merged моделей.

## 2. Скоуп и non-goals

### В скоупе

- Inference Engine (загрузка модели + адаптера)
- Text generation с параметрами (temperature, top_p, max_new_tokens)
- Interactive CLI mode
- Batch inference из файла
- Streaming generation
- Merge & export адаптера в полную модель

### Вне скоупа

- REST API (в Future Enhancements)
- Gradio UI (в Future Enhancements)
- Quantized inference (GPTQ, AWQ) — в Future

## 3. Acceptance Criteria

- [ ] Генерация с LoRA адаптером работает
- [ ] Interactive mode функционирует
- [ ] Batch processing из файла работает
- [ ] Параметры generation (temperature, top_p) применяются
- [ ] Merge & export работает
- [ ] CLI inference команда работает
- [ ] `pytest tests/unit/inference/` проходит

## 4. Решения по архитектуре

| Решение | Обоснование |
|---------|-------------|
| **Merge before inference** | Оптимизация: merged модель быстрее для inference |
| **GenerationConfig dataclass** | Типизация и валидация параметров генерации |
| **Streaming через yield** | Эффективная передача токенов по мере генерации |

## 5. Этап 1. Inference Engine

### Задача 1.1. Generation Config (`app/inference/config.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** XS
- **Зависит от:** —
- **Связанные документы:** `_docs/inference.md`

#### Описание

```python
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
```

#### Definition of Done

- [ ] Dataclass с валидацией значений (temperature > 0, top_p в [0,1])
- [ ] Метод `from_env()` для загрузки из Settings
- [ ] Тесты: валидация, значения по умолчанию

---

### Задача 1.2. Inference Engine (`app/inference/engine.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** M
- **Зависит от:** Задача 1.1, Спринт 02
- **Связанные документы:** `_docs/architecture.md` §2.4, `_docs/inference.md`

#### Описание

```python
class InferenceEngine:
    def __init__(
        self,
        base_model_name: str,
        adapter_path: str = None,
        device: str = "auto",
        load_in_8bit: bool = False
    )
    
    def generate(self, prompt: str, config: GenerationConfig = None) -> str
    def generate_batch(self, prompts: list[str], config: GenerationConfig = None) -> list[str]
    def generate_stream(self, prompt: str, config: GenerationConfig = None) -> Iterator[str]
    def merge_and_unload(self) -> None  # для оптимизации inference
```

#### Definition of Done

- [ ] Загрузка базовой модели через ModelRegistry
- [ ] Загрузка LoRA адаптера (если указан)
- [ ] Generation с параметрами temperature, top_p, top_k
- [ ] Batch generation
- [ ] Streaming generation через yield
- [ ] merge_and_unload для оптимизации
- [ ] Тесты: генерация, batch, streaming

---

## 6. Этап 2. Prompt Builder

### Задача 2.1. Prompt Builder (`app/inference/prompt.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** —
- **Связанные документы:** `_docs/inference.md`

#### Описание

```python
class PromptBuilder:
    def __init__(self, template_name: str = "alpaca")
    
    def build(self, instruction: str, context: str = "") -> str
    
    # Templates:
    # - alpaca: Below is an instruction...\n### Instruction:\n{instruction}\n### Response:\n
    # - raw: {instruction}
    # - chat: apply_chat_template если доступен
```

#### Definition of Done

- [ ] Alpaca шаблон
- [ ] Raw шаблон (только текст)
- [ ] Chat template через tokenizer.apply_chat_template
- [ ] Тесты: построение промптов

---

## 7. Этап 3. CLI Inference

### Задача 3.1. Inference CLI (`app/inference/cli.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задача 1.2, 2.1
- **Связанные документы:** `_docs/architecture.md` §3.1

#### Описание

```bash
# Single prompt
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/run-001/final \
  --prompt "What is LoRA?" \
  --max-new-tokens 200 \
  --temperature 0.7

# Interactive mode
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/run-001/final \
  --interactive

# Batch from file
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/run-001/final \
  --input-file prompts.txt \
  --output-file results.txt
```

#### Definition of Done

- [ ] Single prompt mode с выводом результата
- [ ] Interactive mode с REPL (/quit, /clear, /help)
- [ ] Batch mode из файла (один prompt на строку)
- [ ] Параметры: temperature, top_p, max_new_tokens
- [ ] Graceful shutdown (Ctrl+C)
- [ ] Тесты: все режимы

---

## 8. Этап 4. Model Export

### Задача 4.1. Merge and Export (`app/inference/export.py`)

- **Статус:** ToDo
- **Приоритет:** medium
- **Объём:** S
- **Зависит от:** Задача 1.2
- **Связанные документы:** `_docs/inference.md`

#### Описание

```python
class ModelExporter:
    def merge_and_save(
        self,
        base_model_name: str,
        adapter_path: str,
        output_path: str,
        save_tokenizer: bool = True
    ) -> None
```

```bash
python -m app export \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/run-001/final \
  --output-path outputs/merged-model
```

#### Definition of Done

- [ ] Мерж LoRA в базовую модель через merge_and_unload()
- [ ] Сохранение полной модели (не только адаптера)
- [ ] Сохранение tokenizer
- [ ] CLI команда export
- [ ] Тесты: экспорт, загрузка объединённой модели

---

## 9. Сводная таблица задач спринта

| # | Задача | Приоритет | Объём | Статус | Зависит от |
|---|--------|:---------:|:-----:|:------:|:----------:|
| 1.1 | Generation Config | high | XS | ToDo | — |
| 1.2 | Inference Engine | critical | M | ToDo | 1.1 |
| 2.1 | Prompt Builder | high | S | ToDo | — |
| 3.1 | Inference CLI | high | M | ToDo | 1.2, 2.1 |
| 4.1 | Merge and Export | medium | S | ToDo | 1.2 |

## 10. История

- **YYYY-MM-DD** — спринт открыт

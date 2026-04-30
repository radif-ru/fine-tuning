# Примеры данных для fine-tuning

Эта директория содержит примеры данных в различных форматах для обучения и тестирования.

## Форматы данных

### 1. Alpaca Format (`alpaca_format.jsonl`)

Стандартный формат Alpaca с тремя полями:
- `instruction` - инструкция/задача
- `input` - дополнительный контекст (может быть пустым)
- `output` - ожидаемый ответ

```json
{"instruction": "Explain LoRA", "input": "", "output": "LoRA is..."}
```

### 2. ShareGPT Format (`sharegpt_format.jsonl`)

Формат диалогов для chat-моделей:
- `conversations` - список сообщений
  - `from`: "human" или "gpt"
  - `value`: текст сообщения

```json
{"conversations": [
  {"from": "human", "value": "What is LoRA?"},
  {"from": "gpt", "value": "LoRA is..."}
]}
```

### 3. Raw Text Format (`raw_text.jsonl`)

Простой формат с полем `text`:

```json
{"text": "Any text content here..."}
```

### 4. CSV Format (`sample_csv.csv`)

CSV файл с колонкой `text`:

```csv
text,category,source
"content...",technical,docs
```

## Использование

```bash
# Обучение с Alpaca форматом
python -m app train --config .env --data-path examples/data/alpaca_format.jsonl

# Обучение с CSV
python -m app train --config .env --data-path examples/data/sample_csv.csv
```

## Примечания

- Файлы в формате JSONL (JSON Lines) - по одному JSON объекту на строку
- CSV файл должен содержать колонку `text` или `instruction`/`output` пары
- Реальный датасет должен содержать больше примеров (рекомендуется >1000)

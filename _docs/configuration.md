# Конфигурация

Все параметры системы настраиваются через переменные окружения (`.env` файл) с использованием `pydantic-settings`.

## Основные секции

### Model Configuration

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `BASE_MODEL_NAME` | str | **required** | Имя модели в HuggingFace Hub или локальный путь |
| `HF_CACHE_DIR` | Optional[str] | `~/.cache/huggingface` | Директория кэша моделей |
| `TRUST_REMOTE_CODE` | bool | `false` | Доверять remote code при загрузке модели |

**Примеры значений:**
```bash
BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
BASE_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct
BASE_MODEL_NAME=./models/my-local-model
```

### LoRA Configuration

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `LORA_R` | int | `8` | LoRA rank — размер low-rank матриц |
| `LORA_ALPHA` | int | `16` | Scaling factor для LoRA |
| `LORA_DROPOUT` | float | `0.1` | Dropout rate для LoRA слоёв |
| `LORA_TARGET_MODULES` | list[str] | `q_proj,k_proj,v_proj,o_proj` | Модули для применения LoRA |
| `LORA_BIAS` | str | `none` | Режим обучения bias (`none`, `all`, `lora_only`) |
| `LORA_TASK_TYPE` | str | `CAUSAL_LM` | Тип задачи для LoRA |

**О target modules:**
- `q_proj,k_proj,v_proj,o_proj` — attention layers (стандарт)
- Добавить `gate_proj,up_proj,down_proj` для обучения MLP
- Конкретные модули зависят от архитектуры модели

**LORA_R и LORA_ALPHA:**
- Alpha обычно = 2x rank
- Больше rank → больше обучаемых параметров, лучше качество
- Меньше rank → меньше памяти, быстрее обучение

### Training Configuration

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `OUTPUT_DIR` | str | `./outputs` | Директория для результатов |
| `CHECKPOINT_DIR` | str | `./checkpoints` | Директория для чекпоинтов |
| `NUM_EPOCHS` | int | `3` | Количество эпох обучения |
| `PER_DEVICE_BATCH_SIZE` | int | `4` | Batch size на одно устройство |
| `GRADIENT_ACCUMULATION_STEPS` | int | `4` | Шаги накопления градиента |
| `LEARNING_RATE` | float | `2e-4` | Learning rate (2e-4 = 0.0002) |
| `WARMUP_RATIO` | float | `0.03` | Доля шагов для warmup |
| `WEIGHT_DECAY` | float | `0.01` | Weight decay для регуляризации |
| `MAX_GRAD_NORM` | float | `1.0` | Max norm для gradient clipping |
| `LOGGING_STEPS` | int | `10` | Частота логирования (в шагах) |
| `SAVE_STEPS` | int | `100` | Частота сохранения чекпоинтов |
| `EVAL_STRATEGY` | str | `steps` | Стратегия evaluation (`no`, `steps`, `epoch`) |
| `EVAL_STEPS` | int | `100` | Частота evaluation |
| `LOAD_BEST_MODEL_AT_END` | bool | `true` | Загружать лучшую модель в конце |
| `SAVE_TOTAL_LIMIT` | int | `3` | Максимальное количество сохранённых чекпоинтов |

**Эффективный batch size:**
```
effective_batch = per_device_batch_size * num_devices * gradient_accumulation_steps
```

### Data Configuration

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `TRAIN_DATA_PATH` | str | **required** | Путь к тренировочным данным |
| `VALIDATION_DATA_PATH` | Optional[str] | `None` | Путь к валидационным данным |
| `TEXT_COLUMN` | str | `text` | Имя колонки с текстом |
| `INSTRUCTION_COLUMN` | Optional[str] | `None` | Имя колонки с инструкцией |
| `RESPONSE_COLUMN` | Optional[str] | `None` | Имя колонки с ответом |
| `MAX_SEQ_LENGTH` | int | `512` | Максимальная длина последовательности |

**Форматы данных:**

1. **Простой текст (JSONL):**
```jsonl
{"text": "Ваш текст здесь..."}
{"text": "Ещё один пример..."}
```

2. **Instruction tuning (JSONL):**
```jsonl
{"instruction": "Напиши приветствие", "response": "Привет! Как дела?"}
{"instruction": "Объясни Python", "response": "Python — язык программирования..."}
```

### Optimization

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `USE_8BIT_ADAM` | bool | `false` | Использовать 8-bit AdamW |
| `FP16` | bool | `true` | Mixed precision fp16 |
| `BF16` | bool | `false` | Mixed precision bf16 (приоритетнее fp16) |
| `GRADIENT_CHECKPOINTING` | bool | `true` | Gradient checkpointing для экономии памяти |
| `MAX_MEMORY_MB` | Optional[int] | `None` | Максимальная память на GPU (MB) |

**Рекомендации:**
- BF16 предпочтительнее FP16 на Ampere GPU (A100, RTX 30xx+)
- Gradient checkpointing уменьшает память в ~2 раза, но замедляет на ~20%

### Logging and Monitoring

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `LOG_LEVEL` | str | `INFO` | Уровень логирования (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FILE` | Optional[str] | `None` | Путь к файлу логов |
| `WANDB_PROJECT` | Optional[str] | `llm-fine-tuning` | Имя проекта в Weights & Biases |
| `WANDB_API_KEY` | Optional[str] | `None` | API ключ для W&B |
| `TENSORBOARD_DIR` | Optional[str] | `None` | Директория для TensorBoard logs |

### Inference Configuration

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `INFERENCE_MODEL_PATH` | Optional[str] | `None` | Путь к дообученной модели |
| `INFERENCE_DEVICE` | str | `auto` | Устройство (`auto`, `cpu`, `cuda`, `cuda:0`) |
| `MAX_NEW_TOKENS` | int | `256` | Максимум новых токенов для генерации |
| `TEMPERATURE` | float | `0.7` | Temperature для sampling |
| `TOP_P` | float | `0.9` | Nucleus sampling top_p |
| `TOP_K` | int | `50` | Top-k sampling |
| `REPETITION_PENALTY` | float | `1.0` | Штраф за повторения (1.0 = отключено) |

### Advanced Configuration

| Переменная | Тип | По умолчанию | Описание |
|------------|-----|--------------|----------|
| `RANDOM_SEED` | int | `42` | Seed для воспроизводимости |
| `DATALOADER_NUM_WORKERS` | int | `4` | Количество workers для DataLoader |
| `REMOVE_UNUSED_COLUMNS` | bool | `true` | Удалять неиспользуемые колонки |
| `GROUP_BY_LENGTH` | bool | `true` | Группировать по длине для эффективности |
| `USE_PACKING` | bool | `false` | Использовать packing токенов |

## Примеры конфигураций

### Минимальная конфигурация

```bash
BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
TRAIN_DATA_PATH=./data/train.jsonl
OUTPUT_DIR=./outputs
```

### Для GPU с 8GB VRAM

```bash
BASE_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LORA_R=4
LORA_ALPHA=8
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=8
FP16=true
GRADIENT_CHECKPOINTING=true
MAX_SEQ_LENGTH=512
```

### Полная конфигурация для instruction tuning

```bash
# Model
BASE_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct

# LoRA
LORA_R=16
LORA_ALPHA=32
LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

# Training
NUM_EPOCHS=5
LEARNING_RATE=1e-4
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
MAX_SEQ_LENGTH=2048
BF16=true

# Data
TRAIN_DATA_PATH=./data/instructions_train.jsonl
VALIDATION_DATA_PATH=./data/instructions_val.jsonl
INSTRUCTION_COLUMN=instruction
RESPONSE_COLUMN=response

# Monitoring
WANDB_PROJECT=phi3-instruction-tuning
LOG_FILE=./logs/training.log
```

## Валидация

При старте приложения происходит валидация конфигурации:

1. **Обязательные поля** — проверка наличия `BASE_MODEL_NAME` и `TRAIN_DATA_PATH`
2. **Типы данных** — все значения приводятся к нужным типам
3. **Диапазоны** — например, `LORA_R` должен быть > 0
4. **Пути** — проверка существования `TRAIN_DATA_PATH`

Пример ошибки:
```
ValidationError: 2 validation errors for Settings
BASE_MODEL_NAME
  Field required [type=missing, input_value={'LORA_R': '8'}, input_type=dict]
TRAIN_DATA_PATH
  File not found: ./data/nonexistent.jsonl [type=value_error.path.not_exists]
```

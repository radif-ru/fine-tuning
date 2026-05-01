# Процесс обучения

## Полный workflow обучения

Проект поддерживает полный цикл fine-tuning LLM с LoRA:

### 1. Подготовка данных

Генерация синтетического датасета или подготовка собственных данных:

```bash
# Генерация синтетического датасета (Alpaca формат)
python -m app generate-dataset \
  --type synthetic \
  --output data/synthetic_train.jsonl \
  --count 100 \
  --topics programming science \
  --seed 42
```

Формат данных: JSONL с полями `instruction`, `input`, `output` (Alpaca формат).

Пример формата данных:
```jsonl
{"instruction": "Напиши функцию на Python для сортировки списка", "input": "", "output": "Вот пример функции..."}
{"instruction": "Объясни концепцию рекурсии", "input": "", "output": "Рекурсия — это..."}
```

### 2. Настройка конфигурации

Используйте готовую конфигурацию или создайте свою:

```bash
# Использование готовой конфигурации для TinyLlama
cp configs/tinyllama_lora.env .env

# Или создание своей конфигурации
cp .env.example .env
# Отредактируйте .env с нужными параметрами
```

Ключевые параметры в `configs/tinyllama_lora.env`:
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

### 3. Запуск обучения

```bash
# Обучение с конфигурационным файлом
python -m app --config configs/tinyllama_lora.env train \
  --data-path data/synthetic_train.jsonl

# С переопределением параметров
python -m app --config configs/tinyllama_lora.env train \
  --data-path data/synthetic_train.jsonl \
  --epochs 5 \
  --batch-size 2

# Возобновление из чекпоинта
python -m app --config configs/tinyllama_lora.env train \
  --resume-from-checkpoint ./outputs/tinyllama/checkpoint-500
```

### 4. Тестирование модели

```bash
# Интерактивный режим
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --interactive

# Одиночный промпт
python -m app inference \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --prompt "Напиши функцию на Python для сортировки списка"
```

### 5. Сборка финальной модели

```bash
# Merge LoRA адаптера с базовой моделью
python -m app export \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path outputs/tinyllama/final \
  --output-path ./merged-model
```

## Детальный процесс обучения

## Обзор

Тренировочный цикл состоит из следующих этапов:

1. **Подготовка** — загрузка конфигурации, инициализация логирования
2. **Загрузка модели** — базовая модель из HuggingFace Hub
3. **Применение LoRA** — конфигурация и применение адаптера
4. **Подготовка данных** — загрузка, форматирование, токенизация
5. **Обучение** — тренировочный цикл с логированием метрик
6. **Сохранение** — финальный адаптер и результаты

## Детали реализации

### 1. Инициализация

```python
from app.core.config import Settings
from app.core.logging_config import setup_logging

settings = Settings()
logger = setup_logging(
    level=settings.LOG_LEVEL,
    log_file=settings.LOG_FILE
)
logger.info("Инициализация обучения начата")
```

### 2. Загрузка модели

```python
from app.models.base import BaseModelLoader

loader = BaseModelLoader()
model, tokenizer = loader.load(
    model_name=settings.BASE_MODEL_NAME,
    cache_dir=settings.HF_CACHE_DIR,
    trust_remote_code=settings.TRUST_REMOTE_CODE
)
logger.info(
    "Модель загружена | model=%s params=%d",
    settings.BASE_MODEL_NAME,
    model.num_parameters()
)
```

**Особенности:**
- Автоматическое определение `torch_dtype` (float16 для GPU, float32 для CPU)
- `device_map="auto"` для multi-GPU
- `low_cpu_mem_usage=True` для экономии памяти

### 3. LoRA Configuration

```python
from app.models.lora import LoRAManager, LoRAConfig

lora_config = LoRAConfig(
    r=settings.LORA_R,
    alpha=settings.LORA_ALPHA,
    dropout=settings.LORA_DROPOUT,
    target_modules=settings.LORA_TARGET_MODULES,
    bias=settings.LORA_BIAS,
    task_type=settings.LORA_TASK_TYPE
)

lora_manager = LoRAManager()
model = lora_manager.apply(model, lora_config)
```

**Что происходит:**
- Создаётся `LoraConfig` из PEFT
- Замораживаются веса базовой модели
- Добавляются обучаемые low-rank матрицы
- Выводится количество обучаемых параметров

### 4. Подготовка данных

#### 4.1 Загрузка

```python
from app.data.loader import DataLoader

loader = DataLoader()
train_dataset = loader.load(settings.TRAIN_DATA_PATH)
```

Поддерживаемые форматы:
- **JSONL** — `{"text": "..."}` или `{"instruction": "...", "response": "..."}`
- **JSON** — массив объектов
- **CSV** — колонка `text` или настраиваемые
- **HuggingFace Dataset** — по имени из Hub

#### 4.2 Форматирование

Для instruction tuning:

```python
from app.data.formatter import InstructionFormatter

formatter = InstructionFormatter(
    instruction_column=settings.INSTRUCTION_COLUMN,
    response_column=settings.RESPONSE_COLUMN,
    text_column=settings.TEXT_COLUMN
)

train_dataset = formatter.format(train_dataset)
```

Стандартный формат (Alpaca-style):
```
### Instruction:
{instruction}

### Response:
{response}
```

#### 4.3 Токенизация

```python
from app.data.tokenizer import DatasetTokenizer

tokenizer_fn = DatasetTokenizer(
    tokenizer=tokenizer,
    max_length=settings.MAX_SEQ_LENGTH,
    truncation=True
)

tokenized_dataset = train_dataset.map(
    tokenizer_fn,
    batched=True,
    remove_columns=train_dataset.column_names
)
```

### 5. Обучение

```python
from app.training.trainer import LoRATrainer
from app.training.config import TrainingConfig

training_config = TrainingConfig(
    output_dir=settings.OUTPUT_DIR,
    num_epochs=settings.NUM_EPOCHS,
    batch_size=settings.PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
    learning_rate=settings.LEARNING_RATE,
    warmup_ratio=settings.WARMUP_RATIO,
    weight_decay=settings.WEIGHT_DECAY,
    max_grad_norm=settings.MAX_GRAD_NORM,
    logging_steps=settings.LOGGING_STEPS,
    save_steps=settings.SAVE_STEPS,
    eval_strategy=settings.EVAL_STRATEGY,
    eval_steps=settings.EVAL_STEPS,
    load_best_model_at_end=settings.LOAD_BEST_MODEL_AT_END,
    save_total_limit=settings.SAVE_TOTAL_LIMIT,
    fp16=settings.FP16,
    bf16=settings.BF16,
    gradient_checkpointing=settings.GRADIENT_CHECKPOINTING
)

trainer = LoRATrainer(
    model=model,
    tokenizer=tokenizer,
    config=training_config
)

trainer.train(
    train_dataset=tokenized_dataset,
    eval_dataset=val_dataset if settings.VALIDATION_DATA_PATH else None
)
```

**TrainingArguments включает:**
- Optimizer: AdamW
- Scheduler: cosine with warmup
- Mixed precision (fp16/bf16)
- Gradient accumulation
- Checkpointing
- Logging

### 6. Сохранение

```python
# Сохранение LoRA адаптера
lora_manager.save_adapter(model, f"{settings.CHECKPOINT_DIR}/final")

# Сохранение токенизатора
tokenizer.save_pretrained(f"{settings.OUTPUT_DIR}/tokenizer")

logger.info("Обучение завершено | output_dir=%s", settings.OUTPUT_DIR)
```

## Метрики и логирование

### Логирование в консоль и файл

```
2024-01-15 10:30:45 [INFO] [training.trainer] Начало обучения | epochs=3 lr=2e-04
2024-01-15 10:30:55 [INFO] [training.trainer] Step 10 | loss=2.341 lr=1.8e-04
2024-01-15 10:31:05 [INFO] [training.trainer] Step 20 | loss=1.982 lr=2.0e-04
2024-01-15 10:31:15 [INFO] [training.trainer] Checkpoint saved | step=100 path=./checkpoints/checkpoint-100
```

### Weights & Biases

```python
from transformers import TrainerCallback

class WandbCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)
```

### TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir=settings.TENSORBOARD_DIR)
writer.add_scalar("Loss/train", loss, step)
writer.add_scalar("Learning_Rate", lr, step)
```

## Callbacks

### Стандартные callbacks

1. **LoggingCallback** — логирование в консоль/файл
2. **WandbCallback** — логирование в W&B
3. **CheckpointCallback** — сохранение чекпоинтов
4. **EarlyStoppingCallback** — ранняя остановка (при наличии validation)

### Пользовательский callback

```python
from transformers import TrainerCallback

class CustomCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info("Эпоха %d началась", state.epoch)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info("Эпоха %d завершена", state.epoch)
```

## Resume from checkpoint

```python
# Автоматически определяется из OUTPUT_DIR
last_checkpoint = get_last_checkpoint(settings.OUTPUT_DIR)

if last_checkpoint:
    logger.info("Возобновление с чекпоинта | checkpoint=%s", last_checkpoint)
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()
```

## Оптимизации памяти

### Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
gradient_checkpointing_kwargs={"use_reentrant": False}
```

Снижает потребление памяти примерно в 2 раза за счёт пересчёта активаций во время backward pass.

### Gradient Accumulation

При batch_size=2 и accumulation_steps=8 эффективный batch = 16 без увеличения памяти.

### Mixed Precision

```python
# FP16 (для GPU с compute capability >= 7.0)
FP16=true

# BF16 (для Ampere GPU: A100, RTX 30xx+)
BF16=true  # имеет приоритет над FP16
```

## Best Practices

1. **Начинайте с малого rank** (4-8) и увеличивайте при необходимости
2. **Используйте validation set** для контроля переобучения
3. **Сохраняйте чекпоинты чаще** при долгих тренировках
4. **Мониторьте learning rate** через warmup
5. **Логируйте в W&B** для отслеживания экспериментов

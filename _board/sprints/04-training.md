# Спринт 04. Training

- **Источник:** Дорожная карта §Этап 4
- **Ветка:** `feature/04-training`
- **Открыт:** 2026-05-01
- **Закрыт:** —
- **Статус:** Backlog

## 1. Цель спринта

Реализовать тренировочный цикл с поддержкой LoRA, checkpointing, логированием, resume from checkpoint.

## 2. Скоуп

### В скоупе

- TrainingArguments wrapper с валидацией
- LoRA Trainer на базе transformers.Trainer
- Callbacks (logging, checkpointing, wandb)
- Resume from checkpoint
- CLI команда train
- Поддержка learning rate schedulers
- Gradient accumulation
- Mixed precision (fp16/bf16)

### Вне скоупа

- Distributed training (multi-GPU) — в Future
- DeepSpeed integration

## 3. Acceptance Criteria

- [ ] Тренировка на GPT2 проходит до конца
- [ ] LoRA обучает только adapter параметры
- [ ] Чекпоинты сохраняются каждые N steps
- [ ] Resume from checkpoint работает
- [ ] WandB логирование работает (опционально)
- [ ] CLI train команда работает
- [ ] `pytest tests/unit/training/` проходит

## 4. Этап 1. Training Configuration

### Задача 1.1. Training Config (`app/training/config.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** M

#### Описание

```python
from dataclasses import dataclass
from transformers import TrainingArguments

@dataclass
class LoRATrainingConfig:
    # Training params
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    max_steps: int = -1  # -1 = use epochs
    
    # Optimizer
    optim: str = "adamw_torch"  # adamw_torch, adamw_bnb_8bit
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Scheduler
    lr_scheduler_type: str = "linear"  # linear, cosine, cosine_with_restarts
    
    # Memory optimization
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    
    # Logging & Saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Resume
    resume_from_checkpoint: str = None
    
    def to_hf_training_arguments(self, output_dir: str) -> TrainingArguments
```

#### Definition of Done

- [ ] Все параметры сверху реализованы
- [ ] Валидация значений (learning_rate > 0, batch_size > 0)
- [ ] Конвертация в HF TrainingArguments
- [ ] Загрузка из .env через Settings
- [ ] Тесты: валидация, конвертация

---

## 5. Этап 2. LoRA Trainer

### Задача 2.1. LoRA Trainer (`app/training/trainer.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** L
- **Зависит от:** Задача 1.1, Спринт 02, Спринт 03

#### Описание

```python
from transformers import Trainer

class LoRATrainer:
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        config: LoRATrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset = None
    )
    
    def train(self, resume_from_checkpoint: str = None) -> None
    def save_model(self, output_dir: str) -> None
    def get_trainable_params_count(self) -> int
```

#### Definition of Done

- [ ] Интеграция с transformers.Trainer
- [ ] Подсчёт и логирование trainable parameters
- [ ] Поддержка eval_dataset
- [ ] Checkpointing каждые save_steps
- [ ] Resume from checkpoint работает
- [ ] Тесты: обучение, сохранение, resume

---

## 6. Этап 3. Callbacks

### Задача 3.1. Logging Callback (`app/training/callbacks.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S

#### Описание

```python
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs)
    def on_epoch_end(self, args, state, control, **kwargs)
    def on_save(self, args, state, control, **kwargs)
```

#### Definition of Done

- [ ] Логирование loss/learning rate каждые N steps
- [ ] Логирование в файл через project logger
- [ ] Прогресс bar через tqdm
- [ ] Тесты: callback вызывается

---

### Задача 3.2. WandB Callback (`app/training/callbacks.py`)

- **Статус:** ToDo
- **Приоритет:** medium
- **Объём:** S

#### Описание

```python
class WandBCallback(TrainerCallback):
    def __init__(self, project: str, name: str = None, config: dict = None)
    def on_train_begin(self, args, state, control, **kwargs)
    def on_log(self, args, state, control, logs=None, **kwargs)
```

#### Definition of Done

- [ ] Интеграция с wandb (опционально, только если установлен)
- [ ] Логирование метрик
- [ ] Логирование конфига
- [ ] Тесты: инициализация, логирование

---

## 7. Этап 4. CLI Training Command

### Задача 4.1. Train Command (`app/training/cli.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** M
- **Зависит от:** Задачи 1.1, 2.1, 3.1

#### Описание

```bash
python -m app train \
  --config configs/tinyllama_lora.env \
  --data-path data/train.jsonl \
  --output-dir outputs/run-001 \
  --resume-from-checkpoint outputs/run-001/checkpoint-500  # optional
```

```python
def train_command(args):
    # 1. Load config
    # 2. Load model + LoRA
    # 3. Load tokenizer
    # 4. Load and process dataset
    # 5. Create trainer
    # 6. Train
    # 7. Save final model
```

#### Definition of Done

- [ ] Аргументы: --config, --data-path, --output-dir, --resume-from-checkpoint
- [ ] Загрузка всего pipeline
- [ ] Обработка ошибок с понятными сообщениями
- [ ] Финальное сохранение адаптера
- [ ] Тесты: интеграционные тесты

---

## 8. Риски и смягчение

| # | Риск | Смягчение |
|---|------|-----------|
| 1 | OOM (Out of Memory) при обучении | Gradient accumulation, fp16/bf16, smaller batch |
| 2 | Resume from checkpoint ломается | Сохранять optimizer state, rng state |
| 3 | WandB недоступен (offline) | Локальное логирование в файл, опциональная интеграция |

---

## 9. Сводная таблица задач спринта

| # | Задача | Приоритет | Объём | Статус | Зависит от |
|---|--------|:---------:|:-----:|:------:|:----------:|
| 1.1 | Training Config | critical | M | ToDo | — |
| 2.1 | LoRA Trainer | critical | L | ToDo | 1.1 |
| 3.1 | Logging Callback | high | S | ToDo | 2.1 |
| 3.2 | WandB Callback | medium | S | ToDo | 2.1 |
| 4.1 | Train Command | high | M | ToDo | 1.1, 2.1, 3.1 |

## 10. История

- **YYYY-MM-DD** — спринт открыт

> **Примечание:** Для работы требуется `pip install wandb` для W&B интеграции (опционально)

# Архитектура

## Общая схема

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLI Interface                                   │
│                    (train / inference / evaluate)                           │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Core / Orchestrator                               │
│                     (инициализация, DI, lifecycle)                            │
└─────────────────────────┬───────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          v               v               v
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
│  Model Manager  │ │ Data Manager │ │   Trainer    │
│  ├─ Base model  │ │ ├─ Loader    │ │ ├─ LoRA      │
│  ├─ LoRA config │ │ ├─ Formatter │ │ ├─ Callbacks │
│  └─ Adapter     │ │ └─ Tokenizer │ │ └─ Metrics   │
└────────┬────────┘ └──────┬───────┘ └──────┬───────┘
         │                   │                │
         └───────────────────┴────────────────┘
                             │
                             v
                  ┌──────────────────┐
                  │  Training Loop   │
                  │ (transformers    │
                  │    Trainer)      │
                  └──────────────────┘
```

## Компоненты

### 1. Core (`app/core/`)

#### 1.1 Configuration (`config.py`)

Класс `Settings` на `pydantic-settings`. Загружает все параметры из `.env`:

```python
class Settings(BaseSettings):
    base_model_name: str
    lora_r: int = 8
    lora_alpha: int = 16
    # ...
```

**Принципы:**
- Валидация типов на старте
- Значения по умолчанию
- Понятные error messages

#### 1.2 Logging (`logging_config.py`)

Централизованная конфигурация логирования:

```python
def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    ...
```

**Формат логов:**
```
2024-01-15 10:30:45 [INFO] [training.trainer] Training started | epoch=1/3
```

### 2. Models (`app/models/`)

#### 2.1 Base Model (`base.py`)

Загрузка базовой модели и токенизатора.

**Интерфейс:**
```python
class BaseModelLoader:
    def load(self, model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        ...
```

#### 2.2 LoRA Adapter (`lora.py`)

Конфигурация и применение LoRA.

**Интерфейс:**
```python
class LoRAConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: list[str]

class LoRAManager:
    def apply(self, model: PreTrainedModel, config: LoRAConfig) -> PeftModel:
        ...
    
    def save_adapter(self, model: PeftModel, path: str) -> None:
        ...
    
    def load_adapter(self, model: PreTrainedModel, path: str) -> PeftModel:
        ...
```

#### 2.3 Model Registry (`registry.py`)

Реестр поддерживаемых моделей и их особенностей.

### 3. Data (`app/data/`)

#### 3.1 Data Loader (`loader.py`)

Загрузка данных из различных источников.

**Интерфейс:**
```python
class DataLoader:
    def load(self, path: str) -> Dataset:
        ...

    def validate(self, dataset: Dataset) -> bool:
        ...
```

**Поддерживаемые форматы:**
- JSONL: `{"text": "..."}` или `{"instruction": "...", "response": "..."}`
- CSV: колонка `text` или настраиваемые
- HuggingFace datasets

#### 3.2 Data Formatter (`formatter.py`)

Форматирование данных в нужный формат для обучения.

**Интерфейс:**
```python
class DataFormatter:
    def format_instruction(
        self,
        instruction: str,
        response: str,
        system_prompt: Optional[str] = None
    ) -> str:
        ...
```

#### 3.3 Tokenizer (`tokenizer.py`)

Обёртка над токенизатором с поддержкой:
- Truncation
- Padding
- Batch encoding

#### 3.4 Data Generators (`generators/`)

Генераторы синтетических датасетов для обучения.

**Базовый класс (`generators/base.py`):**
```python
class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, count: int) -> Iterator[Dict[str, Any]]:
        """Генерирует указанное количество примеров"""
        pass

    def save(self, path: str, count: int, format: str = "jsonl") -> None:
        """Сохраняет сгенерированные данные в файл"""
        pass
```

**Синтетический генератор (`generators/synthetic.py`):**
```python
class SyntheticGenerator(BaseGenerator):
    """Генерирует синтетические инструкции и ответы для fine-tuning"""

    def __init__(self, topics: list[str] | None = None, seed: int | None = None):
        self.topics = topics or ["programming", "science", "general"]
```

**Поддерживаемые темы:**
- `programming` — программирование, алгоритмы, концепции
- `science` — научные концепции, явления, процессы
- `general` — общие вопросы, эссе, обзоры

**QA генератор (`generators/qa.py`):**
```python
class QAGenerator(BaseGenerator):
    """Генерирует вопросы и ответы для обучения модели"""

    def __init__(self, categories: list[str] | None = None, seed: int | None = None):
        self.categories = categories or ["general", "technical", "practical"]
```

**Поддерживаемые категории QA:**
- `general` — общие вопросы о концепциях и явлениях
- `technical` — технические вопросы о коде, алгоритмах, инструментах
- `practical` — практические вопросы о том, как что-то сделать

**CLI команда:**
```bash
# Синтетический генератор
python -m app generate-dataset --type synthetic --output data/synthetic.jsonl --count 100

# QA генератор
python -m app generate-dataset --type qa --output data/qa.jsonl --count 100 --categories technical
```

### 4. Training (`app/training/`)

#### 4.1 Trainer (`trainer.py`)

Основной тренировочный цикл на базе `transformers.Trainer`.

**Интерфейс:**
```python
class LoRATrainer:
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig
    ):
        ...
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> None:
        ...
```

#### 4.2 Training Configuration (`config.py`)

Конфигурация тренировки:

```python
@dataclass
class TrainingConfig:
    output_dir: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    # ...
```

#### 4.3 Callbacks (`callbacks.py`)

- LoggingCallback — логирование метрик
- WandbCallback — интеграция с W&B
- CheckpointCallback — сохранение чекпоинтов

### 5. Inference (`app/inference/`)

#### 5.1 Inference Engine (`engine.py`)

```python
class InferenceEngine:
    def __init__(self, model_path: str, device: str = "auto"):
        ...
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        ...
    
    def generate_batch(self, prompts: list[str]) -> list[str]:
        ...
```

### 6. Utils (`app/utils/`)

#### 6.1 Device Utils (`device.py`)

Определение доступных устройств (cuda/cpu/mps).

#### 6.2 Memory Utils (`memory.py`)

Мониторинг и управление памятью GPU.

#### 6.3 File Utils (`file.py`)

Работа с файлами и директориями.

## Поток данных

### Training Flow

```
1. CLI вызывает core.orchestrator.setup_training()
   ↓
2. Загрузка конфигурации из .env (Settings)
   ↓
3. Инициализация логирования
   ↓
4. Загрузка базовой модели (BaseModelLoader)
   ↓
5. Применение LoRA (LoRAManager.apply)
   ↓
6. Загрузка и форматирование данных (DataLoader, DataFormatter)
   ↓
7. Токенизация
   ↓
8. Инициализация Trainer
   ↓
9. Обучение (trainer.train)
   ↓
10. Сохранение адаптера
```

### Inference Flow

```
1. CLI вызывает core.orchestrator.setup_inference()
   ↓
2. Загрузка конфигурации
   ↓
3. Загрузка базовой модели
   ↓
4. Загрузка LoRA адаптера
   ↓
5. Инициализация InferenceEngine
   ↓
6. Генерация ответа
```

## Принципы архитектуры

### 1. Separation of Concerns

Каждый модуль отвечает за одну задачу:
- `models/` — только работа с моделями
- `data/` — только работа с данными
- `training/` — только обучение
- `inference/` — только инференс

### 2. Dependency Injection

Все зависимости передаются через конструкторы, никаких global singletons.

### 3. Configuration as Code

Вся конфигурация типизирована и валидируется на старте.

### 4. Interface-based Design

Ключевые компоненты имеют явные интерфейсы для возможности mocking в тестах.

### 5. Error Handling

- Ожидаемые ошибки (OOM, файл не найден) — обрабатываются с понятными сообщениями
- Неожиданные ошибки — логируются с полным stacktrace

## Расширяемость

### Добавление нового формата данных

1. Создать класс в `app/data/loaders/` наследующий `BaseLoader`
2. Реализовать методы `can_load(path)` и `load(path)`
3. Зарегистрировать в `DataLoaderRegistry`

### Добавление поддержки новой модели

1. Добавить entry в `app/models/registry.py` с особенностями модели
2. При необходимости — создать custom formatter в `app/data/formatters/`

### Добавление нового callback

1. Создать класс наследующий `TrainerCallback`
2. Зарегистрировать в `CallbackManager`

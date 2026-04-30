# Спринт 02. Model Management

- **Источник:** Дорожная карта §Этап 2
- **Ветка:** `feature/02-model-management`
- **Открыт:** —
- **Закрыт:** —
- **Статус:** Active

## 1. Цель спринта

Реализовать загрузку моделей из HuggingFace Hub и локальных путей, интеграцию LoRA через PEFT, сохранение и загрузку адаптеров.

## 2. Скоуп и non-goals

### В скоупе

- Base Model Loader с поддержкой HF Hub и локальных путей
- LoRA Manager (конфигурация, применение, сохранение, загрузка)
- Model Registry с поддержкой TinyLlama, Phi-3, GPT2
- Auto device map (CPU/CUDA/MPS)
- Quantization (8-bit через bitsandbytes)

### Вне скоупа

- 4-bit quantization (QLoRA) — в Future Enhancements
- Другие PEFT методы (IA³, Prompt Tuning)
- Multi-GPU загрузка

## 3. Acceptance Criteria

- [ ] Загрузка GPT2/TinyLlama/Phi-3 из HF Hub работает
- [ ] Загрузка из локального пути работает
- [ ] LoRA применяется корректно через PEFT
- [ ] Адаптер сохраняется и загружается
- [ ] 8-bit quantization работает
- [ ] `pytest tests/unit/models/` проходит

## 4. Решения по архитектуре

| Решение | Обоснование |
|---------|-------------|
| **PEFT для LoRA** | Стандарт де-факто, совместим с HF ecosystem |
| **8-bit через bitsandbytes** | Экономия памяти, стабильная работа |
| **Model Registry** | Централизованная конфигурация моделей |
| **Auto device map** | Автоматический выбор оптимального устройства |

## 5. Этап 1. Base Model Loader

### Задача 1.1. Base Model Loader (`app/models/base.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** M
- **Зависит от:** Спринт 01
- **Связанные документы:** `_docs/architecture.md` §2

#### Описание

Реализовать загрузку моделей:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaseModelLoader:
    def load(self, model_name: str, load_in_8bit: bool = False) -> tuple[PreTrainedModel, PreTrainedTokenizer]
    def load_from_local(self, path: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]
    def get_model_info(self) -> ModelInfo
```

#### Definition of Done

- [ ] Загрузка из HF Hub работает (GPT2, TinyLlama)
- [ ] Загрузка из локального пути работает
- [ ] 8-bit mode работает при `load_in_8bit=True`
- [ ] Auto tokenizer загружается
- [ ] Device map применяется автоматически
- [ ] Тесты: загрузка разных моделей, ошибки при невалидных именах

---

## 6. Этап 2. LoRA Manager

### Задача 2.1. LoRA Config (`app/models/lora_config.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** S
- **Зависит от:** —
- **Связанные документы:** `_docs/architecture.md` §2.2

#### Описание

Dataclass для LoRA конфигурации:

```python
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 8              # Rank
    lora_alpha: int = 16    # Scaling
    lora_dropout: float = 0.05
    target_modules: list[str] = None  # ["q_proj", "v_proj"] и т.д.
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
```

#### Definition of Done

- [ ] Dataclass создан с валидацией полей
- [ ] Метод `from_env()` для загрузки из Settings
- [ ] Метод `to_peft_config()` → `LoraConfig`
- [ ] Тесты: валидация, конвертация в PEFT

---

### Задача 2.2. LoRA Manager (`app/models/lora.py`)

- **Статус:** ToDo
- **Приоритет:** critical
- **Объём:** M
- **Зависит от:** Задача 1.1, 2.1
- **Связанные документы:** `_docs/architecture.md` §2.2

#### Описание

```python
from peft import get_peft_model, PeftModel

class LoRAManager:
    def apply_lora(self, model: PreTrainedModel, config: LoRAConfig) -> PeftModel
    def save_adapter(self, model: PeftModel, path: str)
    def load_adapter(self, model: PreTrainedModel, path: str) -> PeftModel
    def merge_and_unload(self, model: PeftModel) -> PreTrainedModel
```

#### Definition of Done

- [ ] Применение LoRA через `get_peft_model()`
- [ ] Сохранение адаптера в указанный путь
- [ ] Загрузка адаптера к базовой модели
- [ ] Мerge and unload для inference
- [ ] Подсчёт trainable parameters
- [ ] Тесты: apply, save, load, merge

---

## 7. Этап 3. Model Registry

### Задача 3.1. Model Registry (`app/models/registry.py`)

- **Статус:** ToDo
- **Приоритет:** high
- **Объём:** S
- **Зависит от:** Задача 1.1
- **Связанные документы:** `_docs/architecture.md` §2.1

#### Описание

```python
class ModelRegistry:
    def register(self, name: str, config: ModelConfig)
    def get_config(self, name: str) -> ModelConfig
    def list_models() -> list[str]
    
# Предустановленные конфиги:
# - tinyllama-1.1b
# - phi-3-mini  
# - gpt2
```

#### Definition of Done

- [ ] Регистрация моделей с конфигами
- [ ] TinyLlama, Phi-3, GPT2 предустановлены
- [ ] Специфичные настройки (context length, target_modules для LoRA)
- [ ] Тесты: регистрация, получение, листинг

---

## 8. Сводная таблица задач спринта

| # | Задача | Приоритет | Объём | Статус | Зависит от |
|---|--------|:---------:|:-----:|:------:|:----------:|
| 1.1 | Base Model Loader | critical | M | ToDo | — |
| 2.1 | LoRA Config | critical | S | ToDo | — |
| 2.2 | LoRA Manager | critical | M | ToDo | 1.1, 2.1 |
| 3.1 | Model Registry | high | S | ToDo | 1.1 |

## 9. История

- **YYYY-MM-DD** — спринт открыт

> **Примечание:** 8-bit quantization через bitsandbytes требует `pip install bitsandbytes`

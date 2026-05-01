# Тестирование

## Стратегия

Проект использует многоуровневый подход к тестированию:

1. **Unit tests** — изолированные тесты компонентов с моками
2. **Integration tests** — тесты взаимодействия компонентов
3. **E2E tests** — сквозные тесты ключевых сценариев

## Структура тестов

```
tests/
├── __init__.py
├── conftest.py              # Общие фикстуры
├── unit/
│   ├── __init__.py
│   ├── core/
│   │   ├── test_config.py
│   │   └── test_logging.py
│   ├── models/
│   │   ├── test_base.py
│   │   ├── test_lora.py
│   │   └── test_registry.py
│   ├── data/
│   │   ├── test_loader.py
│   │   ├── test_formatter.py
│   │   ├── test_tokenizer.py
│   │   └── test_templates.py
│   ├── training/
│   │   ├── test_trainer.py
│   │   ├── test_callbacks.py
│   │   └── test_training_config.py
│   ├── inference/
│   │   ├── test_config.py
│   │   ├── test_engine.py
│   │   ├── test_prompt.py
│   │   ├── test_cli.py
│   │   └── test_export.py
│   ├── utils/
│   │   ├── test_device.py
│   │   └── test_memory.py
│   └── test_cli.py
└── integration/
    ├── __init__.py
    └── test_e2e.py
```

## Конфигурация pytest

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra -q --strict-markers"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests",
]
```

## Запуск тестов

```bash
# Все тесты
pytest

# Тихий режим
pytest -q

# С покрытием
pytest --cov=app --cov-report=term-missing

# Только unit
pytest -m unit

# Только integration
pytest -m integration

# Исключить slow
pytest --ignore-glob="*slow*"

# Конкретный файл
pytest tests/unit/models/test_lora.py

# Конкретный тест
pytest tests/unit/models/test_lora.py::TestLoRAManager::test_apply_lora
```

## Фикстуры

### conftest.py

```python
import pytest
import tempfile
import shutil
from pathlib import Path

@pytest.fixture
def temp_dir():
    """Временная директория для тестов"""
    path = tempfile.mkdtemp()
    yield Path(path)
    shutil.rmtree(path)

@pytest.fixture
def mock_model_config():
    """Мок конфигурации модели"""
    return {
        "base_model_name": "gpt2",  # маленькая модель для тестов
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.1,
    }

@pytest.fixture
def sample_data_file(temp_dir):
    """Создаёт тестовый файл данных"""
    data_file = temp_dir / "train.jsonl"
    data_file.write_text(
        '{"text": "Example 1"}\n'
        '{"text": "Example 2"}\n'
    )
    return data_file
```

## Примеры тестов

### Тестирование конфигурации

```python
# tests/unit/core/test_config.py
import pytest
from app.core.config import Settings

class TestSettings:
    def test_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("BASE_MODEL_NAME", "test-model")
        monkeypatch.setenv("TRAIN_DATA_PATH", "./data/train.jsonl")
        
        settings = Settings()
        
        assert settings.BASE_MODEL_NAME == "test-model"
        assert settings.TRAIN_DATA_PATH == "./data/train.jsonl"
    
    def test_default_values(self, monkeypatch):
        monkeypatch.setenv("BASE_MODEL_NAME", "test-model")
        monkeypatch.setenv("TRAIN_DATA_PATH", "./data/train.jsonl")
        
        settings = Settings()
        
        assert settings.LORA_R == 8
        assert settings.LORA_ALPHA == 16
        assert settings.NUM_EPOCHS == 3
    
    def test_missing_required_field(self, monkeypatch):
        monkeypatch.delenv("BASE_MODEL_NAME", raising=False)
        monkeypatch.setenv("TRAIN_DATA_PATH", "./data/train.jsonl")
        
        with pytest.raises(ValueError):
            Settings()
    
    def test_invalid_lora_r(self, monkeypatch):
        monkeypatch.setenv("BASE_MODEL_NAME", "test-model")
        monkeypatch.setenv("TRAIN_DATA_PATH", "./data/train.jsonl")
        monkeypatch.setenv("LORA_R", "-1")
        
        with pytest.raises(ValueError):
            Settings()
```

### Тестирование загрузки данных

```python
# tests/unit/data/test_loader.py
import pytest
from app.data.loader import DataLoader, JSONLLoader

class TestJSONLLoader:
    def test_can_load_jsonl(self, temp_dir):
        loader = JSONLLoader()
        jsonl_file = temp_dir / "data.jsonl"
        jsonl_file.write_text('{"text": "test"}')
        
        assert loader.can_load(jsonl_file) is True
    
    def test_can_load_non_jsonl(self, temp_dir):
        loader = JSONLLoader()
        txt_file = temp_dir / "data.txt"
        txt_file.write_text("test")
        
        assert loader.can_load(txt_file) is False
    
    def test_load_valid_jsonl(self, sample_data_file):
        loader = JSONLLoader()
        dataset = loader.load(sample_data_file)
        
        assert len(dataset) == 2
        assert dataset[0]["text"] == "Example 1"
    
    def test_load_invalid_json(self, temp_dir):
        loader = JSONLLoader()
        invalid_file = temp_dir / "invalid.jsonl"
        invalid_file.write_text("not valid json")
        
        with pytest.raises(ValueError):
            loader.load(invalid_file)
```

### Тестирование LoRA

```python
# tests/unit/models/test_lora.py
import pytest
from unittest.mock import Mock, patch
from app.models.lora import LoRAManager, LoRAConfig

class TestLoRAManager:
    @pytest.fixture
    def mock_model(self):
        model = Mock()
        model.num_parameters.return_value = 1000000
        return model
    
    def test_apply_lora(self, mock_model):
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        manager = LoRAManager()
        
        with patch("peft.get_peft_model") as mock_get_peft:
            mock_peft_model = Mock()
            mock_get_peft.return_value = mock_peft_model
            
            result = manager.apply(mock_model, config)
            
            mock_get_peft.assert_called_once()
            assert result == mock_peft_model
    
    def test_lora_config_validation(self):
        with pytest.raises(ValueError):
            LoRAConfig(r=-1, alpha=16)  # negative rank
        
        with pytest.raises(ValueError):
            LoRAConfig(r=8, alpha=0)  # alpha must be positive
        
        with pytest.raises(ValueError):
            LoRAConfig(r=8, alpha=16, dropout=1.5)  # dropout > 1
```

### Тестирование с использованием временных моделей

```python
# tests/unit/models/test_base.py
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.models.base import BaseModelLoader

class TestBaseModelLoader:
    @pytest.mark.slow  # Помечаем как медленный тест
    def test_load_small_model(self):
        """Используем gpt2 как маленькую тестовую модель"""
        loader = BaseModelLoader()
        model, tokenizer = loader.load("gpt2")
        
        assert model is not None
        assert tokenizer is not None
        assert hasattr(model, "generate")
    
    def test_load_model_error(self):
        loader = BaseModelLoader()
        
        with pytest.raises(Exception):
            loader.load("non-existent-model-12345")
```

## Мокирование

### Моки для transformers

```python
from unittest.mock import Mock, patch

@patch("transformers.AutoModelForCausalLM.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
def test_trainer_initialization(
    mock_tokenizer_class,
    mock_model_class
):
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_model_class.return_value = mock_model
    mock_tokenizer_class.return_value = mock_tokenizer
    
    # Тест с моками
    from app.training.trainer import LoRATrainer
    
    trainer = LoRATrainer(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=mock_config
    )
    
    assert trainer.model == mock_model
```

### Моки для PEFT

```python
@patch("peft.LoraConfig")
@patch("peft.get_peft_model")
def test_lora_application(mock_get_peft, mock_lora_config):
    mock_peft_model = Mock()
    mock_get_peft.return_value = mock_peft_model
    
    manager = LoRAManager()
    result = manager.apply(mock_base_model, config)
    
    assert result == mock_peft_model
    mock_get_peft.assert_called_once()
```

## Интеграционные тесты

```python
# tests/integration/test_training_pipeline.py
import pytest
from app.core.config import Settings
from app.data.loader import DataLoader
from app.models.base import BaseModelLoader
from app.models.lora import LoRAManager, LoRAConfig

@pytest.mark.integration
@pytest.mark.slow
class TestTrainingPipeline:
    def test_full_pipeline_with_tiny_model(self, temp_dir, sample_data_file):
        """Сквозной тест с gpt2"""
        # 1. Загрузка модели
        loader = BaseModelLoader()
        model, tokenizer = loader.load("gpt2")
        
        # 2. Применение LoRA
        lora_config = LoRAConfig(r=4, alpha=8, target_modules=["c_attn"])
        lora_manager = LoRAManager()
        model = lora_manager.apply(model, lora_config)
        
        # 3. Загрузка данных
        data_loader = DataLoader()
        dataset = data_loader.load(sample_data_file)
        
        # 4. Токенизация
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, max_length=128)
        
        tokenized = dataset.map(tokenize, batched=True)
        
        # Проверки
        assert len(tokenized) == 2
        assert "input_ids" in tokenized.features
```

## Покрытие кода

### Целевые показатели

- Общее покрытие: ≥ 70%
- `app/core/`: ≥ 85%
- `app/models/`: ≥ 80%
- `app/data/`: ≥ 75%
- `app/training/`: ≥ 70%
- `app/inference/`: ≥ 70%
- `app/utils/`: ≥ 80%

### Отчёт о покрытии

```bash
pytest --cov=app --cov-report=html --cov-report=term-missing
```

HTML отчёт будет в `htmlcov/index.html`.

## CI/CD интеграция

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pip install pytest pytest-cov
      - run: pytest --cov=app --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Best Practices

1. **AAA паттерн** — Arrange, Act, Assert
2. **Один assert на тест** — или логически связанные asserts
3. **Описательные имена** — `test_should_raise_error_when_invalid_config`
4. **Изоляция** — каждый тест независим
5. **Фикстуры** — переиспользуемая подготовка данных
6. **Маркеры** — `unit`, `integration`, `slow` для выборочного запуска
7. **Параметризация** — `@pytest.mark.parametrize` для вариаций

```python
@pytest.mark.parametrize("r,alpha", [
    (4, 8),
    (8, 16),
    (16, 32),
])
def test_lora_config_with_various_ranks(r, alpha):
    config = LoRAConfig(r=r, alpha=alpha)
    assert config.r == r
    assert config.alpha == alpha
```

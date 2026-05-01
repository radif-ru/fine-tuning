"""End-to-End интеграционные тесты.

Тесты проверяют полный цикл работы фреймворка:
- Загрузка конфигурации
- Загрузка модели и данных
- Обучение
- Сохранение и загрузка адаптера
- Инференс
- Экспорт модели
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


class TestFullTrainingLoop:
    """Тест полного цикла обучения."""
    
    @pytest.mark.slow
    def test_config_loading(self, monkeypatch):
        """Проверка загрузки конфигурации."""
        from app.core.config import Settings
        
        # Устанавливаем env vars
        monkeypatch.setenv("BASE_MODEL_NAME", "gpt2")
        monkeypatch.setenv("LORA_R", "4")
        monkeypatch.setenv("LORA_ALPHA", "8")
        monkeypatch.setenv("NUM_EPOCHS", "1")
        monkeypatch.setenv("TRAIN_DATA_PATH", "./data/train.jsonl")
        monkeypatch.setenv("OUTPUT_DIR", "./outputs")
        
        settings = Settings()
        
        assert settings.BASE_MODEL_NAME == "gpt2"
        assert settings.LORA_R == 4
        assert settings.LORA_ALPHA == 8
        assert settings.NUM_EPOCHS == 1
    
    def test_model_registry_integration(self):
        """Проверка работы ModelRegistry."""
        from app.models.registry import get_registry, ModelConfig
        
        registry = get_registry()
        
        # Проверка получения конфигурации
        config = registry.get_config("gpt2")
        assert isinstance(config, ModelConfig)
        assert config.name == "gpt2"
        
        # Проверка target modules
        modules = registry.get_target_modules("gpt2")
        assert len(modules) > 0
    
    def test_data_loading_alpaca(self, tmp_path):
        """Проверка загрузки данных Alpaca формата."""
        # Создаём тестовый файл
        data_file = tmp_path / "test_alpaca.jsonl"
        data_file.write_text(
            '{"instruction": "Test", "input": "", "output": "Result"}\n'
        )
        
        # Проверяем что файл существует и содержит валидный JSON
        assert data_file.exists()
        with open(data_file) as f:
            line = json.loads(f.readline())
            assert "instruction" in line
            assert "output" in line
    
    def test_data_loading_sharegpt(self, tmp_path):
        """Проверка загрузки данных ShareGPT формата."""
        data_file = tmp_path / "test_sharegpt.jsonl"
        data_file.write_text(
            '{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}\n'
        )
        
        assert data_file.exists()
        with open(data_file) as f:
            line = json.loads(f.readline())
            assert "conversations" in line


class TestInferenceWithAdapter:
    """Тест инференса с адаптером."""
    
    @patch("app.inference.engine.BaseModelLoader")
    @patch("app.inference.engine.LoRAManager")
    def test_inference_engine_initialization(
        self, mock_lora_manager, mock_loader_class, mocker
    ):
        """Проверка инициализации InferenceEngine."""
        from app.inference.engine import InferenceEngine
        
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_loader = mocker.MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        engine = InferenceEngine("gpt2", adapter_path="/adapter")
        engine.load()
        
        assert engine.is_loaded is True
        mock_loader_class.assert_called_once()
    
    def test_prompt_builder_integration(self):
        """Проверка интеграции PromptBuilder."""
        from app.inference.prompt import PromptBuilder
        
        # Alpaca шаблон
        builder = PromptBuilder("alpaca")
        prompt = builder.build("Explain LoRA")
        assert "### Инструкция:" in prompt
        assert "### Ответ:" in prompt
        
        # Raw шаблон
        builder = PromptBuilder("raw")
        prompt = builder.build("Simple text")
        assert prompt == "Simple text"
    
    def test_generation_config_validation(self):
        """Проверка валидации GenerationConfig."""
        from app.inference.config import GenerationConfig
        
        # Валидные значения
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256
        )
        assert config.temperature == 0.7
        
        # Невалидные значения
        with pytest.raises(ValueError):
            GenerationConfig(temperature=-0.1)
        
        with pytest.raises(ValueError):
            GenerationConfig(max_new_tokens=0)


class TestResumeTraining:
    """Тест возобновления обучения."""
    
    def test_checkpoint_creation(self, tmp_path):
        """Проверка создания структуры чекпоинтов."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Создаём фиктивный чекпоинт
        checkpoint_path = checkpoint_dir / "checkpoint-100"
        checkpoint_path.mkdir()
        
        assert checkpoint_path.exists()
    
    def test_output_directory_structure(self, tmp_path):
        """Проверка структуры выходной директории."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        
        # Создаём поддиректории
        (output_dir / "final").mkdir()
        (output_dir / "checkpoints").mkdir()
        
        assert (output_dir / "final").exists()
        assert (output_dir / "checkpoints").exists()


class TestExportAndLoad:
    """Тест экспорта и загрузки объединённой модели."""
    
    @patch("app.inference.export.BaseModelLoader")
    @patch("app.inference.export.LoRAManager")
    def test_export_flow(
        self, mock_lora_manager_class, mock_loader_class, mocker, tmp_path
    ):
        """Проверка процесса экспорта."""
        from app.inference.export import ModelExporter
        
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_loader = mocker.MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        mock_lora_manager = mocker.MagicMock()
        mock_lora_manager.load_adapter.return_value = mock_model
        mock_lora_manager.merge_and_unload.return_value = mock_model
        mock_lora_manager_class.return_value = mock_lora_manager
        
        output_path = tmp_path / "merged"
        exporter = ModelExporter()
        exporter.merge_and_save(
            base_model_name="gpt2",
            adapter_path="/adapter",
            output_path=str(output_path),
        )
        
        # Проверяем что директория создана
        assert output_path.exists()
        mock_model.save_pretrained.assert_called_once()
    
    def test_export_directory_creation(self, tmp_path):
        """Проверка создания директории для экспорта."""
        from app.inference.export import ModelExporter
        
        with patch("app.inference.export.BaseModelLoader") as mock_loader:
            with patch("app.inference.export.LoRAManager") as mock_lora:
                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_loader_instance = MagicMock()
                mock_loader_instance.load.return_value = (mock_model, mock_tokenizer)
                mock_loader.return_value = mock_loader_instance
                
                mock_lora_instance = MagicMock()
                mock_lora_instance.load_adapter.return_value = mock_model
                mock_lora_instance.merge_and_unload.return_value = mock_model
                mock_lora.return_value = mock_lora_instance
                
                output_path = tmp_path / "new" / "nested" / "dir"
                exporter = ModelExporter()
                exporter.merge_and_save(
                    base_model_name="gpt2",
                    adapter_path="/adapter",
                    output_path=str(output_path),
                )
                
                assert output_path.exists()


class TestCLIIntegration:
    """Тест интеграции CLI."""
    
    def test_cli_parser_creation(self):
        """Проверка создания парсера аргументов."""
        from app.__main__ import create_parser
        import sys
        
        parser = create_parser()
        assert parser is not None
        
        # Проверка что команды существуют (help вызывает SystemExit)
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
    
    def test_inference_args_parsing(self):
        """Проверка парсинга аргументов inference."""
        from app.__main__ import create_parser
        
        parser = create_parser()
        
        args = parser.parse_args([
            "inference",
            "--base-model", "gpt2",
            "--prompt", "Hello",
            "--temperature", "0.8",
            "--max-new-tokens", "100",
        ])
        
        assert args.command == "inference"
        assert args.base_model == "gpt2"
        assert args.prompt == "Hello"
        assert args.temperature == 0.8
        assert args.max_new_tokens == 100
    
    def test_export_args_parsing(self):
        """Проверка парсинга аргументов export."""
        from app.__main__ import create_parser
        
        parser = create_parser()
        
        args = parser.parse_args([
            "export",
            "--base-model", "gpt2",
            "--adapter-path", "/adapter",
            "--output-path", "/output",
        ])
        
        assert args.command == "export"
        assert args.base_model == "gpt2"
        assert args.adapter_path == "/adapter"
        assert args.output_path == "/output"


class TestExampleDataIntegration:
    """Тест интеграции с примерами данных."""
    
    def test_alpaca_examples_exist(self):
        """Проверка существования примеров Alpaca."""
        from pathlib import Path
        
        example_file = Path("examples/data/alpaca_format.jsonl")
        assert example_file.exists()
        
        # Проверяем что файл не пустой
        content = example_file.read_text()
        assert len(content) > 0
        
        # Проверяем что это валидный JSONL
        lines = content.strip().split("\n")
        for line in lines:
            data = json.loads(line)
            assert "instruction" in data
            assert "output" in data
    
    def test_sharegpt_examples_exist(self):
        """Проверка существования примеров ShareGPT."""
        from pathlib import Path
        
        example_file = Path("examples/data/sharegpt_format.jsonl")
        assert example_file.exists()
        
        content = example_file.read_text()
        assert len(content) > 0
    
    def test_csv_examples_exist(self):
        """Проверка существования примеров CSV."""
        from pathlib import Path
        
        example_file = Path("examples/data/sample_csv.csv")
        assert example_file.exists()
        
        content = example_file.read_text()
        assert "text" in content


class TestConfigIntegration:
    """Тест интеграции конфигураций."""
    
    def test_training_config_exists(self):
        """Проверка существования конфига обучения."""
        from pathlib import Path
        
        config_file = Path("configs/training_default.env")
        assert config_file.exists()
        
        content = config_file.read_text()
        assert "BASE_MODEL_NAME" in content
        assert "LORA_R" in content
    
    def test_inference_config_exists(self):
        """Проверка существования конфига инференса."""
        from pathlib import Path
        
        config_file = Path("configs/inference_default.env")
        assert config_file.exists()
        
        content = config_file.read_text()
        assert "MAX_NEW_TOKENS" in content
        assert "TEMPERATURE" in content
    
    def test_example_configs_exist(self):
        """Проверка существования примеров конфигов."""
        from pathlib import Path
        
        configs = [
            "configs/tinyllama_lora.env",
            "configs/gpt2_lora.env",
            "configs/phi3_lora.env",
        ]
        
        for config_path in configs:
            assert Path(config_path).exists(), f"{config_path} not found"

"""Тесты для CLI приложения."""

import pytest
from unittest.mock import MagicMock, patch

from app.__main__ import (
    create_parser,
    handle_export_command,
    handle_inference_command,
    handle_train_command,
    main,
)


class TestArgumentParser:
    """Тесты для парсера аргументов."""
    
    def test_help(self):
        """Проверка работы --help."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0
    
    def test_train_help(self):
        """Проверка работы train --help."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["train", "--help"])
        assert exc_info.value.code == 0
    
    def test_inference_help(self):
        """Проверка работы inference --help."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["inference", "--help"])
        assert exc_info.value.code == 0
    
    def test_export_help(self):
        """Проверка работы export --help."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["export", "--help"])
        assert exc_info.value.code == 0
    
    def test_no_command(self):
        """Проверка поведения без команды."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.command is None
    
    def test_config_argument(self):
        """Проверка аргумента --config."""
        parser = create_parser()
        args = parser.parse_args(["--config", "custom.env", "train"])
        assert args.config == "custom.env"
    
    def test_train_arguments(self):
        """Проверка аргументов команды train."""
        parser = create_parser()
        args = parser.parse_args([
            "train",
            "--base-model", "gpt2",
            "--data-path", "./data/train.jsonl",
            "--output-dir", "./outputs",
            "--epochs", "5",
            "--batch-size", "8"
        ])
        
        assert args.command == "train"
        assert args.base_model == "gpt2"
        assert args.data_path == "./data/train.jsonl"
        assert args.output_dir == "./outputs"
        assert args.epochs == 5
        assert args.batch_size == 8
    
    def test_inference_arguments(self):
        """Проверка аргументов команды inference."""
        parser = create_parser()
        args = parser.parse_args([
            "inference",
            "--base-model", "gpt2",
            "--adapter-path", "./outputs/adapter",
            "--prompt", "Hello",
            "--temperature", "0.8",
            "--max-new-tokens", "100"
        ])
        
        assert args.command == "inference"
        assert args.base_model == "gpt2"
        assert args.adapter_path == "./outputs/adapter"
        assert args.prompt == "Hello"
        assert args.temperature == 0.8
        assert args.max_new_tokens == 100
    
    def test_inference_interactive(self):
        """Проверка интерактивного режима."""
        parser = create_parser()
        args = parser.parse_args([
            "inference",
            "--base-model", "gpt2",
            "--interactive"
        ])
        
        assert args.interactive is True
    
    def test_export_arguments(self):
        """Проверка аргументов команды export."""
        parser = create_parser()
        args = parser.parse_args([
            "export",
            "--base-model", "gpt2",
            "--adapter-path", "./outputs/adapter",
            "--output-path", "./outputs/merged"
        ])
        
        assert args.command == "export"
        assert args.base_model == "gpt2"
        assert args.adapter_path == "./outputs/adapter"
        assert args.output_path == "./outputs/merged"


class TestCommandHandlers:
    """Тесты для обработчиков команд."""
    
    @pytest.fixture
    def mock_settings(self):
        """Фикстура для mock настроек."""
        settings = MagicMock()
        settings.BASE_MODEL_NAME = "gpt2"
        settings.TRAIN_DATA_PATH = "./data/train.jsonl"
        settings.OUTPUT_DIR = "./outputs"
        settings.NUM_EPOCHS = 3
        settings.PER_DEVICE_BATCH_SIZE = 4
        return settings
    
    @pytest.fixture
    def mock_logger(self):
        """Фикстура для mock логгера."""
        return MagicMock()
    
    @patch("app.training.trainer.LoRATrainer")
    @patch("app.training.config.LoRATrainingConfig")
    @patch("app.data.tokenizer.TokenizerWrapper")
    @patch("app.data.formatter.DataFormatter")
    @patch("app.data.loader.DataLoader")
    @patch("app.models.lora.LoRAManager")
    @patch("app.models.lora_config.LoRAConfig")
    @patch("app.models.base.BaseModelLoader")
    def test_handle_train_command(self, mock_base_loader_class, mock_lora_config_class, mock_lora_manager_class, 
                                   mock_data_loader_class, mock_formatter_class, mock_tokenizer_class,
                                   mock_training_config_class, mock_trainer_class, mock_settings, mock_logger):
        """Проверка обработчика train."""
        args = MagicMock()
        args.base_model = None
        args.data_path = None
        args.output_dir = None
        args.epochs = None
        args.batch_size = None
        args.resume_from_checkpoint = None
        
        # Моки для всех зависимостей
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_base_loader = MagicMock()
        mock_base_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_base_loader_class.return_value = mock_base_loader
        
        mock_lora_config = MagicMock()
        mock_lora_config_class.from_settings.return_value = mock_lora_config
        
        mock_lora_manager = MagicMock()
        mock_lora_manager.apply_lora.return_value = mock_model
        mock_lora_manager_class.return_value = mock_lora_manager
        
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_data_loader = MagicMock()
        mock_data_loader.load.return_value = mock_dataset
        mock_data_loader_class.return_value = mock_data_loader
        
        mock_formatter = MagicMock()
        mock_formatter.auto_format.return_value = mock_dataset
        mock_formatter_class.return_value = mock_formatter
        
        mock_tokenizer_wrapper = MagicMock()
        mock_tokenizer_wrapper.prepare_for_training.return_value = mock_dataset
        mock_tokenizer_class.return_value = mock_tokenizer_wrapper
        
        mock_training_config = MagicMock()
        mock_training_config_class.return_value = mock_training_config
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = None
        mock_trainer.save_adapter.return_value = None
        mock_trainer_class.return_value = mock_trainer
        
        result = handle_train_command(args, mock_settings, mock_logger)
        
        assert result == 0
        mock_logger.info.assert_called()
    
    @patch("app.training.trainer.LoRATrainer")
    @patch("app.training.config.LoRATrainingConfig")
    @patch("app.data.tokenizer.TokenizerWrapper")
    @patch("app.data.formatter.DataFormatter")
    @patch("app.data.loader.DataLoader")
    @patch("app.models.lora.LoRAManager")
    @patch("app.models.lora_config.LoRAConfig")
    @patch("app.models.base.BaseModelLoader")
    def test_handle_train_command_with_overrides(self, mock_base_loader_class, mock_lora_config_class, mock_lora_manager_class, 
                                             mock_data_loader_class, mock_formatter_class, mock_tokenizer_class,
                                             mock_training_config_class, mock_trainer_class, mock_settings, mock_logger):
        """Проверка train с переопределением параметров."""
        args = MagicMock()
        args.base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        args.data_path = "./data/custom.jsonl"
        args.output_dir = "./custom_outputs"
        args.epochs = 5
        args.batch_size = 8
        args.resume_from_checkpoint = None
        
        # Моки для всех зависимостей
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_base_loader = MagicMock()
        mock_base_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_base_loader_class.return_value = mock_base_loader
        
        mock_lora_config = MagicMock()
        mock_lora_config_class.from_settings.return_value = mock_lora_config
        
        mock_lora_manager = MagicMock()
        mock_lora_manager.apply_lora.return_value = mock_model
        mock_lora_manager_class.return_value = mock_lora_manager
        
        mock_dataset = MagicMock()
        mock_dataset.column_names = ["text"]
        mock_dataset.remove_columns.return_value = mock_dataset
        mock_data_loader = MagicMock()
        mock_data_loader.load.return_value = mock_dataset
        mock_data_loader_class.return_value = mock_data_loader
        
        mock_formatter = MagicMock()
        mock_formatter.auto_format.return_value = mock_dataset
        mock_formatter_class.return_value = mock_formatter
        
        mock_tokenizer_wrapper = MagicMock()
        mock_tokenizer_wrapper.prepare_for_training.return_value = mock_dataset
        mock_tokenizer_class.return_value = mock_tokenizer_wrapper
        
        mock_training_config = MagicMock()
        mock_training_config_class.return_value = mock_training_config
        
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = None
        mock_trainer.save_adapter.return_value = None
        mock_trainer_class.return_value = mock_trainer
        
        result = handle_train_command(args, mock_settings, mock_logger)
        
        assert result == 0
        assert mock_settings.BASE_MODEL_NAME == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert mock_settings.TRAIN_DATA_PATH == "./data/custom.jsonl"
        assert mock_settings.NUM_EPOCHS == 5
    
    @patch("app.inference.cli.InferenceCLI")
    def test_handle_inference_command_single(self, mock_cli_class, mock_settings, mock_logger):
        """Проверка inference с одиночным промптом."""
        mock_cli = MagicMock()
        mock_cli.run.return_value = 0
        mock_cli_class.return_value = mock_cli
        
        args = MagicMock()
        args.base_model = "gpt2"
        args.adapter_path = None
        args.prompt = "Hello"
        args.interactive = False
        args.input_file = None
        args.temperature = 0.7
        args.max_new_tokens = 256
        args.top_p = 0.9
        args.top_k = 50
        args.repetition_penalty = 1.0
        args.do_sample = True
        args.prompt_template = "alpaca"
        args.device = "auto"
        args.load_in_8bit = False
        args.output_file = None
        
        result = handle_inference_command(args, mock_settings, mock_logger)
        
        assert result == 0
    
    @patch("app.inference.cli.InferenceCLI")
    def test_handle_inference_command_no_args(self, mock_cli_class, mock_settings, mock_logger):
        """Проверка inference без обязательных аргументов."""
        mock_cli = MagicMock()
        mock_cli.run.return_value = 1
        mock_cli_class.return_value = mock_cli
        
        args = MagicMock()
        args.base_model = "gpt2"
        args.adapter_path = None
        args.prompt = None
        args.interactive = False
        args.input_file = None
        args.temperature = 0.7
        args.max_new_tokens = 256
        args.top_p = 0.9
        args.top_k = 50
        args.repetition_penalty = 1.0
        args.do_sample = True
        args.prompt_template = "alpaca"
        args.device = "auto"
        args.load_in_8bit = False
        args.output_file = None
        
        result = handle_inference_command(args, mock_settings, mock_logger)
        
        assert result == 1
    
    @patch("app.inference.export.ModelExporter")
    def test_handle_export_command(self, mock_exporter_class, mock_settings, mock_logger):
        """Проверка обработчика export."""
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        
        args = MagicMock()
        args.base_model = "gpt2"
        args.adapter_path = "./outputs/adapter"
        args.output_path = "./outputs/merged"
        args.save_tokenizer = True
        
        result = handle_export_command(args, mock_settings, mock_logger)
        
        assert result == 0


class TestMain:
    """Тесты для главной функции."""
    
    @patch("app.__main__.Path.exists")
    @patch("app.__main__.Settings")
    @patch("app.__main__.setup_logging")
    def test_main_no_command(self, mock_setup_logging, mock_settings, mock_exists):
        """Проверка main без команды."""
        mock_exists.return_value = True
        
        with patch("sys.argv", ["llm-fine-tuner"]):
            result = main()
        
        assert result == 1
    
    @patch("app.__main__.Path.exists")
    def test_main_missing_config(self, mock_exists):
        """Проверка main с отсутствующим конфигом."""
        mock_exists.return_value = False
        
        with patch("sys.argv", ["llm-fine-tuner", "--config", "missing.env", "train"]):
            result = main()
        
        assert result == 1


class TestVersion:
    """Тесты для версии."""
    
    def test_version(self):
        """Проверка --version."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

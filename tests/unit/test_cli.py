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
    
    def test_handle_train_command(self, mock_settings, mock_logger):
        """Проверка обработчика train."""
        args = MagicMock()
        args.base_model = None
        args.data_path = None
        args.output_dir = None
        args.epochs = None
        args.batch_size = None
        args.resume_from_checkpoint = None
        
        result = handle_train_command(args, mock_settings, mock_logger)
        
        assert result == 0
        mock_logger.info.assert_called()
    
    def test_handle_train_command_with_overrides(self, mock_settings, mock_logger):
        """Проверка train с переопределением параметров."""
        args = MagicMock()
        args.base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        args.data_path = "./data/custom.jsonl"
        args.output_dir = "./custom_outputs"
        args.epochs = 5
        args.batch_size = 8
        args.resume_from_checkpoint = None
        
        result = handle_train_command(args, mock_settings, mock_logger)
        
        assert result == 0
        assert mock_settings.BASE_MODEL_NAME == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert mock_settings.TRAIN_DATA_PATH == "./data/custom.jsonl"
        assert mock_settings.NUM_EPOCHS == 5
    
    def test_handle_inference_command_single(self, mock_settings, mock_logger):
        """Проверка inference с одиночным промптом."""
        args = MagicMock()
        args.base_model = "gpt2"
        args.adapter_path = None
        args.prompt = "Hello"
        args.interactive = False
        args.input_file = None
        args.temperature = 0.7
        args.max_new_tokens = 256
        
        result = handle_inference_command(args, mock_settings, mock_logger)
        
        assert result == 0
        mock_logger.info.assert_any_call("Промпт: Hello")
    
    def test_handle_inference_command_no_args(self, mock_settings, mock_logger):
        """Проверка inference без обязательных аргументов."""
        args = MagicMock()
        args.base_model = "gpt2"
        args.adapter_path = None
        args.prompt = None
        args.interactive = False
        args.input_file = None
        
        result = handle_inference_command(args, mock_settings, mock_logger)
        
        assert result == 1
        mock_logger.error.assert_called()
    
    def test_handle_export_command(self, mock_settings, mock_logger):
        """Проверка обработчика export."""
        args = MagicMock()
        args.base_model = "gpt2"
        args.adapter_path = "./outputs/adapter"
        args.output_path = "./outputs/merged"
        
        result = handle_export_command(args, mock_settings, mock_logger)
        
        assert result == 0
        mock_logger.info.assert_any_call("Экспорт модели...")


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

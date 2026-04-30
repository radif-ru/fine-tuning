"""Тесты для модуля inference.cli."""

import argparse
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from app.inference.cli import InferenceCLI, run_inference_command
from app.inference.config import GenerationConfig


class TestInferenceCLIInit:
    """Тесты инициализации InferenceCLI."""
    
    def test_init(self):
        """Проверка инициализации."""
        cli = InferenceCLI()
        assert cli.engine is None
        assert cli.prompt_builder is None


class TestInferenceCLIRun:
    """Тесты метода run."""
    
    @patch("app.inference.cli.InferenceEngine")
    @patch("app.inference.cli.PromptBuilder")
    def test_run_single_prompt(self, mock_builder_class, mock_engine_class, mocker):
        """Проверка одиночной генерации."""
        # Моки
        mock_engine = mocker.MagicMock()
        mock_engine.load.return_value = mock_engine
        mock_engine_class.return_value = mock_engine
        
        mock_builder = mocker.MagicMock()
        mock_builder.build.return_value = "formatted prompt"
        mock_builder_class.return_value = mock_builder
        
        cli = InferenceCLI()
        result = cli.run(
            base_model="gpt2",
            prompt="test prompt",
            config=GenerationConfig(),
        )
        
        assert result == 0
        mock_engine.load.assert_called_once()
        mock_engine.generate.assert_called_once_with("formatted prompt", GenerationConfig())
    
    @patch("app.inference.cli.InferenceEngine")
    def test_run_error(self, mock_engine_class, mocker):
        """Проверка обработки ошибки."""
        mock_engine = mocker.MagicMock()
        mock_engine.load.side_effect = Exception("Load error")
        mock_engine_class.return_value = mock_engine
        
        cli = InferenceCLI()
        result = cli.run(
            base_model="gpt2",
            prompt="test",
        )
        
        assert result == 1


class TestInferenceCLIBatch:
    """Тесты batch обработки."""
    
    @patch("app.inference.cli.InferenceEngine")
    @patch("app.inference.cli.PromptBuilder")
    def test_run_batch(self, mock_builder_class, mock_engine_class, mocker, tmp_path):
        """Проверка batch обработки."""
        # Создаём файл с промптами
        input_file = tmp_path / "prompts.txt"
        input_file.write_text("prompt 1\nprompt 2\n")
        
        # Моки
        mock_engine = mocker.MagicMock()
        mock_engine.load.return_value = mock_engine
        mock_engine.generate_batch.return_value = ["result 1", "result 2"]
        mock_engine_class.return_value = mock_engine
        
        mock_builder = mocker.MagicMock()
        mock_builder.build.return_value = "formatted"
        mock_builder_class.return_value = mock_builder
        
        output_file = tmp_path / "results.txt"
        
        cli = InferenceCLI()
        result = cli.run(
            base_model="gpt2",
            input_file=str(input_file),
            output_file=str(output_file),
            config=GenerationConfig(),
        )
        
        assert result == 0
        mock_engine.generate_batch.assert_called_once()
        assert output_file.exists()
    
    @patch("app.inference.cli.InferenceEngine")
    def test_run_batch_file_not_found(self, mock_engine_class, mocker):
        """Проверка ошибки если файл не найден."""
        mock_engine = mocker.MagicMock()
        mock_engine.load.return_value = mock_engine
        mock_engine_class.return_value = mock_engine
        
        cli = InferenceCLI()
        result = cli.run(
            base_model="gpt2",
            input_file="/nonexistent/file.txt",
        )
        
        assert result == 1


class TestRunInferenceCommand:
    """Тесты функции run_inference_command."""
    
    @patch("app.inference.cli.InferenceCLI")
    def test_run_inference_command(self, mock_cli_class, mocker):
        """Проверка run_inference_command."""
        mock_cli = mocker.MagicMock()
        mock_cli.run.return_value = 0
        mock_cli_class.return_value = mock_cli
        
        args = argparse.Namespace(
            base_model="gpt2",
            adapter_path=None,
            prompt="test",
            interactive=False,
            input_file=None,
            output_file=None,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.0,
            do_sample=True,
            prompt_template="alpaca",
            device="auto",
            load_in_8bit=False,
        )
        
        result = run_inference_command(args)
        
        assert result == 0
        mock_cli.run.assert_called_once()
        call_kwargs = mock_cli.run.call_args.kwargs
        assert call_kwargs["base_model"] == "gpt2"
        assert call_kwargs["prompt"] == "test"


class TestInferenceCLIInteractive:
    """Тесты интерактивного режима."""
    
    @patch("app.inference.cli.InferenceEngine")
    @patch("app.inference.cli.PromptBuilder")
    @patch("builtins.input", side_effect=["/quit"])
    def test_interactive_quit(self, mock_input, mock_builder_class, mock_engine_class, mocker):
        """Проверка выхода из интерактивного режима."""
        mock_engine = mocker.MagicMock()
        mock_engine.load.return_value = mock_engine
        mock_engine_class.return_value = mock_engine
        
        mock_builder = mocker.MagicMock()
        mock_builder_class.return_value = mock_builder
        
        cli = InferenceCLI()
        result = cli.run(
            base_model="gpt2",
            interactive=True,
        )
        
        assert result == 0
    
    @patch("app.inference.cli.InferenceEngine")
    @patch("app.inference.cli.PromptBuilder")
    @patch("builtins.input", side_effect=["Hello", "/quit"])
    def test_interactive_generate(self, mock_input, mock_builder_class, mock_engine_class, mocker):
        """Проверка генерации в интерактивном режиме."""
        mock_engine = mocker.MagicMock()
        mock_engine.load.return_value = mock_engine
        mock_engine.generate.return_value = "Generated response"
        mock_engine_class.return_value = mock_engine
        
        mock_builder = mocker.MagicMock()
        mock_builder.build.return_value = "formatted prompt"
        mock_builder_class.return_value = mock_builder
        
        cli = InferenceCLI()
        result = cli.run(
            base_model="gpt2",
            interactive=True,
        )
        
        assert result == 0
        mock_engine.generate.assert_called_once_with("formatted prompt", None)

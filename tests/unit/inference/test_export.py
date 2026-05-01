"""Тесты для модуля inference.export."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from app.inference.export import ModelExporter, export_model


class TestModelExporterInit:
    """Тесты инициализации ModelExporter."""
    
    def test_init(self):
        """Проверка инициализации."""
        exporter = ModelExporter()
        assert exporter._lora_manager is not None


class TestModelExporterMergeAndSave:
    """Тесты merge_and_save."""
    
    @patch("app.inference.export.BaseModelLoader")
    @patch("app.inference.export.LoRAManager")
    def test_merge_and_save_success(
        self, mock_lora_manager_class, mock_loader_class, mocker, tmp_path
    ):
        """Проверка успешного merge и сохранения."""
        # Моки
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_loader = mocker.MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        mock_lora_manager = mocker.MagicMock()
        mock_lora_manager.load_adapter.return_value = mock_model
        mock_lora_manager.merge_and_unload.return_value = mock_model
        mock_lora_manager_class.return_value = mock_lora_manager
        
        output_path = tmp_path / "merged_model"
        
        exporter = ModelExporter()
        exporter.merge_and_save(
            base_model_name="gpt2",
            adapter_path="/path/to/adapter",
            output_path=str(output_path),
        )
        
        # Проверки
        assert output_path.exists()
        mock_model.save_pretrained.assert_called_once_with(output_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(output_path)
    
    @patch("app.inference.export.BaseModelLoader")
    def test_merge_and_save_model_load_error(self, mock_loader_class, mocker, tmp_path):
        """Проверка ошибки загрузки модели."""
        from app.core.exceptions import ExportError
        
        mock_loader = mocker.MagicMock()
        mock_loader.load.side_effect = Exception("Model not found")
        mock_loader_class.return_value = mock_loader
        
        output_path = tmp_path / "merged_model"
        exporter = ModelExporter()
        
        with pytest.raises(ExportError, match="Ошибка экспорта"):
            exporter.merge_and_save(
                base_model_name="invalid_model",
                adapter_path="/path/to/adapter",
                output_path=str(output_path),
            )


class TestModelExporterValidate:
    """Тесты validate_merged_model."""
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_validate_success(self, mock_tokenizer_from, mock_model_from, mocker):
        """Проверка успешной валидации."""
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_tokenizer.return_value = {"input_ids": mocker.MagicMock()}
        mock_tokenizer.decode.return_value = "Generated text"
        
        mock_model.generate.return_value = mocker.MagicMock()
        
        mock_model_from.return_value = mock_model
        mock_tokenizer_from.return_value = mock_tokenizer
        
        exporter = ModelExporter()
        result = exporter.validate_merged_model("/path/to/model")
        
        assert result is True
    
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    def test_validate_failure(self, mock_model_from):
        """Проверка неудачной валидации."""
        mock_model_from.side_effect = Exception("Model corrupted")
        
        exporter = ModelExporter()
        result = exporter.validate_merged_model("/path/to/model")
        
        assert result is False


class TestExportModelFunction:
    """Тесты функции export_model."""
    
    @patch("app.inference.export.ModelExporter")
    def test_export_model(self, mock_exporter_class, mocker):
        """Проверка функции export_model."""
        mock_exporter = mocker.MagicMock()
        mock_exporter_class.return_value = mock_exporter
        
        export_model(
            base_model="gpt2",
            adapter_path="/adapter",
            output_path="/output",
            save_tokenizer=True,
        )
        
        mock_exporter.merge_and_save.assert_called_once_with(
            base_model_name="gpt2",
            adapter_path="/adapter",
            output_path="/output",
            save_tokenizer=True,
        )


class TestModelExporterEdgeCases:
    """Тесты граничных случаев."""
    
    @patch("app.inference.export.BaseModelLoader")
    @patch("app.inference.export.LoRAManager")
    def test_merge_without_tokenizer(
        self, mock_lora_manager_class, mock_loader_class, mocker, tmp_path
    ):
        """Проверка merge без сохранения токенизатора."""
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_loader = mocker.MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        mock_lora_manager = mocker.MagicMock()
        mock_lora_manager.load_adapter.return_value = mock_model
        mock_lora_manager.merge_and_unload.return_value = mock_model
        mock_lora_manager_class.return_value = mock_lora_manager
        
        output_path = tmp_path / "merged_model"
        
        exporter = ModelExporter()
        exporter.merge_and_save(
            base_model_name="gpt2",
            adapter_path="/path/to/adapter",
            output_path=str(output_path),
            save_tokenizer=False,
        )
        
        # Токенизатор не должен сохраняться
        mock_tokenizer.save_pretrained.assert_not_called()
    
    @patch("app.inference.export.BaseModelLoader")
    @patch("app.inference.export.LoRAManager")
    def test_merge_with_8bit(
        self, mock_lora_manager_class, mock_loader_class, mocker, tmp_path
    ):
        """Проверка merge с 8-bit загрузкой."""
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        
        mock_loader = mocker.MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        mock_lora_manager = mocker.MagicMock()
        mock_lora_manager.load_adapter.return_value = mock_model
        mock_lora_manager.merge_and_unload.return_value = mock_model
        mock_lora_manager_class.return_value = mock_lora_manager
        
        output_path = tmp_path / "merged_model"
        
        exporter = ModelExporter()
        exporter.merge_and_save(
            base_model_name="gpt2",
            adapter_path="/path/to/adapter",
            output_path=str(output_path),
            load_in_8bit=True,
        )
        
        # Проверяем что load вызван с load_in_8bit=True
        mock_loader.load.assert_called_once()
        call_kwargs = mock_loader.load.call_args.kwargs
        assert call_kwargs.get("load_in_8bit") is True

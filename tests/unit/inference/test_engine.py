"""Тесты для модуля inference.engine."""

import pytest
from unittest.mock import MagicMock, Mock, patch

from app.inference.config import GenerationConfig
from app.inference.engine import InferenceEngine


class TestInferenceEngineInit:
    """Тесты инициализации InferenceEngine."""
    
    def test_init_basic(self):
        """Проверка базовой инициализации."""
        engine = InferenceEngine("gpt2")
        
        assert engine.base_model_name == "gpt2"
        assert engine.adapter_path is None
        assert engine.device == "auto"
        assert engine.load_in_8bit is False
        assert engine._model is None
        assert engine._tokenizer is None
    
    def test_init_with_adapter(self):
        """Проверка инициализации с адаптером."""
        engine = InferenceEngine(
            base_model_name="gpt2",
            adapter_path="/path/to/adapter",
            device="cpu",
            load_in_8bit=True,
        )
        
        assert engine.base_model_name == "gpt2"
        assert engine.adapter_path == "/path/to/adapter"
        assert engine.device == "cpu"
        assert engine.load_in_8bit is True


class TestInferenceEngineProperties:
    """Тесты свойств InferenceEngine."""
    
    def test_is_loaded_false(self):
        """Проверка is_loaded когда модель не загружена."""
        engine = InferenceEngine("gpt2")
        assert engine.is_loaded is False
    
    def test_is_loaded_true(self, mocker):
        """Проверка is_loaded когда модель загружена."""
        engine = InferenceEngine("gpt2")
        engine._model = mocker.MagicMock()
        engine._tokenizer = mocker.MagicMock()
        
        assert engine.is_loaded is True
    
    def test_is_merged_default(self):
        """Проверка is_merged по умолчанию."""
        engine = InferenceEngine("gpt2")
        assert engine.is_merged is False
    
    def test_get_model_info_not_loaded(self):
        """Проверка get_model_info без загруженной модели."""
        engine = InferenceEngine("gpt2")
        info = engine.get_model_info()
        
        assert info["status"] == "not_loaded"
    
    def test_get_model_info_loaded(self, mocker):
        """Проверка get_model_info с загруженной моделью."""
        engine = InferenceEngine("gpt2", adapter_path="/adapter")
        mock_config = mocker.MagicMock()
        mock_config.architectures = ["GPT2LMHeadModel"]
        mock_config.vocab_size = 50257
        
        engine._model = mocker.MagicMock()
        engine._model.config = mock_config
        engine._is_merged = True
        
        info = engine.get_model_info()
        
        assert info["status"] == "loaded"
        assert info["base_model"] == "gpt2"
        assert info["adapter"] == "/adapter"
        assert info["merged"] is True
        assert info["architecture"] == "GPT2LMHeadModel"
        assert info["vocab_size"] == 50257


class TestInferenceEngineLoad:
    """Тесты загрузки модели."""
    
    @patch("app.inference.engine.BaseModelLoader")
    @patch("app.inference.engine.LoRAManager")
    def test_load_base_model_only(self, mock_lora_manager, mock_loader_class, mocker):
        """Проверка загрузки только базовой модели."""
        mock_loader = mocker.MagicMock()
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        engine = InferenceEngine("gpt2")
        result = engine.load()
        
        assert result is engine  # Проверка chaining
        assert engine.is_loaded is True
        mock_loader.load.assert_called_once()
        mock_lora_manager.return_value.load_adapter.assert_not_called()
    
    @patch("app.inference.engine.BaseModelLoader")
    @patch("app.inference.engine.LoRAManager")
    def test_load_with_adapter(self, mock_lora_manager, mock_loader_class, mocker):
        """Проверка загрузки с адаптером."""
        mock_loader = mocker.MagicMock()
        mock_model = mocker.MagicMock()
        mock_tokenizer = mocker.MagicMock()
        mock_loader.load.return_value = (mock_model, mock_tokenizer)
        mock_loader_class.return_value = mock_loader
        
        mock_lora = mocker.MagicMock()
        mock_lora.load_adapter.return_value = mock_model
        mock_lora_manager.return_value = mock_lora
        
        engine = InferenceEngine("gpt2", adapter_path="/adapter")
        engine.load()
        
        mock_lora.load_adapter.assert_called_once_with(
            model=mock_model,
            path="/adapter",
        )


class TestInferenceEngineGenerate:
    """Тесты генерации текста."""
    
    def test_generate_not_loaded(self):
        """Проверка ошибки если модель не загружена."""
        from app.core.exceptions import InferenceError
        
        engine = InferenceEngine("gpt2")
        
        with pytest.raises(InferenceError, match="не загружена"):
            engine.generate("test prompt")
    
    @patch("torch.no_grad")
    def test_generate_success(self, mock_no_grad, mocker):
        """Проверка успешной генерации."""
        engine = InferenceEngine("gpt2")
        
        # Моки для модели и токенизатора
        mock_model = mocker.MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = mocker.MagicMock()
        
        mock_tokenizer = mocker.MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": mocker.MagicMock(shape=(1, 5)),
            "attention_mask": mocker.MagicMock(),
        }
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        
        engine._model = mock_model
        engine._tokenizer = mock_tokenizer
        
        config = GenerationConfig(max_new_tokens=50)
        result = engine.generate("test prompt", config)
        
        assert result == "Generated text"
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()


class TestInferenceEngineBatch:
    """Тесты batch генерации."""
    
    def test_generate_batch_not_loaded(self):
        """Проверка ошибки batch генерации без загруженной модели."""
        from app.core.exceptions import InferenceError
        
        engine = InferenceEngine("gpt2")
        
        with pytest.raises(InferenceError, match="не загружена"):
            engine.generate_batch(["prompt1", "prompt2"])
    
    def test_generate_batch_success(self, mocker):
        """Проверка batch генерации."""
        engine = InferenceEngine("gpt2")
        engine._model = mocker.MagicMock()
        engine._tokenizer = mocker.MagicMock()
        
        # Мокаем generate для возврата разных результатов
        engine.generate = mocker.MagicMock(side_effect=["Result 1", "Result 2"])
        
        results = engine.generate_batch(["prompt1", "prompt2"])
        
        assert results == ["Result 1", "Result 2"]
        assert engine.generate.call_count == 2


class TestInferenceEngineMerge:
    """Тесты merge_and_unload."""
    
    def test_merge_not_loaded(self):
        """Проверка ошибки merge без загруженной модели."""
        from app.core.exceptions import InferenceError
        
        engine = InferenceEngine("gpt2")
        
        with pytest.raises(InferenceError, match="не загружена"):
            engine.merge_and_unload()
    
    def test_merge_already_merged(self, mocker, caplog):
        """Проверка предупреждения при повторном merge."""
        import logging
        
        engine = InferenceEngine("gpt2")
        engine._model = mocker.MagicMock()
        engine._is_merged = True
        
        with caplog.at_level(logging.WARNING):
            result = engine.merge_and_unload()
        
        assert result is engine
        assert "уже объединена" in caplog.text
    
    def test_merge_success(self, mocker):
        """Проверка успешного merge."""
        engine = InferenceEngine("gpt2")
        mock_model = mocker.MagicMock()
        merged_model = mocker.MagicMock()
        engine._model = mock_model
        
        # Мокаем LoRAManager
        mock_lora_manager = mocker.MagicMock()
        mock_lora_manager.merge_and_unload.return_value = merged_model
        engine._lora_manager = mock_lora_manager
        
        result = engine.merge_and_unload()
        
        assert result is engine
        assert engine._model is merged_model
        assert engine.is_merged is True
        mock_lora_manager.merge_and_unload.assert_called_once_with(mock_model)

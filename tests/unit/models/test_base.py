"""Тесты для BaseModelLoader."""

import pytest
from unittest.mock import MagicMock, patch

from app.core.exceptions import ModelLoadError
from app.models.base import BaseModelLoader, ModelInfo


class TestModelInfo:
    """Тесты для ModelInfo dataclass."""
    
    def test_creation(self):
        """Проверка создания ModelInfo."""
        info = ModelInfo(
            name="gpt2",
            architecture="GPT2LMHeadModel",
            num_parameters=124439808,
            vocab_size=50257,
            max_length=1024
        )
        
        assert info.name == "gpt2"
        assert info.architecture == "GPT2LMHeadModel"
        assert info.num_parameters == 124439808
        assert info.vocab_size == 50257
        assert info.max_length == 1024


class TestBaseModelLoader:
    """Тесты для BaseModelLoader."""
    
    def test_init(self):
        """Проверка инициализации."""
        loader = BaseModelLoader()
        assert loader.cache_dir is None
        assert loader.model is None
        assert loader.tokenizer is None
    
    def test_init_with_cache_dir(self):
        """Проверка инициализации с cache_dir."""
        loader = BaseModelLoader(cache_dir="/tmp/cache")
        assert loader.cache_dir == "/tmp/cache"
    
    @patch("app.models.base.AutoTokenizer.from_pretrained")
    @patch("app.models.base.AutoModelForCausalLM.from_pretrained")
    @patch("app.models.base.get_device")
    def test_load(self, mock_get_device, mock_from_pretrained, mock_tokenizer):
        """Проверка загрузки модели."""
        mock_get_device.return_value = "cpu"
        
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config._name_or_path = "gpt2"
        mock_model.config.architectures = ["GPT2LMHeadModel"]
        mock_model.config.vocab_size = 50257
        mock_model.config.max_position_embeddings = 1024
        
        mock_from_pretrained.return_value = mock_model
        mock_tokenizer.return_value = MagicMock()
        
        loader = BaseModelLoader()
        model, tokenizer = loader.load("gpt2")
        
        assert model is not None
        assert tokenizer is not None
    
    def test_load_invalid_model(self):
        """Проверка загрузки несуществующей модели."""
        loader = BaseModelLoader()
        
        with pytest.raises(ModelLoadError):
            loader.load("nonexistent-model-12345")
    
    def test_get_model_info_not_loaded(self):
        """Проверка получения info без загруженной модели."""
        loader = BaseModelLoader()
        info = loader.get_model_info()
        
        assert info is None
    
    def test_load_from_local_not_exists(self):
        """Проверка загрузки из несуществующего локального пути."""
        loader = BaseModelLoader()
        
        with pytest.raises(ModelLoadError) as exc_info:
            loader.load_from_local("/nonexistent/path")
        
        assert "не существует" in str(exc_info.value)

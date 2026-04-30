"""Тесты для ModelRegistry."""

import pytest

from app.models.registry import ModelConfig, ModelRegistry, get_registry


class TestModelConfig:
    """Тесты для ModelConfig."""
    
    def test_default_creation(self):
        """Проверка создания с дефолтами."""
        config = ModelConfig(name="gpt2", family="gpt")
        
        assert config.name == "gpt2"
        assert config.family == "gpt"
        assert config.context_length == 2048
        assert config.target_modules == []
        assert config.default_lora_r == 8
    
    def test_custom_creation(self):
        """Проверка создания с кастомными значениями."""
        config = ModelConfig(
            name="test",
            family="llama",
            context_length=4096,
            target_modules=["q_proj"],
            default_lora_r=16
        )
        
        assert config.context_length == 4096
        assert config.target_modules == ["q_proj"]
        assert config.default_lora_r == 16


class TestModelRegistry:
    """Тесты для ModelRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Фикстура для ModelRegistry."""
        return ModelRegistry()
    
    def test_init_with_defaults(self, registry):
        """Проверка инициализации с дефолтами."""
        models = registry.list_models()
        
        assert "tinyllama-1.1b" in models
        assert "phi-3-mini" in models
        assert "gpt2" in models
    
    def test_register(self, registry):
        """Проверка регистрации модели."""
        config = ModelConfig(name="custom", family="custom")
        registry.register("custom-model", config)
        
        assert "custom-model" in registry.list_models()
    
    def test_get_config_by_short_name(self, registry):
        """Проверка получения по короткому имени."""
        config = registry.get_config("gpt2")
        
        assert config.name == "gpt2"
        assert config.family == "gpt"
    
    def test_get_config_by_full_name(self, registry):
        """Проверка получения по полному имени."""
        config = registry.get_config("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        assert config.family == "llama"
    
    def test_get_config_unknown(self, registry):
        """Проверка получения неизвестной модели."""
        config = registry.get_config("unknown-model-123")
        
        assert config.name == "unknown-model-123"
        assert config.family == "unknown"
    
    def test_is_registered_by_short_name(self, registry):
        """Проверка is_registered по короткому имени."""
        assert registry.is_registered("gpt2") is True
        assert registry.is_registered("unknown") is False
    
    def test_is_registered_by_full_name(self, registry):
        """Проверка is_registered по полному имени."""
        assert registry.is_registered("gpt2") is True
    
    def test_get_target_modules(self, registry):
        """Проверка получения target modules."""
        modules = registry.get_target_modules("gpt2")
        
        assert modules == ["c_attn", "c_proj"]
    
    def test_get_target_modules_unknown(self, registry):
        """Проверка получения target modules для неизвестной модели."""
        modules = registry.get_target_modules("unknown")
        
        # Возвращаются стандартные модули
        assert "q_proj" in modules


class TestGetRegistry:
    """Тесты для get_registry."""
    
    def test_returns_singleton(self):
        """Проверка что возвращается singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        
        assert reg1 is reg2
    
    def test_has_default_models(self):
        """Проверка что есть дефолтные модели."""
        registry = get_registry()
        
        assert "gpt2" in registry.list_models()

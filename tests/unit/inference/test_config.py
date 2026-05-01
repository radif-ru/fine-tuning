"""Тесты для модуля inference.config."""

import pytest

from app.inference.config import GenerationConfig, get_default_config


class TestGenerationConfig:
    """Тесты для GenerationConfig dataclass."""
    
    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.top_k == 50
        assert config.repetition_penalty == 1.0
        assert config.do_sample is True
    
    def test_custom_values(self):
        """Проверка установки кастомных значений."""
        config = GenerationConfig(
            max_new_tokens=512,
            temperature=1.0,
            top_p=0.95,
            top_k=100,
            repetition_penalty=1.2,
            do_sample=False
        )
        
        assert config.max_new_tokens == 512
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.top_k == 100
        assert config.repetition_penalty == 1.2
        assert config.do_sample is False
    
    def test_to_dict(self):
        """Проверка преобразования в словарь."""
        config = GenerationConfig(max_new_tokens=100, temperature=0.5)
        config_dict = config.to_dict()
        
        assert config_dict == {
            "max_new_tokens": 100,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "do_sample": True,
        }
    
    def test_from_dict(self):
        """Проверка создания из словаря."""
        config_dict = {
            "max_new_tokens": 128,
            "temperature": 0.8,
            "top_p": 0.85,
            "top_k": 40,
        }
        config = GenerationConfig.from_dict(config_dict)
        
        assert config.max_new_tokens == 128
        assert config.temperature == 0.8
        assert config.top_p == 0.85
        assert config.top_k == 40
        # Значения по умолчанию для отсутствующих полей
        assert config.repetition_penalty == 1.0
        assert config.do_sample is True
    
    def test_get_default_config(self):
        """Проверка функции get_default_config."""
        config = get_default_config()
        
        assert isinstance(config, GenerationConfig)
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7


class TestGenerationConfigValidation:
    """Тесты валидации GenerationConfig."""
    
    def test_invalid_max_new_tokens(self):
        """Проверка валидации max_new_tokens."""
        with pytest.raises(ValueError, match="max_new_tokens"):
            GenerationConfig(max_new_tokens=0)
        
        with pytest.raises(ValueError, match="max_new_tokens"):
            GenerationConfig(max_new_tokens=-1)
    
    def test_invalid_temperature(self):
        """Проверка валидации temperature."""
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=0)
        
        with pytest.raises(ValueError, match="temperature"):
            GenerationConfig(temperature=-0.1)
    
    def test_invalid_top_p(self):
        """Проверка валидации top_p."""
        with pytest.raises(ValueError, match="top_p"):
            GenerationConfig(top_p=-0.1)
        
        with pytest.raises(ValueError, match="top_p"):
            GenerationConfig(top_p=1.1)
    
    def test_invalid_top_k(self):
        """Проверка валидации top_k."""
        with pytest.raises(ValueError, match="top_k"):
            GenerationConfig(top_k=0)
    
    def test_invalid_repetition_penalty(self):
        """Проверка валидации repetition_penalty."""
        with pytest.raises(ValueError, match="repetition_penalty"):
            GenerationConfig(repetition_penalty=0.9)
    
    def test_valid_edge_cases(self):
        """Проверка граничных валидных значений."""
        # Минимальные валидные значения
        config = GenerationConfig(
            max_new_tokens=1,
            temperature=0.01,
            top_p=0.0,
            top_k=1,
            repetition_penalty=1.0,
        )
        assert config.max_new_tokens == 1
        assert config.temperature == 0.01
        assert config.top_p == 0.0
        
        # Максимальные top_p
        config2 = GenerationConfig(top_p=1.0)
        assert config2.top_p == 1.0


class TestGenerationConfigFromEnv:
    """Тесты создания GenerationConfig из окружения."""
    
    def test_from_env_default(self, mocker):
        """Проверка создания из окружения без настроек."""
        mock_settings = mocker.MagicMock()
        mock_settings.MAX_NEW_TOKENS = 256
        mock_settings.TEMPERATURE = 0.7
        mock_settings.TOP_P = 0.9
        mock_settings.TOP_K = 50
        mock_settings.REPETITION_PENALTY = 1.0
        mock_settings.DO_SAMPLE = True
        
        config = GenerationConfig.from_env(mock_settings)
        
        assert isinstance(config, GenerationConfig)
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
    
    def test_from_env_with_settings(self, mocker):
        """Проверка создания из Settings."""
        mock_settings = mocker.MagicMock()
        mock_settings.MAX_NEW_TOKENS = 512
        mock_settings.TEMPERATURE = 0.5
        mock_settings.TOP_P = 0.8
        mock_settings.TOP_K = 100
        mock_settings.REPETITION_PENALTY = 1.1
        mock_settings.DO_SAMPLE = False
        
        config = GenerationConfig.from_env(mock_settings)
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.5
        assert config.top_p == 0.8
        assert config.top_k == 100
        assert config.repetition_penalty == 1.1
        assert config.do_sample is False

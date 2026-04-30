"""Тесты для LoRAConfig."""

import pytest
from unittest.mock import MagicMock

from app.models.lora_config import LoRAConfig


class TestLoRAConfig:
    """Тесты для LoRAConfig."""
    
    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = LoRAConfig()
        
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
    
    def test_custom_values(self):
        """Проверка кастомных значений."""
        config = LoRAConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],
            bias="lora_only",
            task_type="SEQ_CLS"
        )
        
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["c_attn", "c_proj"]
        assert config.bias == "lora_only"
        assert config.task_type == "SEQ_CLS"
    
    def test_validation_r_negative(self):
        """Проверка валидации r < 1."""
        with pytest.raises(ValueError, match="r должен быть >= 1"):
            LoRAConfig(r=0)
    
    def test_validation_alpha_negative(self):
        """Проверка валидации alpha < 1."""
        with pytest.raises(ValueError, match="lora_alpha должен быть >= 1"):
            LoRAConfig(lora_alpha=0)
    
    def test_validation_dropout_negative(self):
        """Проверка валидации dropout < 0."""
        with pytest.raises(ValueError, match="lora_dropout должен быть в"):
            LoRAConfig(lora_dropout=-0.1)
    
    def test_validation_dropout_greater_than_1(self):
        """Проверка валидации dropout > 1."""
        with pytest.raises(ValueError, match="lora_dropout должен быть в"):
            LoRAConfig(lora_dropout=1.5)
    
    def test_validation_bias_invalid(self):
        """Проверка валидации bias."""
        with pytest.raises(ValueError, match="bias должен быть"):
            LoRAConfig(bias="invalid")
    
    def test_get_scaling(self):
        """Проверка расчёта scaling."""
        config = LoRAConfig(r=8, lora_alpha=16)
        
        assert config.get_scaling() == 2.0
    
    def test_to_dict(self):
        """Проверка конвертации в словарь."""
        config = LoRAConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["r"] == 8
        assert config_dict["lora_alpha"] == 16
        assert config_dict["bias"] == "none"
    
    def test_from_settings(self):
        """Проверка создания из Settings."""
        settings = MagicMock()
        settings.LORA_R = 4
        settings.LORA_ALPHA = 8
        settings.LORA_DROPOUT = 0.1
        settings.LORA_TARGET_MODULES = ["c_attn"]
        settings.LORA_BIAS = "all"
        settings.LORA_TASK_TYPE = "SEQ_CLS"
        
        config = LoRAConfig.from_settings(settings)
        
        assert config.r == 4
        assert config.lora_alpha == 8
        assert config.target_modules == ["c_attn"]

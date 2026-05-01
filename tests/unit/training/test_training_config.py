"""Тесты для LoRATrainingConfig."""

import pytest

from app.training.config import LoRATrainingConfig


class TestLoRATrainingConfig:
    """Тесты для LoRATrainingConfig."""
    
    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = LoRATrainingConfig()
        
        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 4
        assert config.learning_rate == 2e-4
        assert config.optim == "adamw_torch"
        assert config.gradient_accumulation_steps == 1
        assert config.fp16 is False
        assert config.bf16 is False
    
    def test_custom_values(self):
        """Проверка кастомных значений."""
        config = LoRATrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=8,
            learning_rate=1e-4,
            optim="adamw_8bit",
            fp16=True
        )
        
        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 8
        assert config.learning_rate == 1e-4
        assert config.optim == "adamw_8bit"
        assert config.fp16 is True
    
    def test_validation_epochs_negative(self):
        """Проверка валидации отрицательных эпох."""
        with pytest.raises(ValueError, match="num_train_epochs должен быть >= 1"):
            LoRATrainingConfig(num_train_epochs=0)
    
    def test_validation_batch_size_zero(self):
        """Проверка валидации batch_size = 0."""
        with pytest.raises(ValueError, match="per_device_train_batch_size должен быть >= 1"):
            LoRATrainingConfig(per_device_train_batch_size=0)
    
    def test_validation_lr_negative(self):
        """Проверка валидации отрицательного learning rate."""
        with pytest.raises(ValueError, match="learning_rate должен быть > 0"):
            LoRATrainingConfig(learning_rate=0)
    
    def test_validation_lr_zero(self):
        """Проверка валидации learning rate = 0."""
        with pytest.raises(ValueError, match="learning_rate должен быть > 0"):
            LoRATrainingConfig(learning_rate=0)
    
    def test_validation_invalid_optimizer(self):
        """Проверка валидации невалидного оптимизатора."""
        with pytest.raises(ValueError, match="Неподдерживаемый оптимизатор"):
            LoRATrainingConfig(optim="invalid_optimizer")
    
    def test_to_dict(self):
        """Проверка конвертации в словарь."""
        config = LoRATrainingConfig(num_train_epochs=5)
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["num_train_epochs"] == 5
        assert config_dict["learning_rate"] == 2e-4
    
    def test_valid_optimizers(self):
        """Проверка валидных оптимизаторов."""
        for optim in ["adamw_torch", "adamw_bnb_8bit", "adamw_8bit"]:
            config = LoRATrainingConfig(optim=optim)
            assert config.optim == optim

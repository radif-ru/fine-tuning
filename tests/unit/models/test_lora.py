"""Тесты для LoRAManager."""

import pytest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from app.core.exceptions import LoRAError
from app.models.lora import LoRAManager
from app.models.lora_config import LoRAConfig


class TestLoRAManager:
    """Тесты для LoRAManager."""
    
    @pytest.fixture
    def manager(self):
        """Фикстура для LoRAManager."""
        return LoRAManager()
    
    @pytest.fixture
    def mock_model(self):
        """Фикстура для mock модели."""
        model = nn.Linear(10, 10)
        return model
    
    @pytest.fixture
    def config(self):
        """Фикстура для LoRAConfig."""
        return LoRAConfig(
            r=4,
            lora_alpha=8,
            target_modules=["weight"],
            bias="none"
        )
    
    def test_init(self, manager):
        """Проверка инициализации."""
        assert manager._peft_model is None
    
    def test_get_trainable_params(self, manager):
        """Проверка подсчёта параметров."""
        model = nn.Linear(10, 5)
        
        trainable, total = manager._get_trainable_params(model)
        
        assert trainable == 55  # 10*5 + 5
        assert total == 55
    
    def test_get_trainable_params_with_frozen(self, manager):
        """Проверка подсчёта с замороженными параметрами."""
        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.requires_grad = False
        
        trainable, total = manager._get_trainable_params(model)
        
        assert trainable == 0
        assert total == 55

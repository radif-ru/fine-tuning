"""Тесты для callbacks."""

import pytest
from unittest.mock import MagicMock, patch

from app.training.callbacks import LoggingCallback, WandBCallback


class TestLoggingCallback:
    """Тесты для LoggingCallback."""
    
    def test_init(self):
        """Проверка инициализации."""
        callback = LoggingCallback()
        assert callback.start_time is None
    
    def test_on_train_begin(self):
        """Проверка on_train_begin."""
        callback = LoggingCallback()
        
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        
        callback.on_train_begin(args, state, control)
        
        assert callback.start_time is not None
    
    def test_on_log(self):
        """Проверка on_log."""
        callback = LoggingCallback()
        
        args = MagicMock()
        state = MagicMock()
        state.global_step = 10
        control = MagicMock()
        logs = {"loss": 0.5, "learning_rate": 0.001}
        
        # Не должно выбросить исключение
        callback.on_log(args, state, control, logs=logs)
    
    def test_on_log_none(self):
        """Проверка on_log с None."""
        callback = LoggingCallback()
        
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        
        # Не должно выбросить исключение
        callback.on_log(args, state, control, logs=None)
    
    def test_on_epoch_end(self):
        """Проверка on_epoch_end."""
        callback = LoggingCallback()
        
        args = MagicMock()
        state = MagicMock()
        state.epoch = 1.5
        control = MagicMock()
        
        # Не должно выбросить исключение
        callback.on_epoch_end(args, state, control)
    
    def test_on_save(self):
        """Проверка on_save."""
        callback = LoggingCallback()
        
        args = MagicMock()
        state = MagicMock()
        state.global_step = 100
        control = MagicMock()
        
        # Не должно выбросить исключение
        callback.on_save(args, state, control)
    
    def test_on_train_end(self):
        """Проверка on_train_end."""
        callback = LoggingCallback()
        
        # Устанавливаем start_time
        import time
        callback.start_time = time.time() - 10  # 10 секунд назад
        
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        
        # Не должно выбросить исключение
        callback.on_train_end(args, state, control)


class TestWandBCallback:
    """Тесты для WandBCallback."""
    
    def test_init(self):
        """Проверка инициализации."""
        callback = WandBCallback(
            project="test-project",
            name="test-run",
            config={"lr": 0.001}
        )
        
        assert callback.project == "test-project"
        assert callback.name == "test-run"
        assert callback.config == {"lr": 0.001}
        assert callback._initialized is False
    
    def test_init_defaults(self):
        """Проверка инициализации с дефолтами."""
        callback = WandBCallback()
        
        assert callback.project == "llm-fine-tuning"
        assert callback.name is None
        assert callback.config == {}
    
    def test_on_train_begin_no_wandb(self):
        """Проверка on_train_begin без wandb."""
        callback = WandBCallback()
        
        args = MagicMock()
        state = MagicMock()
        control = MagicMock()
        
        # Патчим import, чтобы симулировать отсутствие wandb
        with patch.object(callback, '_wandb', None):
            callback.on_train_begin(args, state, control)
        
        assert callback._initialized is False

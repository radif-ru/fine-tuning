"""Тесты для утилит работы с памятью."""

import pytest
from unittest.mock import MagicMock, patch

from app.utils.memory import (
    clear_memory_cache,
    format_memory_size,
    get_max_memory_allocated,
    get_memory_stats,
    reset_peak_memory_stats,
)


class TestGetMemoryStats:
    """Тесты для get_memory_stats функции."""
    
    def test_returns_dict(self):
        """Проверка что возвращается словарь."""
        stats = get_memory_stats()
        assert isinstance(stats, dict)
    
    def test_has_required_keys(self):
        """Проверка наличия обязательных ключей."""
        stats = get_memory_stats()
        required_keys = ["allocated_mb", "reserved_mb", "free_mb", "total_mb"]
        for key in required_keys:
            assert key in stats
    
    def test_types(self):
        """Проверка типов значений."""
        stats = get_memory_stats()
        for key, value in stats.items():
            assert isinstance(value, float)
    
    @patch("app.utils.memory.TORCH_AVAILABLE", False)
    def test_no_torch(self):
        """Проверка без torch."""
        stats = get_memory_stats()
        assert stats["allocated_mb"] == 0.0
        assert stats["reserved_mb"] == 0.0
    
    @patch("app.utils.memory.TORCH_AVAILABLE", False)
    def test_no_torch_all_zero(self):
        """Проверка что все значения нули без torch."""
        stats = get_memory_stats()
        assert all(v == 0.0 for v in stats.values())


class TestClearMemoryCache:
    """Тесты для clear_memory_cache функции."""
    
    @patch("app.utils.memory.TORCH_AVAILABLE", False)
    def test_no_torch(self):
        """Проверка без torch (не должно упасть)."""
        clear_memory_cache()  # Не должно выбросить исключение
    
    @patch("app.utils.memory.torch.cuda.empty_cache")
    @patch("app.utils.memory.torch.cuda.is_available")
    @patch("app.utils.memory.TORCH_AVAILABLE", True)
    def test_with_cuda(self, mock_available, mock_empty_cache):
        """Проверка с доступной CUDA."""
        mock_available.return_value = True
        clear_memory_cache()
        mock_empty_cache.assert_called_once()


class TestGetMaxMemoryAllocated:
    """Тесты для get_max_memory_allocated функции."""
    
    @patch("app.utils.memory.TORCH_AVAILABLE", False)
    def test_no_torch(self):
        """Проверка без torch."""
        max_mem = get_max_memory_allocated()
        assert max_mem == 0.0
    
    @patch("app.utils.memory.TORCH_AVAILABLE", True)
    @patch("app.utils.memory.torch.cuda.is_available")
    def test_no_cuda(self, mock_available):
        """Проверка без CUDA."""
        mock_available.return_value = False
        max_mem = get_max_memory_allocated()
        assert max_mem == 0.0


class TestResetPeakMemoryStats:
    """Тесты для reset_peak_memory_stats функции."""
    
    @patch("app.utils.memory.TORCH_AVAILABLE", False)
    def test_no_torch(self):
        """Проверка без torch (не должно упасть)."""
        reset_peak_memory_stats()  # Не должно выбросить исключение


class TestFormatMemorySize:
    """Тесты для format_memory_size функции."""
    
    def test_mb(self):
        """Проверка форматирования MB."""
        result = format_memory_size(512)
        assert result == "512.00 MB"
    
    def test_gb(self):
        """Проверка форматирования GB."""
        result = format_memory_size(1536)
        assert result == "1.50 GB"
    
    def test_exact_gb(self):
        """Проверка точного GB."""
        result = format_memory_size(1024)
        assert result == "1.00 GB"
    
    def test_zero(self):
        """Проверка нуля."""
        result = format_memory_size(0)
        assert result == "0.00 MB"
    
    def test_decimal(self):
        """Проверка десятичных значений."""
        result = format_memory_size(512.5)
        assert result == "512.50 MB"

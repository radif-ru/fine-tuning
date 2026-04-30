"""Тесты для утилит работы с устройствами."""

import pytest
from unittest.mock import MagicMock, patch

from app.utils.device import (
    get_device,
    get_device_info,
    get_device_name,
    is_cuda_available,
    is_mps_available,
)


class TestGetDevice:
    """Тесты для get_device функции."""
    
    def test_explicit_cpu(self):
        """Проверка явного выбора CPU."""
        device = get_device("cpu")
        assert device == "cpu"
    
    @patch("app.utils.device.TORCH_AVAILABLE", False)
    def test_no_torch(self):
        """Проверка поведения без torch."""
        device = get_device("auto")
        assert device == "cpu"
    
    @patch("app.utils.device.torch.cuda.is_available")
    @patch("app.utils.device.TORCH_AVAILABLE", True)
    def test_auto_with_cuda(self, mock_cuda_available):
        """Проверка auto с доступной CUDA."""
        mock_cuda_available.return_value = True
        device = get_device("auto")
        assert device == "cuda"
    
    @patch("app.utils.device.torch.cuda.is_available")
    @patch("app.utils.device.torch.backends.mps.is_available")
    @patch("app.utils.device.TORCH_AVAILABLE", True)
    def test_auto_with_mps_only(self, mock_mps, mock_cuda):
        """Проверка auto с доступной MPS (нет CUDA)."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        device = get_device("auto")
        assert device == "mps"
    
    @patch("app.utils.device.torch.cuda.is_available")
    @patch("app.utils.device.torch.backends.mps.is_available")
    @patch("app.utils.device.TORCH_AVAILABLE", True)
    def test_auto_with_nothing(self, mock_mps, mock_cuda):
        """Проверка auto без GPU."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        device = get_device("auto")
        assert device == "cpu"
    
    @patch("app.utils.device.torch.cuda.is_available")
    @patch("app.utils.device.TORCH_AVAILABLE", True)
    def test_explicit_cuda(self, mock_cuda_available):
        """Проверка явного выбора CUDA."""
        mock_cuda_available.return_value = True
        device = get_device("cuda")
        assert device == "cuda"
    
    @patch("app.utils.device.torch.cuda.is_available")
    @patch("app.utils.device.TORCH_AVAILABLE", True)
    def test_explicit_cuda_not_available(self, mock_cuda_available):
        """Проверка выбора CUDA когда она недоступна."""
        mock_cuda_available.return_value = False
        device = get_device("cuda")
        assert device == "cpu"


class TestGetDeviceInfo:
    """Тесты для get_device_info функции."""
    
    def test_returns_dict(self):
        """Проверка что возвращается словарь."""
        info = get_device_info()
        assert isinstance(info, dict)
    
    def test_has_required_keys(self):
        """Проверка наличия обязательных ключей."""
        info = get_device_info()
        required_keys = [
            "cuda_available",
            "cuda_device_count",
            "cuda_device_name",
            "mps_available",
            "current_device"
        ]
        for key in required_keys:
            assert key in info
    
    def test_types(self):
        """Проверка типов значений."""
        info = get_device_info()
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["cuda_device_count"], int)
        assert isinstance(info["mps_available"], bool)
        assert isinstance(info["current_device"], str)


class TestGetDeviceName:
    """Тесты для get_device_name функции."""
    
    def test_cpu(self):
        """Проверка имени CPU."""
        name = get_device_name("cpu")
        assert name == "CPU"
    
    def test_mps(self):
        """Проверка имени MPS."""
        name = get_device_name("mps")
        assert name == "Apple Silicon (MPS)"
    
    def test_unknown(self):
        """Проверка неизвестного устройства."""
        name = get_device_name("unknown")
        assert name is None


class TestIsCudaAvailable:
    """Тесты для is_cuda_available функции."""
    
    @patch("app.utils.device.TORCH_AVAILABLE", False)
    def test_no_torch(self):
        """Проверка без torch."""
        assert is_cuda_available() is False
    
    @patch("app.utils.device.torch.cuda.is_available")
    @patch("app.utils.device.TORCH_AVAILABLE", True)
    def test_with_cuda(self, mock_cuda):
        """Проверка с CUDA."""
        mock_cuda.return_value = True
        assert is_cuda_available() is True


class TestIsMpsAvailable:
    """Тесты для is_mps_available функции."""
    
    @patch("app.utils.device.TORCH_AVAILABLE", False)
    def test_no_torch(self):
        """Проверка без torch."""
        assert is_mps_available() is False
    
    @patch("app.utils.device.torch.backends.mps")
    @patch("app.utils.device.TORCH_AVAILABLE", True)
    def test_no_mps_attribute(self, mock_mps):
        """Проверка без атрибута mps."""
        del mock_mps.is_available
        assert is_mps_available() is False

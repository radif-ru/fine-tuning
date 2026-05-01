"""Утилиты для работы с устройствами (CPU/CUDA/MPS).

Функции для определения и управления вычислительными устройствами.
"""

from typing import Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device(preferred: Optional[str] = None) -> str:
    """Определить оптимальное устройство для вычислений.
    
    Args:
        preferred: Предпочтительное устройство ('cuda', 'cpu', 'auto', 'mps')
    
    Returns:
        Строка с устройством ('cuda', 'cuda:0', 'cpu', 'mps')
    
    Examples:
        >>> get_device("auto")
        'cuda'  # если CUDA доступна
        >>> get_device("cpu")
        'cpu'
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    if preferred == "cpu":
        return "cpu"
    
    if preferred == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    if preferred in ("cuda", "auto", None):
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    
    return "cpu"


def get_device_info() -> Dict[str, any]:
    """Получить информацию о доступных устройствах.
    
    Returns:
        Словарь с информацией о устройствах:
        - cuda_available: bool
        - cuda_device_count: int
        - cuda_device_name: str или None
        - mps_available: bool
        - current_device: str
    """
    if not TORCH_AVAILABLE:
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "mps_available": False,
            "current_device": "cpu"
        }
    
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False
    
    info = {
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_name": None,
        "mps_available": mps_available,
        "current_device": get_device("auto")
    }
    
    if cuda_available:
        try:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return info


def get_device_name(device: str) -> Optional[str]:
    """Получить имя устройства.
    
    Args:
        device: Устройство ('cuda', 'cuda:0', 'cpu', 'mps')
    
    Returns:
        Имя устройства или None
    """
    if not TORCH_AVAILABLE:
        return None
    
    if device.startswith("cuda"):
        try:
            if device == "cuda":
                return torch.cuda.get_device_name(0)
            else:
                idx = int(device.split(":")[1])
                return torch.cuda.get_device_name(idx)
        except Exception:
            return None
    
    if device == "mps":
        return "Apple Silicon (MPS)"
    
    if device == "cpu":
        return "CPU"
    
    return None


def is_cuda_available() -> bool:
    """Проверить доступность CUDA.
    
    Returns:
        True если CUDA доступна
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Проверить доступность MPS (Apple Silicon).
    
    Returns:
        True если MPS доступна
    """
    if not TORCH_AVAILABLE:
        return False
    if not hasattr(torch.backends, "mps"):
        return False
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        return False

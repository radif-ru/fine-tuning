"""Утилиты для мониторинга и управления памятью.

Функции для отслеживания использования GPU памяти.
"""

from typing import Dict, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
    """Получить статистику использования памяти.
    
    Args:
        device: Устройство ('cuda', 'cuda:0', 'cpu')
    
    Returns:
        Словарь с метриками памяти (в MB):
        - allocated_mb: выделенная память
        - reserved_mb: зарезервированная память
        - free_mb: свободная память
        - total_mb: общая память
    
    Examples:
        >>> get_memory_stats("cuda")
        {'allocated_mb': 1024.5, 'reserved_mb': 2048.0, 'free_mb': 6144.0, 'total_mb': 8192.0}
    """
    if not TORCH_AVAILABLE:
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}
    
    if device and device.startswith("cuda"):
        try:
            if device == "cuda":
                device_idx = 0
            else:
                device_idx = int(device.split(":")[1])
            
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024 ** 2)
            
            return {
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "free_mb": round(total - allocated, 2),
                "total_mb": round(total, 2),
            }
        except Exception:
            pass
    
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "free_mb": 0.0, "total_mb": 0.0}


def clear_memory_cache() -> None:
    """Очистить кэш CUDA памяти.
    
    Вызывает torch.cuda.empty_cache() для освобождения неиспользуемой памяти.
    Безопасно вызывать даже если CUDA недоступна.
    """
    if not TORCH_AVAILABLE:
        return
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_max_memory_allocated(device: Optional[str] = None) -> float:
    """Получить максимальное выделенную память.
    
    Args:
        device: Устройство ('cuda', 'cuda:0')
    
    Returns:
        Максимальная выделенная память в MB
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    if not torch.cuda.is_available():
        return 0.0
    
    try:
        if device and device.startswith("cuda"):
            if device == "cuda":
                device_idx = 0
            else:
                device_idx = int(device.split(":")[1])
        else:
            device_idx = 0
        
        max_allocated = torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2)
        return round(max_allocated, 2)
    except Exception:
        return 0.0


def reset_peak_memory_stats(device: Optional[str] = None) -> None:
    """Сбросить статистику пиковой памяти.
    
    Args:
        device: Устройство ('cuda', 'cuda:0')
    """
    if not TORCH_AVAILABLE:
        return
    
    if not torch.cuda.is_available():
        return
    
    try:
        if device and device.startswith("cuda"):
            if device == "cuda":
                device_idx = 0
            else:
                device_idx = int(device.split(":")[1])
        else:
            device_idx = 0
        
        torch.cuda.reset_peak_memory_stats(device_idx)
    except Exception:
        pass


def format_memory_size(mb: float) -> str:
    """Форматировать размер памяти в читаемый вид.
    
    Args:
        mb: Размер в мегабайтах
    
    Returns:
        Строка с форматированным размером (например, "1.5 GB")
    """
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    else:
        return f"{mb:.2f} MB"

"""Утилиты приложения.

Модуль предоставляет функции для:
- Определения доступных устройств (CUDA, MPS, CPU)
- Получения информации о памяти GPU
- Управления кэшем памяти
"""

from app.utils.device import get_device, get_device_info, get_device_name, is_cuda_available, is_mps_available
from app.utils.memory import get_memory_stats, clear_memory_cache, get_max_memory_allocated, reset_peak_memory_stats, format_memory_size

__all__ = [
    "get_device",
    "get_device_info",
    "get_device_name",
    "is_cuda_available",
    "is_mps_available",
    "get_memory_stats",
    "clear_memory_cache",
    "get_max_memory_allocated",
    "reset_peak_memory_stats",
    "format_memory_size",
]

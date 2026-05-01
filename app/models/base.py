"""Загрузка базовых моделей из HuggingFace Hub и локальных путей.

Класс BaseModelLoader отвечает за загрузку предобученных моделей
и их токенизаторов.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from app.core.exceptions import ModelLoadError
from app.utils.device import get_device


@dataclass
class ModelInfo:
    """Информация о загруженной модели.
    
    Attributes:
        name: Имя модели
        architecture: Архитектура модели
        num_parameters: Количество параметров
        vocab_size: Размер словаря
        max_length: Максимальная длина контекста
    """
    name: str
    architecture: str
    num_parameters: int
    vocab_size: int
    max_length: int


class BaseModelLoader:
    """Загрузчик базовых моделей из HuggingFace Hub.
    
    Поддерживает загрузку:
    - Из HuggingFace Hub по имени модели
    - Из локального пути
    - С различными опциями (8-bit, trust_remote_code)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Инициализация загрузчика.
        
        Args:
            cache_dir: Директория кэша моделей
        """
        self.cache_dir = cache_dir
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
    
    def load(
        self,
        model_name: str,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[str] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Загрузить модель и токенизатор.
        
        Args:
            model_name: Имя модели в HF Hub или локальный путь
            load_in_8bit: Загрузить в 8-bit режиме (через bitsandbytes)
            trust_remote_code: Доверять remote code в репозитории
            device_map: Стратегия размещения на устройствах ('auto', 'cpu', 'cuda')
            torch_dtype: Тип данных ('float16', 'bfloat16', 'float32')
        
        Returns:
            Кортеж (модель, токенизатор)
        
        Raises:
            ModelLoadError: При ошибке загрузки
        """
        try:
            # Загрузка токенизатора
            tokenizer = self._load_tokenizer(
                model_name,
                trust_remote_code=trust_remote_code
            )
            
            # Загрузка модели
            model = self._load_model(
                model_name,
                load_in_8bit=load_in_8bit,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                torch_dtype=torch_dtype
            )
            
            self._model = model
            self._tokenizer = tokenizer
            
            return model, tokenizer
            
        except Exception as e:
            raise ModelLoadError(
                f"Не удалось загрузить модель: {model_name}",
                error=str(e)
            )
    
    def load_from_local(
        self,
        path: Union[str, Path],
        load_in_8bit: bool = False,
        trust_remote_code: bool = False
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Загрузить модель из локального пути.
        
        Args:
            path: Путь к локальной директории с моделью
            load_in_8bit: Загрузить в 8-bit режиме
            trust_remote_code: Доверять remote code
        
        Returns:
            Кортеж (модель, токенизатор)
        """
        path = Path(path)
        if not path.exists():
            raise ModelLoadError(
                f"Локальный путь не существует: {path}"
            )
        
        return self.load(
            str(path),
            load_in_8bit=load_in_8bit,
            trust_remote_code=trust_remote_code
        )
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Получить информацию о загруженной модели.
        
        Returns:
            ModelInfo или None если модель не загружена
        """
        if self._model is None:
            return None
        
        config = self._model.config
        
        # Подсчёт параметров
        num_params = sum(p.numel() for p in self._model.parameters())
        
        return ModelInfo(
            name=getattr(config, "_name_or_path", "unknown"),
            architecture=config.architectures[0] if hasattr(config, "architectures") else "unknown",
            num_parameters=num_params,
            vocab_size=getattr(config, "vocab_size", 0),
            max_length=getattr(config, "max_position_embeddings", 0) or getattr(config, "n_positions", 0)
        )
    
    def _load_tokenizer(
        self,
        model_name: str,
        trust_remote_code: bool = False
    ) -> PreTrainedTokenizer:
        """Загрузить токенизатор.
        
        Args:
            model_name: Имя модели
            trust_remote_code: Доверять remote code
        
        Returns:
            Загруженный токенизатор
        """
        kwargs = {
            "trust_remote_code": trust_remote_code,
            "use_fast": True,
        }
        
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **kwargs
        )
        
        # Установка pad_token если не задан
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model(
        self,
        model_name: str,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
        device_map: Optional[str] = "auto",
        torch_dtype: Optional[str] = None
    ) -> PreTrainedModel:
        """Загрузить модель.
        
        Args:
            model_name: Имя модели
            load_in_8bit: 8-bit режим
            trust_remote_code: Доверять remote code
            device_map: Стратегия размещения
            torch_dtype: Тип данных
        
        Returns:
            Загруженная модель
        """
        import torch
        
        kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        
        # Device map
        if device_map == "auto":
            kwargs["device_map"] = get_device("auto")
        elif device_map:
            kwargs["device_map"] = device_map
        
        # 8-bit quantization
        if load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                raise ModelLoadError(
                    "8-bit quantization требует установки bitsandbytes: "
                    "pip install bitsandbytes"
                )
        
        # Torch dtype
        if torch_dtype:
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            if torch_dtype in dtype_map:
                kwargs["torch_dtype"] = dtype_map[torch_dtype]
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs
        )
        
        return model
    
    @property
    def model(self) -> Optional[PreTrainedModel]:
        """Получить загруженную модель."""
        return self._model
    
    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """Получить загруженный токенизатор."""
        return self._tokenizer

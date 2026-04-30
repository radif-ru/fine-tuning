# Инференс и генерация

## Обзор

Модуль инференса предоставляет API для генерации текста дообученными моделями с поддержкой:
- Загрузки базовой модели + LoRA адаптера
- Гибких параметров генерации
- Batch processing
- Интерактивного режима

## Архитектура

```
┌─────────────────────────────────────────────┐
│           InferenceEngine                    │
│  ┌───────────────────────────────────────┐  │
│  │  Model Loader (base + LoRA adapter)   │  │
│  └─────────────────┬─────────────────────┘  │
│                    │                        │
│  ┌─────────────────▼─────────────────────┐  │
│  │         Tokenizer                   │  │
│  └─────────────────┬─────────────────────┘  │
│                    │                        │
│  ┌─────────────────▼─────────────────────┐  │
│  │      Generation Pipeline              │  │
│  │  ┌─────────────────────────────┐      │  │
│  │  │  Sampling (temperature,     │      │  │
│  │  │  top_p, top_k)            │      │  │
│  │  └─────────────────────────────┘      │  │
│  └─────────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Использование

### Простой инференс

```python
from app.inference.engine import InferenceEngine

engine = InferenceEngine(
    model_path="./checkpoints/final",
    device="auto"  # auto, cpu, cuda, cuda:0
)

response = engine.generate(
    prompt="Напиши функцию на Python для сортировки списка",
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9
)

print(response)
```

### Интерактивный режим

```python
engine = InferenceEngine(model_path="./checkpoints/final")

while True:
    prompt = input("User: ")
    if prompt.lower() in ["exit", "quit"]:
        break
    
    response = engine.generate(prompt)
    print(f"Assistant: {response}")
```

### Batch processing

```python
prompts = [
    "Объясни Python",
    "Напиши приветствие",
    "Что такое LoRA?"
]

responses = engine.generate_batch(
    prompts,
    batch_size=4,
    max_new_tokens=128
)
```

## Параметры генерации

| Параметр | Тип | По умолчанию | Описание |
|----------|-----|--------------|----------|
| `max_new_tokens` | int | 256 | Максимальное количество новых токенов |
| `temperature` | float | 0.7 | Температура сэмплинга (0 = deterministic) |
| `top_p` | float | 0.9 | Nucleus sampling (0.9 = top 90% probability mass) |
| `top_k` | int | 50 | Top-k sampling (50 = top 50 токенов) |
| `repetition_penalty` | float | 1.0 | Штраф за повторения (1.0 = отключено) |
| `do_sample` | bool | True | Использовать sampling (False = greedy) |

### Temperature

- **0.0-0.3**: Детерминированный, консервативный вывод
- **0.5-0.7**: Сбалансированный (рекомендуется)
- **0.8-1.0**: Креативный, разнообразный вывод
- **>1.0**: Хаотичный, менее связный

### Top-p (Nucleus sampling)

```
top_p=0.9: сэмплируем из минимального набора токенов,
          суммарная вероятность которых >= 0.9
```

### Top-k

```
top_k=50: рассматриваем только 50 токенов с наибольшей вероятностью
```

## Реализация

### InferenceEngine

```python
class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None
    ):
        self.device = self._get_device(device)
        self.model, self.tokenizer = self._load_model(model_path)
    
    def _get_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        return torch.device(device)
    
    def _load_model(
        self,
        model_path: str
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Загрузка базовой модели
        base_model_path = self._get_base_model_path(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )
        
        # Загрузка LoRA адаптера
        if self._has_lora_adapter(model_path):
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()  # Опционально: слияние адаптера
        
        model.to(self.device)
        model.eval()
        
        return model, tokenizer
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> str:
        # Токенизация
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.device)
        
        # Генерация
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Декодирование (убираем prompt из ответа)
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def generate_batch(
        self,
        prompts: list[str],
        batch_size: int = 4,
        **generation_kwargs
    ) -> list[str]:
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            
            # Токенизация батча
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Декодирование
            batch_responses = self.tokenizer.batch_decode(
                outputs[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            responses.extend(batch_responses)
        
        return responses
```

## Форматы промптов

### Base model

```python
prompt = "Вопрос: Что такое LoRA?\nОтвет:"
```

### Chat model (instruction tuned)

```python
# Alpaca-style
prompt = """### Instruction:
Напиши функцию для сортировки

### Response:
"""

# ChatML-style
prompt = """<|im_start|>user
Напиши функцию для сортировки<|im_end|>
<|im_start|>assistant
"""
```

## Оптимизации

### Mixed Precision

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16  # или bfloat16
)
```

### CUDA Graphs (для повторяющихся запросов)

```python
# Compile model for faster inference (PyTorch 2.0+)
model = torch.compile(model)
```

### KV Cache

```python
# PyTorch автоматически использует KV cache
# Для очистки между сессиями:
model.generation_config.cache_implementation = None
```

## Обработка ошибок

```python
class InferenceError(Exception):
    """Базовый класс для ошибок инференса"""
    pass

class ModelNotFoundError(InferenceError):
    """Модель не найдена по указанному пути"""
    pass

class GenerationError(InferenceError):
    """Ошибка во время генерации"""
    pass

# Использование
try:
    engine = InferenceEngine(model_path)
except ModelNotFoundError as e:
    logger.error("Модель не найдена: %s", e)
except torch.cuda.OutOfMemoryError:
    logger.error("Недостаточно памяти GPU, попробуйте уменьшить max_new_tokens")
```

## Мониторинг

```python
import time

start = time.time()
response = engine.generate(prompt)
latency = time.time() - start

logger.info(
    "Инференс завершен | latency=%.3fs tokens=%d",
    latency, len(response.split())
)
```

## Best Practices

1. **Используйте `model.eval()`** — отключает dropout и batch norm updates
2. **Оберните в `torch.no_grad()`** — отключает вычисление градиентов
3. **Укажите `pad_token_id`** — предотвращает warnings и ошибки
4. **Очищайте CUDA cache** при длительных сессиях
5. **Используйте batch processing** для множественных запросов

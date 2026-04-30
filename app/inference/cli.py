"""CLI для инференса моделей.

Поддерживает:
- Single prompt mode
- Interactive mode (REPL)
- Batch processing из файла
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from app.core.exceptions import InferenceError
from app.core.logging_config import get_logger

from .config import GenerationConfig
from .engine import InferenceEngine
from .prompt import PromptBuilder

logger = get_logger("inference.cli")


class InferenceCLI:
    """CLI для инференса языковых моделей."""
    
    def __init__(self):
        """Инициализация CLI."""
        self.engine: Optional[InferenceEngine] = None
        self.prompt_builder: Optional[PromptBuilder] = None
    
    def run(
        self,
        base_model: str,
        adapter_path: Optional[str] = None,
        prompt: Optional[str] = None,
        interactive: bool = False,
        input_file: Optional[str] = None,
        output_file: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        prompt_template: str = "alpaca",
        device: str = "auto",
        load_in_8bit: bool = False,
    ) -> int:
        """Запустить CLI инференса.
        
        Args:
            base_model: Имя базовой модели
            adapter_path: Путь к LoRA адаптеру
            prompt: Промпт для одиночной генерации
            interactive: Интерактивный режим
            input_file: Файл с промптами для batch режима
            output_file: Файл для сохранения результатов
            config: Конфигурация генерации
            prompt_template: Шаблон промптов
            device: Устройство для инференса
            load_in_8bit: Загрузить в 8-bit режиме
        
        Returns:
            Exit code (0 - успех, 1 - ошибка)
        """
        try:
            # Инициализация engine
            logger.info(f"Инициализация модели | model={base_model}")
            self.engine = InferenceEngine(
                base_model_name=base_model,
                adapter_path=adapter_path,
                device=device,
                load_in_8bit=load_in_8bit,
            )
            self.engine.load()
            
            # Инициализация prompt builder
            self.prompt_builder = PromptBuilder(prompt_template)
            if self.engine._tokenizer:
                self.prompt_builder.set_tokenizer(self.engine._tokenizer)
            
            logger.info("Модель загружена и готова к инференсу")
            
            # Определение режима работы
            if interactive:
                return self._run_interactive(config)
            elif input_file:
                return self._run_batch(input_file, output_file, config)
            elif prompt:
                return self._run_single(prompt, config)
            else:
                logger.error("Укажите --prompt, --interactive или --input-file")
                return 1
                
        except KeyboardInterrupt:
            logger.info("Прервано пользователем (Ctrl+C)")
            return 0
        except Exception as e:
            logger.error(f"Ошибка инференса: {e}")
            return 1
    
    def _run_single(self, prompt: str, config: Optional[GenerationConfig]) -> int:
        """Одиночная генерация.
        
        Args:
            prompt: Промпт
            config: Конфигурация генерации
        
        Returns:
            Exit code
        """
        try:
            # Форматирование промпта
            formatted_prompt = self.prompt_builder.build(prompt)
            
            logger.info(f"Генерация | prompt_length={len(formatted_prompt)}")
            
            # Генерация
            result = self.engine.generate(formatted_prompt, config)
            
            # Вывод результата
            print("\n" + "=" * 50)
            print("Результат:")
            print("=" * 50)
            print(result)
            print("=" * 50 + "\n")
            
            return 0
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return 1
    
    def _run_interactive(self, config: Optional[GenerationConfig]) -> int:
        """Интерактивный режим (REPL).
        
        Args:
            config: Конфигурация генерации
        
        Returns:
            Exit code
        """
        print("\n" + "=" * 50)
        print("Интерактивный режим инференса")
        print("=" * 50)
        print("Команды:")
        print("  /quit, /q     - выход")
        print("  /clear, /c    - очистить экран")
        print("  /help, /h     - справка")
        print("  Ctrl+C        - выход")
        print("=" * 50 + "\n")
        
        while True:
            try:
                # Чтение ввода
                user_input = input("> ").strip()
                
                # Обработка команд
                if not user_input:
                    continue
                
                if user_input in ("/quit", "/q"):
                    print("Выход...")
                    break
                
                if user_input in ("/clear", "/c"):
                    print("\n" * 50)  # Простая очистка
                    continue
                
                if user_input in ("/help", "/h"):
                    print("\nКоманды:")
                    print("  /quit, /q     - выход")
                    print("  /clear, /c    - очистить экран")
                    print("  /help, /h     - справка")
                    print("  Введите текст для генерации ответа модели.\n")
                    continue
                
                # Генерация
                formatted_prompt = self.prompt_builder.build(user_input)
                
                print("\nГенерация...")
                result = self.engine.generate(formatted_prompt, config)
                
                print(f"\n{result}\n")
                
            except KeyboardInterrupt:
                print("\n\nВыход...")
                break
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                print(f"\nОшибка: {e}\n")
        
        return 0
    
    def _run_batch(
        self,
        input_file: str,
        output_file: Optional[str],
        config: Optional[GenerationConfig]
    ) -> int:
        """Batch обработка из файла.
        
        Args:
            input_file: Путь к файлу с промптами
            output_file: Путь для сохранения результатов
            config: Конфигурация генерации
        
        Returns:
            Exit code
        """
        try:
            # Чтение промптов
            input_path = Path(input_file)
            if not input_path.exists():
                logger.error(f"Файл не найден: {input_file}")
                return 1
            
            prompts = input_path.read_text(encoding="utf-8").strip().split("\n")
            prompts = [p.strip() for p in prompts if p.strip()]
            
            logger.info(f"Batch обработка | prompts={len(prompts)}")
            
            # Форматирование промптов
            formatted_prompts = [
                self.prompt_builder.build(p) for p in prompts
            ]
            
            # Генерация
            results = self.engine.generate_batch(formatted_prompts, config)
            
            # Вывод результатов
            output_lines = []
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                output_lines.append(f"# Prompt {i+1}: {prompt}")
                output_lines.append(result)
                output_lines.append("-" * 50)
            
            output_text = "\n".join(output_lines)
            
            if output_file:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(output_text, encoding="utf-8")
                logger.info(f"Результаты сохранены | path={output_file}")
            else:
                print("\n" + output_text + "\n")
            
            return 0
            
        except Exception as e:
            logger.error(f"Ошибка batch обработки: {e}")
            return 1


def run_inference_command(args: argparse.Namespace) -> int:
    """Запустить команду инференса из argparse.
    
    Args:
        args: Аргументы командной строки
    
    Returns:
        Exit code
    """
    # Создание GenerationConfig из аргументов
    config = GenerationConfig(
        max_new_tokens=getattr(args, "max_new_tokens", 256),
        temperature=getattr(args, "temperature", 0.7),
        top_p=getattr(args, "top_p", 0.9),
        top_k=getattr(args, "top_k", 50),
        repetition_penalty=getattr(args, "repetition_penalty", 1.0),
        do_sample=getattr(args, "do_sample", True),
    )
    
    cli = InferenceCLI()
    
    return cli.run(
        base_model=args.base_model,
        adapter_path=getattr(args, "adapter_path", None),
        prompt=getattr(args, "prompt", None),
        interactive=getattr(args, "interactive", False),
        input_file=getattr(args, "input_file", None),
        output_file=getattr(args, "output_file", None),
        config=config,
        prompt_template=getattr(args, "prompt_template", "alpaca"),
        device=getattr(args, "device", "auto"),
        load_in_8bit=getattr(args, "load_in_8bit", False),
    )

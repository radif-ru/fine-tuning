"""Точка входа для CLI фреймворка fine-tuning LLM.

Запуск: python -m app [command] [options]
"""

import argparse
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.exceptions import ConfigurationError, FineTuningError
from app.core.logging_config import setup_logging


def create_parser() -> argparse.ArgumentParser:
    """Создание парсера аргументов командной строки."""
    parser = argparse.ArgumentParser(
        prog="llm-fine-tuner",
        description="Фреймворк для дообучения LLM с использованием LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python -m app --config .env train --data-path ./data/train.jsonl
  python -m app inference --base-model ./outputs/final --interactive
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default=".env",
        help="Путь к файлу конфигурации (.env)"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    # Подкоманды
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")
    
    # Команда обучения
    train_parser = subparsers.add_parser(
        "train",
        help="Запустить обучение модели",
        description="Обучение LLM с LoRA на предоставленных данных"
    )
    train_parser.add_argument(
        "--base-model",
        help="Базовая модель (переопределяет BASE_MODEL_NAME из .env)"
    )
    train_parser.add_argument(
        "--data-path",
        help="Путь к данным (переопределяет TRAIN_DATA_PATH из .env)"
    )
    train_parser.add_argument(
        "--output-dir",
        help="Директория вывода (переопределяет OUTPUT_DIR из .env)"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Количество эпох (переопределяет NUM_EPOCHS из .env)"
    )
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        help="Batch size (переопределяет PER_DEVICE_BATCH_SIZE из .env)"
    )
    train_parser.add_argument(
        "--resume-from-checkpoint",
        help="Продолжить обучение из чекпоинта"
    )
    
    # Команда инференса
    inference_parser = subparsers.add_parser(
        "inference",
        help="Запустить инференс",
        description="Генерация текста с дообученной моделью"
    )
    inference_parser.add_argument(
        "--base-model",
        required=True,
        help="Базовая модель или путь к ней"
    )
    inference_parser.add_argument(
        "--adapter-path",
        help="Путь к LoRA адаптеру (опционально)"
    )
    inference_parser.add_argument(
        "--prompt", "-p",
        help="Промпт для генерации (одиночный режим)"
    )
    inference_parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Интерактивный режим (REPL)"
    )
    inference_parser.add_argument(
        "--input-file",
        help="Файл с промптами (batch режим, одна строка = один промпт)"
    )
    inference_parser.add_argument(
        "--output-file",
        help="Файл для сохранения результатов (batch режим)"
    )
    inference_parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Максимум новых токенов для генерации"
    )
    inference_parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Temperature для sampling"
    )
    inference_parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling"
    )
    inference_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling"
    )
    inference_parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Штраф за повторения"
    )
    inference_parser.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Использовать sampling"
    )
    inference_parser.add_argument(
        "--prompt-template",
        default="alpaca",
        choices=["alpaca", "raw", "chat"],
        help="Шаблон промптов"
    )
    inference_parser.add_argument(
        "--device",
        default="auto",
        help="Устройство (auto, cpu, cuda, cuda:0)"
    )
    inference_parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Загрузить в 8-bit режиме"
    )
    
    # Команда экспорта
    export_parser = subparsers.add_parser(
        "export",
        help="Экспортировать модель (merge LoRA)",
        description="Объединение LoRA адаптера с базовой моделью"
    )
    export_parser.add_argument(
        "--base-model",
        required=True,
        help="Базовая модель"
    )
    export_parser.add_argument(
        "--adapter-path",
        required=True,
        help="Путь к LoRA адаптеру"
    )
    export_parser.add_argument(
        "--output-path",
        required=True,
        help="Путь для сохранения объединённой модели"
    )
    
    # Команда генерации датасета
    generate_parser = subparsers.add_parser(
        "generate-dataset",
        help="Сгенерировать синтетический датасет",
        description="Генерация синтетических данных для обучения"
    )
    generate_parser.add_argument(
        "--type", "-t",
        default="synthetic",
        choices=["synthetic", "qa"],
        help="Тип генератора"
    )
    generate_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Путь к выходному файлу"
    )
    generate_parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Количество примеров для генерации"
    )
    generate_parser.add_argument(
        "--format", "-f",
        default="jsonl",
        choices=["jsonl", "csv"],
        help="Формат выходного файла"
    )
    generate_parser.add_argument(
        "--topics",
        nargs="+",
        choices=["programming", "science", "general"],
        help="Темы для генерации synthetic (если не указаны, используются все)"
    )
    generate_parser.add_argument(
        "--categories",
        nargs="+",
        choices=["general", "technical", "practical"],
        help="Категории для генерации qa (если не указаны, используются все)"
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        help="Seed для воспроизводимости"
    )
    
    return parser


def handle_train_command(args: argparse.Namespace, settings: Settings, logger) -> int:
    """Обработка команды train.
    
    Args:
        args: Аргументы командной строки
        settings: Конфигурация приложения
        logger: Логгер
    
    Returns:
        Exit code (0 - успех, 1 - ошибка)
    """
    logger.info("Запуск обучения...")
    
    # Переопределяем настройки из аргументов
    if args.base_model:
        settings.BASE_MODEL_NAME = args.base_model
    if args.data_path:
        settings.TRAIN_DATA_PATH = args.data_path
    if args.output_dir:
        settings.OUTPUT_DIR = args.output_dir
    if args.epochs:
        settings.NUM_EPOCHS = args.epochs
    if args.batch_size:
        settings.PER_DEVICE_BATCH_SIZE = args.batch_size
    
    logger.info(f"Модель: {settings.BASE_MODEL_NAME}")
    logger.info(f"Данные: {settings.TRAIN_DATA_PATH}")
    logger.info(f"Вывод: {settings.OUTPUT_DIR}")
    logger.info(f"Эпохи: {settings.NUM_EPOCHS}")

    if args.resume_from_checkpoint:
        logger.info(f"Продолжение из: {args.resume_from_checkpoint}")

    try:
        from app.models.base import BaseModelLoader
        from app.models.lora import LoRAManager
        from app.models.lora_config import LoRAConfig
        from app.training.trainer import LoRATrainer
        from app.training.config import LoRATrainingConfig
        from app.data.loader import DataLoader
        from app.data.formatter import DataFormatter
        from app.data.tokenizer import TokenizerWrapper

        # Загрузка модели
        logger.info("Загрузка базовой модели...")
        model_loader = BaseModelLoader(cache_dir=settings.HF_CACHE_DIR)
        model, tokenizer = model_loader.load(
            model_name=settings.BASE_MODEL_NAME,
            load_in_8bit=settings.USE_8BIT_ADAM,
            trust_remote_code=settings.TRUST_REMOTE_CODE,
            device_map="auto",
            torch_dtype="float16" if settings.FP16 else ("bfloat16" if settings.BF16 else None)
        )

        # Применение LoRA
        logger.info("Применение LoRA адаптеров...")
        lora_config = LoRAConfig.from_settings(settings)
        lora_manager = LoRAManager()
        model = lora_manager.apply_lora(model, lora_config)

        # Загрузка и подготовка данных
        logger.info("Загрузка данных...")
        data_loader = DataLoader()
        raw_dataset = data_loader.load(settings.TRAIN_DATA_PATH, format="jsonl")

        formatter = DataFormatter(
            instruction_column=settings.INSTRUCTION_COLUMN or "instruction",
            input_column="input",
            response_column=settings.RESPONSE_COLUMN or "output",
            text_column=settings.TEXT_COLUMN
        )
        formatted_dataset = formatter.auto_format(raw_dataset)

        logger.info(f"Колонки после форматирования: {formatted_dataset.column_names}")

        # Удаляем исходные колонки, оставляем только text
        original_columns = ["instruction", "input", "output"]
        columns_to_remove = [col for col in original_columns if col in formatted_dataset.column_names]
        if columns_to_remove:
            formatted_dataset = formatted_dataset.remove_columns(columns_to_remove)
            logger.info(f"Удалены колонки: {columns_to_remove}")

        tokenizer_wrapper = TokenizerWrapper(
            tokenizer,
            max_length=settings.MAX_SEQ_LENGTH,
            truncation=True,
            padding="max_length"
        )
        tokenized_dataset = tokenizer_wrapper.prepare_for_training(formatted_dataset)

        logger.info(f"Колонки после токенизации: {tokenized_dataset.column_names}")

        # Создание конфигурации обучения
        training_config = LoRATrainingConfig(
            num_train_epochs=settings.NUM_EPOCHS,
            per_device_train_batch_size=settings.PER_DEVICE_BATCH_SIZE,
            learning_rate=settings.LEARNING_RATE,
            warmup_steps=int(settings.WARMUP_RATIO * 1000),
            weight_decay=settings.WEIGHT_DECAY,
            logging_steps=settings.LOGGING_STEPS,
            save_steps=settings.SAVE_STEPS,
            eval_steps=settings.EVAL_STEPS,
            save_total_limit=settings.SAVE_TOTAL_LIMIT,
            gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
            fp16=settings.FP16,
            bf16=settings.BF16,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

        # Создание тренера
        logger.info("Создание тренера...")
        trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            config=training_config,
            train_dataset=tokenized_dataset,
        )

        # Обучение
        logger.info("Начало обучения...")
        trainer.train(
            output_dir=settings.OUTPUT_DIR,
            resume_from_checkpoint=args.resume_from_checkpoint
        )

        # Сохранение адаптера
        logger.info("Сохранение LoRA адаптера...")
        adapter_path = Path(settings.OUTPUT_DIR) / "final"
        trainer.save_adapter(str(adapter_path))

        logger.info(f"Обучение завершено. Адаптер сохранён в: {adapter_path}")
        return 0

    except Exception as e:
        logger.error(f"Ошибка обучения: {e}")
        return 1


def handle_inference_command(args: argparse.Namespace, settings: Settings, logger) -> int:
    """Обработка команды inference.
    
    Args:
        args: Аргументы командной строки
        settings: Конфигурация приложения
        logger: Логгер
    
    Returns:
        Exit code
    """
    from app.inference.cli import InferenceCLI
    from app.inference.config import GenerationConfig
    
    logger.info("Запуск инференса...")
    
    logger.info(f"Модель: {args.base_model}")
    if args.adapter_path:
        logger.info(f"Адаптер: {args.adapter_path}")
    
    # Создание GenerationConfig из аргументов
    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
    )
    
    cli = InferenceCLI()
    
    return cli.run(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        prompt=args.prompt,
        interactive=args.interactive,
        input_file=args.input_file,
        output_file=args.output_file,
        config=config,
        prompt_template=args.prompt_template,
        device=args.device,
        load_in_8bit=args.load_in_8bit,
    )


def handle_export_command(args: argparse.Namespace, settings: Settings, logger) -> int:
    """Обработка команды export.

    Args:
        args: Аргументы командной строки
        settings: Конфигурация приложения
        logger: Логгер

    Returns:
        Exit code
    """
    from app.inference.export import ModelExporter

    logger.info("Экспорт модели...")
    logger.info(f"Базовая модель: {args.base_model}")
    logger.info(f"Адаптер: {args.adapter_path}")
    logger.info(f"Вывод: {args.output_path}")

    try:
        exporter = ModelExporter()
        exporter.merge_and_save(
            base_model_name=args.base_model,
            adapter_path=args.adapter_path,
            output_path=args.output_path,
            save_tokenizer=True,
        )
        logger.info("Экспорт завершён успешно")
        return 0
    except Exception as e:
        logger.error(f"Ошибка экспорта: {e}")
        return 1


def handle_generate_dataset_command(args: argparse.Namespace, settings: Settings, logger) -> int:
    """Обработка команды generate-dataset.

    Args:
        args: Аргументы командной строки
        settings: Конфигурация приложения
        logger: Логгер

    Returns:
        Exit code
    """
    from app.data.generators.synthetic import SyntheticGenerator
    from app.data.generators.qa import QAGenerator

    logger.info("Генерация датасета...")
    logger.info(f"Тип генератора: {args.type}")
    logger.info(f"Вывод: {args.output}")
    logger.info(f"Количество: {args.count}")
    logger.info(f"Формат: {args.format}")
    if args.topics:
        logger.info(f"Темы: {', '.join(args.topics)}")
    if args.categories:
        logger.info(f"Категории: {', '.join(args.categories)}")
    if args.seed:
        logger.info(f"Seed: {args.seed}")

    try:
        if args.type == "synthetic":
            generator = SyntheticGenerator(topics=args.topics, seed=args.seed)
            generator.save(path=args.output, count=args.count, format=args.format)
            logger.info(f"Датасет сохранён в: {args.output}")
            return 0
        elif args.type == "qa":
            generator = QAGenerator(categories=args.categories, seed=args.seed)
            generator.save(path=args.output, count=args.count, format=args.format)
            logger.info(f"Датасет сохранён в: {args.output}")
            return 0
        else:
            logger.error(f"Неподдерживаемый тип генератора: {args.type}")
            return 1
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        return 1


def main() -> int:
    """Главная функция CLI.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Для команды generate-dataset не требуется полная конфигурация
        if args.command == "generate-dataset":
            # Используем базовую настройку логирования с файлом
            logger = setup_logging(level="INFO", log_file="./logs/generator.log")
            logger.debug(f"Команда: {args.command}")
            return handle_generate_dataset_command(args, None, logger)

        # Загрузка конфигурации для остальных команд
        env_file = Path(args.config)
        if not env_file.exists():
            print(f"Ошибка: файл конфигурации не найден: {args.config}", file=sys.stderr)
            return 1

        settings = Settings(_env_file=str(env_file))

        # Настройка логирования
        logger = setup_logging(
            level=settings.LOG_LEVEL,
            log_file=settings.LOG_FILE
        )

        logger.debug(f"Загружена конфигурация из: {args.config}")
        logger.debug(f"Команда: {args.command}")

        # Маршрутизация команд
        if args.command == "train":
            return handle_train_command(args, settings, logger)
        elif args.command == "inference":
            return handle_inference_command(args, settings, logger)
        elif args.command == "export":
            return handle_export_command(args, settings, logger)
        else:
            parser.print_help()
            return 1

    except ConfigurationError as e:
        print(f"Ошибка конфигурации: {e}", file=sys.stderr)
        return 1
    except FineTuningError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Неожиданная ошибка: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

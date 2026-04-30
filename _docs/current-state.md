# Текущее состояние

## Статус проекта: Bootstrap (Спринт 00)

**Последнее обновление:** 2026-04-30

## Что реализовано

### Инфраструктура ✅

- Структура каталогов проекта
- Корневые файлы:
  - `.gitignore` — игнорирование временных файлов
  - `README.md` — описание проекта
  - `.env.example` — шаблон конфигурации (русский язык)
  - `requirements.txt` — зависимости
  - `pyproject.toml` — конфигурация Python проекта

### Документация ✅

- `_docs/README.md` — индекс документации
- `_docs/requirements.md` — требования (FR/NFR/CON)
- `_docs/architecture.md` — архитектура системы
- `_docs/configuration.md` — конфигурация через .env
- `_docs/training.md` — процесс обучения
- `_docs/inference.md` — инференс и генерация
- `_docs/testing.md` — стратегия тестирования
- `_docs/instructions.md` — правила разработки
- `_docs/project-structure.md` — структура каталогов
- `_docs/roadmap.md` — дорожная карта
- `_docs/current-state.md` — этот файл

### Доска задач ✅

- `_board/README.md` — описание процесса
- `_board/plan.md` — план спринтов
- `_board/process.md` — пошаговый процесс задачи
- `_board/sprints/00-bootstrap.md` — спринт 00

### Скелет кода ✅

- `app/` — пакеты с `__init__.py`
- `tests/` — пакеты с `__init__.py`
- `.gitkeep` файлы в data/, logs/, checkpoints/, outputs/

## Что не реализовано

### Core Infrastructure ❌

- `app/core/config.py` — Settings
- `app/core/logging_config.py` — логирование
- `app/core/exceptions.py` — исключения

### Models ❌

- `app/models/base.py` — загрузка моделей
- `app/models/lora.py` — LoRA конфигурация
- `app/models/registry.py` — реестр моделей

### Data ❌

- `app/data/loader.py` — загрузка данных
- `app/data/formatter.py` — форматирование
- `app/data/tokenizer.py` — токенизация

### Training ❌

- `app/training/trainer.py` — тренировочный цикл
- `app/training/config.py` — конфигурация
- `app/training/callbacks.py` — callbacks

### Inference ❌

- `app/inference/engine.py` — inference engine
- `app/inference/cli.py` — CLI для инференса

### Utils ❌

- `app/utils/device.py` — определение устройств
- `app/utils/memory.py` — мониторинг памяти
- `app/utils/file.py` — работа с файлами

### Тесты ❌

- `tests/unit/core/test_config.py`
- `tests/unit/models/test_base.py`
- `tests/unit/models/test_lora.py`
- `tests/unit/data/test_loader.py`
- `tests/unit/training/test_trainer.py`
- `tests/integration/test_training_pipeline.py`

### CLI ❌

- `app/__main__.py` — точка входа
- Команды: `train`, `inference`

## Готовность к следующему спринту

Спринт 00 (Bootstrap) завершён. Готов к началу Спринта 01 — реализации Core Infrastructure.

## Зависимости

- Python 3.10+ — требуется
- PyTorch — будет установлен через requirements.txt
- Transformers — будет установлен
- PEFT — будет установлен
- GPU — рекомендуется, но не обязательен для разработки

## Известные риски

1. **Модели HuggingFace** — требуется интернет для загрузки
2. **GPU память** — может потребоваться уменьшение batch size
3. **Совместимость** — разные версии transformers/peft могут иметь breaking changes

## Следующие шаги

1. Реализовать `app/core/config.py` и `app/core/logging_config.py`
2. Написать unit-тесты для core
3. Создать CLI скелет
4. Закрыть Спринт 01

## Метрики

| Метрика | Значение |
|---------|----------|
| Строк кода | 0 |
| Покрытие тестами | 0% |
| Документация | 100% (планирование) |
| Рабочих спринтов | 0 |

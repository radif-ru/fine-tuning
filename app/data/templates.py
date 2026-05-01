"""Шаблоны промптов для форматирования данных.

Поддерживаемые форматы: Alpaca, ShareGPT, Raw.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.exceptions import DataFormatError
from app.core.logging_config import get_logger

logger = get_logger("data.templates")


@dataclass
class PromptTemplate:
    """Шаблон для форматирования промптов.
    
    Attributes:
        name: Имя шаблона
        template: Строка шаблона с плейсхолдерами
        description: Описание шаблона
    """
    name: str
    template: str
    description: str = ""
    
    def format(self, **kwargs) -> str:
        """Форматировать шаблон с данными.
        
        Args:
            **kwargs: Значения для плейсхолдеров
        
        Returns:
            Отформатированная строка
        
        Raises:
            DataFormatError: Если не хватает плейсхолдеров
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise DataFormatError(
                f"Отсутствует плейсхолдер {e} в шаблоне {self.name}"
            )
    
    def get_placeholders(self) -> List[str]:
        """Получить список плейсхолдеров в шаблоне.
        
        Returns:
            Список имён плейсхолдеров
        """
        import re
        return re.findall(r'\{(\w+)\}', self.template)


# Предустановленные шаблоны

ALPACA_TEMPLATE = PromptTemplate(
    name="alpaca",
    description="Стандартный Alpaca формат с instruction/input/output",
    template="""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
)

ALPACA_NO_INPUT_TEMPLATE = PromptTemplate(
    name="alpaca_no_input",
    description="Alpaca формат без input поля",
    template="""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
)

SHAREGPT_TEMPLATE = PromptTemplate(
    name="sharegpt",
    description="ShareGPT формат для conversational данных",
    template="""{system}

{conversation}"""
)

RAW_TEMPLATE = PromptTemplate(
    name="raw",
    description="Сырые данные без форматирования",
    template="{text}"
)

CHAT_ML_TEMPLATE = PromptTemplate(
    name="chatml",
    description="ChatML формат с system/user/assistant ролями",
    template="""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
)


class TemplateRegistry:
    """Реестр шаблонов промптов.
    
    Хранит предустановленные и пользовательские шаблоны.
    """
    
    def __init__(self):
        """Инициализация с предустановленными шаблонами."""
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_defaults()
    
    def register(self, template: PromptTemplate) -> None:
        """Зарегистрировать шаблон.
        
        Args:
            template: Шаблон для регистрации
        """
        self._templates[template.name] = template
        logger.debug(f"Зарегистрирован шаблон: {template.name}")
    
    def get(self, name: str) -> PromptTemplate:
        """Получить шаблон по имени.
        
        Args:
            name: Имя шаблона
        
        Returns:
            Шаблон
        
        Raises:
            DataFormatError: Если шаблон не найден
        """
        if name not in self._templates:
            raise DataFormatError(f"Шаблон не найден: {name}")
        
        return self._templates[name]
    
    def list_templates(self) -> List[str]:
        """Получить список доступных шаблонов.
        
        Returns:
            Список имён шаблонов
        """
        return list(self._templates.keys())
    
    def has_template(self, name: str) -> bool:
        """Проверить существование шаблона.
        
        Args:
            name: Имя шаблона
        
        Returns:
            True если шаблон существует
        """
        return name in self._templates
    
    def _register_defaults(self) -> None:
        """Регистрация предустановленных шаблонов."""
        self.register(ALPACA_TEMPLATE)
        self.register(ALPACA_NO_INPUT_TEMPLATE)
        self.register(SHAREGPT_TEMPLATE)
        self.register(RAW_TEMPLATE)
        self.register(CHAT_ML_TEMPLATE)


# Глобальный реестр
_default_registry = TemplateRegistry()


def get_template(name: str) -> PromptTemplate:
    """Получить шаблон из глобального реестра.
    
    Args:
        name: Имя шаблона
    
    Returns:
        Шаблон
    """
    return _default_registry.get(name)


def format_alpaca(
    instruction: str,
    output: str,
    input_text: str = "",
    system_prompt: str = ""
) -> str:
    """Форматировать данные в Alpaca формат.
    
    Args:
        instruction: Инструкция
        output: Ответ
        input_text: Дополнительный контекст
        system_prompt: Системный промпт (игнорируется для Alpaca)
    
    Returns:
        Отформатированная строка
    """
    if input_text:
        return ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
    else:
        return ALPACA_NO_INPUT_TEMPLATE.format(
            instruction=instruction,
            output=output
        )


def format_sharegpt(
    conversations: List[Dict[str, str]],
    system_prompt: str = ""
) -> str:
    """Форматировать данные в ShareGPT формат.
    
    Args:
        conversations: Список сообщений [{"from": "human"/"gpt", "value": "..."}]
        system_prompt: Системный промпт
    
    Returns:
        Отформатированная строка
    """
    lines = []
    
    for msg in conversations:
        role = msg.get("from", "")
        content = msg.get("value", "")
        
        if role == "human" or role == "user":
            lines.append(f"User: {content}")
        elif role == "gpt" or role == "assistant":
            lines.append(f"Assistant: {content}")
    
    conversation_text = "\n".join(lines)
    
    return SHAREGPT_TEMPLATE.format(
        system=system_prompt,
        conversation=conversation_text
    )

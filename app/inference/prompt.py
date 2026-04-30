"""Prompt Builder для форматирования промптов.

Поддерживает различные шаблоны: alpaca, raw, chat.
"""

from typing import List, Optional

from app.core.logging_config import get_logger

logger = get_logger("inference.prompt")


class PromptBuilder:
    """Строитель промптов с поддержкой различных шаблонов.
    
    Templates:
    - alpaca: Стандартный Alpaca формат
    - raw: Сырой текст без форматирования
    - chat: Chat формат через apply_chat_template
    """
    
    # Шаблон Alpaca
    ALPACA_TEMPLATE = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )
    
    ALPACA_TEMPLATE_WITH_CONTEXT = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{context}\n\n"
        "### Response:\n"
    )
    
    def __init__(self, template_name: str = "alpaca"):
        """Инициализация PromptBuilder.
        
        Args:
            template_name: Имя шаблона ('alpaca', 'raw', 'chat')
        """
        self.template_name = template_name
        self._tokenizer = None
        
        if template_name not in ("alpaca", "raw", "chat"):
            logger.warning(f"Неизвестный шаблон '{template_name}', используется 'alpaca'")
            self.template_name = "alpaca"
        
        logger.debug(f"PromptBuilder инициализирован | template={self.template_name}")
    
    def build(self, instruction: str, context: str = "") -> str:
        """Построить промпт по шаблону.
        
        Args:
            instruction: Инструкция/вопрос
            context: Дополнительный контекст (опционально)
        
        Returns:
            Отформатированный промпт
        """
        if self.template_name == "alpaca":
            return self._build_alpaca(instruction, context)
        elif self.template_name == "raw":
            return self._build_raw(instruction, context)
        elif self.template_name == "chat":
            return self._build_chat(instruction, context)
        else:
            return self._build_alpaca(instruction, context)
    
    def _build_alpaca(self, instruction: str, context: str = "") -> str:
        """Построить Alpaca формат промпта.
        
        Args:
            instruction: Инструкция
            context: Контекст
        
        Returns:
            Отформатированный Alpaca промпт
        """
        if context:
            return self.ALPACA_TEMPLATE_WITH_CONTEXT.format(
                instruction=instruction,
                context=context
            )
        else:
            return self.ALPACA_TEMPLATE.format(instruction=instruction)
    
    def _build_raw(self, instruction: str, context: str = "") -> str:
        """Построить raw формат (просто текст).
        
        Args:
            instruction: Текст
            context: Дополнительный контекст
        
        Returns:
            Объединённый текст
        """
        if context:
            return f"{context}\n\n{instruction}"
        return instruction
    
    def _build_chat(self, instruction: str, context: str = "") -> str:
        """Построить chat формат.
        
        Использует apply_chat_template если доступен.
        
        Args:
            instruction: Сообщение пользователя
            context: Системный контекст
        
        Returns:
            Отформатированный chat промпт
        """
        # Базовый chat формат (если нет tokenizer)
        if self._tokenizer is None:
            if context:
                return f"System: {context}\nUser: {instruction}\nAssistant:"
            return f"User: {instruction}\nAssistant:"
        
        # Если есть tokenizer с apply_chat_template
        try:
            messages = []
            if context:
                messages.append({"role": "system", "content": context})
            messages.append({"role": "user", "content": instruction})
            
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"apply_chat_template не сработал: {e}, используется fallback")
            if context:
                return f"System: {context}\nUser: {instruction}\nAssistant:"
            return f"User: {instruction}\nAssistant:"
    
    def set_tokenizer(self, tokenizer) -> "PromptBuilder":
        """Установить токенизатор для chat шаблона.
        
        Args:
            tokenizer: Токенизатор HuggingFace
        
        Returns:
            self для chaining
        """
        self._tokenizer = tokenizer
        return self
    
    def build_conversation(
        self,
        messages: List[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """Построить промпт из истории сообщений.
        
        Args:
            messages: Список сообщений [{"role": "user", "content": "..."}, ...]
            system_prompt: Системный промпт
        
        Returns:
            Отформатированный промпт
        """
        if self.template_name == "chat" and self._tokenizer:
            try:
                all_messages = []
                if system_prompt:
                    all_messages.append({"role": "system", "content": system_prompt})
                all_messages.extend(messages)
                
                return self._tokenizer.apply_chat_template(
                    all_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"apply_chat_template не сработал: {e}")
        
        # Fallback для других шаблонов
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        
        parts.append("Assistant:")
        return "\n".join(parts)


# Фабричные функции для удобства

def create_alpaca_prompt(instruction: str, context: str = "") -> str:
    """Создать Alpaca промпт.
    
    Args:
        instruction: Инструкция
        context: Контекст
    
    Returns:
        Отформатированный промпт
    """
    builder = PromptBuilder("alpaca")
    return builder.build(instruction, context)


def create_raw_prompt(instruction: str, context: str = "") -> str:
    """Создать raw промпт.
    
    Args:
        instruction: Текст
        context: Контекст
    
    Returns:
        Объединённый текст
    """
    builder = PromptBuilder("raw")
    return builder.build(instruction, context)


def create_chat_prompt(instruction: str, context: str = "", tokenizer=None) -> str:
    """Создать chat промпт.
    
    Args:
        instruction: Сообщение
        context: Системный контекст
        tokenizer: Токенизатор для apply_chat_template
    
    Returns:
        Отформатированный промпт
    """
    builder = PromptBuilder("chat")
    if tokenizer:
        builder.set_tokenizer(tokenizer)
    return builder.build(instruction, context)

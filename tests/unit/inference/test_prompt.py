"""Тесты для модуля inference.prompt."""

import pytest

from app.inference.prompt import (
    PromptBuilder,
    create_alpaca_prompt,
    create_raw_prompt,
    create_chat_prompt,
)


class TestPromptBuilder:
    """Тесты для PromptBuilder."""
    
    def test_alpaca_template_basic(self):
        """Проверка базового Alpaca шаблона."""
        builder = PromptBuilder("alpaca")
        prompt = builder.build("What is LoRA?")
        
        assert "Below is an instruction" in prompt
        assert "### Instruction:" in prompt
        assert "What is LoRA?" in prompt
        assert "### Response:" in prompt
        assert "### Input:" not in prompt
    
    def test_alpaca_template_with_context(self):
        """Проверка Alpaca шаблона с контекстом."""
        builder = PromptBuilder("alpaca")
        prompt = builder.build(
            instruction="Explain this concept.",
            context="We are discussing machine learning."
        )
        
        assert "Below is an instruction" in prompt
        assert "paired with an input" in prompt
        assert "### Instruction:" in prompt
        assert "Explain this concept." in prompt
        assert "### Input:" in prompt
        assert "We are discussing machine learning." in prompt
        assert "### Response:" in prompt
    
    def test_raw_template(self):
        """Проверка raw шаблона."""
        builder = PromptBuilder("raw")
        prompt = builder.build("Simple text")
        
        assert prompt == "Simple text"
    
    def test_raw_template_with_context(self):
        """Проверка raw шаблона с контекстом."""
        builder = PromptBuilder("raw")
        prompt = builder.build(
            instruction="Question",
            context="Context"
        )
        
        assert "Context" in prompt
        assert "Question" in prompt
    
    def test_chat_template_without_tokenizer(self):
        """Проверка chat шаблона без токенизатора."""
        builder = PromptBuilder("chat")
        prompt = builder.build("Hello")
        
        assert "User: Hello" in prompt
        assert "Assistant:" in prompt
    
    def test_chat_template_with_context(self):
        """Проверка chat шаблона с контекстом."""
        builder = PromptBuilder("chat")
        prompt = builder.build(
            instruction="Hello",
            context="You are a helpful assistant."
        )
        
        assert "System: You are a helpful assistant." in prompt
        assert "User: Hello" in prompt
        assert "Assistant:" in prompt
    
    def test_unknown_template_fallback(self):
        """Проверка fallback на alpaca для неизвестного шаблона."""
        builder = PromptBuilder("unknown_template")
        
        assert builder.template_name == "alpaca"
    
    def test_set_tokenizer(self):
        """Проверка установки токенизатора."""
        builder = PromptBuilder("chat")
        mock_tokenizer = {"name": "mock"}  # Простой mock
        
        result = builder.set_tokenizer(mock_tokenizer)
        
        assert result is builder  # Проверка chaining
        assert builder._tokenizer == mock_tokenizer
    
    def test_build_conversation_basic(self):
        """Проверка построения разговора."""
        builder = PromptBuilder("raw")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        
        prompt = builder.build_conversation(messages)
        
        assert "User: Hello" in prompt
        assert "Assistant: Hi there" in prompt
        assert "User: How are you?" in prompt
        assert "Assistant:" in prompt  # Финальный prompt
    
    def test_build_conversation_with_system(self):
        """Проверка построения разговора с системным промптом."""
        builder = PromptBuilder("raw")
        messages = [{"role": "user", "content": "Hello"}]
        
        prompt = builder.build_conversation(
            messages,
            system_prompt="You are helpful"
        )
        
        assert "System: You are helpful" in prompt
        assert "User: Hello" in prompt


class TestFactoryFunctions:
    """Тесты фабричных функций."""
    
    def test_create_alpaca_prompt(self):
        """Проверка create_alpaca_prompt."""
        prompt = create_alpaca_prompt("What is AI?")
        
        assert "Below is an instruction" in prompt
        assert "What is AI?" in prompt
    
    def test_create_raw_prompt(self):
        """Проверка create_raw_prompt."""
        prompt = create_raw_prompt("Just text")
        
        assert prompt == "Just text"
    
    def test_create_chat_prompt(self):
        """Проверка create_chat_prompt."""
        prompt = create_chat_prompt("Hi")
        
        assert "User: Hi" in prompt
        assert "Assistant:" in prompt
    
    def test_create_chat_prompt_with_tokenizer(self):
        """Проверка create_chat_prompt с токенизатором."""
        mock_tokenizer = type("MockTokenizer", (), {
            "apply_chat_template": lambda self, msgs, **kwargs: "<chat>formatted</chat>"
        })()
        
        prompt = create_chat_prompt("Hi", tokenizer=mock_tokenizer)
        
        assert prompt == "<chat>formatted</chat>"


class TestPromptBuilderEdgeCases:
    """Тесты граничных случаев."""
    
    def test_empty_instruction(self):
        """Проверка пустой инструкции."""
        builder = PromptBuilder("alpaca")
        prompt = builder.build("")
        
        assert "### Instruction:" in prompt
        assert "### Response:" in prompt
    
    def test_empty_context(self):
        """Проверка пустого контекста (не должен добавлять Input секцию)."""
        builder = PromptBuilder("alpaca")
        prompt = builder.build("Question", context="")
        
        # С пустым контекстом не должно быть Input секции
        assert "### Input:" not in prompt
    
    def test_special_characters(self):
        """Проверка специальных символов."""
        builder = PromptBuilder("raw")
        prompt = builder.build("Text with <tags> & \"quotes\"")
        
        assert "<tags>" in prompt
        assert "\"quotes\"" in prompt
    
    def test_multiline_instruction(self):
        """Проверка многострочной инструкции."""
        builder = PromptBuilder("raw")
        prompt = builder.build("Line 1\nLine 2\nLine 3")
        
        assert "Line 1" in prompt
        assert "Line 2" in prompt
        assert "Line 3" in prompt

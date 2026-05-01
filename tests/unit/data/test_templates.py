"""Тесты для templates."""

import pytest

from app.core.exceptions import DataFormatError
from app.data.templates import (
    PromptTemplate,
    TemplateRegistry,
    format_alpaca,
    format_sharegpt,
    get_template,
)


class TestPromptTemplate:
    """Тесты для PromptTemplate."""
    
    def test_creation(self):
        """Проверка создания."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!")
        
        assert template.name == "test"
        assert template.template == "Hello {name}!"
    
    def test_format(self):
        """Проверка форматирования."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!")
        
        result = template.format(name="World")
        assert result == "Hello World!"
    
    def test_format_missing_placeholder(self):
        """Проверка форматирования с отсутствующим плейсхолдером."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!")
        
        with pytest.raises(DataFormatError):
            template.format()
    
    def test_get_placeholders(self):
        """Проверка получения плейсхолдеров."""
        template = PromptTemplate(
            name="test",
            template="{greeting} {name}!")
        
        placeholders = template.get_placeholders()
        assert set(placeholders) == {"greeting", "name"}


class TestTemplateRegistry:
    """Тесты для TemplateRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Фикстура для TemplateRegistry."""
        return TemplateRegistry()
    
    def test_has_default_templates(self, registry):
        """Проверка наличия дефолтных шаблонов."""
        templates = registry.list_templates()
        
        assert "alpaca" in templates
        assert "alpaca_no_input" in templates
        assert "sharegpt" in templates
        assert "raw" in templates
    
    def test_get_template(self, registry):
        """Проверка получения шаблона."""
        template = registry.get("alpaca")
        
        assert template.name == "alpaca"
        assert "### Инструкция:" in template.template
    
    def test_get_nonexistent(self, registry):
        """Проверка получения несуществующего шаблона."""
        with pytest.raises(DataFormatError):
            registry.get("nonexistent")
    
    def test_register(self, registry):
        """Проверка регистрации шаблона."""
        template = PromptTemplate(
            name="custom",
            template="Custom {value}")
        
        registry.register(template)
        
        assert "custom" in registry.list_templates()


class TestFormatFunctions:
    """Тесты для функций форматирования."""
    
    def test_format_alpaca_with_input(self):
        """Проверка форматирования Alpaca с input."""
        result = format_alpaca(
            instruction="Напиши привет",
            input_text="на русском",
            output="Привет!"
        )
        
        assert "### Инструкция:" in result
        assert "Напиши привет" in result
        assert "### Входные данные:" in result
        assert "### Ответ:" in result
    
    def test_format_alpaca_without_input(self):
        """Проверка форматирования Alpaca без input."""
        result = format_alpaca(
            instruction="Напиши привет",
            output="Привет!"
        )
        
        assert "### Инструкция:" in result
        assert "### Ответ:" in result
        assert "### Входные данные:" not in result
    
    def test_format_sharegpt(self):
        """Проверка форматирования ShareGPT."""
        conversations = [
            {"from": "human", "value": "Привет"},
            {"from": "gpt", "value": "Здравствуй!"}
        ]
        
        result = format_sharegpt(conversations)
        
        assert "User: Привет" in result
        assert "Assistant: Здравствуй!" in result


class TestGetTemplate:
    """Тесты для get_template."""
    
    def test_get_alpaca(self):
        """Проверка получения alpaca шаблона."""
        template = get_template("alpaca")
        
        assert template.name == "alpaca"

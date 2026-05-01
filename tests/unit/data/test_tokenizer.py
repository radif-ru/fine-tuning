"""Тесты для TokenizerWrapper."""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from app.data.tokenizer import TokenizerWrapper


class TestTokenizerWrapper:
    """Тесты для TokenizerWrapper."""
    
    @pytest.fixture
    def tokenizer(self):
        """Фикстура для токенизатора."""
        return AutoTokenizer.from_pretrained("gpt2")
    
    @pytest.fixture
    def wrapper(self, tokenizer):
        """Фикстура для TokenizerWrapper."""
        return TokenizerWrapper(
            tokenizer=tokenizer,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    
    @pytest.fixture
    def text_dataset(self):
        """Датасет с текстом."""
        return Dataset.from_list([
            {"text": "Hello world"},
            {"text": "Another example"}
        ])
    
    def test_init(self, wrapper):
        """Проверка инициализации."""
        assert wrapper.max_length == 128
        assert wrapper.truncation is True
    
    def test_tokenize(self, wrapper):
        """Проверка токенизации."""
        result = wrapper.tokenize("Hello world")
        
        assert "input_ids" in result
        assert "attention_mask" in result
    
    def test_tokenize_dataset(self, wrapper, text_dataset):
        """Проверка токенизации датасета."""
        result = wrapper.tokenize_dataset(text_dataset)
        
        assert "input_ids" in result.column_names
        assert "attention_mask" in result.column_names
    
    def test_prepare_for_training(self, wrapper, text_dataset):
        """Проверка подготовки для обучения."""
        result = wrapper.prepare_for_training(text_dataset)
        
        assert "input_ids" in result.column_names
        assert "labels" in result.column_names
        # Labels = input_ids для causal LM
        assert result[0]["labels"] == result[0]["input_ids"]
    
    def test_decode(self, wrapper):
        """Проверка декодирования."""
        # Токенизируем
        tokens = wrapper.tokenize("Hello world")
        
        # Декодируем
        text = wrapper.decode(tokens["input_ids"])
        
        assert "Hello" in text or "world" in text
    
    def test_get_vocab_size(self, wrapper):
        """Проверка получения размера словаря."""
        vocab_size = wrapper.get_vocab_size()
        
        assert vocab_size > 0
    
    def test_pad_token_id(self, wrapper):
        """Проверка pad_token_id."""
        assert wrapper.pad_token_id is not None
    
    def test_eos_token_id(self, wrapper):
        """Проверка eos_token_id."""
        assert wrapper.eos_token_id is not None

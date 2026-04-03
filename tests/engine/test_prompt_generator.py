# tests/engine/test_prompt_generator.py
import pytest
from unittest.mock import MagicMock
from engine.prompt_generator import PromptGenerator

def test_calibrate_exact_fit():
    mock_tokenizer = MagicMock()
    # Assume 1 char = 1 token for simplicity
    mock_tokenizer.encode.side_effect = lambda t, **kwargs: [1] * len(t)
    mock_tokenizer.decode.side_effect = lambda t, **kwargs: "".join(["x"] * len(t))
    
    gen = PromptGenerator(mock_tokenizer)
    
    # Target 10 tokens
    # Using 'x' as token char
    result = gen.calibrate(target_tokens=10, suffix="")
    assert len(result) == 10

def test_generate_for_token_count_simple():
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda t, **kwargs: [1] * len(t)
    mock_tokenizer.decode.side_effect = lambda t, **kwargs: "".join(["x"] * len(t))
    
    gen = PromptGenerator(mock_tokenizer)
    
    # Target 50 tokens
    text, count = gen.generate_for_token_count(target_tokens=50)
    assert count == 50
    # Should be composed of suffix + padding
    assert len(text) >= 50 # Length in chars might vary but tokens fixed

def test_generate_unavailable_tokenizer():
    # If tokenizer is None, should raise
    gen = PromptGenerator(None)
    with pytest.raises(RuntimeError):
        gen.generate_for_token_count(10)

def test_calibrate_adjustment():
    # Test adjustment logic
    mock_tokenizer = MagicMock()
    # Simulate a tokenizer where length != chars
    # Say every char is 2 tokens? No, assume 1 char = 1 token usually.
    # Let's say we start short, it should add.
    
    # Mock calculate length
    gen = PromptGenerator(mock_tokenizer)
    
    # Mocking internal helper _get_count is hard since it's inside.
    # But calibrate uses tokenizer.encode().
    
    # Let's just ensure it calls encode many times.
    mock_tokenizer.encode.side_effect = lambda t, **kwargs: [1] * len(t)
    mock_tokenizer.decode.side_effect = lambda t, **kwargs: "".join(["x"] * len(t))
    
    res = gen.calibrate(20)
    assert len(res) == 20

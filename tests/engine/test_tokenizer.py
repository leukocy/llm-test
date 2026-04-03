# tests/engine/test_tokenizer.py
import pytest
from unittest.mock import MagicMock, patch
from engine.models import TestConfig
from engine.tokenizer import TokenizerManager, get_cached_tokenizer

@pytest.fixture
def mock_transformers():
    # Mock the module import itself
    mock_module = MagicMock()
    mock_tokenizer = MagicMock()
    mock_module.AutoTokenizer = mock_tokenizer
    # Mock name_or_path for instances
    mock_instance = MagicMock()
    mock_instance.name_or_path = "gpt2"
    mock_tokenizer.from_pretrained.return_value = mock_instance
    # Reset global cache
    import engine.tokenizer
    engine.tokenizer._AutoTokenizer = None
    engine.tokenizer.get_cached_tokenizer.cache_clear()
    
    with patch.dict("sys.modules", {"transformers": mock_module}):
        yield mock_tokenizer

def test_tokenizer_manager_auto(mock_transformers):
    # Test Auto-Inferred
    config = TestConfig(
        api_base_url="",
        model_id="Qwen2.5-7B-Instruct", # Should map to Qwen/Qwen2.5-7B-Instruct
        api_key="",
        tokenizer_option="Auto",
    )
    
    # We need to mock HF_MODEL_MAPPING or ensure config uses a model in it.
    # Assuming Qwen2.5-7B-Instruct matches something or fallback.
    
    # Let's mock HF_MODEL_MAPPING via patch
    with patch("engine.tokenizer.HF_MODEL_MAPPING", {"qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct"}):
        tok = TokenizerManager.get_tokenizer(config)
        assert tok is not None
        mock_transformers.from_pretrained.assert_called()

def test_tokenizer_manager_explicit(mock_transformers):
    config = TestConfig(
        api_base_url="",
        model_id="whatever",
        api_key="",
        tokenizer_option="HuggingFace Tokenizer",
        hf_tokenizer_model_id="deepseek-ai/deepseek-coder-33b-instruct"
    )
    
    tok = TokenizerManager.get_tokenizer(config)
    mock_transformers.from_pretrained.assert_called_with(
        "deepseek-ai/deepseek-coder-33b-instruct",
        trust_remote_code=True
    )
    # Note: local_files_only=True is tried first in Loop, then Fallback.
    # My mock setup is simple, so checking arguments exactly is tricky if multiple calls happen.
    # But it should call from_pretrained at least once.

def test_fallback_gpt2(mock_transformers):
    # If explicit fails, fallback to GPT2?
    # Or strict error?
    # Old logic: Fallback to GPT2 if option is Auto/Fallback.
    
    config = TestConfig(
        api_base_url="",
        model_id="unknown-model",
        api_key="",
        tokenizer_option="Auto"
    )
    
    # Force failure for inference?
    # Unknown model -> No inference -> fallback to GPT2.
    
    tok = TokenizerManager.get_tokenizer(config)
    assert tok is not None
    # Provide arg 'gpt2'
    assert mock_transformers.from_pretrained.call_args[0][0] == "gpt2"

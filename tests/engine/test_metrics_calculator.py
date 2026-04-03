# tests/engine/test_metrics_calculator.py
import pytest
from unittest.mock import MagicMock
from engine.metrics_calculator import calculate_request_metrics, calculate_tokens, extract_cache_hit_tokens, empty_metrics
from engine.models import MetricsSnapshot

def test_calculate_request_metrics():
    # 0s: start
    # 1s: first token (TTFT = 1.0)
    # 5s: end (Total = 5.0, Decode = 4.0)
    # Tokens: 101. (Decode tokens = 100).
    # TPS = 100 / 4.0 = 25.0
    start_time = 1000.0
    first_token_time = 1001.0
    end_time = 1005.0
    completion_tokens = 101
    
    m = calculate_request_metrics(start_time, first_token_time, end_time, completion_tokens)
    
    assert m.ttft == 1.0
    assert m.total_time == 5.0
    assert m.decode_time == 4.0
    assert m.tps == 25.25
    assert m.tpot == 0.04  # 1/25

def test_calculate_request_metrics_zero_decode():
    # Should handle division by zero
    start_time = 1000.0
    first_token_time = 1001.0
    end_time = 1001.0
    completion_tokens = 1
    
    m = calculate_request_metrics(start_time, first_token_time, end_time, completion_tokens)
    assert m.tps == 0.0
    assert m.tpot == 0.0

def test_calculate_tokens():
    mock_tokenizer = MagicMock()
    # Mock encode returning list of IDs
    # Handle both signatures (transfomers vs tiktoken)
    mock_tokenizer.encode.side_effect = lambda t, **kwargs: [1] * len(t.split())
    # Ensure it looks like transformers to hit that path, or delete encode_plus to hit tiktoken path
    # Let's make it look like transformers

    
    prompt = "hello user"
    response = "hello world"
    usage_info = {"prompt_tokens": 10, "completion_tokens": 20}
    
    # Method 1: usage_info
    prefill, decode, method = calculate_tokens(prompt, response, usage_info, mock_tokenizer)
    assert prefill == 10
    assert decode == 20
    assert method == "api_usage"
    
    # Method 2: tokenizer
    prefill, decode, method = calculate_tokens(prompt, response, None, mock_tokenizer)
    assert prefill == 2 # hello user
    assert decode == 2 # hello world
    assert "tokenizer" in method

    
    # Method 3: estimate
    prefill, decode, method = calculate_tokens(prompt, response, None, None)
    # 10 chars / 4 = 2.5 -> 2
    # 11 chars / 4 = 2.75 -> 2
    assert prefill > 0
    assert decode > 0
    assert method == "estimated"

def test_extract_cache_hit_tokens_openai():
    usage = {
        "prompt_tokens_details": {
            "cached_tokens": 123
        }
    }
    assert extract_cache_hit_tokens(usage) == 123

def test_extract_cache_hit_tokens_anthropic():
    usage = {
        "cache_read_input_tokens": 456
    }
    assert extract_cache_hit_tokens(usage) == 456

def test_extract_cache_hit_tokens_deepseek():
    usage = {
        "prompt_cache_hit_tokens": 789
    }
    assert extract_cache_hit_tokens(usage) == 789

def test_empty_metrics():
    m = empty_metrics()
    assert isinstance(m, MetricsSnapshot)
    assert m.tps == 0.0

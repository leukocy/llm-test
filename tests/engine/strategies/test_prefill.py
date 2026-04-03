# tests/engine/strategies/test_prefill.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from engine.strategies.prefill import PrefillStrategy
from engine.models import TestConfig, TestResult

@pytest.fixture
def mock_event_bus():
    mock = MagicMock()
    mock.is_stop_requested.return_value = False
    return mock

@pytest.fixture
def strategy(mock_event_bus):
    s = PrefillStrategy(mock_event_bus)
    return s

def test_param_schema(strategy):
    schema = strategy.param_schema()
    assert "input_tokens_list" in schema
    assert "rounds" in schema
    assert "max_tokens" in schema

def test_calculate_total_requests(strategy):
    params = {
        "input_tokens_list": [10, 20],
        "rounds": 5
    }
    # 2 levels * 5 rounds = 10
    total = strategy.calculate_total_requests(params)
    assert total == 10

@pytest.mark.asyncio
async def test_execute(strategy, mock_event_bus):
    config = TestConfig(api_base_url="", model_id="", api_key="")
    params = {
        "input_tokens_list": [10],
        "rounds": 1,
        "max_tokens": 5
    }
    
    # Mock dependencies
    mock_provider = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pg = MagicMock()
    mock_pg.generate_for_token_count.return_value = ("prompt_10", 10)
    
    # Use real ConcurrencyEngine or Mock it?
    # Prefill usually runs with concurrency=1? Or configurable? 
    # Usually strictly sequential (concurrency=1) to measure pure latency.
    # But some might want concurrent prefill.
    # Assuming concurrency=1 for standard prefill test.
    
    with patch("engine.strategies.prefill.ConcurrencyEngine") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        mock_engine.run_batch = AsyncMock(return_value=[
            TestResult(session_id=1, ttft=0.05, prefill_tokens=10)
        ])
        
        results = await strategy.execute(
            config, params, mock_provider, mock_tokenizer, mock_pg
        )
        
        assert len(results) == 1
        assert results[0].prefill_tokens == 10
        
        # Verify prompt generation called
        mock_pg.generate_for_token_count.assert_called_with(10)

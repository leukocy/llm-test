# tests/engine/strategies/test_concurrency.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from engine.strategies.concurrency import ConcurrencyStrategy
from engine.models import TestConfig, TestResult

@pytest.fixture
def mock_event_bus():
    mock = MagicMock()
    mock.is_stop_requested.return_value = False
    return mock

@pytest.fixture
def strategy(mock_event_bus):
    s = ConcurrencyStrategy(mock_event_bus)
    return s

def test_param_schema(strategy):
    schema = strategy.param_schema()
    assert "concurrencies" in schema
    assert "rounds" in schema
    assert "max_tokens" in schema
    assert "prompt" in schema

def test_calculate_total_requests(strategy):
    params = {
        "concurrencies": [1, 2, 4],
        "rounds": 3,
        "max_tokens": 10
    }
    # 1*3 + 2*3 + 4*3 = 3 + 6 + 12 = 21 requests
    total = strategy.calculate_total_requests(params)
    assert total == 21

@pytest.mark.asyncio
async def test_execute(strategy, mock_event_bus):
    config = TestConfig(api_base_url="", model_id="", api_key="")
    params = {
        "concurrencies": [2],
        "rounds": 1,
        "max_tokens": 100,
        "prompt": "test"
    }
    
    mock_provider = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pg = MagicMock()
    mock_pg.generate_for_token_count.return_value = ("prompt_text", 10)
    
    # Mock ConcurrencyEngine.run_batch
    # We need to mock the ConcurrencyEngine implicitly created or injected?
    # Strategy creates it inside. If we want to mock execution, we patch 'engine.strategies.concurrency.ConcurrencyEngine'.
    
    with patch("engine.strategies.concurrency.ConcurrencyEngine") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        mock_engine.run_batch = AsyncMock(return_value=[
            TestResult(session_id=1, ttft=0.1, tps=100.0)
        ])
        
        results = await strategy.execute(
            config, params, mock_provider, mock_tokenizer, mock_pg
        )
        
        assert len(results) == 1
        assert results[0].ttft == 0.1
        
        mock_engine.run_batch.assert_called_once()
        # Verify arguments passed to run_batch
        call_kwargs = mock_engine.run_batch.call_args[1]
        assert call_kwargs["concurrency"] == 2
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["prompts"] == "test"

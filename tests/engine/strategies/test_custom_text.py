
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from engine.strategies.custom_text import CustomTextStrategy
from engine.models import TestConfig, TestResult

@pytest.fixture
def mock_event_bus():
    mock = MagicMock()
    mock.is_stop_requested.return_value = False
    return mock

@pytest.fixture
def strategy(mock_event_bus):
    s = CustomTextStrategy(mock_event_bus)
    return s

def test_calculate_total_requests(strategy):
    params = {
        "concurrencies": [1, 2],
        "rounds_per_level": 3
    }
    # 1*3 + 2*3 = 3+6 = 9
    total = strategy.calculate_total_requests(params)
    assert total == 9

@pytest.mark.asyncio
async def test_execute(strategy, mock_event_bus):
    config = TestConfig(api_base_url="", model_id="", api_key="")
    params = {
        "concurrencies": [2],
        "rounds_per_level": 1,
        "base_prompt": "Hello",
        "max_tokens": 10
    }
    
    mock_provider = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pg = MagicMock()
    
    with patch("engine.strategies.custom_text.ConcurrencyEngine") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        
        # Concurrency 2, Rounds 1 -> Total 2 requests
        results = [TestResult(session_id=1, ttft=0.5, prefill_tokens=5, decode_tokens=10),
                   TestResult(session_id=2, ttft=0.5, prefill_tokens=5, decode_tokens=10)]
        
        mock_engine.run_continuous = AsyncMock(return_value=results)
        
        results = await strategy.execute(
            config, params, mock_provider, mock_tokenizer, mock_pg
        )
        
        assert len(results) == 2
        assert mock_engine.run_continuous.call_count == 1
        
        # Verify call args
        args, kwargs = mock_engine.run_continuous.call_args
        assert kwargs["concurrency"] == 2
        assert kwargs["total_requests"] == 2
        assert kwargs["prompt_func_or_str"] == "Hello"
        
        assert results[0].test_type == "custom_text"


import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from engine.strategies.long_context import LongContextStrategy
from engine.models import TestConfig, TestResult

@pytest.fixture
def mock_event_bus():
    mock = MagicMock()
    mock.is_stop_requested.return_value = False
    return mock

@pytest.fixture
def strategy(mock_event_bus):
    s = LongContextStrategy(mock_event_bus)
    return s

def test_calculate_total_requests(strategy):
    params = {
        "context_lengths": [1000, 2000],
        "rounds_per_level": 3
    }
    # 2 levels * 3 rounds = 6
    total = strategy.calculate_total_requests(params)
    assert total == 6

@pytest.mark.asyncio
async def test_execute(strategy, mock_event_bus):
    config = TestConfig(api_base_url="", model_id="", api_key="")
    params = {
        "context_lengths": [10, 100],
        "rounds_per_level": 2,
        "max_tokens": 5
    }
    
    # Mock dependencies
    mock_provider = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pg = MagicMock()
    
    # Mock prompt generation
    # For < 20, force_random is True
    # For >= 20, calibrate is called
    mock_pg.generate_for_token_count.return_value = ("prompt_10", 10)
    mock_pg.calibrate.return_value = "prompt_100"
    
    with patch("engine.strategies.long_context.ConcurrencyEngine") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        
        # Determine expected calls:
        # Level 1 (10 tokens): 2 rounds -> 2 calls to run_batch
        # Level 2 (100 tokens): 2 rounds -> 2 calls to run_batch
        # Total 4 calls.
        
        # Mock run_batch return values
        mock_engine.run_batch = AsyncMock(side_effect=[
            [TestResult(session_id=1, ttft=0.1, prefill_tokens=10)],
            [TestResult(session_id=2, ttft=0.1, prefill_tokens=10)],
            [TestResult(session_id=3, ttft=0.2, prefill_tokens=100)],
            [TestResult(session_id=4, ttft=0.2, prefill_tokens=100)]
        ])
        
        results = await strategy.execute(
            config, params, mock_provider, mock_tokenizer, mock_pg
        )
        
        assert len(results) == 4
        assert results[0].context_length_target == 10
        assert results[2].context_length_target == 100
        
        # Verify prompt generator calls
        # 2 rounds * 1 low-level = 2 calls to generate_for_token_count
        assert mock_pg.generate_for_token_count.call_count == 2
        mock_pg.generate_for_token_count.assert_called_with(10, force_random=True)
        
        # 2 rounds * 1 high-level = 2 calls to calibrate
        assert mock_pg.calibrate.call_count == 2
        mock_pg.calibrate.assert_called_with(100, suffix="")

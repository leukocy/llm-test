
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from engine.strategies.matrix import MatrixStrategy
from engine.models import TestConfig, TestResult

@pytest.fixture
def mock_event_bus():
    mock = MagicMock()
    mock.is_stop_requested.return_value = False
    return mock

@pytest.fixture
def strategy(mock_event_bus):
    s = MatrixStrategy(mock_event_bus)
    return s

def test_calculate_total_requests(strategy):
    # params: concurrencies=[1, 5], context_lengths=[1000, 2000], rounds=3
    # total requests = (1*3 + 5*3) * 2 = (3+15)*2 = 18 * 2 = 36 (?)
    # Logic in matrix.py:
    # total_reqs_per_context = sum(c * rounds for c in concurrencies)
    # total = total_reqs_per_context * len(context_lengths)
    # Sum: 1*3 + 5*3 = 18
    # Levels: 2 (1000, 2000)
    # Total: 18 * 2 = 36
    
    params = {
        "concurrencies": [1, 5],
        "context_lengths": [1000, 2000],
        "rounds": 3
    }
    total = strategy.calculate_total_requests(params)
    assert total == 36

@pytest.mark.asyncio
async def test_execute(strategy, mock_event_bus):
    config = TestConfig(api_base_url="", model_id="", api_key="")
    params = {
        "concurrencies": [1, 2],
        "context_lengths": [10, 20],
        "rounds": 2, # reqs per level = c * rounds -> 1*2=2, 2*2=4
    }
    
    mock_provider = MagicMock()
    mock_tokenizer = MagicMock()
    mock_pg = MagicMock()
    mock_pg.generate_for_token_count.return_value = ("prompt_10", 10)
    mock_pg.calibrate.return_value = "prompt_20"
    
    with patch("engine.strategies.matrix.ConcurrencyEngine") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        
        # Determine execution flow:
        # C=1:
        #   L=10: run_continuous(total=2). Returns 2 results.
        #   L=20: run_continuous(total=2). Returns 2 results.
        # C=2:
        #   L=10: run_continuous(total=4). Returns 4 results.
        #   L=20: run_continuous(total=4). Returns 4 results.
        
        # Total calls to run_continuous: 4
        
        # Mock run_continuous return values for each call
        
        results_c1_l10 = [TestResult(session_id=i, ttft=0.1, prefill_tokens=10) for i in range(2)]
        results_c1_l20 = [TestResult(session_id=i, ttft=0.1, prefill_tokens=20) for i in range(2)]
        results_c2_l10 = [TestResult(session_id=i, ttft=0.1, prefill_tokens=10) for i in range(4)]
        results_c2_l20 = [TestResult(session_id=i, ttft=0.1, prefill_tokens=20) for i in range(4)]
        
        mock_engine.run_continuous = AsyncMock(side_effect=[
            results_c1_l10, results_c1_l20,
            results_c2_l10, results_c2_l20
        ])
        
        results = await strategy.execute(
            config, params, mock_provider, mock_tokenizer, mock_pg
        )
        
        # Total results = 2 + 2 + 4 + 4 = 12
        assert len(results) == 12
        
        # Verify run_continuous calls count
        assert mock_engine.run_continuous.call_count == 4
        
        # Verify first call args
        args, kwargs = mock_engine.run_continuous.call_args_list[0]
        # concurrency=1, total_requests=2, max_tokens=default(100)
        assert kwargs["concurrency"] == 1
        assert kwargs["total_requests"] == 2
        
        # Verify strategy type injected
        assert results[0].test_type == "matrix"
        assert results[0].concurrency == 1
        assert results[0].context_length_target == 10

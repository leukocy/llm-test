
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from engine.strategies.segmented import SegmentedStrategy
from engine.models import TestConfig, TestResult

@pytest.fixture
def mock_event_bus():
    mock = MagicMock()
    mock.is_stop_requested.return_value = False
    return mock

@pytest.fixture
def strategy(mock_event_bus):
    s = SegmentedStrategy(mock_event_bus)
    return s

def test_calculate_total_requests(strategy):
    params = {
        "segment_levels": [1000, 2000],
        "requests_per_segment": 2,
        "rounds": 3
    }
    # 2 levels * 2 requests * 3 rounds = 12
    total = strategy.calculate_total_requests(params)
    assert total == 12

@pytest.mark.asyncio
async def test_execute_cumulative(strategy, mock_event_bus):
    config = TestConfig(api_base_url="", model_id="", api_key="")
    params = {
        "segment_levels": [10, 20],
        "requests_per_segment": 1,
        "cumulative_mode": True,
        "concurrency": 1,
        "per_round_unique": False,
        "rounds": 1
    }
    
    mock_provider = MagicMock()
    mock_tokenizer = MagicMock()
    # Mock encode/decode for truncation simulation
    mock_tokenizer.encode.return_value = [0]*100 # ample tokens
    mock_tokenizer.decode.side_effect = lambda t, **kwargs: "prompt_" + str(len(t))
    
    mock_pg = MagicMock()
    mock_pg.calibrate.return_value = "base_prompt_calibrated"
    
    with patch("engine.strategies.segmented.ConcurrencyEngine") as MockEngineCls:
        mock_engine = MockEngineCls.return_value
        
        # Determine execution:
        # A) Generate Base Prompts (1 time, max_segment=20)
        # B) Round 0:
        #    1. Segment 10: run_batch(concurrency=1)
        #    2. Segment 20: run_batch(concurrency=1)
        
        # Mock run_batch return values
        # Assume normal TTFT to populate baseline
        # Baseline TTFT ~ 0.1 for 10 tokens
        # Segment 20: 0.1 TTFT (implies cache hit if baseline speed is 10/0.1=100 tps)
        # Wait, if 20 tokens prefilled in 0.1s -> 200 tps > baseline(100) -> fast.
        # But our inference logic: expected = effective / baseline.
        # effective=20. baseline=100. expected=0.2s.
        # actual=0.1s. expected > actual * 2 (0.2 !> 0.2). Not hit.
        # Let's make it very fast. 0.01s. expected=0.2 > 0.02. Hit.
        
        results_seg10 = [TestResult(session_id=1, ttft=0.1, prefill_tokens=10, decode_tokens=5)]
        results_seg20 = [TestResult(session_id=2, ttft=0.01, prefill_tokens=20, decode_tokens=5)] # Fast!
        
        mock_engine.run_batch = AsyncMock(side_effect=[results_seg10, results_seg20])
         
        results = await strategy.execute(
            config, params, mock_provider, mock_tokenizer, mock_pg
        )
        
        assert len(results) == 2
        
        # Verify run_batch calls
        assert mock_engine.run_batch.call_count == 2
        
        # Verify result annotations
        assert results[0].test_type == "segmented"
        assert results[0].context_length_target == 10
        assert results[1].context_length_target == 20
        
        # Verify Cache Inference
        # First segment establishes baseline: 10 / 0.1 = 100 tokens/sec
        # Second segment: 20 tokens. Expected time = 20/100 = 0.2s.
        # Actual time = 0.01s.
        # 0.2 > 0.01 * 2 (=0.02). Condition met.
        # Uncached est = 0.01 * 100 = 1 token.
        # Inferred hit = 20 - 1 = 19.
        # Let's check results[1].cache_hit_tokens
        
        # Note: logic: uncached_est = res.ttft * baseline
        # inferred_hit = effective_prefill - uncached_est
        
        # res[1].ttft = 0.01
        # baseline = 100
        # uncached_est = 0.01 * 100 = 1.0
        # effective = 20
        # hit = 20 - 1 = 19
        
        # Result might be integer.
        assert results[1].cache_hit_tokens == 19
        assert results[1].prefill_speed > 0 # Should be recalculated

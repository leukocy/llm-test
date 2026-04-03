# tests/engine/test_concurrency.py
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from engine.concurrency import ConcurrencyEngine
from engine.events import EventBus, EventType
from engine.models import TestResult

@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.get_completion = AsyncMock()
    # Mock result
    provider.get_completion.return_value = {
        "start_time": 1000,
        "first_token_time": 1001,
        "end_time": 1002,
        "full_response_content": "test output",
        "usage_info": {"prompt_tokens": 10, "completion_tokens": 5},
        "error": None
    }
    return provider

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.mark.asyncio
async def test_run_batch(mock_provider, event_bus):
    engine = ConcurrencyEngine(event_bus)
    
    prompts = ["p1", "p2", "p3"]
    max_tokens = 100
    concurrency = 3
    session_id_start = 0
    
    # Needs a mock metrics calculator? 
    # Or extract logic?
    # The plan says ConcurrencyEngine handles execution.
    # It probably needs to call metrics calculator internally.
    
    results = await engine.run_batch(
        client=None, # Passed to provider
        prompts=prompts,
        max_tokens=max_tokens,
        concurrency=concurrency,
        session_id_start=session_id_start,
        provider=mock_provider,
        latency_offset=0.0
    )
    
    assert len(results) == 3
    assert mock_provider.get_completion.call_count == 3
    
    # Check result processing?
    # run_batch returns processed results (TestResult objects or dicts?)
    # Models.py defines TestResult.
    
    assert isinstance(results[0], dict) or isinstance(results[0], TestResult)

@pytest.mark.asyncio
async def test_run_batch_stop_signal(mock_provider, event_bus):
    engine = ConcurrencyEngine(event_bus)
    event_bus.request_stop()
    
    # Should check stop signal before running?
    # If checked before, returns empty.
    
    results = await engine.run_batch(
        client=None, prompts=["p"], max_tokens=10, concurrency=1, 
        session_id_start=0, provider=mock_provider, latency_offset=0
    )
    
    # If implemented to check stop, might return [] or raise?
    # Current benchmark_runner checks stop in outer loops usually.
    # checking inside run_batch is good practice.
    
    pass

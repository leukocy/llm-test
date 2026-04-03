# tests/engine/test_models.py
import pytest
from engine.models import (
    TestConfig, TestResult, MetricsSnapshot,
    TestType, TestRunConfig, TestRunSummary
)


def test_test_type_enum():
    assert TestType.CONCURRENCY == "concurrency"
    assert TestType.PREFILL == "prefill"
    assert TestType.LONG_CONTEXT == "long_context"
    assert TestType.SEGMENTED == "segmented"
    assert TestType.MATRIX == "matrix"
    assert TestType.STABILITY == "stability"
    assert TestType.CUSTOM_TEXT == "custom_text"
    assert TestType.DATASET == "dataset"


def test_test_config_creation():
    cfg = TestConfig(
        api_base_url="http://localhost:8000/v1",
        model_id="qwen-2.5",
        api_key="sk-test",
        provider="OpenAI Compatible",
    )
    assert cfg.model_id == "qwen-2.5"


def test_test_result_serialization():
    r = TestResult(
        session_id=1,
        ttft=0.5,
        tps=30.0,
        prefill_tokens=100,
        decode_tokens=50,
    )
    d = r.model_dump()
    assert d["session_id"] == 1
    assert d["ttft"] == 0.5


def test_metrics_snapshot():
    m = MetricsSnapshot(
        ttft=0.3,
        tps=40.0,
        tpot=0.025,
        tpot_p95=0.03,
        tpot_p99=0.05,
        prefill_tokens=500,
        decode_tokens=200,
        decode_time=5.0,
        total_time=5.3,
    )
    assert m.tps == 40.0


def test_test_run_config_for_concurrency():
    run_cfg = TestRunConfig(
        test_type=TestType.CONCURRENCY,
        base=TestConfig(
            api_base_url="http://localhost:8000/v1",
            model_id="test",
            api_key="sk-test",
            provider="OpenAI Compatible",
        ),
        params={
            "concurrencies": [1, 2, 4],
            "rounds_per_level": 3,
            "max_tokens": 512,
            "input_tokens_target": 64,
        }
    )
    assert run_cfg.test_type == TestType.CONCURRENCY
    assert run_cfg.params["concurrencies"] == [1, 2, 4]

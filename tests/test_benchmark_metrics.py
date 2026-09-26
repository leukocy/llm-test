import pytest

from core.benchmark.metrics import calculate_request_metrics, empty_metrics


def test_calculate_request_metrics_applies_latency_offset_and_inter_token_percentiles():
    result = calculate_request_metrics(
        start_time=100.0,
        first_token_time=100.6,
        end_time=102.0,
        completion_tokens=4,
        latency_offset=0.1,
        token_timestamps=[100.6, 101.0, 101.5, 102.0],
    )

    assert result.ttft == pytest.approx(0.5)
    assert result.tps == pytest.approx(3 / 1.4)
    assert result.tpot == pytest.approx(1.4 / 3)
    assert result.tps * result.tpot == pytest.approx(1)
    assert result.tpot_p95 == pytest.approx(0.5)
    assert result.tpot_p99 == pytest.approx(0.5)
    assert result.generation_time == pytest.approx(1.4)


def test_calculate_request_metrics_without_first_token_returns_zero_metrics():
    result = calculate_request_metrics(
        start_time=100.0,
        first_token_time=None,
        end_time=102.0,
        completion_tokens=0,
    )

    assert result.as_tuple() == (0, 0, 0, 0, 0, 0)


def test_skip_first_token_uses_matching_token_timestamps():
    result = calculate_request_metrics(
        start_time=100.0,
        first_token_time=100.4,
        end_time=101.6,
        completion_tokens=4,
        token_timestamps=[100.4, 100.8, 101.2, 101.6],
        skip_first_token=True,
    )

    assert result.generation_time == pytest.approx(0.8)
    assert result.tps == pytest.approx(2 / 0.8)
    assert result.tpot == pytest.approx(0.8 / 2)


def test_skip_first_token_falls_back_when_chunk_count_differs_from_token_count():
    result = calculate_request_metrics(
        start_time=100.0,
        first_token_time=100.4,
        end_time=101.6,
        completion_tokens=4,
        token_timestamps=[100.4, 101.6],
        skip_first_token=True,
    )

    assert result.generation_time == pytest.approx(1.2)
    assert result.tps == pytest.approx(3 / 1.2)
    assert result.tpot == pytest.approx(1.2 / 3)


def test_single_output_token_has_no_decode_interval():
    result = calculate_request_metrics(
        start_time=100.0,
        first_token_time=100.4,
        end_time=100.5,
        completion_tokens=1,
        token_timestamps=[100.4],
    )

    assert result.ttft == pytest.approx(0.4)
    assert result.tps == 0
    assert result.tpot == 0


def test_empty_metrics_preserves_legacy_error_shape():
    metrics = empty_metrics()

    assert metrics["ttft"] == 0
    assert metrics["tps"] == 0
    assert metrics["error"] is None
    assert metrics["token_calc_method"] == "Error"

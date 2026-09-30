"""Phase estimates preserve clocks, batch units, calibration and cache evidence."""

import csv
import io
import json

import pytest

from core.benchmark.phase_observations import PHASE_CONTRACT
from server.extended_analytics import extended_observations
from server.history import HistoryError, SavedCsvHistory
from server.phase_analytics import PHASE_METRICS
from tests.server.test_extended_observations import batch, serializable
from tests.server.test_scenario_figures import platform


def phased_batch(count=2, *, origin=100.0, offset=0.1, elapsed=4.0):
    rows = batch(count, elapsed)
    for index, row in enumerate(rows):
        row.update(cache_hit_source="API", cache_hit_tokens=index * 20, api_prefill=100)
        row["extra_metrics"]["request_phase"] = {
            "version": PHASE_CONTRACT,
            "clock": "client_monotonic",
            "start_seconds": origin + index * 0.1,
            "first_token_seconds": origin + 0.5 + index * 0.1,
            "end_seconds": origin + 2 + index * 0.1,
            "latency_offset_seconds": offset,
        }
    return rows


def test_phase_windows_reproduce_formula_and_do_not_replace_wall_measurements():
    data = extended_observations(serializable(phased_batch()))
    expected = {
        "phase_input": 400,
        "phase_input_uncached_api": 360,
        "phase_output": 25,
        "phase_total": 120,
        "phase_qpm": 60,
        "system_output_wall": 10,
        "system_qpm": 30,
    }
    for metric, value in expected.items():
        assert data["metrics"][metric]["count"] == 1
        assert data["metrics"][metric]["median"] == pytest.approx(value)
    assert data["metrics"]["phase_input_uncached_inferred"]["count"] == 0
    assert data["phase"]["valid_batches"] == 1


def test_batches_have_equal_weight_and_clock_origin_does_not_change_rates():
    rows = phased_batch() + phased_batch(4, origin=1_000_000_000.0)
    data = extended_observations(serializable(rows))
    assert data["metrics"]["phase_input"]["count"] == 2
    assert data["metrics"]["phase_input"]["mean"] == pytest.approx((400 + 400 / 0.7) / 2)
    assert data["metrics"]["phase_qpm"]["count"] == 2


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wall_clock",
        "version",
        "reversed",
        "nan",
        "boolean",
        "mixed_offset",
        "beyond_wall",
        "duplicate_member",
        "partial_batch",
    ],
)
def test_incomplete_or_conflicting_evidence_never_uses_legacy_rates(fault):
    rows = phased_batch()
    phase = rows[0]["extra_metrics"]["request_phase"]
    if fault == "missing":
        del rows[0]["extra_metrics"]["request_phase"]
    elif fault == "wall_clock":
        phase["clock"] = "wall"
    elif fault == "version":
        phase["version"] = "future"
    elif fault == "reversed":
        phase["first_token_seconds"] = 99
    elif fault == "nan":
        phase["end_seconds"] = float("nan")
    elif fault == "boolean":
        phase["latency_offset_seconds"] = False
    elif fault == "mixed_offset":
        phase["latency_offset_seconds"] = 0.2
    elif fault == "beyond_wall":
        phase["end_seconds"] = 110
    elif fault == "duplicate_member":
        rows[1]["request_index"] = 0
    else:
        rows.pop()
    for row in rows:
        row.update(system_input_throughput=99999, system_output_throughput=99999, rps=99999)
    data = extended_observations(serializable(rows))
    assert all(data["metrics"][metric]["count"] == 0 for metric in PHASE_METRICS)


def test_first_token_missing_preserves_total_window_but_does_not_invent_decode_window():
    rows = phased_batch()
    rows[0]["extra_metrics"]["request_phase"]["first_token_seconds"] = None
    data = extended_observations(serializable(rows))
    assert data["phase"]["missing_first_token_batches"] == 1
    assert data["metrics"]["phase_qpm"]["median"] == pytest.approx(60)
    assert data["metrics"]["phase_input"]["count"] == 0
    assert data["metrics"]["phase_output"]["count"] == 0


def test_nonpositive_calibrated_windows_are_missing_without_epsilon_clamping():
    data = extended_observations(serializable(phased_batch(offset=3)))
    assert data["phase"]["nonpositive_windows"] == {"input": 1, "output": 0, "total": 1}
    assert data["metrics"]["phase_input"]["count"] == 0
    assert data["metrics"]["phase_qpm"]["count"] == 0
    assert data["metrics"]["phase_output"]["median"] == pytest.approx(25)


def test_failures_do_not_enter_success_phase_window_or_token_numerator():
    rows = phased_batch(elapsed=10)
    rows[1].update(error="timeout", decode_tokens=999, prefill_tokens=999)
    rows[1]["extra_metrics"].pop("request_phase")
    data = extended_observations(serializable(rows))
    assert data["metrics"]["phase_output"]["median"] == pytest.approx(20 / 1.5)
    assert data["metrics"]["phase_qpm"]["median"] == pytest.approx(60 / 1.9)
    assert data["metrics"]["system_qpm"]["median"] == 6
    rows[0]["error"] = "timeout"
    data = extended_observations(serializable(rows))
    assert data["phase"]["no_success_batches"] == 1
    assert data["metrics"]["system_qpm"]["median"] == 0
    assert data["metrics"]["phase_qpm"]["count"] == 0


def test_unknown_cache_is_not_zero_and_inference_is_separately_sampled():
    rows = phased_batch()
    rows[0]["cache_hit_source"] = "unknown"
    data = extended_observations(serializable(rows))
    assert data["metrics"]["phase_input"]["median"] == pytest.approx(400)
    assert data["metrics"]["phase_input_uncached_api"]["count"] == 0
    rows[0].update(
        cache_hit_source="TTFT_inferred", cache_hit_tokens=80, effective_prefill_tokens=100
    )
    data = extended_observations(serializable(rows))
    assert data["metrics"]["phase_input_uncached_api"]["count"] == 0
    assert data["metrics"]["phase_input_uncached_inferred"]["median"] == pytest.approx(200)
    rows[0]["cache_hit_tokens"] = 101
    data = extended_observations(serializable(rows))
    assert data["metrics"]["phase_input_uncached_inferred"]["count"] == 0


def test_authenticated_api_reports_figures_and_csv_preserve_phase_evidence(tmp_path):
    rows = phased_batch()
    rows[0]["extra_metrics"]["request_phase"]["private"] = "NEVER EXPORTED"
    client, _, job, headers = platform(tmp_path, "concurrency", serializable(rows))
    root = f"/api/v1/jobs/{job['job_id']}"
    summary = client.get(root + "/summary", headers=headers).json()
    for metric in PHASE_METRICS:
        expected = summary["groups"][0]["metrics"][metric]["median"]
        response = client.get(root + f"/figure?metric={metric}&view=profile", headers=headers)
        assert response.status_code == (200 if expected is not None else 409)
        if expected is not None:
            figure = response.json()["figure"]
            assert figure["data"][0]["y"] == [expected]
            assert "测量批次" in figure["layout"]["title"]["text"]
            assert "client-phase-v1" in figure["layout"]["annotations"][0]["text"]
    assert client.get(root + "/figure?metric=phase_output").status_code == 401
    for format in ("html", "markdown"):
        text = client.get(root + f"/report?format={format}", headers=headers).text
        assert (
            "Client output phase" in text and "client-phase-v1" in text and "最晚首 Token" in text
        )
        assert "NEVER EXPORTED" not in text
    csv_text = client.get(root + "/export.csv", headers=headers).text
    assert "NEVER EXPORTED" not in csv_text
    exported = list(csv.DictReader(io.StringIO(csv_text)))
    phase = json.loads(exported[0]["request_phase_json"])
    assert phase["first_token_seconds"] == 100.5 and phase["latency_offset_seconds"] == 0.1
    history = SavedCsvHistory(tmp_path / "history")
    entry = history.upload(
        csv_text.encode(), "concurrency.csv", json.dumps({"test_type": "concurrency"}).encode()
    )
    restored = history.load(entry["id"])["summary"]
    assert not restored["integrity"]["verified"]
    assert (
        restored["overall"]["metrics"]["phase_output"]
        == summary["overall"]["metrics"]["phase_output"]
    )
    assert restored["overall"]["metrics"]["phase_input_uncached_api"]["median"] == pytest.approx(
        360
    )


def test_csv_rejects_conflicting_duplicate_phase_evidence(tmp_path):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ttft", "error", "request_phase_json", "extra_metrics"])
    phase = phased_batch()[0]["extra_metrics"]["request_phase"]
    writer.writerow(
        [0.2, "", json.dumps(phase), json.dumps({"request_phase": {**phase, "start_seconds": 99}})]
    )
    with pytest.raises(HistoryError, match="来源互相冲突"):
        SavedCsvHistory(tmp_path / "history").upload(output.getvalue().encode(), "conflict.csv")

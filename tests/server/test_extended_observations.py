"""Batch sampling, cache source separation and persisted API/report integration."""

import csv
import io
import json

import pytest

from core.benchmark.batch_observations import api_cached_tokens, tag_batch_window
from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from core.models.test_result import TestResult as ResultModel
from server.extended_analytics import extended_observations
from server.scenario_reports import profile_svg
from tests.server.test_scenario_figures import platform


def batch(count=2, elapsed=2.0):
    rows = [
        {
            "concurrency_level": 2,
            "ttft": 0.2,
            "prefill_tokens": 100,
            "decode_tokens": 20,
            "token_source": "API",
            "extra_metrics": {"metric_contract_version": METRIC_CONTRACT_VERSION},
        }
        for _ in range(count)
    ]
    tag_batch_window(rows, elapsed_seconds=elapsed, expected_requests=count)
    return rows


def serializable(rows):
    return [{**row, "extra_metrics": json.dumps(row["extra_metrics"])} for row in rows]


def test_shared_batch_is_sampled_once_and_batches_have_equal_weight():
    rows = batch(2, 2) + batch(4, 4)
    data = extended_observations(serializable(rows))
    assert data["metrics"]["system_output_wall"]["count"] == 2
    assert data["metrics"]["system_output_wall"]["median"] == 20
    assert data["metrics"]["system_qpm"]["median"] == 60
    assert data["system"]["valid_batches"] == 2
    assert ResultModel.from_api_result(1, rows[0]).batch_id == rows[0]["batch_id"]
    assert ResultModel.from_api_result(1, rows[1]).request_index == 1


def test_failed_wait_stays_in_denominator_and_failed_tokens_do_not_enter_totals():
    rows = batch(2, 10)
    rows[1].update(error="timeout", decode_tokens=999, prefill_tokens=999)
    data = extended_observations(serializable(rows))
    assert data["metrics"]["system_output_wall"]["median"] == 2
    assert data["metrics"]["system_input_wall"]["median"] == 10
    assert data["metrics"]["system_qpm"]["median"] == 6


@pytest.mark.parametrize(
    "fault",
    [
        "missing_member",
        "duplicate_index",
        "batch_id",
        "conflicting_window",
        "zero_window",
        "infinite_window",
        "wrong_version",
        "mixed_conditions",
    ],
)
def test_invalid_batch_is_excluded_instead_of_silently_recomputed(fault):
    rows = batch()
    if fault == "missing_member":
        rows.pop()
    elif fault == "duplicate_index":
        rows[1]["request_index"] = 0
    elif fault == "batch_id":
        rows[1]["batch_id"] = "b" * 32
    elif fault == "conflicting_window":
        rows[1]["extra_metrics"]["system_measurement"]["elapsed_seconds"] = 3
    elif fault == "mixed_conditions":
        rows[1]["concurrency_level"] = 4
    else:
        key, value = (
            ("version", "future")
            if fault == "wrong_version"
            else ("elapsed_seconds", 0 if fault == "zero_window" else float("inf"))
        )
        for row in rows:
            row["extra_metrics"]["system_measurement"][key] = value
    data = extended_observations(serializable(rows))
    assert data["system"]["invalid_batches"] == 1
    assert all(data["metrics"][key]["count"] == 0 for key in ["system_output_wall", "system_qpm"])


def test_partial_token_collection_does_not_fabricate_zero_but_qpm_is_observed():
    rows = batch()
    rows[1]["decode_tokens"] = None
    data = extended_observations(serializable(rows))
    assert data["metrics"]["system_output_wall"]["median"] is None
    assert data["metrics"]["system_total_wall"]["median"] is None
    assert data["metrics"]["system_input_wall"]["count"] == 1
    assert data["metrics"]["system_qpm"]["count"] == 1


def test_all_failed_batch_has_real_zero_successful_throughput():
    rows = batch()
    for row in rows:
        row["error"] = "timeout"
    data = extended_observations(serializable(rows))
    assert data["metrics"]["system_qpm"]["median"] == 0
    assert data["metrics"]["system_qpm"]["count"] == 1
    assert data["metrics"]["system_output_wall"]["median"] == 0


@pytest.mark.parametrize(
    "usage, expected",
    [
        (None, None),
        ({}, None),
        ({"prompt_tokens_details": {}}, None),
        ({"prompt_tokens_details": {"cached_tokens": 0}}, 0),
        ({"prompt_tokens_details": {"cached_tokens": 20}}, 20),
        ({"cache_hit_tokens": 0}, 0),
        ({"prompt_cache_hit_tokens": 30}, 30),
        ({"disk_cache_hit_tokens": 40}, 40),
        ({"cache_read_input_tokens": 50}, 50),
        ({"cache_hit_tokens": -1}, None),
        ({"cache_hit_tokens": True}, None),
        ({"cache_hit_tokens": "30"}, None),
        ({"prompt_tokens_details": {"cached_tokens": 0}, "cache_hit_tokens": 50}, 0),
    ],
)
def test_explicit_api_zero_is_distinct_from_missing_cache_field(usage, expected):
    assert api_cached_tokens(usage) == expected


def test_cache_api_inference_zero_unknown_and_failed_are_separate():
    rows = [
        {"cache_hit_source": "API", "cache_hit_tokens": 0, "api_prefill": 100, "ttft": 0.4},
        {"cache_hit_source": "API", "cache_hit_tokens": 20, "api_prefill": 100, "ttft": 0.2},
        {
            "cache_hit_source": "TTFT_inferred",
            "cache_hit_tokens": 90,
            "effective_prefill_tokens": 100,
            "ttft": 0.1,
        },
        {"cache_hit_source": "none", "cache_hit_tokens": 0, "ttft": 0.6},
        {
            "cache_hit_source": "API",
            "cache_hit_tokens": 50,
            "api_prefill": 100,
            "ttft": 99,
            "error": "timeout",
        },
    ]
    data = extended_observations(rows)
    assert data["metrics"]["cache_tokens_api"]["count"] == 2
    assert data["metrics"]["cache_rate_api"]["median"] == 10
    assert data["metrics"]["cache_rate_inferred"]["median"] == 90
    assert data["metrics"]["ttft_zero_cache_api"]["median"] == 0.4
    assert data["metrics"]["ttft_cache_api"]["median"] == 0.2
    assert data["cache"]["unknown_successes"] == 1


def test_api_ratio_never_uses_estimated_input_denominator_and_bad_counts_are_excluded():
    data = extended_observations(
        [
            {"cache_hit_source": "API", "cache_hit_tokens": 20, "prefill_tokens": 100, "ttft": 0.2},
            {"cache_hit_source": "API", "cache_hit_tokens": 101, "api_prefill": 100, "ttft": 0.3},
        ]
    )
    assert data["metrics"]["cache_tokens_api"]["count"] == 1
    assert data["metrics"]["cache_rate_api"]["count"] == 0
    assert data["metrics"]["ttft_cache_api"]["count"] == 1
    assert data["cache"]["invalid_observations"] == 1


def test_legacy_repeated_engine_rate_is_not_treated_as_a_verified_batch():
    data = extended_observations([{"system_output_throughput": 999, "rps": 999}])
    assert data["metrics"]["system_qpm"]["count"] == 0
    assert data["system"]["untagged_requests"] == 1


def test_zero_api_prompt_denominator_is_not_confused_with_missing_denominator():
    data = extended_observations(
        [
            {"cache_hit_source": "API", "cache_hit_tokens": 5, "api_prefill": 0, "ttft": 0.2},
            {"cache_hit_source": "API", "cache_hit_tokens": 0, "api_prefill": 0, "ttft": 0.3},
        ]
    )
    assert data["cache"]["invalid_observations"] == 1
    assert data["metrics"]["cache_tokens_api"]["count"] == 1
    assert data["metrics"]["cache_tokens_api"]["median"] == 0
    assert data["metrics"]["cache_rate_api"]["count"] == 0


def test_extended_metrics_survive_storage_and_share_report_plot_and_csv_contract(tmp_path):
    rows = batch()
    rows[0].update(cache_hit_source="API", cache_hit_tokens=0, api_prefill=100)
    rows[1].update(cache_hit_source="API", cache_hit_tokens=50, api_prefill=100)
    rows[0]["extra_metrics"]["private"] = "never-export-this"
    client, _, job, headers = platform(tmp_path, "concurrency", serializable(rows))
    root = f"/api/v1/jobs/{job['job_id']}"
    data = client.get(root + "/summary", headers=headers).json()
    assert data["groups"][0]["metrics"]["system_qpm"]["median"] == 60
    for metric in [
        "system_input_wall",
        "system_output_wall",
        "system_total_wall",
        "system_qpm",
        "cache_tokens_api",
        "cache_rate_api",
        "ttft_zero_cache_api",
        "ttft_cache_api",
    ]:
        result = client.get(root + f"/figure?metric={metric}&view=profile", headers=headers)
        assert result.status_code == 200
        assert result.json()["figure"]["data"][0]["y"] == [
            data["groups"][0]["metrics"][metric]["median"]
        ]
    assert client.get(root + "/figure?metric=system_qpm").status_code == 401
    html = client.get(root + "/report?format=html", headers=headers).text
    markdown = client.get(root + "/report?format=markdown", headers=headers).text
    assert "QPM p50" in html and "batch-wall-v1" in html and "API cache ratio" in markdown
    export = client.get(root + "/export.csv", headers=headers).text
    assert "never-export-this" not in export
    exported = list(csv.DictReader(io.StringIO(export)))
    assert json.loads(exported[0]["system_measurement_json"])["version"] == "batch-wall-v1"
    assert exported[1]["cache_hit_tokens"] == "50"


def test_zero_only_svg_has_finite_geometry(tmp_path):
    rows = batch()
    for row in rows:
        row["error"] = "timeout"
    client, _, job, headers = platform(tmp_path, "concurrency", serializable(rows))
    data = client.get(f"/api/v1/jobs/{job['job_id']}/summary", headers=headers).json()
    svg = profile_svg(data, "system_qpm")
    assert "<circle" in svg and "nan" not in svg.lower() and "inf" not in svg.lower()

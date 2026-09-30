"""Time curves require explicit shared-clock provenance, never persistence timestamps."""

import csv
import io
import json
import xml.etree.ElementTree as ET

import pytest

from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from server.analytics import run_results_csv, run_summary
from server.reports import render_html, render_markdown
from server.time_series import REQUEST_METRICS, stability_time_series, timeline_svg
from tests.server.test_scenario_figures import platform


def timed(index, end, *, expected=4, window=4.0, **row):
    timing = {
        "version": "stability-clock-v1",
        "clock": "monotonic",
        "anchor": "scheduler_start",
        "id": "a" * 32,
        "index": index,
        "start_seconds": max(0, end - 0.5),
        "end_seconds": end,
        "window_seconds": window,
        "planned_seconds": 3,
        "expected_requests": expected,
    }
    return {
        "ttft": 0.2,
        **row,
        "extra_metrics": json.dumps(
            {
                "metric_contract_version": METRIC_CONTRACT_VERSION,
                "timing_observation": timing,
                "private": "DO NOT EXPORT",
            }
        ),
    }


def alter(row, **values):
    extra = json.loads(row["extra_metrics"])
    extra["timing_observation"].update(values)
    return {**row, "extra_metrics": json.dumps(extra)}


@pytest.fixture
def rows():
    return [
        timed(3, 4.0),
        timed(0, 0.5),
        timed(2, 3.2, error="timeout", ttft=999),
        timed(1, 0.8, ttft=0.4),
    ]


def test_full_completion_time_bins_keep_gaps_failures_and_final_boundary(rows):
    result = stability_time_series(rows)
    assert result["complete"] and result["timed_requests"] == 4
    assert result["bin_seconds"] == 1
    assert [b["requests"] for b in result["bins"]] == [2, 0, 0, 2]
    assert [b["failures"] for b in result["bins"]] == [0, 0, 0, 1]
    assert [b["metrics"]["ttft"]["median"] for b in result["bins"]] == [
        pytest.approx(0.3),
        None,
        None,
        0.2,
    ]
    assert result["window_seconds"] > result["planned_seconds"]


@pytest.mark.parametrize(
    "values",
    [
        {"clock": "wall"},
        {"version": "unknown"},
        {"anchor": "database_created_at"},
        {"start_seconds": -1},
        {"start_seconds": 3},
        {"end_seconds": 8},
        {"end_seconds": float("nan")},
        {"end_seconds": float("inf")},
        {"end_seconds": True},
        {"window_seconds": 0},
        {"window_seconds": 86401},
        {"window_seconds": 10**400},
        {"expected_requests": True},
        {"index": -1},
        {"index": 4},
        {"id": "<script>"},
        {"planned_seconds": 0},
    ],
)
def test_invalid_clock_or_bounds_cannot_enter_time_axis(values):
    result = stability_time_series([alter(timed(0, 1), **values)])
    assert not result["bins"] and result["invalid_requests"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"id": "b" * 32},
        {"window_seconds": 5},
        {"planned_seconds": 2},
        {"expected_requests": 5},
        {"index": 0},
    ],
)
def test_conflicting_window_or_duplicate_membership_is_rejected(change):
    result = stability_time_series([timed(0, 1), alter(timed(1, 2), **change)])
    assert result["conflict"] and result["invalid_requests"] == 2 and not result["bins"]


def test_partial_timing_is_diagnostic_and_never_uses_database_dates():
    result = stability_time_series(
        [timed(0, 0.5), {"created_at": "2026-09-30", "start_time": 9000000, "end_time": 9000001}]
    )
    assert result["timed_requests"] == result["missing_requests"] == 1
    assert not result["complete"] and sum(b["requests"] for b in result["bins"]) == 1


def test_bounded_bins_use_all_120_rows_and_exclude_uncollected_or_failed_metrics():
    rows = [
        timed(i, i + 0.5, expected=120, window=120, ttft=0 if i == 0 else 0.1) for i in range(120)
    ]
    result = stability_time_series(rows)
    assert len(result["bins"]) == 60 and result["complete"]
    assert sum(b["requests"] for b in result["bins"]) == 120
    assert sum(b["metrics"]["ttft"]["count"] for b in result["bins"]) == 119


def test_all_failed_has_real_counts_and_no_fabricated_latency():
    result = stability_time_series([timed(0, 2, expected=1, error="timeout", ttft=999)])
    assert sum(b["failures"] for b in result["bins"]) == 1
    assert all(b["metrics"]["ttft"]["median"] is None for b in result["bins"])


def test_api_and_reports_share_all_time_bins_and_safe_exports(tmp_path, rows):
    client, store, job, headers = platform(tmp_path, "stability", rows)
    root = f"/api/v1/jobs/{job['job_id']}"
    summary = run_summary(str(store.path), 1, job=job)
    timeline = summary["time_series"]
    response = client.get(root + "/figure?view=timeline", headers=headers)
    assert response.status_code == 200
    traces = response.json()["figure"]["data"]
    assert traces[0]["x"] == [0.5, 1.5, 2.5, 3.5]
    assert traces[0]["y"] == [pytest.approx(0.3), None, None, 0.2]
    assert traces[0]["connectgaps"] is False
    assert traces[1]["y"] == [2, 0, 0, 2] and traces[2]["y"] == [0, 0, 0, 1]
    assert traces[0]["customdata"][3] == [3, 4, 1, 2, 1]
    assert client.get(root + "/figure?view=timeline").status_code == 401
    assert (
        client.get(root + "/figure?view=timeline&metric=system_qpm", headers=headers).status_code
        == 409
    )
    assert (
        client.get(root + "/figure?view=timeline&statistic=bogus", headers=headers).status_code
        == 422
    )
    html = render_html(job, summary)
    md = render_markdown(job, summary)
    assert "稳定性完成时间序列" in html and "稳定性完成时间序列" in md
    assert "3.000–4.000" in html and "3.000–4.000" in md
    for note in timeline["notes"]:
        assert note in html and note in md
    svg = timeline_svg(timeline, "ttft", "<script>", "s")
    root_svg = ET.fromstring(svg[svg.index("<svg") : svg.index("</svg>") + 6])
    assert len(root_svg.findall(".//polyline")) == 2
    assert "<script>" not in svg and "nan" not in svg
    exported = list(csv.DictReader(io.StringIO(run_results_csv(str(store.path), 1))))
    assert len(exported) == 4
    assert json.loads(exported[0]["timing_observation_json"])["end_seconds"] == 4
    assert "DO NOT EXPORT" not in json.dumps(exported)
    for metric in REQUEST_METRICS:
        for stat in ["mean", "median", "p95", "p99", "min", "max"]:
            assert (
                client.get(
                    root + f"/figure?view=timeline&metric={metric}&statistic={stat}",
                    headers=headers,
                ).status_code
                == 200
            )


@pytest.mark.parametrize("kind", ["stability", "concurrency"])
def test_old_rows_do_not_get_fake_time_figures(tmp_path, kind):
    client, _, job, headers = platform(
        tmp_path, kind, [{"ttft": 0.2, "start_time": 100, "end_time": 101}]
    )
    assert (
        client.get(
            f"/api/v1/jobs/{job['job_id']}/figure?view=timeline", headers=headers
        ).status_code
        == 409
    )

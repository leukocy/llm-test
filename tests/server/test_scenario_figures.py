"""Scenario plots preserve dimensions, full samples, missing cells and metric semantics."""

import json
import sqlite3
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from core.benchmark.metrics import METRIC_CONTRACT_VERSION
from core.run_lifecycle import RunStatus
from server.analytics import NUMERIC_FIELDS, run_summary
from server.api import create_app
from server.figures import performance_report_figure
from server.reports import render_html, render_markdown
from server.scenario_reports import METRICS, STATISTICS, profile_svg, scenario_analysis
from server.settings import Settings
from server.store import JobStore


def platform(tmp_path, kind, rows):
    store = JobStore(tmp_path / "db.sqlite")
    job = store.submit(
        test_type=kind,
        endpoint_id="synthetic",
        model_id="model",
        parameters={},
        progress_total=len(rows),
    )
    store.claim("fixture")
    version = json.dumps({"metric_contract_version": METRIC_CONTRACT_VERSION})
    with sqlite3.connect(store.path) as conn:
        conn.execute(
            "INSERT INTO test_runs(test_id,test_type,status,model_id,config_json) VALUES (?,?,'completed','model',?)",
            (job["job_id"], kind, version),
        )
        for row in rows:
            record = {"run_id": 1, "extra_metrics": version, **row}
            fields = list(record)
            conn.execute(
                f"INSERT INTO test_results ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                tuple(record.values()),
            )
    store.finish(job["job_id"], "fixture", outcome=RunStatus.COMPLETED, result_run_id=1)
    job = store.get(job["job_id"])
    settings = Settings(
        api_token="a" * 40, db_path=store.path, artifact_root=tmp_path / "artifacts", endpoints={}
    )
    return (
        TestClient(create_app(settings, store)),
        store,
        job,
        {"Authorization": "Bearer " + "a" * 40},
    )


@pytest.mark.parametrize(
    "kind,field,axis",
    [
        ("concurrency", "concurrency_level", "并发数"),
        ("custom_text", "concurrency_level", "并发数"),
        ("prefill", "input_tokens_target", "目标输入长度 (token)"),
        ("long_context", "context_length_target", "目标上下文长度 (token)"),
        ("segmented_prefill", "context_length_target", "目标分段长度 (token)"),
        ("stability", "concurrency_level", "并发数"),
    ],
)
def test_profiles_have_numeric_scenario_axes(tmp_path, kind, field, axis):
    client, _, job, headers = platform(
        tmp_path,
        kind,
        [
            {field: 100, "ttft": 0.8},
            {field: 2, "ttft": 0.2},
            {field: 10, "ttft": 0.4},
        ],
    )
    result = client.get(f"/api/v1/jobs/{job['job_id']}/figure?view=profile", headers=headers).json()
    assert result["figure"]["data"][0]["x"] == [2, 10, 100]
    assert result["figure"]["data"][0]["y"] == [0.2, 0.4, 0.8]
    assert result["figure"]["layout"]["xaxis"] == {
        "title": {"text": axis},
        "type": "linear",
        "automargin": True,
    }
    notes = " ".join(result["analysis"]["notes"])
    if kind == "stability":
        assert "不是时间序列" in notes
    if kind == "segmented_prefill":
        assert "不能称为未缓存 TTFT 或缓存收益" in notes


@pytest.fixture
def matrix(tmp_path):
    return platform(
        tmp_path,
        "throughput_matrix",
        [
            {"concurrency_level": 1, "context_length_target": 512, "ttft": 0.2, "tps": 50},
            {"concurrency_level": 1, "context_length_target": 512, "ttft": 0.4, "tps": 100},
            {"concurrency_level": 1, "context_length_target": 2048, "ttft": 99, "error": "timeout"},
            {"concurrency_level": 4, "context_length_target": 2048, "ttft": 0.8, "tps": 40},
        ],
    )


def test_matrix_keeps_missing_cells_and_all_failed_cells(matrix):
    client, _, job, headers = matrix
    root = f"/api/v1/jobs/{job['job_id']}"
    profile = client.get(root + "/figure?view=profile", headers=headers).json()["figure"]
    assert [s["name"] for s in profile["data"]] == ["并发 1", "并发 4"]
    assert profile["data"][0]["y"] == [pytest.approx(0.3), None]
    assert profile["data"][1]["y"] == [None, 0.8]
    assert all(not s["connectgaps"] for s in profile["data"])
    chart = client.get(root + "/figure?view=heatmap", headers=headers).json()["figure"]
    heat = chart["data"][0]
    assert heat["x"] == ["512", "2048"] and heat["y"] == ["1", "4"]
    assert heat["z"] == [[pytest.approx(0.3), None], [None, 0.8]]
    assert heat["customdata"] == [[[2, 2, 0], [0, 1, 1]], [[None, None, None], [1, 1, 0]]]
    assert heat["hoverongaps"] is False


@pytest.mark.parametrize("statistic", list(STATISTICS))
@pytest.mark.parametrize("metric", list(NUMERIC_FIELDS))
def test_every_metric_and_statistic_matches_full_summary(tmp_path, metric, statistic):
    rows = [
        {"concurrency_level": 2, **dict.fromkeys(NUMERIC_FIELDS, value)} for value in range(1, 81)
    ]
    rows.extend(
        [
            {"concurrency_level": 2, **dict.fromkeys(NUMERIC_FIELDS, 999), "error": "timeout"},
            {"concurrency_level": 2, **dict.fromkeys(NUMERIC_FIELDS, 0)},
        ]
    )
    _, store, job, _ = platform(tmp_path, "concurrency", rows)
    summary = run_summary(str(store.path), 1, job=job)
    result = performance_report_figure(job, summary, metric, view="profile", statistic=statistic)
    assert result["figure"]["data"][0]["y"] == [summary["overall"]["metrics"][metric][statistic]]
    assert result["figure"]["data"][0]["customdata"] == [[80, 82, 1]]
    assert summary["overall"]["metrics"][metric]["median"] == 40.5
    assert summary["overall"]["metrics"][metric]["max"] == 80
    assert METRICS[metric][2] in result["figure"]["layout"]["yaxis"]["title"]["text"]


def test_api_bounds_auth_no_sample_and_wrong_scenario(tmp_path):
    client, _, job, headers = platform(
        tmp_path, "prefill", [{"input_tokens_target": 512, "ttft": 0.2}]
    )
    root = f"/api/v1/jobs/{job['job_id']}/figure"
    assert client.get(root + "?view=heatmap").status_code == 401
    for query in [
        "metric=system_throughput",
        "view=bogus",
        "statistic=count",
        "metric=" + "x" * 2000,
    ]:
        assert client.get(root + "?" + query, headers=headers).status_code == 422
    assert client.get(root + "?metric=tpot", headers=headers).status_code == 409
    assert client.get(root + "?view=heatmap", headers=headers).status_code == 409
    assert client.get(root, headers=headers).status_code == 200


def test_exports_share_analysis_and_svg_does_not_bridge_missing_cells(matrix):
    client, store, job, headers = matrix
    summary = run_summary(str(store.path), 1, job=job)
    root = f"/api/v1/jobs/{job['job_id']}"
    html = client.get(root + "/report?format=html", headers=headers).text
    markdown = client.get(root + "/report?format=markdown", headers=headers).text
    for item in summary["scenario_analysis"]["observations"]:
        assert item["text"] in html and item["text"] in markdown
    assert "<svg" in html and "<script" not in html and "https://" not in html
    assert "完整指标切片" in markdown and "观测与" not in markdown
    svg = profile_svg(summary, "ttft")
    root_svg = ET.fromstring(svg[svg.index("<svg") : svg.index("</svg>") + 6])
    assert len(root_svg.findall("{http://www.w3.org/2000/svg}circle")) == 2
    assert "失败=1" not in svg


def test_unknown_dimensions_and_hostile_labels_are_safe(matrix):
    _, store, job, _ = matrix
    summary = run_summary(str(store.path), 1, job=job)
    for group in summary["groups"]:
        group.pop("dimensions")
    summary["groups"][0]["label"] = "</title><script>alert(1)</script>"
    figure = performance_report_figure(job, summary, view="profile")["figure"]
    assert figure["layout"]["xaxis"]["type"] == "category"
    assert "<script>" not in json.dumps(figure)
    html = render_html(job, summary)
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<script>" not in render_markdown(job, summary)
    with pytest.raises(ValueError, match="完整结构化条件"):
        performance_report_figure(job, summary, view="heatmap")


def test_historical_source_never_claims_verified_warmup(matrix):
    _, store, job, _ = matrix
    summary = run_summary(str(store.path), 1, job=job)
    summary["origin"] = {"kind": "saved_csv"}
    notes = scenario_analysis(summary)["notes"]
    assert "原运行的预热、计划与执行条件未经核验" in notes[0]
    assert "状态未知" in notes[4]


def test_planned_but_entirely_missing_matrix_axes_remain_visible(matrix):
    _, store, job, _ = matrix
    summary = run_summary(str(store.path), 1, job=job)
    summary["measurement_protocol"] = {
        "cells": [
            {"concurrency": c, "input_tokens_target": x}
            for c in [1, 4, 8]
            for x in [512, 2048, 4096]
        ]
    }
    heat = performance_report_figure(job, summary, view="heatmap")["figure"]["data"][0]
    assert heat["x"] == ["512", "2048", "4096"]
    assert heat["y"] == ["1", "4", "8"]
    assert heat["z"][2] == [None, None, None]
    assert all(row[2] is None for row in heat["z"])


def test_plan_matches_numeric_conditions_instead_of_comma_formatted_label(tmp_path):
    _, store, job, _ = platform(
        tmp_path, "prefill", [{"input_tokens_target": 2048, "prefill_tokens": 2048, "ttft": 0.2}]
    )
    config = {
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "measurement_protocol": {
            "measured_requests": 1,
            "warmup_recorded": 0,
            "warmup_requests": 0,
            "cells": [
                {"label": "2048 tokens", "input_tokens_target": 2048, "measured_requests": 1}
            ],
        },
    }
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE test_runs SET config_json=?", (json.dumps(config),))
    summary = run_summary(str(store.path), 1, job=job)
    assert summary["groups"][0]["label"] == "2,048 tokens"
    assert summary["groups"][0]["planned_requests"] == 1
    assert summary["groups"][0]["input_tokens"]["target_deviation_pct"] == 0
    assert summary["integrity"]["verified"] is True

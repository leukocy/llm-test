"""Finite workload counts match the real executor and report group verification."""

import pytest

from core.benchmark_runner import BenchmarkRunner
from core.run_lifecycle import RunStatus
from server.analytics import run_summary
from server.reports import render_html, render_markdown
from server.runner_adapter import execute_job
from server.specs import JobSubmission, expected_requests, measurement_plan
from tests.server.test_measurement_checkpoints import ReferenceEncoding, lab, result  # noqa: F401

CASES = {
    "segmented_prefill": {
        "segment_levels": [8, 16],
        "requests_per_segment": 2,
        "total_rounds": 2,
        "concurrency": 2,
        "max_tokens": 4,
    },
    "long_context": {"context_lengths": [8, 16, 8], "rounds_per_level": 2, "max_tokens": 4},
    "custom_text": {
        "selected_concurrencies": [1, 2],
        "rounds_per_level": 2,
        "base_prompt": "fixed",
        "max_tokens": 4,
    },
    "dataset": {
        "rows": [{"prompt": f"row-{i}"} for i in range(5)],
        "rounds": 2,
        "concurrency": 3,
        "max_tokens": 4,
    },
}


def test_segmented_budget_includes_concurrency_and_uses_absolute_targets():
    p = CASES["segmented_prefill"]
    body = JobSubmission(endpoint_id="lab", test_type="segmented_prefill", parameters=p)
    plan = measurement_plan(body.test_type, body.parameters)
    assert expected_requests(body.test_type, body.parameters) == plan["total_requests"] == 16
    assert [cell["measured_requests"] for cell in plan["cells"]] == [8, 8]
    assert plan["configured_input_token_volume"] == 192
    excessive = {
        **p,
        "segment_levels": [8] * 12,
        "requests_per_segment": 20,
        "total_rounds": 20,
        "concurrency": 1024,
    }
    with pytest.raises(ValueError, match="100000 requests"):
        JobSubmission(endpoint_id="lab", test_type="segmented_prefill", parameters=excessive)


def test_dataset_partial_batch_counts_and_unknown_file_plan():
    plan = measurement_plan("dataset", CASES["dataset"])
    assert [(cell["concurrency"], cell["measured_requests"]) for cell in plan["cells"]] == [
        (3, 6),
        (2, 4),
    ]
    assert plan["configured_input_token_volume"] is None
    assert plan["maximum_output_token_volume"] == 40
    assert (
        measurement_plan(
            "dataset", {"dataset": "local.json", "concurrency": 3, "rounds": 2, "max_tokens": 4}
        )["protocol_version"]
        is None
    )


def test_duplicate_targets_aggregate_expected_counts_without_losing_budget():
    plan = measurement_plan("long_context", CASES["long_context"])
    assert [(cell["input_tokens_target"], cell["measured_requests"]) for cell in plan["cells"]] == [
        (8, 4),
        (16, 2),
    ]
    assert plan["total_requests"] == 6 and plan["configured_input_token_volume"] == 64
    assert (
        measurement_plan("custom_text", CASES["custom_text"])["configured_input_token_volume"]
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(CASES))
async def test_real_executor_saved_protocol_and_report_conditions(lab, monkeypatch, kind):  # noqa: F811
    store, endpoint, settings, _ = lab
    if kind == "segmented_prefill":
        monkeypatch.setattr(BenchmarkRunner, "_get_tokenizer", lambda _: ReferenceEncoding())
    calls = []

    async def response(self, client, session_id, prompt, max_tokens, barrier=None):
        if barrier:
            await barrier.wait()
        calls.append(session_id)
        value = result(session_id, prompt)
        if kind == "segmented_prefill":
            value.update(prefill_tokens=len(prompt), api_prefill=len(prompt), api_decode=2)
        return value

    monkeypatch.setattr(BenchmarkRunner, "get_completion", response)
    body = JobSubmission(endpoint_id="lab", test_type=kind, parameters=CASES[kind])
    preview = measurement_plan(kind, body.parameters)
    job = store.submit(
        test_type=kind,
        endpoint_id="lab",
        model_id="m",
        parameters=body.parameters,
        progress_total=expected_requests(kind, body.parameters),
    )
    output = await execute_job(store.claim("w"), endpoint, settings, store, "w")
    final = store.finish(
        job["job_id"], "w", outcome=RunStatus.COMPLETED, result_run_id=output.result_run_id
    )
    summary = run_summary(str(store.path), output.result_run_id, job=final)
    assert (
        len(calls)
        == preview["measured_requests"]
        == summary["measurement_protocol"]["measured_requests"]
    )
    assert summary["integrity"]["verified"], summary["integrity"]["reasons"]
    assert all(group["planned_requests"] == group["requests"] for group in summary["groups"])
    if kind == "dataset":
        assert {group["dimensions"]["concurrency_level"] for group in summary["groups"]} == {2, 3}
    if kind == "segmented_prefill":
        assert any("独立随机样本" in text for text in summary["data_quality"]["warnings"])
        assert "独立随机样本" in render_html(final, summary)
        assert "独立随机样本" in render_markdown(final, summary)

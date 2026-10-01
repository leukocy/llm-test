"""Code grading provenance and native tests; no model/container execution."""

import json
import subprocess
import threading
from pathlib import Path

import httpx
import pytest

from core.safe_executor import (
    SandboxUnavailableError,
    require_sandbox_available,
    run_untrusted_code,
    sandbox_session,
)
from core.sandbox_identity import validate_identity
from core.standard_report import StandardReport
from evaluators.base_evaluator import EvaluationResult
from evaluators.humaneval_evaluator import HumanEvalEvaluator
from evaluators.mbpp_evaluator import MBPPEvaluator
from sandbox_worker import SandboxServer
from tests.sandbox_fixtures import sandbox_identity


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("LLM_TEST_SANDBOX_URL", "http://synthetic-worker:8765")
    monkeypatch.setenv("LLM_TEST_SANDBOX_TOKEN", "x" * 40)
    proof = sandbox_identity()
    calls = []

    def request(method, url, token, timeout, **kwargs):
        calls.append((method, kwargs))
        if method == "GET":
            return httpx.Response(200, json={"status": "ok", "sandbox_identity": proof})
        code = kwargs["json"]["code"]
        good = "return 999" not in code
        return httpx.Response(
            200,
            json={
                "success": good,
                "error": None if good else "AssertionError",
                "output": None,
                "sandbox_identity": proof,
            },
        )

    monkeypatch.setattr("core.safe_executor._worker_request", request)
    return proof, calls


@pytest.mark.asyncio
async def test_humaneval_really_invokes_check_function(worker):
    proof, calls = worker

    async def respond(*args, **kwargs):
        return {"content": "    return 999"}

    evaluator = HumanEvalEvaluator(num_shots=0)
    with sandbox_session(proof):
        result = await evaluator.evaluate_single(
            {
                "task_id": "t",
                "prompt": "def f():\n",
                "test": "def check(candidate):\n    assert candidate() == 1",
                "entry_point": "f",
            },
            respond,
        )
    assert not result.is_correct
    assert calls[0][1]["json"]["code"].endswith("check(f)\n")
    assert calls[0][1]["json"]["sandbox_contract"] == proof["sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,correct", [("def f():\n    return 999", False), ("def f():\n    return 1", True)]
)
async def test_mbpp_scores_execution_not_function_shape(worker, code, correct):
    proof, calls = worker

    async def respond(*args, **kwargs):
        return {"content": code, "input_tokens": 10, "output_tokens": 12}

    evaluator = MBPPEvaluator(num_shots=0)
    assert evaluator.requires_code_execution
    with sandbox_session(proof):
        result = await evaluator.evaluate_single(
            {
                "task_id": 7,
                "text": "Return one",
                "test_setup_code": "import math",
                "test_list": ["assert f() == 1"],
                "code": "def f(): return 1",
            },
            respond,
        )
    assert result.is_correct is correct
    assert result.sample_id == "7" and result.question == "Return one"
    sent = calls[0][1]["json"]["code"]
    assert sent.startswith("import math\n") and sent.endswith("assert f() == 1")
    assert "check(" not in sent
    assert result.input_tokens == 10 and result.output_tokens == 12
    assert result.error is None and result.evaluation_method == "code_execution"
    assert result.execution_error == (None if correct else "AssertionError")
    if not correct:
        exported = StandardReport.from_evaluation_result(
            EvaluationResult("mbpp", "synthetic", 0, 1, 0, details=[result])
        )
        assert exported.failure_analysis["mbpp"][0].analysis == "AssertionError"


@pytest.mark.asyncio
async def test_mbpp_without_tests_fails_before_model_request(worker):
    async def respond(*args, **kwargs):
        pytest.fail("missing tests must not consume model requests")

    with pytest.raises(Exception, match="no usable execution tests"):
        await MBPPEvaluator().evaluate_single({"text": "Question", "test_list": []}, respond)


@pytest.mark.parametrize("changed", [None, sandbox_identity("d")])
def test_execution_rejects_missing_or_changed_worker_identity(worker, monkeypatch, changed):
    proof, _ = worker
    monkeypatch.setattr(
        "core.safe_executor._worker_request",
        lambda *a, **k: httpx.Response(200, json={"success": True, "sandbox_identity": changed}),
    )
    with sandbox_session(proof), pytest.raises(SandboxUnavailableError):
        run_untrusted_code("fixed-synthetic-source")
    # Context does not leak to a separate grading session.
    monkeypatch.setattr(
        "core.safe_executor._worker_request",
        lambda *a, **k: httpx.Response(200, json={"success": True}),
    )
    assert run_untrusted_code("fixed-synthetic-source")[0]


def test_health_without_environment_evidence_is_not_ready(worker, monkeypatch):
    monkeypatch.setattr(
        "core.safe_executor._worker_request",
        lambda *a, **k: httpx.Response(200, json={"status": "ok"}),
    )
    with pytest.raises(SandboxUnavailableError, match="health check"):
        require_sandbox_available()


def test_worker_resolves_tag_once_and_records_runtime_policy(monkeypatch):
    inspections = []

    def docker(command, **kwargs):
        inspections.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout="sha256:" + "a" * 64 + "\n")
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(sandbox_identity()["runtime"])
        )

    monkeypatch.setattr("sandbox_worker.subprocess.run", docker)
    server = SandboxServer(("127.0.0.1", 0), "x" * 40, "mutable-tag")
    try:
        first = server.execution_identity()
        second = server.execution_identity()
        assert first == second and validate_identity(first) == first
        assert first["image_id"] == "sha256:" + "a" * 64
        assert sum(command[1] == "image" for command in inspections) == 1
    finally:
        server.server_close()


def test_http_worker_contract_mismatch_blocks_execution(monkeypatch):
    proof = sandbox_identity()
    monkeypatch.setattr(SandboxServer, "execution_identity", lambda self: proof)
    executed = []
    monkeypatch.setattr(
        "sandbox_worker.execute_in_container",
        lambda *args: executed.append(args) or (True, None, None),
    )
    server = SandboxServer(("127.0.0.1", 0), "x" * 40, "mutable-tag")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with httpx.Client(trust_env=False, timeout=3) as client:
            base = f"http://127.0.0.1:{server.server_port}"
            headers = {"Authorization": "Bearer " + "x" * 40}
            assert client.get(base + "/health", headers=headers).json()["sandbox_identity"] == proof
            body = {
                "code": "synthetic",
                "timeout_seconds": 2,
                "mem_limit_mb": 128,
                "sandbox_contract": "d" * 64,
            }
            assert client.post(base + "/execute", headers=headers, json=body).status_code == 409
            assert not executed
            body["sandbox_contract"] = proof["sha256"]
            response = client.post(base + "/execute", headers=headers, json=body)
            assert response.json()["sandbox_identity"] == proof
            assert executed[0][-1] == proof["image_id"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_environment_evidence_cannot_be_modified_without_checksum():
    proof = sandbox_identity()
    proof["runtime"]["KernelVersion"] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        validate_identity(proof)


def test_mbpp_split_keeps_all_test_samples_and_separate_examples(tmp_path: Path):
    rows = [
        {
            "task_id": identity,
            "text": f"q-{identity}",
            "code": "def f(): return 1",
            "test_list": ["assert f() == 1"],
            "test_imports": ["import math"],
        }
        for identity in [1, 2, 3, 4, 11, 12, 13, 601]
    ]
    (tmp_path / "mbpp.json").write_text(json.dumps(rows))
    evaluator = MBPPEvaluator(dataset_path=str(tmp_path), num_shots=3, max_samples=None)
    samples = evaluator.load_dataset()
    assert {sample["task_id"] for sample in samples} == {11, 12, 13}
    assert [sample["task_id"] for sample in evaluator.few_shot_examples] == [2, 3, 4]
    assert all(sample["test_setup_code"] == "import math" for sample in samples)
    assert evaluator.evaluation_split == "test" and evaluator.few_shot_split == "prompt"
    # Prepared test.json takes priority; no prompting examples taken from test.
    (tmp_path / "test.json").write_text(json.dumps(rows[4:7]))
    with pytest.raises(Exception, match="few-shot"):
        evaluator.load_dataset()
    (tmp_path / "prompt.json").write_text(json.dumps(rows[:4]))
    assert len(evaluator.load_dataset()) == 3


@pytest.mark.parametrize(
    "tests,entry", [("pass", "f"), ("def check(:", "f"), ("def check(candidate): pass", "for")]
)
def test_invalid_humaneval_tests_are_rejected_during_plan_load(monkeypatch, tests, entry):
    monkeypatch.setattr(
        "core.dataset_manager.get_dataset",
        lambda **kwargs: [
            {"task_id": "t", "prompt": "def f():", "test": tests, "entry_point": entry}
        ],
    )
    with pytest.raises(Exception, match="HumanEval"):
        HumanEvalEvaluator().load_dataset()

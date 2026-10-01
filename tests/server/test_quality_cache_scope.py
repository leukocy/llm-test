"""Exercise real SQLite cache isolation with synthetic model responses."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from core.quality_evaluator import QualityEvaluator
from core.response_cache import ResponseCache
from evaluators.base_evaluator import ProviderRequestError
from server.paired_quality import compare_dataset
from tests.server.test_paired_quality import dataset


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(QualityEvaluator, "_init_tokenizer", lambda *args: None)
    cache = ResponseCache(str(tmp_path / "cache"))
    calls = []

    def make(url="http://127.0.0.1:9/v1", credential="synthetic-credential", provider="OpenAI"):
        evaluator = QualityEvaluator(
            url,
            "same-model",
            credential,
            provider,
            enable_cache=False,
            output_dir=str(tmp_path / "outputs"),
        )

        async def complete(**kwargs):
            assert "retries" not in kwargs
            calls.append(kwargs)
            return {
                "full_response_content": f"response-{len(calls)}",
                "start_time": 1,
                "first_token_time": 2,
                "end_time": 3,
                "usage_info": {"prompt_tokens": 5, "completion_tokens": 2},
                "timing_clock": "client_monotonic",
            }

        evaluator.provider = SimpleNamespace(get_completion=complete)
        evaluator.count_tokens = lambda text: 7
        evaluator.cache = cache
        evaluator._cache_enabled = True
        return evaluator

    return make, cache, calls, tmp_path


@pytest.mark.asyncio
async def test_identical_context_hits_and_reopening_retains_scoped_key(lab):
    make, cache, calls, _ = lab
    evaluator = make()
    first = await evaluator._get_response_with_metrics("Q")
    second = await evaluator._get_response_with_metrics("Q", retries=0, request_timeout=12)
    assert first["content"] == second["content"] and len(calls) == 1
    assert second["from_cache"] and second["ttft_ms"] == 0
    assert second["measurement_provenance"]["cache_context_version"] == "quality-cache-v2"
    assert second["measurement_provenance"]["input_token_source"] == "tokenizer"
    evaluator.cache = ResponseCache(str(cache.cache_dir))
    assert (await evaluator._get_response_with_metrics("Q"))["from_cache"]
    assert (cache.cache_dir / ".request_scope_key").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"temperature": 0.5},
        {"max_tokens": 1024},
        {"thinking_enabled": True, "thinking_budget": 2048},
        {"reasoning_effort": "high"},
        {"_custom_extra_body": {"top_p": 0.7}},
    ],
)
async def test_generation_parameter_changes_do_not_reuse_answers(lab, changed):
    make, _, calls, _ = lab
    evaluator = make()
    first = await evaluator._get_response_with_metrics("Q")
    second = await evaluator._get_response_with_metrics("Q", **changed)
    assert first["content"] != second["content"] and len(calls) == 2
    assert not second["from_cache"]
    assert (await evaluator._get_response_with_metrics("Q", **changed))["from_cache"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed",
    [
        {"url": "http://127.0.0.1:10/v1"},
        {"credential": "different-synthetic-credential"},
        {"provider": "Anthropic"},
    ],
)
async def test_same_model_id_isolated_across_endpoints_auth_and_protocol(lab, changed):
    make, cache, calls, _ = lab
    first = await make()._get_response_with_metrics("Q")
    second = await make(**changed)._get_response_with_metrics("Q")
    assert first["content"] != second["content"] and len(calls) == 2
    assert not second["from_cache"]
    assert b"synthetic-credential" not in cache.db_path.read_bytes()


@pytest.mark.asyncio
async def test_prompt_text_cannot_collide_with_structured_messages_or_legacy_keys(lab):
    make, cache, calls, _ = lab
    messages = [{"role": "user", "content": "Q"}]
    prompt = json.dumps(messages, ensure_ascii=False)
    cache.set(prompt, "unsafe-legacy-response", model_id="same-model")
    evaluator = make()
    flat = await evaluator._get_response_with_metrics(prompt)
    structured = await evaluator._get_response_with_metrics("", messages=messages)
    assert flat["content"] != "unsafe-legacy-response"
    assert flat["content"] != structured["content"] and len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["corrupt", "public", "symlink"])
async def test_unverifiable_key_fails_before_any_model_request(lab, mode):
    make, cache, calls, root = lab
    path = cache.cache_dir / ".request_scope_key"
    if mode == "symlink":
        target = root / "synthetic-key"
        target.write_bytes(b"x" * 32)
        path.symlink_to(target)
    else:
        path.write_bytes(b"x" if mode == "corrupt" else b"x" * 32)
        path.chmod(0o644 if mode == "public" else 0o600)
    with pytest.raises(ProviderRequestError, match="response-cache scope unavailable"):
        await make()._get_response_with_metrics("Q")
    assert not calls


@pytest.mark.parametrize(
    "provenance,verified",
    [
        (None, False),
        ({"from_cache": True}, False),
        ({"from_cache": False}, True),
        ({"from_cache": True, "cache_context_version": "quality-cache-v2"}, True),
    ],
)
def test_paired_statistics_require_cache_origin_when_enabled(provenance, verified):
    left, right = dataset(), dataset()
    for side in (left, right):
        side["config"]["use_cache"] = True
        if provenance is not None:
            for row in side["details"]:
                row["measurement_provenance"] = provenance
    result = compare_dataset(left, right)
    assert result["verified"] is verified
    if not verified:
        assert result["p_value"] is None


@pytest.mark.asyncio
async def test_cache_database_failure_does_not_become_a_model_score(lab, monkeypatch):
    make, cache, calls, _ = lab

    def broken(*args, **kwargs):
        raise sqlite3.DatabaseError("synthetic storage failure")

    monkeypatch.setattr(cache, "get", broken)
    with pytest.raises(ProviderRequestError, match="response-cache lookup unavailable"):
        await make()._get_response_with_metrics("Q")
    assert not calls


def test_cache_connections_close_after_operations_and_transactions_commit(tmp_path, monkeypatch):
    original = sqlite3.connect
    opened = []

    class Tracked(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            return super().close()

    def connect(*args, **kwargs):
        conn = original(*args, **kwargs, factory=Tracked)
        opened.append(conn)
        return conn

    monkeypatch.setattr("core.response_cache.sqlite3.connect", connect)
    cache = ResponseCache(str(tmp_path / "cache"))
    cache.set("prompt", "answer", model_id="model")
    assert cache.get("prompt", model_id="model") == "answer"
    cache.get_stats()
    cache.delete("prompt", model_id="model")
    cache.clear()
    assert opened and all(conn.closed for conn in opened)

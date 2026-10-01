"""Native Gemini protocol evidence through mocked HTTP, measurement and reports."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from core import cancel_state
from core.benchmark_runner import BenchmarkRunner
from core.providers import openai
from core.providers.gemini import GeminiProvider
from core.providers.gemini_stream import normalize_usage
from core.quality_evaluator import QualityEvaluator
from core.response_parser import parse_stream_response
from server.quality_analysis import quality_analysis, quality_analysis_html
from tests.server.test_quality_analysis import fixture_report

USAGE = {
    "promptTokenCount": 10,
    "candidatesTokenCount": 3,
    "thoughtsTokenCount": 7,
    "cachedContentTokenCount": 4,
    "totalTokenCount": 20,
}


def frame(parts=None, finish=None, usage=None):
    candidate = {"content": {"parts": parts or []}}
    if finish:
        candidate["finishReason"] = finish
    chunk = {"candidates": [candidate]}
    if usage is not None:
        chunk["usageMetadata"] = usage
    return chunk


def stream(*chunks):
    return "".join("data:" + json.dumps(chunk) + "\n\n" for chunk in chunks)


def answer():
    return stream(
        frame([{"text": "B is tempting", "thought": True}]), frame([{"text": "A"}], "STOP", USAGE)
    )


@pytest.fixture(autouse=True)
def reset_cancel():
    cancel_state.reset_all()
    yield
    cancel_state.reset_all()


@pytest.fixture(autouse=True)
def isolate_request_logger(monkeypatch):
    monkeypatch.setattr("core.request_logger.get_request_logger", lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base",
    [
        "https://generativelanguage.googleapis.com",
        "https://generativelanguage.googleapis.com/v1beta/",
    ],
)
async def test_native_thought_answer_usage_and_answer_clock(base, monkeypatch):
    stamps = iter([100.0, 100.1, 100.4, 101.0])
    monkeypatch.setattr(
        "core.providers.gemini.time",
        SimpleNamespace(monotonic=lambda: next(stamps), time=lambda: 2000000000.0),
    )

    def handle(request):
        assert (
            str(request.url)
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        )
        assert request.headers["x-goog-api-key"] == "synthetic"
        assert "key=" not in str(request.url)
        payload = json.loads(request.content)
        assert payload["systemInstruction"]["parts"][0]["text"] == "first\n\nsecond"
        assert payload["contents"][0]["role"] == "user"
        return httpx.Response(200, text=answer())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await GeminiProvider(base, "synthetic", "models/gemini-2.5-flash").get_completion(
            client,
            1,
            messages=[
                {"role": "system", "content": "first"},
                {"role": "system", "content": "second"},
                {"role": "user", "content": "Q"},
            ],
        )
        assert not client.is_closed
    assert result["error"] is None
    assert result["full_response_content"] == "A"
    assert result["reasoning_content"] == "B is tempting"
    assert result["first_token_time"] == 100.4 and result["first_reasoning_time"] == 100.1
    assert result["usage_info"]["completion_tokens"] == 3
    assert result["usage_info"]["completion_tokens_details"]["reasoning_tokens"] == 7
    assert result["usage_info"]["prompt_tokens_details"]["cached_tokens"] == 4
    assert asyncio.current_task() not in openai._active_streams


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        stream(frame([{"text": "A"}])),
        stream(frame([{"text": "secret thought", "thought": True}], "MAX_TOKENS")),
        stream(frame([{"text": "A"}], "SAFETY")),
        stream({"error": {"message": "synthetic-error"}}),
        "data: not-json\n\n",
        stream(frame([{"text": "A", "thought": "old-invalid-format"}], "STOP")),
        stream(frame([{"text": "A"}], "STOP"), frame([{"text": "late"}])),
        stream(frame([{"text": "A"}], "STOP", {"promptTokenCount": True})),
    ],
)
async def test_incomplete_blocked_or_invalid_stream_never_succeeds(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))
    ) as client:
        result = await GeminiProvider(
            "https://generativelanguage.googleapis.com", "synthetic", "gemini-2.5-flash"
        ).get_completion(client, 1, prompt="Q")
    assert result["error"] and "full_response_content" not in result
    assert asyncio.current_task() not in openai._active_streams


def test_missing_counts_not_zero_and_native_parser_uses_boolean_marker():
    usage = normalize_usage({"promptTokenCount": 0})
    assert usage["prompt_tokens"] == 0 and usage["completion_tokens"] is None
    parsed = parse_stream_response(
        [frame([{"thought": True, "text": "B"}]), frame([{"text": "A"}], "STOP", USAGE)], "gemini"
    )
    assert parsed.full_content == "A" and parsed.full_reasoning == "B"
    assert parsed.usage["completion_tokens"] == 3


@pytest.mark.parametrize(
    "raw",
    [
        {"candidatesTokenCount": -1},
        {"promptTokenCount": "10"},
        {"promptTokenCount": 1, "cachedContentTokenCount": 2},
        {"promptTokenCount": 10, "totalTokenCount": 1},
    ],
)
def test_invalid_usage_rejected(raw):
    with pytest.raises(ValueError):
        normalize_usage(raw)


@pytest.mark.asyncio
async def test_quality_measurement_keeps_answer_only_and_report_usage(monkeypatch, tmp_path):
    monkeypatch.setattr(QualityEvaluator, "_init_tokenizer", lambda *a: None)
    evaluator = QualityEvaluator(
        api_base_url="https://generativelanguage.googleapis.com",
        model_id="gemini-2.5-flash",
        provider="Gemini",
        api_key="synthetic",  # pragma: allowlist secret
        enable_cache=False,
        output_dir=str(tmp_path),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=answer()))
    ) as client:
        complete = evaluator.provider.get_completion

        async def mocked_complete(**kwargs):
            kwargs["client"] = client
            return await complete(**kwargs)

        monkeypatch.setattr(evaluator.provider, "get_completion", mocked_complete)
        monkeypatch.setattr(
            "core.providers.gemini.httpx.AsyncClient",
            lambda **kw: pytest.fail("real network client"),
        )
        result = await evaluator._get_response_with_metrics("Q", use_cache=False, retries=0)
    assert (
        result["content"] == "A" and result["output_tokens"] == 3 and result["input_tokens"] == 10
    )
    assert result["measurement_provenance"]["provider_usage"]["thoughtsTokenCount"] == 7
    payload = fixture_report()
    payload["datasets"]["gsm8k"]["details"][0]["measurement_provenance"] = result[
        "measurement_provenance"
    ]
    projected = quality_analysis(payload)
    assert any("Gemini" in warning for warning in projected["datasets"]["gsm8k"]["warnings"])
    assert "Gemini" in quality_analysis_html(projected)


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gemini-3-flash-preview", "gemini-2.5-pro"])
async def test_unsupported_thinking_off_fails_before_http(model):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("unexpected request"))
    ) as client:
        result = await GeminiProvider(
            "https://generativelanguage.googleapis.com", "synthetic", model
        ).get_completion(client, 1, thinking_enabled=False)
    assert result["error"] and "cannot disable" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,expected", [(False, 0), (True, -1)])
async def test_flash_thinking_config_matches_native_budget(enabled, expected):
    def handle(request):
        config = json.loads(request.content)["generationConfig"]["thinkingConfig"]
        assert config["thinkingBudget"] == expected and "thinkingLevel" not in config
        assert config["includeThoughts"] is enabled
        return httpx.Response(200, text=answer())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await GeminiProvider(
            "https://generativelanguage.googleapis.com", "synthetic", "gemini-2.5-flash"
        ).get_completion(client, 1, thinking_enabled=enabled)
    assert result["error"] is None


@pytest.mark.asyncio
async def test_performance_observation_preserves_native_usage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_TEST_DB_PATH", str(tmp_path / "isolated.db"))
    placeholder = MagicMock()
    runner = BenchmarkRunner(
        placeholder=placeholder,
        progress_bar=placeholder,
        status_text=placeholder,
        api_base_url="https://generativelanguage.googleapis.com",
        model_id="gemini-2.5-flash",
        tokenizer_option="字符数 (Fallback)",
        csv_filename=str(tmp_path / "result.csv"),
        api_key="synthetic",  # pragma: allowlist secret
        log_placeholder=placeholder,
        provider="Gemini",
        enable_live_log_server=False,
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=answer()))
    ) as client:
        observed = await runner.get_completion(client, 1, "Q", 100)
    assert observed["decode_tokens"] == 3 and observed["prefill_tokens"] == 10
    assert observed["cache_hit_tokens"] == 4
    assert observed["output_text"] == "A"
    metadata = observed["extra_metrics"]["provider_measurement"]
    assert metadata["provider_usage"]["thoughtsTokenCount"] == 7
    assert metadata["ttft_scope"] == "first_answer_text"
    partial = runner._calculate_tokens("Q", "answer", {"prompt_tokens": 10})
    assert partial[0] == 10 and partial[1] is not None
    assert partial[4]["completion_tokens"] is None and "API input /" in partial[2]


@pytest.mark.asyncio
async def test_request_log_preserves_usage_and_masks_native_key(tmp_path, monkeypatch):
    from core.request_logger import RequestLogger

    logger = RequestLogger(log_dir=str(tmp_path / "logs"))
    monkeypatch.setattr("core.request_logger.get_request_logger", lambda: logger)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=answer()))
    ) as client:
        result = await GeminiProvider(
            "https://generativelanguage.googleapis.com", "synthetic-native-key", "gemini-2.5-flash"
        ).get_completion(client, 1, prompt="Q")
    assert result["error"] is None
    files = list((tmp_path / "logs").rglob("*.json"))
    contents = "\n".join(path.read_text() for path in files)
    assert "synthetic-native-key" not in contents
    assert '"thoughtsTokenCount": 7' in contents and "B is tempting" in contents

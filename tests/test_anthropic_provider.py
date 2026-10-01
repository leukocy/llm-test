"""Native wire protocol tests with synthetic HTTP responses only."""

import asyncio
import json

import httpx
import pytest

from core import cancel_state
from core.benchmark.batch_observations import api_cached_tokens
from core.benchmark.phase_observations import provider_phase_observation
from core.providers.anthropic import AnthropicProvider
from core.providers.factory import get_provider


def stream(*events):
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def answer(usage=None, finish=True):
    events = [
        {
            "type": "message_start",
            "message": {"usage": usage or {"input_tokens": 10, "output_tokens": 1}},
        },
        {"type": "ping"},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "reason"}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}},
        {"type": "message_delta", "usage": {"output_tokens": 4}},
        {"type": "message_delta", "usage": {"output_tokens": 7}},
    ]
    if finish:
        events.append({"type": "message_stop"})
    return stream(*events)


@pytest.fixture(autouse=True)
def reset_cancel(monkeypatch):
    cancel_state.reset_all()
    monkeypatch.setattr("core.request_logger.get_request_logger", lambda: None)
    yield
    cancel_state.reset_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("base", ["https://api.anthropic.com", "https://api.anthropic.com/v1/"])
async def test_native_messages_usage_and_monotonic_observations(base):
    def handler(request):
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "synthetic"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        assert payload["system"] == "first\n\nsecond"
        assert payload["messages"] == [{"role": "user", "content": "Q"}]
        return httpx.Response(
            200,
            text=answer(
                {
                    "input_tokens": 10,
                    "cache_creation_input_tokens": 20,
                    "cache_read_input_tokens": 30,
                    "output_tokens": 1,
                }
            ),
        )

    provider = get_provider("anthropic", base, "synthetic", "claude-test")
    assert isinstance(provider, AnthropicProvider)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await provider.get_completion(
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
    assert result["reasoning_content"] == "reason"
    assert result["usage_info"]["prompt_tokens"] == 60
    assert result["usage_info"]["completion_tokens"] == 7
    assert api_cached_tokens(result["usage_info"]) == 30
    assert provider_phase_observation(result, 0) is not None
    assert len(result["token_timestamps"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        answer(finish=False),
        stream({"type": "error", "error": {"message": "synthetic-sensitive"}}),
        "data: not-json\n\n",
        answer({"input_tokens": -1}),
        answer({"input_tokens": True}),
    ],
)
async def test_invalid_stream_is_never_a_success(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body))
    ) as client:
        result = await AnthropicProvider(
            "https://api.anthropic.com", "synthetic", "model"
        ).get_completion(client, 1, prompt="Q")
    assert result["error"]
    assert "synthetic-sensitive" not in str(result)
    assert "full_response_content" not in result


@pytest.mark.asyncio
async def test_missing_cache_is_unknown_not_zero():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=answer()))
    ) as client:
        result = await AnthropicProvider(
            "https://api.anthropic.com/v1", "synthetic", "model"
        ).get_completion(client, 1, prompt="Q")
    assert api_cached_tokens(result["usage_info"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"thinking_enabled": True, "thinking_budget": 256},
        {"thinking_enabled": True, "thinking_budget": 1024, "max_tokens": 1024},
        {"thinking_enabled": True, "temperature": 0},
        {"_custom_extra_body": {"model": "other"}},
    ],
)
async def test_invalid_parameters_fail_before_request(params):
    def handler(_):
        pytest.fail("invalid configuration must not call a model")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnthropicProvider(
            "https://api.anthropic.com", "synthetic", "model"
        ).get_completion(client, 1, prompt="Q", **params)
    assert result["error"]


@pytest.mark.asyncio
async def test_thinking_payload_and_http_failure():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert payload["output_config"] == {"effort": "high"}
        assert "temperature" not in payload
        return httpx.Response(401, text="synthetic-sensitive")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnthropicProvider(
            "https://api.anthropic.com", "synthetic", "model"
        ).get_completion(
            client,
            1,
            prompt="Q",
            max_tokens=2048,
            thinking_enabled=True,
            thinking_budget=1024,
            temperature=1,
            reasoning_effort="high",
        )
    assert result == {"error": "Anthropic API returned HTTP 401"}


@pytest.mark.asyncio
async def test_cancel_propagates():
    class CancelStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield stream({"type": "ping"}).encode()
            raise asyncio.CancelledError()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=CancelStream()))
    ) as client:
        with pytest.raises(asyncio.CancelledError):
            await AnthropicProvider(
                "https://api.anthropic.com", "synthetic", "model"
            ).get_completion(client, 1, prompt="Q")
        assert not client.is_closed

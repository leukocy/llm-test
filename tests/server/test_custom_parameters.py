"""Typed extra parameters survive admission, presets and native payload dispatch."""

import json
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from core.benchmark_runner import BenchmarkRunner
from server.api import create_app
from server.specs import CustomParameter, RunConfig
from tests.server.test_measurement_checkpoints import lab  # noqa: F401

PARAMS = [
    {"name": "top_p", "location": "top_level", "value": 0.9},
    {"name": "chat_template_kwargs", "location": "extra_body", "value": {"enable_thinking": False}},
    {"name": "user_tag", "location": "top_level", "value": "sample"},
    {"name": "stop", "location": "top_level", "value": ["END", "STOP"]},
    {"name": "optional_flag", "location": "extra_body", "value": None},
    {"name": "optional_top", "location": "top_level", "value": None},
]


def test_native_json_types_and_duplicate_names():
    assert RunConfig(custom_params=PARAMS).model_dump()["custom_params"] == PARAMS
    with pytest.raises(ValueError, match="不能重复"):
        RunConfig(custom_params=[*PARAMS, {"name": "top_p", "location": "extra_body", "value": 1}])


@pytest.mark.parametrize(
    "name",
    [
        "max_tokens",
        "temperature",
        "prompt",
        "model",
        "api_key",
        "headers",
        "_barrier",
        "n",
        "stream_options",
    ],
)
def test_dedicated_fields_cannot_be_overridden(name):
    with pytest.raises(ValueError):
        CustomParameter(name=name, value=1)


@pytest.mark.parametrize(
    "value", [math.inf, math.nan, "x" * 16384, {"headers": {"x-api-key": "synthetic"}}]
)
def test_invalid_or_credential_values_rejected(value):
    with pytest.raises(ValueError):
        CustomParameter(name="custom", value=value)


def test_json_schema_can_describe_credential_named_properties_without_containing_credentials():
    value = {"type": "object", "properties": {"api_key": {"type": "string"}}}
    assert CustomParameter(name="response_format", value=value).value == value


def test_auth_api_preset_roundtrip_and_invalid_rows(lab):  # noqa: F811
    store, endpoint, settings, _ = lab
    client = TestClient(create_app(settings, store))
    headers = {"Authorization": "Bearer " + settings.api_token}
    body = {
        "endpoint_id": endpoint.id,
        "test_type": "concurrency",
        "parameters": {"selected_concurrencies": [1], "rounds_per_level": 1, "max_tokens": 4},
        "run_config": {"custom_params": PARAMS},
    }
    assert client.post("/api/v1/jobs/plan", json=body).status_code == 401
    assert client.post("/api/v1/jobs/plan", json=body, headers=headers).status_code == 200
    job = client.post("/api/v1/jobs", json=body, headers=headers)
    assert job.status_code == 201
    assert store.get(job.json()["job_id"])["parameters"]["_run_config"]["custom_params"] == PARAMS
    preset = client.post(
        "/api/v1/presets", json={**body, "name": "Typed params", "description": ""}, headers=headers
    )
    assert preset.status_code == 201 and preset.json()["run_config"]["custom_params"] == PARAMS
    saved = client.get("/api/v1/presets", headers=headers).json()["items"]
    assert saved[0]["run_config"]["custom_params"] == PARAMS
    body["run_config"]["custom_params"] = [{"name": "top_p", "location": "top_level"}]
    assert client.post("/api/v1/jobs/plan", json=body, headers=headers).status_code == 422


@pytest.mark.asyncio
async def test_native_http_payload_preserves_typed_values(lab, monkeypatch):  # noqa: F811
    _, endpoint, settings, _ = lab
    from unittest.mock import MagicMock

    placeholder = MagicMock()
    runner = BenchmarkRunner(
        placeholder=placeholder,
        progress_bar=placeholder,
        status_text=placeholder,
        api_base_url=endpoint.api_base_url,
        model_id=endpoint.model_id,
        api_key=endpoint.api_key(),
        tokenizer_option="字符数 (Fallback)",
        csv_filename=str(settings.artifact_root / "typed.csv"),
        log_placeholder=placeholder,
        provider="OpenAI",
        enable_live_log_server=False,
        custom_params=RunConfig(custom_params=PARAMS).model_dump()["custom_params"],
    )

    def handle(request):
        payload = json.loads(request.content)
        assert payload["top_p"] == 0.9 and type(payload["top_p"]) is float
        assert payload["chat_template_kwargs"]["enable_thinking"] is False
        assert payload["stop"] == ["END", "STOP"] and payload["optional_flag"] is None
        assert payload["optional_top"] is None
        assert payload["max_tokens"] == 4
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\ndata: [DONE]\n\n',
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await runner.get_completion(client, 1, "Q", 4)
    assert result["error"] is None and result["decode_tokens"] == 2


@pytest.mark.asyncio
async def test_raw_extra_body_cannot_override_model_or_allocate_client(monkeypatch):
    from core.providers.openai import OpenAIProvider

    monkeypatch.setattr(
        "core.providers.openai.httpx.AsyncClient", lambda **kw: pytest.fail("client allocation")
    )
    provider = OpenAIProvider("http://127.0.0.1:9/v1", "synthetic", "model")
    result = await provider.get_completion(
        None, 1, prompt="Q", _custom_extra_body={"model": "other"}
    )
    assert result["error"] and "override" in result["error"]

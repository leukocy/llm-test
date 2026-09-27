"""Model discovery only reads bounded catalogs from configured providers."""

from __future__ import annotations

import httpx
import pytest

from server.model_discovery import ModelDiscoveryError, discover_models, measure_reference_latency
from server.settings import Endpoint


@pytest.mark.asyncio
async def test_openai_model_list_filters_invalid_ids_and_keeps_credential_private():
    key = "model-list-secret"  # pragma: allowlist secret

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.deepseek.com/v1/models"
        assert request.headers["Authorization"] == f"Bearer {key}"
        return httpx.Response(
            200,
            json={"data": [{"id": "good/model"}, {"id": "good/model"}, {"id": "../bad"}]},
        )

    endpoint = Endpoint(
        "lab", "Lab", "OpenAI", "https://api.deepseek.com/v1", "model", "KEY", api_key_value=key
    )
    result = await discover_models(endpoint, transport=httpx.MockTransport(handler))
    assert result == {"items": ["good/model"], "truncated": False}
    assert key not in str(result)


@pytest.mark.asyncio
async def test_gemini_catalog_uses_header_and_does_not_follow_redirect():
    endpoint = Endpoint(
        "lab",
        "Lab",
        "Gemini",
        "https://generativelanguage.googleapis.com",
        "gemini-2.5-flash",
        "KEY",
        api_key_value="gemini-secret",  # pragma: allowlist secret
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://generativelanguage.googleapis.com/v1beta/models"
        assert request.headers["x-goog-api-key"] == "gemini-secret"
        return httpx.Response(
            200, json={"models": [{"name": "models/gemini-2.5-flash"}], "nextPageToken": "next"}
        )

    result = await discover_models(endpoint, transport=httpx.MockTransport(handler))
    assert result == {"items": ["gemini-2.5-flash"], "truncated": True}

    with pytest.raises(ModelDiscoveryError, match="HTTP 302"):
        await discover_models(
            endpoint,
            transport=httpx.MockTransport(
                lambda _: httpx.Response(302, headers={"Location": "http://169.254.169.254"})
            ),
        )


@pytest.mark.asyncio
async def test_reference_latency_uses_response_headers_without_download():
    endpoint = Endpoint(
        "lab",
        "Lab",
        "OpenAI",
        "https://api.deepseek.com/v1",
        "model",
        "KEY",
        api_key_value="secret",  # pragma: allowlist secret
    )
    result = await measure_reference_latency(
        endpoint,
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"large-body")),
    )
    assert result["reference_ms"] >= 0
    assert result["method"] == "GET /models response headers"

"""Bounded model listing for an already configured measurement endpoint."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from server.settings import Endpoint, validate_endpoint

_MODEL_ID = re.compile(r"[A-Za-z0-9._/@:+-]{1,200}")
_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_MODELS = 200


class ModelDiscoveryError(RuntimeError):
    pass


async def measure_reference_latency(
    endpoint: Endpoint, *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    """Time to response headers for a small authenticated catalog GET.

    This includes provider processing and is deliberately labelled a reference
    value rather than a pure network RTT.
    """
    validate_endpoint(endpoint)
    key = endpoint.api_key()
    gemini = endpoint.provider == "Gemini"
    url = (
        f"{endpoint.api_base_url.rstrip('/')}/v1beta/models"
        if gemini
        else f"{endpoint.api_base_url.rstrip('/')}/models"
    )
    headers = {"x-goog-api-key": key} if gemini else {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(5.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            start = time.monotonic()
            async with client.stream("GET", url, headers=headers) as response:
                elapsed = round((time.monotonic() - start) * 1000)
                if response.status_code >= 300:
                    raise ModelDiscoveryError(
                        f"模型列表接口返回 HTTP {response.status_code}，无法测量参考耗时"
                    )
    except httpx.TimeoutException as exc:
        raise ModelDiscoveryError("参考耗时测量超时") from exc
    except httpx.RequestError as exc:
        raise ModelDiscoveryError("参考耗时测量连接失败") from exc
    return {"reference_ms": elapsed, "method": "GET /models response headers"}


async def discover_models(
    endpoint: Endpoint, *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    """Read one page of a provider model catalog without exposing its credential.

    The endpoint is revalidated immediately before the outbound request. Redirects,
    environment proxies and unbounded response bodies are disabled for this lookup.
    """
    validate_endpoint(endpoint)
    key = endpoint.api_key()
    base = endpoint.api_base_url.rstrip("/")
    gemini = endpoint.provider == "Gemini"
    url = f"{base}/v1beta/models" if gemini else f"{base}/models"
    headers = {"x-goog-api-key": key} if gemini else {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(10.0),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code in {401, 403}:
                    raise ModelDiscoveryError("认证失败，请检查 API key")
                if response.status_code == 404:
                    raise ModelDiscoveryError("此服务商未提供模型列表，请手动填写模型 ID")
                if response.status_code != 200:
                    raise ModelDiscoveryError(f"模型列表请求失败（HTTP {response.status_code}）")
                content = bytearray()
                async for chunk in response.aiter_bytes():
                    content.extend(chunk)
                    if len(content) > _MAX_RESPONSE_BYTES:
                        raise ModelDiscoveryError("模型列表响应过大")
    except httpx.TimeoutException as exc:
        raise ModelDiscoveryError("模型列表请求超时") from exc
    except httpx.RequestError as exc:
        raise ModelDiscoveryError("模型列表连接失败") from exc

    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelDiscoveryError("模型列表不是有效 JSON") from exc
    if not isinstance(payload, dict):
        raise ModelDiscoveryError("模型列表格式无效")
    entries = payload.get("models" if gemini else "data")
    if not isinstance(entries, list):
        raise ModelDiscoveryError("模型列表格式无效")
    ids: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidate = entry.get("name" if gemini else "id")
        if gemini and isinstance(candidate, str) and candidate.startswith("models/"):
            candidate = candidate.removeprefix("models/")
        if not isinstance(candidate, str) or not _MODEL_ID.fullmatch(candidate):
            continue
        if candidate.startswith("/") or any(
            part in {"", ".", ".."} for part in candidate.split("/")
        ):
            continue
        if candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
        if len(ids) >= _MAX_MODELS:
            break
    return {
        "items": sorted(ids, key=str.casefold),
        "truncated": len(entries) > _MAX_MODELS or bool(payload.get("nextPageToken")),
    }

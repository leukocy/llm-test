"""Native Messages API with cumulative usage and explicit stream completion."""

import asyncio
import json
import time
from typing import Any

import httpx

from .base import LLMProvider, get_request_timeout_seconds
from .openai import (
    is_stop_requested,
    register_client,
    register_stream,
    unregister_client,
    unregister_stream,
)


def anthropic_url(base: str, resource: str) -> str:
    base = base.rstrip("/")
    return f"{base}/{resource}" if base.endswith("/v1") else f"{base}/v1/{resource}"


def anthropic_headers(key: str) -> dict[str, str]:
    return {"x-api-key": key, "anthropic-version": "2023-06-01"}


class AnthropicProvider(LLMProvider):
    async def get_completion(
        self,
        client,
        session_id: int,
        prompt: str = "",
        max_tokens: int = 256,
        log_callback=None,
        messages: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        if is_stop_requested():
            raise asyncio.CancelledError("Test stopped by user.")
        timeout = get_request_timeout_seconds(
            prompt=prompt,
            messages=messages,
            input_tokens=kwargs.pop("input_tokens_hint", None),
            request_timeout=kwargs.pop("request_timeout", None),
        )
        barrier = kwargs.pop("_barrier", None)
        payload: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "stream": True,
            "messages": [],
        }
        system = []
        for message in messages or [{"role": "user", "content": prompt}]:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                if not isinstance(content, str):
                    return {"error": "Anthropic system messages must contain text"}
                system.append(content)
            elif role in {"user", "assistant"}:
                payload["messages"].append({"role": role, "content": content})
            else:
                return {"error": "Unsupported Anthropic message role"}
        if system:
            payload["system"] = "\n\n".join(system)
        if not payload["messages"]:
            return {"error": "Anthropic requires at least one user/assistant message"}
        temperature = kwargs.pop("temperature", None)
        enabled = kwargs.pop("thinking_enabled", None)
        budget = kwargs.pop("thinking_budget", None)
        effort = kwargs.pop("reasoning_effort", None)
        if enabled is False:
            payload["thinking"] = {"type": "disabled"}
        elif enabled or budget:
            if budget is not None:
                if (
                    not isinstance(budget, int)
                    or isinstance(budget, bool)
                    or not 1024 <= budget < max_tokens
                ):
                    return {"error": "Anthropic thinking budget must be >=1024 and < max_tokens"}
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            else:
                payload["thinking"] = {"type": "adaptive"}
            if temperature not in (None, 1, 1.0):
                return {"error": "Anthropic thinking requires default temperature (1)"}
        elif temperature is not None:
            payload["temperature"] = temperature
        if effort and enabled is not False:
            payload["output_config"] = {"effort": effort}
        extra = kwargs.pop("_custom_extra_body", None) or {}
        if any(
            key in payload or key in {"system", "messages", "model", "stream", "max_tokens"}
            for key in extra
        ):
            return {
                "error": "Anthropic extra parameters cannot override measured request parameters"
            }
        payload.update(extra)
        payload.update({key: value for key, value in kwargs.items() if key not in payload})
        own_client = client is None
        if own_client:
            client = httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False)
        client_id = register_client(client)
        task = asyncio.current_task()
        register_stream(task)
        text = ""
        reasoning = ""
        first = None
        timestamps = []
        usage: dict[str, Any] = {}
        stopped = False
        try:
            if barrier is not None:
                await barrier.wait()
            created_at = time.time()
            start = time.monotonic()
            async with client.stream(
                "POST",
                anthropic_url(self.api_base_url, "messages"),
                json=payload,
                headers=anthropic_headers(self.api_key),
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    return {"error": f"Anthropic API returned HTTP {response.status_code}"}
                pending: list[str] = []
                async for line in response.aiter_lines():
                    if is_stop_requested():
                        raise asyncio.CancelledError("Test stopped by user.")
                    if line.startswith("data:"):
                        pending.append(line[5:].lstrip())
                        continue
                    if line or not pending:
                        continue
                    event = json.loads("\n".join(pending))
                    pending.clear()
                    kind = event.get("type")
                    if kind == "error":
                        return {"error": "Anthropic stream reported an API error"}
                    if kind == "message_start":
                        usage.update(event.get("message", {}).get("usage", {}))
                    elif kind == "message_delta":
                        usage.update(event.get("usage", {}))
                    elif kind == "message_stop":
                        stopped = True
                    elif kind in {"content_block_start", "content_block_delta"}:
                        block = (
                            event.get("content_block", {})
                            if kind == "content_block_start"
                            else event.get("delta", {})
                        )
                        part = (
                            block.get("text", "")
                            if block.get("type") in {"text", "text_delta"}
                            else ""
                        )
                        thought = (
                            block.get("thinking", "")
                            if block.get("type") in {"thinking", "thinking_delta"}
                            else ""
                        )
                        if part or thought:
                            now = time.monotonic()
                            if first is None:
                                first = now
                            timestamps.append(now)
                            text += part
                            reasoning += thought
            if not stopped or pending:
                return {"error": "Anthropic stream ended without a complete message_stop event"}
            if not text.strip():
                return {"error": "Anthropic returned no answer text"}
            normalized: dict[str, Any] = {"anthropic_usage": usage}
            inputs = [
                usage.get(key)
                for key in (
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            ]
            if any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int) or value < 0)
                for value in inputs + [usage.get("output_tokens")]
            ):
                return {"error": "Anthropic returned invalid token usage"}
            # Cache reads/writes are disjoint from uncached input, not subsets.
            if isinstance(inputs[0], int) and not isinstance(inputs[0], bool):
                normalized["prompt_tokens"] = sum(value or 0 for value in inputs)
            if usage.get("output_tokens") is not None:
                normalized["completion_tokens"] = usage["output_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                normalized["prompt_tokens_details"] = {
                    "cached_tokens": usage["cache_read_input_tokens"]
                }
            end = time.monotonic()
            from ..request_logger import get_request_logger

            logger = get_request_logger()
            if logger:
                logger.log_request(
                    session_id=str(session_id),
                    provider="AnthropicProvider",
                    model_id=self.model_id,
                    platform="anthropic",
                    api_base_url=self.api_base_url,
                    headers=anthropic_headers(self.api_key),
                    payload=payload,
                    thinking_enabled=enabled,
                    thinking_budget=budget,
                    reasoning_effort=effort,
                    full_response_content=text,
                    reasoning_content=reasoning,
                    usage_info=normalized,
                    created_at=created_at,
                    start_time=start,
                    first_token_time=first,
                    end_time=end,
                    token_timestamps=timestamps,
                )
            return {
                "created_at": created_at,
                "timing_clock": "client_monotonic",
                "start_time": start,
                "first_token_time": first,
                "end_time": end,
                "full_response_content": text,
                "reasoning_content": reasoning,
                "usage_info": normalized,
                "token_timestamps": timestamps,
                "error": None,
            }
        except asyncio.CancelledError:
            raise
        except (httpx.HTTPError, ValueError, TypeError, AttributeError):
            return {"error": "Anthropic request failed or returned an invalid stream"}
        finally:
            unregister_stream(task)
            unregister_client(client_id)
            if own_client:
                await client.aclose()

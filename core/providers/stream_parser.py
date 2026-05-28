"""Pure parsers for provider streaming response lines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenAIStreamEvent:
    """Parsed OpenAI-compatible stream event."""

    done: bool = False
    raw_chunk: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    content: str = ""
    reasoning: str = ""
    has_choice: bool = False

    @property
    def text(self) -> str:
        return f"{self.reasoning}{self.content}"


def parse_openai_stream_line(line: str) -> OpenAIStreamEvent | None:
    """Parse one OpenAI-compatible SSE/plain JSON stream line.

    Empty lines, keep-alive comments, non-JSON payloads, and malformed JSON return
    ``None`` so callers can continue consuming the stream.
    """
    stripped = line.strip()
    if not stripped:
        return None

    if stripped.startswith("data:"):
        payload = stripped[5:].strip()
        if payload == "[DONE]":
            return OpenAIStreamEvent(done=True)
    elif stripped.startswith("{"):
        payload = stripped
    else:
        return None

    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None

    usage = chunk.get("usage") or None
    choices = chunk.get("choices") or []
    if not choices:
        return OpenAIStreamEvent(raw_chunk=chunk, usage=usage)

    choice = choices[0]
    delta = choice.get("delta") or choice.get("message") or choice.get("text") or {}
    if isinstance(delta, str):
        content = delta
        reasoning = ""
    else:
        content = delta.get("content") or ""
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("thought")
            or ""
        )

    return OpenAIStreamEvent(
        raw_chunk=chunk,
        usage=usage,
        content=content,
        reasoning=reasoning,
        has_choice=True,
    )

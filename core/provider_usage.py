"""Retain bounded OpenAI usage counts without copying arbitrary response metadata."""

from typing import Any


def openai_usage_counts(usage: dict[str, Any]) -> dict[str, Any] | None:
    def count(value):
        return value if type(value) is int and 0 <= value <= 10**12 else None

    result: dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = count(usage.get(key))
        if value is not None:
            result[key] = value
    for key, fields in (
        (
            "completion_tokens_details",
            (
                "reasoning_tokens",
                "audio_tokens",
                "accepted_prediction_tokens",
                "rejected_prediction_tokens",
            ),
        ),
        ("prompt_tokens_details", ("cached_tokens", "audio_tokens")),
    ):
        detail = usage.get(key)
        if isinstance(detail, dict):
            values = {field: count(detail.get(field)) for field in fields}
            values = {field: value for field, value in values.items() if value is not None}
            if values:
                result[key] = values
    return result or None

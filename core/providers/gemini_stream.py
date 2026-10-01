"""Native GenerateContent stream contracts and reported token scopes."""

import json
from typing import Any


def normalize_usage(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Invalid Gemini usage metadata")
    fields = {
        "promptTokenCount": "prompt_tokens",
        "candidatesTokenCount": "completion_tokens",
        "thoughtsTokenCount": "reasoning_tokens",
        "totalTokenCount": "total_tokens",
        "cachedContentTokenCount": "cached_tokens",
    }
    result: dict[str, Any] = {}
    for source, target in fields.items():
        value = raw.get(source)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Invalid Gemini token count")
        result[target] = value
    prompt, cached = result["prompt_tokens"], result["cached_tokens"]
    if prompt is not None and cached is not None and cached > prompt:
        raise ValueError("Gemini cached count exceeds prompt count")
    total = result["total_tokens"]
    known = [result[key] for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens")]
    if total is not None and total < sum(value for value in known if value is not None):
        raise ValueError("Gemini total count is smaller than reported component counts")
    result["prompt_tokens_details"] = {"cached_tokens": cached}
    result["completion_tokens_details"] = {"reasoning_tokens": result["reasoning_tokens"]}
    result["output_token_scope"] = "response_candidates_excluding_thoughts"
    result["provider_usage"] = raw
    return result


def split_parts(parts: list[dict]) -> tuple[str, str]:
    answer: list[str] = []
    reasoning: list[str] = []
    for part in parts:
        text, thought = part.get("text", ""), part.get("thought", False)
        if not isinstance(text, str) or type(thought) is not bool:
            raise ValueError("Invalid Gemini text/thought part")
        (reasoning if thought else answer).append(text)
    return "".join(answer), "".join(reasoning)


async def stream_chunks(response):
    """Decode SSE data fields, including multiline frames and a final EOF frame."""
    pending = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            pending.append(line[5:].removeprefix(" "))
        elif not line and pending:
            yield json.loads("\n".join(pending))
            pending = []
    if pending:
        yield json.loads("\n".join(pending))

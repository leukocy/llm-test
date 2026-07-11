from abc import ABC, abstractmethod
from typing import Any

DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
ULTRA_LONG_CONTEXT_TIMEOUTS: tuple[tuple[int, float], ...] = (
    (1_000_000, 7200.0),
    (520_000, 3600.0),
    (256_001, 1800.0),
    (128_000, 1200.0),
)


def _content_length(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, dict):
                total += _content_length(item.get("text", ""))
            else:
                total += _content_length(item)
        return total
    return len(str(content)) if content is not None else 0


def _coerce_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def estimate_request_context_size(
    prompt: str = "",
    messages: list[dict] | None = None,
    input_tokens: int | None = None,
) -> int:
    token_hint = _coerce_nonnegative_int(input_tokens)
    if token_hint is not None:
        return token_hint

    total = _content_length(prompt)
    for message in messages or []:
        total += _content_length(message.get("content", ""))
    return total


def get_request_timeout_seconds(
    *,
    prompt: str = "",
    messages: list[dict] | None = None,
    input_tokens: int | None = None,
    request_timeout: float | int | None = None,
    default_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> float:
    if request_timeout is not None:
        try:
            return max(1.0, float(request_timeout))
        except (TypeError, ValueError):
            pass

    context_size = estimate_request_context_size(prompt, messages, input_tokens)
    for threshold, timeout_seconds in ULTRA_LONG_CONTEXT_TIMEOUTS:
        if context_size >= threshold:
            return max(default_timeout, timeout_seconds)
    return default_timeout


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_base_url: str, api_key: str, model_id: str):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_id = model_id

    @abstractmethod
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
        """
        Get completion from the LLM provider.

        Args:
            client: The HTTP client to use (e.g., httpx.AsyncClient).
            session_id: Unique ID for the session.
            prompt: The prompt text (used as fallback if messages is None).
            max_tokens: Maximum tokens to generate.
            log_callback: Optional callback for logging updates.
            messages: Optional structured chat messages list.
                      If provided, takes precedence over prompt.

        Returns:
            A dictionary containing metrics and result data.
        """
        pass

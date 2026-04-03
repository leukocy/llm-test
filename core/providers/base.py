from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_base_url: str, api_key: str, model_id: str):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_id = model_id

    @abstractmethod
    async def get_completion(self, client, session_id: int, prompt: str, max_tokens: int, log_callback=None, **kwargs) -> dict[str, Any]:
        """
        Get completion from the LLM provider.

        Args:
            client: The HTTP client to use (e.g., httpx.AsyncClient).
            session_id: Unique ID for the session.
            prompt: The prompt text.
            max_tokens: Maximum tokens to generate.
            log_callback: Optional callback for logging updates.

        Returns:
            A dictionary containing metrics and result data.
        """
        pass

from .base import LLMProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider


def get_provider(provider_name: str, api_base_url: str, api_key: str, model_id: str) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.

    Args:
        provider_name: Name of the provider (e.g., "OpenAI", "Gemini (非兼容)")
        api_base_url: Base URL for the API
        api_key: API key
        model_id: Model ID

    Returns:
        An instance of the appropriate LLMProvider subclass
    """
    if "Gemini" in provider_name:
        return GeminiProvider(api_base_url, api_key, model_id)
    else:
        # Default to OpenAI-compatible
        return OpenAIProvider(api_base_url, api_key, model_id)

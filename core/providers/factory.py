from .base import LLMProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider

# 本地推理端点是合法场景(llama.cpp/LM Studio/Ollama/vLLM 均为私网地址),
# 放行私网但保留协议白名单与格式校验; 公网 URL 走默认拦截规则。
_LOCAL_ENDPOINT_PREFIXES = (
    "http://127.",
    "http://localhost",
    "http://0.0.0.0",
    "http://[::1]",
    "https://127.",
    "https://localhost",
)


def _validate_base_url(api_base_url: str) -> str:
    """SSRF 校验(安全审查 #4): 拒绝 file:/ftp: 等非 HTTP 协议与畸形 URL。

    返回规范化 URL; 校验失败抛 SSRFError, 由调用方转为用户可见错误。
    """
    from urllib.parse import urlparse

    from ..url_validator import SSRFError, is_safe_url, validate_and_normalize_url

    if not api_base_url:
        return api_base_url

    parsed = urlparse(api_base_url)
    # Gemini 端点无 /v1 路径且代码里按 "{base}/v1beta" 拼接,
    # 不能让 normalize 补尾斜杠(会产生 //v1beta), 这里只校验不重写。
    if "generativelanguage.googleapis.com" in (parsed.hostname or ""):
        is_safe, error = is_safe_url(api_base_url)
        if not is_safe:
            raise SSRFError(f"URL validation failed: {error}")
        return api_base_url

    # 空字符串占位("Custom (OpenAI Compatible)" 未填)交由后续请求阶段报错
    allow_private = api_base_url.strip().lower().startswith(_LOCAL_ENDPOINT_PREFIXES)
    return validate_and_normalize_url(api_base_url, allow_private=allow_private)


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
    validated_url = _validate_base_url(api_base_url)
    if "Gemini" in provider_name:
        return GeminiProvider(validated_url, api_key, model_id)
    else:
        # Default to OpenAI-compatible
        return OpenAIProvider(validated_url, api_key, model_id)

# 全局Configure与常量


# --- API ProviderConfigure ---
# 公共云服务商 API 端点
PROVIDER_OPTIONS = {
    "Custom (OpenAI Compatible)": "",
    "火山引擎 (Volcano)": "https://ark.cn-beijing.volces.com/api/v3",
    "深度求索 (DeepSeek)": "https://api.deepseek.com/v1",
    "MiMo": "https://api.xiaomimimo.com/v1",
    "智谱开放平台 (ZhiPu)": "https://open.bigmodel.cn/api/paas/v4",
    "阿里云百炼 (Alibaba)": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "MiniMax": "https://api.minimax.chat/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "硅基流动 (SiliconFlow)": "https://api.siliconflow.cn/v1",
    "月之暗面 (Kimi)": "https://api.moonshot.cn/v1",
    "Gemini": "https://generativelanguage.googleapis.com",
}

# --- Model预设 ---
MODEL_OPTIONS = [
    "DeepSeek-V3.1",
    "DeepSeek-V3.2",
    "mimo-v2-flash",
    "Qwen3.5-397B-A17B-FP8",
    "Qwen3-Coder-480B-A35B",
    "Qwen3-Next-80B-A3B",
    "Kimi-K2.5",
    "Qwen3-235B-A22B",
    "MiniMax-M2.5",
    "MiniMax-M2.1",
    "GLM-5",
    "gpt-oss-120b",
    "deepseek-v3-1-terminus",
    "deepseek-v3-2-251201",
    "XiaomiMiMo/MiMo-V2-Flash",
    "DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro"
]

# --- HuggingFace Model映射 (用于自动Align) ---
HF_MODEL_MAPPING = {
    "qwen3.5-397b": "./tokenizers/Qwen3.5-397B-A17B-FP8",
    "qwen3.5": "./tokenizers/Qwen3.5-397B-A17B-FP8",
    "deepseek-v4-flash": "./tokenizers/DeepSeek-V4-Flash",
    "deepseek-v4-pro": "./tokenizers/DeepSeek-V4-Pro",
    "deepseek-v4": "./tokenizers/DeepSeek-V4-Pro",
    "deepseek-v3.2": "./tokenizers/DeepSeek-V3.2",
    "deepseek": "./tokenizers/DeepSeek-V3.1-Terminus",
    "kimi-k2.5": "./tokenizers/Kimi-K2.5",
    "kimi-k2-thinking": "./tokenizers/Kimi-K2-Thinking",
    "kimi-k2": "./tokenizers/Kimi-K2.5",
    "kimi": "./tokenizers/Kimi-K2-Instruct-0905",
    "qwen3-235b": "./tokenizers/Qwen3-235B-A22B-Instruct-2507",
    "qwen3-coder": "./tokenizers/Qwen3-Coder-480B-A35B-Instruct",
    "qwen3-next": "./tokenizers/Qwen3-Next-80B-A3B-Instruct",
    "qwen3-vl": "./tokenizers/Qwen3-VL-32B-Instruct",
    "glm-5": "./tokenizers/GLM-5",
    "glm": "./tokenizers/GLM-4.6",
    "minimax-m2.5": "./tokenizers/MiniMax-M2.5",
    "minimax": "./tokenizers/MiniMax-M2",
    "gpt-oss": "./tokenizers/gpt-oss-120b",
    "llama": "./tokenizers/Llama-3.3-70B-Instruct",
    "mimo": "./tokenizers/MiMo-V2-Flash",
    "qwen": "./tokenizers/Qwen3-235B-A22B-Instruct-2507"
}

# --- Tokenizer本地目录 -> HuggingFace Hub Repo映射 (用于自动下载) ---
TOKENIZER_HF_MAPPING = {
    # DeepSeek系列
    "DeepSeek-V3.1-Terminus": "deepseek-ai/DeepSeek-V3.1-Terminus",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    # Qwen系列
    "Qwen3.5-397B-A17B-FP8": "Qwen/Qwen3.5-397B-A17B-FP8",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B",
    "Qwen3-Coder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen3-VL-32B-Instruct": "Qwen/Qwen3-VL-30B-Instruct",
    # Kimi系列
    "Kimi-K2.5": "moonshotai/Kimi-K2.5",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Thinking",
    "Kimi-K2-Instruct-0905": "moonshotai/Kimi-K2-Instruct",
    # GLM系列
    "GLM-5": "zai-org/GLM-5",
    "GLM-4.6": "zai-org/GLM-4.7",
    # MiniMax系列
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
    "MiniMax-M2": "MiniMaxAI/MiniMax-M2",
    # Llama系列
    "Llama-3.3-70B-Instruct": "meta-llama/Llama-3.3-70B-Instruct",
    # MiMo系列
    "MiMo-V2-Flash": "XiaomiMiMo/MiMo-V2-Flash",
    # 其他
    "gpt-oss-120b": "openai/gpt-oss-120b",  # fallback用gpt2做token计数
}

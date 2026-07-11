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
    "Gemini (非兼容)": "https://generativelanguage.googleapis.com",
    "llama.cpp": "http://127.0.0.1:8080/v1",
    "LM Studio": "http://127.0.0.1:1234/v1",
    "Ollama": "http://127.0.0.1:11434/v1",
}

# --- Model预设 ---
MODEL_OPTIONS = [
    "DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro",
    "DeepSeek-V3.1",
    "DeepSeek-V3.2",
    "mimo-v2-flash",
    "XiaomiMiMo/MiMo-V2.5",
    "Qwen3.5-397B-A17B-FP8",
    "Qwen3-Coder-480B-A35B",
    "Qwen3-Next-80B-A3B",
    "Kimi-K2.5",
    "Qwen3-235B-A22B",
    "MiniMax-M3",
    "MiniMax-M2.5",
    "MiniMax-M2.1",
    "GLM-5",
    "gpt-oss-120b",
    "deepseek-v3-1-terminus",
    "deepseek-v3-2-251201",
    "XiaomiMiMo/MiMo-V2-Flash",
    "stepfun-ai/Step-3.7-Flash",
    "google/gemma-4-31B-it",
]

# --- HuggingFace Model映射 (用于自动Align) ---
HF_MODEL_MAPPING = {
    "qwen3.5-397b": "./tokenizers/Qwen3.5-397B-A17B-FP8",
    "qwen3.5": "./tokenizers/Qwen3.5-397B-A17B-FP8",
    "qwen3.6": "./tokenizers/Qwen3.5-397B-A17B-FP8",
    "deepseek-v4-flash": "./tokenizers/DeepSeek-V4-Pro",
    "deepseek-v4-pro": "./tokenizers/DeepSeek-V4-Pro",
    "deepseek-v4": "./tokenizers/DeepSeek-V4-Pro",
    "deepseek-v3.2": "./tokenizers/DeepSeek-V3.2",
    "deepseek": "./tokenizers/DeepSeek-V3.1-Terminus",
    # kimi-k2.7 优先匹配:指向模型自带完整 tokenizer(与 vLLM 同源)。
    # ./tokenizers/Kimi-K2.5 缺 tool_declaration_ts.py 会加载失败→校准回退错 tokenizer→欠生成。
    "kimi-k2.7": "/DATA/Model/Kimi-K2.7-Code",
    "kimi-k2.5": "./tokenizers/Kimi-K2.5",
    "kimi-k2-thinking": "./tokenizers/Kimi-K2-Thinking",
    "kimi-k2": "./tokenizers/Kimi-K2.5",
    "kimi": "./tokenizers/Kimi-K2-Instruct-0905",
    "qwen3-235b": "./tokenizers/Qwen3-Next-80B-A3B-Instruct",
    "qwen3-coder": "./tokenizers/Qwen3-Coder-480B-A35B-Instruct",
    "qwen3-next": "./tokenizers/Qwen3-Next-80B-A3B-Instruct",
    "qwen3-vl": "./tokenizers/Qwen3-VL-32B-Instruct",
    "glm-5": "./tokenizers/GLM-5",
    "glm": "./tokenizers/GLM-4.6",
    "minimax-m3": "./tokenizers/MiniMax-M3",
    "minimax-m2.5": "./tokenizers/MiniMax-M2.5",
    "minimax": "./tokenizers/MiniMax-M3",
    "gpt-oss": "./tokenizers/gpt-oss-120b",
    "llama": "./tokenizers/Llama-3.3-70B-Instruct",
    "mimo-v2.5": "./tokenizers/MiMo-V2.5",
    "mimo": "./tokenizers/MiMo-V2-Flash",
    "qwen": "./tokenizers/Qwen3-Next-80B-A3B-Instruct",
    "step-3.7": "./tokenizers/Step-3.7-Flash",
    "step-3.7-flash": "./tokenizers/Step-3.7-Flash",
    "stepfun": "./tokenizers/Step-3.7-Flash",
    "stepfun-3.7": "./tokenizers/Step-3.7-Flash",
    "gemma-4-31b": "./tokenizers/Gemma-4-31B-it",
    "gemma-4": "./tokenizers/Gemma-4-31B-it",
    "gemma": "./tokenizers/Gemma-4-31B-it",
}

# --- Tokenizer 源配置 (统一维护，自动兼容) ---
TOKENIZER_SOURCES: dict[str, str | dict[str, str]] = {
    # 字符串值：MS 和 HF 地址相同
    "DeepSeek-V3.1-Terminus": "deepseek-ai/DeepSeek-V3.1-Terminus",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    "Qwen3.5-397B-A17B-FP8": "Qwen/Qwen3.5-397B-A17B-FP8",
    "Qwen3-Coder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen3-VL-32B-Instruct": "Qwen/Qwen3-VL-30B-Instruct",
    "Kimi-K2.5": "moonshotai/Kimi-K2.5",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Thinking",
    "Kimi-K2-Instruct-0905": "moonshotai/Kimi-K2-Instruct",
    "MiMo-V2-Flash": "XiaomiMiMo/MiMo-V2-Flash",
    "MiMo-V2.5": "XiaomiMiMo/MiMo-V2.5",
    "Step-3.7-Flash": "stepfun-ai/Step-3.7-Flash",
    "Gemma-4-31B-it": "google/gemma-4-31B-it",
    # 字典值：MS 和 HF 地址不同
    "GLM-5": {"ms": "ZhipuAI/GLM-5", "hf": "zai-org/GLM-5"},
    "GLM-4.6": {"ms": "ZhipuAI/GLM-4.7", "hf": "zai-org/GLM-4.7"},
    "MiniMax-M2.5": {"ms": "MiniMax/MiniMax-M2.5", "hf": "MiniMaxAI/MiniMax-M2.5"},
    "Llama-3.3-70B-Instruct": {
        "ms": "LLM-Research/Llama-3.3-70B-Instruct",
        "hf": "meta-llama/Llama-3.3-70B-Instruct",
    },
    "gpt-oss-120b": {"ms": "openai-mirror/gpt-oss-120b", "hf": "openai/gpt-oss-120b"},
    "MiniMax-M3": "MiniMax/MiniMax-M3",
}

# 兼容层：从 TOKENIZER_SOURCES 自动生成旧变量
TOKENIZER_MODELSCOPE_MAPPING = {
    k: (v if isinstance(v, str) else v["ms"]) for k, v in TOKENIZER_SOURCES.items()
}

TOKENIZER_HF_MAPPING = {
    k: (v if isinstance(v, str) else v["hf"]) for k, v in TOKENIZER_SOURCES.items()
}

# --- Tokenizer本地目录 -> ModelScope 魔搭社区 Repo映射 (国内优先) ---
TOKENIZER_MODELSCOPE_MAPPING = {
    # DeepSeek系列
    "DeepSeek-V3.1-Terminus": "deepseek-ai/DeepSeek-V3.1-Terminus",
    "DeepSeek-V3.2": "deepseek-ai/DeepSeek-V3.2",
    "DeepSeek-V4-Flash": "deepseek-ai/DeepSeek-V4-Flash",
    "DeepSeek-V4-Pro": "deepseek-ai/DeepSeek-V4-Pro",
    # Qwen系列
    "Qwen3.5-397B-A17B-FP8": "Qwen/Qwen3.5-397B-A17B-FP8",
    "Qwen3-235B-A22B-Instruct-2507": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-Coder-480B-A35B-Instruct": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen3-Next-80B-A3B-Instruct": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "Qwen3-VL-32B-Instruct": "Qwen/Qwen3-VL-32B-Instruct",
    # Kimi系列
    "Kimi-K2.5": "moonshotai/Kimi-K2.5",
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Thinking",
    "Kimi-K2-Instruct-0905": "moonshotai/Kimi-K2-Instruct-0905",
    # GLM系列
    "GLM-5": "ZhipuAI/GLM-5",
    "GLM-4.6": "ZhipuAI/GLM-4.7",
    # MiniMax系列
    "MiniMax-M2.5": "MiniMaxAI/MiniMax-M2.5",
    "MiniMax-M2": "MiniMaxAI/MiniMax-M2",
    # Llama系列
    "Llama-3.3-70B-Instruct": "LLM-Research/Llama-3.3-70B-Instruct",
    # MiMo系列
    "MiMo-V2-Flash": "XiaomiMiMo/MiMo-V2-Flash",
}

"""
混合推理Model参数管理模块

not同平台对推理Model参数命名and格式各not相同:
- 小米 MiMo: enable_thinking (bool), thinking_budget (int)
- 硅基流动: enable_thinking (bool), thinking_budget (int)
- 火山引擎: enable_thinking (bool), thinking_budget (int)
- 阿里百炼: enable_thinking (bool), thinking_budget (int)
- OpenAI: reasoning_effort (str: "low"/"medium"/"high")
"""

from typing import Any, Dict

# 平台识别规则
PLATFORM_PATTERNS = {
    "mimo": ["mimo", "xiaomi"],
    "siliconflow": ["siliconflow", "silicon"],
    "volcano": ["volcengine", "volcano", "doubao"],
    "aliyun": ["aliyun", "dashscope", "qwen"],
    "minimax": ["minimax", "minimaxi"],
    "deepseek": ["deepseek.com", "api.deepseek"],  # DeepSeek官方API
    "zhipu": ["zhipuai", "bigmodel.cn"],  # 智谱AI
    "gemini": ["gemini", "generativelanguage.googleapis.com", "google"],
    "openrouter": ["openrouter"],
    "openai": ["openai", "api.openai.com"]
}


# ============================================================================
# Phase 1: 平台特性表 (Platform Features)
# 记录各平台参数位置、格式、支持能力and响应字段
# ============================================================================
PLATFORM_FEATURES = {
    "mimo": {
        "name": "小米 MiMo",
        "thinking_param_location": "top_level",  # 参数位置: top_level / extra_body / generationConfig
        "thinking_field": "thinking",  # 参数字段名
        "thinking_format": {"type": "enabled|disabled"},  # 参数格式
        "supports_budget": False,  # is否支持 thinking_budget
        "supports_effort": False,  # is否支持 reasoning_effort
        "reasoning_output_field": "reasoning_content",  # 响应in推理内容字段名
        "content_output_field": "content",  # 响应in正文内容字段名
        "special_headers": {"api-key": True},  # 特殊请求头 (MiMo 用 api-key 而非 Authorization)
        "special_params": {"max_completion_tokens": True},  # 特殊参数名
        "notes": "thinking mustis顶级参数，not能放 extra_body"
    },
    "deepseek": {
        "name": "DeepSeek",
        "thinking_param_location": "extra_body",
        "thinking_field": "thinking",
        "thinking_format": {"type": "enabled|disabled"},
        "supports_budget": False,
        "supports_effort": False,
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "thinking 需放in extra_body 内"
    },
    "zhipu": {
        "name": "智谱 AI",
        "thinking_param_location": "top_level",
        "thinking_field": "thinking",
        "thinking_format": {"type": "enabled|disabled"},
        "supports_budget": False,
        "supports_effort": False,
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "thinking is顶级参数，与 MiMo 类似"
    },
    "gemini": {
        "name": "Google Gemini",
        "thinking_param_location": "generationConfig.thinkingConfig",
        "thinking_field": "thinkingConfig",
        "thinking_format": {"includeThoughts": True, "thinkingLevel": "HIGH", "thinkingBudget": -1},
        "supports_budget": True,  # thinkingBudget: -1 表示自动
        "supports_effort": True,  # thinkingLevel: MINIMAL/LOW/MEDIUM/HIGH
        "reasoning_output_field": "thought",  # Gemini 用 thought 而非 reasoning_content
        "content_output_field": "text",  # Gemini 用 text 而非 content
        "special_headers": {},
        "special_params": {},
        "notes": "REST API v1beta Not supported thinkingConfig，需用 SDK or特定Model隐式Trigger"
    },
    "volcano": {
        "name": "火山引擎 (豆包)",
        "thinking_param_location": "extra_body",
        "thinking_field": "thinking",
        "thinking_format": {"type": "enabled|disabled"},
        "supports_budget": False,
        "supports_effort": True,  # reasoning.effort: minimal/low/medium/high
        "effort_levels": ["minimal", "low", "medium", "high"],
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "支持 reasoning.effort 控制思考深度，仅部分Model支持"
    },
    "aliyun": {
        "name": "阿里云百炼",
        "thinking_param_location": "extra_body",
        "thinking_field": "enable_thinking",  # 布尔值，notis对象
        "thinking_format": True,  # 直接传 True/False
        "supports_budget": True,  # thinking_budget: 128-32768
        "budget_field": "thinking_budget",
        "budget_range": [128, 32768],
        "supports_effort": False,
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "use enable_thinking (bool) + thinking_budget (int)"
    },
    "siliconflow": {
        "name": "硅基流动",
        "thinking_param_location": "top_level",
        "thinking_field": "enable_thinking",
        "thinking_format": True,
        "supports_budget": True,
        "budget_field": "thinking_budget",
        "budget_range": [128, 32768],
        "supports_effort": False,
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "use enable_thinking + thinking_budget 作is顶级参数"
    },
    "minimax": {
        "name": "MiniMax",
        "thinking_param_location": "extra_body",
        "thinking_field": "reasoning_split",
        "thinking_format": True,  # 布尔值
        "supports_budget": False,
        "supports_effort": False,
        "reasoning_output_field": "reasoning_details",  # MiniMax 用 reasoning_details
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "use Interleaved Thinking（交错思维）"
    },
    "openrouter": {
        "name": "OpenRouter",
        "thinking_param_location": "top_level",
        "thinking_field": "reasoning",
        "thinking_format": {"effort": "high"},
        "supports_budget": False,
        "supports_effort": True,
        "effort_levels": ["low", "medium", "high"],
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {"HTTP-Referer": True, "X-Title": True},
        "special_params": {},
        "notes": "与 OpenAI 格式兼容"
    },
    "openai": {
        "name": "OpenAI",
        "thinking_param_location": "top_level",
        "thinking_field": "reasoning",
        "thinking_format": {"effort": "high"},
        "supports_budget": False,
        "supports_effort": True,
        "effort_levels": ["none", "minimal", "low", "medium", "high", "xhigh"],
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "o1/o3 系列use reasoning.effort"
    }
}


def get_platform_features(platform: str) -> dict[str, Any]:
    """
    Get指定平台特性Configure

    Args:
        platform: 平台标识

    Returns:
        平台特性字典，if平台未知则ReturndefaultConfigure
    """
    return PLATFORM_FEATURES.get(platform, {
        "name": "未知平台",
        "thinking_param_location": "top_level",
        "thinking_field": "enable_thinking",
        "thinking_format": True,
        "supports_budget": True,
        "supports_effort": False,
        "reasoning_output_field": "reasoning_content",
        "content_output_field": "content",
        "special_headers": {},
        "special_params": {},
        "notes": "通用Configure"
    })


def get_reasoning_field(platform: str) -> str:
    """
    based on平台Return推理内容字段名

    Args:
        platform: 平台标识

    Returns:
        推理内容字段名 (如 reasoning_content, thought, reasoning_details)
    """
    features = get_platform_features(platform)
    return features.get("reasoning_output_field", "reasoning_content")


def get_content_field(platform: str) -> str:
    """
    based on平台Return正文内容字段名

    Args:
        platform: 平台标识

    Returns:
        正文内容字段名 (如 content, text)
    """
    features = get_platform_features(platform)
    return features.get("content_output_field", "content")


def supports_thinking_budget(platform: str) -> bool:
    """
    Check平台is否支持 thinking_budget 参数

    Args:
        platform: 平台标识

    Returns:
        is否支持
    """
    features = get_platform_features(platform)
    return features.get("supports_budget", False)


def supports_reasoning_effort(platform: str) -> bool:
    """
    Check平台is否支持 reasoning_effort 参数

    Args:
        platform: 平台标识

    Returns:
        is否支持
    """
    features = get_platform_features(platform)
    return features.get("supports_effort", False)


def get_effort_levels(platform: str) -> list:
    """
    Get平台支持 effort etc.级列表

    Args:
        platform: 平台标识

    Returns:
        支持etc.级列表，如 ["low", "medium", "high"]
    """
    features = get_platform_features(platform)
    return features.get("effort_levels", ["low", "medium", "high"])



def detect_platform(api_base_url: str, model_id: str = "") -> str:
    """
    based on API URL orModelID检测平台

    Args:
        api_base_url: API基础URL
        model_id: ModelID

    Returns:
        平台标识: "mimo", "siliconflow", "volcano", "aliyun", "openai" or "unknown"
    """
    url_lower = api_base_url.lower()
    model_lower = model_id.lower()

    # Pass 1: 特定的 URL 匹配 (Highest priority)
    URL_PATTERNS = {
        "aliyun": ["dashscope.aliyuncs.com"],
        "volcano": ["ark.cn-beijing.volces.com", "ark.volcengine.com"],
        "siliconflow": ["api.siliconflow.cn"],
        "deepseek": ["api.deepseek.com"],
        "mimo": ["api.xiaomimimo.com"],
        "zhipu": ["open.bigmodel.cn", "paas/v4"],
        "minimax": ["api.minimax.chat"],
        "gemini": ["generativelanguage.googleapis.com"],
        "openrouter": ["openrouter.ai"],
        "openai": ["api.openai.com"]
    }

    for platform, patterns in URL_PATTERNS.items():
        if any(p in url_lower for p in patterns):
            return platform

    # Pass 2: Local IP/Host Detection
    # ifis本地部署，typicallyuse standard OpenAI/vLLM 格式，returunknown以避免注入平台特定参数
    is_local = any(p in url_lower for p in ["127.0.0.1", "localhost", "10.", "192.168.", "::1"])
    if is_local:
        return "unknown"

    # Pass 3: 基于 PLATFORM_PATTERNS 模糊匹配 (Fallback)
    for platform, patterns in PLATFORM_PATTERNS.items():
        for pattern in patterns:
            if pattern in url_lower or pattern in model_lower:
                return platform

    return "unknown"


def build_thinking_params(
    thinking_enabled: bool,
    thinking_budget: int,
    reasoning_effort: str,
    platform: str
) -> dict[str, Any]:
    """
    based on平台Build推理参数

    Args:
        thinking_enabled: is否启用Thinking mode
        thinking_budget: 思考token预算
        reasoning_effort: OpenAI风格推理强度 ("low", "medium", "high")
        platform: 平台标识

    Returns:
        适合该平台参数字典（可能包含 extra_body）
    """
    params = {}

    if platform == "openai" or platform == "openrouter":
        # OpenAI and OpenRouter use reasoning 对象: {"effort": "low" | "medium" | "high" | "xhigh"}
        if thinking_enabled is not None or reasoning_effort:
            effort_value = "high"  # default高档
            if reasoning_effort:
                # 支持: none, minimal, low, medium, high, xhigh
                effort_map = {
                    "low": "low",
                    "medium": "medium",
                    "high": "high"
                }
                effort_value = effort_map.get(reasoning_effort, "high")

            params["reasoning"] = {"effort": effort_value}

    elif platform == "gemini":
        # Google Gemini use generationConfig.thinkingConfig
        # includeThoughts: bool, thinkingLevel: str, thinkingBudget: int
        if thinking_enabled is not None or reasoning_effort or thinking_budget:
            thinking_level = "HIGH"  # default高档
            if reasoning_effort:
                level_map = {
                    "low": "LOW",
                    "medium": "MEDIUM",
                    "high": "HIGH"
                }
                thinking_level = level_map.get(reasoning_effort, "HIGH")

            t_config = {
                "includeThoughts": True if thinking_enabled is not None else True,
            }

            # if启用Thinking modebut没has指定预算，canuse -1 (由Model自行决定)
            # or者直接透传用户 thinking_budget
            if thinking_budget is not None:
                t_config["thinkingBudget"] = thinking_budget
            elif thinking_enabled:
                t_config["thinkingLevel"] = thinking_level

            params["_generation_config_gemini"] = {
                "thinkingConfig": t_config
            }

    elif platform == "minimax":
        # MiniMax use reasoning_split (bool)
        # needvia extra_body 传递
        if thinking_enabled is not None:
            # 标记isneedin extra_body inProcess
            params["_extra_body_reasoning_split"] = thinking_enabled

    elif platform == "aliyun":
        # 阿里云百炼use extra_body 传递 enable_thinking and thinking_budget
        extra_body_params = {}
        if thinking_enabled is not None:
            extra_body_params["enable_thinking"] = thinking_enabled
        if thinking_budget and thinking_budget > 0:
            extra_body_params["thinking_budget"] = thinking_budget

        if extra_body_params:
            # 标记isneedin extra_body inProcess
            params["_extra_body_aliyun"] = extra_body_params

    elif platform == "mimo":
        # 小米 MiMo use thinking 对象，顶级参数，格式: {"type": "enabled" | "disabled"}
        if thinking_enabled is not None:
            thinking_type = "enabled" if thinking_enabled else "disabled"
            params["thinking"] = {"type": thinking_type}
        # MiMo Not supported thinking_budget

    elif platform == "deepseek":
        # DeepSeek 官方API use与 MiMo 相同 thinking 对象格式
        # 格式: {"type": "enabled" | "disabled"}
        # needvia extra_body 传递
        if thinking_enabled is not None:
            thinking_type = "enabled" if thinking_enabled else "disabled"
            params["_extra_body_deepseek"] = {
                "thinking": {"type": thinking_type}
            }
        # DeepSeek notuse thinking_budget，而isuse max_tokens (default32K，最大64K)

    elif platform == "zhipu":
        # 智谱 AI use与 MiMo 相似 thinking 对象格式，直接作is顶级参数
        # 格式: {"thinking": {"type": "enabled" | "disabled"}}
        if thinking_enabled is not None:
            thinking_type = "enabled" if thinking_enabled else "disabled"
            params["thinking"] = {"type": thinking_type}

    elif platform == "volcano":
        # 火山引擎use thinking 对象 + reasoning.effort
        # needvia extra_body 传递
        extra_body_params = {}

        if thinking_enabled is not None:
            thinking_type = "enabled" if thinking_enabled else "disabled"
            extra_body_params["thinking"] = {"type": thinking_type}

        # reasoning.effort: minimal/low/medium/high
        if reasoning_effort:
            effort_map = {
                "low": "low",
                "medium": "medium",
                "high": "high"
            }
            effort_value = effort_map.get(reasoning_effort, "high")
            extra_body_params["reasoning"] = {"effort": effort_value}

        if extra_body_params:
            params["_extra_body_volcano"] = extra_body_params

    elif platform == "siliconflow":
        # 硅基流动 use顶级参数 enable_thinking and thinking_budget
        if thinking_enabled is not None:
            params["enable_thinking"] = thinking_enabled
        if thinking_budget and thinking_budget > 0:
            params["thinking_budget"] = thinking_budget

    else:
        # unknown or standard OpenAI-compatible platforms
        # typically do not use special thinking parameters, as thinking is often model-implicit (e.g. R1)
        pass

    return params


def get_default_thinking_config(platform: str) -> dict[str, Any]:
    """
    Get各平台default推理Configure

    Args:
        platform: 平台标识

    Returns:
        defaultConfigure字典
    """
    defaults = {
        "mimo": {
            "thinking": {"type": "enabled"}  # MiMo use thinking 对象，顶级参数
        },
        "siliconflow": {
            "enable_thinking": True,
            "thinking_budget": 32768  # 最大值
        },
        "deepseek": {
            "thinking": {"type": "enabled"}  # DeepSeek 官方API，与 MiMo 格式相同
        },
        "zhipu": {
            "thinking": {"type": "enabled"}  # 智谱 AI 格式
        },
        "volcano": {
            # 火山引擎via extra_body 传递
            "thinking": {"type": "enabled"},
            "reasoning": {"effort": "high"}  # Highest档位
        },
        "aliyun": {
            # 阿里云via extra_body 传递
            "enable_thinking": True,
            "thinking_budget": 32768 # 最大值（范围 128-32768）
        },
        "minimax": {
            "reasoning_split": True  # MiniMax via extra_body 传递
        },
        "gemini": {
            "thinkingConfig": {"thinkingLevel": "HIGH"}  # Gemini via generationConfig 传递
        },
        "openrouter": {
            "reasoning": {"effort": "high"}  # 与 OpenAI 相同格式
        },
        "openai": {
            "reasoning": {"effort": "high"}  # Highest档位 (none/minimal/low/medium/high/xhigh)
        }
    }

    return defaults.get(platform, {})


# 支持混合推理Model列表（部分示例，实际应该更完整）
THINKING_MODELS = {
    "mimo": ["mimo-v2-flash"],
    "siliconflow": [
        "zai-org/GLM-4.6",
        "Qwen/Qwen3-",
        "tencent/Hunyuan-A13B-Instruct",
        "deepseek-ai/DeepSeek-V3.1",
        "Pro/deepseek-ai/DeepSeek-V3.1"
    ],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],  # DeepSeek官方API
    "zhipu": ["glm-4.7", "glm-4-plus"],  # 智谱AI
    "volcano": ["doubao-pro", "doubao-thinking", "doubao-seed-"],
    "aliyun": [
        "qwen3-", "qwq-", "deepseek-v3", "deepseek-r1",
        "glm-4.6", "glm-4.5", "kimi-k2-thinking"
    ],
    "minimax": ["minimax-m2"],
    "gemini": ["gemini-3-flash-preview", "gemini-2.0-flash-thinking-exp", "gemini-flash-latest", "gemini-2.5-flash-preview"],
    "openrouter": ["*"],  # OpenRouter 支持多种推理Model
    "openai": ["o1-", "o3-", "gpt-5"]
}


def is_thinking_model(model_id: str, platform: str) -> bool:
    """
    判断is否is推理Model

    Args:
        model_id: ModelID
        platform: 平台标识

    Returns:
        is否is推理Model
    """
    model_lower = model_id.lower()
    model_patterns = THINKING_MODELS.get(platform, [])

    return any(pattern.lower() in model_lower for pattern in model_patterns)


def get_intelligent_preset(api_base_url: str, model_id: str) -> dict[str, Any]:
    """
    智能参数预设：based on平台andModel自动推荐最优Configure

    Args:
        api_base_url: API基础URL
        model_id: ModelID

    Returns:
        推荐thinking参数Configure
    """
    # 1. 检测平台
    platform = detect_platform(api_base_url, model_id)

    # 2. Checkis否is推理Model
    if not is_thinking_model(model_id, platform):
        return {
            "thinking_enabled": False,
            "thinking_budget": 0,
            "reasoning_effort": "medium",
            "platform": platform,
            "is_thinking_model": False
        }

    # 3. based on平台ReturnBestConfigure
    preset_configs = {
        "mimo": {
            "thinking_enabled": True,
            "thinking_budget": 0,  # MiMoNot supportedbudget
            "reasoning_effort": "medium",  # 仅用于记录，MiMonotuse
            "description": "小米MiMo - 官方推理模式"
        },
        "siliconflow": {
            "thinking_enabled": True,
            "thinking_budget": 32768,  # 最大值
            "reasoning_effort": "medium",
            "description": "硅基流动 - 最大Thinking budget"
        },
        "deepseek": {
            "thinking_enabled": True,
            "thinking_budget": 0,  # DeepSeekusemax_tokens而非thinking_budget
            "reasoning_effort": "medium",
            "description": "DeepSeek官方 - 深度Thinking mode"
        },
        "zhipu": {
            "thinking_enabled": True,
            "thinking_budget": 0,
            "reasoning_effort": "medium",
            "description": "智谱AI - GLM-4.7 Thinking mode"
        },
        "volcano": {
            "thinking_enabled": True,
            "thinking_budget": 0,  # 火山引擎notusebudget，用effort
            "reasoning_effort": "high",
            "description": "火山引擎 - 高强度推理"
        },
        "aliyun": {
            "thinking_enabled": True,
            "thinking_budget": 32768,  # 最大值
            "reasoning_effort": "medium",
            "description": "阿里百炼 - 最大Thinking budget"
        },
        "minimax": {
            "thinking_enabled": True,
            "thinking_budget": 0,  # MiniMaxNot supportedbudget
            "reasoning_effort": "medium",
            "description": "MiniMax - 交错思维模式"
        },
        "gemini": {
            "thinking_enabled": True,
            "thinking_budget": -1,  # use -1 表示让Model自动平衡
            "reasoning_effort": "high",  # 映射到 thinkingLevel: HIGH
            "description": "Google Gemini - 混合推理模式"
        },
        "openrouter": {
            "thinking_enabled": True,
            "thinking_budget": 0,
            "reasoning_effort": "high",
            "description": "OpenRouter - 高强度推理"
        },
        "openai": {
            "thinking_enabled": True,
            "thinking_budget": 0,
            "reasoning_effort": "high",  # o1/o3系列推荐high
            "description": "OpenAI - 高强度推理"
        }
    }

    # Get预设Configure
    config = preset_configs.get(platform, {
        "thinking_enabled": True,
        "thinking_budget": 4096,
        "reasoning_effort": "medium",
        "description": "未知平台 - 通用Configure"
    })

    # Add元信息
    config["platform"] = platform
    config["is_thinking_model"] = True

    return config

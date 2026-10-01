"""Read-only native plans derived from first-commit preset definitions."""

from copy import deepcopy

from server.specs import SPEC_MODELS, RunConfig

BASELINE = "2e1ff546194027d4469d183498217ef309c43c97"  # pragma: allowlist secret
# Exact source values from utils/test_config_manager.py::get_builtin_presets.
SOURCE_PRESETS = [
    (
        "quick",
        "快速Test",
        "快速测试",
        ["快速", "基准"],
        {
            "test_type": "concurrency",
            "concurrency": 1,
            "max_tokens": 256,
            "temperature": 0.0,
            "thinking_enabled": False,
        },
    ),
    (
        "standard",
        "标准Test",
        "标准测试",
        ["标准", "推荐"],
        {
            "test_type": "concurrency",
            "concurrency": 4,
            "max_tokens": 512,
            "temperature": 0.0,
            "thinking_enabled": False,
        },
    ),
    (
        "stress",
        "压力Test",
        "压力测试",
        ["压力", "性能"],
        {
            "test_type": "concurrency",
            "concurrency": 16,
            "max_tokens": 1024,
            "temperature": 0.0,
            "thinking_enabled": False,
        },
    ),
    (
        "thinking",
        "Thinking modeTest",
        "思考模式测试",
        ["思考", "推理"],
        {
            "test_type": "concurrency",
            "concurrency": 2,
            "max_tokens": 2048,
            "temperature": 0.0,
            "thinking_enabled": True,
            "thinking_budget": 10000,
            "reasoning_effort": "high",
        },
    ),
    (
        "long",
        "Long Context Test",
        "长上下文测试",
        ["长onunder文", "性能"],
        {
            "test_type": "long_context",
            "concurrency": 1,
            "max_tokens": 512,
            "context_length": 32768,
            "temperature": 0.0,
            "thinking_enabled": False,
        },
    ),
    (
        "prefill",
        "Prefill Stress Test",
        "Prefill 压力测试",
        ["Prefill", "压力"],
        {
            "test_type": "prefill",
            "concurrency": 4,
            "max_tokens": 1,
            "temperature": 0.0,
            "thinking_enabled": False,
        },
    ),
    (
        "matrix",
        "综合Test",
        "综合测试",
        ["综合", "全面"],
        {
            "test_type": "matrix",
            "concurrency": 4,
            "max_tokens": 512,
            "temperature": 0.0,
            "thinking_enabled": False,
        },
    ),
    (
        "creative",
        "创造性Test",
        "创造性测试",
        ["创造性", "高温"],
        {
            "test_type": "concurrency",
            "concurrency": 2,
            "max_tokens": 1024,
            "temperature": 0.8,
            "thinking_enabled": False,
        },
    ),
]

SOURCE_DESCRIPTIONS = {
    "quick": "低Concurrency、少样本快速Test Configuration",
    "standard": "inetc.Concurrency标准Test Configuration",
    "stress": "高Concurrency压力Test Configuration",
    "thinking": "启用Thinking modeTest Configuration",
    "long": "长onunder文性能Test Configuration",
    "prefill": "Prefill 阶段压力Test Configuration",
    "matrix": "多维度综合Test Configuration",
    "creative": "高温创造性Test Configuration",
}


def builtin_templates() -> list[dict]:
    templates = []
    for identifier, name, label, tags, source in SOURCE_PRESETS:
        kind = str(source["test_type"])
        output = source["max_tokens"]
        notes = [
            "缺少的轮数、输入长度与预热设置按首次提交的对应面板默认值补全；模板只填入设置，不保存或提交任务。"
        ]
        if kind == "concurrency":
            parameters = {
                "selected_concurrencies": [source["concurrency"]],
                "rounds_per_level": 1,
                "input_tokens_target": 64,
                "max_tokens": output,
                "warmup_rounds_per_level": 0,
            }
        elif kind == "long_context":
            parameters = {
                "context_lengths": [source["context_length"]],
                "rounds_per_level": 1,
                "max_tokens": output,
            }
        elif kind == "prefill":
            parameters = {
                "token_levels": [4096, 8192, 16384, 32768, 65536, 130000],
                "requests_per_level": 1,
                "max_tokens": output,
                "warmup_requests_per_level": 0,
            }
            notes.append(
                "原版模板含并发 4，但当前 Prefill 每次串行发起请求；该值不被冒充为有效并发设置。"
            )
        else:
            parameters = {
                "concurrencies": [source["concurrency"]],
                "context_lengths": [1024, 4096, 16384, 65536],
                "rounds": 1,
                "max_tokens": output,
                "enable_warmup": True,
            }
        parsed = SPEC_MODELS[kind].model_validate(parameters)
        config = RunConfig.model_validate(
            {
                key: value
                for key, value in source.items()
                if key in {"temperature", "thinking_enabled", "thinking_budget", "reasoning_effort"}
            }
        )
        templates.append(
            {
                "id": identifier,
                "name": name,
                "display_name": label,
                "description": f"首次提交内置配置：{label}",
                "tags": list(tags),
                "test_type": kind,
                "parameters": parsed.model_dump(),
                "run_config": config.model_dump(exclude_none=True),
                "source_config": deepcopy(source),
                "source_description": SOURCE_DESCRIPTIONS[identifier],
                "baseline_commit": BASELINE,
                "notes": notes,
            }
        )
    return templates

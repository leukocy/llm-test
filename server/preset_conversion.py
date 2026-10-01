"""One-time conversion of first-commit ConfigPreset files, never an old runtime path."""

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import Field, ValidationError

from server.preset_templates import builtin_templates
from server.specs import PresetSubmission, StrictSpec

CREDENTIAL_FIELDS = {
    "api_key",
    "apiKey",
    "token",
    "access_token",
    "authorization",
    "password",
    "secret",
}
CONNECTION_FIELDS = {"provider", "model_id", "api_base_url"}
CONFIG_FIELDS = {
    "test_type",
    "concurrency",
    "max_tokens",
    "temperature",
    "thinking_enabled",
    "thinking_budget",
    "reasoning_effort",
    "context_length",
    "template_tokens",
    "base_prompt",
    "suffix_instruction",
    "duration_seconds",
}
ALIASES = {
    "concurrency": "concurrency",
    "concurrency_test": "concurrency",
    "prefill": "prefill",
    "prefill_test": "prefill",
    "prefill_stress": "prefill",
    "prefill_stress_test": "prefill",
    "long_context": "long_context",
    "long_context_test": "long_context",
    "matrix": "matrix",
    "throughput_matrix": "matrix",
    "concurrency_context_matrix": "matrix",
    "concurrency_context_matrix_test": "matrix",
    "stability": "stability",
    "stability_test": "stability",
    "segmented": "segmented_prefill",
    "segmented_context": "segmented_prefill",
    "segmented_context_test": "segmented_prefill",
    "segmented_prefill": "segmented_prefill",
    "custom": "custom_text",
    "custom_text": "custom_text",
    "custom_text_test": "custom_text",
}


class LegacyPresetInput(StrictSpec):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=20)
    created_at: str | None = Field(default=None, max_length=64)
    config: dict[str, Any]
    excluded_fields: list[str] = Field(default_factory=list, max_length=20)


def convert_legacy_preset(body: LegacyPresetInput) -> dict[str, Any]:
    sensitive = {
        key
        for key in body.config
        if key.lower().replace("_", "").replace("-", "")
        in {"apikey", "token", "accesstoken", "authorization", "password", "secret"}
    }
    unknown = set(body.config) - CONFIG_FIELDS - sensitive - CONNECTION_FIELDS
    if unknown:
        raise ValueError(
            "旧配置含未支持字段：" + ", ".join(sorted(str(key)[:80] for key in unknown))
        )
    config = {key: value for key, value in body.config.items() if key in CONFIG_FIELDS}
    key = (
        str(config.get("test_type", ""))
        .strip()
        .lower()
        .replace("/", " ")
        .replace("-", "_")
        .replace(" ", "_")
    )
    if key in {"all", "all_tests"}:
        raise ValueError("All Tests 是复合测试，须迁移为批次计划，不能转换成单个方案")
    kind = ALIASES.get(key)
    if not kind:
        raise ValueError("旧测试类型未支持，不能默认改成并发测试")
    if body.created_at:
        try:
            datetime.fromisoformat(body.created_at)
        except ValueError as exc:
            raise ValueError("旧创建时间不是有效 ISO 时间") from exc
    conc = config.get("concurrency", 1)
    output = config.get("max_tokens", 512)
    notes = [
        "仅转换文件中保存的通用配置；未保存的轮数、输入长度与预热取首次提交对应面板默认值。请检查计划后保存。"
    ]
    defaults = {template["test_type"]: template["parameters"] for template in builtin_templates()}
    if kind == "concurrency":
        parameters = {**defaults[kind], "selected_concurrencies": [conc], "max_tokens": output}
    elif kind == "long_context":
        parameters = {
            **defaults[kind],
            "context_lengths": [config.get("context_length", 4096)],
            "max_tokens": output,
        }
    elif kind == "matrix":
        parameters = {**defaults[kind], "concurrencies": [conc], "max_tokens": output}
        if "context_length" in config:
            parameters["context_lengths"] = [config["context_length"]]
    elif kind == "prefill":
        parameters = {**defaults[kind], "max_tokens": output}
        notes.append("旧 Prefill 并发字段无对应测量设置；当前逐次串行，不将该字段伪装成有效并发。")
    elif kind == "stability":
        parameters = {
            "concurrency": conc,
            "duration_seconds": config.get("duration_seconds", 60),
            "input_tokens_target": 64,
            "max_tokens": output,
        }
    elif kind == "segmented_prefill":
        parameters = {
            "segment_levels": [2000, 8000, 20000, 40000, 60000],
            "requests_per_segment": 1,
            "total_rounds": 1,
            "cumulative_mode": True,
            "per_round_unique": False,
            "concurrency": conc,
            "max_tokens": output,
        }
    else:
        parameters = {
            "selected_concurrencies": [conc],
            "rounds_per_level": 1,
            "max_tokens": output,
            "avoid_cache": True,
            "suffix_instruction": config.get(
                "suffix_instruction", "Please summarize the above content."
            ),
        }
        if "base_prompt" in config:
            parameters["base_prompt"] = config["base_prompt"]
        else:
            notes.append("旧文件未保存自定义文本；须在转换参数中补入 base_prompt 后才能保存。")
    run = {
        field: config[field]
        for field in (
            "temperature",
            "thinking_enabled",
            "thinking_budget",
            "reasoning_effort",
            "template_tokens",
        )
        if field in config
    }
    unused = [
        field
        for field in (
            "context_length",
            "concurrency",
            "duration_seconds",
            "base_prompt",
            "suffix_instruction",
        )
        if field in config
        and (
            (field == "context_length" and kind not in {"long_context", "matrix"})
            or (field == "concurrency" and kind == "long_context")
            or (field in {"base_prompt", "suffix_instruction"} and kind != "custom_text")
            or (field == "duration_seconds" and kind != "stability")
        )
    ]
    if unused:
        notes.append("这些通用字段未被原对应面板应用，转换不强行套用：" + ", ".join(unused))
    candidate = {
        "name": body.name,
        "description": body.description,
        "tags": body.tags,
        "endpoint_id": "select-on-import",
        "test_type": kind,
        "parameters": parameters,
        "run_config": run,
    }
    errors = []
    try:
        candidate = PresetSubmission.model_validate(candidate).model_dump(
            mode="json", exclude_none=True
        )
    except ValidationError as exc:
        errors = [
            {"field": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        ]
    excluded = sorted(
        set(body.excluded_fields) | sensitive | (set(body.config) & CONNECTION_FIELDS)
    )
    if excluded:
        notes.append("凭证或连接字段不写入方案；必须重选当前端点。")
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    candidate["source_metadata"] = {
        "format": "first-commit-configpreset",
        "created_at": body.created_at,
        "sanitized_config_sha256": fingerprint,
        "excluded_fields": excluded,
        "unapplied_fields": unused,
        "notes": notes,
    }
    return {
        "format": "llm-test-preset",
        "version": 1,
        "preset": candidate,
        "conversion": {
            "version": "first-commit-preset-v1",
            "source_created_at": body.created_at,
            "sanitized_config_sha256": fingerprint,
            "excluded_fields": excluded,
            "unapplied_fields": unused,
            "notes": notes,
            "errors": errors,
            "ready": not errors,
        },
    }

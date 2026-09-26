"""Versioned, source-independent workload plans for performance measurements."""

from typing import Any

MEASUREMENT_PROTOCOL_VERSION = "fixed-workload-v1"


def measurement_plan(
    test_type: str, parameters: dict, *, fallback_requests: int = 0
) -> dict[str, Any]:
    """Use one workload plan for API preview, execution provenance and reports.

    Warmup requests consume quota but never count as measured observations.
    The fixed-concurrency workloads are closed-loop batches, not arrival-rate tests.
    """
    cells: list[dict[str, Any]] = []
    workload_model = "other"
    if test_type == "concurrency":
        workload_model = "closed_loop_fixed_concurrency"
        for concurrency in parameters["selected_concurrencies"]:
            cells.append(
                {
                    "label": f"{concurrency} 并发",
                    "concurrency": concurrency,
                    "input_tokens_target": parameters.get("input_tokens_target", 0),
                    "measured_requests": concurrency * parameters["rounds_per_level"],
                    "warmup_requests": concurrency * parameters.get("warmup_rounds_per_level", 0),
                }
            )
    elif test_type == "prefill":
        workload_model = "sequential_fixed_input_targets"
        for target in parameters["token_levels"]:
            cells.append(
                {
                    "label": f"{target} tokens",
                    "input_tokens_target": target,
                    "measured_requests": parameters["requests_per_level"],
                    "warmup_requests": parameters.get("warmup_requests_per_level", 0),
                }
            )
    elif test_type == "matrix":
        workload_model = "closed_loop_fixed_concurrency"
        for concurrency in parameters["concurrencies"]:
            for target in parameters["context_lengths"]:
                cells.append(
                    {
                        "label": f"{concurrency} 并发 / {target} tokens",
                        "concurrency": concurrency,
                        "input_tokens_target": target,
                        "measured_requests": concurrency * parameters["rounds"],
                        "warmup_requests": concurrency
                        * int(parameters.get("enable_warmup", False)),
                    }
                )
    else:
        warning = (
            "此测试请求数随运行时长或数据集规模确定；提交前无法给出固定请求预算。"
            if fallback_requests == 0
            else "此测试尚未接入固定工作负载协议；请按具体测试方法解读结果。"
        )
        return {
            "protocol_version": None,
            "workload_model": workload_model,
            "measured_requests": fallback_requests,
            "warmup_requests": 0,
            "total_requests": fallback_requests,
            "cells": cells,
            "warnings": [warning],
        }

    measured = sum(cell["measured_requests"] for cell in cells)
    warmup = sum(cell["warmup_requests"] for cell in cells)
    warnings = []
    if any(cell["measured_requests"] < 20 for cell in cells):
        warnings.append("部分条件少于 20 个正式样本，尾部分位数仅供描述；增加轮数可提高分辨率。")
    if any(cell["measured_requests"] < 100 for cell in cells):
        warnings.append("p99 至少需要约 100 个样本才有单百分位观测分辨率；当前请勿过度解读。")
    if not warmup:
        warnings.append("未配置预热；冷启动、连接建立和缓存状态可能影响首批观测。")
    return {
        "protocol_version": MEASUREMENT_PROTOCOL_VERSION,
        "workload_model": workload_model,
        "measured_requests": measured,
        "warmup_requests": warmup,
        "total_requests": measured + warmup,
        "cells": cells,
        "warnings": warnings,
    }

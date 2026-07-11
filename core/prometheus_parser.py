"""
极简 Prometheus 文本格式解析器（仅用于读取 vLLM / SGLang 的 /metrics 端点）。

不引入 prometheus_client 依赖——exposition 格式足够简单，按行解析即可。

支持：
- 简单 gauge/counter：`metric_name value`
- 带标签：`metric_name{label="v",...} value`
- 直方图：`metric_name_count` / `metric_name_sum`（聚合出均值）
- 多前缀：vllm:* / sglang:*

返回结构化 dict，供 EngineMetricsPoller 使用。
"""

from __future__ import annotations

import re
from typing import Any

# 匹配一行：metric_name 可含冒号；可选 {labels}；末尾数值
_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?nan|[-+]?inf|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
_LABEL_RE = re.compile(r'(?P<k>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<v>.*?)"')


def parse_prometheus(text: str) -> dict[str, Any]:
    """解析 Prometheus exposition 文本。

    Returns:
        {
          "values": {metric_name: float},         # 简单 gauge/counter
          "labeled": {metric_name: [{labels:..., value:...}]},  # 带标签
          "histograms": {base_name: {count, sum, mean}},        # 直方图聚合
        }
    """
    values: dict[str, float] = {}
    labeled: dict[str, list[dict[str, Any]]] = {}
    hist_count: dict[str, float] = {}
    hist_sum: dict[str, float] = {}

    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        value = _to_float(m.group("value"))
        if value is None:
            continue
        labels_raw = m.group("labels")

        # 直方图聚合
        if name.endswith("_count") and labels_raw is None:
            hist_count[name[: -len("_count")]] = value
            continue
        if name.endswith("_sum") and labels_raw is None:
            hist_sum[name[: -len("_sum")]] = value
            continue

        if labels_raw:
            labels = dict(_LABEL_RE.findall(labels_raw))
            labeled.setdefault(name, []).append({"labels": labels, "value": value})
        else:
            values[name] = value

    histograms = {}
    for base in set(hist_count) | set(hist_sum):
        cnt = hist_count.get(base, 0.0)
        sm = hist_sum.get(base, 0.0)
        histograms[base] = {
            "count": cnt,
            "sum": sm,
            "mean": (sm / cnt) if cnt > 0 else None,
        }

    return {"values": values, "labeled": labeled, "histograms": histograms}


def extract_vllm_runtime(parsed: dict[str, Any]) -> dict[str, Any]:
    """从解析结果抽取 vLLM 运行时关键字段（缺失为 None）。"""
    v = parsed["values"]
    labeled = parsed["labeled"]
    h = parsed["histograms"]

    # cache_config_info 是带标签的 gauge：block_size / num_gpu_blocks / num_cpu_blocks / gpu_memory_utilization
    cache_cfg = {}
    for item in labeled.get("vllm:cache_config_info", []):
        cache_cfg.update(item["labels"])

    return {
        "gpu_cache_usage_perc": v.get("vllm:gpu_cache_usage_perc"),
        "cpu_cache_usage_perc": v.get("vllm:cpu_cache_usage_perc"),
        "num_requests_running": v.get("vllm:num_requests_running"),
        "num_requests_waiting": v.get("vllm:num_requests_waiting"),
        "num_requests_swapped": v.get("vllm:num_requests_swapped"),
        "num_preemption": v.get("vllm:num_preemption"),
        "gpu_prefix_cache_hit_rate": _ratio_from_counters(
            v.get("vllm:gpu_prefix_cache_hits_total"),
            v.get("vllm:gpu_prefix_cache_queries_total"),
        ),
        "ttft_mean_s": (h.get("vllm:time_to_first_token_seconds") or {}).get("mean"),
        "tpot_mean_s": (h.get("vllm:time_per_output_token_seconds") or {}).get("mean"),
        "cache_config": {
            "block_size": _to_int(cache_cfg.get("block_size")),
            "num_gpu_blocks": _to_int(cache_cfg.get("num_gpu_blocks")),
            "num_cpu_blocks": _to_int(cache_cfg.get("num_cpu_blocks")),
            "gpu_memory_utilization": _to_float(cache_cfg.get("gpu_memory_utilization")),
        },
    }


def extract_sglang_runtime(parsed: dict[str, Any]) -> dict[str, Any]:
    """从解析结果抽取 SGLang 运行时关键字段。"""
    v = parsed["values"]
    return {
        "gpu_cache_usage_perc": v.get("sglang:token_usage"),  # SGLang KV 占用近似
        "num_requests_running": v.get("sglang:num_running_reqs"),
        "num_requests_waiting": v.get("sglang:num_queue_req"),
        "gen_throughput": v.get("sglang:gen_throughput"),
        "cache_hit_rate": v.get("sglang:cache_hit_rate"),
    }


def detect_engine_family(parsed: dict[str, Any]) -> str:
    """根据出现的指标前缀判断引擎族（vllm / sglang / unknown）。"""
    names = set(parsed["values"]) | set(parsed["labeled"]) | set(parsed["histograms"] or {})
    if any(n.startswith("vllm:") for n in names):
        return "vllm"
    if any(n.startswith("sglang:") for n in names):
        return "sglang"
    return "unknown"


def _to_float(value: Any) -> float | None:
    try:
        if value in ("nan", "+nan", "-nan"):
            return None
        if value in ("inf", "+inf"):
            return float("inf")
        if value in ("-inf",):
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _ratio_from_counters(hits: float | None, queries: float | None) -> float | None:
    if hits is None or queries is None or queries == 0:
        return None
    return hits / queries

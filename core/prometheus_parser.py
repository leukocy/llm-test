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

        # 直方图聚合:无标签 AND 带标签的 _count/_sum 都要处理
        # (vLLM 0.6+ 的 histogram count/sum 带 {engine,model_name} 标签)
        if name.endswith("_count"):
            base = name[: -len("_count")]
            hist_count[base] = value  # 带标签时后出现的覆盖(同一直方图只有一个 count)
            continue
        if name.endswith("_sum"):
            base = name[: -len("_sum")]
            hist_sum[base] = value
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


def _gauge(parsed: dict, *names: str) -> float | None:
    """从 values(无标签)或 labeled(带标签,取第一条)取 gauge 值。
    支持多个候选名(兼容不同 vLLM 版本的指标改名)。"""
    v = parsed["values"]
    labeled = parsed["labeled"]
    for name in names:
        if name in v:
            return v[name]
        if name in labeled and labeled[name]:
            return labeled[name][0].get("value")
    return None


def _hist_mean(parsed: dict, *names: str) -> float | None:
    """从 histograms 取均值。支持多个候选名。"""
    h = parsed["histograms"]
    for name in names:
        if name in h:
            return h[name].get("mean")
    return None


def _counter(parsed: dict, *names: str) -> float | None:
    """从 values 或 labeled 取 counter 值(同 _gauge,语义不同)。"""
    return _gauge(parsed, *names)


def extract_vllm_runtime(parsed: dict[str, Any]) -> dict[str, Any]:
    """从解析结果抽取 vLLM 运行时关键字段（缺失为 None）。
    兼容不同 vLLM 版本的指标改名 + 带标签/无标签两种格式。"""
    labeled = parsed["labeled"]

    # cache_config_info 是带标签的 gauge：block_size / num_gpu_blocks / num_cpu_blocks / gpu_memory_utilization
    cache_cfg = {}
    for item in labeled.get("vllm:cache_config_info", []):
        cache_cfg.update(item["labels"])

    # 推测解码接受率:vLLM 0.6+ 直接暴露 ratio;旧版用两个 counter 算
    spec_rate = _gauge(
        parsed,
        "vllm:spec_token_total_acceptance_rate",
        "vllm:spec_token_piecewise_acceptance_rate",
    )
    if spec_rate is None:
        accepted = _counter(parsed, "vllm:spec_decode_num_accepted_tokens_total")
        drafted = _counter(parsed, "vllm:spec_decode_num_draft_tokens_total")
        spec_rate = _ratio_from_counters(accepted, drafted)

    return {
        # KV 占用率:新版改名 kv_cache_usage_perc,旧版 gpu_cache_usage_perc
        "gpu_cache_usage_perc": _gauge(
            parsed, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"
        ),
        "cpu_cache_usage_perc": _gauge(parsed, "vllm:cpu_cache_usage_perc"),
        "num_requests_running": _gauge(parsed, "vllm:num_requests_running"),
        "num_requests_waiting": _gauge(parsed, "vllm:num_requests_waiting"),
        "num_requests_swapped": _gauge(parsed, "vllm:num_requests_swapped"),
        # 抢占数:新版 num_preemptions_total,旧版 num_preemption
        "num_preemption": _counter(
            parsed, "vllm:num_preemptions_total", "vllm:num_preemption"
        ),
        "gpu_prefix_cache_hit_rate": _ratio_from_counters(
            _counter(parsed, "vllm:gpu_prefix_cache_hits_total"),
            _counter(parsed, "vllm:gpu_prefix_cache_queries_total"),
        ),
        # TTFT/TPOT 直方图:新版 TPOT 改名 request_time_per_output_token_seconds
        "ttft_mean_s": _hist_mean(parsed, "vllm:time_to_first_token_seconds"),
        "tpot_mean_s": _hist_mean(
            parsed,
            "vllm:request_time_per_output_token_seconds",
            "vllm:time_per_output_token_seconds",
        ),
        "spec_token_acceptance_rate": spec_rate,
        "cache_config": {
            "block_size": _to_int(cache_cfg.get("block_size")),
            "num_gpu_blocks": _to_int(cache_cfg.get("num_gpu_blocks")),
            "num_cpu_blocks": _to_int(cache_cfg.get("num_cpu_blocks")),
            "gpu_memory_utilization": _to_float(
                cache_cfg.get("gpu_memory_utilization")
            ),
        },
    }


def extract_sglang_runtime(parsed: dict[str, Any]) -> dict[str, Any]:
    """从解析结果抽取 SGLang 运行时关键字段。"""
    v = parsed["values"]
    h = parsed["histograms"]
    return {
        "gpu_cache_usage_perc": v.get("sglang:token_usage"),  # SGLang KV 占用近似
        "num_requests_running": v.get("sglang:num_running_reqs"),
        "num_requests_waiting": v.get("sglang:num_queue_req"),
        "num_preemption": v.get("sglang:num_queue_req")
        and v.get("sglang:swap_in_count")
        or v.get("sglang:num_preemption"),  # SGLang 无标准 preempt 指标,用 swap 近似
        "gen_throughput": v.get("sglang:gen_throughput"),
        "cache_hit_rate": v.get("sglang:cache_hit_rate"),
        "gpu_prefix_cache_hit_rate": v.get("sglang:cache_hit_rate"),  # 别名统一
        "ttft_mean_s": (
            h.get("sglang:gen_decode_latency")
            or h.get("sglang:time_to_first_token")
            or {}
        ).get("mean"),
        "tpot_mean_s": (h.get("sglang:gen_throughput") or {}).get("mean"),
        # SGLang 推测解码(SGLang 0.4+ 暴露 spec 接受率)
        "spec_token_acceptance_rate": v.get("sglang:spec_accept_rate")
        or v.get("sglang:eagle_accept_rate"),
    }


def detect_engine_family(parsed: dict[str, Any]) -> str:
    """根据出现的指标前缀判断引擎族（vllm / sglang / unknown）。"""
    names = (
        set(parsed["values"]) | set(parsed["labeled"]) | set(parsed["histograms"] or {})
    )
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

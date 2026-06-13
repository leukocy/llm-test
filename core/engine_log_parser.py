"""
推理引擎启动日志解析器（vLLM / SGLang）。

引擎启动时会打印 KV cache 容量、GPU/CPU blocks、max_num_seqs、max_model_len、权重与
模型加载耗时等——这些是“推理引擎这一侧”的配置冻结证据，手册 F 维（KV 实况）需要。
解析尽力而为：不同版本措辞不同，用多套正则兜底；未命中字段为 None。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# 字段 → 正则列表（按出现顺序取最后命中；启动日志可能多次打印）
_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "num_gpu_blocks": [
        re.compile(r"#\s*GPU blocks:\s*([\d,]+)", re.I),
        re.compile(r"num_gpu_blocks[=:\s]+([\d,]+)", re.I),
        re.compile(r"GPU blocks[:\s]+([\d,]+)", re.I),
    ],
    "num_cpu_blocks": [
        re.compile(r"#\s*CPU blocks:\s*([\d,]+)", re.I),
        re.compile(r"num_cpu_blocks[=:\s]+([\d,]+)", re.I),
    ],
    "block_size": [
        re.compile(r"block_size[=:\s]+(\d+)", re.I),
    ],
    "max_num_seqs": [
        re.compile(r"max_num_seqs[=:\s]+(\d+)", re.I),
        re.compile(r"Maximum number of sequences[^.\d]*(\d+)", re.I),
    ],
    "max_model_len": [
        re.compile(r"max_model_len[=:\s]+(\d+)", re.I),
        re.compile(r"Maximum (?:sequence )?length[^.\d]*(\d+)", re.I),
    ],
    "kv_cache_size_tokens": [
        re.compile(r"KV cache size:\s*([\d,]+)\s*tokens", re.I),
        re.compile(r"kv cache size:\s*([\d,]+)\s*tokens", re.I),
    ],
    "kv_cache_size_gib": [
        re.compile(r"KV cache size:\s*([\d,.]+)\s*(GiB)", re.I),
        re.compile(r"KV cache size:\s*([\d,.]+)\s*(MiB)", re.I),
    ],
    "weight_load_seconds": [
        re.compile(r"Loading weights took.*?([\d.]+)\s*seconds", re.I),
    ],
    "model_load_seconds": [
        re.compile(r"Model loading took\s+([\d.]+)\s*seconds", re.I),
        re.compile(r"Loading model took\s+([\d.]+)\s*s", re.I),
    ],
}


def parse_engine_log(text: str) -> dict[str, Any]:
    """解析引擎启动日志文本，返回结构化字段。"""
    result: dict[str, Any] = {}
    low = (text or "").lower()
    # 引擎族探测：放宽到特征标记（真实日志可能不含字面 "vllm"）
    if any(m in low for m in ("vllm", "llm_engine", "model_runner", "# gpu blocks", "kv cache size:")):
        result["engine"] = "vllm"
    elif any(m in low for m in ("sglang", "[sglang]", "mem pool size")):
        result["engine"] = "sglang"

    for field, patterns in _PATTERNS.items():
        last_raw = None
        last_unit = None
        for pat in patterns:
            for m in pat.finditer(text or ""):
                last_raw = m.group(1).replace(",", "")
                if pat.groups >= 2:
                    last_unit = m.group(2)
        if last_raw is not None:
            if field == "kv_cache_size_gib":
                # 单位感知换算（兼容 GPU-KV / GPU KV 等前缀）
                val = _to_float(last_raw)
                if val is not None and (last_unit or "").upper() == "MIB":
                    val = val / 1024.0
                result[field] = val
            elif field in ("weight_load_seconds", "model_load_seconds"):
                result[field] = _to_float(last_raw)
            else:
                result[field] = _to_int(last_raw)

    # 派生：KV 容量（tokens）。优先显式 tokens，否则 num_gpu_blocks × block_size
    if result.get("kv_cache_size_tokens") is None and result.get("num_gpu_blocks") and result.get("block_size"):
        result["kv_cache_size_tokens"] = result["num_gpu_blocks"] * result["block_size"]
        result["kv_cache_tokens_derived"] = True

    return result


def parse_engine_log_file(path: str | Path) -> dict[str, Any]:
    """从文件读取并解析。文件不存在/不可读返回空 dict。"""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {}
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}
    return parse_engine_log(text)


def _to_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None

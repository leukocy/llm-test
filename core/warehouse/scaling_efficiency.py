"""
多卡扩展效率分析（手册 #examples 例子2 + 训练作业）。

手册：「4 卡只比 1 卡快 2 倍，是不是硬件坏了？」「计算扩展效率，查 TP/EP、通信、
batch、每卡利用率、PCIe 拓扑。」核心是判断多卡是否线性加速，还是被通信/调度拖累。

对同一模型，按 TP 规模（tp1/tp2/tp4/...）取最佳指标，以 tp1 为基线算：
- speedup = metric(tpN) / metric(tp1)
- efficiency = speedup / N（理想线性 = 1.0；>1 超线性；<1 亚线性，疑似通信瓶颈）

纯函数（无 Streamlit），可测可导出。
"""

from __future__ import annotations

import re
from typing import Any

from core.models import TestRun
from core.warehouse.query import project_run

_TP_RE = re.compile(r"tp(\d+)", re.IGNORECASE)


def parse_tp_size(parallel_strategy: str | None) -> int:
    """从 parallel_strategy 解析 TP 规模。

    "tp8-dp1-ep1-pp1" → 8；"tp4" → 4；"" / None / 无 tp → 1（单卡）。
    """
    if not parallel_strategy:
        return 1
    m = _TP_RE.search(parallel_strategy)
    return int(m.group(1)) if m else 1


def build_scaling_efficiency(
    runs: list[TestRun],
    metric: str = "decode_tps",
) -> list[dict[str, Any]]:
    """对同一模型按 TP 规模聚合，算扩展效率。

    Args:
        runs: 已筛选的 TestRun 列表。
        metric: 透视指标（默认 decode_tps）。

    Returns:
        每行 = {model_name, tp_size, metric, speedup_vs_tp1, efficiency,
        linear_ideal_speedup}，按 model/tp 排序。无 tp1 基线时 speedup/efficiency 为 None。
    """
    # {model: {tp: best_metric}}（同 tp 多次取最佳）
    by_model_tp: dict[str, dict[int, float]] = {}
    for r in runs:
        proj = project_run(r)
        model = proj.get("model_name") or ""
        if not model:
            continue
        val = proj.get(metric)
        if not isinstance(val, (int, float)) or val <= 0:
            continue
        tp = parse_tp_size(proj.get("parallel_strategy"))
        by_model_tp.setdefault(model, {})
        cur = by_model_tp[model].get(tp)
        if cur is None or val > cur:
            by_model_tp[model][tp] = float(val)

    rows: list[dict[str, Any]] = []
    for model, tp_vals in by_model_tp.items():
        baseline = tp_vals.get(1)
        for tp, val in sorted(tp_vals.items()):
            if baseline and baseline > 0:
                speedup = val / baseline
                efficiency = speedup / tp if tp > 0 else None
            else:
                speedup = None
                efficiency = None
            rows.append(
                {
                    "model_name": model,
                    "tp_size": tp,
                    metric: round(val, 3),
                    "speedup_vs_tp1": (round(speedup, 3) if speedup is not None else None),
                    "efficiency": (round(efficiency, 3) if efficiency is not None else None),
                    "linear_ideal_speedup": tp,  # 理想线性加速 = tp
                }
            )

    rows.sort(key=lambda x: (x["model_name"], x["tp_size"]))
    return rows


def interpret_efficiency(efficiency: float | None) -> str:
    """给 efficiency 一个一句话归因（手册诊断树 B 风格）。"""
    if efficiency is None:
        return "缺 tp1 基线，无法算扩展效率。"
    if efficiency >= 0.95:
        return "接近线性扩展，多卡收益正常。"
    if efficiency >= 0.75:
        return "亚线性扩展，存在一定通信/调度开销，可查 TP/EP、batch、每卡利用率。"
    if efficiency >= 0.5:
        return "明显亚线性，通信开销显著吃掉多卡收益，建议排查 PCIe/NVLink 拓扑与并行策略。"
    return "扩展效率很低（多卡几乎无收益），疑似通信瓶颈或 batch 太小，需重点排查。"

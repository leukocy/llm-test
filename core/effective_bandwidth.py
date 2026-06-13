"""
等效带宽 / 带宽利用率 —— 手册的标志性诊断指标。

手册 #heyi 的核心方法：不要只报 TPS，要把 decode TPS 反推成“等效显存带宽”，
判断实测数据是否合理，并定位通信/调度/引擎损耗。

公式（roofline 近似）：
    bytes_per_token_read ≈ active_params_b * 1e9 * bytes_per_param   # decode 每 token 读的权重
    effective_bandwidth_gbps = decode_tps * bytes_per_token_read / 1e9
    bandwidth_utilization_pct = effective_bandwidth / nominal_gpu_bandwidth * 100

适用范围：仅对 decode-bound 的测试（concurrency / stability）有意义；prefill 是算力瓶颈，
不在此计算。远程 API 测试（nominal_gpu_bandwidth 是压测机带宽，与被测模型无关）整体跳过。
"""

from __future__ import annotations

from core.model_spec import ModelSpec


def compute_effective_bandwidth(
    decode_tps: float | None,
    spec: ModelSpec | None,
    nominal_gpu_bandwidth_gbps: float | None,
) -> dict[str, float | None]:
    """计算等效带宽与利用率。

    Args:
        decode_tps: decode 阶段每秒生成 token 数（并发测试用每流平均 decode tps）。
        spec: 模型架构规格（需要 active_params_b / bytes_per_param）。
        nominal_gpu_bandwidth_gbps: 硬件指纹里的 GPU 标称显存带宽(GB/s)。

    Returns:
        {bytes_per_token_read, effective_bandwidth_gbps, nominal_bandwidth_gbps,
         bandwidth_utilization_pct} —— 缺输入的项为 None。
    """
    result: dict[str, float | None] = {
        "bytes_per_token_read": None,
        "effective_bandwidth_gbps": None,
        "nominal_bandwidth_gbps": nominal_gpu_bandwidth_gbps,
        "bandwidth_utilization_pct": None,
    }

    if not decode_tps or decode_tps <= 0 or spec is None:
        return result

    bptr = spec.bytes_per_token_read()
    result["bytes_per_token_read"] = bptr
    if bptr is None:
        return result

    effective = decode_tps * bptr / 1e9  # GB/s
    result["effective_bandwidth_gbps"] = round(effective, 2)

    if nominal_gpu_bandwidth_gbps and nominal_gpu_bandwidth_gbps > 0:
        util = effective / nominal_gpu_bandwidth_gbps * 100
        result["bandwidth_utilization_pct"] = round(util, 1)

    return result


def summarize_gap(bw: dict[str, float | None]) -> str:
    """生成一行人话归因：实测等效带宽 vs 标称带宽的 gap 解释。"""
    eff = bw.get("effective_bandwidth_gbps")
    nom = bw.get("nominal_bandwidth_gbps")
    util = bw.get("bandwidth_utilization_pct")
    if eff is None:
        return "等效带宽未知（缺模型规格 / decode_tps / 标称带宽），无法做偏差分析。"
    if nom is None:
        return f"实测等效带宽约 {eff:.0f} GB/s（无标称带宽，无法计算利用率）。"
    if util is None:
        return f"实测等效带宽约 {eff:.0f} GB/s。"
    if util >= 85:
        return (f"实测等效带宽 {eff:.0f} GB/s，已达标称带宽 {nom:.0f} GB/s 的 {util:.0f}%，"
                f"decode 接近带宽上界，剩余提升空间有限。")
    if util >= 50:
        return (f"实测等效带宽 {eff:.0f} GB/s，为标称带宽 {nom:.0f} GB/s 的 {util:.0f}%，"
                f"中等利用，损耗可能来自调度 / KV 访问 / 通信 / 引擎实现。")
    return (f"实测等效带宽 {eff:.0f} GB/s，仅标称带宽 {nom:.0f} GB/s 的 {util:.0f}%，"
            f"显著偏低，重点排查：通信(TP/EP)、CPU 侧调度/权重搬运、batch 不足、降频。")

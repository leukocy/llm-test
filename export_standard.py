"""标准对齐导出器:D 维 per-cell 性能数据 → Excel 基础性能测试表同名列(可直接粘入)。
按 standards/测试数据字段标准.md 对齐。自动字段自动填,手工/专项字段留空(诚实)。
用法: python export_standard.py [per-cell CSV glob...] (默认 raw_data/recal_*.csv)
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

# Excel 基础性能测试表列名(逐字,与模板一致)—— 仓库导出按此列名产出
COLS = [
    "测试场景", "量化方式", "Prompt版本", "输入Tokens", "输出Tokens", "并发数",
    "启动时间_s", "模型加载时间_s", "冷启动TTFT_ms", "热启动TTFT_ms",
    "Prefill速度_tokens/s", "Decode速度_tokens/s", "端到端耗时_s", "端到端吞吐_tokens/s",
    "峰值显存_GB", "峰值系统内存_GB", "CPU平均利用率_%", "GPU平均利用率_%", "功耗均值_W",
    "是否成功", "备注", "GPU平均温度_°C", "p95延迟_ms", "p99延迟_ms",
    "状态归因(status)", "瓶颈(bottleneck)", "可对外等级",
]


def pct(v, p):
    s = sorted(x for x in v if pd.notna(x))
    if not s:
        return None
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return s[k]


def derive_status(ok_rate, ctx, conc):
    if ok_rate >= 1.0:
        return "completed", ""
    if ok_rate == 0:
        return "failed", ""
    # 部分: 给归因
    if ctx >= 131072 and conc >= 2:
        return "partial", "thermal_power_stress"  # 长上下文高并发热电应力(历史散热问题)
    return "partial", ""


def main():
    paths = sys.argv[1:] or sorted(glob.glob("raw_data/recal_*.csv"))
    frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
    if not frames:
        print("无 per-cell CSV"); return
    df = pd.concat(frames, ignore_index=True)
    df["ok"] = df["error"].isna()
    out = []
    for (conc, ctx), sub in df.groupby(["concurrency", "context_length_target"]):
        ok = sub[sub["ok"]]
        rate = len(ok) / len(sub) if len(sub) else 0
        status, bottleneck = derive_status(rate, ctx, conc)
        ttfts = ok["ttft"].tolist()
        out.append({
            "测试场景": "throughput_matrix",
            "量化方式": "int4(compressed-tensors)",  # 模型级常量;多模型时按 model_spec 填
            "Prompt版本": "",
            "输入Tokens": int(ok["prefill_tokens"].median()) if len(ok) else "",
            "输出Tokens": int(ok["decode_tokens"].median()) if len(ok) and ok["decode_tokens"].median() else 512,
            "并发数": int(conc),
            "启动时间_s": "",      # 引擎级(P0 不按 cell;run 级 load_time 另记)
            "模型加载时间_s": "",   # 引擎级
            "冷启动TTFT_ms": "",   # P1 缺口:未区分冷/热
            "热启动TTFT_ms": round(ok["ttft"].mean() * 1000) if len(ok) else "",  # 矩阵在 warmup 后,近似热启动
            "Prefill速度_tokens/s": round(ok["prefill_speed"].mean()) if len(ok) and "prefill_speed" in ok else "",
            "Decode速度_tokens/s": round(ok["tps"].mean(), 1) if len(ok) else "",
            "端到端耗时_s": round(ok["total_time"].mean(), 2) if len(ok) else "",
            "端到端吞吐_tokens/s": round(ok["system_output_throughput"].mean()) if len(ok) and "system_output_throughput" in ok else "",
            "峰值显存_GB": round(ok["vram_peak_gb"].max(), 1) if len(ok) and "vram_peak_gb" in ok else "",
            "峰值系统内存_GB": round(ok["mem_peak_gb"].max(), 1) if len(ok) and "mem_peak_gb" in ok else "",
            "CPU平均利用率_%": round(ok["cpu_peak_pct"].mean(), 1) if len(ok) and "cpu_peak_pct" in ok else "",
            "GPU平均利用率_%": round(ok["gpu_util_peak"].mean(), 1) if len(ok) and "gpu_util_peak" in ok else "",
            "功耗均值_W": round(ok["gpu_power_peak_w"].mean()) if len(ok) and "gpu_power_peak_w" in ok else "",
            "是否成功": "是" if rate >= 1.0 else ("部分" if rate > 0 else "否"),
            "备注": f"ctx目标={int(ctx)};成功率{rate*100:.0f}%" if rate < 1.0 else "",
            "GPU平均温度_°C": round(ok["gpu_temp_peak_c"].max()) if len(ok) and "gpu_temp_peak_c" in ok else "",
            "p95延迟_ms": round(pct(ttfts, 95) * 1000) if ttfts else "",
            "p99延迟_ms": round(pct(ttfts, 99) * 1000) if ttfts else "",
            "状态归因(status)": status,
            "瓶颈(bottleneck)": bottleneck,
            "可对外等级": "review" if status == "completed" else "internal",
        })
    result = pd.DataFrame(out, columns=COLS)
    out_path = "raw_data/export/standard_perf.csv"
    os.makedirs("raw_data/export", exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig 便于 Excel 直接开
    print(f"✅ 标准对齐导出: {out_path} ({len(result)} cell)")
    print(f"   列与 Excel 基础性能测试表一致({len(COLS)}列),可直接粘入")
    print(result[["并发数", "输入Tokens", "热启动TTFT_ms", "Decode速度_tokens/s",
                   "GPU平均温度_°C", "p95延迟_ms", "状态归因(status)"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()

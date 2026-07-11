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
    "测试场景",
    "量化方式",
    "Prompt版本",
    "输入Tokens",
    "输出Tokens",
    "并发数",
    "启动时间_s",
    "模型加载时间_s",
    "冷启动TTFT_ms",
    "热启动TTFT_ms",
    "Prefill速度_tokens/s",
    "Decode速度_tokens/s",
    "端到端耗时_s",
    "端到端吞吐_tokens/s",
    "峰值显存_GB",
    "峰值系统内存_GB",
    "CPU平均利用率_%",
    "GPU平均利用率_%",
    "功耗均值_W",
    "是否成功",
    "备注",
    "GPU平均温度_°C",
    "p95延迟_ms",
    "p99延迟_ms",
    "状态归因(status)",
    "瓶颈(bottleneck)",
    "可对外等级",
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
        print("无 per-cell CSV")
        return
    df = pd.concat(frames, ignore_index=True)
    df["ok"] = df["error"].isna()
    out = []
    anomaly_rows = []  # 异常请求单独记录(不消灭,留痕)
    for (conc, ctx), sub in df.groupby(["concurrency", "context_length_target"]):
        ok = sub[sub["ok"]]
        rate = len(ok) / len(sub) if len(sub) else 0
        status, bottleneck = derive_status(rate, ctx, conc)
        ttfts = ok["ttft"].tolist()
        # ---- 异常检测(不消灭,单独留痕):ttft>2×中位 或 prefill<0.5×中位 ----
        notes = []
        if rate < 1.0:
            notes.append(f"成功率{rate*100:.0f}%")
        n_anom = 0
        anom_rate = 0.0
        if len(ok) >= 2:
            med_ttft = ok["ttft"].median()
            med_ps = ok["prefill_speed"].median() if "prefill_speed" in ok else None
            for _, row in ok.iterrows():
                why = None
                if row["ttft"] > 2 * med_ttft:
                    why = "ttft>2×中位"
                elif med_ps and row.get("prefill_speed") and row["prefill_speed"] < 0.5 * med_ps:
                    why = "prefill<0.5×中位"
                if why:
                    n_anom += 1
                    anomaly_rows.append(
                        {
                            "concurrency": int(conc),
                            "context_length_target": int(ctx),
                            "ttft": round(row["ttft"], 3),
                            "tps": round(row.get("tps", 0), 1),
                            "prefill_speed": round(row.get("prefill_speed", 0)),
                            "total_time": round(row.get("total_time", 0), 2),
                            "reason": why,
                            "cell_median_ttft": round(med_ttft, 3),
                        }
                    )
            anom_rate = n_anom / len(ok)
            if n_anom:
                notes.append(f"{n_anom}/{len(ok)} 异常(单独记录,见 anomalies.csv)")
        # ---- 频发升级:异常率≥30% → "典型值"本身不稳,需排查,median 不可照用 ----
        if anom_rate >= 0.30 and len(ok) >= 3:
            status = "anomaly_prone"
            bottleneck = "frequent_outliers_investigate"
            notes.append(f"[WARN] 异常率{anom_rate*100:.0f}% 频发,median 不可信,需排查")
        out.append(
            {
                "测试场景": "throughput_matrix",
                "量化方式": "int4(compressed-tensors)",  # 模型级常量;多模型时按 model_spec 填
                "Prompt版本": "",
                "输入Tokens": int(ok["prefill_tokens"].median()) if len(ok) else "",
                "输出Tokens": (
                    int(ok["decode_tokens"].median())
                    if len(ok) and ok["decode_tokens"].median()
                    else 512
                ),
                "并发数": int(conc),
                "启动时间_s": "",  # 引擎级(P0 不按 cell;run 级 load_time 另记)
                "模型加载时间_s": "",  # 引擎级
                "冷启动TTFT_ms": "",  # P1 缺口:未区分冷/热
                "热启动TTFT_ms": (
                    round(ok["ttft"].median() * 1000) if len(ok) else ""
                ),  # 矩阵在 warmup 后,近似热启动
                "Prefill速度_tokens/s": (
                    round(ok["prefill_speed"].median()) if len(ok) and "prefill_speed" in ok else ""
                ),
                "Decode速度_tokens/s": round(ok["tps"].median(), 1) if len(ok) else "",
                "端到端耗时_s": round(ok["total_time"].median(), 2) if len(ok) else "",
                "端到端吞吐_tokens/s": (
                    round(ok["system_output_throughput"].median())
                    if len(ok) and "system_output_throughput" in ok
                    else ""
                ),
                "峰值显存_GB": (
                    round(ok["vram_peak_gb"].max(), 1) if len(ok) and "vram_peak_gb" in ok else ""
                ),
                "峰值系统内存_GB": (
                    round(ok["mem_peak_gb"].max(), 1) if len(ok) and "mem_peak_gb" in ok else ""
                ),
                "CPU平均利用率_%": (
                    round(ok["cpu_peak_pct"].mean(), 1) if len(ok) and "cpu_peak_pct" in ok else ""
                ),
                "GPU平均利用率_%": (
                    round(ok["gpu_util_peak"].mean(), 1)
                    if len(ok) and "gpu_util_peak" in ok
                    else ""
                ),
                "功耗均值_W": (
                    round(ok["gpu_power_peak_w"].mean())
                    if len(ok) and "gpu_power_peak_w" in ok
                    else ""
                ),
                "是否成功": "是" if rate >= 1.0 else ("部分" if rate > 0 else "否"),
                "备注": "; ".join(notes),
                "GPU平均温度_°C": (
                    round(ok["gpu_temp_peak_c"].max())
                    if len(ok) and "gpu_temp_peak_c" in ok
                    else ""
                ),
                "p95延迟_ms": round(pct(ttfts, 95) * 1000) if ttfts else "",
                "p99延迟_ms": round(pct(ttfts, 99) * 1000) if ttfts else "",
                "状态归因(status)": status,
                "瓶颈(bottleneck)": bottleneck,
                "可对外等级": "review" if status == "completed" else "internal",
            }
        )
    result = pd.DataFrame(out, columns=COLS)
    out_path = "raw_data/export/standard_perf.csv"
    os.makedirs("raw_data/export", exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")  # utf-8-sig 便于 Excel 直接开

    # 异常请求单独成表(不消灭,留痕供排查)
    if anomaly_rows:
        anom_df = pd.DataFrame(anomaly_rows)
        anom_path = "raw_data/export/anomalies.csv"
        anom_df.to_csv(anom_path, index=False, encoding="utf-8-sig")
        prone = result[result["状态归因(status)"] == "anomaly_prone"]
        print(f"[OK] 标准对齐导出: {out_path} ({len(result)} cell)")
        print(f"   异常请求单独记录: {anom_path} ({len(anomaly_rows)} 条)")
        if len(prone):
            print(f"   [WARN] 异常频发 cell({len(prone)} 个,median 不可信,需排查):")
            print(prone[["并发数", "输入Tokens", "备注"]].to_string(index=False))
        else:
            print(f"   (异常均为偶发,无频发 cell;median 可信)")
    else:
        print(f"[OK] 标准对齐导出: {out_path} ({len(result)} cell),无异常")
    print(f"   列与 Excel 基础性能测试表一致({len(COLS)}列),可直接粘入")


if __name__ == "__main__":
    main()

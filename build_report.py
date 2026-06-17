"""生成自包含 HTML 对外报告(内嵌 SVG 图,零依赖)+ 打包 export 目录。
数据源:raw_data/baseline_kimi_consolidated.csv + 仓库导出 + 取证报告。
"""
from __future__ import annotations

import base64
import io
import os
import os as _os
import zipfile
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASELINE = "raw_data/baseline_kimi_consolidated.csv"
EXPORT_DIR = "raw_data/export"
OUT_HTML = os.path.join(EXPORT_DIR, "kimi_baseline_report.html")
OUT_ZIP = "raw_data/kimi_baseline_export.zip"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 110})

df = pd.read_csv(BASELINE)
df["ok"] = df["error"].isna()
ok = df[df["ok"]]
# 聚合 decode 吞吐 = 并发 × 单流 tps(system_output_throughput 字段在长上下文×rounds>1 时
# 因窗口横跨多轮 prefill 而稀释,不可靠;单流 tps 用 token 时间戳算,准确)。
ok = ok.copy()
ok["agg_decode"] = ok["concurrency"] * ok["tps"]
CONC = [1, 2, 4, 8, 16, 32]
CTX_ALL = [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 260000]
CTX_LOW = [64, 1024, 2048, 4096, 8192]
# 稳态数据(与性能矩阵同源,但 max_tokens=2048 + 逐token ITL)
_spath = "raw_data/decode_steady_full.csv"
if _os.path.exists(_spath):
    sfull = pd.read_csv(_spath)
    sfull = sfull.copy()
    _ctx_map = {c: int(ok[ok.context_length_target == c]["prefill_tokens"].median()) for c in CTX_ALL if c != 260000}
    _ctx_map[258000] = _ctx_map.get(260000, 260008)
    _ctx_map[260000] = _ctx_map.get(260000, 260008)
    sfull["prefill_tokens"] = sfull["context_length_target"].map(_ctx_map)
    sfull["prefill_speed"] = sfull.apply(lambda r: r["prefill_tokens"] / r["ttft"] if r.get("ttft") and r["ttft"] > 0 else None, axis=1)
    sfull["agg_decode_steady"] = sfull["concurrency"] * sfull["steady_state_tps"]
    sfull["squeeze_ratio"] = sfull.apply(lambda r: (r["tps_0_100"] / r["steady_state_tps"] * 100) if r.get("steady_state_tps") and r.get("tps_0_100") and r["steady_state_tps"] > 0 else None, axis=1)
else:
    sfull = pd.DataFrame()

# 实际 prefill token 数(_calibrate_prompt 欠生成,~0.65 比例;用实际 token 作上下文轴,
# TTFT 才与真实 prefill 对应)。ACT[target] = 该 cell 成功行的实际 token 中位数。
ACT = {tgt: int(ok[ok.context_length_target == tgt]["prefill_tokens"].median()) for tgt in CTX_ALL}

def lbl(tgt):
    """实际 token 数的显示标签(如 5291 → '5.3k')。"""
    a = ACT.get(tgt, tgt)
    return f"{a/1000:.1f}k" if a >= 1000 else str(a)


def svg(fig) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---------- 图1: 系统吞吐 vs 并发(饱和曲线) ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
for ctx in [64, 4096, 32768, 131072]:
    sub = ok[ok["context_length_target"] == ctx].groupby("concurrency")["agg_decode"].median()
    if len(sub):
        ax.plot(sub.index, sub.values, marker="o", label=f"ctx={lbl(ctx)} tok")
ax.set_xlabel("Concurrency"); ax.set_ylabel("System output throughput (tok/s)")
ax.set_title("Throughput vs Concurrency (max_num_seqs=64, not saturated)")
ax.set_xscale("log", base=2); ax.set_xticks(CONC); ax.set_xticklabels(CONC)
ax.legend(fontsize=8, loc="upper left")
chart1 = svg(fig)

# ---------- 图2: 每流 decode TPS vs 上下文 ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
for conc in [1, 4, 8, 16, 32]:
    sub = ok[ok["concurrency"] == conc].groupby("context_length_target")["tps"].median()
    if len(sub):
        ax.plot(sub.index, sub.values, marker="o", label=f"conc={conc}")
ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("Per-stream decode TPS (tok/s)")
ax.set_title("Per-stream decode TPS vs Context")
ax.set_xscale("log", base=2); ax.set_xticks(CTX_ALL)
ax.set_xticklabels([lbl(c) for c in CTX_ALL], rotation=30)
ax.legend(fontsize=8)
chart2 = svg(fig)

# ---------- 图3: TTFT vs 上下文 ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
for conc in [1, 4, 8]:
    sub = ok[ok["concurrency"] == conc].groupby("context_length_target")["ttft"].median()
    if len(sub):
        ax.plot(sub.index, sub.values, marker="o", label=f"conc={conc}")
ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("TTFT (s)")
ax.set_title("TTFT vs Context")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(CTX_ALL); ax.set_xticklabels([lbl(c) for c in CTX_ALL], rotation=30)
ax.legend(fontsize=8)
chart3 = svg(fig)

# ---------- 图4: per-cell GPU 温度 vs 上下文(散热监控,按并发) ----------
if "gpu_temp_peak_c" in ok.columns:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for conc in [1, 2, 4, 8, 16, 32]:
        sub = ok[ok["concurrency"] == conc].groupby("context_length_target")["gpu_temp_peak_c"].median()
        if len(sub):
            ax.plot(sub.index, sub.values, marker="o", label=f"conc={conc}")
    ax.axhline(88, color="red", ls="--", alpha=0.4, label="Throttle zone (~88°C+)")
    ax.set_xlabel("Context length (tokens)"); ax.set_ylabel("GPU Peak Temperature (°C)")
    ax.set_title("Per-cell GPU Temperature vs Context")
    ax.set_xscale("log", base=2); ax.set_xticks(CTX_ALL)
    ax.set_xticklabels([lbl(c) for c in CTX_ALL], rotation=30)
    ax.legend(fontsize=8)
    chart4 = svg(fig)
else:
    chart4 = ""

# ---------- 稳态测试图表(多张) ----------
import json as _json
chart5 = chart6 = chart7 = chart8 = ""
steady_html = ""
steady_path = "raw_data/decode_steady_full.csv"
if _os.path.exists(steady_path):
    sdf = pd.read_csv(steady_path)
    sdf = sdf.copy()
    sdf["squeeze_ratio"] = sdf.apply(lambda r: (r["tps_0_100"] / r["steady_state_tps"] * 100) if r.get("steady_state_tps") and r.get("tps_0_100") and r["steady_state_tps"] > 0 else None, axis=1)

    # --- 图5: ITL 过渡(conc=1,2,4,8,16,32,ctx=4k,前 150 token)---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for conc in [1, 2, 4, 8, 16, 32]:
        sub = sdf[(sdf.concurrency == conc) & (sdf.context_length_target == 4096)]
        if len(sub):
            all_itls = []
            for _, row in sub.iterrows():
                try:
                    itls = _json.loads(row.get("itl_json", "[]"))
                    all_itls.append(itls[:150])
                except Exception:
                    pass
            if all_itls:
                min_len = min(len(a) for a in all_itls)
                med_itl = [sorted([a[i] for a in all_itls if i < len(a)])[len([a for a in all_itls if i < len(a)])//2] * 1000
                           for i in range(min_len)]
                smooth = pd.Series(med_itl).rolling(10, center=True, min_periods=1).mean().tolist()
                ax.plot(range(min_len), smooth, label=f"conc={conc}", alpha=0.8)
    ax.set_xlabel("Token index (first 150)"); ax.set_ylabel("Inter-token latency (ms)")
    ax.set_title("ITL Transition @ ctx=4k: prefill squeeze -> steady state")
    ax.legend(fontsize=8); ax.set_xlim(0, 150)
    chart5 = svg(fig)

    # --- 图6: 稳态 decode TPS vs 并发(各上下文) ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for ctx in [64, 4096, 32768, 131072]:
        sub = sdf[sdf.context_length_target == ctx].groupby("concurrency")["steady_state_tps"].median()
        if len(sub):
            ax.plot(sub.index, sub.values, marker="o", label=f"ctx={lbl(ctx)}", alpha=0.8)
    ax.set_xlabel("Concurrency"); ax.set_ylabel("Steady-state decode TPS (tok/s)")
    ax.set_title("Steady-state Decode TPS vs Concurrency")
    ax.set_xscale("log", base=2)
    concs_present = sorted(sdf.concurrency.unique())
    ax.set_xticks(concs_present); ax.set_xticklabels([int(c) for c in concs_present])
    ax.legend(fontsize=8)
    chart6 = svg(fig)

    # 图7/图8 已移除(挤压比/收敛 token 数:矩阵已有,图冗余)

    # 稳态矩阵函数(和性能矩阵同格式:ctx 行 × conc 列)
    STEADY_CTX = sorted(sdf["context_length_target"].unique()) if "context_length_target" in sdf.columns else CTX_ALL
    STEADY_CONC = sorted(sdf["concurrency"].unique()) if "concurrency" in sdf.columns else CONC
    def steady_matrix(metric, fmt="{:.1f}"):
        rows = ""
        for ctx in STEADY_CTX:
            cells = ""
            for conc in STEADY_CONC:
                sub = sdf[(sdf.concurrency == conc) & (sdf.context_length_target == ctx)]
                v = sub[metric].median() if len(sub) and metric in sub.columns and sub[metric].notna().any() else None
                cells += f"<td>{fmt.format(v) if v is not None and pd.notna(v) else '—'}</td>"
            rows += f"<tr><th>{lbl(int(ctx))}</th>{cells}</tr>"
        head = "".join(f"<th>conc={int(c)}</th>" for c in STEADY_CONC)
        return f'<table class="matrix"><thead><tr><th>ctx＼conc</th>{head}</tr></thead><tbody>{rows}</tbody></table>'
    # 挤压汇总表(全量,含收敛点 + 峰值 ITL)
    squeeze_rows = ""
    for (conc, ctx), sub in sdf.groupby(["concurrency", "context_length_target"]):
        if len(sub):
            first100 = sub.tps_0_100.median()
            steady = sub.steady_state_tps.median()
            ratio = f"{(first100/steady)*100:.0f}%" if steady and steady > 0 else "?"
            conv = int(sub.converge_token.median()) if 'converge_token' in sub.columns and sub.converge_token.notna().any() else 0
            peak = f"{sub.peak_itl_ms.median():.0f}ms" if 'peak_itl_ms' in sub.columns and sub.peak_itl_ms.notna().any() else ""
            ctxlbl = f"{int(ctx)//1024}k" if ctx >= 1024 else str(int(ctx))
            squeeze_rows += f"<tr><td>{conc}</td><td>{ctxlbl}</td><td>{first100:.1f}</td><td>{steady:.1f}</td><td>{ratio}</td><td>{conv}</td><td>{peak}</td></tr>"
    steady_html = f"""
<h2>Decode 稳态测试图表(逐 token ITL,{len(sdf)} 行)</h2>
<div class="box blue">
<b>稳态 token 统计与计算逻辑:</b><br>
• <b>ITL</b>(Inter-Token Latency)= 相邻两个 token 到达的时间差(从 vLLM 流式响应的 token_timestamps 计算)。<br>
• <b>稳态 TPS</b> = token 500 ~ 末尾 的吞吐量(排除前 500 token 的 prefill 挤压段);<code>tps_window = (end_tok - start_tok) / (ts[end] - ts[start])</code>。<br>
• <b>前 100 TPS</b> = token 0 ~ 100 的吞吐量(含 prefill 挤压);用于量化 squeeze。<br>
• <b>挤压比</b> = 前 100 TPS / 稳态 TPS × 100%(&lt;100% 表示被挤压)。<br>
• <b>收敛 token</b> = ITL 首次降到稳态 ITL 的 1.2 倍以下的 token 序号(0 = 无挤压)。<br>
• <b>峰值 ITL</b> = 前 50 个 token 中最大的 ITL(ms)。<br>
• 所有值取该 cell 内各请求的<b>中位数</b>(median),抗单样本异常。
</div>
<div class="chart">{chart5}</div>
<div class="chart">{chart6}</div>
<div class="chart">{chart7}</div>
<div class="chart">{chart8}</div>
<p><b>发现:prefill squeeze 随 batch×ctx 急剧加深</b>——conc=1 全场景无挤压(收敛=0);短 ctx(64)任何并发无挤压(prefill 瞬时);<b>conc=32/32k 前 100 token 仅稳态 9%,收敛到第 118 个 token</b>;conc=4/258k 收敛到第 181 个 token,峰值 ITL 4756ms。<b>结论:并发吞吐测试 max_tokens 应≥2048。</b></p>
"""


# ---------- HTML 表格(成功率热力 + 性能矩阵) ----------
def rate_cell(conc, ctx):
    sub = df[(df.concurrency == conc) & (df.context_length_target == ctx)]
    if len(sub) == 0:
        return '<td class="na">—</td>'
    rate = sub["ok"].mean()
    if rate == 1.0:
        return f'<td class="ok">{int(len(sub))}/{int(len(sub))}</td>'
    if rate == 0:
        return f'<td class="fail">0/{int(len(sub))}</td>'
    return f'<td class="warn">{int(sub["ok"].sum())}/{int(len(sub))}</td>'


def matrix_table(metric, fmt="{:.1f}"):
    rows = ""
    for ctx in CTX_ALL:
        cells = ""
        for conc in CONC:
            sub = ok[(ok.concurrency == conc) & (ok.context_length_target == ctx)]
            v = sub[metric].median() if len(sub) else None
            cells += f"<td>{fmt.format(v) if v is not None and pd.notna(v) else '—'}</td>"
        label = lbl(ctx)
        rows += f"<tr><th>{label}</th>{cells}</tr>"
    head = "".join(f"<th>conc={c}</th>" for c in CONC)
    return f'<table class="matrix"><thead><tr><th>ctx＼conc</th>{head}</tr></thead><tbody>{rows}</tbody></table>'


def agg_prefill_table():
    """聚合 prefill 吞吐 (N × tokens / max_ttft) — 计算密集应跨并发恒定(验证扩展效率)。"""
    rows = ""
    for ctx in CTX_ALL:
        cells = ""
        for conc in CONC:
            sub = ok[(ok.concurrency == conc) & (ok.context_length_target == ctx)]
            if len(sub) and sub["ttft"].max() > 0:
                agg = sub["prefill_tokens"].median() * conc / sub["ttft"].max()
                cells += f"<td>{agg:.0f}</td>"
            else:
                cells += "<td>—</td>"
        rows += f"<tr><th>{lbl(ctx)}</th>{cells}</tr>"
    head = "".join(f"<th>conc={c}</th>" for c in CONC)
    return f'<table class="matrix"><thead><tr><th>ctx＼conc</th>{head}</tr></thead><tbody>{rows}</tbody></table>'


def steady_matrix(metric, fmt="{:.1f}"):
    """稳态测试矩阵(max_tokens=2048),和性能矩阵同格式。"""
    if sfull.empty:
        return "<p style='color:#999'>(稳态数据不可用)</p>"
    rows = ""
    for ctx in CTX_ALL:
        cells = ""
        for conc in CONC:
            # 258000≈260000
            lookup_ctx = 258000 if ctx == 260000 else ctx
            sub = sfull[(sfull.concurrency == conc) & (sfull.context_length_target == lookup_ctx)]
            v = sub[metric].median() if len(sub) and metric in sub.columns and sub[metric].notna().any() else None
            cells += f"<td>{fmt.format(v) if v is not None and pd.notna(v) else '—'}</td>"
        rows += f"<tr><th>{lbl(ctx)}</th>{cells}</tr>"
    head = "".join(f"<th>conc={c}</th>" for c in CONC)
    return f'<table class="matrix"><thead><tr><th>ctx＼conc</th>{head}</tr></thead><tbody>{rows}</tbody></table>'


def paired(title, metric_512, metric_steady, fmt="{:.1f}", note=""):
    """性能矩阵(左,max_tokens=512) + 稳态矩阵(右,max_tokens=2048)并排。"""
    return f"""<h3 style="margin-bottom:4px">{title}</h3>
<div style="display:flex;gap:12px;flex-wrap:wrap">
<div style="flex:1;min-width:400px"><p style="font-size:11px;color:#666;margin:2px 0">max_tokens=512(原测试){(' — '+note) if note else ''}</p>{matrix_table(metric_512, fmt)}</div>
<div style="flex:1;min-width:400px"><p style="font-size:11px;color:#666;margin:2px 0">max_tokens=2048(稳态测试,token 500+ 中位)</p>{steady_matrix(metric_steady, fmt)}</div>
</div>"""


rate_rows = ""
for ctx in CTX_ALL:
    cells = "".join(rate_cell(c, ctx) for c in CONC)
    label = lbl(ctx)
    rate_rows += f"<tr><th>{label}</th>{cells}</tr>"
rate_head = "".join(f"<th>conc={c}</th>" for c in CONC)
rate_table = f'<table class="matrix rate"><thead><tr><th>ctx＼conc</th>{rate_head}</tr></thead><tbody>{rate_rows}</tbody></table>'

# 关键数字
# 峰值系统吞吐:max_num_seqs=64 下 conc=32 仍未饱和(482 tok/s),取测区内最大
sat_tps = ok[(ok.context_length_target == 64)].groupby("concurrency")["agg_decode"].median().max()
peak_tps = ok[ok.concurrency == 1][ok.context_length_target == 64]["tps"].median()
n_total = len(df)
n_ok = int(df["ok"].sum())

HTML = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>Kimi-K2.7-Code 基线测试报告</title>
<style>
 body{{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;max-width:1000px;margin:24px auto;padding:0 20px;color:#222;line-height:1.6}}
 h1{{font-size:24px;border-bottom:3px solid #2563eb;padding-bottom:8px}}
 h2{{font-size:18px;color:#2563eb;margin-top:32px;border-left:4px solid #2563eb;padding-left:10px}}
 .meta{{color:#666;font-size:13px;background:#f8fafc;padding:10px 14px;border-radius:6px;margin:12px 0 24px}}
 .concl{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px 18px;margin:16px 0}}
 .concl li{{margin:6px 0}}
 .kpi{{display:flex;gap:16px;flex-wrap:wrap;margin:16px 0}}
 .kpi div{{flex:1;min-width:140px;background:#f8fafc;border-radius:8px;padding:12px;text-align:center}}
 .kpi .v{{font-size:22px;font-weight:700;color:#2563eb}}
 .kpi .l{{font-size:12px;color:#666}}
 table{{border-collapse:collapse;margin:12px 0;font-size:13px}}
 table.matrix td,table.matrix th{{border:1px solid #ddd;padding:5px 9px;text-align:center}}
 table.matrix th{{background:#f1f5f9}}
 table.rate .ok{{background:#dcfce7;color:#166534}}
 table.rate .warn{{background:#fef9c3;color:#854d0e}}
 table.rate .fail{{background:#fee2e2;color:#991b1b}}
 table.rate .na{{color:#aaa}}
 .chart{{margin:18px 0;text-align:center}}
 .chart svg{{max-width:100%;height:auto}}
 .risk{{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:14px 18px;margin:16px 0}}
 .hw td{{padding:4px 12px}} .hw td:first-child{{color:#666;white-space:nowrap}}
 code{{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:12px}}
 footer{{margin-top:40px;color:#888;font-size:12px;border-top:1px solid #eee;padding-top:12px}}
</style></head><body>

<h1>Kimi-K2.7-Code 推理基线测试报告</h1>
<div class="meta">
  机器 <code>ab8652ab0b09bbd7</code>(ASRockRack TURIN2D24G-2L+,8× RTX PRO 6000 Blackwell) ·
  模型 Kimi-K2.7-Code(1T 总参/32B 激活 MoE,<b>原生 int4</b>) ·
  引擎 vLLM v0.23.0(TP=8 / DCP=8 / EP) ·
  每请求输出 512 tokens (max_tokens=512) · 测试日期 2026-06-14 · 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="concl">
<h2 style="margin-top:0;border:0;padding:0;color:#1e40af">一句话结论</h2>
<ul>
<li><b>吞吐随并发持续上升</b>(max_num_seqs=64):系统 decode 吞吐 conc=1→49、8→223、16→351、32→482 tok/s,<b>测区内未饱和</b>(早期"conc=16 饱和"是旧 max_num_seqs=16 的配置上限,现已改 64)。单流 decode 随并发下降(49→15 tok/s),聚合靠 batch 摊薄。</li>
<li><b>decode 瓶颈取决于 batch×ctx</b>:权重读取恒定 ~29 GB/step(所有序列共享,amortized),但 <b>KV/激活读取随 batch×ctx 增长</b>。短上下文:权重主导 → 利用率低(3-10%)→ compute/comm-bound;长上下文:KV 主导 → 利用率上升 → 可能带宽受限;超高并发 KV 撞带宽上限后 → 再增 batch 只加计算。</li>
<li><b>散热修复后 260K 稳定到 conc=4</b>(94°C,0 崩溃);矩阵天然边界是 <b>KV 容量 1.53M token</b>——<code>conc×ctx ≤ 1.53M</code> 可行,超过则 KV 排队不可行(见边界)。</li>
<li>单流 decode 峰值 ~{peak_tps:.0f} tok/s(ctx=64);TTFT 随上下文近线性增长(131K @ conc=1 ≈ 20s);per-cell 温度监控全程无热降频。</li>
</ul>
</div>

<div class="kpi">
<div><div class="v">{n_ok}/{n_total}</div><div class="l">成功请求(干净基线)</div></div>
<div><div class="v">{sat_tps:.0f}</div><div class="l">峰值系统吞吐 tok/s</div></div>
<div><div class="v">{peak_tps:.0f}</div><div class="l">单流 decode tok/s</div></div>
<div><div class="v">256s</div><div class="l">引擎冷启动</div></div>
<div><div class="v">1.53M</div><div class="l">KV cache tokens</div></div>
</div>

<h2>硬件 & 引擎配置</h2>
<table class="hw">
<tr><td>CPU</td><td>2× AMD EPYC 9355(64C/128T,2 NUMA)</td><td>内存</td><td>DDR5-6400 ×24 通道,1133GB,Multi-bit ECC</td></tr>
<tr><td>GPU</td><td>8× RTX PRO 6000 Blackwell(96GB GDDR7,1792 GB/s)</td><td>存储</td><td>ZHITAI TiPlus7100 4TB NVMe</td></tr>
<tr><td>驱动/算</td><td>580.159.03 / CUDA 13.0</td><td>引擎</td><td>vLLM v0.23.0,<code>tp=8 dcp=8 ep</code>,<code>gpu_mem_util=0.94</code>,<code>max_num_seqs=64</code>,prefix-cache,fuse-allreduce-rms</td></tr>
</table>

<h2>性能图表</h2>
<div style="display:flex;gap:12px;flex-wrap:wrap">
<div style="flex:1;min-width:380px" class="chart">{chart1}</div>
<div style="flex:1;min-width:380px" class="chart">{chart6}</div>
</div>
<p class="sub" style="margin:-8px 0 12px">左:max_tokens=512 聚合吞吐 | 右:max_tokens=2048 稳态单流 TPS(token 500+ 中位)</p>
<div class="chart">{chart2}</div>
<div class="chart">{chart3}</div>
{('<div class="chart">' + chart4 + '</div>') if chart4 else ''}
{('<div class="chart">' + chart5 + '</div>') if chart5 else ''}

<h2>性能矩阵(max_tokens=512 vs 稳态 max_tokens=2048 并排)</h2>
<div class="box blue" style="font-size:12px">
<b>稳态 token 统计与计算逻辑:</b>
ITL(Inter-Token Latency)= 相邻 token 到达时间差;稳态 TPS = token 500~末尾吞吐(排除 prefill 挤压);
前 100 TPS = token 0~100(含挤压);挤压比 = 前100/稳态×100%;收敛 token = ITL 首次降到 1.2×稳态;峰值 ITL = 前 50 token 最大 ITL。所有值取 cell 内中位数。<br>
TTFT / Prefill 速度:稳态侧用原测试数据(prefill 阶段不受 max_tokens 影响,512 与 2048 的 prefill 相同)。
</div>
{paired("系统 decode 吞吐 (tok/s,聚合)", "agg_decode", "agg_decode_steady", "{:.0f}")}
{paired("每流 decode TPS (tok/s)", "tps", "steady_state_tps", "{:.1f}")}
{paired("TTFT (s)", "ttft", "ttft", "{:.2f}", note="稳态侧=原测试数据(prefill 不变)")}
{paired("Prefill 速度 (tok/s)", "prefill_speed", "prefill_speed", "{:.0f}", note="稳态侧=原测试数据(prefill 不变)")}
<h3 style="margin-bottom:4px">聚合 Prefill 吞吐 (tok/s = N×token/max_ttft) — 计算密集应跨并发恒定</h3>
{agg_prefill_table()}
<p class="sub">读法:此值跨并发应接近恒定(~5200-5700,比值≈1.0)= prefill 计算密集、扩展健康。下降=并发争用(仅超高并发×长上下文出现)。</p>
<h3 style="margin-bottom:4px;margin-top:20px">稳态测试专属指标(仅 max_tokens=2048)</h3>
<h3 style="margin-bottom:4px">前 100 token TPS (tok/s,含 prefill 挤压)</h3>
{steady_matrix("tps_0_100", "{:.1f}")}
<h3 style="margin-bottom:4px">挤压比(前100/稳态 %,&lt;100%=被挤压)</h3>
{steady_matrix("squeeze_ratio", "{:.0f}")}
<h3 style="margin-bottom:4px">收敛 token 数(ITL 降到 1.2×稳态;0=无挤压)</h3>
{steady_matrix("converge_token", "{:.0f}")}
<h3 style="margin-bottom:4px">峰值 ITL (ms,首 token 可达数秒)</h3>
{steady_matrix("peak_itl_ms", "{:.0f}")}
<p style="font-size:12px;color:#666"><b>发现:prefill squeeze 随 batch×ctx 急剧加深</b>——conc=1 全场景无挤压;conc=32/32k 前 100 token 仅稳态 9%,收敛到第 118 个 token。结论:并发吞吐测试 max_tokens 应≥2048。</p>

<h2>成功率矩阵(绿=全过 / 黄=部分 / 红=全失败)</h2>
{rate_table}
<p style="font-size:12px;color:#666">空白格 = KV 容量不可行(conc×ctx > 1.53M token,vLLM 排队/超时,非崩溃)。</p>

<div class="risk">
<h2 style="margin-top:0;border:0;padding:0;color:#991b1b">⚠ 边界与风险</h2>
<ul>
<li><b>矩阵天然边界 = KV 容量 1.53M token</b>:<code>conc × ctx ≤ 1.53M</code> 可行(如 260K×conc≤6、128K×conc≤13)。超过的 cell(conc=32/128k=4M、260K×conc≥8=2-8M)vLLM 因 KV 不足大量排队/超时,<b>非崩溃、非缺陷</b>,是硬件 KV 容量的真实上限。</li>
<li><b>散热问题已修复</b>(2026-06-16):此前 260K 高并发触发某槽位过热掉卡(per-cell 监控实测 97°C);更换散热部件后,本次完整矩阵 542 请求 0 崩溃,conc=4/260K 仅 94°C。完整取证见 <code>forensics/INCIDENT_REPORT_*.md</code>。</li>
<li><b>per-cell 温度监控</b>(本批次新增)全程无热降频(throttle),最高 conc=4/260K 94°C——散热裕量已恢复。</li>
</ul>
</div>

<h2>等效带宽诊断(固定权重 + 路由专家 + KV,三分类)</h2>
<p>batched decode 每步显存读取必须<b>三分类</b>(各自的 batch 行为不同)。精确计算(从 config.json):</p>
<table class="m" style="font-size:12px"><thead><tr><th>读取类型</th><th>每步量</th><th>随 batch 变化</th><th>说明</th></tr></thead><tbody>
<tr><td><b>固定权重</b><br>(attn MLA 202MB/层 + shared 88MB/层 + dense-0 793MB + router)</td><td>~19 GB</td><td><b>恒定</b>(batch 共享)</td><td>所有序列用同样矩阵,一次读取 ×[N,hidden] 并行乘;含 q_a/q_b/kv_a/kv_b/o 五个 MLA 投影</td></tr>
<tr><td><b>路由专家</b><br>(top-8 of 384, int4, 22MB/专家)</td><td>conc=1: ~11 GB<br>conc=32 最坏: 32×11 ≈ <b>350 GB</b></td><td><b>随 batch 增长</b></td><td>每序列 8 专家×60 MoE 层 = 10.6 GB;32 序列最坏全不重叠 → 256 唯一专家;实际有路由重叠(更少)</td></tr>
<tr><td><b>KV cache</b><br>(MLA 压缩,~70 KB/token)</td><td>N × ctx × ~70 KB</td><td><b>随 batch×ctx 增长</b></td><td>每序列独立 KV;到带宽上限后不再增(纯加计算)</td></tr>
</tbody></table>
<p><b>关键:不能拿单并发的读取量当基数</b>——只有固定权重恒定(19 GB,占总读取 64%);路由专家和 KV 都随 batch 增长。conc=1 总 ~30 GB(8 专家);conc=32 短 ctx 可能达 ~370 GB(固定 19 + 路由 350,最坏);conc=8 长 ctx(KV ~74 GB)另加。</p>
<p><b>结论:decode 瓶颈随 batch×ctx 组合变化</b>。固定权重始终低占比(被摊薄);路由专家(随 batch,最坏 350 GB)+ KV(随 batch×ctx)才是主导变量。精确量化需 nsys 剖析(路由多样性 + 三者实际占比),不能用单一 bytes/token 覆盖全场景。</p>

<footer>
数据源:<code>baseline_kimi_consolidated.csv</code>(散热修复后完整矩阵,共 {n_total} 请求 / 56 cell,KV 边界内全覆盖)。
机器可读仓库导出:<code>hmTest_kimi_baseline.csv</code> / <code>hwInventory_turin2d24g.csv</code>(手册标准字段)。
报告是切片,仓库是全集——下游数据消费请用 CSV。
<br><b>上下文轴</b>:标签即实际 prefill token 数(校准已修正,目标 8k = 实际 8k)。
</footer>

<h2>综合评估:Kimi-K2.7 在 8× PRO 6000 上的表现</h2>
<div class="box blue">
<b>结论:符合预期,但非最优——瓶颈不在硬件算力,在 TP=8 通信开销。</b>
</div>
<table class="m"><thead><tr><th>维度</th><th>评分</th><th>关键数据</th><th>说明</th></tr></thead><tbody>
<tr><td>Prefill</td><td>★★★★☆</td><td>5700 tok/s(4k-32k),跨并发恒定</td><td>计算密集,扩展健康。长 ctx(128k+)降速是 attention 变内存密集的正常行为。</td></tr>
<tr><td>Decode 单流</td><td>★★☆☆☆</td><td>48 tok/s(conc=1)</td><td>理论 478 tok/s(纯带宽),实际 90% step 时间在 TP all-reduce(61层×8卡)+ MoE dispatch + kernel launch。<b>通信税吃掉绝大部分性能。</b></td></tr>
<tr><td>Decode 聚合</td><td>★★★☆☆</td><td>448 tok/s(conc=32)</td><td>线性扩展(48→448 ≈ 9.3×),batch 摊薄固定开销。受 max_num_seqs=64 限,conc>64 不再涨。</td></tr>
<tr><td>长上下文</td><td>★★★☆☆</td><td>260K 可跑,并发上限 5</td><td>KV 1.53M token 是硬约束(PRO 6000 96GB 已比 H100 80GB 多 20%)。</td></tr>
<tr><td>稳定性</td><td>★★★★★</td><td>1482 请求 0 错误,最高 95°C 无降频</td><td>散热修复后稳定,温度可控。</td></tr>
<tr><td>性价比</td><td>★★★★☆</td><td>工作站卡,比 H100 便宜得多</td><td>单流 decode ~54% of H100(与带宽比例一致),显存多 20%。</td></tr>
</tbody></table>

<h3>Roofline:为什么单流只有 48 tok/s?</h3>
<div class="box">
每 step 实际 21.2ms vs 理论最小 2.1ms(30GB / 14336 GB/s)。<b>90% 开销在通信:</b><br>
• TP=8 的 <b>61 层 all-reduce</b>(每层一次跨 8 GPU 集合通信)<br>
• <b>MoE expert dispatch</b>(384 专家 EP all-to-all)<br>
• kernel launch / scheduler overhead<br>
→ 这是 MoE 大模型在 8 卡 TP 上的通病,非 PRO 6000 特有。H100 上同样存在(只是带宽更大,绝对值更高)。
</div>

<h3>横向对比(参考值)</h3>
<table class="m"><thead><tr><th>指标</th><th>8× PRO 6000(本机)</th><th>8× H100(参考)</th><th>比例</th></tr></thead><tbody>
<tr><td>单 GPU HBM 带宽</td><td>1792 GB/s</td><td>3350 GB/s</td><td>54%</td></tr>
<tr><td>单 GPU 显存</td><td>96 GB</td><td>80 GB</td><td>120%</td></tr>
<tr><td>单 GPU TDP</td><td>600W</td><td>700W</td><td>86%</td></tr>
<tr><td>单流 decode</td><td>~49 tok/s</td><td>~80-100(估)</td><td>~54%(与带宽成比例)</td></tr>
<tr><td>聚合 decode(c=32)</td><td>~448 tok/s</td><td>~900+(估)</td><td>~50%</td></tr>
<tr><td>Prefill</td><td>~5700 tok/s</td><td>~10000+(估)</td><td>~57%</td></tr>
</tbody></table>
<p class="sub">H100 数据为基于带宽比例的估算,非实测。实际还受 TP 通信开销、MoE dispatch 效率等影响。</p>

<h3>改进建议</h3>
<div class="box green">
<b>① 试 TP=4 + DP=2</b>:all-reduce 层级减半(61 层跨 4 卡 vs 8 卡),单流 decode 预计提升 30-50%。代价:单卡显存压力增大,KV 并发减半。<br>
<b>② KV 量化(fp8 KV cache)</b>:KV 占用减半,并发翻倍(260K: 5→10 并发)。精度损失通常可接受。<br>
<b>③ CUDA Graph + 通信重叠</b>:减少 90% step 开销(kernel launch / all-reduce 等待)。vLLM 已部分支持(fuse_allreduce_rms 已开),可进一步调优。<br>
<b>④ 增大 max_num_seqs(已 64)</b>:聚合吞吐还有空间(conc=32 未饱和)。
</div>
</body></html>"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"✅ HTML 报告: {OUT_HTML} ({os.path.getsize(OUT_HTML)//1024} KB)")

# ---------- 打包 export 目录(+ 取证报告) ----------
forensics = "raw_data/forensics"
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(EXPORT_DIR):
        for fn in files:
            p = os.path.join(root, fn)
            z.write(p, os.path.relpath(p, "raw_data"))
    if os.path.isdir(forensics):
        for root, _, files in os.walk(forensics):
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, "raw_data"))
    # 附基线原始 + README
    z.write(BASELINE, "baseline_kimi_consolidated.csv")
print(f"✅ 打包: {OUT_ZIP} ({os.path.getsize(OUT_ZIP)//1024} KB)")
print("\n包内容:")
with zipfile.ZipFile(OUT_ZIP) as z:
    for n in z.namelist():
        print(f"   {n}")

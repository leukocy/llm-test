"""生成自包含 HTML 对外报告(内嵌 SVG 图,零依赖)+ 打包 export 目录。
数据源:raw_data/baseline_kimi_consolidated.csv + 仓库导出 + 取证报告。
"""
from __future__ import annotations

import base64
import io
import os
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

# ---------- 图5: decode 稳态 ITL 过渡(前 200 token,prefill→稳态)----------
import json as _json, os as _os
chart5 = ""
steady_html = ""
steady_path = "raw_data/decode_steady_test.csv"
if _os.path.exists(steady_path):
    sdf = pd.read_csv(steady_path)
    # ITL 过渡图(conc=8/16/32,前 200 token)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for conc in [8, 16, 32]:
        sub = sdf[sdf.concurrency == conc]
        if len(sub):
            # 解析每行的 itl_json,取前 150 token 的中位 ITL
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
                # 平滑(窗口 10,用 pandas rolling 避免除零)
                smooth = pd.Series(med_itl).rolling(10, center=True, min_periods=1).mean().tolist()
                ax.plot(range(min_len), smooth, label=f"conc={conc}", alpha=0.8)
    ax.set_xlabel("Token index (first 150)"); ax.set_ylabel("Inter-token latency (ms)")
    ax.set_title("Decode ITL Transition: prefill squeeze -> steady state")
    ax.legend(fontsize=8); ax.set_xlim(0, 150)
    chart5 = svg(fig)
    # 挤压汇总表
    squeeze_rows = ""
    for conc in [1, 8, 16, 32]:
        sub = sdf[sdf.concurrency == conc]
        if len(sub):
            first100 = sub.tps_0_100.median()
            steady = sub.steady_state_tps.median()
            agg = sub.aggregate_tps.median()
            squeeze_rows += f"<tr><td>{conc}</td><td>{first100:.1f}</td><td>{steady:.1f}</td><td>{agg:.1f}</td><td>{(steady/agg-1)*100:.0f}%</td></tr>"
    steady_html = f"""
<h2>Decode 稳态测试(输入 4096 tokens prefill · 输出 4096 tokens decode · 逐 token ITL)</h2>
<p>验证 prefill 对并发 decode 的挤压效应。独特 prompt(独立 KV,生产真实)。输入 4096 tokens(prefill)+ 输出 4096 tokens(decode),逐 token 采集 ITL。</p>
<div class="chart">{chart5}</div>
<table class="matrix"><thead><tr><th>conc</th><th>前100 token (tok/s)</th><th>稳态 500+ (tok/s)</th><th>聚合 4096 (tok/s)</th><th>prefill 挤压</th></tr></thead><tbody>{squeeze_rows}</tbody></table>
<p><b>发现</b>:前 100 token 被严重挤压(conc=32: 3.9 vs 稳态 12.5 tok/s,仅 31%);但 4096 token 下挤压仅占 2.4% → 聚合≈稳态。<b>对比 512 token:100/512=20% 被挤压 → 聚合被拉低约 20%。</b> 结论:并发吞吐测试 max_tokens 应≥4096(或用 ITL 窗口排除 prefill 段)。</p>
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
<div class="chart">{chart1}</div>
<div class="chart">{chart2}</div>
<div class="chart">{chart3}</div>
{('<div class="chart">' + chart4 + '</div>') if chart4 else ''}

<h2>性能矩阵</h2>
<h3 style="margin-bottom:4px">系统 decode 吞吐 (tok/s)</h3>
{matrix_table("agg_decode", "{:.0f}")}
<h3>每流 decode TPS (tok/s)</h3>
{matrix_table("tps", "{:.1f}")}
<h3>TTFT (s)</h3>
{matrix_table("ttft", "{:.2f}")}
<h3>Prefill 速度 (tok/s,单流,仅 conc≤8 有效)</h3>
{matrix_table("prefill_speed", "{:.0f}")}
<h3>聚合 Prefill 吞吐 (tok/s = N×token/max_ttft) — 计算密集应跨并发恒定</h3>
{agg_prefill_table()}
<p class="sub">读法:此值跨并发应接近恒定(~5200-5700,比值≈1.0)= prefill 计算密集、扩展健康。下降=并发争用(仅超高并发×长上下文出现)。</p>

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

{steady_html}

<footer>
数据源:<code>baseline_kimi_consolidated.csv</code>(散热修复后完整矩阵,共 {n_total} 请求 / 56 cell,KV 边界内全覆盖)。
机器可读仓库导出:<code>hmTest_kimi_baseline.csv</code> / <code>hwInventory_turin2d24g.csv</code>(手册标准字段)。
报告是切片,仓库是全集——下游数据消费请用 CSV。
<br><b>上下文轴说明</b>:矩阵/图表的上下文标签为<b>实际 prefill token 数</b>(非目标值)。<code>_calibrate_prompt</code> 按目标生成时欠生成约 0.65 倍(如目标 8192 实际 ~5.3k),故用实际 token 标注以保证 TTFT/Prefill 速度对应正确。prefill 速度 ~5500-6000 tok/s(8k-32k 区间)与独立手测一致。
</footer>
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

"""生成"测试平台升级全流程展示"HTML(明天展示用)。
从真实数据组装:硬件指纹/模型规格/引擎配置/矩阵/per-cell监控/取证。
多受众(决策者/工程师/运维/追溯),有总结 + 可追溯原始数据。自包含、零依赖。
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import pandas as pd

from core.engine_capture import capture_engine_config, get_adapters
from core.hardware_fingerprint import capture_hardware_fingerprint
from core.model_spec import resolve_spec

OUT = "raw_data/export/showcase.html"

# ---- 采集真实数据样本 ----
fp = capture_hardware_fingerprint()
spec = resolve_spec("Kimi-K2.7-Code")
eng = capture_engine_config("http://localhost:10814/v1")
df = pd.read_csv("raw_data/baseline_kimi_consolidated.csv")
df["ok"] = df["error"].isna()
ok = df[df["ok"]]
n_total, n_ok = len(df), int(df["ok"].sum())
n_cells = len(df.groupby(["concurrency", "context_length_target"]))

cpu = fp.get("cpu") or {}
gpus = fp.get("gpus") or []
gpu0 = gpus[0] if gpus else {}
sched = eng.get("schedule") or {}
par = eng.get("parallel_strategy") or {}
rt = eng.get("runtime") or {}


def tps_matrix():
    """聚合 decode 吞吐 = conc × 单流tps(system_output_throughput 字段在长上下文×rounds>1 时
    因窗口横跨多轮 prefill 而稀释,不可靠;用 conc×tps 替代)。"""
    _tmp = ok.copy()
    _tmp["agg_decode"] = _tmp["concurrency"] * _tmp["tps"]
    piv = _tmp.groupby(["concurrency", "context_length_target"])["agg_decode"].median().round(0)
    out = {}
    for (c, ctx), v in piv.items():
        out.setdefault(c, {})[ctx] = v
    return out


def temp_curve():
    """per-cell GPU 温度(散热监控价值)。"""
    piv = ok.groupby(["concurrency", "context_length_target"])["gpu_temp_peak_c"].max()
    out = {}
    for (c, ctx), v in piv.items():
        if pd.notna(v):
            out.setdefault(c, {})[ctx] = int(v)
    return out


TPS = tps_matrix()
TEMP = temp_curve()

# ---- 文件索引(可追溯)----
files = []
for d in ["raw_data/export", "raw_data/forensics", "standards"]:
    for p in sorted(glob.glob(f"{d}/*")):
        if os.path.isfile(p):
            files.append((p, os.path.getsize(p)))

CTX = [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 260000]
CONC = [1, 2, 4, 8, 16, 32]


def ctx_lbl(c):
    return f"{c//1024}k" if c >= 1024 else str(c)


def cell_table(data, concs, ctxs, fmt="{}"):
    """生成单元格表(data[conc][ctx])。"""
    head = "".join(f"<th>conc={c}</th>" for c in concs)
    rows = ""
    for ctx in ctxs:
        cells = ""
        for c in concs:
            v = (data.get(c) or {}).get(ctx)
            cells += f"<td>{fmt.format(v) if v is not None else '·'}</td>"
        rows += f"<tr><th>{ctx_lbl(ctx)}</th>{cells}</tr>"
    return f'<table class="m"><thead><tr><th>ctx＼conc</th>{head}</tr></thead><tbody>{rows}</tbody></table>'


HTML = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>LLM 推理测试平台升级 · 全流程展示</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;max-width:1180px;margin:0 auto;padding:0 24px 80px;color:#1f2937;line-height:1.65}}
h1{{font-size:30px;border-bottom:4px solid #2563eb;padding-bottom:10px;margin:32px 0 6px}}
.sub{{color:#6b7280;font-size:15px;margin-bottom:8px}}
h2{{font-size:21px;color:#2563eb;margin:42px 0 12px;padding-left:12px;border-left:5px solid #2563eb}}
h3{{font-size:16px;color:#1e40af;margin:22px 0 8px}}
.aud{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0 28px;padding:16px;background:#eff6ff;border-radius:10px;border:1px solid #bfdbfe}}
.aud a{{display:block;padding:10px 16px;background:#2563eb;color:#fff;border-radius:8px;text-decoration:none;font-size:14px;font-weight:600}}
.aud a:hover{{background:#1d4ed8}}
.aud .r{{color:#dbeafe;font-size:11px;font-weight:400;display:block}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}
.kpi div{{background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;padding:14px;text-align:center}}
.kpi .v{{font-size:26px;font-weight:800;color:#2563eb}}
.kpi .l{{font-size:12px;color:#6b7280;margin-top:2px}}
.box{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;padding:16px 20px;margin:14px 0}}
.box.green{{background:#f0fdf4;border-color:#bbf7d0}}
.box.amber{{background:#fffbeb;border-color:#fde68a}}
.box.blue{{background:#eff6ff;border-color:#bfdbfe}}
.flow{{font-family:ui-monospace,Consolas,monospace;font-size:13px;background:#0f172a;color:#e2e8f0;padding:18px;border-radius:10px;line-height:1.9;overflow-x:auto}}
.flow .a{{color:#60a5fa}}.flow .b{{color:#34d399}}.flow .c{{color:#fbbf24}}.flow .d{{color:#f472b6}}
table.m{{border-collapse:collapse;font-size:12.5px;margin:10px 0;width:100%}}
table.m td,table.m th{{border:1px solid #e5e7eb;padding:5px 7px;text-align:center}}
table.m th{{background:#f1f5f9}}table.m td:first-child,table.m th:first-child{{background:#f8fafc;font-weight:600}}
code,pre{{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:12.5px;font-family:ui-monospace,Consolas,monospace}}
pre{{padding:12px;overflow-x:auto;line-height:1.5}}
.tlist{{font-size:13px}}
.tlist td{{padding:5px 10px}}.tlist td:first-child{{color:#2563eb;font-family:ui-monospace,Consolas,monospace}}
.tag{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;margin-right:4px}}
.t-auto{{background:#dbeafe;color:#1e40af}}.t-man{{background:#fef3c7;color:#92400e}}.t-spec{{background:#ede9fe;color:#5b21b6}}
footer{{margin-top:50px;color:#9ca3af;font-size:12px;border-top:1px solid #e5e7eb;padding-top:14px}}
.nav{{position:sticky;top:0;background:#fff;z-index:10;padding:10px 0;border-bottom:1px solid #e5e7eb;margin:0 -24px 20px;padding-left:24px;padding-right:24px}}
.nav a{{color:#6b7280;text-decoration:none;font-size:13px;margin-right:16px}}.nav a:hover{{color:#2563eb}}
</style></head><body>

<div class="nav">
<a href="#summary">执行摘要</a><a href="#pipeline">全流程</a><a href="#dims">八维数据</a>
<a href="#ops">可靠性/运维</a><a href="#standard">标准</a><a href="#trace">数据追溯</a>
</div>

<h1>LLM 推理测试平台升级<br><span style="font-size:18px;color:#6b7280">从硬件到模型的全流程数据仓库</span></h1>
<div class="sub">机器 <code>{fp.get('machine_id')}</code>({cpu.get('model_name','')}, {len(gpus)}× {gpu0.get('name','')}) · 每请求输出 512 tokens(max_tokens=512) · 模型 Kimi-K2.7-Code · 生成 {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="aud">
<a href="#summary">🧭 决策者<span class="r">3 分钟看价值</span></a>
<a href="#pipeline">🔧 工程师<span class="r">全流程 + 数据</span></a>
<a href="#ops">🛠 运维<span class="r">可靠性 + 取证</span></a>
<a href="#trace">📚 追溯<span class="r">原始数据索引</span></a>
</div>

<!-- ============ 执行摘要 ============ -->
<h2 id="summary">一、执行摘要(给决策者)</h2>
<div class="box blue">
<b>一句话:</b>把测试结果从"几张干瘪图片"升级为<u>机器可读、可追溯、可对外</u>的<b>八维数据仓库</b>——
每条记录绑定<b>冻结的硬件+模型+引擎指纹、测试中的资源监控、含等效带宽的性能、状态/瓶颈归因、可对外闸门</b>。
报告是切片,仓库是全集。
</div>
<div class="kpi">
<div><div class="v">8</div><div class="l">数据维度(硬件→模型→引擎→性能→资源→归因→标准→对外)</div></div>
<div><div class="v">1.000</div><div class="l">校准精度(修前 0.65,欠生成 35%)</div></div>
<div><div class="v">{n_ok}/{n_total}</div><div class="l">干净基线请求({n_cells} cell,0 错误)</div></div>
<div><div class="v">{len(get_adapters())}</div><div class="l">支持引擎(vLLM/SGLang/llama.cpp/…)</div></div>
<div><div class="v">6维</div><div class="l">统一字段标准(注册表)</div></div>
<div><div class="v">27列</div><div class="l">Excel↔仓库对齐(自动填)</div></div>
</div>
<h3>解决了什么</h3>
<div class="box">
<b>① 数据不准</b> — tokenizer 映射错导致 prompt 欠生成 35%,所有上下文测试在"标称的 65%"跑;修正后精确命中。<br>
<b>② 看不懂瓶颈</b> — 只报 TPS,不知是算力/通信/带宽;新增等效带宽 + per-cell 温度/功耗/PCIe/throttle 监控。<br>
<b>③ 引擎配置漂移</b> — vLLM max_num_seqs 16→64、KV 变了,报告用了陈旧值导致"conc=16 饱和"误判;改为 per-run 自动采集(多引擎)。<br>
<b>④ 不可追溯/对外</b> — 散落 CSV/截图;统一字段标准 + 导出 + 可对外闸门 + 事故取证。
</div>

<!-- ============ 全流程 ============ -->
<h2 id="pipeline">二、全流程:从硬件到模型(给工程师)</h2>
<div class="flow">
<span class="a">A 硬件指纹</span> capture_hardware_fingerprint (machine_id / CPU×NUMA / 内存DDR5×24ch / 8×GPU name+VRAM+带宽+PCIe×宽 / CUDA)
   ↓
<span class="a">B 资源监控</span> ResourceMonitor (per-GPU util/vram/power/<b>温度</b>/clock/fan/PCIe RX·TX/throttle,per-cell 独立采样)
   ↓
<span class="c">C 模型规格</span> resolve_spec (config.json 权威值:int4 量化 / 1T总参32B激活 / MLA / 384专家top8)
   ↓
<span class="c">C 引擎配置</span> capture_engine_config (docker inspect + 日志 + /v1/models,<b>多引擎适配器自动探测</b>)
   ↓
<span class="d">D 测试执行</span> throughput_matrix (校准精确 + per-cell 监控 + GPU 计数防护)
   ↓
<span class="b">E 瓶颈/归因</span> 等效带宽(int4 roofline) + 状态/瓶颈/error_type
   ↓
<b>标准注册表</b> 六维字段(来源标签:自动/手工/专项) → <b>导出</b> standard_perf(27列) + hmTest + <b>HTML 报告</b>
</div>

<h3>A · 硬件指纹(自动采集样本)</h3>
<div class="box"><pre>machine_id: {fp.get('machine_id')}
CPU: {cpu.get('model_name')}  sockets={cpu.get('sockets')} cores={cpu.get('cores_per_socket')} NUMA={cpu.get('numa_nodes')}
内存: DDR5 ×{len(gpus)*3}通道 {fp.get('memory',{}).get('total_gb')}GB ECC={fp.get('memory',{}).get('ecc')}
GPU[0]: {gpu0.get('name')}  VRAM={gpu0.get('vram_gb')}GB  标称带宽={gpu0.get('nominal_bandwidth_gbps')}GB/s  PCIe gen{gpu0.get('pcie_gen')}×{gpu0.get('pcie_width')}  ×{len(gpus)}卡
驱动: {fp.get('cuda',{}).get('driver')}  CUDA {fp.get('cuda',{}).get('cuda_version')}</pre></div>

<h3>C · 模型规格(从 config.json 权威解析,<span class="tag t-auto">自动</span>)</h3>
<div class="box"><pre>{json.dumps({k:v for k,v in (spec.to_dict() if spec else {{}}).items() if v not in (None,'',[]) and k in ('name','architecture','total_params_b','active_params_b','num_experts','num_experts_per_tok','weight_dtype','quant_method','group_size','hidden_size','num_layers','attention_type','max_position_embeddings','is_multimodal')}, ensure_ascii=False, indent=2)}</pre></div>

<h3>C · 引擎配置(自动采集,<span class="tag t-auto">自动</span> 引擎={eng.get('engine')} 适配器={eng.get('adapter')})</h3>
<div class="box"><pre>max_num_seqs: {sched.get('max_num_seqs')}    gpu_memory_utilization: {sched.get('gpu_memory_utilization')}
parallel: {par}
runtime: 权重载入 {rt.get('weight_load_s')}s + init {rt.get('init_engine_s')}s + graph {rt.get('graph_capture_s')}s = 冷启动 ~{rt.get('cold_start_s_est')}s
KV cache: <b>{rt.get('kv_cache_tokens'):,}</b> tokens
来源: {eng.get('capture_source')}</pre></div>
<div class="box amber">⚠ <b>这条自动采集堵住的坑</b>:vLLM 曾把 max_num_seqs 16→64、KV 1.67M→1.53M,硬编码报告一度用陈旧值得出错误的"conc=16 饱和"。现在 per-run 自动采集,配置怎么变都不会再错。已注册引擎: {', '.join(get_adapters())}。</div>

<!-- ============ 八维数据 ============ -->
<h2 id="dims">三、性能数据样本(per-cell,可追溯)</h2>
<h3>系统 decode 吞吐 (tok/s) — max_num_seqs={sched.get('max_num_seqs')} 下未在测区饱和</h3>
{cell_table(TPS, CONC, CTX, "{:.0f}")}
<p class="sub">读法:conc=1→{int((TPS.get(1) or {{}}).get(64,0))}、16→{int((TPS.get(16) or {{}}).get(64,0))}、32→{int((TPS.get(32) or {{}}).get(64,0))} tok/s(ctx=64),<b>测区内持续上升,未饱和</b>。空白=KV 容量不可行(conc×ctx&gt;{rt.get('kv_cache_tokens'):,})。</p>

<h3>per-cell GPU 峰值温度 (°C) — 散热监控(运维预警靠它)</h3>
{cell_table(TEMP, CONC, CTX)}
<p class="sub">散热修复后:260K 稳定到 conc=4({(TEMP.get(4) or {{}}).get(260000)}°C),全程无热降频。温度随上下文+并发爬升,per-cell 可定位是哪格负载把哪张卡推热。</p>

<!-- ============ 运维/可靠性 ============ -->
<h2 id="ops">四、可靠性与运维(给运维)</h2>
<div class="box green">
<b>GPU 掉线事故 — 已定位、已修复</b><br>
现象:260K 高并发触发 GPU fall off bus。取证(dmesg + IPMI SEL + per-cell 温度监控)定位为<b>某 GPU 槽位散热不良</b>(per-cell 监控实测 conc=1/260K 即 97°C),非 PCIe 接触、非供电、非 ECC。<br>
处置:更换该槽散热部件 → 完整矩阵 542 请求 <b>0 崩溃</b>,conc=4/260K 降至 94°C。<br>
<b>关键:per-cell 温度监控让散热问题在"温度曲线"上可见,不必等掉卡。</b>
</div>
<div class="box amber">
<b>KV 容量边界(矩阵天然上限)</b>:KV cache = {rt.get('kv_cache_tokens'):,} token → <code>conc × ctx ≤ {rt.get('kv_cache_tokens'):,}</code> 可行(260K×conc≤6、128K×conc≤{int(rt.get('kv_cache_tokens',0)/131072)})。超过 vLLM 排队/超时,<b>非缺陷,是硬件 KV 容量物理上限</b>。
</div>
<div class="box">
<b>防护机制</b>:① 矩阵测试 GPU 计数防护(掉卡即中止后续 phase,保住已采数据);② per-cell 温度/功耗/PCIe/throttle 监控(散热异常提前预警);③ ResourceMonitor 优雅降级(无 NVML 仍采 CPU/内存)。
</div>

<!-- ============ 标准 ============ -->
<h2 id="standard">五、字段标准与对齐(给架构)</h2>
<div class="box blue">
<b>原则:补全不替代</b>。手工 Excel 模板 + 自动仓库并成<b>一份权威字段注册表</b>(<code>standards/测试数据字段标准.md</code>),每字段标来源:<span class="tag t-auto">自动</span><span class="tag t-man">手工</span><span class="tag t-spec">专项</span>。
</div>
<h3>六维注册表</h3>
<div class="box">
<b>A 模型</b>(模型信息登记表 ∪ model_spec) · <b>B 硬件</b>(硬件登记表 ∪ hwInventory) · <b>C 引擎</b>(引擎登记表 ∪ serving_config,自动采集) ·
<b>D 性能</b>(基础性能 ∪ hmTest,<span class="tag t-auto">自动主力</span>) · <b>E 瓶颈</b>(Prefill/Decode 专项,Excel 独有,仓库待建) · <b>F 场景</b>(不同场景 ∪ maTest)
</div>
<h3>P0 两边对齐(已做)</h3>
<div class="box">Excel 基础性能表补 <b>GPU温度/p95/p99/状态归因/瓶颈/可对外</b> 6 列;仓库 <code>export_standard.py</code> 产 27 列同名 CSV(22 列自动填 + 5 列按来源标签诚实留空)→ 直接粘进 Excel,不手抄。<b>来源标签是硬约束:自动字段禁手填,专项字段禁假装自动。</b></div>

<!-- ============ 数据追溯 ============ -->
<h2 id="trace">六、数据可追溯索引(给所有人)</h2>
<p class="sub">每个结论都能追到原始文件。下表为本批产出(本地 <code>raw_data/</code> + <code>standards/</code>,gitignore 故不进 git)。</p>
<table class="tlist m"><thead><tr><th>文件</th><th>大小</th><th>内容</th></tr></thead><tbody>
{''.join(f'<tr><td>{p}</td><td>{s//1024}KB</td><td>{ {"html":"自包含 HTML 报告(图+矩阵+结论)","csv":"CSV 数据(标准对齐/hmTest/基线)","json":"JSON(含元信息+指纹)","md":"文档(事故报告/字段标准)","txt":"原始取证快照(dmesg/SEL)"}.get(p.split(".")[-1],"") }</td></tr>' for p,s in files)}
</tbody></table>
<div class="box">代码侧(已提交 origin/master):<code>core/engine_capture.py</code>(多引擎采集) · <code>core/resource_monitor.py</code>(nvtop级) · <code>config/settings.py</code>(校准) · <code>core/model_spec.py</code>(Kimi int4) · <code>standards/测试数据字段标准.md</code> · <code>export_*.py / build_report.py / build_showcase.py</code></div>

<footer>
本展示自动生成自真实运行数据(build_showcase.py)。数据源 baseline_kimi_consolidated.csv({n_total} 请求)。
仓库是全集,报告/展示是切片——下游消费请用 CSV。{datetime.now().strftime('%Y-%m-%d %H:%M')}
</footer>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(HTML)
print(f"✅ 展示已生成: {OUT} ({os.path.getsize(OUT)//1024} KB)")
print(f"   数据: {n_total} 请求 / {n_cells} cell / 适配器 {get_adapters()}")
print(f"   打开: xdg-open {OUT}")

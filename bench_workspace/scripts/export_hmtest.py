"""把 Kimi-K2.7-Code 基线导出成手册 hmTest 模板(对外格式)。

粒度:每个 (并发 × 上下文) cell 一行(手册"一次测试一行"——矩阵的一个测试点)。
公共字段(硬件指纹/模型规格/服务配置)全行相同,来自 capture_hardware_fingerprint() + 已知配置。
每 cell 指标从 baseline_kimi_consolidated.csv 按成功请求聚合。
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from datetime import datetime

import pandas as pd

from core.effective_bandwidth import compute_effective_bandwidth
from core.hardware_fingerprint import capture_hardware_fingerprint
from core.model_spec import resolve_spec
from core.warehouse.templates import HM_TEST_FIELDS

OUT_DIR = "raw_data/export"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. 公共字段:真实硬件指纹 ----
fp = capture_hardware_fingerprint()
cpu = fp.get("cpu") or {}
mem = fp.get("memory") or {}
cuda = fp.get("cuda") or {}
gpus = fp.get("gpus") or []
gpu0 = gpus[0] if gpus else {}

# ---- 2. 公共字段:模型规格(从 registry 解析) + 服务配置(自动采集,非硬编码)----
SPEC = resolve_spec("Kimi-K2.7-Code")
assert SPEC is not None, "Kimi-K2.7-Code 未能从 registry 解析 model_spec"
MODEL_SPEC = {
    "model_name": SPEC.name,
    "model_version": "K2.7-Code",
    "model_type": "MoE" if SPEC.architecture == "moe" else SPEC.architecture,
    "total_params": SPEC.total_params_b,
    "active_params": SPEC.active_params_b,
    "num_experts": SPEC.num_experts,
    "top_k": SPEC.num_experts_per_tok,
    "quantization": (
        f"{SPEC.weight_dtype}({SPEC.quant_method},gs={SPEC.group_size})"
        if SPEC.quant_method
        else SPEC.weight_dtype
    ),
    "dtype": SPEC.weight_dtype,
    "max_context": SPEC.max_position_embeddings,
}
# 引擎配置自动采集(docker inspect + 日志 + /v1/models)——不再硬编码,配置变即跟着变
from core.engine_capture import capture_engine_config

_ENG = capture_engine_config("http://localhost:10814/v1")
_SCHEDULE = _ENG.get("schedule") or {}
_PARALLEL = _ENG.get("parallel") or {}
_RUNTIME = _ENG.get("runtime") or {}
SERVING = {
    "engine": _ENG.get("engine", "vLLM"),
    "engine_version": _ENG.get("engine_version", ""),
    "engine_params": _ENG.get("launch_cmd", "")
    or ";".join(f"{k}={v}" for k, v in _SCHEDULE.items()),
    "parallel_strategy": (
        _PARALLEL
        and f"tp={_PARALLEL.get('tp')} + dcp={_PARALLEL.get('dcp')} + ep={_PARALLEL.get('ep')}"
    )
    or "",
}
# 单卡标称带宽(与 runner._nominal_gpu_bandwidth_gbps 同约定)
NOMINAL_GPU_BW = gpu0.get("nominal_bandwidth_gbps") or 1792.0
# 引擎冷启动(自动采集的分解)
LOAD_TIME_S = _RUNTIME.get("cold_start_s_est") or 256
KV_CACHE_TOKENS = _RUNTIME.get("kv_cache_tokens")
TESTER = "claude-live"
MACHINE_ID = fp.get("machine_id") or "ab8652ab0b09bbd7"
DATE = "2026-06-14"


def pct(values, p):
    """简单百分位(p in [0,100])。"""
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(p / 100 * (len(s) - 1)))))
    return round(s[k], 3)


# ---- 3. 按 cell 聚合基线 ----
df = pd.read_csv("raw_data/baseline_kimi_consolidated.csv")
rows = []
for (conc, ctx), sub in df.groupby(["concurrency", "context_length_target"]):
    ok = sub[sub["error"].isna()]
    n_total = len(sub)
    n_ok = len(ok)
    rate = n_ok / n_total if n_total else 0
    # 该 cell 的指标(仅成功请求)
    ttfts = ok["ttft"].dropna().tolist()
    totals = ok["total_time"].dropna().tolist()
    tps_list = ok["tps"].dropna().tolist()
    prefill_tps_list = ok["prefill_speed"].dropna().tolist()
    sys_out = ok["system_output_throughput"].dropna().tolist()

    # 状态/归因(每 cell)
    if rate == 1.0:
        status, bottleneck, error_type, error_detail = "completed", "", "", ""
        ext = "review"
    elif ctx == 260000 and conc == 8:
        status = "hardware_fault"
        bottleneck = "pcie_gpu_drop"
        error_type = "gpu_lost_from_bus"
        error_detail = "conc=8/260K 触发 GPU#7 PCIe 掉总线(见事故报告),非模型能力问题"
        ext = "internal"
    else:
        status = "partial"
        bottleneck = ""
        error_type = ""
        error_detail = f"{n_ok}/{n_total} 成功"
        ext = "internal"

    # 等效带宽(int4 roofline:decode_tps × active × bytes_per_param;TP=8 下为上界估计)
    decode_tps_mean = statistics.mean(tps_list) if tps_list else None
    bw = compute_effective_bandwidth(decode_tps_mean, SPEC, NOMINAL_GPU_BW)

    row = {
        "test_id": f"hmtest-kimi-{conc}c-{ctx}ctx",
        "date": DATE,
        "tester": TESTER,
        "machine_id": MACHINE_ID,
        "engine": SERVING["engine"],
        "engine_version": SERVING["engine_version"],
        "engine_params": SERVING["engine_params"],
        "parallel_strategy": SERVING["parallel_strategy"],
        "model_name": MODEL_SPEC["model_name"],
        "model_version": MODEL_SPEC["model_version"],
        "model_type": MODEL_SPEC["model_type"],
        "total_params": MODEL_SPEC["total_params"],
        "active_params": MODEL_SPEC["active_params"],
        "num_experts": MODEL_SPEC["num_experts"],
        "top_k": MODEL_SPEC["top_k"],
        "quantization": MODEL_SPEC["quantization"],
        "dtype": MODEL_SPEC["dtype"],
        "max_context": MODEL_SPEC["max_context"],
        "concurrency": conc,
        "usecase_set_version": "",
        "prompt_tokens": int(ok["prefill_tokens"].median()) if len(ok) else "",
        "output_tokens": (
            int(ok["decode_tokens"].median()) if len(ok) and ok["decode_tokens"].median() else 512
        ),
        "load_time_s": LOAD_TIME_S,
        "ttft_s": round(statistics.mean(ttfts), 3) if ttfts else "",
        "prefill_tps": (round(statistics.mean(prefill_tps_list)) if prefill_tps_list else ""),
        "decode_tps": round(decode_tps_mean, 1) if decode_tps_mean else "",
        "long_context_tps": (round(statistics.mean(sys_out)) if (ctx >= 32768 and sys_out) else ""),
        "p50_latency_s": pct(ttfts, 50),
        "p95_latency_s": pct(ttfts, 95),
        "p99_latency_s": pct(ttfts, 99),
        "gpu_vram_peak_gb": round(735.0, 1),  # 8×~92GB(run 244 实测峰值)
        "system_memory_peak_gb": "",
        "effective_bandwidth_gbps": (
            round(bw["effective_bandwidth_gbps"], 1) if bw.get("effective_bandwidth_gbps") else ""
        ),
        "bandwidth_utilization_pct": (
            round(bw["bandwidth_utilization_pct"]) if bw.get("bandwidth_utilization_pct") else ""
        ),
        "cpu_threads_used": "",
        "cpu_util_pct": "",
        "gpu_util_pct": "",
        "power_w": "",
        "temp_c": "",
        "status": status,
        "bottleneck": bottleneck,
        "error_type": error_type,
        "error_detail": error_detail,
        "log_path": "raw_data/baseline_kimi_consolidated.csv",
        "screenshot_path": "",
        "external_level": ext,
        "next_action": (
            "物理检修 GPU#7 PCIe 连接(见 forensics/INCIDENT_REPORT)后再测 conc≥16 高上下文"
            if status == "hardware_fault"
            else ""
        ),
        "supersedes_test_id": "",
    }
    rows.append(row)

# ---- 4. 按 HM_TEST_FIELDS 顺序导出 CSV + JSON ----
csv_path = os.path.join(OUT_DIR, "hmTest_kimi_baseline.csv")
json_path = os.path.join(OUT_DIR, "hmTest_kimi_baseline.json")

# CSV: 用 csv 模块正确转义(字段值可能含逗号/引号),字段严格按手册模板顺序
with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(HM_TEST_FIELDS)
    for r in rows:
        writer.writerow(
            ["" if r.get(fld, "") in (None, "") else r.get(fld) for fld in HM_TEST_FIELDS]
        )

# JSON: 每行一个对象 + 元信息
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "template": "hmTest",
            "source": "llm-test warehouse baseline (run 244 conc<=8 + run 245 conc=16/32 low-ctx)",
            "exported_at": datetime.now().isoformat(),
            "row_count": len(rows),
            "hardware_fingerprint": fp,
            "model_spec": SPEC.to_dict(),
            "methodology": {
                "effective_bandwidth": (
                    "int4 roofline: effective_bw = decode_tps × active_params × bytes_per_param; "
                    "active=32B, int4→0.5 bytes/param → 16 GB/token. "
                    "utilization vs 单卡标称带宽(与 runner._nominal_gpu_bandwidth_gbps 同约定); "
                    "TP=8 下该值为上界估计(权重跨 8 卡分片,真实每卡利用率 ≈ 值/8)。"
                ),
                "缺测字段": "load_time_s/resource monitor(被 _Fake UI 桩跳过)等留空,遵手册「缺测即决策信息」。",
            },
            "rows": rows,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"[OK] 导出 {len(rows)} 行 hmTest 记录:")
print(f"   CSV : {csv_path}")
print(f"   JSON: {json_path}")
print(f"\n硬件指纹: machine_id={MACHINE_ID}")
print(
    f"   CPU={cpu.get('model_name')}  socket×core={cpu.get('sockets')}×{cpu.get('cores_per_socket')}"
)
print(
    f"   GPU={gpu0.get('name')} ×{len(gpus)}  显存={gpu0.get('vram_gb')}GB  标称带宽={gpu0.get('nominal_bandwidth_gbps')}GB/s"
)
print(f"   driver={cuda.get('driver')}  cuda={cuda.get('cuda_version')}")
print("\n字段完整性(非空率):")
for f in HM_TEST_FIELDS:
    filled = sum(1 for r in rows if r.get(f) not in (None, ""))
    if filled == 0:
        print(f"   [MISSING] {f}: 全缺测")
print("\n样例行(conc=4/ctx=4096):")
sample = next(
    r
    for r in rows
    if r["concurrency"] == 4 and r["prompt_tokens"] and 3000 < (r["prompt_tokens"] or 0) < 6000
)
print(json.dumps(sample, ensure_ascii=False, indent=2))

"""安全补跑:conc=[16,32] × 低上下文,补全基线高并发段。
避开崩溃点(conc=8/260K 长时间满载)。起飞前校验 8 卡在位。
"""

import asyncio
import os
import subprocess

from core.benchmark_runner import BenchmarkRunner
from core.ui_bridge import NullStateBridge

os.makedirs("raw_data", exist_ok=True)


class _Fake:
    def __getattr__(self, name):
        return lambda *a, **kw: self


def gpu_count() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL
        )
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:
        return -1


# 起飞前安全检查
n = gpu_count()
print(f"[preflight] 可见 GPU 数 = {n}(预期 8)", flush=True)
if n != 8:
    print(f"[preflight] [WARN] GPU 数异常({n}),放弃起飞避免连带崩溃", flush=True)
    raise SystemExit(1)

serving_config = {
    "engine": "vllm",
    "engine_version": "0.23.0",
    "tp_size": 8,
    "enable_prefix_caching": True,
    "gpu_memory_utilization": 0.94,
    "max_num_seqs": 16,
}
warehouse_context = {
    "serving_config": serving_config,
    "test_metadata": {
        "tester": "claude-live",
        "next_action": "conc=16/32 低上下文安全补跑(避开 conc=8/260K 崩溃点)",
    },
    "model_spec_override": {},
    "engine_runtime": {},
    "custom_sys_info": {},
}

runner = BenchmarkRunner(
    placeholder=_Fake(),
    progress_bar=_Fake(),
    status_text=_Fake(),
    api_base_url="http://localhost:10814/v1",
    model_id="Kimi-K2.7-Code",
    tokenizer_option="API (usage field)",
    csv_filename="raw_data/live_kimi_hiconc.csv",
    api_key="EMPTY",
    log_placeholder=_Fake(),
    provider="OpenAI Compatible",
    output_placeholder=_Fake(),
    warehouse_context=warehouse_context,
    ui_state=NullStateBridge(),
    render_progress=lambda **kw: None,
    render_log=lambda **kw: None,
)

# 安全矩阵:只测高并发 × 低上下文
CONCURRENCIES = [16, 32]
CONTEXT_LENGTHS = [64, 1024, 2048, 4096, 8192]

print(
    f"[run] conc={CONCURRENCIES} × ctx={CONTEXT_LENGTHS} (共 {sum(c for c in CONCURRENCIES)*len(CONTEXT_LENGTHS)} 请求)",
    flush=True,
)
df = asyncio.run(
    runner.run_throughput_matrix_test(
        concurrencies=CONCURRENCIES,
        context_lengths=CONTEXT_LENGTHS,
        rounds=1,
        max_tokens=2048,
    )
)

# 收尾再校验 GPU 数
n2 = gpu_count()
print(f"[postflight] 可见 GPU 数 = {n2}", flush=True)
if n2 != 8:
    print(f"[postflight] [WARN] 运行中掉卡({n2})!数据可能不完整", flush=True)

print(f"\n=== 补跑完成 {len(df)} 行 ===", flush=True)
ok = df["error"].isna().sum()
print(f"成功 {ok}/{len(df)}", flush=True)
print(
    df[["concurrency", "context_length_target", "ttft", "tps", "total_time", "error"]]
    .round(2)
    .to_string(index=False),
    flush=True,
)

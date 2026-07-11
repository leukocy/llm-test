"""全部重测(校准已修正:tokenizer=模型自带,实际 token=目标)。
安全包络:避开 conc=8/260K 崩溃点(GPU#7 未修)。
  Phase A: conc=[1,2] × 全 10 上下文(含真 260K)
  Phase B: conc=[4,8] × [64..131072](无 260K)
  Phase C: conc=[16,32] × 低上下文
每 phase 前校验 8 卡在位;掉卡即中止。
"""

import asyncio
import os
import subprocess

from core.benchmark_runner import BenchmarkRunner
from core.ui_bridge import NullStateBridge

os.makedirs("raw_data", exist_ok=True)


def gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:
        return -1


class _F:
    def __getattr__(self, n):
        return lambda *a, **k: self


def make_runner(tag):
    return BenchmarkRunner(
        placeholder=_F(),
        progress_bar=_F(),
        status_text=_F(),
        api_base_url="http://localhost:10814/v1",
        model_id="Kimi-K2.7-Code",
        tokenizer_option="API (usage field)",
        csv_filename=f"raw_data/recal_{tag}.csv",
        api_key="EMPTY",
        log_placeholder=_F(),
        provider="OpenAI Compatible",
        output_placeholder=_F(),
        warehouse_context={
            "serving_config": {"engine": "vllm", "tp_size": 8, "max_num_seqs": 16},
            "test_metadata": {"tester": "claude-recal", "note": "校准修正后重测"},
            "model_spec_override": {},
            "engine_runtime": {},
            "custom_sys_info": {},
        },
        ui_state=NullStateBridge(),
        render_progress=lambda **k: None,
        render_log=lambda **k: None,
    )


PHASES = [
    (
        "A_conc12_full",
        [1, 2],
        [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 260000],
    ),
    (
        "B_conc48_no260k",
        [4, 8],
        [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    ),
    ("C_conc1632_low", [16, 32], [64, 1024, 2048, 4096, 8192]),
]


async def main():
    total = sum(sum(c for c in conc) * len(ctx) for _, conc, ctx in PHASES)
    print(f"[plan] 3 phases, 共 {total} 请求", flush=True)
    for tag, conc, ctx in PHASES:
        n = gpu_count()
        print(f"\n[phase {tag}] GPU={n} (预期8)  conc={conc} ctx={ctx}", flush=True)
        if n != 8:
            print(f"[ABORT] GPU 数 {n}≠8,停止避免连带崩溃", flush=True)
            return
        r = make_runner(tag)
        await r.run_throughput_matrix_test(
            concurrencies=conc, context_lengths=ctx, rounds=1, max_tokens=2048
        )
        n2 = gpu_count()
        print(f"[phase {tag} done] GPU={n2}", flush=True)
        if n2 != 8:
            print(f"[WARN] 运行后掉卡({n2}),后续 phase 取消", flush=True)
            return
    print("\n[all done] 全部 phase 完成", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

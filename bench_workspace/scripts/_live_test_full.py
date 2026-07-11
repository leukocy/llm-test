"""完整矩阵(散热修复后,含 260K)+ per-cell 监控。风险递增分 phase,GPU 防护。
Phase A: conc=[1,2,4] × 全 10 上下文(安全)
Phase B: conc=[8,16,32] × 低上下文[64..8192](安全)
Phase C: conc=[8,16,32] × 中上下文[16384..131072]
Phase D: conc=[8,16,32] × 260000(历史崩溃点,最后跑)
每 phase 前/后校验 8 卡;掉卡即中止后续(保住已采数据)。
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
        csv_filename=f"raw_data/full_{tag}.csv",
        api_key="EMPTY",
        log_placeholder=_F(),
        provider="OpenAI Compatible",
        output_placeholder=_F(),
        warehouse_context={
            "serving_config": {"engine": "vllm", "tp_size": 8, "max_num_seqs": 16},
            "test_metadata": {"tester": "claude-full", "note": "散热修复后完整矩阵"},
            "model_spec_override": {},
            "engine_runtime": {},
            "custom_sys_info": {},
        },
        ui_state=NullStateBridge(),
        render_progress=lambda **k: None,
        render_log=lambda **k: None,
    )


# KV 可行三角(conc×ctx ≤ ~1.5M KV 容量,避免排队挂死),rounds=3 median 级
PHASES = [
    (
        "A_lowconc_full",
        [1, 2, 4],
        [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 260000],
    ),
    (
        "B_conc8_upto131k",
        [8],
        [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
    ),
    ("C_conc16_upto64k", [16], [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536]),
    ("D_conc32_upto32k", [32], [64, 1024, 2048, 4096, 8192, 16384, 32768]),
]


async def main():
    total = sum(sum(c for c in conc) * len(ctx) for _, conc, ctx in PHASES)
    print(f"[plan] 完整矩阵 {total} 请求,4 phase 风险递增", flush=True)
    for tag, conc, ctx in PHASES:
        n = gpu_count()
        print(f"\n[phase {tag}] GPU={n} conc={conc} ctx={ctx}", flush=True)
        if n != 8:
            print(f"[ABORT] GPU={n}≠8,停止(保住已采数据)", flush=True)
            return
        r = make_runner(tag)
        await r.run_throughput_matrix_test(
            concurrencies=conc, context_lengths=ctx, rounds=3, max_tokens=2048
        )
        n2 = gpu_count()
        print(f"[phase {tag} done] GPU={n2}", flush=True)
        if n2 != 8:
            print(f"[WARN] 掉卡({n2}),后续 phase 取消", flush=True)
            return
    print("\n[all done] 完整矩阵完成", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

"""全量 decode 稳态测试:KV 可行域内所有 (conc×ctx) × max_tokens=4096 + 逐 token ITL。
风险递增,GPU 计数防护。预计 ~4-5h。
"""
import asyncio, json, os, subprocess, time
import pandas as pd
from core.benchmark_runner import BenchmarkRunner
from core.providers.openai import OpenAIProvider
from core.ui_bridge import NullStateBridge

os.makedirs("raw_data", exist_ok=True)
MAX_TOKENS = 2048

# KV 可行域(conc×ctx ≤ ~1.3M),风险递增分 phase
PHASES = [
    ("A_lowconc", [1,2,4], [64,1024,2048,4096,8192,16384,32768,65536,131072,258000]),
    ("B_highconc_lowctx", [8,16,32], [64,1024,2048,4096,8192]),
    ("C_highconc_middctx", [8,16,32], [16384,32768]),
    ("D_conc8_highctx", [8], [65536,131072]),
]

class _F:
    def __getattr__(self, n): return lambda *a, **k: self

runner = BenchmarkRunner(placeholder=_F(), progress_bar=_F(), status_text=_F(),
    api_base_url="http://localhost:10814/v1", model_id="Kimi-K2.7-Code",
    tokenizer_option="API (usage field)", csv_filename="/tmp/x.csv", api_key="EMPTY",
    log_placeholder=_F(), provider="OpenAI Compatible", output_placeholder=_F(),
    warehouse_context={}, ui_state=NullStateBridge(),
    render_progress=lambda **k: None, render_log=lambda **k: None)
tok = runner._get_tokenizer()
provider = OpenAIProvider("http://localhost:10814/v1", "EMPTY", "Kimi-K2.7-Code")

def gpu_count():
    try:
        out = subprocess.check_output(["nvidia-smi","-L"], text=True, stderr=subprocess.DEVNULL)
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except: return -1

async def one_req(i, prompt):
    res = await provider.get_completion(None, i, prompt=prompt, max_tokens=MAX_TOKENS)
    if res.get("error"): return {"idx": i, "error": res["error"]}
    ts = res.get("token_timestamps") or []
    n = len(ts)
    itl = [round(ts[j+1]-ts[j], 4) for j in range(n-1)]
    def w(s, e):
        e = min(e, n-1)
        if s >= e or s >= n: return None
        dt = ts[e]-ts[s]
        return round((e-s)/dt, 1) if dt > 0 else None
    # 收敛点:ITL 降到 1.2× 稳态
    steady_itl = sorted(itl[-50:])[len(itl[-50:])//2] if len(itl)>=50 else (sorted(itl)[len(itl)//2] if itl else 1)
    converge = 0
    for idx, v in enumerate(itl):
        if v <= steady_itl * 1.2:
            converge = idx; break
    return {"idx": i, "n_tokens": n,
            "aggregate_tps": round(n/(ts[-1]-ts[0]), 1) if n > 1 else 0,
            "tps_0_100": w(0, 100), "tps_0_50": w(0, 50),
            "tps_100_500": w(100, 500), "tps_500_2000": w(500, 2000),
            "tps_2000_plus": w(2000, n-1),
            "steady_state_tps": w(min(500, n-2), n-1),
            "steady_itl_ms": round(steady_itl*1000, 1) if itl else None,
            "peak_itl_ms": round(max(itl[:min(50,n)])*1000, 1) if itl else None,
            "converge_token": converge,
            "itl_json": json.dumps(itl[:300])}

async def main():
    all_rows = []
    for tag, concs, ctxs in PHASES:
        n = gpu_count()
        print(f"\n[phase {tag}] GPU={n} concs={concs} ctxs={ctxs}", flush=True)
        if n != 8:
            print(f"[ABORT] GPU={n}≠8", flush=True); break
        for conc in concs:
            for ctx in ctxs:
                # 258000 时 max_tokens 受限(262144-258000=4144, 4096 ok)
                actual_max = min(MAX_TOKENS, 262144 - ctx) if ctx > 250000 else MAX_TOKENS
                if actual_max < 100:
                    print(f"  conc={conc} ctx={ctx}: max_tokens too small ({actual_max}), skip", flush=True)
                    continue
                print(f"  conc={conc} ctx={ctx} max_tokens={actual_max}...", flush=True, end="")
                prompts = [runner._calibrate_prompt(ctx, "", tok) for _ in range(conc)]
                t0 = time.monotonic()
                results = await asyncio.gather(*[one_req(i, p) for i, p in enumerate(prompts)])
                elapsed = time.monotonic() - t0
                ok = [x for x in results if not x.get("error")]
                for x in ok:
                    x["concurrency"] = conc; x["context_length_target"] = ctx
                    x["max_tokens"] = actual_max
                    all_rows.append(x)
                if ok:
                    agg = sorted([x["aggregate_tps"] for x in ok])[len(ok)//2]
                    steady = ok[0].get("steady_state_tps", "?")
                    first100 = ok[0].get("tps_0_100", "?")
                    conv = ok[0].get("converge_token", "?")
                    print(f" {len(ok)}/{conc} ok {elapsed:.0f}s | agg={agg} steady={steady} 前100={first100} 收敛={conv}", flush=True)
                else:
                    print(f" 0/{conc} ok", flush=True)
                # 保存增量(防止中途挂掉丢数据)
                if all_rows:
                    pd.DataFrame(all_rows).to_csv("raw_data/decode_steady_full.csv", index=False, encoding="utf-8-sig")
        n2 = gpu_count()
        print(f"[phase {tag} done] GPU={n2}", flush=True)
        if n2 != 8:
            print(f"[WARN] 掉卡({n2}),后续取消", flush=True); break
    print(f"\n[ALL DONE] {len(all_rows)} 行 → raw_data/decode_steady_full.csv", flush=True)

if __name__ == "__main__":
    asyncio.run(main())

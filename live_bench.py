"""通用长上下文并发基准 — 针对 OpenAI 兼容推理端点(vLLM / SGLang / 等)。

设计目标:不硬编码任何模型名、路径或引擎指纹。所有运行时事实来自
  1. 对 /v1/models 的探测(拿 served-model-name / max_model_len / 模型 root);
  2. CLI 覆盖(--api-base / --model-id / --tokenizer / --max-tokens / ...);
  3. 可选的引擎指纹 JSON(--engine-json 或 --kv KEY=VAL ...),仅用于结果标注,不阻塞测试。

测量精度对齐 llm-test 项目官方 BenchmarkRunner:
  - 共发请求用 asyncio.Barrier 对齐,真正同时发出;
  - 共享 httpx.AsyncClient(max_connections=2048),避免每请求各建连接;
  - 单请求指标用 core.benchmark.metrics.calculate_request_metrics
    (TTFT / TPS / TPOT / TPOT P95 / TPOT P99 / generation_time);
  - 批次级聚合吞吐(system_output_throughput / system_input_throughput /
    system_throughput / rps),按 prefill/decode 阶段拆分,剔除 cache_hit;
  - latency_offset 校准(本地端点可设 0);
  - cache_hit_tokens 从 usage_info.prompt_tokens_details.cached_tokens 提取,
    计算 uncached prefill 速度(prefix caching 开启时避免高估)。

保留自原 GLM-5.2 测试脚本并经多轮验证的辅助逻辑:
  - 精确 token 数的 prompt 构建(迭代逼近,20 次微调);
  - ITL 统计(peak/steady/converge);
  - ResourceMonitor 采样(GPU util/vram/power/temp + CPU/内存);
  - GPU 掉卡防护 + 温度告警;
  - 每 cell 后增量落盘 CSV。

用法示例:
  # 先探测当前在跑的端点
  python live_bench.py --api-base http://127.0.0.1:10814/v1 --probe

  # 自动探测模型,跑默认矩阵(按 KV 预算自适应)
  python live_bench.py --api-base http://127.0.0.1:10814/v1 \
      --kv-budget 1027072 --max-conc 16 \
      --engine-json /home/ai/llm-test/engines/glm52_nvfp4_vllm.json

  # 完全自定义矩阵
  python live_bench.py --api-base http://127.0.0.1:10814/v1 \
      --conc 1,2,4,8 --ctx 1024,8192,65536 --max-tokens 512 --rounds 2 \
      --out raw_data/mybench.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd
from transformers import AutoTokenizer

from core.benchmark.metrics import calculate_request_metrics
from core.engine_capture import capture_engine_config, find_vllm_container
from core.engine_metrics import EngineMetricsPoller
from core.hardware_fingerprint import capture_hardware_fingerprint
from core.model_spec import from_local_config, resolve_spec
from core.providers.openai import OpenAIProvider
from core.resource_monitor import ResourceMonitor
from core.system_info import capture_system_info, get_library_versions

# ──────────────────────────────────────────────────────────────────────────
# prompt 词库(中英混合,保证 tokenizer 不会过度聚合,长上下文填充稳定)
# ──────────────────────────────────────────────────────────────────────────
_WORD_POOL = ["system", "network", "kernel", "memory", "buffer", "cache", "latency", "throughput", "token", "context", "vector", "matrix", "tensor", "attention", "expert", "router", "sparse", "dense", "quantization", "inference", "prefill", "decode", "batch", "concurrency", "latency", "bandwidth", "gigabyte", "flops", "utilization", "queue", "schedule", "shard", "tensor", "parallel", "pipeline", "datacenter", "consistency", "byzantine", "erasure", "replication", "consensus", "checkpoint", "gradient", "optimizer", "attention", "fusion", "墨子", "兼爱", "非攻", "逻辑", "三段论", "诸子", "先秦", "因果", "分布式", "一致性", "容错", "存储", "缓存", "调度", "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "request", "response", "timeout", "retry", "backoff", "idempotent", "stream", "fragment", "aggregate"]


# ──────────────────────────────────────────────────────────────────────────
# 端点探测
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class EndpointInfo:
    model_id: str | None = None
    max_model_len: int | None = None
    model_root: str | None = None
    raw: dict = field(default_factory=dict)


def probe_endpoint(
    api_base: str, model_id_override: str | None, timeout: float = 15.0
) -> EndpointInfo:
    """探测 /v1/models,返回 served-model-name / max_model_len / 模型 root。"""
    info = EndpointInfo()
    url = api_base.rstrip("/") + "/models"
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[probe] 无法访问 {url}: {type(e).__name__}: {e}", file=sys.stderr)
        return info
    info.raw = data if isinstance(data, dict) else {}
    models = data.get("data") if isinstance(data, dict) else None
    if not models:
        return info
    chosen = None
    if model_id_override:
        chosen = next((m for m in models if m.get("id") == model_id_override), None)
    if chosen is None and len(models) == 1:
        chosen = models[0]
    if chosen is None and models:
        chosen = models[0]
    if not chosen:
        return info
    info.model_id = chosen.get("id")
    info.max_model_len = chosen.get("max_model_len")
    info.model_root = chosen.get("root")
    return info


# ──────────────────────────────────────────────────────────────────────────
# 引擎指纹(仅标注,不参与逻辑)
# ──────────────────────────────────────────────────────────────────────────
def parse_kv_pairs(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" in p:
            k, v = p.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def coerce(val: str) -> Any:
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    return val


# ──────────────────────────────────────────────────────────────────────────
# 矩阵自动生成(按 KV 预算自适应,保证 conc*ctx + conc*max_tokens ≤ kv_budget)
# ──────────────────────────────────────────────────────────────────────────
def build_phases(
    kv_budget: int,
    max_tokens: int,
    max_conc: int,
    max_model_len: int | None,
) -> list[tuple[str, list[int], list[int]]]:
    """生成 (tag, concs, ctxs) 列表。每个 cell 满足 conc*ctx + conc*max_tokens ≤ kv_budget。"""
    ladder = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 196608, 262144, 393216, 524288]
    if max_model_len:
        ladder = [c for c in ladder if c <= max_model_len]
        if not ladder:
            ladder = [max_model_len]
    phases: list[tuple[str, list[int], list[int]]] = []
    concs = [c for c in [1, 2, 4, 8, 16, 32] if c <= max_conc]
    for conc in concs:
        room = kv_budget - conc * max_tokens
        if room <= 0:
            continue
        max_ctx = room // conc
        ctxs = [c for c in ladder if c <= max_ctx]
        if not ctxs:
            continue
        if conc >= 8:
            ctxs = [c for c in ctxs if c <= 32768]
        if conc >= 16:
            ctxs = [c for c in ctxs if c <= 16384]
        if conc >= 32:
            ctxs = [c for c in ctxs if c <= 8192]
        if ctxs:
            phases.append((f"conc{conc}", [conc], ctxs))
    return phases


# ──────────────────────────────────────────────────────────────────────────
# prompt 构建(精确 token 数,迭代逼近 — 对齐 BenchmarkRunner._calibrate_prompt)
# ──────────────────────────────────────────────────────────────────────────
def make_prompt_builder(tok):
    """返回 build_prompt(target_tokens, seed) -> str,精确到 target 个 token。"""
    filler = "The quick brown fox jumps over the lazy dog. "

    def _encode(text: str) -> list[int]:
        try:
            if hasattr(tok, "encode_plus"):
                return tok.encode(text, add_special_tokens=False)
            return tok.encode(text)
        except Exception:
            return []

    def build_prompt(target: int, seed: int) -> str:
        rng = random.Random(seed)
        seed_line = f"[doc-{seed}] 以下是编号 {seed} 的测试文档材料,请通读全文后再作答:\n"
        inst = "\n\n阅读以上全部材料,然后用中文写一段不超过150字的摘要,概括其主题与关键信息。"
        overhead = len(_encode(seed_line + inst))
        body_target = max(64, target - overhead)
        # 粗估:token ≈ word*1.3,逆向多 25% 余量
        n_words = int(body_target / 1.3 * 1.25) + 16
        words = [rng.choice(_WORD_POOL) for _ in range(n_words)]
        body_text = " ".join(words)
        # 迭代逼近(对齐官方 _calibrate_prompt 的 20 次微调)
        for _ in range(20):
            cur = len(_encode(body_text + seed_line + inst))
            diff = cur - target
            if diff == 0:
                break
            if diff > 0:
                body_text = body_text[max(1, int(diff * 2)) :]
            else:
                body_text = (
                    "".join(random.choices(filler, k=max(1, int(abs(diff) * 3)))) + body_text
                )
        return seed_line + body_text + inst

    return build_prompt


# ──────────────────────────────────────────────────────────────────────────
# ITL 统计(peak/steady/converge — 原脚本保留,补充 TPOT 分位数走官方函数)
# ──────────────────────────────────────────────────────────────────────────
def itl_stats(ts: list[float]) -> dict[str, Any]:
    n = len(ts)
    if n < 2:
        return {"n_tokens": n, "peak_itl_ms": None, "steady_itl_ms": None, "converge_token": None}
    itl = [ts[j + 1] - ts[j] for j in range(n - 1)]
    tail = itl[-50:] if n >= 50 else itl
    s_itl = sorted(tail)[len(tail) // 2]
    peak = max(itl[: min(50, n - 1)]) if itl else 0.0
    conv = 0
    for i, v in enumerate(itl):
        if s_itl > 0 and v <= s_itl * 1.2:
            conv = i
            break
    return {
        "n_tokens": n,
        "peak_itl_ms": round(peak * 1000, 1),
        "steady_itl_ms": round(s_itl * 1000, 1),
        "converge_token": conv,
    }


# ──────────────────────────────────────────────────────────────────────────
# cache_hit_tokens 提取(对齐 BenchmarkRunner._get_cache_hit_tokens)
# ──────────────────────────────────────────────────────────────────────────
def extract_cache_hit(usage_info: dict | None) -> int:
    if not usage_info:
        return 0
    details = usage_info.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens"):
        return details.get("cached_tokens")
    for key in ("cache_hit_tokens", "prompt_cache_hit_tokens", "disk_cache_hit_tokens"):
        if usage_info.get(key):
            return usage_info[key]
    if usage_info.get("cache_read_input_tokens"):
        return usage_info.get("cache_read_input_tokens")
    return 0


# ──────────────────────────────────────────────────────────────────────────
# GPU 防护
# ──────────────────────────────────────────────────────────────────────────
def gpu_count() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
        return len([l for l in out.splitlines() if l.startswith("GPU ")])
    except Exception:
        return -1


def gpu_temp_max() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=8,
        )
        return max(int(x) for x in out.split())
    except Exception:
        return -1


# ──────────────────────────────────────────────────────────────────────────
# 单次请求(用共享 client + barrier 对齐;指标走 calculate_request_metrics)
# ──────────────────────────────────────────────────────────────────────────
async def one_req(
    provider,
    client,
    barrier,
    idx,
    prompt,
    max_tokens,
    req_timeout,
    latency_offset,
    skip_first_token,
    reasoning_effort=None,
):
    kwargs: dict[str, Any] = {"request_timeout": req_timeout, "_barrier": barrier}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    res = await provider.get_completion(client, idx, prompt=prompt, max_tokens=max_tokens, **kwargs)
    if res.get("error"):
        return {"idx": idx, "error": res["error"]}

    ts = res.get("token_timestamps") or []
    st = itl_stats(ts)
    u = res.get("usage_info") or {}
    usage = u if isinstance(u, dict) else {}
    start, first, end = res.get("start_time"), res.get("first_token_time"), res.get("end_time")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens") or st["n_tokens"]
    cache_hit = extract_cache_hit(usage)

    # 官方指标:TTFT / TPS / TPOT / P95 / P99 / generation_time
    m = calculate_request_metrics(
        start or 0.0,
        first,
        end or 0.0,
        completion_tokens,
        latency_offset=latency_offset,
        token_timestamps=ts,
        skip_first_token=skip_first_token,
    )

    total = round((end - start) - latency_offset, 2) if (end and start) else None
    decode_span = m.generation_time  # 已减 latency_offset,已处理 skip_first_token
    # 单请求 decode_tps:用官方 tps(skip_first_token 一致)
    decode_tps = round(m.tps, 1) if m.tps else None
    # 单请求 prefill_tps:用未缓存 token / TTFT
    uncached = (prompt_tokens or 0) - cache_hit
    prefill_tps = round(uncached / m.ttft, 1) if (m.ttft and uncached > 0) else None
    # 聚合用原始时间(不减 offset,批次级自己减)
    return {
        "idx": idx,
        "ttft": round(m.ttft, 3),
        "tps": decode_tps,
        "tpot_ms": round(m.tpot * 1000, 2) if m.tpot else None,
        "tpot_p95_ms": round(m.tpot_p95 * 1000, 2) if m.tpot_p95 else None,
        "tpot_p99_ms": round(m.tpot_p99 * 1000, 2) if m.tpot_p99 else None,
        "total_time": total,
        "start_time": start,
        "first_token_time": first,
        "end_time": end,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "cache_hit_tokens": cache_hit,
        "prefill_tps": prefill_tps,
        "aggregate_tps": (
            round(completion_tokens / total, 1)
            if (total and completion_tokens and total > 0)
            else None
        ),
        "finish_reason": res.get("finish_reason"),
        "peak_itl_ms": st["peak_itl_ms"],
        "steady_itl_ms": st["steady_itl_ms"],
        "converge_token": st["converge_token"],
        "_raw_usage": usage,  # 批次聚合用
    }


# ──────────────────────────────────────────────────────────────────────────
# 单 cell(共享 client + barrier;计算批次级聚合吞吐)
# ──────────────────────────────────────────────────────────────────────────
async def run_cell(
    provider,
    build_prompt,
    conc,
    ctx,
    rounds,
    max_tokens,
    req_timeout,
    latency_offset,
    skip_first_token,
    reasoning_effort,
    metrics_url=None,
    metrics_poll_interval=5.0,
):
    mon = ResourceMonitor(interval=1.0)
    mon.start()
    # 引擎 /metrics 轮询(KV 占用/队列/抢占/TTFT/TPOT/推测接受率)
    poller = (
        EngineMetricsPoller(metrics_url, interval=metrics_poll_interval) if metrics_url else None
    )
    if poller:
        poller.start()
    all_rows = []
    t0 = time.monotonic()
    # 共享 client(对齐 _run_concurrency_batch):避免每请求各建连接
    client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(
            limits=httpx.Limits(max_connections=2048, max_keepalive_connections=256),
        ),
        timeout=req_timeout,
    )
    try:
        for r in range(rounds):
            prompts = [
                build_prompt(ctx, seed=ctx * 1_000_000 + r * 1000 + i + 1) for i in range(conc)
            ]
            # barrier 让所有请求在 HTTP 发出前对齐
            barrier = asyncio.Barrier(conc) if conc > 1 else None
            results = await asyncio.gather(
                *[
                    one_req(
                        provider,
                        client,
                        barrier,
                        i,
                        p,
                        max_tokens,
                        req_timeout,
                        latency_offset,
                        skip_first_token,
                        reasoning_effort,
                    )
                    for i, p in enumerate(prompts)
                ]
            )
            # 批次聚合吞吐(对齐 _run_concurrency_batch 的 system_* 指标)
            ok = [x for x in results if not x.get("error")]
            batch_agg = _batch_aggregate(ok, latency_offset)
            for x in results:
                x["concurrency"] = conc
                x["context_length_target"] = ctx
                x["round"] = r
                x.update(batch_agg)
            all_rows.extend(results)
    finally:
        await client.aclose()
        mon_summary = mon.stop()
        engine_summary = poller.stop() if poller else {}
    return all_rows, mon_summary, engine_summary, time.monotonic() - t0


def _batch_aggregate(ok_rows: list[dict], latency_offset: float) -> dict[str, Any]:
    """批次级聚合吞吐(对齐 BenchmarkRunner._run_concurrency_batch)。"""
    if not ok_rows:
        return {
            "sys_output_throughput": None,
            "sys_input_throughput_uncached": None,
            "sys_throughput": None,
            "rps": None,
        }
    total_out = sum(x.get("completion_tokens") or 0 for x in ok_rows)
    total_in = sum(x.get("prompt_tokens") or 0 for x in ok_rows)
    total_cache = sum(x.get("cache_hit_tokens") or 0 for x in ok_rows)
    n_ok = len(ok_rows)
    starts = [x["start_time"] for x in ok_rows if x.get("start_time")]
    ends = [x["end_time"] for x in ok_rows if x.get("end_time")]
    firsts = [x["first_token_time"] for x in ok_rows if x.get("first_token_time")]
    if not starts or not ends:
        return {
            "sys_output_throughput": None,
            "sys_input_throughput_uncached": None,
            "sys_throughput": None,
            "rps": None,
        }
    min_start = min(starts)
    max_end = max(ends)
    batch_dur = max(0.001, (max_end - min_start) - latency_offset)
    # decode 阶段:最早 first_token → 最晚 end
    if firsts:
        decode_dur = max(0.001, (max_end - min(firsts)) - latency_offset)
    else:
        decode_dur = batch_dur
    # prefill 阶段:min_start → 最晚 first_token
    if firsts:
        prefill_dur = max(0.001, (max(firsts) - min_start) - latency_offset)
    else:
        prefill_dur = batch_dur
    uncached_in = max(0, total_in - total_cache)
    return {
        "sys_output_throughput": round(total_out / decode_dur, 1) if decode_dur else None,
        "sys_input_throughput_uncached": (
            round(uncached_in / prefill_dur, 1) if prefill_dur else None
        ),
        "sys_throughput": round((total_in + total_out) / batch_dur, 1) if batch_dur else None,
        "rps": round(n_ok / batch_dur, 2) if batch_dur else None,
    }


def fmt_cell(rows, elapsed):
    ok = [x for x in rows if not x.get("error")]
    if not ok:
        return f"0/{len(rows)} ok ({elapsed:.0f}s)"
    ttfts = sorted(x["ttft"] for x in ok if x.get("ttft") is not None)
    decs = sorted(x["tps"] for x in ok if x.get("tps") is not None)
    prefs = sorted(x["prefill_tps"] for x in ok if x.get("prefill_tps") is not None)
    agg = ok[0].get("sys_output_throughput")

    def med(lst):
        return lst[len(lst) // 2] if lst else None

    spec_rate = ok[0].get("spec_acceptance_rate")
    spec_str = f" spec_acc={spec_rate}" if spec_rate is not None else ""

    return (
        f"{len(ok)}/{len(rows)} ok {elapsed:.0f}s | ttft={med(ttfts)}s "
        f"prefill={med(prefs)}tps decode={med(decs)}tps agg_out={agg}tps{spec_str}"
    )


def rounds_for(ctx, r_short, r_med, r_long):
    if ctx <= 8192:
        return r_short
    if ctx <= 65536:
        return r_med
    return r_long


# ──────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="通用长上下文并发基准(OpenAI 兼容端点)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--api-base", default="http://127.0.0.1:10814/v1", help="推理端点 base URL")
    p.add_argument("--model-id", default=None, help="served-model-name;不填则探测")
    p.add_argument("--tokenizer", default=None, help="tokenizer 路径;不填则用模型 root 或 model-id")
    p.add_argument("--api-key", default="EMPTY", help="API key(本地端点通常 EMPTY)")
    p.add_argument("--max-tokens", type=int, default=512, help="每个请求的 completion 上限")
    p.add_argument(
        "--reasoning-effort",
        default=None,
        help="透传给 provider 的 reasoning_effort(min/low/medium/high/max)",
    )
    # 矩阵
    p.add_argument(
        "--kv-budget", type=int, default=None, help="KV 池 token 数;用于自动跳过超预算 cell"
    )
    p.add_argument("--max-conc", type=int, default=16, help="最大并发级别(自动矩阵的上界)")
    p.add_argument("--conc", default=None, help="显式并发列表,逗号分隔;覆盖自动矩阵")
    p.add_argument("--ctx", default=None, help="显式 ctx 列表,逗号分隔;对所有 conc 复用")
    p.add_argument("--rounds-short", type=int, default=3, help="ctx<=8192 的轮数")
    p.add_argument("--rounds-med", type=int, default=2, help="8192<ctx<=65536 的轮数")
    p.add_argument("--rounds-long", type=int, default=1, help="ctx>65536 的轮数")
    # 超时
    p.add_argument(
        "--req-timeout-short", type=float, default=900.0, help="ctx<128k 的单请求超时(秒)"
    )
    p.add_argument(
        "--req-timeout-long", type=float, default=1800.0, help="ctx>=128k 的单请求超时(秒)"
    )
    # 测量精度(对齐 BenchmarkRunner)
    p.add_argument(
        "--latency-offset",
        type=float,
        default=0.0,
        help="客户端测量偏差校准(秒);本地端点设 0,远程端点可空跑测得",
    )
    p.add_argument(
        "--skip-first-token",
        action="store_true",
        default=True,
        help="TPS/TPOT 跳过首 token(prefill→decode 间隙不扭曲生成速度);默认开",
    )
    p.add_argument(
        "--no-skip-first-token",
        dest="skip_first_token",
        action="store_false",
        help="不跳首 token(与官方默认一致时用)",
    )
    # GPU 防护
    p.add_argument("--expected-gpu", type=int, default=8, help="预期 GPU 数;掉卡则中止(0=不检查)")
    p.add_argument("--temp-warn", type=int, default=88, help="温度告警阈值(℃)")
    # 引擎 /metrics 轮询
    p.add_argument(
        "--metrics-url",
        default=None,
        help="引擎 /metrics 端点 URL;不填则不轮询引擎指标。可设为 'auto' 从 api-base 推导",
    )
    p.add_argument(
        "--metrics-poll-interval",
        type=float,
        default=5.0,
        help="引擎 /metrics 轮询间隔(秒),越大引擎日志越安静;默认 5.0",
    )
    # 指纹
    p.add_argument("--engine-json", default=None, help="引擎指纹 JSON 文件路径(仅标注)")
    p.add_argument(
        "--kv", action="append", default=[], help="引擎指纹键值对 KEY=VAL,可多次(仅标注)"
    )
    p.add_argument("--container", default=None, help="Docker 容器名;不填则自动探测(含 host 网络)")
    p.add_argument(
        "--no-auto-capture",
        action="store_true",
        help="跳过硬件/引擎/模型自动采集(仅用 --kv/--engine-json)",
    )
    # 输出
    p.add_argument("--out", default=None, help="输出 CSV 路径;默认 raw_data/live_bench_<ts>.csv")
    p.add_argument("--probe", action="store_true", help="只探测端点并打印,不跑测试")
    return p.parse_args(argv)


def resolve_phases(args, kv_budget, max_model_len):
    """决定最终 (tag, concs, ctxs) 列表。"""
    if args.conc or args.ctx:
        concs = [int(x) for x in args.conc.split(",")] if args.conc else [1]
        ctxs = [int(x) for x in args.ctx.split(",")] if args.ctx else [8192]
        return [(f"conc{c}", [c], ctxs) for c in concs]
    if not kv_budget:
        kv_budget = max_model_len or 131072
        print(f"[warn] 未给 --kv-budget,回退到 {kv_budget} 自动生成矩阵", file=sys.stderr)
    return build_phases(kv_budget, args.max_tokens, args.max_conc, max_model_len)


def _print_fingerprint_summary(cfg: dict[str, Any]) -> None:
    """打印结构化指纹摘要(探测/调试用)。"""
    env = cfg.get("environment") or {}
    # 硬件
    hw = env.get("hardware") or {}
    if hw:
        cpu = hw.get("cpu") or {}
        mem = hw.get("memory") or {}
        gpus = hw.get("gpus") or []
        cuda = hw.get("cuda") or {}
        print(
            f"[hw] CPU={cpu.get('model_name','?')} {cpu.get('sockets')}s×{cpu.get('cores_per_socket')}c "
            f"NUMA={cpu.get('numa_nodes')} | MEM={mem.get('total_gb','?')}GB {mem.get('type','')} "
            f"{mem.get('speed_mt_s','')}MT/s ECC={mem.get('ecc','?')}",
            flush=True,
        )
        if gpus:
            g = gpus[0]
            print(
                f"[hw] GPU={g.get('name','?')} ×{len(gpus)} {g.get('vram_gb','?')}GB "
                f"{g.get('nominal_bandwidth_gbps','?')}GB/s PCIeGen{g.get('pcie_gen','?')}x{g.get('pcie_width','?')} "
                f"| CUDA={cuda.get('cuda_version','?')} driver={cuda.get('driver','?')} "
                f"| machine_id={hw.get('machine_id','?')}",
                flush=True,
            )
    # 引擎
    ec = env.get("engine_config") or {}
    if ec:
        eng = ec.get("engine") or ec.get("adapter") or "?"
        img = ec.get("image") or ""
        sched = ec.get("schedule") or {}
        par = ec.get("parallel_strategy") or {}
        rt = ec.get("runtime") or {}
        print(
            f"[engine] {eng} image={img.split('/')[-1] if img else '?'} "
            f"container={env.get('container','?')}",
            flush=True,
        )
        if par:
            print(f"[engine] parallel={par}", flush=True)
        if sched:
            print(f"[engine] schedule={sched}", flush=True)
        if rt:
            print(f"[engine] runtime={rt}", flush=True)
    # serving config(归一化后)
    sc = env.get("serving_config") or {}
    if sc:
        print(
            f"[serving] quant={sc.get('serving_quant','?')} kv={sc.get('kv_cache_dtype','?')} "
            f"attn={sc.get('attention_backend','?')} moe={sc.get('moe_backend','?')} "
            f"cuda={sc.get('cuda_version','?')} env_flags={len(sc.get('env_flags',{}))}",
            flush=True,
        )
    # 模型
    ms = env.get("model_spec") or {}
    if ms:
        print(
            f"[model] {ms.get('name','?')} arch={ms.get('architecture','?')} "
            f"layers={ms.get('num_layers','?')} experts={ms.get('num_experts','?')} "
            f"attn={ms.get('attention_type','?')} dtype={ms.get('weight_dtype','?')} "
            f"quant={ms.get('quant_method','?')} kv={ms.get('kv_dtype','?')}",
            flush=True,
        )
    # 用户显式覆盖
    user_kv = {
        k: v
        for k, v in cfg.items()
        if k
        not in (
            "environment",
            "api_base",
            "model_id_probed",
            "max_model_len_probed",
            "model_root_probed",
            "bench_cli",
        )
    }
    if user_kv:
        print(
            f"[override] {json.dumps(user_kv, ensure_ascii=False, default=str)[:200]}", flush=True
        )


def _auto_capture_environment(args, info: EndpointInfo) -> dict[str, Any]:
    """自动采集硬件指纹 + 引擎配置 + 模型架构 + 系统信息。每块独立 try/except,不阻塞测试。"""
    env: dict[str, Any] = {}

    # 1. 硬件指纹(CPU 拓扑/内存/GPU/CUDA/驱动/machine_id)
    try:
        env["hardware"] = capture_hardware_fingerprint()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] hardware_fingerprint 采集失败: {e}", file=sys.stderr)
        env["hardware"] = {}

    # 2. 系统信息(Python/OS/git hash/库版本)
    try:
        env["system"] = {
            "python_version": capture_system_info().get("python_version"),
            "os_name": capture_system_info().get("os_name"),
            "os_version": capture_system_info().get("os_version"),
            "hostname": capture_system_info().get("hostname"),
            "git_hash": capture_system_info().get("git_hash"),
            "project_version": capture_system_info().get("project_version"),
        }
    except Exception:  # noqa: BLE001
        env["system"] = {}
    try:
        env["system"]["libraries"] = get_library_versions()
    except Exception:  # noqa: BLE001
        pass

    # 3. 引擎配置(docker inspect + 日志解析 + /v1/models + 引擎适配器)
    try:
        container = args.container
        if not container:
            container = find_vllm_container(args.api_base)
        if container:
            env["container"] = container
        ec = capture_engine_config(args.api_base, container_name=container)
        env["engine_config"] = ec
        # 归一化成 ServingConfig(填充 serving_quant/backends/cuda_version/env_flags 等)
        try:
            from core.serving_config import from_engine_capture

            env["serving_config"] = from_engine_capture(ec).to_dict()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        print(f"[warn] engine_capture 采集失败: {e}", file=sys.stderr)
        env["engine_config"] = {}

    # 4. 模型架构:注册表优先(权威 total/active params),from_local_config 补充结构字段
    try:
        model_root = info.model_root or ""
        base_spec = None
        # 4a. 注册表优先(有权威 total_params_b / active_params_b / family)
        if info.model_id:
            base_spec = resolve_spec(info.model_id)
            if base_spec:
                env["model_spec_source"] = "registry"
        # 4b. from_local_config 补充注册表没有的结构字段(quant/attention_type/moe_intermediate 等)
        local_spec = None
        if model_root:
            cfg_path = os.path.join(model_root, "config.json")
            if os.path.exists(cfg_path):
                local_spec = from_local_config(cfg_path)
        # 4c. 合并:注册表为底,local 覆盖空字段(不覆盖注册表的权威值)
        if base_spec and local_spec:
            merged = base_spec.to_dict()
            for k, v in local_spec.to_dict().items():
                if merged.get(k) in (None, "", [], {}) and v not in (None, "", [], {}):
                    merged[k] = v
            env["model_spec"] = merged
        elif base_spec:
            env["model_spec"] = base_spec.to_dict()
        elif local_spec:
            env["model_spec"] = local_spec.to_dict()
            env["model_spec_source"] = "local_config"
    except Exception as e:  # noqa: BLE001
        print(f"[warn] model_spec 采集失败: {e}", file=sys.stderr)

    return env


def build_engine_fingerprint(args, info: EndpointInfo) -> dict[str, Any]:
    """组装引擎指纹:--engine-json > --kv > 自动采集 > 探测事实。
    用户显式给的(--engine-json / --kv)优先级最高,覆盖自动采集。"""
    cfg: dict[str, Any] = {}

    # 1. 自动采集(硬件/引擎/模型/系统) — 最低优先级
    if not args.no_auto_capture:
        cfg["environment"] = _auto_capture_environment(args, info)

    # 2. --engine-json 文件(覆盖自动采集的同级字段)
    if args.engine_json and os.path.exists(args.engine_json):
        try:
            with open(args.engine_json, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 读取 --engine-json 失败: {e}", file=sys.stderr)

    # 3. --kv KEY=VAL(最高优先级,逐键覆盖)
    for k, v in parse_kv_pairs(args.kv).items():
        cfg[k] = coerce(v)

    # 4. 探测到的端点事实(不覆盖用户显式给的)
    cfg.setdefault("api_base", args.api_base)
    cfg.setdefault("model_id_probed", info.model_id)
    cfg.setdefault("max_model_len_probed", info.max_model_len)
    cfg.setdefault("model_root_probed", info.model_root)

    # 5. bench CLI 参数(测试条件)
    cfg["bench_cli"] = {
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "latency_offset": args.latency_offset,
        "skip_first_token": args.skip_first_token,
        "expected_gpu": args.expected_gpu,
    }
    return cfg


async def main_async(args):
    # 1) 探测端点
    print(f"[probe] GET {args.api_base.rstrip('/')}/models ...", flush=True)
    info = probe_endpoint(args.api_base, args.model_id)
    model_id = args.model_id or info.model_id
    if not model_id:
        print("[fatal] 探测不到 model_id,且未显式 --model-id", file=sys.stderr)
        sys.exit(2)
    print(
        f"[probe] model_id={model_id} max_model_len={info.max_model_len} root={info.model_root}",
        flush=True,
    )

    # 2) tokenizer
    tok_path = args.tokenizer or info.model_root or model_id
    print(f"[init] 加载 tokenizer: {tok_path}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    except Exception as e:  # noqa: BLE001
        print(f"[fatal] tokenizer 加载失败: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"[init] tokenizer 就绪 vocab={getattr(tok, 'vocab_size', '?')}", flush=True)

    # 3) 引擎指纹 + 环境采集
    engine_cfg = build_engine_fingerprint(args, info)
    _print_fingerprint_summary(engine_cfg)

    if args.probe:
        print("[probe] --probe 模式,不跑测试,退出。", flush=True)
        return

    # 4) 矩阵
    phases = resolve_phases(args, args.kv_budget, info.max_model_len)
    if not phases:
        print("[fatal] 矩阵为空(检查 --kv-budget / --max-conc / --ctx)", file=sys.stderr)
        sys.exit(2)
    total_cells = sum(len(c) for _, _, c in phases)
    kv_budget = args.kv_budget or (info.max_model_len or 0)
    print(
        f"[plan] {total_cells} cells, max_tokens={args.max_tokens}, "
        f"kv_budget={kv_budget}, max_conc={args.max_conc}, "
        f"latency_offset={args.latency_offset}, skip_first_token={args.skip_first_token}",
        flush=True,
    )

    # 5) 输出路径
    out_csv = args.out or f"raw_data/live_bench_{int(time.time())}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    # 6) provider + metrics URL
    provider = OpenAIProvider(args.api_base, args.api_key, model_id)
    build_prompt = make_prompt_builder(tok)
    # 解析 metrics URL:'auto' 从 api_base 推导(同 host:port + /metrics)
    metrics_url = args.metrics_url
    if metrics_url == "auto":
        from core.engine_metrics import default_metrics_url

        metrics_url = default_metrics_url(args.api_base)

    # 7) 跑矩阵
    all_rows: list[dict] = []
    for tag, concs, ctxs in phases:
        n = gpu_count()
        tmax = gpu_temp_max()
        print(
            f"\n[phase {tag}] GPU={n}(预期{args.expected_gpu}) temp_max={tmax}C concs={concs} ctxs={ctxs}",
            flush=True,
        )
        if args.expected_gpu and n != args.expected_gpu:
            print(f"[ABORT] GPU={n}≠{args.expected_gpu},停止(保住已采数据)", flush=True)
            break
        for conc in concs:
            for ctx in ctxs:
                if kv_budget:
                    need = conc * ctx + conc * args.max_tokens
                    if need > kv_budget:
                        print(
                            f"  conc={conc} ctx={ctx}: 超 KV 预算({need}>{kv_budget}),skip",
                            flush=True,
                        )
                        all_rows.append(
                            {
                                "concurrency": conc,
                                "context_length_target": ctx,
                                "error": f"over_kv_budget ({need}>{kv_budget})",
                                "round": 0,
                            }
                        )
                        continue
                if args.expected_gpu and gpu_count() != args.expected_gpu:
                    print("  [ABORT] 掉卡,停止", flush=True)
                    break
                rnd = rounds_for(ctx, args.rounds_short, args.rounds_med, args.rounds_long)
                req_timeout = args.req_timeout_long if ctx >= 128000 else args.req_timeout_short
                print(f"  conc={conc} ctx={ctx} rounds={rnd} ...", flush=True, end="")
                try:
                    rows, mon, eng_metrics, elapsed = await run_cell(
                        provider,
                        build_prompt,
                        conc,
                        ctx,
                        rnd,
                        args.max_tokens,
                        req_timeout,
                        args.latency_offset,
                        args.skip_first_token,
                        args.reasoning_effort,
                        metrics_url=args.metrics_url,
                        metrics_poll_interval=args.metrics_poll_interval,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"\n  [cell-err] {type(e).__name__}: {e}", flush=True)
                    rows = [
                        {
                            "concurrency": conc,
                            "context_length_target": ctx,
                            "error": f"{type(e).__name__}: {e}",
                            "round": 0,
                        }
                    ]
                    mon, eng_metrics, elapsed = {}, {}, 0.0
                peaks = (mon or {}).get("peaks", {}) or {}
                per_gpu = (mon or {}).get("per_gpu_peaks", []) or []
                gpu_static = (mon or {}).get("gpu_static_info", []) or []
                throttle_seen = (mon or {}).get("throttle_reasons_seen", []) or []
                eng_means = (eng_metrics or {}).get("engine_means", {}) or {}
                for x in rows:
                    x["elapsed_cell_s"] = round(elapsed, 1)
                    x["gpu_util_peak"] = peaks.get("gpu_util_percent")
                    x["gpu_vram_peak_gb"] = peaks.get("gpu_vram_gb")
                    x["gpu_power_peak_w"] = peaks.get("gpu_power_w")
                    x["gpu_temp_peak_c"] = peaks.get("gpu_temp_c")
                    x["cpu_peak_pct"] = peaks.get("cpu_percent")
                    x["system_memory_peak_gb"] = peaks.get("system_memory_gb")
                    x["spec_acceptance_rate"] = eng_means.get("spec_token_acceptance_rate")
                    x["prefix_cache_hit_rate"] = eng_means.get("gpu_prefix_cache_hit_rate")
                    x["resource_monitor_json"] = json.dumps(
                        {
                            "peaks": peaks,
                            "per_gpu_peaks": per_gpu,
                            "gpu_static_info": gpu_static,
                            "throttle_reasons_seen": throttle_seen,
                            "timeline_len": len((mon or {}).get("timeline", [])),
                        },
                        ensure_ascii=False,
                        default=str,
                    )
                    x["engine_metrics_json"] = (
                        json.dumps(eng_metrics, ensure_ascii=False, default=str)
                        if eng_metrics
                        else None
                    )
                    x["engine_config_json"] = json.dumps(
                        engine_cfg, ensure_ascii=False, default=str
                    )
                    x.pop("_raw_usage", None)  # 不写入 CSV
                all_rows.extend(rows)
                tmax = gpu_temp_max()
                print(f" {fmt_cell(rows, elapsed)} [GPU_temp_max={tmax}C]", flush=True)
                if tmax >= args.temp_warn:
                    print(f"  [WARN] GPU 最高 {tmax}C 接近节流线,持续关注", flush=True)
                pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
            else:
                continue
            break
        n2 = gpu_count()
        print(f"[phase {tag} done] GPU={n2} temp_max={gpu_temp_max()}C", flush=True)
        if args.expected_gpu and n2 != args.expected_gpu:
            print(f"[WARN] 掉卡({n2}),后续 phase 取消", flush=True)
            break

    print(f"\n[ALL DONE] {len(all_rows)} 行 → {out_csv}", flush=True)
    ok = [r for r in all_rows if not r.get("error")]
    print(f"成功 {len(ok)}/{len(all_rows)}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

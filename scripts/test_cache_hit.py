"""
用 ~30K tokens 长 prompt Test prefix caching TTFT 差异
"""

import json
import os
import time

import requests

ENDPOINTS = {
    name: url
    for name, url in [
        ("Server1", os.getenv("SERVER1_URL")),
        ("Server2", os.getenv("SERVER2_URL")),
    ]
    if url
}

# 构造 ~30K tokens  prompt（每段约 150 tokens，重复200次）
base_text = """人工智能（Artificial Intelligence，AI）isCalculate机科学一分支，它企图解智能实质，
并生产出一种新能以人类智能相似方式做出反应智能机器。该领域研究包括机器人、语言识别、
图像识别、自然语言Processand专家系统etc.。人工智能从诞生以来，理论and技术日益成熟，Apply领域也not断扩大。
深度学习作is机器学习一分支，in图像识别、语音识别and自然语言Processetc.领域取得突破性进展。
大型语言Model（LLM）如GPT、Claude、Qwenetc.展示强大文本Generateand理解能力。
Transformer架构提出标志着自然语言Process进入新时代。注意力机制允许Model关注输入序列innot同部分，
thus更好地捕捉长距离依赖关系。预训练加微调范式already成isNLP领域主流方法。
强化学习从人类反馈（RLHF）进一步提升ModelAlign性能。混合专家Model（MoE）via稀疏激活实现
更高效Calculate。推理优化技术如Quantize、剪枝and蒸馏使得大Model能够in更多设备on部署。
"""
long_prompt = base_text * 150  # ~150 * 150 ≈ 22500 tokens

out = open("cache_ttft_results_30k.txt", "w", encoding="utf-8")


def log(msg):
    print(msg, flush=True)
    out.write(msg + "\n")
    out.flush()


def measure_request(base_url, model_id, prompt, max_tokens=10):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    usage_info = None
    t_start = time.perf_counter()
    t_first_token = None

    with requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=300,
    ) as r:
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
        for line in r.iter_lines():
            if not line:
                continue
            l = line.decode("utf-8").strip()
            if l.startswith("data: "):
                d = l[6:].strip()
                if d == "[DONE]":
                    break
                try:
                    c = json.loads(d)
                    if c.get("usage"):
                        usage_info = c["usage"]
                    if c.get("choices") and len(c["choices"]) > 0:
                        delta = c["choices"][0].get("delta", {})
                        content = delta.get("content") or delta.get("reasoning_content") or ""
                        if content and t_first_token is None:
                            t_first_token = time.perf_counter()
                except:
                    pass
    t_end = time.perf_counter()
    return {
        "ttft": (t_first_token - t_start) if t_first_token else None,
        "total_time": t_end - t_start,
        "usage": usage_info,
    }


for name, base_url in ENDPOINTS.items():
    log(f"\n{'='*60}")
    log(f"引擎: {name} ({base_url})")
    log(f"{'='*60}")

    try:
        resp = requests.get(f"{base_url}/models", timeout=5)
        models = resp.json().get("data", [])
        model_id = models[0]["id"] if models else None
        if not model_id:
            log("noModel")
            continue
        log(f"Model: {model_id}")
    except Exception as e:
        log(f"Connection failed: {e}")
        continue

    log("\n--- 相同 Prompt 连续4次 (~30K tokens) ---")
    ttfts = []
    for i in range(1, 5):
        result = measure_request(base_url, model_id, long_prompt, max_tokens=10)
        if "error" in result:
            log(f"  #{i}: {result['error']}")
            break

        ttft = result["ttft"]
        total = result["total_time"]
        ttfts.append(ttft)
        prompt_tokens = result["usage"].get("prompt_tokens", "?") if result["usage"] else "?"

        cache_hit = 0
        if result["usage"]:
            ptd = result["usage"].get("prompt_tokens_details")
            if ptd and isinstance(ptd, dict):
                cache_hit = ptd.get("cached_tokens", 0)
            for key in ["cache_hit_tokens", "prompt_cache_hit_tokens"]:
                if result["usage"].get(key):
                    cache_hit = result["usage"][key]

        marker = ""
        if i > 1 and ttfts[0] and ttft and ttfts[0] > 0:
            speedup = ttfts[0] / ttft
            marker = f"  ← 相对#1 加速 {speedup:.2f}x" if speedup > 1.1 else f"  ← {speedup:.2f}x"

        log(
            f"  #{i}: TTFT={ttft:.4f}s | 总耗时={total:.2f}s | prompt={prompt_tokens} | cache_hit_usage={cache_hit}{marker}"
        )
        time.sleep(2)

    if len(ttfts) >= 3:
        log(
            f"\n  结论: #1 TTFT={ttfts[0]:.4f}s → 后续平均={sum(ttfts[1:])/len(ttfts[1:]):.4f}s, 加速比={ttfts[0]/(sum(ttfts[1:])/len(ttfts[1:])):.2f}x"
        )

log("\n✅ Test完成")
out.close()

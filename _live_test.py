"""完整矩阵 live 测试：容器 Kimi-K2.7-Code (vLLM)。
并发 [1,2,4,8,16,32] × 上下文 [64,1k,2k,4k,8k,16k,32k,64k,128k,260k]，max_tokens=2048。"""
import asyncio
import os

from core.benchmark_runner import BenchmarkRunner
from core.ui_bridge import NullStateBridge

os.makedirs("raw_data", exist_ok=True)


class _Fake:
    def __getattr__(self, name):
        return lambda *a, **kw: self


serving_config = {
    "engine": "vllm", "engine_version": "0.22.1", "tp_size": 8,
    "enable_prefix_caching": True, "gpu_memory_utilization": 0.94, "max_num_seqs": 16,
}
warehouse_context = {
    "serving_config": serving_config,
    "test_metadata": {"tester": "claude-live", "next_action": "完整并发×上下文矩阵"},
    "model_spec_override": {}, "engine_runtime": {}, "custom_sys_info": {},
}

runner = BenchmarkRunner(
    placeholder=_Fake(), progress_bar=_Fake(), status_text=_Fake(),
    api_base_url="http://localhost:10814/v1",
    model_id="Kimi-K2.7-Code",
    tokenizer_option="API (usage field)",
    csv_filename="raw_data/live_kimi_matrix.csv",
    api_key="EMPTY",
    log_placeholder=_Fake(),
    provider="OpenAI Compatible",
    output_placeholder=_Fake(),
    warehouse_context=warehouse_context,
    ui_state=NullStateBridge(),
    render_progress=lambda **kw: None,
    render_log=lambda **kw: None,
)

# 完整矩阵：10 上下文档 × 6 并发档，rounds=1
CONCURRENCIES = [1, 2, 4, 8, 16, 32]
CONTEXT_LENGTHS = [64, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 260000]

df = asyncio.run(runner.run_throughput_matrix_test(
    concurrencies=CONCURRENCIES,
    context_lengths=CONTEXT_LENGTHS,
    rounds=1,
    max_tokens=2048,
))
print(f"\n=== 矩阵完成 {len(df)} 行（{len(CONCURRENCIES)}并发 × {len(CONTEXT_LENGTHS)}上下文）===")
print(df[["concurrency", "context_length_target", "ttft", "tps", "decode_tokens", "error"]].to_string())

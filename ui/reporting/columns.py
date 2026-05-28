"""Column labels and tooltips shared by report renderers."""

TPOT_CHUNK_P95_LABEL = "TPOT_Chunk_P95 (ms)"
TPOT_CHUNK_P99_LABEL = "TPOT_Chunk_P99 (ms)"

COLUMN_RENAME_MAP = {
    "Best_TTFT": "Best_TTFT (s)",
    "Max_System_Output_Throughput": "Max_System_Output_Throughput (tokens/s)",
    "Max_System_Input_Throughput": "Max_System_Input_Throughput (tokens/s)",
    "Max_System_Throughput": "Max_System_Throughput (tokens/s)",
    "Max_RPS": "Max_RPS (req/s)",
    "TPOT_Mean": "TPOT_Mean (ms)",
    "TPOT_P95": TPOT_CHUNK_P95_LABEL,
    "TPOT_P99": TPOT_CHUNK_P99_LABEL,
    "Max_Prefill_Speed": "Max_Prefill_Speed (tokens/s)",
    "Max_TPS": "Max_TPS (tokens/s)",
    "Success_Rate": "Success_Rate (%)",
}

COLUMN_TOOLTIPS = {
    "Max_System_Output_Throughput": (
        "System Output Throughput (Decode Output Throughput): Token generation rate during "
        "the Decode phase only (Total Output Tokens / Decode Duration). Excludes Prefill phase."
    ),
    "Max_System_Input_Throughput": (
        "System Input Throughput (Prefill Input Throughput): Token processing rate during "
        "the Prefill phase only (Total Input Tokens / Prefill Duration). Excludes Decode phase."
    ),
    "Max_System_Throughput": (
        "Total System Throughput: Total tokens processed per unit time "
        "(Input + Output) / Total Duration."
    ),
    "Max_RPS": (
        "Requests Per Second: Number of requests system can complete per second "
        "(Total Requests / Total Duration)."
    ),
    "TPOT_Mean": "Average Time Per Output Token. Measures generation fluency.",
    "Max_Prefill_Speed": (
        "Max Prefill Speed: Rate at which the system processes input prompt tokens (Tokens/s)."
    ),
    "Best_TTFT": "Best Time To First Token (TTFT). Measures response agility.",
    "Success_Rate": "Request success rate.",
    "TPOT_P95": (
        "P95 stream chunk latency from streamed delta arrival gaps. This is close to token "
        "latency only when the provider emits one token per chunk; token-level granularity "
        "is not guaranteed."
    ),
    "TPOT_P99": (
        "P99 stream chunk latency from streamed delta arrival gaps. This is close to token "
        "latency only when the provider emits one token per chunk; token-level granularity "
        "is not guaranteed."
    ),
    TPOT_CHUNK_P95_LABEL: (
        "P95 stream chunk latency from streamed delta arrival gaps. This is close to token "
        "latency only when the provider emits one token per chunk; token-level granularity "
        "is not guaranteed."
    ),
    TPOT_CHUNK_P99_LABEL: (
        "P99 stream chunk latency from streamed delta arrival gaps. This is close to token "
        "latency only when the provider emits one token per chunk; token-level granularity "
        "is not guaranteed."
    ),
    "Max_TPS": "Max Single-stream Throughput: Generation rate of a single request (Decode Only).",
    "Actual_Tokens_Mean": "Average actual tokens generated.",
    "Best_Prefill_Speed": "Best Prefill Speed.",
    "Input Tokens": "Target input token count.",
    "Context Length": "Target context length.",
    "Concurrency": "Number of concurrent requests.",
}

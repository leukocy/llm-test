import base64
import csv

import pandas as pd
import streamlit as st


def initialize_csv(columns, filename):
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(columns)
    except OSError as e:
        st.error(f"no法Initialize CSV 文件: {e}")

def append_to_csv(result_dict, columns, filename):
    try:
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow({k: result_dict.get(k) for k in columns})
    except OSError as e:
        st.error(f"no法写入 CSV: {e}")

def get_table_download_link(df, filename, text):
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def get_log_download_link(log_list, filename, text):
    """Generates a download link for a list of log strings."""
    try:
        # 移除 Markdown 粗体标记，保留纯文本
        log_text = "\n".join(line.replace("**", "") for line in log_list)
        b64 = base64.b64encode(log_text.encode()).decode()
        return f'<a href="data:file/text;base64,{b64}" download="{filename}">{text}</a>'
    except Exception as e:
        st.error(f"GenerateLogunder载链接失败: {e}")
        return ""

def reorder_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder DataFrame columns to ensure a logical, intuitive order.
    Follows the natural data flow of a benchmark request:

    1. Identity         — 这一行is谁？(session_id, test_type)
    2. Test Config      — in什么条件underTest？(concurrency, round, targets...)
    3. Input (Prefill)  — 输入Process性能 (ttft, prefill_speed, prefill_tokens...)
    4. Output (Decode)  — 输出Generate性能 (tps, decode_tokens, decode_time, tpot...)
    5. System Throughput — 系统级Aggregate吞吐 (system_*_throughput, rps)
    6. Total Time       — 端到端总耗时
    7. Metadata         — 技术细节，放in最右侧
    """
    if df.empty:
        return df

    # 1. Identity — 标识列
    identity_cols = ['session_id', 'test_type']

    # 2. Test Config — Test条件/Configure
    config_cols = [
        'concurrency', 'round',
        'input_tokens_target', 'context_length_target',
        'cumulative_mode', 'timestamp',
    ]

    # 3. Input (Prefill) Metrics — 输入Process阶段
    #    ttft → prefill_speed → prefill_tokens → cache_hit_tokens
    #    (先看Latency，再看速度，再看 token 数)
    input_cols = [
        'ttft', 'prefill_speed',
        'prefill_tokens', 'cache_hit_tokens',
    ]

    # 4. Output (Decode) Metrics — 输出Generate阶段
    #    tps → decode_tokens → decode_time → tpot → tpot_p95 → tpot_p99
    #    (先看速度，再看 token 数and时间，最后看尾部Latency)
    output_cols = [
        'tps', 'decode_tokens', 'decode_time',
        'tpot', 'tpot_p95', 'tpot_p99',
    ]

    # 5. System Throughput — 系统级Aggregate指标
    system_cols = [
        'system_input_throughput', 'system_output_throughput',
        'system_throughput', 'system_total_throughput',
        'rps',
    ]

    # 6. Total Time — 端到端总耗时
    time_cols = ['total_time']

    # 7. Metadata — 技术细节 (放in最右侧)
    meta_cols = [
        'start_time', 'first_token_time', 'end_time',
        'token_calc_method',
        'api_prefill', 'api_decode',
        'effective_prefill_tokens', 'effective_decode_tokens', 'token_source',
        'cache_hit_source',
        'error',
    ]

    # Construct final order based on what exists in the dataframe
    all_defined = (identity_cols + config_cols + input_cols + output_cols
                   + system_cols + time_cols + meta_cols)

    final_order = []
    final_order.extend([c for c in identity_cols if c in df.columns])
    final_order.extend([c for c in config_cols if c in df.columns])
    final_order.extend([c for c in input_cols if c in df.columns])
    final_order.extend([c for c in output_cols if c in df.columns])
    final_order.extend([c for c in system_cols if c in df.columns])
    final_order.extend([c for c in time_cols if c in df.columns])

    # Add any remaining columns that weren't explicitly listed
    defined_set = set(all_defined)
    remaining_cols = [c for c in df.columns if c not in defined_set]
    final_order.extend(remaining_cols)

    # Add metadata at the end
    final_order.extend([c for c in meta_cols if c in df.columns])

    return df[final_order]

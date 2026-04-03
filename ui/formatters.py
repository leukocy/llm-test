"""
Result Display Formatting Module

Provides unified result table formatting, including:
- Column name renaming (with units)
- Number precision
- Column order adjustment
- Hidden column filtering
"""

import pandas as pd

# =====================================================================
# Unified Raw Data Display Configuration
# =====================================================================
# Column name mapping: raw column name → display name (with units)
COLUMN_DISPLAY_NAMES = {
    # Identifier columns
    'session_id':               'Session ID',
    'round':                    'Round',
    'concurrency':              'Concurrency',
    # Test parameter columns
    'context_length_target':    'Target Context (tokens)',
    'input_tokens_target':      'Target Input (tokens)',
    'cumulative_mode':          'Cumulative Mode',
    'timestamp':                'Timestamp',
    # Performance metric columns
    'ttft':                     'TTFT (s)',
    'prefill_speed':            'Prefill Speed (t/s)',
    'tps':                      'TPS (t/s)',
    'tpot':                     'TPOT (s)',
    'tpot_p95':                 'TPOT P95 (s)',
    'tpot_p99':                 'TPOT P99 (s)',
    'rps':                      'RPS (req/s)',
    # Token statistics columns
    'prefill_tokens':           'Prefill Tokens',
    'decode_tokens':            'Decode Tokens',
    'effective_prefill_tokens': 'Prefill Tokens (Effective)',
    'effective_decode_tokens':  'Decode Tokens (Effective)',
    'cache_hit_tokens':         'Cache Hit (tokens)',
    'cache_hit_source':         'Cache Source',
    # Throughput columns
    'system_throughput':        'System Throughput (t/s)',
    'system_input_throughput':  'Input Throughput (t/s)',
    'system_output_throughput': 'Output Throughput (t/s)',
    'system_total_throughput':  'Total Throughput (t/s)',
    # Meta information columns
    'token_source':             'Token Source',
    'token_calc_method':        'Token Calc Method',
    'error':                    'Error',
}

# Always hidden internal/debug columns (not shown in main data table)
HIDDEN_COLUMNS = {
    'start_time', 'end_time', 'first_token_time',
    'total_time', 'decode_time',
    'api_prefill', 'api_decode',
    'test_type',
}

# Display column order for each test type
# Columns not in this list but present in data → appended at end (except hidden columns)
TEST_TYPE_DISPLAY_ORDER = {
    'Concurrency Test': [
        'session_id', 'concurrency', 'round', 'input_tokens_target',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot', 'tpot_p95', 'tpot_p99',
        'system_throughput', 'system_input_throughput', 'system_output_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    'Prefill Stress Test': [
        'session_id', 'input_tokens_target',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_throughput', 'system_input_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    'Long Context Test': [
        'session_id', 'context_length_target', 'round',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_throughput', 'system_input_throughput', 'system_output_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    'Segmented Context Test': [
        'session_id', 'round', 'context_length_target', 'cumulative_mode',
        'effective_prefill_tokens', 'effective_decode_tokens', 'cache_hit_tokens', 'cache_hit_source',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_input_throughput', 'system_output_throughput',
        'token_source', 'token_calc_method', 'error',
    ],
    'Concurrency-Context Matrix Test': [
        'session_id', 'concurrency', 'context_length_target', 'round',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_input_throughput', 'system_output_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    'Stability Test': [
        'session_id', 'concurrency', 'timestamp', 'input_tokens_target',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_throughput', 'system_input_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    # Backward compatibility: keep Chinese keys as aliases
    '并发性能Test': [
        'session_id', 'concurrency', 'round', 'input_tokens_target',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot', 'tpot_p95', 'tpot_p99',
        'system_throughput', 'system_input_throughput', 'system_output_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    'Prefill 压力Test': [
        'session_id', 'input_tokens_target',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_throughput', 'system_input_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    '长onunder文Test': [
        'session_id', 'context_length_target', 'round',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_throughput', 'system_input_throughput', 'system_output_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    '分段onunder文Test': [
        'session_id', 'round', 'context_length_target', 'cumulative_mode',
        'effective_prefill_tokens', 'effective_decode_tokens', 'cache_hit_tokens', 'cache_hit_source',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_input_throughput', 'system_output_throughput',
        'token_source', 'token_calc_method', 'error',
    ],
    '并发-onunder文 综合Test': [
        'session_id', 'concurrency', 'context_length_target', 'round',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_input_throughput', 'system_output_throughput', 'rps',
        'token_calc_method', 'error',
    ],
    '稳定性Test': [
        'session_id', 'concurrency', 'timestamp', 'input_tokens_target',
        'prefill_tokens', 'decode_tokens', 'cache_hit_tokens',
        'ttft', 'prefill_speed', 'tps', 'tpot',
        'system_throughput', 'system_input_throughput', 'rps',
        'token_calc_method', 'error',
    ],
}

# Segmented test column fallback (when effective_* is missing, use tokenizer columns)
SEGMENTED_COLUMN_FALLBACK = {
    'effective_prefill_tokens': 'prefill_tokens',
    'effective_decode_tokens':  'decode_tokens',
}

# Numeric formatting rules
FORMAT_RULES = {
    'round_4': {'TTFT (s)', 'TPOT (s)', 'TPOT P95 (s)', 'TPOT P99 (s)'},
    'round_1': {'Prefill Speed (t/s)', 'TPS (t/s)', 'RPS (req/s)',
                'System Throughput (t/s)', 'Input Throughput (t/s)', 'Output Throughput (t/s)', 'Total Throughput (t/s)'},
    'int':     {'Prefill Tokens', 'Decode Tokens', 'Prefill Tokens (Effective)',
                'Decode Tokens (Effective)', 'Cache Hit (tokens)', 'Target Context (tokens)',
                'Target Input (tokens)', 'Concurrency', 'Round'},
}


def format_results_for_display(df: pd.DataFrame, test_type: str = None) -> pd.DataFrame:
    """Unified formatting for all test type display data

    Args:
        df: Raw result DataFrame
        test_type: Test type string

    Returns:
        Formatted DataFrame (for display only)
    """
    if df.empty:
        return df

    # 1. Determine display column order
    # If test_type is None, try to infer from df (if test_type column exists and is unique)
    if not test_type and 'test_type' in df.columns:
        unique_types = df['test_type'].dropna().unique()
        if len(unique_types) == 1:
            test_type = unique_types[0]

    ordered = TEST_TYPE_DISPLAY_ORDER.get(test_type, [])

    # Handle column fallback (segmented test effective_* columns)
    fallback = SEGMENTED_COLUMN_FALLBACK if test_type in ('Segmented Context Test', '分段onunder文Test') else {}

    # Build final column list
    final_cols = []
    
    # First add columns from the ordered list
    for col in ordered:
        if col in df.columns:
            final_cols.append(col)
        elif col in fallback and fallback[col] in df.columns:
            final_cols.append(fallback[col])

    # Append columns not in ordered list but present in data (except hidden)
    for col in df.columns:
        if col not in final_cols and col not in HIDDEN_COLUMNS:
            final_cols.append(col)

    # If no config, just remove hidden columns
    if not final_cols:
        final_cols = [c for c in df.columns if c not in HIDDEN_COLUMNS]

    # Create a copy to avoid modifying original data
    display_df = df[final_cols].copy()

    # 2. Rename columns
    rename = {col: COLUMN_DISPLAY_NAMES[col]
              for col in final_cols if col in COLUMN_DISPLAY_NAMES}
    display_df = display_df.rename(columns=rename)

    # 3. Numeric formatting
    for col in display_df.columns:
        if col in FORMAT_RULES['round_4']:
            display_df[col] = display_df[col].apply(
                lambda x: round(x, 4) if isinstance(x, (int, float)) else x)
        elif col in FORMAT_RULES['round_1']:
            display_df[col] = display_df[col].apply(
                lambda x: round(x, 1) if isinstance(x, (int, float)) else x)
        elif col in FORMAT_RULES['int']:
            display_df[col] = display_df[col].apply(
                lambda x: int(x) if isinstance(x, (int, float)) and not pd.isna(x) else x)

    return display_df

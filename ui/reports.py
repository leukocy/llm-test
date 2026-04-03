import pandas as pd
import streamlit as st

from ui.charts import apply_theme, plot_plotly_line
from ui.export import create_static_chart_download_link, export_benchmark_summary_chart
from ui.insights import generate_performance_insights, get_performance_grade
from ui.styled_tables import create_styled_summary_table
from ui.markdown_summary import _generate_markdown_summary


def safe_to_markdown(df, **kwargs):
    """Safely convert DataFrame to markdown, with fallback if tabulate is not installed."""
    try:
        return df.to_markdown(**kwargs)
    except ImportError:
        # Fallback: create a simple text table representation
        return df.to_string(**kwargs)

# Column rename map to add units to table headers
COLUMN_RENAME_MAP = {
    'Best_TTFT': 'Best_TTFT (s)',
    'Max_System_Output_Throughput': 'Max_System_Output_Throughput (tokens/s)',
    'Max_System_Input_Throughput': 'Max_System_Input_Throughput (tokens/s)',
    'Max_System_Throughput': 'Max_System_Throughput (tokens/s)',
    'Max_RPS': 'Max_RPS (req/s)',
    'TPOT_Mean': 'TPOT_Mean (ms)',
    'TPOT_P95': 'TPOT_P95 (ms)',
    'TPOT_P99': 'TPOT_P99 (ms)',
    'Max_Prefill_Speed': 'Max_Prefill_Speed (tokens/s)',
    'Max_TPS': 'Max_TPS (tokens/s)',
    'Success_Rate': 'Success_Rate (%)'
}

COLUMN_TOOLTIPS = {
    "Max_System_Output_Throughput": "System Output Throughput (Decode Output Throughput): Token generation rate during the Decode phase only (Total Output Tokens / Decode Duration). Excludes Prefill phase.",
    "Max_System_Input_Throughput": "System Input Throughput (Prefill Input Throughput): Token processing rate during the Prefill phase only (Total Input Tokens / Prefill Duration). Excludes Decode phase.",
    "Max_System_Throughput": "Total System Throughput: Total tokens processed per unit time (Input + Output) / Total Duration.",
    "Max_RPS": "Requests Per Second: Number of requests system can complete per second (Total Requests / Total Duration).",
    "TPOT_Mean": "Average Time Per Output Token. Measures generation fluency.",
    "Max_Prefill_Speed": "Max Prefill Speed: Rate at which the system processes input prompt tokens (Tokens/s).",
    "Best_TTFT": "Best Time To First Token (TTFT). Measures response agility.",
    "Success_Rate": "Request success rate.",
    "TPOT_P95": "P95 Token Latency: 95% of tokens are generated faster than this value.",
    "TPOT_P99": "P99 Token Latency: 99% of tokens are generated faster than this value.",
    "Max_TPS": "Max Single-stream Throughput: Generation rate of a single request (Decode Only).",
    "Actual_Tokens_Mean": "Average actual tokens generated.",
    "Best_Prefill_Speed": "Best Prefill Speed.",
    "Input Tokens": "Target input token count.",
    "Context Length": "Target context length.",
    "Concurrency": "Number of concurrent requests."
}

def _render_test_summary_card(model_id, provider, duration, test_config, system_info=None, df=None, test_type=None):
    """
    Renders a test-type-aware summary card with meaningful insights.
    Instead of generic averages, shows the specific 'story' relevant to each test type.
    """
    import numpy as np

    with st.expander("⏱️ Test Summary", expanded=True):
        # --- Row 1: Basic Info & Context ---
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.markdown(f"**🤖 Model**: `{model_id}`")
        with info_col2:
            st.markdown(f"**🔌 Provider**: `{provider}`")
        with info_col3:
            if duration >= 60:
                mins = int(duration // 60)
                secs = duration % 60
                st.markdown(f"**⏱️ Total Duration**: `{mins}m {secs:.1f}s`")
            else:
                st.markdown(f"**⏱️ Total Duration**: `{duration:.2f}s`")

        # --- Row 2: Test-type-specific Performance Insights ---
        if df is not None and not df.empty:
            st.markdown("---")

            # Helper
            def _fmt(n):
                if n >= 1_000_000: return f"{n / 1_000_000:.2f}M"
                if n >= 1_000: return f"{n / 1_000:.1f}K"
                return f"{int(n)}"

            total_requests = len(df)
            error_col = df.get('error')
            successful = int(error_col.isnull().sum()) if error_col is not None else total_requests
            total_input = int(df['prefill_tokens'].sum()) if 'prefill_tokens' in df.columns else 0
            total_output = int(df['decode_tokens'].sum()) if 'decode_tokens' in df.columns else 0
            df_ok = df[df['error'].isnull()] if 'error' in df.columns else df

            # ===== CONCURRENCY =====
            if test_type == 'concurrency' and 'concurrency' in df.columns:
                conc_levels = sorted(df_ok['concurrency'].unique())
                if len(conc_levels) >= 2:
                    lo, hi = conc_levels[0], conc_levels[-1]
                    lo_df, hi_df = df_ok[df_ok['concurrency'] == lo], df_ok[df_ok['concurrency'] == hi]
                    ttft_lo = lo_df['ttft'].mean() if 'ttft' in lo_df.columns else 0
                    ttft_hi = hi_df['ttft'].mean() if 'ttft' in hi_df.columns else 0
                    ttft_ratio = ttft_hi / ttft_lo if ttft_lo > 0 else 0
                    tp_col = 'system_output_throughput' if 'system_output_throughput' in df_ok.columns else ('system_throughput' if 'system_throughput' in df_ok.columns else None)
                    if tp_col:
                        tp_by_c = df_ok.groupby('concurrency')[tp_col].max()
                        peak_tp, peak_c = tp_by_c.max(), int(tp_by_c.idxmax())
                    else:
                        peak_tp, peak_c = 0, "N/A"
                    tps_lo = lo_df['tps'].mean() if 'tps' in lo_df.columns else 0
                    tps_hi = hi_df['tps'].mean() if 'tps' in hi_df.columns else 0

                    st.markdown(f"##### 🔍 Concurrency Scaling Analysis (Concurrency {int(lo)} → {int(hi)})")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("Total Requests", f"{total_requests}", delta=f"{successful} succeeded")
                    with m2: st.metric(f"TTFT @ C{int(lo)}", f"{ttft_lo:.3f}s", help="Low-concurrency baseline latency")
                    with m3:
                        d = f"↑ {ttft_ratio:.1f}x" if ttft_ratio > 1.2 else "Stable"
                        st.metric(f"TTFT @ C{int(hi)}", f"{ttft_hi:.3f}s", delta=d, delta_color="inverse")
                    with m4: st.metric("Peak System Throughput", f"{peak_tp:.1f} t/s", delta=f"@ C{peak_c}" if peak_c != "N/A" else None)

                    m5, m6, m7, m8 = st.columns(4)
                    with m5: st.metric(f"Single TPS @ C{int(lo)}", f"{tps_lo:.1f} t/s")
                    with m6:
                        d2 = f"↓ {((1 - tps_hi/tps_lo)*100):.0f}%" if tps_lo > 0 and tps_hi < tps_lo else "Stable"
                        st.metric(f"Single TPS @ C{int(hi)}", f"{tps_hi:.1f} t/s", delta=d2, delta_color="inverse")
                    with m7: st.metric("Total Input Tokens", _fmt(total_input))
                    with m8: st.metric("Total Output Tokens", _fmt(total_output))

                    # --- Row 3: Peak & Trend Analysis ---
                    ttft_by_c = df_ok.groupby('concurrency')['ttft'].mean()
                    best_ttft_val = ttft_by_c.min()
                    best_ttft_c = int(ttft_by_c.idxmin())
                    tps_by_c = df_ok.groupby('concurrency')['tps'].mean()
                    peak_tps_val = tps_by_c.max()
                    peak_tps_c = int(tps_by_c.idxmax())

                    # Detect TTFT trend: check if monotonically increasing or has a dip
                    ttft_vals = [ttft_by_c[c] for c in conc_levels]
                    if len(ttft_vals) >= 3:
                        diffs = [ttft_vals[i+1] - ttft_vals[i] for i in range(len(ttft_vals)-1)]
                        max_jump_idx = max(range(len(diffs)), key=lambda i: diffs[i])
                        jump_from = int(conc_levels[max_jump_idx])
                        jump_to = int(conc_levels[max_jump_idx + 1])
                        jump_pct = (diffs[max_jump_idx] / ttft_vals[max_jump_idx] * 100) if ttft_vals[max_jump_idx] > 0 else 0
                        ttft_trend = f"C{jump_from}→C{jump_to} jumped {jump_pct:.0f}%" if jump_pct > 10 else "Gradual increase"
                    else:
                        ttft_trend = f"↑ {ttft_ratio:.1f}x" if ttft_ratio > 1.2 else "Stable"

                    # Detect throughput saturation point
                    if tp_col:
                        tp_vals = [tp_by_c[c] for c in conc_levels if c in tp_by_c.index]
                        if len(tp_vals) >= 3:
                            # Find where throughput stops growing significantly (<5% gain)
                            saturation_c = None
                            for i in range(1, len(tp_vals)):
                                gain = (tp_vals[i] - tp_vals[i-1]) / tp_vals[i-1] if tp_vals[i-1] > 0 else 0
                                if gain < 0.05 and saturation_c is None:
                                    saturation_c = int(conc_levels[i])
                            saturation_label = f"@ C{saturation_c}" if saturation_c else "Not saturated"
                        else:
                            saturation_label = "Insufficient data"
                    else:
                        saturation_label = "N/A"

                    st.markdown("---")
                    st.markdown("##### 📈 Peak & Trend Analysis")
                    m9, m10, m11, m12 = st.columns(4)
                    with m9: st.metric("Best TTFT", f"{best_ttft_val:.3f}s", delta=f"@ C{best_ttft_c}")
                    with m10: st.metric("Peak Single TPS", f"{peak_tps_val:.1f} t/s", delta=f"@ C{peak_tps_c}")
                    with m11: st.metric("TTFT Trend", ttft_trend, help="TTFT change pattern as concurrency increases")
                    with m12: st.metric("Throughput Saturation", saturation_label, help="Inflection point where throughput growth drops below 5%")
                else:
                    _render_generic_summary(st, df_ok, total_requests, successful, total_input, total_output, duration)

            # ===== PREFILL =====
            elif test_type == 'prefill' and 'input_tokens_target' in df.columns:
                levels = sorted(df_ok['input_tokens_target'].unique())
                if len(levels) >= 2:
                    sm, lg = levels[0], levels[-1]
                    sm_df, lg_df = df_ok[df_ok['input_tokens_target'] == sm], df_ok[df_ok['input_tokens_target'] == lg]
                    ps_sm = sm_df['prefill_speed'].mean() if 'prefill_speed' in sm_df.columns else 0
                    ps_lg = lg_df['prefill_speed'].mean() if 'prefill_speed' in lg_df.columns else 0
                    ttft_sm = sm_df['ttft'].mean() if 'ttft' in sm_df.columns else 0
                    ttft_lg = lg_df['ttft'].mean() if 'ttft' in lg_df.columns else 0
                    ratio = ps_lg / ps_sm if ps_sm > 0 else 0

                    st.markdown(f"##### 🔍 Prefill Speed Change ({_fmt(sm)} → {_fmt(lg)} tokens)")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("Total Requests", f"{total_requests}", delta=f"{successful} succeeded")
                    with m2: st.metric(f"Prefill @ {_fmt(sm)}t", f"{ps_sm:.0f} t/s")
                    with m3:
                        d = f"↓ {((1 - ratio)*100):.0f}%" if ratio < 0.95 else "Stable"
                        st.metric(f"Prefill @ {_fmt(lg)}t", f"{ps_lg:.0f} t/s", delta=d, delta_color="inverse")
                    with m4: st.metric("Speed Retention", f"{ratio*100:.1f}%")

                    m5, m6, m7, m8 = st.columns(4)
                    with m5: st.metric(f"TTFT @ {_fmt(sm)}t", f"{ttft_sm:.3f}s")
                    with m6:
                        g = ttft_lg / ttft_sm if ttft_sm > 0 else 0
                        st.metric(f"TTFT @ {_fmt(lg)}t", f"{ttft_lg:.3f}s", delta=f"↑ {g:.1f}x" if g > 1.5 else "Stable", delta_color="inverse")
                    with m7: st.metric("Total Input Tokens", _fmt(total_input))
                    with m8: st.metric("Total Output Tokens", _fmt(total_output))

                    # --- Row 3: Peak & Trend Analysis ---
                    ps_by_level = df_ok.groupby('input_tokens_target')['prefill_speed'].mean()
                    peak_ps_val = ps_by_level.max()
                    peak_ps_level = int(ps_by_level.idxmax())
                    ttft_by_level = df_ok.groupby('input_tokens_target')['ttft'].mean()
                    best_ttft_val = ttft_by_level.min()
                    best_ttft_level = int(ttft_by_level.idxmin())

                    # Detect biggest prefill speed drop between adjacent levels
                    ps_vals = [ps_by_level[lv] for lv in levels]
                    if len(ps_vals) >= 3:
                        drops = [(ps_vals[i] - ps_vals[i+1]) / ps_vals[i] * 100 if ps_vals[i] > 0 else 0 for i in range(len(ps_vals)-1)]
                        max_drop_idx = max(range(len(drops)), key=lambda i: drops[i])
                        drop_from = levels[max_drop_idx]
                        drop_to = levels[max_drop_idx + 1]
                        drop_pct = drops[max_drop_idx]
                        speed_trend = f"{_fmt(drop_from)}→{_fmt(drop_to)} dropped {drop_pct:.0f}%" if drop_pct > 10 else "Gradual decline"
                    else:
                        speed_trend = f"↓ {((1 - ratio)*100):.0f}%" if ratio < 0.95 else "Stable"

                    # Detect TTFT scaling pattern (linear vs super-linear)
                    if len(levels) >= 3:
                        input_ratio = levels[-1] / levels[0] if levels[0] > 0 else 1
                        ttft_ratio_full = ttft_by_level[levels[-1]] / ttft_by_level[levels[0]] if ttft_by_level[levels[0]] > 0 else 1
                        if ttft_ratio_full > input_ratio * 1.5:
                            ttft_scaling = "Super-linear growth ⚠️"
                        elif ttft_ratio_full > input_ratio * 0.8:
                            ttft_scaling = "Near-linear"
                        else:
                            ttft_scaling = "Sub-linear ✅"
                    else:
                        ttft_scaling = f"↑ {g:.1f}x"

                    st.markdown("---")
                    st.markdown("##### 📈 Peak & Trend Analysis")
                    m9, m10, m11, m12 = st.columns(4)
                    with m9: st.metric("Peak Prefill Speed", f"{peak_ps_val:.0f} t/s", delta=f"@ {_fmt(peak_ps_level)}t")
                    with m10: st.metric("Best TTFT", f"{best_ttft_val:.3f}s", delta=f"@ {_fmt(best_ttft_level)}t")
                    with m11: st.metric("Speed Decline Point", speed_trend, help="Location of the largest single drop in Prefill speed")
                    with m12: st.metric("TTFT Growth Pattern", ttft_scaling, help="TTFT growth relationship relative to input length")
                else:
                    _render_generic_summary(st, df_ok, total_requests, successful, total_input, total_output, duration)

            # ===== LONG CONTEXT =====
            elif test_type == 'long_context' and 'context_length_target' in df.columns:
                levels = sorted(df_ok['context_length_target'].unique())
                if len(levels) >= 2:
                    sh, lo = levels[0], levels[-1]
                    sh_df, lo_df = df_ok[df_ok['context_length_target'] == sh], df_ok[df_ok['context_length_target'] == lo]
                    ttft_sh = sh_df['ttft'].mean() if 'ttft' in sh_df.columns else 0
                    ttft_lo2 = lo_df['ttft'].mean() if 'ttft' in lo_df.columns else 0
                    growth = ttft_lo2 / ttft_sh if ttft_sh > 0 else 0
                    ps_sh = sh_df['prefill_speed'].mean() if 'prefill_speed' in sh_df.columns else 0
                    ps_lo2 = lo_df['prefill_speed'].mean() if 'prefill_speed' in lo_df.columns else 0
                    tps_sh = sh_df['tps'].mean() if 'tps' in sh_df.columns else 0
                    tps_lo2 = lo_df['tps'].mean() if 'tps' in lo_df.columns else 0
                    tps_r = tps_lo2 / tps_sh if tps_sh > 0 else 0

                    st.markdown(f"##### 🔍 Long Context Performance Degradation ({_fmt(sh)} → {_fmt(lo)} ctx)")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("Total Requests", f"{total_requests}", delta=f"{successful} succeeded")
                    with m2: st.metric(f"TTFT @ {_fmt(sh)}", f"{ttft_sh:.3f}s")
                    with m3: st.metric(f"TTFT @ {_fmt(lo)}", f"{ttft_lo2:.3f}s", delta=f"↑ {growth:.1f}x" if growth > 1.5 else "Stable", delta_color="inverse")
                    with m4: st.metric("TTFT Growth Factor", f"{growth:.1f}x")

                    m5, m6, m7, m8 = st.columns(4)
                    with m5: st.metric(f"Prefill @ {_fmt(sh)}", f"{ps_sh:.0f} t/s")
                    with m6: st.metric(f"Prefill @ {_fmt(lo)}", f"{ps_lo2:.0f} t/s")
                    with m7:
                        d = f"↓ {((1 - tps_r)*100):.0f}%" if tps_r < 0.9 else "Stable"
                        st.metric("TPS Stability", f"{tps_r*100:.1f}%", delta=d, delta_color="inverse" if tps_r < 0.9 else "normal")
                    with m8: st.metric("Total Output Tokens", _fmt(total_output))

                    # --- Row 3: Peak & Trend Analysis ---
                    ps_by_ctx = df_ok.groupby('context_length_target')['prefill_speed'].mean()
                    peak_ps_val = ps_by_ctx.max()
                    peak_ps_ctx = int(ps_by_ctx.idxmax())
                    tps_by_ctx = df_ok.groupby('context_length_target')['tps'].mean()
                    peak_tps_val = tps_by_ctx.max()
                    peak_tps_ctx = int(tps_by_ctx.idxmax())

                    # Find the biggest TTFT jump between adjacent levels
                    ttft_by_ctx = df_ok.groupby('context_length_target')['ttft'].mean()
                    ttft_vals = [ttft_by_ctx[lv] for lv in levels]
                    if len(ttft_vals) >= 3:
                        jumps = [(ttft_vals[i+1] - ttft_vals[i]) / ttft_vals[i] * 100 if ttft_vals[i] > 0 else 0 for i in range(len(ttft_vals)-1)]
                        max_jump_idx = max(range(len(jumps)), key=lambda i: jumps[i])
                        jump_from = levels[max_jump_idx]
                        jump_to = levels[max_jump_idx + 1]
                        jump_pct = jumps[max_jump_idx]
                        ttft_inflection = f"{_fmt(jump_from)}→{_fmt(jump_to)} jumped {jump_pct:.0f}%" if jump_pct > 20 else "Uniform growth"
                    else:
                        ttft_inflection = f"↑ {growth:.1f}x"

                    # Overall TPS stability across all levels
                    tps_vals = [tps_by_ctx[lv] for lv in levels]
                    tps_cv = (np.std(tps_vals) / np.mean(tps_vals) * 100) if np.mean(tps_vals) > 0 else 0
                    tps_stability = f"Volatile {tps_cv:.0f}%" if tps_cv > 10 else f"Stable (CV={tps_cv:.0f}%)"

                    st.markdown("---")
                    st.markdown("##### 📈 Peak & Trend Analysis")
                    m9, m10, m11, m12 = st.columns(4)
                    with m9: st.metric("Peak Prefill Speed", f"{peak_ps_val:.0f} t/s", delta=f"@ {_fmt(peak_ps_ctx)} ctx")
                    with m10: st.metric("Peak TPS", f"{peak_tps_val:.1f} t/s", delta=f"@ {_fmt(peak_tps_ctx)} ctx")
                    with m11: st.metric("TTFT Inflection", ttft_inflection, help="Context length segment with the largest single TTFT increase")
                    with m12: st.metric("TPS Volatility", tps_stability, help="Coefficient of variation of TPS across context lengths")
                else:
                    _render_generic_summary(st, df_ok, total_requests, successful, total_input, total_output, duration)

            # ===== SEGMENTED (Prefix Caching) =====
            elif test_type == 'segmented' and 'context_length_target' in df.columns:
                levels = sorted(df_ok['context_length_target'].unique())
                cache_total = int(df_ok['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in df_ok.columns else 0
                cache_rate = (cache_total / total_input * 100) if total_input > 0 else 0

                if len(levels) >= 2:
                    f_seg, l_seg = levels[0], levels[-1]
                    f_df, l_df = df_ok[df_ok['context_length_target'] == f_seg], df_ok[df_ok['context_length_target'] == l_seg]
                    # Use max TTFT which represents the uncached request
                    ttft_f = f_df['ttft'].max() if 'ttft' in f_df.columns else 0
                    ttft_l = l_df['ttft'].max() if 'ttft' in l_df.columns else 0
                    ch_f = int(f_df['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in f_df.columns else 0
                    in_f = int(f_df['prefill_tokens'].sum()) if 'prefill_tokens' in f_df.columns else 0
                    r_f = (ch_f / in_f * 100) if in_f > 0 else 0
                    ch_l = int(l_df['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in l_df.columns else 0
                    in_l = int(l_df['prefill_tokens'].sum()) if 'prefill_tokens' in l_df.columns else 0
                    r_l = (ch_l / in_l * 100) if in_l > 0 else 0
                    ps_f = f_df['prefill_speed'].max() if 'prefill_speed' in f_df.columns else 0
                    ps_l = l_df['prefill_speed'].max() if 'prefill_speed' in l_df.columns else 0

                    st.markdown(f"##### 🔍 Prefix Caching Effect ({len(levels)} segments)")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("Total Requests", f"{total_requests}", delta=f"{successful} succeeded")
                    with m2: st.metric("Total Cache Hit Rate", f"{cache_rate:.1f}%", help=f"Hit {_fmt(cache_total)} / Total {_fmt(total_input)}")
                    with m3:
                        d = f"↑ {r_l - r_f:.0f}pp" if r_l > r_f + 5 else "No growth"
                        st.metric("Cache Rate Change", f"{r_f:.0f}% → {r_l:.0f}%", delta=d)
                    with m4: st.metric("Uncached TTFT First→Last", f"{ttft_f:.3f}s → {ttft_l:.3f}s")

                    m5, m6, m7, m8 = st.columns(4)
                    with m5: st.metric("Prefill @ First", f"{ps_f:.0f} t/s")
                    with m6: st.metric("Prefill @ Last", f"{ps_l:.0f} t/s")
                    with m7: st.metric("Total Input Tokens", _fmt(total_input))
                    with m8: st.metric("Total Output Tokens", _fmt(total_output))

                    # --- Row 3: Peak & Trend Analysis ---
                    # Per-segment cache rate and uncached TTFT
                    cache_by_seg = {}
                    ttft_by_seg = df_ok.groupby('context_length_target')['ttft'].max()
                    ps_by_seg = df_ok.groupby('context_length_target')['prefill_speed'].max()
                    for lv in levels:
                        lv_df = df_ok[df_ok['context_length_target'] == lv]
                        ch = int(lv_df['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in lv_df.columns else 0
                        inp = int(lv_df['prefill_tokens'].sum()) if 'prefill_tokens' in lv_df.columns else 0
                        cache_by_seg[lv] = (ch / inp * 100) if inp > 0 else 0

                    # Peak cache rate segment
                    peak_cache_seg = max(cache_by_seg, key=cache_by_seg.get)
                    peak_cache_rate = cache_by_seg[peak_cache_seg]
                    # Best TTFT segment (smallest Uncached TTFT)
                    best_ttft_seg = int(ttft_by_seg.idxmin())
                    best_ttft_val = ttft_by_seg.min()
                    # Peak prefill speed segment
                    peak_ps_seg = int(ps_by_seg.idxmax())
                    peak_ps_val = ps_by_seg.max()

                    # Cache effectiveness trend
                    cache_vals = [cache_by_seg[lv] for lv in levels]
                    if len(cache_vals) >= 3:
                        # Check if cache rate is consistently growing
                        increasing = all(cache_vals[i+1] >= cache_vals[i] - 1 for i in range(len(cache_vals)-1))
                        cache_trend = "Increasing ✅" if increasing and cache_vals[-1] > cache_vals[0] + 5 else ("Volatile" if not increasing else "Stable")
                    else:
                        cache_trend = f"{r_f:.0f}% → {r_l:.0f}%"

                    st.markdown("---")
                    st.markdown("##### 📈 Peak & Trend Analysis")
                    m9, m10, m11, m12 = st.columns(4)
                    with m9: st.metric("Peak Cache Rate", f"{peak_cache_rate:.0f}%", delta=f"@ {_fmt(peak_cache_seg)} ctx")
                    with m10: st.metric("Min Uncached TTFT", f"{best_ttft_val:.3f}s", delta=f"@ {_fmt(best_ttft_seg)} ctx")
                    with m11: st.metric("Peak Prefill Speed", f"{peak_ps_val:.0f} t/s", delta=f"@ {_fmt(peak_ps_seg)} ctx")
                    with m12: st.metric("Cache Rate Trend", cache_trend, help="Overall trend of cache hit rates across segments")
                else:
                    _render_generic_summary(st, df_ok, total_requests, successful, total_input, total_output, duration)

            # ===== MATRIX =====
            elif test_type == 'matrix' and 'concurrency' in df.columns and 'context_length_target' in df.columns:
                tp_col = 'system_output_throughput' if 'system_output_throughput' in df_ok.columns else ('system_throughput' if 'system_throughput' in df_ok.columns else None)
                conc_levels = sorted(df_ok['concurrency'].unique())
                ctx_levels = sorted(df_ok['context_length_target'].unique())

                st.markdown("##### 🔍 Matrix Best/Worst Configuration")
                m1, m2, m3, m4 = st.columns(4)
                with m1: st.metric("Total Requests", f"{total_requests}", delta=f"{successful} succeeded")
                with m2: st.metric("Test Matrix", f"{len(conc_levels)}×{len(ctx_levels)}", help=f"Concurrency: {conc_levels}\nContext: {ctx_levels}")
                if tp_col:
                    combo = df_ok.groupby(['concurrency', 'context_length_target'])[tp_col].max().reset_index()
                    best = combo.loc[combo[tp_col].idxmax()]
                    worst = combo.loc[combo[tp_col].idxmin()]
                    with m3: st.metric("Max Throughput", f"{best[tp_col]:.1f} t/s", delta=f"C{int(best['concurrency'])}×{_fmt(best['context_length_target'])}ctx")
                    with m4: st.metric("Min Throughput", f"{worst[tp_col]:.1f} t/s", delta=f"C{int(worst['concurrency'])}×{_fmt(worst['context_length_target'])}ctx")
                else:
                    with m3: st.metric("Total Input Tokens", _fmt(total_input))
                    with m4: st.metric("Total Output Tokens", _fmt(total_output))

                ttft_all = df_ok[df_ok['ttft'] > 0]['ttft'] if 'ttft' in df_ok.columns else pd.Series()
                m5, m6, m7, m8 = st.columns(4)
                with m5: st.metric("Best TTFT", f"{ttft_all.min():.3f}s" if not ttft_all.empty else "N/A")
                with m6: st.metric("Worst TTFT", f"{ttft_all.max():.3f}s" if not ttft_all.empty else "N/A")
                with m7: st.metric("Total Input Tokens", _fmt(total_input))
                with m8: st.metric("Total Output Tokens", _fmt(total_output))

            # ===== GENERIC FALLBACK =====
            else:
                _render_generic_summary(st, df_ok, total_requests, successful, total_input, total_output, duration)

        # --- Row 3: Test Configuration (filtered) ---
        if test_config:
            st.markdown("---")
            st.markdown("**📋 Test Configuration:**")

            # Filter out keys that are already shown above
            skip_keys = {'Test Type', 'Model ID', 'Provider', 'Timestamp'}
            filtered_config = {k: v for k, v in test_config.items() if k not in skip_keys}

            if filtered_config:
                config_cols = st.columns(min(len(filtered_config), 4))
                for i, (k, v) in enumerate(filtered_config.items()):
                    with config_cols[i % len(config_cols)]:
                        if isinstance(v, list):
                            v_str = ", ".join(map(str, v))
                        else:
                            v_str = str(v)
                        st.markdown(f"**{k}**: `{v_str}`")

        # --- Row 4: System Environment ---
        if system_info:
            st.markdown("---")
            st.markdown("**🖥️ System Environment:**")
            display_keys = {
                'processor': 'Processor',
                'gpu': 'GPU',
                'memory': 'Memory',
                'system': 'OS',
                'engine_name': 'Engine'
            }

            # Only show keys that have values
            valid_items = [(label, system_info.get(key)) for key, label in display_keys.items() if system_info.get(key)]
            if valid_items:
                sys_cols = st.columns(min(len(valid_items), 3))
                for i, (label, val) in enumerate(valid_items):
                    with sys_cols[i % len(sys_cols)]:
                        st.markdown(f"**{label}**: `{val}`")


def _render_generic_summary(st_mod, df, total_requests, successful, total_input, total_output, duration):
    """Fallback generic metrics when test type is unknown or has insufficient data."""
    import numpy as np

    def _fmt(n):
        if n >= 1_000_000: return f"{n / 1_000_000:.2f}M"
        if n >= 1_000: return f"{n / 1_000:.1f}K"
        return f"{int(n)}"

    success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
    ttft_data = df[df['ttft'] > 0]['ttft'] if 'ttft' in df.columns else pd.Series()
    avg_tps = df[df['tps'] > 0]['tps'].mean() if 'tps' in df.columns else 0

    # Data quality warning
    if total_input == 0 and total_output == 0 and ttft_data.empty:
        st_mod.warning("⚠️ **Data Quality Alert**: TTFT, TPS, and Token counts are all 0 or missing. The test may not have correctly captured performance metrics. Please check the API response usage info and streaming output.")

    m1, m2, m3, m4 = st_mod.columns(4)
    with m1: st_mod.metric("Total Requests", f"{total_requests}", delta=f"{successful} succeeded")
    with m2: st_mod.metric("Request Success Rate", f"{success_rate:.1f}%")
    with m3:
        avg_ttft = ttft_data.mean() if not ttft_data.empty else 0
        st_mod.metric("Avg TTFT", f"{avg_ttft:.3f}s" if avg_ttft > 0 else "N/A")
    with m4:
        st_mod.metric("Avg TPS", f"{avg_tps:.1f} t/s" if avg_tps > 0 else "N/A")

    m5, m6, m7, m8 = st_mod.columns(4)
    with m5:
        p95 = np.percentile(ttft_data, 95) if not ttft_data.empty else 0
        st_mod.metric("P95 TTFT", f"{p95:.3f}s" if p95 > 0 else "N/A")
    with m6:
        pf = df[df['prefill_speed'] > 0]['prefill_speed'].mean() if 'prefill_speed' in df.columns else 0
        st_mod.metric("Avg Prefill Speed", f"{pf:.1f} t/s" if pf > 0 else "N/A")
    with m7: st_mod.metric("Total Input Tokens", _fmt(total_input))
    with m8: st_mod.metric("Total Output Tokens", _fmt(total_output))


def generate_concurrency_report(df_group, model_id, provider="Unknown", duration=0, test_config=None, system_info=None):
    st.subheader("Concurrency Test Charts")
    
    # Use standard summary card
    _render_test_summary_card(model_id, provider, duration, test_config, system_info, df=df_group, test_type='concurrency')

    # Add Test Summary Section for Markdown Report
    report_md = "# Concurrency Performance Test Report\n\n"
    report_md += f"**Provider**: {provider} | **Model**: {model_id}\n"
    report_md += f"**Total Duration**: {duration:.2f}s\n\n"

    if test_config:
        report_md += "### Test Configuration\n"
        for k, v in test_config.items():
            report_md += f"- **{k}**: {v}\n"
        report_md += "\n"

    # Add Enhanced Summary
    report_md += _generate_markdown_summary(df_group, 'concurrency', duration)

    report_md += "## Detailed Results\n\n"

    if 'concurrency' not in df_group.columns:
        st.error("Concurrency test report failed: missing 'concurrency' column in data.")
        return ""
    df_group = df_group[pd.to_numeric(df_group['concurrency'], errors='coerce').notna()]
    df_group = df_group[df_group['concurrency'] > 0].copy()
    if df_group.empty:
        st.warning("Concurrency test report: no valid concurrency data.")
        return ""

    rounds_per_level = df_group.groupby('concurrency')['round'].max().max()

    round_stats = df_group.groupby(['concurrency', 'round']).agg({
        'ttft': 'mean',
        'system_throughput': 'mean',
        'session_id': 'count'
    }).reset_index()

    best_ttft_idx = round_stats.groupby('concurrency')['ttft'].idxmin()
    summary_ttft = round_stats.loc[best_ttft_idx].copy().rename(columns={'ttft': 'Best_TTFT'})[['concurrency', 'Best_TTFT']]

    # Backward compatibility: Ensure system_output_throughput exists
    if 'system_output_throughput' not in df_group.columns:
        if 'system_throughput' in df_group.columns:
            df_group['system_output_throughput'] = df_group['system_throughput']
        else:
             df_group['system_output_throughput'] = 0

    best_sys_tps_idx = df_group.groupby('concurrency')['system_output_throughput'].idxmax()
    summary_sys_tps = df_group.loc[best_sys_tps_idx].copy()

    # Ensure new columns exist for backward compatibility
    if 'system_input_throughput' not in summary_sys_tps.columns:
        summary_sys_tps['system_input_throughput'] = 0
    if 'rps' not in summary_sys_tps.columns:
        summary_sys_tps['rps'] = 0

    summary_sys_tps = summary_sys_tps.rename(columns={
        'system_output_throughput': 'Max_System_Output_Throughput',
        'system_input_throughput': 'Max_System_Input_Throughput',
        'rps': 'Max_RPS'
    })[['concurrency', 'Max_System_Output_Throughput', 'Max_System_Input_Throughput', 'Max_RPS']]

    # Calculate QPM based on RPS
    summary_sys_tps['Max_QPM'] = summary_sys_tps['Max_RPS'] * 60

    if 'tpot' not in df_group.columns:
        df_group['tpot'] = 0

    # Ensure new columns exist for backward compatibility
    if 'tpot_p95' not in df_group.columns:
        df_group['tpot_p95'] = df_group['tpot'] # Fallback
    if 'tpot_p99' not in df_group.columns:
        df_group['tpot_p99'] = df_group['tpot'] # Fallback

    req_stats = df_group.groupby('concurrency').agg(
        Success_Rate=('error', lambda x: x.isnull().mean()),
        Token_Calc_Method=('token_calc_method', 'first'),
        Num_Requests=('session_id', 'size'),
        Actual_Tokens_Mean=('prefill_tokens', 'mean'), # Input tokens
        Actual_Decode_Max=('decode_tokens', 'max'),     # Output tokens
        TPOT_Mean=('tpot', lambda x: x[x > 0].mean() * 1000),
        # Use the pre-calculated token-level P95/P99 (averaged across requests)
        TPOT_P95=('tpot_p95', lambda x: x[x > 0].mean() * 1000),
        TPOT_P99=('tpot_p99', lambda x: x[x > 0].mean() * 1000)
    ).reset_index().fillna(0)

    summary = pd.merge(summary_ttft, summary_sys_tps, on='concurrency', how='left')
    summary = pd.merge(summary, req_stats, on='concurrency', how='left')

    st.subheader(f"Best Performance Stats (across {rounds_per_level} rounds per level)")

    display_columns = [
        'concurrency', 'Num_Requests', 'Success_Rate',
        'Best_TTFT',
        'Max_System_Output_Throughput', 'Max_System_Input_Throughput',
        'Max_RPS', 'Max_QPM', # Added Max_QPM here
        'TPOT_Mean', 'TPOT_P95', 'TPOT_P99'
    ]
    display_columns = [col for col in display_columns if col in summary.columns]

    # Rename columns to include units
    summary = summary.rename(columns=COLUMN_RENAME_MAP)
    # Manually add QPM mapping if not in map
    summary = summary.rename(columns={'Max_QPM': 'Max_QPM (req/min)'})

    # Update display columns with new names
    display_columns = [COLUMN_RENAME_MAP.get(col, col) for col in display_columns]
    display_columns = ['Max_QPM (req/min)' if c == 'Max_QPM' else c for c in display_columns]

    # === Enhanced: Styled table ===
    styled_table = create_styled_summary_table(
        summary[display_columns].round(4),
        highlight_cols=[COLUMN_RENAME_MAP.get(c, c) for c in ['Best_TTFT', 'Max_System_Output_Throughput']],
        highlight_best=True
    )
    st.dataframe(styled_table,
        column_config={
            **{k: st.column_config.Column(help=v) for k, v in COLUMN_TOOLTIPS.items()},
            "Best_TTFT (s)": st.column_config.ProgressColumn(
                "Best_TTFT (s)",
                help="TTFT - Time To First Token (lower is better)",
                format="%.4f s",
                min_value=0,
                max_value=summary['Best_TTFT (s)'].max() * 1.2 if not summary.empty else 1,
            ),
            "Max_System_Output_Throughput (tokens/s)": st.column_config.ProgressColumn(
                "System Output Throughput",
                help="System Output Throughput (higher is better)",
                format="%d t/s",
                min_value=0,
                max_value=summary['Max_System_Output_Throughput (tokens/s)'].max() * 1.1 if not summary.empty else 100,
            ),
             "Max_QPM (req/min)": st.column_config.NumberColumn(
                "Max QPM (req/min)",
                help="Maximum requests per minute (estimated from RPS × 60)",
                format="%.1f"
            ),
            "Success_Rate (%)": st.column_config.ProgressColumn(
                "Success Rate",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            )
        }
    )
    report_md += safe_to_markdown(summary[display_columns].round(4), index=False) + "\n\n"

    # === Enhanced: Performance insights ===
    insights = generate_performance_insights(summary, 'concurrency', model_id)
    if insights:
        with st.expander("📊 Performance Insights & Analysis", expanded=True):
            grade, color, description = get_performance_grade(insights)
            st.markdown(f"**Overall Grade**: <span style='color:{color};font-size:20px;font-weight:bold'>{grade}</span> - {description}", unsafe_allow_html=True)
            st.markdown("---")
            for insight in insights:
                st.markdown(insight)
            report_md += "\n### Performance Insights\n\n" + "\n".join([f"- {i}" for i in insights]) + "\n\n"

    summary['concurrency_str'] = summary['concurrency'].astype(int).astype(str)

    # === Add Static Chart Download Button ===
    st.markdown("### 📈 Chart Analysis")
    chart_col1, chart_col2 = st.columns([1, 4])
    with chart_col1:
        # Generate static chart bytes
        # Check for column (renamed or original)
        tput_col = 'Max_System_Output_Throughput'
        if tput_col not in summary.columns and f"{tput_col} (tokens/s)" in summary.columns:
            tput_col = f"{tput_col} (tokens/s)"

        if tput_col in summary.columns:
            # Try to get system_info from session state
            system_info = st.session_state.get('system_info', None)
            static_fig_bytes = export_benchmark_summary_chart(
                summary, 'concurrency', model_id, provider, system_info=system_info
            )
            if static_fig_bytes:
                st.markdown(
                    create_static_chart_download_link(
                        static_fig_bytes,
                        f"concurrency_report_{model_id}.png",
                        "📷 Download Static Chart"
                    ),
                    unsafe_allow_html=True
                )

    # Calculate Labels for Charts
    avg_in = int(summary['Actual_Tokens_Mean'].mean()) if 'Actual_Tokens_Mean' in summary.columns else 0
    max_out = int(summary['Actual_Decode_Max'].max()) if 'Actual_Decode_Max' in summary.columns else 0
    io_label = f"(In: ~{avg_in}, Out: ~{max_out})"

    # --- Row 1: Latency Metrics ---
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_plotly_line(summary, 'concurrency_str', 'Best_TTFT (s)', f"Time To First Token (TTFT) {io_label}", "Concurrency", "Time (s)", model_id, rounds_per_level,
                               show_relative=False, force_linear_scale=True, provider=provider, line_color='#4bc0c0') # Teal for Prefill/TTFT
        if fig1:
            fig1 = apply_theme(fig1)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig3 = plot_plotly_line(summary, 'concurrency_str', 'TPOT_Mean (ms)', f"Avg Time Per Output Token (TPOT) {io_label}", "Concurrency", "Time (ms)", model_id, rounds_per_level,
                               show_relative=False, provider=provider, line_color='#ff9f40') # Orange for Decode/TPOT
        if fig3:
            fig3 = apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    # --- Row 2: Throughput Metrics ---
    col3, col4 = st.columns(2)
    with col3:
        fig_input = plot_plotly_line(summary, 'concurrency_str', 'Max_System_Input_Throughput (tokens/s)', f"System Input Throughput (Prefill) {io_label}", "Concurrency", "Tokens/s", model_id, rounds_per_level,
                               show_relative=False, provider=provider, line_color='#4bc0c0') # Teal for Input/Prefill Tput
        if fig_input:
            fig_input = apply_theme(fig_input)
            st.plotly_chart(fig_input, use_container_width=True)

    with col4:
        fig2 = plot_plotly_line(summary, 'concurrency_str', 'Max_System_Output_Throughput (tokens/s)', f"System Output Throughput (Decode) {io_label}", "Concurrency", "Tokens/s", model_id, rounds_per_level,
                               show_relative=False, provider=provider, line_color='#ff9f40') # Orange for Output/Decode Tput
        if fig2:
            fig2 = apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # --- Row 3: Request Throughput (QPM) ---
    # QPM chart to help users intuitively assess system capacity bottleneck
    # When QPM stops growing with concurrency, the system limit is reached
    fig_qpm = plot_plotly_line(summary, 'concurrency_str', 'Max_QPM (req/min)', f"System Processing Capacity (QPM Trend) {io_label}", "Concurrency", "Requests/min (QPM)", model_id, rounds_per_level,
                               show_relative=False, provider=provider, line_color='#ff9f40') # Orange for Request Metric
    if fig_qpm:
        fig_qpm = apply_theme(fig_qpm)
        st.plotly_chart(fig_qpm, use_container_width=True)

    # === Export Full Report (HTML) ===
    # Check for duplicate figures or None
    figs_list = [f for f in [fig1, fig3, fig_input, fig2, fig_qpm] if f is not None]

    # Tables
    from ui.export import create_html_download_link, export_interactive_html

    if figs_list:
        html_report = export_interactive_html(
            figures_list=figs_list,
            tables_list=[styled_table],
            insights_list=insights,
            title=f"Concurrency Performance Test Report - {model_id}"
        )

        with chart_col2:
            st.markdown(
                create_html_download_link(
                    html_report,
                    f"concurrency_report_full_{model_id}.html",
                    "🌐 Download Full HTML Report (All Charts)"
                ),
                unsafe_allow_html=True
            )

    report_md += "(See charts in Web UI)\n"
    return report_md

def generate_prefill_report(df_group, model_id, provider="Unknown", duration=0, test_config=None, system_info=None):
    st.subheader("Prefill Stress Test Charts")

    # Use standard summary card
    _render_test_summary_card(model_id, provider, duration, test_config, system_info, df=df_group, test_type='prefill')

    # Add Test Summary Section
    report_md = "# Prefill Stress Test Report\n\n"
    report_md += f"**Provider**: {provider} | **Model**: {model_id}\n"
    report_md += f"**Total Duration**: {duration:.2f}s\n\n"

    if test_config:
        report_md += "### Test Configuration\n"
        for k, v in test_config.items():
            report_md += f"- **{k}**: {v}\n"
        report_md += "\n"

    # Add Enhanced Summary
    report_md += _generate_markdown_summary(df_group, 'prefill', duration)

    report_md += "## Detailed Results\n\n"

    if 'input_tokens_target' not in df_group.columns:
        st.error("Prefill test report failed: missing 'input_tokens_target' column in data.")
        return ""
    group_copy = df_group.copy()
    if group_copy.empty:
        st.warning("Prefill test report: no valid Prefill data.")
        return ""

    requests_per_level = group_copy.groupby('input_tokens_target').size().max()

    idx = group_copy.groupby('input_tokens_target')['prefill_speed'].idxmax()
    summary_best = group_copy.loc[idx].copy().rename(columns={'prefill_speed': 'Best_Prefill_Speed', 'ttft': 'TTFT_at_Best_Speed'})

    if 'tpot' not in group_copy.columns:
        group_copy['tpot'] = 0

    stats = group_copy.groupby('input_tokens_target').agg(
        Num_Requests=('session_id', 'size'),
        Success_Rate=('error', lambda x: x.isnull().mean()),
        Token_Calc_Method=('token_calc_method', 'first'),
        Actual_Tokens_Mean=('prefill_tokens', 'mean'),
        Best_TTFT=('ttft', 'min'),
        Max_Prefill_Speed=('prefill_speed', 'max'),
        TPOT_Mean=('tpot', lambda x: x.mean() * 1000),
        TPOT_P95=('tpot', lambda x: x.quantile(0.95) * 1000),
        TPOT_P99=('tpot', lambda x: x.quantile(0.99) * 1000)
    ).reset_index().fillna(0)

    summary = pd.merge(stats, summary_best[['input_tokens_target', 'TTFT_at_Best_Speed']], on='input_tokens_target', how="left")

    summary['x_label'] = (summary['input_tokens_target'] / 1024).round(1).astype(str) + 'k'
    summary = summary.sort_values(by='input_tokens_target')

    st.subheader(f"Best Performance Stats ({requests_per_level} requests per level)")

    display_columns = [
        'input_tokens_target', 'Num_Requests', 'Success_Rate',
        'Best_TTFT',
        'Max_Prefill_Speed',
        'TPOT_Mean', 'TPOT_P95', 'TPOT_P99',
        'Actual_Tokens_Mean'
    ]
    display_columns = [col for col in display_columns if col in summary.columns]

    # Rename columns to include units
    summary = summary.rename(columns=COLUMN_RENAME_MAP)
    display_columns = [COLUMN_RENAME_MAP.get(col, col) for col in display_columns]

    # === Enhanced: Styled table ===
    styled_table = create_styled_summary_table(
        summary[display_columns].round(4),
        highlight_cols=[COLUMN_RENAME_MAP.get(c, c) for c in ['Best_TTFT', 'Max_Prefill_Speed']],
        highlight_best=True
    )
    st.dataframe(styled_table,
        column_config={
            **{k: st.column_config.Column(help=v) for k, v in COLUMN_TOOLTIPS.items()},
            "Best_TTFT (s)": st.column_config.ProgressColumn(
                "Best_TTFT (s)",
                help="TTFT - Time To First Token (lower is better)",
                format="%.4f s",
                min_value=0,
                max_value=summary['Best_TTFT (s)'].max() * 1.2 if not summary.empty else 1,
            ),
            "Max_Prefill_Speed (tokens/s)": st.column_config.ProgressColumn(
                "Prefill Speed",
                help="Prefill Speed (higher is better)",
                format="%d t/s",
                min_value=0,
                max_value=summary['Max_Prefill_Speed (tokens/s)'].max() * 1.1 if not summary.empty else 1000,
            ),
            "Success_Rate (%)": st.column_config.ProgressColumn(
                "Success Rate",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            )
        }
    )
    report_md += safe_to_markdown(summary[display_columns].round(4), index=False) + "\n\n"

    # === Enhanced: Performance insights ===
    insights = generate_performance_insights(summary, 'prefill', model_id)
    if insights:
        with st.expander("📊 Performance Insights", expanded=True):
            for insight in insights:
                st.markdown(insight)
            report_md += "\n### Performance Insights\n\n" + "\n".join([f"- {i}" for i in insights]) + "\n\n"

    hover_data_prefill = ['Actual_Tokens_Mean', 'input_tokens_target']

    # === Add Static Chart Download Button ===
    st.markdown("### 📈 Chart Analysis")
    chart_col1, chart_col2 = st.columns([1, 4])
    with chart_col1:
        # Generate static chart bytes
        # Check for column (renamed or original)
        prefill_col = 'Max_Prefill_Speed'
        if prefill_col not in summary.columns and f"{prefill_col} (tokens/s)" in summary.columns:
             prefill_col = f"{prefill_col} (tokens/s)"

        if 'input_tokens_target' in summary.columns and prefill_col in summary.columns:
            # Try to get system_info from session state
            system_info = st.session_state.get('system_info', None)
            static_fig_bytes = export_benchmark_summary_chart(
                summary, 'prefill', model_id, provider, system_info=system_info
            )
            if static_fig_bytes:
                st.markdown(
                    create_static_chart_download_link(
                        static_fig_bytes,
                        f"prefill_report_{model_id}.png",
                        "📷 Download Static Chart"
                    ),
                    unsafe_allow_html=True
                )

    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_plotly_line(summary, 'x_label', 'Best_TTFT (s)', "Time To First Token (TTFT)", "Input Tokens", "Time (s)", model_id, requests_per_level,
                               show_relative=False, force_linear_scale=True, hover_data=hover_data_prefill, provider=provider, line_color='#4bc0c0') # Teal
        if fig1:
            fig1 = apply_theme(fig1)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = plot_plotly_line(summary, 'x_label', 'Max_Prefill_Speed (tokens/s)', "Prefill Speed", "Input Tokens", "Tokens/s", model_id, requests_per_level,
                               show_relative=False, hover_data=hover_data_prefill, provider=provider, line_color='#4bc0c0') # Teal
        if fig2:
            fig2 = apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # === Export Full Report (HTML) ===
    figs_list = [f for f in [fig1, fig2] if f is not None]

    from ui.export import create_html_download_link, export_interactive_html

    if figs_list:
        html_report = export_interactive_html(
            figures_list=figs_list,
            tables_list=[styled_table],
            insights_list=insights,
            title=f"Prefill Performance Test Report - {model_id}"
        )

        with chart_col2:
            st.markdown(
                create_html_download_link(
                    html_report,
                    f"prefill_report_full_{model_id}.html",
                    "🌐 Download Full HTML Report (All Charts)"
                ),
                unsafe_allow_html=True
            )

    report_md += "(See charts in Web UI)\n"
    return report_md

def generate_long_context_report(df_group, model_id, provider="Unknown", duration=0, test_config=None, system_info=None):
    st.subheader("Long Context Test Charts")

    # Use standard summary card
    _render_test_summary_card(model_id, provider, duration, test_config, system_info, df=df_group, test_type='long_context')

    # Add Test Summary Section
    report_md = "# Long Context Test Report\n\n"
    report_md += f"**Provider**: {provider} | **Model**: {model_id}\n"
    report_md += f"**Total Duration**: {duration:.2f}s\n\n"

    if test_config:
        report_md += "### Test Configuration\n"
        for k, v in test_config.items():
            report_md += f"- **{k}**: {v}\n"
        report_md += "\n"

    # Add Enhanced Summary
    report_md += _generate_markdown_summary(df_group, 'long_context', duration)

    report_md += "## Detailed Results\n\n"

    if 'context_length_target' not in df_group.columns:
        st.error("Long context test report failed: missing 'context_length_target' column in data.")
        return ""
    group_copy = df_group.copy()
    if group_copy.empty:
        st.warning("Long context test report: no valid long context data.")
        return ""

    requests_per_level = group_copy.groupby('context_length_target').size().max()

    if 'tpot' not in group_copy.columns:
        group_copy['tpot'] = 0

    stats = group_copy.groupby('context_length_target').agg(
        Num_Requests=('session_id', 'size'),
        Success_Rate=('error', lambda x: x.isnull().mean()),
        Token_Calc_Method=('token_calc_method', 'first'),
        Actual_Tokens_Mean=('prefill_tokens', 'mean'),
        Actual_Decode_Max=('decode_tokens', 'max'), # Output tokens
        TPOT_Mean=('tpot', lambda x: x.mean() * 1000),
        TPOT_P95=('tpot', lambda x: x.quantile(0.95) * 1000),
        TPOT_P99=('tpot', lambda x: x.quantile(0.99) * 1000)
    ).reset_index().fillna(0)

    best_ttft_idx = group_copy.groupby('context_length_target')['ttft'].idxmin()
    summary_ttft = group_copy.loc[best_ttft_idx].copy()
    summary_ttft = summary_ttft.rename(columns={
        'ttft': 'Best_TTFT'
    })[['context_length_target', 'Best_TTFT']]

    best_prefill_idx = group_copy.groupby('context_length_target')['prefill_speed'].idxmax()
    summary_prefill = group_copy.loc[best_prefill_idx].copy()
    summary_prefill = summary_prefill.rename(columns={
        'prefill_speed': 'Max_Prefill_Speed'
    })[['context_length_target', 'Max_Prefill_Speed']]

    best_tps_idx = group_copy.groupby('context_length_target')['tps'].idxmax()
    summary_tps = group_copy.loc[best_tps_idx].copy()
    summary_tps = summary_tps.rename(columns={
        'tps': 'Max_TPS'
    })[['context_length_target', 'Max_TPS']]

    summary = pd.merge(stats, summary_ttft, on='context_length_target', how='left')
    summary = pd.merge(summary, summary_prefill, on='context_length_target', how='left')
    summary = pd.merge(stats, summary_ttft, on='context_length_target', how='left')
    summary = pd.merge(summary, summary_prefill, on='context_length_target', how='left')
    summary = pd.merge(summary, summary_tps, on='context_length_target', how='left')

    # New: Aggregate System Throughputs
    if 'system_input_throughput' in group_copy.columns:
        best_sys_input_idx = group_copy.groupby('context_length_target')['system_input_throughput'].idxmax()
        summary_sys_input = group_copy.loc[best_sys_input_idx].copy().rename(columns={'system_input_throughput': 'Max_System_Input_Throughput'})[['context_length_target', 'Max_System_Input_Throughput']]
        summary = pd.merge(summary, summary_sys_input, on='context_length_target', how='left')

    if 'system_output_throughput' in group_copy.columns:
        best_sys_output_idx = group_copy.groupby('context_length_target')['system_output_throughput'].idxmax()
        summary_sys_output = group_copy.loc[best_sys_output_idx].copy().rename(columns={'system_output_throughput': 'Max_System_Output_Throughput'})[['context_length_target', 'Max_System_Output_Throughput']]
        summary = pd.merge(summary, summary_sys_output, on='context_length_target', how='left')

    if 'system_throughput' in group_copy.columns:
        best_sys_total_idx = group_copy.groupby('context_length_target')['system_throughput'].idxmax()
        summary_sys_total = group_copy.loc[best_sys_total_idx].copy().rename(columns={'system_throughput': 'Max_System_Throughput'})[['context_length_target', 'Max_System_Throughput']]
        summary = pd.merge(summary, summary_sys_total, on='context_length_target', how='left')


    summary['x_label'] = (summary['context_length_target'] / 1024).round(1).astype(str) + 'k'


    st.subheader("Best Performance Stats (Peak)")

    display_columns = [
        'context_length_target', 'Num_Requests', 'Success_Rate',
        'Best_TTFT',
        'Max_Prefill_Speed',
        'Max_System_Input_Throughput', 'Max_System_Output_Throughput', 'Max_System_Throughput',
        'TPOT_Mean', 'TPOT_P95', 'TPOT_P99',
        'Actual_Tokens_Mean'
    ]
    display_columns = [col for col in display_columns if col in summary.columns]

    # Rename columns to include units
    summary = summary.rename(columns=COLUMN_RENAME_MAP)
    display_columns = [COLUMN_RENAME_MAP.get(col, col) for col in display_columns]

    # === Enhanced: Styled table ===
    styled_table = create_styled_summary_table(
        summary[display_columns].round(4),
        highlight_cols=[COLUMN_RENAME_MAP.get(c, c) for c in ['Best_TTFT', 'Max_Prefill_Speed', 'Max_System_Output_Throughput']],
        highlight_best=True
    )
    st.dataframe(styled_table,
        column_config={
            **{k: st.column_config.Column(help=v) for k, v in COLUMN_TOOLTIPS.items()},
             "Best_TTFT (s)": st.column_config.ProgressColumn(
                "Best_TTFT (s)",
                help="TTFT - Time To First Token (lower is better)",
                format="%.4f s",
                min_value=0,
                max_value=summary['Best_TTFT (s)'].max() * 1.2 if not summary.empty else 1,
            ),
            "Max_System_Input_Throughput (tokens/s)": st.column_config.ProgressColumn(
                "Max Input Tput",
                format="%d t/s",
                min_value=0,
                max_value=summary['Max_System_Input_Throughput (tokens/s)'].max() * 1.1 if not summary.empty and 'Max_System_Input_Throughput (tokens/s)' in summary else 100,
            ),
            "Max_System_Output_Throughput (tokens/s)": st.column_config.ProgressColumn(
                "Max Output Tput",
                format="%d t/s",
                min_value=0,
                max_value=summary['Max_System_Output_Throughput (tokens/s)'].max() * 1.1 if not summary.empty and 'Max_System_Output_Throughput (tokens/s)' in summary else 100,
            ),
             "Max_System_Throughput (tokens/s)": st.column_config.ProgressColumn(
                "Max Total Tput",
                format="%d t/s",
                min_value=0,
                max_value=summary['Max_System_Throughput (tokens/s)'].max() * 1.1 if not summary.empty and 'Max_System_Throughput (tokens/s)' in summary else 100,
            ),
            "Success_Rate (%)": st.column_config.ProgressColumn(
                "Success Rate",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            )
        }
    )
    report_md += safe_to_markdown(summary[display_columns].round(4), index=False) + "\n\n"

    # === Enhanced: Performance insights ===
    insights = generate_performance_insights(summary, 'long_context', model_id)
    if insights:
        with st.expander("📊 Performance Insights", expanded=True):
            for insight in insights:
                st.markdown(insight)
            report_md += "\n### Performance Insights\n\n" + "\n".join([f"- {i}" for i in insights]) + "\n\n"

    hover_data_ttft = ['Actual_Tokens_Mean']
    hover_data_prefill = ['Actual_Tokens_Mean']
    hover_data_decode = ['Actual_Tokens_Mean']

    # === Add Static Chart Download Button ===
    st.markdown("### 📈 Chart Analysis")
    chart_col1, chart_col2 = st.columns([1, 4])
    with chart_col1:
        # Generate static chart bytes
        # Try to get system_info from session state
        system_info = st.session_state.get('system_info', None)
        static_fig_bytes = export_benchmark_summary_chart(
            summary, 'long_context', model_id, provider, system_info=system_info
        )
        if static_fig_bytes:
            st.markdown(
                create_static_chart_download_link(
                    static_fig_bytes,
                    f"long_context_report_{model_id}.png",
                    "📷 Download Static Chart"
                ),
                unsafe_allow_html=True
            )

    # Determine Output Metrics Label
    max_out_tokens = int(summary['Actual_Decode_Max'].max()) if 'Actual_Decode_Max' in summary.columns else 0
    out_label = f"(Output: ~{max_out_tokens})"

    # --- Row 1: Latency Metrics (TTFT & TPOT) ---
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_plotly_line(summary, 'x_label', 'Best_TTFT (s)', "Time To First Token (TTFT)", "Context Length", "Time (s)", model_id, requests_per_level,
                                hover_data=hover_data_ttft, show_relative=False, force_linear_scale=True, provider=provider, line_color='#4bc0c0') # Teal
        if fig1:
            fig1 = apply_theme(fig1)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig_tpot = plot_plotly_line(summary, 'x_label', 'TPOT_Mean (ms)', f"Avg Time Per Output Token (TPOT) {out_label}", "Context Length", "Time (ms)", model_id, requests_per_level,
                                    show_relative=False, provider=provider, line_color='#ff9f40') # Orange
        if fig_tpot:
            fig_tpot = apply_theme(fig_tpot)
            st.plotly_chart(fig_tpot, use_container_width=True)

    # --- Row 2: Phase Throughputs (Input & Output) ---
    col3, col4 = st.columns(2)
    with col3:
        fig3 = plot_plotly_line(summary, 'x_label', 'Max_System_Input_Throughput (tokens/s)', "Sys Input Throughput", "Context Length", "Tokens/s", model_id, requests_per_level,
                                hover_data=hover_data_prefill, show_relative=False, provider=provider, line_color='#4bc0c0') # Teal
        if fig3:
            fig3 = apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig2 = plot_plotly_line(summary, 'x_label', 'Max_System_Output_Throughput (tokens/s)', f"Sys Output Throughput {out_label}", "Context Length", "Tokens/s", model_id, requests_per_level,
                                hover_data=hover_data_decode, show_relative=False, provider=provider, line_color='#ff9f40') # Orange
        if fig2:
            fig2 = apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # --- Row 3: Total Throughput ---
    # User requested 'Sys Total Throughput' at the bottom.
    fig4 = plot_plotly_line(summary, 'x_label', 'Max_System_Throughput (tokens/s)', f"Sys Total Throughput {out_label}", "Context Length", "Tokens/s", model_id, requests_per_level,
                            show_relative=False, provider=provider, line_color='#ff9f40') # Orange for Total/Output
    if fig4:
        fig4 = apply_theme(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # === Export Full Report (HTML) ===
    figs_list = [f for f in [fig1, fig_tpot, fig3, fig2, fig4] if f is not None]

    from ui.export import create_html_download_link, export_interactive_html

    if figs_list:
        html_report = export_interactive_html(
            figures_list=figs_list,
            tables_list=[styled_table],
            insights_list=insights,
            title=f"Long Context Performance Test Report - {model_id}"
        )

        with chart_col2:
            st.markdown(
                create_html_download_link(
                    html_report,
                    f"long_context_report_full_{model_id}.html",
                    "🌐 Download Full HTML Report (All Charts)"
                ),
                unsafe_allow_html=True
            )

    report_md += "(See charts in Web UI)\n"
    return report_md

def generate_matrix_report(df_group, model_id, provider="Unknown", duration=0, test_config=None, system_info=None):
    st.subheader("Concurrency-Context Matrix Test Charts")

    # Use standard summary card
    _render_test_summary_card(model_id, provider, duration, test_config, system_info, df=df_group, test_type='matrix')

    # Add Test Summary Section
    report_md = "# Concurrency-Context Matrix Test Report\n\n"
    report_md += f"**Provider**: {provider} | **Model**: {model_id}\n"
    report_md += f"**Total Duration**: {duration:.2f}s\n\n"

    if test_config:
        report_md += "### Test Configuration\n"
        for k, v in test_config.items():
            report_md += f"- **{k}**: {v}\n"
        report_md += "\n"

    # Add Enhanced Summary
    report_md += _generate_markdown_summary(df_group, 'matrix', duration)

    report_md += "## Detailed Results\n\n"

    if 'context_length_target' not in df_group.columns or 'concurrency' not in df_group.columns:
        st.error("Matrix test report failed: missing 'context_length_target' or 'concurrency' column.")
        return ""
    group_copy = df_group.copy()
    if group_copy.empty:
        st.warning("Matrix test report: no valid matrix test data.")
        return ""

    if 'system_output_throughput' not in group_copy.columns:
        # Backward compatibility for old columns or if update missed
        if 'system_throughput' in group_copy.columns:
             group_copy['system_output_throughput'] = group_copy['system_throughput']
        else:
             group_copy['system_output_throughput'] = 0

    if 'tpot' not in group_copy.columns:
        group_copy['tpot'] = 0

    stats = group_copy.groupby(['context_length_target', 'concurrency']).agg(
        Num_Requests=('session_id', 'size'),
        Success_Rate=('error', lambda x: x.isnull().mean()),
        Token_Calc_Method=('token_calc_method', 'first'),
        Actual_Tokens_Mean=('prefill_tokens', 'mean'),
        Actual_Decode_Max=('decode_tokens', 'max'), # Capture max output tokens
        TPOT_Mean=('tpot', lambda x: x.mean() * 1000),
        TPOT_P95=('tpot', lambda x: x.quantile(0.95) * 1000),
        TPOT_P99=('tpot', lambda x: x.quantile(0.99) * 1000)
    ).reset_index().fillna(0)

    best_ttft_idx = group_copy.groupby(['context_length_target', 'concurrency'])['ttft'].idxmin()
    summary_ttft = group_copy.loc[best_ttft_idx].copy()
    summary_ttft = summary_ttft.rename(columns={
        'ttft': 'Best_TTFT'
    })[['context_length_target', 'concurrency', 'Best_TTFT']]

    best_prefill_idx = group_copy.groupby(['context_length_target', 'concurrency'])['prefill_speed'].idxmax()
    summary_prefill = group_copy.loc[best_prefill_idx].copy()
    summary_prefill = summary_prefill.rename(columns={
        'prefill_speed': 'Max_Prefill_Speed'
    })[['context_length_target', 'concurrency', 'Max_Prefill_Speed']]

    best_sys_tps_idx = group_copy.groupby(['context_length_target', 'concurrency'])['system_output_throughput'].idxmax()
    summary_sys_tps = group_copy.loc[best_sys_tps_idx].copy()

    # Ensure new columns exist for backward compatibility
    if 'system_input_throughput' not in summary_sys_tps.columns:
        summary_sys_tps['system_input_throughput'] = 0
    if 'rps' not in summary_sys_tps.columns:
        summary_sys_tps['rps'] = 0

    if 'system_throughput' not in summary_sys_tps.columns:
        summary_sys_tps['system_throughput'] = 0

    summary_sys_tps = summary_sys_tps.rename(columns={
        'system_output_throughput': 'Max_System_Output_Throughput',
        'system_input_throughput': 'Max_System_Input_Throughput',
        'system_throughput': 'Max_System_Throughput',
        'rps': 'Max_RPS'
    })[['context_length_target', 'concurrency', 'Max_System_Output_Throughput', 'Max_System_Input_Throughput', 'Max_System_Throughput', 'Max_RPS']]

    # Calculate Max Single TPS
    if 'tps' in group_copy.columns:
        best_tps_idx = group_copy.groupby(['context_length_target', 'concurrency'])['tps'].idxmax()
        summary_tps = group_copy.loc[best_tps_idx].copy()
        summary_tps = summary_tps.rename(columns={'tps': 'Max_Single_TPS'})[['context_length_target', 'concurrency', 'Max_Single_TPS']]
    else:
        # Fallback if tps is missing
        summary_tps = pd.DataFrame(columns=['context_length_target', 'concurrency', 'Max_Single_TPS'])

    summary = pd.merge(stats, summary_ttft, on=['context_length_target', 'concurrency'], how='left')
    summary = pd.merge(summary, summary_prefill, on=['context_length_target', 'concurrency'], how='left')
    summary = pd.merge(summary, summary_sys_tps, on=['context_length_target', 'concurrency'], how='left')
    if not summary_tps.empty:
        summary = pd.merge(summary, summary_tps, on=['context_length_target', 'concurrency'], how='left')
    else:
        summary['Max_Single_TPS'] = 0

    summary['x_label'] = (summary['context_length_target'] / 1024).round(1).astype(str) + 'k'

    summary = summary[summary['concurrency'] > 0].copy()
    summary['concurrency'] = summary['concurrency'].astype(int)

    summary = summary.sort_values(by=['context_length_target', 'concurrency'])

    # === Enhanced: Performance insights (Generate before renaming) ===
    insights = generate_performance_insights(summary, 'matrix', model_id)

    # Rename columns to include units for DISPLAY ONLY
    # We keep 'summary' with raw column names for Plotly charts below
    summary_display = summary.rename(columns=COLUMN_RENAME_MAP)

    # Determine Output Metrics Label
    # Use max observed output tokens as the label
    max_out_tokens = int(summary['Actual_Decode_Max'].max()) if 'Actual_Decode_Max' in summary.columns else 0
    out_label = f"(Output: ~{max_out_tokens})"

    st.subheader("Best Performance Stats (Peak)")

    # === Enhanced: Styled table ===
    styled_table = create_styled_summary_table(
        summary_display,
        highlight_cols=[COLUMN_RENAME_MAP.get(c, c) for c in ['Best_TTFT', 'Max_System_Output_Throughput', 'Max_System_Input_Throughput', 'Max_RPS', 'Max_Prefill_Speed']],
        highlight_best=True
    )
    st.dataframe(styled_table,
        column_config={
            **{k: st.column_config.Column(help=v) for k, v in COLUMN_TOOLTIPS.items()},
            "Best_TTFT (s)": st.column_config.ProgressColumn(
                "Best_TTFT (s)",
                format="%.4f s",
                min_value=0,
                max_value=summary_display['Best_TTFT (s)'].max() * 1.2 if not summary_display.empty else 1,
            ),
            "Max_System_Output_Throughput (tokens/s)": st.column_config.ProgressColumn(
                "Max_Sys_Output_TPS",
                format="%d t/s",
                min_value=0,
                max_value=summary_display['Max_System_Output_Throughput (tokens/s)'].max() * 1.1 if not summary_display.empty else 100,
            ),
             "Success_Rate (%)": st.column_config.ProgressColumn(
                "Success Rate",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            )
        }
    )
    report_md += safe_to_markdown(summary_display, index=False) + "\n\n"

    # === Enhanced: Performance insights ===
    # Insights generated above before renaming
    if insights:
        with st.expander("📊 Comprehensive Performance Insights", expanded=True):
            grade, color, description = get_performance_grade(insights)
            st.markdown(f"**Grade**: {grade} ({description})")
            st.markdown("---")
            for insight in insights:
                st.markdown(insight)
            report_md += "\n### Performance Insights\n\n" + "\n".join([f"- {i}" for i in insights]) + "\n\n"

    # Previous duplicate charts and heatmap removed

    # === Add Static Chart Download Button ===
    st.markdown("### 📈 Chart Analysis")
    chart_col1, chart_col2 = st.columns([1, 4])
    with chart_col1:
        # Generate static chart bytes
        # Try to get system_info from session state
        system_info = st.session_state.get('system_info', None)
        static_fig_bytes = export_benchmark_summary_chart(
            summary, 'matrix', model_id, provider, system_info=system_info
        )
        if static_fig_bytes:
            st.markdown(
                create_static_chart_download_link(
                    static_fig_bytes,
                    f"matrix_report_{model_id}.png",
                    "📷 Download Static Chart"
                ),
                unsafe_allow_html=True
            )
    hover_data_ttft = ['Actual_Tokens_Mean', 'concurrency']
    # --- Row 1: Latency Metrics (TTFT & TPOT) ---
    # --- Row 1: Latency Metrics (TTFT & TPOT) ---
    col1, col2 = st.columns(2)
    with col1:
        hover_data_ttft = ['Actual_Tokens_Mean', 'concurrency']
        # TTFT is Input-related -> Teal
        fig1 = plot_plotly_line(summary, 'x_label', 'Best_TTFT', "Time To First Token (TTFT)", "Context Length", "Time (s)", model_id, "N/A",
                                color='concurrency',
                                hover_data=hover_data_ttft,
                                show_relative=False, force_linear_scale=True, provider=provider, line_color='#4bc0c0')
        if fig1:
            fig1 = apply_theme(fig1)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        hover_data_tpot = ['Actual_Tokens_Mean', 'concurrency']
        # TPOT is Output-related -> Orange
        fig_tpot = plot_plotly_line(summary, 'x_label', 'TPOT_Mean', f"Avg Time Per Output Token (TPOT) {out_label}", "Context Length", "Time (ms)", model_id, "N/A",
                                    color='concurrency',
                                    hover_data=hover_data_tpot,
                                    show_relative=False, provider=provider, line_color='#ff9f40')
        if fig_tpot:
            fig_tpot = apply_theme(fig_tpot)
            st.plotly_chart(fig_tpot, use_container_width=True)

    # --- Row 2: Phase Throughputs (Input & Output) ---
    col3, col4 = st.columns(2)
    with col3:
        hover_data_prefill = ['Actual_Tokens_Mean', 'concurrency']
        # Input Throughput -> Teal
        fig3 = plot_plotly_line(summary, 'x_label', 'Max_System_Input_Throughput', "Sys Input Throughput", "Context Length", "Tokens/s", model_id, "N/A",
                                color='concurrency',
                                hover_data=hover_data_prefill,
                                show_relative=False, provider=provider, line_color='#4bc0c0')
        if fig3:
            fig3 = apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        hover_data_decode = ['Actual_Tokens_Mean', 'concurrency']
        # Output Throughput -> Orange
        fig2 = plot_plotly_line(summary, 'x_label', 'Max_System_Output_Throughput', f"Sys Output Throughput {out_label}", "Context Length", "Tokens/s", model_id, "N/A",
                                color='concurrency',
                                hover_data=hover_data_decode,
                                show_relative=False, provider=provider, line_color='#ff9f40')
        if fig2:
            fig2 = apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # Calculate QPM based on RPS
    # ... (QPM calculation code remains, implicit) ...
    if 'Max_RPS' in summary.columns:
         summary['Max_QPM'] = summary['Max_RPS'] * 60
    else:
         summary['Max_QPM'] = 0

    # --- Row 3: Total Throughput & QPM ---

    col5, col6 = st.columns(2)
    with col5:
        hover_data_total = ['Actual_Tokens_Mean', 'concurrency']
        # Total Throughput -> Orange
        fig4 = plot_plotly_line(summary, 'x_label', 'Max_System_Throughput', f"Sys Total Throughput {out_label}", "Context Length", "Tokens/s", model_id, "N/A",
                                color='concurrency',
                                hover_data=hover_data_total,
                                show_relative=False, provider=provider, line_color='#ff9f40')
        if fig4:
            fig4 = apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

    with col6:
        hover_data_qpm = ['Max_RPS', 'concurrency']
        # QPM -> Orange (Output capacity)
        fig_qpm = plot_plotly_line(summary, 'x_label', 'Max_QPM', f"System QPM (by Input Length) {out_label}", "Context Length", "Req/min (QPM)", model_id, "N/A",
                                   color='concurrency',
                                   hover_data=hover_data_qpm,
                                   show_relative=False, provider=provider, line_color='#ff9f40')
    if fig_qpm:
        fig_qpm = apply_theme(fig_qpm)
        st.plotly_chart(fig_qpm, use_container_width=True)

    # === Export Full Report (HTML) ===
    figs_list = [f for f in [fig1, fig_tpot, fig3, fig2, fig4, fig_qpm] if f is not None]

    from ui.export import create_html_download_link, export_interactive_html

    if figs_list:
        html_report = export_interactive_html(
            figures_list=figs_list,
            tables_list=[styled_table],
            insights_list=insights,
            title=f"Concurrency-Context Matrix Test Report - {model_id}"
        )

        with chart_col2:
            st.markdown(
                create_html_download_link(
                    html_report,
                    f"matrix_report_full_{model_id}.html",
                    "🌐 Download Full HTML Report (All Charts)"
                ),
                unsafe_allow_html=True
            )

    return report_md


def generate_segmented_report(df_group, model_id, provider="Unknown", duration=0, test_config=None, system_info=None):
    """Segmented Context Test (Prefix Caching) Report"""
    st.subheader("Segmented Context Test Charts (Prefix Caching)")

    # Use standard summary card
    _render_test_summary_card(model_id, provider, duration, test_config, system_info, df=df_group, test_type='segmented')

    # Add Test Summary Section
    report_md = "# Segmented Context Test Report (Prefix Caching)\n\n"
    report_md += f"**Provider**: {provider} | **Model**: {model_id}\n"
    report_md += f"**Total Duration**: {duration:.2f}s\n\n"

    if test_config:
        report_md += "### Test Configuration\n"
        for k, v in test_config.items():
            report_md += f"- **{k}**: {v}\n"
        report_md += "\n"

    # Add Enhanced Summary
    report_md += _generate_markdown_summary(df_group, 'segmented', duration)

    report_md += "## Detailed Results\n\n"


    if 'context_length_target' not in df_group.columns:
        st.error("Segmented context test report failed: missing 'context_length_target' column in data.")
        return ""
    group_copy = df_group.copy()
    if group_copy.empty:
        st.warning("Segmented context test report: no valid test data.")
        return ""

    # Ensure required columns exist with fallbacks
    if 'prefill_speed' not in group_copy.columns:
        group_copy['prefill_speed'] = 0
    if 'tpot' not in group_copy.columns:
        group_copy['tpot'] = 0
    if 'cache_hit_tokens' not in group_copy.columns:
        group_copy['cache_hit_tokens'] = 0
    if 'tps' not in group_copy.columns:
        group_copy['tps'] = 0
    if 'rps' not in group_copy.columns:
        group_copy['rps'] = 0
    if 'system_input_throughput' not in group_copy.columns:
        group_copy['system_input_throughput'] = 0
    if 'system_output_throughput' not in group_copy.columns:
        group_copy['system_output_throughput'] = 0
    if 'system_total_throughput' not in group_copy.columns:
        if 'system_throughput' in group_copy.columns:
            group_copy['system_total_throughput'] = group_copy['system_throughput']
        else:
            group_copy['system_total_throughput'] = 0

    # Use effective tokens (API priority) for stats, fallback to tokenizer values
    if 'effective_prefill_tokens' in group_copy.columns:
        # When effective column exists, fill missing values with tokenizer values
        prefill_col = 'effective_prefill_tokens'
        group_copy[prefill_col] = group_copy[prefill_col].fillna(group_copy['prefill_tokens'])
        group_copy.loc[group_copy[prefill_col] == 0, prefill_col] = group_copy.loc[group_copy[prefill_col] == 0, 'prefill_tokens']
    else:
        prefill_col = 'prefill_tokens'

    if 'effective_decode_tokens' in group_copy.columns:
        decode_col = 'effective_decode_tokens'
        group_copy[decode_col] = group_copy[decode_col].fillna(group_copy['decode_tokens'])
        group_copy.loc[group_copy[decode_col] == 0, decode_col] = group_copy.loc[group_copy[decode_col] == 0, 'decode_tokens']
    else:
        decode_col = 'decode_tokens'

    requests_per_level = group_copy.groupby('context_length_target').size().max()

    # Check if we have cumulative_mode data for comparative analysis
    has_mode_col = 'cumulative_mode' in group_copy.columns

    # Build summary statistics per segment level
    agg_dict = {
        'Num_Requests': ('session_id', 'size'),
        'Success_Rate': ('error', lambda x: x.isnull().mean()),
        'Token_Calc_Method': ('token_calc_method', 'first'),
        'Actual_Tokens_Mean': (prefill_col, 'mean'),
        'Actual_Decode_Mean': (decode_col, 'mean'),
        'Uncached_TTFT': ('ttft', 'max'),
        'Cached_TTFT': ('ttft', 'min'),
        'TTFT_Mean': ('ttft', 'mean'),
        'Max_Prefill_Speed': ('prefill_speed', 'max'),
        'Prefill_Speed_Mean': ('prefill_speed', 'mean'),
        'Cache_Hit_Mean': ('cache_hit_tokens', 'mean'),
        'Cache_Hit_Max': ('cache_hit_tokens', 'max'),
        'Max_TPS': ('tps', 'max'),
        'TPOT_Mean': ('tpot', lambda x: x[x > 0].mean() * 1000 if (x > 0).any() else 0),
        'Max_System_Input_Throughput': ('system_input_throughput', 'max'),
        'Max_System_Output_Throughput': ('system_output_throughput', 'max'),
        'Max_System_Total_Throughput': ('system_total_throughput', 'max'),
    }

    # Add token_source stats (if available)
    if 'token_source' in group_copy.columns:
        agg_dict['Token_Source'] = ('token_source', 'first')

    stats = group_copy.groupby('context_length_target').agg(**agg_dict).reset_index().fillna(0)

    stats['x_label'] = (stats['context_length_target'] / 1024).round(1).astype(str) + 'k'
    stats = stats.sort_values(by='context_length_target')

    # Calculate Cache Hit Rate
    stats['Cache_Hit_Rate'] = 0.0
    mask = stats['Actual_Tokens_Mean'] > 0
    stats.loc[mask, 'Cache_Hit_Rate'] = (
        stats.loc[mask, 'Cache_Hit_Mean'] / stats.loc[mask, 'Actual_Tokens_Mean'] * 100
    ).round(1)

    st.subheader(f"Best Performance Stats ({requests_per_level} requests per level)")

    display_columns = [
        'context_length_target', 'Num_Requests', 'Success_Rate',
        'Uncached_TTFT', 'Cached_TTFT', 'TTFT_Mean',
        'Max_Prefill_Speed',
        'Cache_Hit_Mean', 'Cache_Hit_Rate',
        'Max_System_Input_Throughput', 'Max_System_Output_Throughput',
        'TPOT_Mean',
        'Actual_Tokens_Mean'
    ]
    display_columns = [col for col in display_columns if col in stats.columns]

    # Rename columns to include units
    seg_rename_map = {
        'Uncached_TTFT': 'Uncached_TTFT (s)',
        'Cached_TTFT': 'Cached_TTFT (s)',
        'TTFT_Mean': 'TTFT_Mean (s)',
        'Max_Prefill_Speed': 'Max_Prefill_Speed (tokens/s)',
        'Cache_Hit_Mean': 'Cache_Hit_Mean (tokens)',
        'Cache_Hit_Rate': 'Cache_Hit_Rate (%)',
        'Max_System_Input_Throughput': 'Max_System_Input_Throughput (tokens/s)',
        'Max_System_Output_Throughput': 'Max_System_Output_Throughput (tokens/s)',
        'TPOT_Mean': 'TPOT_Mean (ms)',
        'Success_Rate': 'Success_Rate (%)',
    }
    stats = stats.rename(columns=seg_rename_map)
    display_columns = [seg_rename_map.get(col, col) for col in display_columns]

    # === Enhanced: Styled table ===
    styled_table = create_styled_summary_table(
        stats[display_columns].round(4),
        highlight_cols=[seg_rename_map.get(c, c) for c in ['Uncached_TTFT', 'Max_Prefill_Speed', 'Cache_Hit_Rate']],
        highlight_best=True
    )

    seg_tooltips = {
        **COLUMN_TOOLTIPS,
        "Cache_Hit_Mean (tokens)": "Average Cache Hit Tokens: Mean of cached_tokens returned by API. Higher values indicate better Prefix Caching.",
        "Cache_Hit_Rate (%)": "Cache Hit Rate: Cache Hit Tokens / Input Tokens. Higher values indicate better Prefix Caching.",
        "TTFT_Mean (s)": "Average TTFT: Mean TTFT across all requests.",
        "Uncached_TTFT (s)": "Uncached TTFT: Maximum TTFT among requests at this segment, effectively the first request (cache miss).",
        "Cached_TTFT (s)": "Cached TTFT: Minimum TTFT among requests at this segment, effectively the fully cached requests.",
    }

    st.dataframe(styled_table,
        column_config={
            **{k: st.column_config.Column(help=v) for k, v in seg_tooltips.items()},
            "Uncached_TTFT (s)": st.column_config.ProgressColumn(
                "Uncached_TTFT (s)",
                help="Uncached TTFT - Non-cache-hit time (lower is better)",
                format="%.4f s",
                min_value=0,
                max_value=stats['Uncached_TTFT (s)'].max() * 1.2 if not stats.empty else 1,
            ),
            "Cached_TTFT (s)": st.column_config.NumberColumn(
                "Cached_TTFT (s)",
                help="Cached TTFT - Time after cache-hit",
                format="%.4f s"
            ),
            "Max_Prefill_Speed (tokens/s)": st.column_config.ProgressColumn(
                "Prefill Speed",
                help="Prefill Speed (higher is better)",
                format="%d t/s",
                min_value=0,
                max_value=stats['Max_Prefill_Speed (tokens/s)'].max() * 1.1 if not stats.empty else 1000,
            ),
            "Cache_Hit_Rate (%)": st.column_config.ProgressColumn(
                "Cache Hit Rate",
                help="Cache Hit Rate (higher indicates better Prefix Caching)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "Success_Rate (%)": st.column_config.ProgressColumn(
                "Success Rate",
                format="%.1f%%",
                min_value=0,
                max_value=1,
            )
        }
    )
    report_md += safe_to_markdown(stats[display_columns].round(4), index=False) + "\n\n"

    # === Performance insights ===
    insights = generate_performance_insights(stats, 'segmented', model_id)
    if not stats.empty:
        best_ttft_row = stats.loc[stats['Uncached_TTFT (s)'].idxmin()]
        insights.append(f"🏆 **Best Uncached TTFT**: `{best_ttft_row['Uncached_TTFT (s)']:.4f}s` (segment: {best_ttft_row['context_length_target']})")

        best_prefill_row = stats.loc[stats['Max_Prefill_Speed (tokens/s)'].idxmax()]
        insights.append(f"⚡ **Peak Prefill Speed**: `{best_prefill_row['Max_Prefill_Speed (tokens/s)']:.0f} tokens/s` (segment: {best_prefill_row['context_length_target']})")

        if 'Cache_Hit_Rate (%)' in stats.columns:
            max_cache_rate = stats['Cache_Hit_Rate (%)'].max()
            if max_cache_rate > 0:
                best_cache_row = stats.loc[stats['Cache_Hit_Rate (%)'].idxmax()]
                insights.append(f"💾 **Peak Cache Hit Rate**: `{max_cache_rate:.1f}%` (segment: {best_cache_row['context_length_target']})")

                # Check for increasing cache hit rate pattern (Prefix Caching working)
                if len(stats) >= 2:
                    first_rate = stats.iloc[0]['Cache_Hit_Rate (%)']
                    last_rate = stats.iloc[-1]['Cache_Hit_Rate (%)']
                    if last_rate > first_rate + 5:
                        insights.append("📈 **Cache Effect Increasing**: Cache hit rate grows with cumulative segments, indicating Prefix Caching is working effectively.")
            else:
                insights.append("⚠️ **No Cache Hits**: No Prefix Caching effect detected. Please verify if the API supports this feature.")

        # TTFT degradation check
        if len(stats) >= 2:
            ttft_first = stats.iloc[0]['Uncached_TTFT (s)']
            ttft_last = stats.iloc[-1]['Uncached_TTFT (s)']
            if ttft_last > ttft_first * 3 and ttft_first > 0:
                ratio = ttft_last / ttft_first
                insights.append(f"⚠️ **Significant TTFT Growth**: From {ttft_first:.4f}s to {ttft_last:.4f}s ({ratio:.1f}x increase). Linear TTFT growth with input length is expected behavior.")

    if insights:
        with st.expander("📊 Performance Insights & Analysis", expanded=True):
            for insight in insights:
                st.markdown(insight)
            report_md += "\n### Performance Insights\n\n" + "\n".join([f"- {i}" for i in insights]) + "\n\n"

    # === Charts ===
    st.markdown("### 📈 Chart Analysis")
    chart_col1, chart_col2 = st.columns([1, 4])
    with chart_col1:
        # Static chart download
        system_info = st.session_state.get('system_info', None)
        static_fig_bytes = export_benchmark_summary_chart(
            stats, 'segmented', model_id, provider, system_info=system_info
        )
        if static_fig_bytes:
            st.markdown(
                create_static_chart_download_link(
                    static_fig_bytes,
                    f"segmented_report_{model_id}.png",
                    "📷 Download Static Chart"
                ),
                unsafe_allow_html=True
            )

    hover_data_seg = ['Actual_Tokens_Mean', 'context_length_target']

    # --- Row 1: TTFT & Prefill Speed ---
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_plotly_line(
            stats, 'x_label', 'Uncached_TTFT (s)', "Time To First Token (Uncached) by Segment",
            "Context Segment", "Time (s)", model_id, requests_per_level,
            show_relative=False, force_linear_scale=True,
            hover_data=hover_data_seg, provider=provider, line_color='#4bc0c0'
        )
        if fig1:
            fig1 = apply_theme(fig1)
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = plot_plotly_line(
            stats, 'x_label', 'Max_Prefill_Speed (tokens/s)', "Prefill Speed",
            "Context Segment", "Tokens/s", model_id, requests_per_level,
            show_relative=False,
            hover_data=hover_data_seg, provider=provider, line_color='#4bc0c0'
        )
        if fig2:
            fig2 = apply_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    # --- Row 2: Cache Hit & System Throughput ---
    col3, col4 = st.columns(2)
    with col3:
        # Cache Hit Rate chart
        if stats['Cache_Hit_Rate (%)'].max() > 0:
            fig3 = plot_plotly_line(
                stats, 'x_label', 'Cache_Hit_Rate (%)', "Cache Hit Rate",
                "Context Segment", "Cache Hit Rate (%)", model_id, requests_per_level,
                show_relative=False,
                hover_data=hover_data_seg, provider=provider, line_color='#36a2eb'
            )
        else:
            # Show Cache Hit Mean tokens instead
            fig3 = plot_plotly_line(
                stats, 'x_label', 'Cache_Hit_Mean (tokens)', "Cache Hit Tokens",
                "Context Segment", "Tokens", model_id, requests_per_level,
                show_relative=False,
                hover_data=hover_data_seg, provider=provider, line_color='#36a2eb'
            )
        if fig3:
            fig3 = apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = plot_plotly_line(
            stats, 'x_label', 'Max_System_Input_Throughput (tokens/s)',
            "System Input Throughput",
            "Context Segment", "Tokens/s", model_id, requests_per_level,
            show_relative=False,
            hover_data=hover_data_seg, provider=provider, line_color='#ff9f40'
        )
        if fig4:
            fig4 = apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

    # --- Row 3: TPOT ---
    if 'TPOT_Mean (ms)' in stats.columns and stats['TPOT_Mean (ms)'].max() > 0:
        fig5 = plot_plotly_line(
            stats, 'x_label', 'TPOT_Mean (ms)', "Avg Time Per Output Token (TPOT)",
            "Context Segment", "Time (ms)", model_id, requests_per_level,
            show_relative=False,
            provider=provider, line_color='#ff9f40'
        )
        if fig5:
            fig5 = apply_theme(fig5)
            st.plotly_chart(fig5, use_container_width=True)
    else:
        fig5 = None

    # === Export Full Report (HTML) ===
    figs_list = [f for f in [fig1, fig2, fig3, fig4, fig5] if f is not None]

    from ui.export import create_html_download_link, export_interactive_html

    if figs_list:
        html_report = export_interactive_html(
            figures_list=figs_list,
            tables_list=[styled_table],
            insights_list=insights,
            title=f"Segmented Context Test Report (Prefix Caching) - {model_id}"
        )

        with chart_col2:
            st.markdown(
                create_html_download_link(
                    html_report,
                    f"segmented_report_full_{model_id}.html",
                    "🌐 Download Full HTML Report (All Charts)"
                ),
                unsafe_allow_html=True
            )

    report_md += "(See charts in Web UI)\n"
    return report_md

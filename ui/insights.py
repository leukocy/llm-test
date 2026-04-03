"""
Performance insights generation for benchmark results.

Automatically analyzes test results and generates actionable insights with statistical depth.
Includes data quality validation to avoid misleading conclusions from invalid/zero metrics.

Insight emoji convention:
  ✅ 🚀 🏆 📈 ⚡  → Strong positive (no action needed)
  📊 ⚖️ 🎯        → Neutral / informational (facts, moderate results)
  ⚠️ 🐢 📉 🎢     → Warning (optimization recommended)
  ❌ 🛑            → Critical (requires immediate attention)
"""
from scipy import stats
import pandas as pd
import numpy as np


def _check_data_quality(df, test_type):
    """
    Check if the test data has meaningful metrics before analysis.
    Returns a list of warning insights if data is poor, or empty list if OK.
    """
    warnings = []

    # Check if key token metrics are all zero or missing
    # Support both raw data columns and aggregated summary columns
    total_input = 0
    for col in ['prefill_tokens', 'Actual_Tokens_Mean']:
        if col in df.columns:
            total_input = df[col].sum()
            break
    total_output = 0
    for col in ['decode_tokens', 'Actual_Decode_Max']:
        if col in df.columns:
            total_output = df[col].sum()
            break

    if total_input == 0 and total_output == 0:
        warnings.append("❌ **Data Anomaly**: Input/output token counts are both 0. The test may not have correctly collected token statistics. Please check if the API returns usage info.")

    # Check if TTFT data is available
    if 'ttft' in df.columns:
        valid_ttft = df[df['ttft'] > 0]['ttft']
        if valid_ttft.empty or len(valid_ttft) == 0:
            warnings.append("❌ **TTFT Missing**: All requests have TTFT of 0 or missing. Cannot perform latency analysis.")

    # Check if TPS data is available
    if 'tps' in df.columns:
        valid_tps = df[df['tps'] > 0]['tps']
        if valid_tps.empty or len(valid_tps) == 0:
            warnings.append("❌ **TPS Missing**: All requests have TPS of 0 or missing. Cannot perform throughput analysis.")

    return warnings


def generate_performance_insights(df, test_type, model_id=''):
    """
    Generate performance insights from benchmark results.

    Args:
        df: Results DataFrame
        test_type: Type of test ('concurrency', 'prefill', 'long_context', 'matrix', 'segmented')
        model_id: Model identifier

    Returns:
        List of insight strings (markdown formatted)
    """
    insights = []
    
    # Common check
    if df is None or df.empty:
        return ["⚠️ **Insufficient Data**: Cannot generate valid insights."]

    # === Data Quality Gate ===
    quality_warnings = _check_data_quality(df, test_type)
    if quality_warnings:
        insights.extend(quality_warnings)
        # If there are critical data issues (❌), skip further analysis to avoid misleading insights
        has_critical = any("❌" in w for w in quality_warnings)
        if has_critical:
            insights.append("⚠️ **Analysis Skipped**: Due to missing core metrics, further analysis may be misleading and has been skipped. Please investigate data collection issues and re-test.")
            return insights

    if test_type == 'concurrency':
        insights.extend(_analyze_concurrency(df, model_id))
    elif test_type == 'prefill':
        insights.extend(_analyze_prefill(df, model_id))
    elif test_type == 'long_context':
        insights.extend(_analyze_long_context(df, model_id))
    elif test_type == 'matrix':
        insights.extend(_analyze_matrix(df, model_id))
    elif test_type == 'segmented':
        insights.extend(_analyze_segmented(df, model_id))

    return insights


def _get_col(df, base_col):
    """Get column name handling potential unit suffixes added by reports.py."""
    if base_col in df.columns: return base_col
    # Common suffixes used in formatters
    suffixes = [" (s)", " (ms)", " (tokens/s)", " (%)", " (req/min)", " (req/s)"]
    for s in suffixes:
        if f"{base_col}{s}" in df.columns:
            return f"{base_col}{s}"
    return None


def _get_val(df, idx, base_col):
    """Safely get value handling column names."""
    col = _get_col(df, base_col)
    if col:
        return df.loc[idx, col]
    return 0


# ============================================
# Concurrency Analysis
# ============================================

def _analyze_concurrency(df, model_id):
    """Analyze concurrency test results with comprehensive multi-dimensional assessment."""
    insights = []
    
    if 'concurrency' not in df.columns: return insights
    
    df = df.sort_values('concurrency')
    n_levels = len(df)
    
    # --- 1. Throughput: Peak, Absolute Level, and Scaling ---
    tp_col = _get_col(df, 'Max_System_Output_Throughput') or _get_col(df, 'Max_System_Throughput')

    if tp_col:
        peak_idx = df[tp_col].idxmax()
        peak_conc = df.loc[peak_idx, 'concurrency']
        peak_tp = df.loc[peak_idx, tp_col]
        
        if peak_tp > 0:
            # 1a. Report peak (always informational)
            insights.append(f"🎯 **Peak Throughput**: Reached **{peak_tp:.1f} tokens/s** at **{int(peak_conc)} concurrency**.")
            
            # 1b. Throughput level assessment - contextualize with concurrency
            # At single concurrency, throughput = single-stream decode speed, a model characteristic
            # Only assess as "low" when throughput doesn't scale with concurrency
            max_conc_tested = df['concurrency'].max()
            if max_conc_tested > 1:
                # At higher concurrency, assess throughput per concurrency unit
                tp_per_conc = peak_tp / peak_conc
                if tp_per_conc < 5:
                    insights.append(f"⚠️ **Low Concurrency Throughput Efficiency**: Peak throughput {peak_tp:.1f} t/s at {int(peak_conc)} concurrency, only {tp_per_conc:.1f} t/s per concurrent request. System concurrency utilization is insufficient.")
                elif peak_tp >= 300:
                    insights.append(f"✅ **Ample Throughput**: Peak {peak_tp:.1f} t/s, capable of sustaining high concurrent loads.")
            
            # 1c. Saturation / Degradation detection
            max_conc = df['concurrency'].max()
            if peak_conc < max_conc:
                last_tp = df.iloc[-1][tp_col]
                drop = (peak_tp - last_tp) / peak_tp * 100
                if drop > 10:
                    insights.append(f"⚠️ **Overload Degradation**: Beyond {int(peak_conc)} concurrency, system throughput dropped {drop:.1f}% due to increasing resource contention.")
            
            # 1d. Throughput scaling efficiency (need at least 2 levels)
            if n_levels >= 2:
                first_tp = df.iloc[0][tp_col]
                first_conc = df.iloc[0]['concurrency']
                last_conc = df.iloc[-1]['concurrency']
                last_tp_val = df.iloc[-1][tp_col]
                
                if first_tp > 0 and first_conc > 0 and last_conc > first_conc:
                    conc_ratio = last_conc / first_conc
                    tp_ratio = last_tp_val / first_tp
                    efficiency = tp_ratio / conc_ratio * 100
                    
                    if efficiency > 85:
                        insights.append(f"📈 **Excellent Scaling Efficiency**: Concurrency {int(first_conc)}→{int(last_conc)} (×{conc_ratio:.0f}), throughput scales near-linearly (efficiency {efficiency:.0f}%).")
                    elif efficiency > 50:
                        insights.append(f"📊 **Average Scaling Efficiency**: Concurrency increased {conc_ratio:.1f}×, but throughput only grew {tp_ratio:.1f}× (efficiency {efficiency:.0f}%), resource contention present.")
                    elif efficiency > 0:
                        insights.append(f"📉 **Poor Scaling Efficiency**: Concurrency increased {conc_ratio:.1f}×, but throughput only grew {tp_ratio:.1f}× (efficiency {efficiency:.0f}%), system near bottleneck.")
        else:
            insights.append(f"❌ **Throughput Anomaly**: System throughput is 0. Please check test configuration and API response data.")

    # --- 2. TPOT Trend Analysis (Mean) ---
    tpot_col = _get_col(df, 'TPOT_Mean')
    if tpot_col and n_levels >= 2:
        valid_tpot = df[df[tpot_col] > 0]
        if len(valid_tpot) >= 2:
            first_tpot = valid_tpot.iloc[0][tpot_col]
            last_tpot = valid_tpot.iloc[-1][tpot_col]
            first_conc = valid_tpot.iloc[0]['concurrency']
            last_conc = valid_tpot.iloc[-1]['concurrency']
            
            if first_tpot > 0:
                tpot_increase_pct = (last_tpot - first_tpot) / first_tpot * 100
                
                if tpot_increase_pct > 100:
                    insights.append(f"📉 **TPOT Severe Degradation**: From concurrency {int(first_conc)}→{int(last_conc)}, avg TPOT increased from {first_tpot:.1f}ms to {last_tpot:.1f}ms (↑{tpot_increase_pct:.0f}%), significant user experience decline.")
                elif tpot_increase_pct > 30:
                    insights.append(f"⚠️ **TPOT Notable Increase**: Avg TPOT from {first_tpot:.1f}ms to {last_tpot:.1f}ms (↑{tpot_increase_pct:.0f}%), users may notice slower generation speed.")
                elif tpot_increase_pct > 5:
                    insights.append(f"📊 **TPOT Slight Increase**: Avg TPOT from {first_tpot:.1f}ms to {last_tpot:.1f}ms (↑{tpot_increase_pct:.0f}%), minimal impact from concurrency.")
                else:
                    insights.append(f"✅ **TPOT Highly Stable**: Avg TPOT stays between {first_tpot:.1f}ms~{last_tpot:.1f}ms, virtually unaffected by concurrency.")

    # --- 3. Latency SLA (P99 TPOT) ---
    tpot_p99_col = _get_col(df, 'TPOT_P99')
    if tpot_p99_col:
        valid_tpot = df[df[tpot_p99_col] > 0]
        if not valid_tpot.empty:
            sla_limit = 150
            sla_breach = valid_tpot[valid_tpot[tpot_p99_col] > sla_limit]
            
            if not sla_breach.empty:
                safe_conc = valid_tpot[valid_tpot[tpot_p99_col] <= sla_limit]['concurrency'].max()
                if pd.isna(safe_conc):
                    insights.append(f"❌ **High Latency**: Even at lowest concurrency, P99 TPOT exceeds {sla_limit}ms, poor real-time interaction experience.")
                else:
                    insights.append(f"⚖️ **Latency SLA Suggestion**: To maintain P99 TPOT < {sla_limit}ms, recommend keeping concurrency at **{int(safe_conc)}** or below.")
            # Don't blindly say "excellent" - check if values are close to threshold
            else:
                max_p99 = valid_tpot[tpot_p99_col].max()
                if max_p99 > sla_limit * 0.8:
                    insights.append(f"📊 **P99 Latency Near Threshold**: Max P99 TPOT is {max_p99:.0f}ms, approaching {sla_limit}ms threshold. May breach with higher concurrency.")
                elif max_p99 < sla_limit * 0.3:
                    insights.append(f"✅ **Excellent Tail Latency**: P99 TPOT max only {max_p99:.0f}ms, well below {sla_limit}ms threshold.")
                else:
                    insights.append(f"📊 **P99 Latency Within Limits**: Max P99 TPOT is {max_p99:.0f}ms, within {sla_limit}ms threshold.")

    # --- 4. TTFT & Prefill Speed Analysis ---
    # TTFT alone is misleading - must be evaluated against input length via Prefill Speed
    ttft_col = _get_col(df, 'Best_TTFT')
    tokens_col = _get_col(df, 'Actual_Tokens_Mean')  # avg input tokens per request
    if ttft_col:
        valid_ttft = df[df[ttft_col] > 0]
        if not valid_ttft.empty:
            min_ttft = valid_ttft[ttft_col].min()
            max_ttft = valid_ttft[ttft_col].max()
            
            # 4a. Prefill Speed assessment (input_tokens / TTFT = true prefill efficiency)
            if tokens_col:
                valid_both = df[(df[ttft_col] > 0) & (df[tokens_col] > 0)]
                if not valid_both.empty:
                    avg_input = valid_both[tokens_col].mean()
                    # Use best (lowest) TTFT row for prefill speed
                    best_ttft_idx = valid_both[ttft_col].idxmin()
                    best_ttft = valid_both.loc[best_ttft_idx, ttft_col]
                    best_input = valid_both.loc[best_ttft_idx, tokens_col]
                    prefill_speed = best_input / best_ttft if best_ttft > 0 else 0
                    
                    if prefill_speed > 0:
                        if prefill_speed < 50:
                            insights.append(f"⚠️ **Low Prefill Speed**: ~{prefill_speed:.0f} t/s (input ~{int(best_input)} tokens, TTFT {best_ttft:.2f}s). Prefill phase may be a latency bottleneck.")
                        elif prefill_speed < 500:
                            insights.append(f"📊 **Average Prefill Speed**: ~{prefill_speed:.0f} t/s (input ~{int(best_input)} tokens, TTFT {best_ttft:.2f}s).")
                        elif prefill_speed < 5000:
                            insights.append(f"✅ **Good Prefill Speed**: ~{prefill_speed:.0f} t/s (input ~{int(best_input)} tokens, TTFT {best_ttft:.2f}s).")
                        else:
                            insights.append(f"🚀 **Blazing Fast Prefill**: ~{prefill_speed:.0f} t/s (input ~{int(best_input)} tokens, TTFT {best_ttft:.3f}s).")
            else:
                # No input token info available - fall back to raw TTFT reporting
                if min_ttft > 3.0:
                    insights.append(f"⚠️ **High TTFT**: Best TTFT is {min_ttft:.2f}s (no input token data available to calculate Prefill Speed).")
                elif min_ttft <= 0.3:
                    insights.append(f"✅ **Fast First Token**: Best TTFT only {min_ttft:.3f}s.")
                else:
                    insights.append(f"📊 **TTFT**: Best TTFT is {min_ttft:.2f}s (needs to be evaluated in context of input length).")
            
            # 4b. TTFT stability across concurrency
            if n_levels >= 2 and min_ttft > 0:
                if max_ttft > min_ttft * 3:
                    insights.append(f"📉 **Severe Queuing Effect**: TTFT surges to {max_ttft/min_ttft:.1f}× of baseline at high concurrency. System scheduling has a severe bottleneck.")
                elif max_ttft > min_ttft * 1.5:
                    ttft_growth = (max_ttft - min_ttft) / min_ttft * 100
                    insights.append(f"📊 **TTFT Grows with Concurrency**: From {min_ttft:.2f}s to {max_ttft:.2f}s (↑{ttft_growth:.0f}%), mild queuing observed.")

    # --- 5. Success Rate ---
    sr_col = _get_col(df, 'Success_Rate')
    if sr_col:
        min_sr = df[sr_col].min()
        if min_sr < 0.95:
            fail_conc = df[df[sr_col] < 0.95]['concurrency'].min()
            insights.append(f"🛑 **Stability Risk**: At {int(fail_conc)} concurrency, success rate dropped to {min_sr*100:.1f}%, some requests failed.")

    return insights


# ============================================
# Prefill Analysis
# ============================================

def _analyze_prefill(df, model_id):
    """Analyze prefill scaling and compute density."""
    insights = []
    
    target_col = 'input_tokens_target'
    speed_col = _get_col(df, 'Max_Prefill_Speed')
    
    if target_col not in df.columns or not speed_col: return insights
    
    df = df.sort_values(target_col)
    
    # 1. Speed Peak and Assessment
    max_speed_idx = df[speed_col].idxmax()
    max_speed = df.loc[max_speed_idx, speed_col]
    optimal_len = df.loc[max_speed_idx, target_col]
    
    if max_speed > 0:
        insights.append(f"🎯 **Peak Prefill Speed**: **{max_speed:.0f} tokens/s** (at input length {int(optimal_len)}).")
        
        # Absolute level
        if max_speed < 500:
            insights.append(f"⚠️ **Low Prefill Speed**: Peak only {max_speed:.0f} t/s, may impact TTFT in long-text scenarios.")
        elif max_speed >= 5000:
            insights.append(f"✅ **Blazing Fast Prefill**: Peak {max_speed:.0f} t/s, can efficiently process long input text.")
    else:
        insights.append(f"❌ **Prefill Speed Anomaly**: Prefill speed is 0 at all levels. Please check token counting and TTFT data.")
        return insights
    
    # 2. Compute Density / Speed Scaling
    if len(df) >= 2:
        first_speed = df.iloc[0][speed_col]
        last_speed = df.iloc[-1][speed_col]
        first_len = df.iloc[0][target_col]
        last_len = df.iloc[-1][target_col]
        
        if first_speed > 0:
            ratio = last_speed / first_speed
            if ratio < 0.5:
                insights.append(f"📉 **Severe Long-Text Compute Bottleneck**: Prefill speed dropped {(1 - ratio)*100:.0f}% from input {int(first_len)}→{int(last_len)}, quadratic attention complexity impact is significant.")
            elif ratio < 0.7:
                insights.append(f"⚠️ **Long-Text Compute Bottleneck**: Prefill speed dropped {(1 - ratio)*100:.0f}% at longest input, typically caused by quadratic attention complexity.")
            elif ratio > 1.2:
                insights.append(f"📈 **Compute-Dense Advantage**: As input grows, GPU utilization improves, Prefill efficiency up {(ratio - 1)*100:.0f}%.")
            else:
                insights.append(f"✅ **Stable Prefill Speed**: Speed variation stays within ±{abs(ratio - 1)*100:.0f}% across different input lengths.")
    
    # 3. TTFT at various input lengths
    ttft_col = _get_col(df, 'Best_TTFT')
    if ttft_col and len(df) >= 2:
        max_ttft = df[ttft_col].max()
        max_ttft_len = df.loc[df[ttft_col].idxmax(), target_col]
        if max_ttft > 5.0:
            insights.append(f"🐢 **Very High TTFT for Long Input**: At {int(max_ttft_len)} tokens input, TTFT reaches {max_ttft:.2f}s, severely impacting interaction experience.")
        elif max_ttft > 2.0:
            insights.append(f"⚠️ **High TTFT for Long Input**: At {int(max_ttft_len)} tokens input, TTFT is {max_ttft:.2f}s.")

    # 4. Success Rate
    success_col = _get_col(df, 'Success_Rate')
    if success_col:
        failures = df[df[success_col] < 1.0]
        if not failures.empty:
            fail_len = failures[target_col].min()
            insights.append(f"🛑 **Stability Risk**: Request failures starting at input length **{int(fail_len)}**, possibly triggered VRAM OOM or context limit.")

    return insights


# ============================================
# Long Context Analysis
# ============================================

def _analyze_long_context(df, model_id):
    """Analyze long context retention and decoding stability."""
    insights = []
    
    target_col = 'context_length_target'
    if target_col not in df.columns: return insights
    
    df = df.sort_values(target_col)
    
    # 1. Decoding Stability (TPS)
    tps_col = _get_col(df, 'Max_TPS')
    if tps_col:
        valid_rows = df[df[tps_col] > 0]
        if len(valid_rows) >= 2:
            base_tps = valid_rows.iloc[0][tps_col]
            final_tps = valid_rows.iloc[-1][tps_col]
            base_ctx = valid_rows.iloc[0][target_col]
            final_ctx = valid_rows.iloc[-1][target_col]
            
            if base_tps > 0:
                retention = final_tps / base_tps * 100
                
                if retention > 90:
                    insights.append(f"✅ **Highly Stable Long-Text Generation**: Context from {int(base_ctx)}→{int(final_ctx)} tokens, generation speed retains {retention:.1f}% of baseline.")
                elif retention > 70:
                    insights.append(f"📊 **Generally Stable Long-Text Generation**: Speed retains {retention:.1f}% of baseline, TPS from {base_tps:.1f}→{final_tps:.1f}, acceptable decline.")
                elif retention > 50:
                    insights.append(f"⚠️ **KV Cache Pressure Increasing**: Long context reduces generation speed to {retention:.1f}% of baseline, memory bandwidth under pressure.")
                else:
                    insights.append(f"📉 **Severe KV Cache Bottleneck**: Long context reduces generation speed to only {retention:.1f}% of baseline, memory bandwidth may be insufficient.")
        
        # Absolute TPS assessment
        if not valid_rows.empty:
            min_tps = valid_rows[tps_col].min()
            if min_tps < 10:
                insights.append(f"⚠️ **Low Minimum TPS**: TPS drops to only {min_tps:.1f} t/s at longest context, user experience may suffer.")

    # 2. TTFT Scaling
    ttft_col = _get_col(df, 'Best_TTFT')
    if ttft_col:
        valid_ttft = df[df[ttft_col] > 0]
        if not valid_ttft.empty:
            min_ttft = valid_ttft[ttft_col].min()
            long_ttft = valid_ttft.iloc[-1][ttft_col]
            
            if long_ttft > 10.0:
                insights.append(f"❌ **Extremely High TTFT**: TTFT reaches {long_ttft:.2f}s at longest context, completely unsuitable for interactive applications.")
            elif long_ttft > 5.0:
                insights.append(f"🐢 **Very High TTFT**: TTFT reaches {long_ttft:.2f}s at longest context, impacting interaction experience. Consider checking attention operator optimization.")
            elif long_ttft > 2.0:
                insights.append(f"⚠️ **High TTFT**: TTFT is {long_ttft:.2f}s at longest context, noticeable waiting time.")
            elif long_ttft <= 1.0:
                insights.append(f"✅ **Fast First Token at Long Context**: Even at longest context, TTFT is only {long_ttft:.3f}s.")
            
            # TTFT growth pattern
            if len(valid_ttft) >= 2 and min_ttft > 0:
                growth = long_ttft / min_ttft
                if growth > 10:
                    insights.append(f"📉 **Dramatic TTFT Growth**: TTFT surges {growth:.1f}× with context growth, prefill phase is a clear bottleneck.")
                elif growth > 3:
                    insights.append(f"📊 **Significant TTFT Growth**: TTFT increases {growth:.1f}× with context growth, monitor response times for long-text scenarios.")

    # 3. Prefill Speed Trend
    ps_col = _get_col(df, 'Max_Prefill_Speed')
    if ps_col:
        valid_ps = df[df[ps_col] > 0]
        if len(valid_ps) >= 2:
            first_ps = valid_ps.iloc[0][ps_col]
            last_ps = valid_ps.iloc[-1][ps_col]
            if first_ps > 0:
                ratio = last_ps / first_ps
                if ratio > 1.2:
                    insights.append(f"📈 **Prefill Efficiency Improves**: Prefill speed actually increases {(ratio-1)*100:.0f}% at longer context, higher GPU compute utilization.")
                elif ratio < 0.5:
                    insights.append(f"⚠️ **Significant Prefill Speed Drop**: Prefill speed drops {(1-ratio)*100:.0f}% at longer context.")

    # 4. Success Rate
    sr_col = _get_col(df, 'Success_Rate')
    if sr_col:
        failures = df[df[sr_col] < 1.0]
        if not failures.empty:
            fail_ctx = failures[target_col].min()
            insights.append(f"🛑 **Context Limit**: Request failures at context length **{int(fail_ctx)}**, possibly triggered context window or VRAM limit.")

    return insights


# ============================================
# Matrix Analysis
# ============================================

def _analyze_matrix(df, model_id):
    """Analyze multi-dimensional configuration sweet spots."""
    insights = []
    
    tp_col = _get_col(df, 'Max_System_Output_Throughput')
    if not tp_col: return insights
    
    # Guard: check for meaningful throughput data
    valid_tp = df[df[tp_col] > 0]
    if valid_tp.empty:
        insights.append("❌ **Throughput Data Anomaly**: Throughput is 0 for all configurations, cannot perform optimization analysis.")
        return insights
    
    # 1. Global Optimum
    best_idx = valid_tp[tp_col].idxmax()
    best_row = valid_tp.loc[best_idx]
    worst_idx = valid_tp[tp_col].idxmin()
    worst_row = valid_tp.loc[worst_idx]
    
    insights.append(f"🏆 **Global Best Configuration**: **{int(best_row['concurrency'])} concurrency + {int(best_row['context_length_target'])} context**, throughput reaches {best_row[tp_col]:.1f} t/s.")
    
    # Performance gap
    if worst_row[tp_col] > 0:
        gap = best_row[tp_col] / worst_row[tp_col]
        if gap > 3:
            insights.append(f"⚠️ **Configuration Sensitive**: Best vs worst configuration throughput gap is {gap:.1f}×, wrong configuration is costly.")
        elif gap > 1.5:
            insights.append(f"📊 **Configuration Matters**: Best vs worst throughput gap is {gap:.1f}×, proper tuning has noticeable benefits.")
    
    # 2. Concurrency Resilience
    try:
        if 'context_length_target' in valid_tp.columns:
            grouped = valid_tp.groupby('context_length_target')[tp_col].std()
            mean_std = grouped.mean()
            mean_tp = valid_tp[tp_col].mean()
            
            cv = mean_std / mean_tp if mean_tp > 0 else 0
            
            if cv < 0.15:
                insights.append(f"✅ **Load Balanced**: Model performs consistently across different concurrency and context combinations (CV={cv*100:.0f}%).")
            elif cv < 0.3:
                insights.append(f"📊 **Some Variation**: Throughput varies somewhat across configurations (CV={cv*100:.0f}%).")
            else:
                insights.append(f"🎢 **Load Sensitive**: Model performance is very sensitive to configuration (CV={cv*100:.0f}%), consider fine-grained auto-scaling strategies.")
    except Exception:
        pass

    return insights


# ============================================
# Segmented (Prefix Caching) Analysis
# ============================================

def _analyze_segmented(df, model_id):
    """Analyze prefix caching effectiveness."""
    insights = []
    
    target_col = 'context_length_target'
    if target_col not in df.columns: return insights
    
    df = df.sort_values(target_col)
    
    if len(df) >= 2:
        first = df.iloc[0]
        last = df.iloc[-1]
        
        # Check TTFT Benefit
        uncached_ttft_col = _get_col(df, 'Uncached_TTFT')
        cached_ttft_col = _get_col(df, 'Cached_TTFT')
        
        if uncached_ttft_col and cached_ttft_col:
            # Evaluate caching effect at the longest context (last segment)
            t_uncached = last[uncached_ttft_col]
            t_cached = last[cached_ttft_col]
            
            # Guard: both values must be positive to compare
            if t_uncached > 0 and t_cached > 0:
                save = t_uncached - t_cached
                
                if save > 0.05:
                    pct = (save/t_uncached)*100
                    insights.append(f"⚡ **Cache Hit Acceleration**: At longest context ({int(last[target_col])}), TTFT reduced by {save:.3f}s from caching (improvement {pct:.1f}%).")
                elif save > 0:
                    insights.append(f"📊 **Weak Cache Effect**: At longest context, TTFT reduced by only {save:.3f}s from caching.")
                elif save < -0.05:
                    insights.append(f"⚠️ **Cache Negative Impact**: At longest context, TTFT actually increased by {abs(save):.3f}s after cache hit?!")
        
        # Check Prefill Throughput
        pf_col = _get_col(df, 'Max_Prefill_Speed')
        if pf_col and first[pf_col] > 0 and last[pf_col] > 0:
            speedup = last[pf_col] / first[pf_col]
            if speedup > 2.0:
                insights.append(f"🚀 **Throughput Multiplied**: Caching mechanism boosts effective Prefill speed by {speedup:.1f}× (from {first[pf_col]:.0f} to {last[pf_col]:.0f} t/s).")
            elif speedup > 1.2:
                insights.append(f"📈 **Throughput Improved**: Caching boosts effective Prefill speed by {(speedup-1)*100:.0f}%.")
            elif speedup < 0.8:
                insights.append(f"⚠️ **Prefill Speed Drop**: Evaluated prefix caching did not improve speed; dropped {(1-speedup)*100:.0f}%.")

    return insights


# ============================================
# Grading System
# ============================================

def get_performance_grade(insights):
    """
    Calculate overall performance grade based on insight distribution.
    
    Uses a weighted scoring system rather than simple counting to produce
    fair and accurate grades. Guards against positive grades when data
    quality is poor.

    Returns:
        Tuple of (grade, color, description)
    """
    if not insights:
        return 'N/A', '#6c757d', 'Insufficient data for grading'
    
    if all('Insufficient Data' in i or 'Analysis Skipped' in i for i in insights):
        return 'N/A', '#6c757d', 'Insufficient data for grading'

    # Categorize each insight
    positive_emojis = ['✅', '🚀', '🏆', '📈', '⚡']
    neutral_emojis = ['📊', '⚖️', '🎯']
    warning_emojis = ['⚠️', '🐢', '📉', '🎢']
    critical_emojis = ['❌', '🛑']
    
    positive_count = sum(1 for i in insights if any(e in i for e in positive_emojis))
    neutral_count = sum(1 for i in insights if any(e in i for e in neutral_emojis) and not any(e in i for e in positive_emojis + warning_emojis + critical_emojis))
    warning_count = sum(1 for i in insights if any(e in i for e in warning_emojis))
    critical_count = sum(1 for i in insights if any(e in i for e in critical_emojis))
    
    total_scored = positive_count + neutral_count + warning_count + critical_count
    if total_scored == 0:
        return 'B', '#6c757d', 'Performance adequate'

    # Weighted score: positive=+2, neutral=0, warning=-1, critical=-3
    score = positive_count * 2 + neutral_count * 0 + warning_count * (-1) + critical_count * (-3)
    # Normalize to a per-insight score
    avg_score = score / total_scored

    # Critical issues always take priority
    if critical_count > 0:
        if critical_count >= 2:
            return 'D', '#dc3545', 'Multiple critical issues detected'
        return 'C', '#dc3545', 'Critical issue detected'
    
    # Grade based on avg score and distribution
    if avg_score >= 1.5 and warning_count == 0:
        return 'A+', '#28a745', 'Outstanding performance'
    elif avg_score >= 1.0 and warning_count == 0:
        return 'A', '#28a745', 'Excellent performance'
    elif avg_score >= 0.5 and warning_count <= 1:
        return 'B+', '#17a2b8', 'Good performance, minor optimization opportunities'
    elif avg_score >= 0 and warning_count <= positive_count:
        return 'B', '#6c757d', 'Adequate performance, room for optimization'
    elif avg_score >= -0.5:
        return 'B-', '#ffc107', 'Average performance, optimization recommended'
    else:
        return 'C+', '#ffc107', 'Below average, targeted optimization needed'


# ============================================
# Export
# ============================================

def generate_insights_markdown(insights, model_id=''):
    """
    Format insights as markdown for export.

    Returns:
        Markdown formatted string
    """
    if not insights:
        return "## Performance Insights\n\nNo performance insights available.\n"

    grade, color, description = get_performance_grade(insights)

    md = "## Performance Insights\n\n"
    md += f"**Overall Grade**: {grade} - {description}\n\n"

    if model_id:
        md += f"**Model**: {model_id}\n\n"

    md += "### Detailed Analysis\n\n"

    for insight in insights:
        md += f"- {insight}\n"

    return md


def _generate_markdown_summary(df, test_type, duration):
    """
    Generate a markdown summary section mirroring the enhanced UI card.
    Provides text-based insights for the report file, including peak values and trend analysis.
    """
    if df is None or df.empty:
        return ""
        
    import numpy as np

    def _fmt(n):
        if n >= 1_000_000: return f"{n / 1_000_000:.2f}M"
        if n >= 1_000: return f"{n / 1_000:.1f}K"
        return f"{int(n)}"
    
    total_requests = len(df)
    error_col = df.get('error')
    successful = int(error_col.isnull().sum()) if error_col is not None else total_requests
    success_rate = (successful / total_requests * 100) if total_requests > 0 else 0
    total_input = int(df['prefill_tokens'].sum()) if 'prefill_tokens' in df.columns else 0
    total_output = int(df['decode_tokens'].sum()) if 'decode_tokens' in df.columns else 0
    df_ok = df[df['error'].isnull()] if 'error' in df.columns else df

    md = "### 📊 Test Summary\n\n"
    
    # Common stats
    md += f"- **Total Requests**: {total_requests} (succeeded: {successful}, Success Rate: {success_rate:.1f}%)\n"
    md += f"- **Total Data Volume**: Input {_fmt(total_input)} tokens, Output {_fmt(total_output)} tokens\n" 

    if test_type == 'concurrency':
        conc_levels = sorted(df_ok['concurrency'].unique()) if 'concurrency' in df_ok.columns else []
        if len(conc_levels) >= 2:
            lo, hi = conc_levels[0], conc_levels[-1]
            lo_df, hi_df = df_ok[df_ok['concurrency'] == lo], df_ok[df_ok['concurrency'] == hi]
            ttft_lo = lo_df['ttft'].mean() if 'ttft' in lo_df.columns else 0
            ttft_hi = hi_df['ttft'].mean() if 'ttft' in hi_df.columns else 0
            
            tp_col = 'system_output_throughput' if 'system_output_throughput' in df_ok.columns else ('system_throughput' if 'system_throughput' in df_ok.columns else None)
            if tp_col:
                tp_by_c = df_ok.groupby('concurrency')[tp_col].max()
                peak_tp, peak_c = tp_by_c.max(), int(tp_by_c.idxmax())
            else:
                peak_tp, peak_c = 0, "N/A"
                
            md += f"- **Concurrency Scalability**: When concurrency increased from {int(lo)} to {int(hi)}, TTFT went from {ttft_lo:.3f}s {'increased to' if ttft_hi > ttft_lo else 'changed to'} {ttft_hi:.3f}s.\n"
            md += f"- **Peak Performance**: at concurrency {peak_c} reached peak system throughput of {peak_tp:.1f} tokens/s.\n"

            # --- Peak & Trend Analysis ---
            ttft_by_c = df_ok.groupby('concurrency')['ttft'].mean()
            best_ttft_val = ttft_by_c.min()
            best_ttft_c = int(ttft_by_c.idxmin())
            tps_by_c = df_ok.groupby('concurrency')['tps'].mean()
            peak_tps_val = tps_by_c.max()
            peak_tps_c = int(tps_by_c.idxmax())

            md += f"- **Best TTFT**: {best_ttft_val:.3f}s (@ Concurrency {best_ttft_c})\n"
            md += f"- **Peak Single-Stream TPS**: {peak_tps_val:.1f} t/s (@ Concurrency {peak_tps_c})\n"

            # TTFT trend
            ttft_vals = [ttft_by_c[c] for c in conc_levels]
            if len(ttft_vals) >= 3:
                diffs = [ttft_vals[i+1] - ttft_vals[i] for i in range(len(ttft_vals)-1)]
                max_jump_idx = max(range(len(diffs)), key=lambda i: diffs[i])
                jump_from = int(conc_levels[max_jump_idx])
                jump_to = int(conc_levels[max_jump_idx + 1])
                jump_pct = (diffs[max_jump_idx] / ttft_vals[max_jump_idx] * 100) if ttft_vals[max_jump_idx] > 0 else 0
                if jump_pct > 10:
                    md += f"- **TTFT Trend**: Concurrency {jump_from}→{jump_to} showed largest spike ({jump_pct:.0f}%).\n"
                else:
                    md += f"- **TTFT Trend**: increased steadily with concurrency.\n"

            # Throughput saturation
            if tp_col:
                tp_vals = [tp_by_c[c] for c in conc_levels if c in tp_by_c.index]
                if len(tp_vals) >= 3:
                    saturation_c = None
                    for i in range(1, len(tp_vals)):
                        gain = (tp_vals[i] - tp_vals[i-1]) / tp_vals[i-1] if tp_vals[i-1] > 0 else 0
                        if gain < 0.05 and saturation_c is None:
                            saturation_c = int(conc_levels[i])
                    if saturation_c:
                        md += f"- **Throughput Saturation**: Concurrency {saturation_c} throughput growth fell below 5%.\n"
                    else:
                        md += f"- **Throughput Saturation**: not yet saturated, still has growth capacity.\n"
            
    elif test_type == 'prefill':
        levels = sorted(df_ok['input_tokens_target'].unique()) if 'input_tokens_target' in df_ok.columns else []
        if len(levels) >= 2:
            sm, lg = levels[0], levels[-1]
            sm_df, lg_df = df_ok[df_ok['input_tokens_target'] == sm], df_ok[df_ok['input_tokens_target'] == lg]
            ps_sm = sm_df['prefill_speed'].mean() if 'prefill_speed' in sm_df.columns else 0
            ps_lg = lg_df['prefill_speed'].mean() if 'prefill_speed' in lg_df.columns else 0
            ratio = ps_lg / ps_sm if ps_sm > 0 else 0
            
            md += f"- **Prefill Speed Trend**: When input length increased from {_fmt(sm)} to {_fmt(lg)}, prefill speed went from {ps_sm:.0f} t/s to {ps_lg:.0f} t/s (retention rate {ratio*100:.1f}%).\n"

            # --- Peak & Trend Analysis ---
            ps_by_level = df_ok.groupby('input_tokens_target')['prefill_speed'].mean()
            peak_ps_val = ps_by_level.max()
            peak_ps_level = int(ps_by_level.idxmax())
            ttft_by_level = df_ok.groupby('input_tokens_target')['ttft'].mean()
            best_ttft_val = ttft_by_level.min()
            best_ttft_level = int(ttft_by_level.idxmin())

            md += f"- **Peak Prefill Speed**: {peak_ps_val:.0f} t/s (@ {_fmt(peak_ps_level)} tokens)\n"
            md += f"- **Best TTFT**: {best_ttft_val:.3f}s (@ {_fmt(best_ttft_level)} tokens)\n"

            # Speed drop detection
            ps_vals = [ps_by_level[lv] for lv in levels]
            if len(ps_vals) >= 3:
                drops = [(ps_vals[i] - ps_vals[i+1]) / ps_vals[i] * 100 if ps_vals[i] > 0 else 0 for i in range(len(ps_vals)-1)]
                max_drop_idx = max(range(len(drops)), key=lambda i: drops[i])
                drop_from = levels[max_drop_idx]
                drop_to = levels[max_drop_idx + 1]
                drop_pct = drops[max_drop_idx]
                if drop_pct > 10:
                    md += f"- **Speed Degradation Inflection**: {_fmt(drop_from)}→{_fmt(drop_to)} showed largest drop ({drop_pct:.0f}%).\n"
                else:
                    md += f"- **Speed Degradation Pattern**: decreased steadily with increasing input length.\n"

            # TTFT scaling pattern
            if len(levels) >= 3:
                input_ratio = levels[-1] / levels[0] if levels[0] > 0 else 1
                ttft_ratio_full = ttft_by_level[levels[-1]] / ttft_by_level[levels[0]] if ttft_by_level[levels[0]] > 0 else 1
                if ttft_ratio_full > input_ratio * 1.5:
                    md += f"- **TTFT Growth Pattern**: Super-linear growth (TTFT grew {ttft_ratio_full:.1f}x while input only grew {input_ratio:.1f}x) ⚠️\n"
                elif ttft_ratio_full > input_ratio * 0.8:
                    md += f"- **TTFT Growth Pattern**: approximately linear.\n"
                else:
                    md += f"- **TTFT Growth Pattern**: Sub-linear (TTFT grew slower than input) ✅\n"
            
    elif test_type == 'long_context':
        levels = sorted(df_ok['context_length_target'].unique()) if 'context_length_target' in df_ok.columns else []
        if len(levels) >= 2:
            sh, lo = levels[0], levels[-1]
            sh_df, lo_df = df_ok[df_ok['context_length_target'] == sh], df_ok[df_ok['context_length_target'] == lo]
            ttft_sh = sh_df['ttft'].mean() if 'ttft' in sh_df.columns else 0
            ttft_lo2 = lo_df['ttft'].mean() if 'ttft' in lo_df.columns else 0
            tps_sh = sh_df['tps'].mean() if 'tps' in sh_df.columns else 0
            tps_lo2 = lo_df['tps'].mean() if 'tps' in lo_df.columns else 0
            
            md += f"- **Long-Text Performance Degradation**: when context grew from {_fmt(sh)} to {_fmt(lo)}, TTFT went from {ttft_sh:.3f}s to {ttft_lo2:.3f}s.\n"
            md += f"- **Generation Stability**: single-stream TPS went from {tps_sh:.1f} t/s to {tps_lo2:.1f} t/s.\n"

            # --- Peak & Trend Analysis ---
            ps_by_ctx = df_ok.groupby('context_length_target')['prefill_speed'].mean()
            peak_ps_val = ps_by_ctx.max()
            peak_ps_ctx = int(ps_by_ctx.idxmax())
            tps_by_ctx = df_ok.groupby('context_length_target')['tps'].mean()
            peak_tps_val = tps_by_ctx.max()
            peak_tps_ctx = int(tps_by_ctx.idxmax())

            md += f"- **Peak Prefill Speed**: {peak_ps_val:.0f} t/s (@ {_fmt(peak_ps_ctx)} ctx)\n"
            md += f"- **Peak TPS**: {peak_tps_val:.1f} t/s (@ {_fmt(peak_tps_ctx)} ctx)\n"

            # TTFT inflection detection
            ttft_by_ctx = df_ok.groupby('context_length_target')['ttft'].mean()
            ttft_vals = [ttft_by_ctx[lv] for lv in levels]
            if len(ttft_vals) >= 3:
                jumps = [(ttft_vals[i+1] - ttft_vals[i]) / ttft_vals[i] * 100 if ttft_vals[i] > 0 else 0 for i in range(len(ttft_vals)-1)]
                max_jump_idx = max(range(len(jumps)), key=lambda i: jumps[i])
                jump_from = levels[max_jump_idx]
                jump_to = levels[max_jump_idx + 1]
                jump_pct = jumps[max_jump_idx]
                if jump_pct > 20:
                    md += f"- **TTFT Inflection Point**: {_fmt(jump_from)}→{_fmt(jump_to)} showed largest spike ({jump_pct:.0f}%).\n"
                else:
                    md += f"- **TTFT Trend**: grew uniformly with increasing context.\n"

            # TPS stability
            tps_vals = [tps_by_ctx[lv] for lv in levels]
            tps_cv = (np.std(tps_vals) / np.mean(tps_vals) * 100) if np.mean(tps_vals) > 0 else 0
            if tps_cv > 10:
                md += f"- **Overall TPS Fluctuation**: Coefficient of variation {tps_cv:.0f}%,showing significant fluctuation.\n"
            else:
                md += f"- **Overall TPS Stability**: Coefficient of variation {tps_cv:.0f}%,showing stable performance.\n"

    elif test_type == 'segmented':
         cache_total = int(df_ok['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in df_ok.columns else 0
         cache_rate = (cache_total / total_input * 100) if total_input > 0 else 0
         
         levels = sorted(df_ok['context_length_target'].unique()) if 'context_length_target' in df_ok.columns else []
         if len(levels) >= 2:
            f_seg, l_seg = levels[0], levels[-1]
            f_df, l_df = df_ok[df_ok['context_length_target'] == f_seg], df_ok[df_ok['context_length_target'] == l_seg]
            
            ch_f = int(f_df['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in f_df.columns else 0
            in_f = int(f_df['prefill_tokens'].sum()) if 'prefill_tokens' in f_df.columns else 0
            r_f = (ch_f / in_f * 100) if in_f > 0 else 0
            
            ch_l = int(l_df['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in l_df.columns else 0
            in_l = int(l_df['prefill_tokens'].sum()) if 'prefill_tokens' in l_df.columns else 0
            r_l = (ch_l / in_l * 100) if in_l > 0 else 0
            
            md += f"- **Prefix Caching**: Overall Cache Hit Rate {cache_rate:.1f}% (hit {_fmt(cache_total)} tokens).\n"
            md += f"- **Cache Growth**: first-segment cache rate {r_f:.0f}% → last-segment cache rate {r_l:.0f}%.\n"

            # --- Peak & Trend Analysis ---
            cache_by_seg = {}
            ps_by_seg = df_ok.groupby('context_length_target')['prefill_speed'].max()
            ttft_by_seg = df_ok.groupby('context_length_target')['ttft'].max()
            for lv in levels:
                lv_df = df_ok[df_ok['context_length_target'] == lv]
                ch = int(lv_df['cache_hit_tokens'].sum()) if 'cache_hit_tokens' in lv_df.columns else 0
                inp = int(lv_df['prefill_tokens'].sum()) if 'prefill_tokens' in lv_df.columns else 0
                cache_by_seg[lv] = (ch / inp * 100) if inp > 0 else 0

            peak_cache_seg = max(cache_by_seg, key=cache_by_seg.get)
            peak_cache_rate = cache_by_seg[peak_cache_seg]
            best_ttft_seg = int(ttft_by_seg.idxmin())
            best_ttft_val = ttft_by_seg.min()
            peak_ps_seg = int(ps_by_seg.idxmax())
            peak_ps_val = ps_by_seg.max()

            md += f"- **Peak Cache Rate**: {peak_cache_rate:.0f}% (@ {_fmt(peak_cache_seg)} ctx)\n"
            md += f"- **Min Uncached TTFT**: {best_ttft_val:.3f}s (@ {_fmt(best_ttft_seg)} ctx)\n"
            md += f"- **Peak Prefill Speed**: {peak_ps_val:.0f} t/s (@ {_fmt(peak_ps_seg)} ctx)\n"

            # Cache trend
            cache_vals = [cache_by_seg[lv] for lv in levels]
            if len(cache_vals) >= 3:
                increasing = all(cache_vals[i+1] >= cache_vals[i] - 1 for i in range(len(cache_vals)-1))
                if increasing and cache_vals[-1] > cache_vals[0] + 5:
                    md += f"- **Cache Rate Trend**: continuously increasing, cache effectiveness grows significantly with segments.\n"
                elif not increasing:
                    md += f"- **Cache Rate Trend**: shows fluctuation, cache effectiveness is unstable.\n"
                else:
                    md += f"- **Cache Rate Trend**: overall stable.\n"

    elif test_type == 'matrix':
        tp_col = 'system_output_throughput' if 'system_output_throughput' in df_ok.columns else ('system_throughput' if 'system_throughput' in df_ok.columns else None)
        if tp_col:
             combo = df_ok.groupby(['concurrency', 'context_length_target'])[tp_col].max().reset_index()
             best = combo.loc[combo[tp_col].idxmax()]
             worst = combo.loc[combo[tp_col].idxmin()]
             md += f"- **Best Configuration**: Concurrency {int(best['concurrency'])} × context {_fmt(best['context_length_target'])} reached peak throughput of {best[tp_col]:.1f} t/s.\n"
             md += f"- **Worst Configuration**: Concurrency {int(worst['concurrency'])} × context {_fmt(worst['context_length_target'])} throughput was {worst[tp_col]:.1f} t/s.\n"

             # TTFT range
             ttft_all = df_ok[df_ok['ttft'] > 0]['ttft'] if 'ttft' in df_ok.columns else None
             if ttft_all is not None and not ttft_all.empty:
                 md += f"- **TTFT Range**: {ttft_all.min():.3f}s ~ {ttft_all.max():.3f}s.\n"

    md += "\n"
    return md

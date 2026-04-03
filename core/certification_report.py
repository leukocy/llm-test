import json
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# MQC Certification Standard Version
MQC_VERSION = "1.2"

# Configurable thresholds for certification criteria
MQC_THRESHOLDS = {
    "burst_success_rate_pass": 0.999,    # 99.9% success rate for full pass
    "burst_success_rate_warn": 0.99,      # 99% success rate for pass with warning
    "boundary_max_failures": 0,           # Maximum allowed boundary test failures
    "niah_min_pass_rate": 0.8,            # Minimum NIAH pass rate across all tests
}

@dataclass
class CertificationResult:
    test_id: str
    status: str  # PASS / FAIL
    criteria_met: list[str] = field(default_factory=list)
    criteria_failed: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class CertificationReportGenerator:
    """
    Generates specialized productivity certification reports.
    """
    def __init__(self, output_dir: str = "quality_results/certification"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, model_id: str, results_df: Any) -> CertificationResult:
        """
        Analyzes test results and generates a pass/fail certificate based on Stage 1 & 2.
        """
        status = "PASS"
        met = []
        failed = []
        
        # Initialize metrics with version and configuration info
        max_context_tested = None
        if 'context_length_target' in results_df.columns:
            max_context_tested = int(results_df['context_length_target'].max())
        
        metrics = {
            "model_id": model_id,
            "total_requests": len(results_df),
            "mqc_version": MQC_VERSION,
            "max_context_tested": max_context_tested,
        }

        # Stage 1: The Qwen Standard Analysis
        s1_df = results_df[results_df['test_type'].str.startswith('mqc_s1_')]
        if not s1_df.empty:
            # 1.1 Zero/Short
            if s1_df[s1_df['test_type'] == 'mqc_s1_zero']['error'].isnull().all():
                met.append("Stage 1 - Zero/Short: Successfully handled 10-token prompt.")
            else:
                status = "FAIL"
                failed.append("Stage 1 - Zero/Short: Failed basic connectivity test.")

            # 1.2 Boundaries - use configurable threshold
            boundary_fails = s1_df[s1_df['test_type'].str.contains('boundary')]['error'].notnull().sum()
            if boundary_fails <= MQC_THRESHOLDS['boundary_max_failures']:
                met.append("Stage 1 - Boundaries: Passed all fill levels (50%, 80%, 95%, 100%).")
            else:
                status = "FAIL"
                failed.append(f"Stage 1 - Boundaries: {boundary_fails} boundary levels failed.")

            # 1.3 Overflow
            ov_res = s1_df[s1_df['test_type'] == 'mqc_s1_overflow']
            if not ov_res.empty:
                if ov_res['error'].isnull().all():
                    met.append("Stage 1 - Overflow (105%): Gracefully handled (Model processed or engine truncated).")
                else:
                    met.append("Stage 1 - Overflow (105%): Engine returned error (Acceptable if non-crashing).")

            # 1.4 NIAH - now supports multiple context/depth combinations
            niah_res = s1_df[s1_df['test_type'] == 'mqc_s1_niah']
            if not niah_res.empty:
                niah_total = len(niah_res)
                niah_passed = niah_res['niah_correct'].sum() if 'niah_correct' in niah_res.columns else 0
                niah_pass_rate = niah_passed / niah_total if niah_total > 0 else 0
                metrics['niah_pass_rate'] = niah_pass_rate
                metrics['niah_tests_total'] = niah_total
                metrics['niah_tests_passed'] = int(niah_passed)
                
                if niah_pass_rate >= MQC_THRESHOLDS['niah_min_pass_rate']:
                    met.append(f"Stage 1 - NIAH: {int(niah_passed)}/{niah_total} tests passed ({niah_pass_rate*100:.0f}%).")
                else:
                    status = "FAIL"
                    failed.append(f"Stage 1 - NIAH: Only {int(niah_passed)}/{niah_total} passed ({niah_pass_rate*100:.0f}% < {MQC_THRESHOLDS['niah_min_pass_rate']*100:.0f}% threshold).")

            # Performance Meta (Stage 1)
            ttfts = s1_df['ttft'].dropna() * 1000 # to ms
            if not ttfts.empty:
                metrics["s1_ttft_p50"] = np.percentile(ttfts, 50)
                metrics["s1_ttft_p95"] = np.percentile(ttfts, 95)
                metrics["s1_ttft_p99"] = np.percentile(ttfts, 99)
                met.append(f"Stage 1 Performance: P50={metrics['s1_ttft_p50']:.0f}ms, P99={metrics['s1_ttft_p99']:.0f}ms.")
            
            # Cache Hit Ratio Analysis (Phase 2 improvement)
            boundary_df = s1_df[s1_df['test_type'].str.contains('boundary')]
            if not boundary_df.empty and 'cache_hit_tokens' in boundary_df.columns and 'prefill_tokens' in boundary_df.columns:
                # Calculate cache hit ratio for each boundary level
                boundary_df = boundary_df.copy()
                boundary_df['cache_hit_ratio'] = boundary_df.apply(
                    lambda row: (row['cache_hit_tokens'] / row['prefill_tokens'] * 100) 
                    if row.get('prefill_tokens', 0) > 0 else 0, axis=1
                )
                
                avg_cache_ratio = boundary_df['cache_hit_ratio'].mean()
                max_cache_ratio = boundary_df['cache_hit_ratio'].max()
                metrics['boundary_cache_hit_avg'] = avg_cache_ratio
                metrics['boundary_cache_hit_max'] = max_cache_ratio
                
                # Generate cache efficiency insights
                if avg_cache_ratio > 50:
                    met.append(f"Stage 1 - Cache Efficiency: Excellent ({avg_cache_ratio:.1f}% average hit rate).")
                elif avg_cache_ratio > 20:
                    met.append(f"Stage 1 - Cache Efficiency: Moderate ({avg_cache_ratio:.1f}% average hit rate).")
                elif avg_cache_ratio > 0:
                    met.append(f"Stage 1 - Cache Efficiency: Low ({avg_cache_ratio:.1f}% average hit rate, consider enabling Prefix Caching).")
                
                # Check if cache hits increase with context length (expected behavior)
                if len(boundary_df) >= 2:
                    sorted_df = boundary_df.sort_values('context_length_target')
                    first_cache = sorted_df.iloc[0]['cache_hit_ratio']
                    last_cache = sorted_df.iloc[-1]['cache_hit_ratio']
                    if last_cache > first_cache:
                        metrics['cache_trend'] = 'increasing'
                        met.append(f"Stage 1 - Cache Trend: Progressive caching working ({first_cache:.1f}% → {last_cache:.1f}%).")

        # Stage 2: The 1000 Requests Standard Analysis
        s2_df = results_df[results_df['test_type'].str.startswith('mqc_s2_')]
        if not s2_df.empty:
            # 2.1 Scenario A (Burst)
            burst_df = s2_df[s2_df['test_type'] == 'mqc_s2_burst']
            success_rate = burst_df['error'].isnull().mean()
            metrics['burst_success_rate'] = success_rate
            
            # Use configurable thresholds
            if success_rate >= MQC_THRESHOLDS['burst_success_rate_pass']:
                met.append(f"Stage 2 - Burst: {success_rate*100:.1f}% Success rate across {len(burst_df)} short prompts.")
            elif success_rate >= MQC_THRESHOLDS['burst_success_rate_warn']:
                met.append(f"Stage 2 - Burst: {success_rate*100:.2f}% Success rate (Pass with minor jitter).")
            else:
                status = "FAIL"
                failed.append(f"Stage 2 - Burst: Stability below {MQC_THRESHOLDS['burst_success_rate_warn']*100:.0f}% ({success_rate*100:.2f}%).")
            
            # Add throughput metrics if available
            if 'system_output_throughput' in burst_df.columns:
                avg_throughput = burst_df['system_output_throughput'].dropna().mean()
                if avg_throughput > 0:
                    metrics['burst_avg_throughput_tps'] = avg_throughput
                    met.append(f"Stage 2 - Burst Throughput: {avg_throughput:.1f} tokens/sec average.")

            # Performance Meta (Stage 2 Burst)
            burst_ttfts = burst_df['ttft'].dropna() * 1000
            if not burst_ttfts.empty:
                metrics["s2_burst_ttft_p50"] = np.percentile(burst_ttfts, 50)
                metrics["s2_burst_ttft_p99"] = np.percentile(burst_ttfts, 99)
                met.append(f"Stage 2 Burst Performance: P50={metrics['s2_burst_ttft_p50']:.0f}ms, P99={metrics['s2_burst_ttft_p99']:.0f}ms.")

            # 2.2 Scenario B (Mixed)
            mixed_df = s2_df[s2_df['test_type'] == 'mqc_s2_mixed']
            long_tasks = mixed_df[mixed_df['is_long'] == True]
            short_tasks = mixed_df[mixed_df['is_long'] == False]
            if long_tasks['error'].isnull().all() and short_tasks['error'].isnull().all():
                met.append(f"Stage 2 - Mixed Load: Handled {len(mixed_df)} mixed tasks (80/20 s/l).")
            else:
                status = "FAIL"
                failed.append("Stage 2 - Mixed Load: Failed in mixed-load batch processing.")

        # Stage 3: Multi-round Dialogue Stability Analysis
        s3_df = results_df[results_df['test_type'].str.startswith('mqc_s3_')]
        if not s3_df.empty:
            for version in ['A', 'B', 'C']:
                v_df = s3_df[s3_df['test_type'] == f'mqc_s3_dialogue_{version}']
                if not v_df.empty:
                    last_round = v_df.iloc[-1]
                    if last_round['error'] is None and last_round['semantic_correct']:
                        met.append(f"Stage 3 - Sessions {version}: Maintained secret key across {len(v_df)} rounds.")
                    elif last_round['error'] is not None:
                        status = "FAIL"
                        failed.append(f"Stage 3 - Sessions {version}: Crashed at round {last_round['round']}.")
                    else:
                        status = "FAIL"
                        failed.append(f"Stage 3 - Sessions {version}: Lost context/secret key by round {last_round['round']}.")

        cert_result = CertificationResult(
            test_id=f"MQC-{model_id}-{datetime.now().strftime('%Y%m%d%H%M')}",
            status=status,
            criteria_met=met,
            criteria_failed=failed,
            metrics={"success_rate": results_df['error'].isnull().mean(), "total_requests": len(results_df), "model_id": model_id}
        )
        
        self._save_report_md(cert_result)
        return cert_result

    def _save_report_md(self, result: CertificationResult):
        path = self.output_dir / f"{result.test_id}.md"
        badge = "✅ PASS" if result.status == "PASS" else "❌ FAIL"
        
        lines = [
            f"# Model Quality Certification Report: {result.test_id}",
            "",
            f"## Status: {badge}",
            f"**Model ID**: {result.metrics['model_id']}",
            f"**Timestamp**: {result.timestamp}",
            "",
            "## Certification Criteria",
            "### Met Criteria",
        ]
        for m in result.criteria_met:
            lines.append(f"- {m}")
        
        if result.criteria_failed:
            lines.append("\n### Failed Criteria")
            for f in result.criteria_failed:
                lines.append(f"- {f}")
        
        lines.append("\n## Detailed Metrics")
        lines.append(f"- Success Rate: {result.metrics['success_rate']*100:.2f}%")
        lines.append(f"- Total Requests: {result.metrics['total_requests']}")
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        print(f"Certification report saved to {path}")

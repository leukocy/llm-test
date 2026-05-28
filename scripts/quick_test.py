"""
Comprehensive test for all benchmark types without UI
"""
import asyncio
import sys
import os
# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
from core.benchmark_runner import BenchmarkRunner


async def test_all():
    # Mock UI components
    placeholder = MagicMock()
    progress_bar = MagicMock()
    status_text = MagicMock()
    log_placeholder = MagicMock()

    runner = BenchmarkRunner(
        placeholder=placeholder,
        progress_bar=progress_bar,
        status_text=status_text,
        api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000/v1"),
        model_id="DeepSeek-V3.1",
        tokenizer_option="字符数 (Fallback)",
        csv_filename="quick_test.csv",
        api_key="",
        log_placeholder=log_placeholder,
        provider="openai"
    )

    print("=" * 70)
    print(f"API: {runner.api_base_url}")
    print(f"Model: {runner.model_id}")
    print("=" * 70)

    all_results = {}

    # ============================================================
    # 1. Concurrency Test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: Concurrency Test (1, 2, 4 concurrency)")
    print("=" * 70)
    try:
        results = await runner.run_concurrency_test(
            selected_concurrencies=[1, 2, 4],
            rounds_per_level=1,
            max_tokens=100,
            context_length_target=512
        )
        all_results['concurrency'] = results
        if not results.empty:
            print(f"\nResults ({len(results)} rows):")
            for _, r in results.iterrows():
                print(f"  Concurrency {int(r['concurrency'])}: TTFT={r['ttft']:.4f}s, "
                      f"TPS={r['tps']:.2f}, Output={r['system_output_throughput']:.2f} tok/s")
        else:
            print("  No results returned!")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # 2. Prefill Test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: Prefill Test (256, 512, 1024 tokens)")
    print("=" * 70)
    try:
        results = await runner.run_prefill_test(
            token_levels=[256, 512, 1024],
            requests_per_level=1,
            max_tokens=50
        )
        all_results['prefill'] = results
        if not results.empty:
            print(f"\nResults ({len(results)} rows):")
            for _, r in results.iterrows():
                print(f"  {int(r['context_length_target'])} tokens: TTFT={r['ttft']:.4f}s, "
                      f"Prefill Speed={r['prefill_speed']:.2f} tok/s")
        else:
            print("  No results returned!")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # 3. Long Context Test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: Long Context Test (1K, 2K, 4K tokens)")
    print("=" * 70)
    try:
        results = await runner.run_long_context_test(
            context_lengths=[1024, 2048, 4096],
            rounds_per_level=1,
            max_tokens=50
        )
        all_results['long_context'] = results
        if not results.empty:
            print(f"\nResults ({len(results)} rows):")
            for _, r in results.iterrows():
                print(f"  {int(r['context_length_target'])} tokens: TTFT={r['ttft']:.4f}s, "
                      f"TPS={r['tps']:.2f}, Prefill={r['prefill_speed']:.2f} tok/s")
        else:
            print("  No results returned!")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # 4. Throughput Matrix Test (concurrency × context)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 4: Throughput Matrix Test (1,2 × 256,512)")
    print("=" * 70)
    try:
        results = await runner.run_throughput_matrix_test(
            concurrencies=[1, 2],
            context_lengths=[256, 512],
            rounds=1,
            max_tokens=50
        )
        all_results['matrix'] = results
        if not results.empty:
            print(f"\nResults ({len(results)} rows):")
            for _, r in results.iterrows():
                print(f"  Concurrency={int(r['concurrency'])}, Context={int(r['context_length_target'])}: "
                      f"Total Throughput={r['system_total_throughput']:.2f} tok/s")
        else:
            print("  No results returned!")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # 5. Custom Text Test
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 5: Custom Text Test")
    print("=" * 70)
    try:
        results = await runner.run_custom_text_test(
            selected_concurrencies=[1],
            rounds_per_level=1,
            base_prompt="Explain quantum computing in simple terms",
            suffix_instruction="Keep the answer under 100 words.",
            max_tokens=100
        )
        all_results['custom_text'] = results
        if not results.empty:
            print(f"\nResults ({len(results)} rows):")
            for _, r in results.iterrows():
                print(f"  TTFT={r['ttft']:.4f}s, TPS={r['tps']:.2f}")
        else:
            print("  No results returned!")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # 6. Stability Test (short 10s)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 6: Stability Test (10 seconds)")
    print("=" * 70)
    try:
        results = await runner.run_stability_test(
            concurrency=1,
            duration_seconds=10,
            max_tokens=50,
            context_length_target=256
        )
        all_results['stability'] = results
        if not results.empty:
            print(f"\nResults ({len(results)} rows):")
            avg_ttft = results['ttft'].mean()
            avg_tps = results['tps'].mean()
            print(f"  Average TTFT: {avg_ttft:.4f}s")
            print(f"  Average TPS: {avg_tps:.2f} tokens/s")
            print(f"  Total Requests: {len(results)}")
        else:
            print("  No results returned!")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY - All Tests Completed")
    print("=" * 70)
    for test_name, results in all_results.items():
        if not results.empty:
            print(f"  {test_name}: {len(results)} result(s)")
        else:
            print(f"  {test_name}: No results")

    return all_results


if __name__ == "__main__":
    print("Running comprehensive benchmark tests...\n")
    results = asyncio.run(test_all())

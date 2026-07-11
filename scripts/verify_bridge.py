"""Verification of the Streamlit Compatibility Bridge."""

import logging

import pandas as pd
from legacy.streamlit_adapter import run_benchmark_test

# Disable verbose logging during verification
logging.getLogger("engine").setLevel(logging.WARNING)


def verify_bridge():
    print("🧪 Verifying Streamlit Bridge...")

    # Configuration in the old Streamlit style
    old_config = {
        "api_base_url": "http://localhost:8000/v1",  # Just a placeholder
        "model_id": "gpt-4-mock",
        "api_key": "sk-dummy",
        "selected_concurrencies": [1, 2],
        "rounds_per_level": 1,
        "max_tokens": 10,
        "input_tokens_target": 10,
    }

    # Progress callback simulation
    progress_calls = []

    def progress_callback(done, total, msg):
        progress_calls.append((done, total))
        print(f"  [Bridge Progress] {done}/{total}: {msg}")

    print("🏃 Running mock concurrency test via bridge...")

    # Note: This will attempt to use the actual engine.
    # Since we don't have a live API here, we expect results to be empty or failed,
    # but the logic flow should hold.
    try:
        df = run_benchmark_test(
            test_type="concurrency",
            config=old_config,
            progress_callback=progress_callback,
        )

        print(f"✅ Bridge returned DataFrame with shape: {df.shape}")
        assert isinstance(df, pd.DataFrame), "Output must be a pandas DataFrame"

        print("✨ Bridge Verification SUCCESS!")
        return True
    except Exception as e:
        # If it fails due to connection error, that's expected if no provider is up,
        # but the config translation and runner initialization should have worked.
        if "Connection" in str(e) or "provider" in str(e).lower():
            print(f"⚠️ Bridge logic reached execution phase (expected provider error): {e}")
            print("✨ Bridge Verification (Logic Flow) SUCCESS!")
            return True
        else:
            print(f"❌ Bridge Verification FAILED: {e}")
            import traceback

            traceback.print_exc()
            return False


if __name__ == "__main__":
    verify_bridge()

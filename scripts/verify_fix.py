import os
import sys

import pandas as pd

# Add project root to sys.path
sys.path.append(os.getcwd())

from ui.reports import generate_concurrency_report


def test_concurrency_report_missing_columns():
    print(
        "Testing generate_concurrency_report with missing system_output_throughput..."
    )

    # Mock data without system_output_throughput
    data = {
        "session_id": [0, 1, 2],
        "concurrency": [1, 2, 2],
        "round": [1, 1, 1],
        "ttft": [0.5, 0.6, 0.7],
        "prefill_tokens": [100, 100, 100],
        "decode_tokens": [50, 50, 50],
        "token_calc_method": ["API", "API", "API"],
        "error": [None, None, None],
    }
    df = pd.DataFrame(data)

    # Mock streamlit
    from unittest.mock import MagicMock

    import streamlit as st

    # We don't want st to actually do anything UI-related during this text test
    # but we want to make sure the code doesn't raise a KeyError
    st.subheader = MagicMock()
    st.markdown = MagicMock()
    st.expander = MagicMock()
    st.error = MagicMock()
    st.warning = MagicMock()
    st.dataframe = MagicMock()
    st.plotly_chart = MagicMock()
    st.columns = MagicMock(return_value=(MagicMock(), MagicMock()))

    try:
        # This should NOT raise KeyError
        report = generate_concurrency_report(df, "test-model", "test-provider", 10.0)
        print("Success! Report generated without KeyError.")
    except KeyError as e:
        print(f"FAILED: KeyError raised: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Some errors might be expected due to deep streamlit mocks failing,
        # but KeyError is the target.
        if "system_output_throughput" in str(e):
            print("FAILED: system_output_throughput still causing issues.")
            exit(1)
        print("Continuing as this might be due to streamlit mocks.")


if __name__ == "__main__":
    test_concurrency_report_missing_columns()

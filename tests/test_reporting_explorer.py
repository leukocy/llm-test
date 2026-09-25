"""Request explorer filters are views over the complete recorded run."""

import pandas as pd

from ui.page_layout import filter_result_rows


def test_request_explorer_filters_status_and_literal_text_without_mutating_source():
    rows = pd.DataFrame(
        {
            "session_id": ["ok[1]", "bad[2]", "ok[3]"],
            "error": [None, "timeout", ""],
            "test_type": ["concurrency"] * 3,
        }
    )

    assert len(filter_result_rows(rows, "Succeeded")) == 2
    assert len(filter_result_rows(rows, "Failed")) == 1
    assert filter_result_rows(rows, query="[2]").iloc[0]["error"] == "timeout"
    assert len(rows) == 3

import pandas as pd

from ui.insights import _get_col
from ui.reporting.columns import COLUMN_RENAME_MAP


def test_insight_column_lookup_handles_renamed_tpot_chunk_columns():
    tpot_p99_label = COLUMN_RENAME_MAP["TPOT_P99"]
    df = pd.DataFrame({tpot_p99_label: [120.0]})

    assert _get_col(df, "TPOT_P99") == tpot_p99_label

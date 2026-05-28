import pandas as pd
import pytest

from core.result_metrics import (
    calculate_success_rate_percent,
    sanitize_performance_metrics,
    success_mask_from_error,
)


def test_blank_error_values_count_as_success():
    errors = pd.Series([None, "", "  ", "null", "timeout"])

    assert success_mask_from_error(errors).tolist() == [True, True, True, True, False]
    assert calculate_success_rate_percent(errors) == pytest.approx(80.0)


def test_zero_performance_values_are_masked_but_counts_are_preserved():
    df = pd.DataFrame({"ttft": [0, 1.2], "Num_Requests": [0, 2]})

    result = sanitize_performance_metrics(df)

    assert pd.isna(result.loc[0, "ttft"])
    assert result.loc[0, "Num_Requests"] == 0

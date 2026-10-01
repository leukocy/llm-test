"""Offline bars preserve units, genuine zero, bounds and escaped labels."""

from server.robustness_figures import sensitivity_svg


def test_sensitivity_bars_sort_and_escape_without_fabricating_missing_scores():
    svg = sensitivity_svg(
        {"zero": 0, "<script>": 0.5, "missing": None, "nan": float("nan"), "bad": 2, "bool": True}
    )
    assert "&lt;script&gt;：50.0%" in svg and "zero：0.0%" in svg
    assert svg.index("&lt;script&gt;") < svg.index("zero")
    assert 'width="200.00"' in svg and 'width="0.00"' in svg
    assert "<script>" not in svg and "missing" not in svg and "nan" not in svg
    assert "不是显著性或合格标准" in svg
    assert "没有有效" in sensitivity_svg({"missing": None})

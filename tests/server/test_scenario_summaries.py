"""Initial scenario cards retain comparisons without inventing missing evidence."""

import pytest

from server.analytics import describe_observations
from server.extended_analytics import extended_observations
from server.scenario_summaries import summary_cards, token_totals


def row(**updates):
    return dict(
        error=None,
        ttft=1,
        tpot=0.1,
        tps=10,
        total_time=2,
        prefill_speed=100,
        prefill_tokens=100,
        decode_tokens=10,
        **updates,
    )


def summary(kind, levels=(1, 2, 3), **updates):
    field = (
        "concurrency_level"
        if kind == "concurrency"
        else "input_tokens_target"
        if kind == "prefill"
        else "context_length_target"
    )
    groups = []
    rows = []
    for level in levels:
        item = row()
        item["ttft"] = level
        rows.append(item)
        groups.append(
            {
                **describe_observations([item]),
                "label": f"条件 {level}",
                "dimensions": {field: level, "concurrency_level": level},
            }
        )
    return dict(
        run={"test_type": kind},
        overall=describe_observations(rows),
        groups=groups,
        group_axis="条件",
        **updates,
    )


def cards(data):
    return {card["key"]: card for card in summary_cards(data)}


@pytest.mark.parametrize(
    "kind",
    [
        "concurrency",
        "custom_text",
        "prefill",
        "long_context",
        "segmented_prefill",
        "throughput_matrix",
        "stability",
    ],
)
def test_all_scenarios_keep_success_token_totals_and_unique_cards(kind):
    result = summary_cards(summary(kind))
    assert len({c["key"] for c in result}) == len(result)
    assert cards(summary(kind))["prefill_tokens"]["value"] == "300"


def test_token_total_coverage_excludes_failure_invalid_and_keeps_zero():
    rows = [row(), row(), row(), row(), row()]
    rows[1].update(error="failed")
    rows[2].update(prefill_tokens=None)
    rows[3].update(prefill_tokens=True)
    rows[4].update(prefill_tokens=0, decode_tokens=-1)
    result = token_totals(rows)
    assert result["prefill_tokens"] == {"total": 100, "count": 2, "successes": 4}
    assert result["decode_tokens"] == {"total": 30, "count": 3, "successes": 4}


def test_missing_planned_endpoint_not_replaced_by_observed_extreme():
    data = summary(
        "concurrency",
        (2, 3),
        measurement_protocol={
            "cells": [{"concurrency": 1}, {"concurrency": 2}, {"concurrency": 3}]
        },
    )
    result = cards(data)
    assert result["ttft_first"]["value"] == "未采集 s"
    assert result["ttft_ratio"]["value"] == "未采集 ×"
    assert "条件 2 → 条件 3" in result["ttft_adjacent"]["note"]


def test_missing_middle_does_not_create_adjacent_comparison():
    data = summary(
        "prefill",
        (1, 3),
        measurement_protocol={"cells": [{"input_tokens_target": level} for level in (1, 2, 3)]},
    )
    assert cards(data)["prefill_speed_adjacent"]["value"] == "未采集 %"


def test_concurrency_jump_selects_initial_absolute_increment_then_reports_percent():
    data = summary("concurrency")
    for group, ttft in zip(data["groups"], (1, 3, 6), strict=True):
        group["metrics"]["ttft"]["mean"] = ttft
    result = cards(data)["ttft_adjacent"]
    assert result["value"] == "100 %"
    assert "条件 2 → 条件 3" in result["note"]
    assert "按绝对增量选择，增量 3 s" in result["note"]


def test_long_context_cv_uses_equal_condition_means_and_requires_all():
    data = summary("long_context", (100, 200))
    data["groups"][0]["metrics"]["tps"].update(mean=10, count=100)
    data["groups"][1]["metrics"]["tps"].update(mean=30, count=1)
    assert cards(data)["tps_cv"]["value"] == "50 %"
    data["groups"][1]["metrics"]["tps"]["mean"] = None
    assert cards(data)["tps_cv"]["value"] == "未采集 %"


def test_segmented_cache_is_token_weighted_source_separated_and_zero_cache_explicit():
    rows = [row() for _ in range(4)]
    rows[0].update(cache_hit_source="API", cache_hit_tokens=0, api_prefill=100)
    rows[1].update(cache_hit_source="API", cache_hit_tokens=900, api_prefill=900)
    rows[2].update(
        cache_hit_source="TTFT_inferred", cache_hit_tokens=25, effective_prefill_tokens=50
    )
    extended = extended_observations(rows)
    assert extended["cache"]["weighted"]["API"] == {
        "hits": 900,
        "tokens": 1000,
        "count": 2,
        "rate": 90,
    }
    data = summary("segmented_prefill")
    data["extended_observations"] = extended
    data["groups"][0]["metrics"].update(extended["metrics"])
    data["groups"][0]["extended_observations"] = extended
    result = cards(data)
    assert result["cache_total_API"]["value"] == "90 %"
    assert result["cache_total_TTFT_inferred"]["value"] == "50 %"
    assert result["ttft_zero_cache_api_first"]["value"] == "1 s"
    assert result["ttft_zero_cache_api_last"]["value"] == "未采集 s"
    assert result["cache_change"]["value"] == "未采集 pp"


def test_throughput_never_falls_back_to_single_request_tps_times_concurrency():
    result = cards(summary("concurrency"))
    assert result["phase_output_max_high"]["value"] == "未采集 token/s"
    assert "不证明饱和容量" in result["throughput_gain"]["note"]


def test_prefill_initial_growth_rule_has_formula_and_requires_three_conditions():
    result = cards(summary("prefill"))
    assert result["ttft_pattern"]["value"] == "近线性规则"
    assert "不是回归拟合" in result["ttft_pattern"]["note"]
    assert cards(summary("prefill", (1, 2)))["ttft_pattern"]["value"] == "无法判断"

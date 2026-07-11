"""Tests for stable test-type identifiers and presentation metadata."""

import re

import pytest


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("concurrency", "concurrency"),
        ("Concurrency Test", "concurrency"),
        ("prefill_test", "prefill"),
        ("Prefill Stress Test", "prefill"),
        ("legacy-prefix Prefill Stress Test", "prefill"),
        ("Concurrency-Context Matrix Test", "matrix"),
        ("Model Quality Test", "quality"),
        ("A/B Model Comparison", "comparison"),
    ],
)
def test_normalizes_ids_labels_and_legacy_prefixed_labels(value, expected):
    from config.test_types import normalize_test_type

    assert normalize_test_type(value) == expected


def test_registry_has_stable_order_and_plain_labels():
    from config.test_types import TEST_TYPE_IDS, test_type_label

    assert TEST_TYPE_IDS == (
        "concurrency",
        "prefill",
        "segmented",
        "long_context",
        "matrix",
        "custom",
        "all",
        "stability",
        "environment",
        "batch",
        "quality",
        "comparison",
        "advanced",
        "data_warehouse",
    )
    assert test_type_label("concurrency") == "Concurrency Test"
    assert test_type_label("unknown") == "Concurrency Test"
    assert all(not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", test_type_label(item)) for item in TEST_TYPE_IDS)


def test_registry_exposes_native_material_icon_names():
    from config.test_types import test_type_icon

    assert test_type_icon("concurrency") == "speed"
    assert test_type_icon("quality") == "fact_check"
    assert test_type_icon("unknown") == "speed"


def test_allowed_options_fall_back_to_first_available_id():
    from config.test_types import normalize_test_type

    assert normalize_test_type("missing", ("prefill", "matrix")) == "prefill"
    assert normalize_test_type("Prefill Stress Test", ("prefill", "matrix")) == "prefill"

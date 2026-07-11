"""Stable test-type identifiers and presentation metadata."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TestTypeSpec:
    """Describe one test type without coupling its ID to presentation text."""

    id: str
    label: str
    material_icon: str
    aliases: tuple[str, ...] = ()


TEST_TYPE_SPECS = (
    TestTypeSpec("concurrency", "Concurrency Test", "speed", ("concurrency_test",)),
    TestTypeSpec(
        "prefill",
        "Prefill Stress Test",
        "input",
        ("prefill_test", "prefill_stress", "prefill_stress_test"),
    ),
    TestTypeSpec(
        "segmented",
        "Segmented Context Test",
        "view_timeline",
        ("segmented_context", "segmented_context_test", "segmented_prefill"),
    ),
    TestTypeSpec(
        "long_context",
        "Long Context Test",
        "article",
        ("long_context_test",),
    ),
    TestTypeSpec(
        "matrix",
        "Concurrency-Context Matrix Test",
        "grid_view",
        (
            "throughput_matrix",
            "concurrency_context_matrix",
            "concurrency_context_matrix_test",
        ),
    ),
    TestTypeSpec("custom", "Custom Text Test", "edit_note", ("custom_text", "custom_text_test")),
    TestTypeSpec("all", "All Tests", "select_all", ("all_tests",)),
    TestTypeSpec("stability", "Stability Test", "monitor_heart", ("stability_test",)),
    TestTypeSpec(
        "environment",
        "Environment Information",
        "memory",
        ("environment_info", "环境信息"),
    ),
    TestTypeSpec("batch", "Batch Test", "stacks"),
    TestTypeSpec("quality", "Model Quality Test", "fact_check", ("dataset",)),
    TestTypeSpec("comparison", "A/B Model Comparison", "compare_arrows", ("ab_comparison",)),
    TestTypeSpec("advanced", "Advanced Evaluation", "science", ("advanced_evaluation",)),
    TestTypeSpec("data_warehouse", "Data Warehouse", "database", ("warehouse", "data_warehouse_test")),
)

TEST_TYPE_IDS = tuple(spec.id for spec in TEST_TYPE_SPECS)
DEFAULT_TEST_TYPE = "concurrency"

_SPECS_BY_ID = {spec.id: spec for spec in TEST_TYPE_SPECS}
_ALIASES = {
    alias: spec.id
    for spec in TEST_TYPE_SPECS
    for alias in (spec.id, *spec.aliases)
}


def _comparison_key(value: object) -> str:
    return (
        str(value or "")
        .strip()
        .casefold()
        .replace("/", " ")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _normalize_known(value: object) -> str | None:
    text = str(value or "").strip()
    if "\u6570\u636e\u4ed3\u5e93" in text:
        return "data_warehouse"
    key = _comparison_key(text)
    if key in _ALIASES:
        return _ALIASES[key]

    folded = text.casefold()
    for spec in TEST_TYPE_SPECS:
        if folded.endswith(spec.label.casefold()):
            return spec.id
    return None


def normalize_test_type(
    value: object,
    allowed_options: Sequence[str] | None = None,
) -> str:
    """Normalize IDs, legacy aliases, and display labels to a stable ID."""

    selected = _normalize_known(value) or DEFAULT_TEST_TYPE
    if not allowed_options:
        return selected

    allowed_ids = tuple(
        normalized
        for option in allowed_options
        if (normalized := _normalize_known(option)) is not None
    )
    if not allowed_ids:
        return DEFAULT_TEST_TYPE
    return selected if selected in allowed_ids else allowed_ids[0]


def test_type_label(value: object) -> str:
    """Return the user-facing label for an ID or legacy value."""

    return _SPECS_BY_ID[normalize_test_type(value)].label


def test_type_icon(value: object) -> str:
    """Return the Material Symbol name for an ID or legacy value."""

    return _SPECS_BY_ID[normalize_test_type(value)].material_icon

"""Strict Streamlit AppTest smoke checks executed outside pytest's Streamlit mock.

Run: ``python tests/e2e_app_smoke.py``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from config.test_types import TEST_TYPE_IDS, test_type_label

EMOJI_PATTERN = re.compile(
    r"["
    r"\U0001F1E6-\U0001F1FF"
    r"\U0001F300-\U0001FAFF"
    r"\u2139"
    r"\u2300-\u23FF"
    r"\u25A0-\u27BF"
    r"\u2B00-\u2BFF"
    r"\uFE0F"
    r"]"
)


def load_app() -> AppTest:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
    at.run(timeout=30)
    assert not at.exception, f"App crashed on load:\n{at.exception}"
    return at


def find_test_type_select(at: AppTest):
    selector = next(
        (item for item in at.sidebar.selectbox if item.label == "Select Test Type"),
        None,
    )
    assert selector is not None, "Could not locate the test type selector"
    return selector


def rendered_copy(at: AppTest) -> str:
    values: list[str] = []
    for collection_name in (
        "button",
        "caption",
        "error",
        "header",
        "info",
        "markdown",
        "subheader",
        "success",
        "text",
        "title",
        "warning",
    ):
        for element in getattr(at, collection_name):
            value = getattr(element, "value", None)
            label = getattr(element, "label", None)
            text = value if isinstance(value, str) else label
            if isinstance(text, str) and not text.lstrip().startswith("<style>"):
                values.append(text)
    return "\n".join(values)


def verify_default_page(at: AppTest) -> None:
    selector = find_test_type_select(at)
    expected_labels = [test_type_label(test_type) for test_type in TEST_TYPE_IDS]
    assert list(selector.options) == expected_labels

    button_labels = [button.label for button in at.button]
    assert "Run concurrency test" in button_labels

    copy = rendered_copy(at)
    assert "LLM Benchmark Platform" in copy
    assert "[DEBUG]" not in copy
    assert not EMOJI_PATTERN.search(copy), "Rendered user-visible copy contains Emoji"
    print("PASS: Default page copy and primary action verified")


def verify_provider_and_model(at: AppTest) -> None:
    provider_select = at.sidebar.selectbox[0]
    providers = list(provider_select.options)
    assert providers, "Provider selector has no options"
    target_provider = next(
        (
            provider
            for provider in providers
            if provider != "Custom (OpenAI Compatible)"
        ),
        providers[0],
    )
    provider_select.set_value(target_provider).run()
    assert not at.exception, f"Provider change crashed the app:\n{at.exception}"

    model_select = at.sidebar.selectbox[1]
    models = list(model_select.options)
    assert models, "Model selector has no options"
    model_select.set_value(models[0]).run()
    assert not at.exception, f"Model change crashed the app:\n{at.exception}"
    print("PASS: Provider and model selection verified")


def verify_custom_prompt_flow(at: AppTest) -> None:
    find_test_type_select(at).set_value("custom").run()
    assert not at.exception, f"Custom test panel crashed:\n{at.exception}"

    source_mode = next(
        (radio for radio in at.main.radio if radio.label == "Prompt Source"),
        None,
    )
    assert source_mode is not None, "Prompt Source control is missing"
    source_mode.set_value("Manual Input").run()

    prompt_area = next(
        (
            area
            for area in at.main.text_area
            if area.label and "prompt" in area.label.casefold()
        ),
        None,
    )
    assert prompt_area is not None, "Custom prompt input is missing"
    prompt_area.set_value("Hello, benchmark test!").run()
    assert (
        not at.exception
    ), f"Entering a custom prompt crashed the app:\n{at.exception}"
    print("PASS: Custom prompt flow verified")


def verify_advanced_panels() -> None:
    for test_type in ("quality", "comparison", "advanced", "batch", "data_warehouse"):
        at = load_app()
        find_test_type_select(at).set_value(test_type).run()
        assert not at.exception, f"{test_type} panel crashed:\n{at.exception}"
    print("PASS: Advanced panels verified")


def main() -> None:
    print("=" * 60)
    print("E2E Smoke Test: Streamlit User Flows")
    print("=" * 60)

    at = load_app()
    verify_default_page(at)
    verify_provider_and_model(at)
    verify_custom_prompt_flow(at)
    verify_advanced_panels()

    print("=" * 60)
    print("All strict smoke checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Regression policy for user-visible runtime text."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

PRIMARY_SURFACES = (
    "app.py",
    "config/auth.py",
    "ui/sidebar.py",
    "ui/test_panels.py",
    "ui/test_control_panel.py",
    "ui/page_layout.py",
    "ui/onboarding.py",
    "ui/test_runner.py",
)

REPORTING_SURFACES = (
    "ui/insights.py",
    "ui/markdown_summary.py",
    "ui/reports.py",
    "ui/export.py",
    "ui/quality_reports.py",
    "ui/evaluation_dashboard.py",
    "ui/thinking_components.py",
    "ui/static_chart_generator.py",
    "ui/styled_tables.py",
)

RUNTIME_SOURCE_ROOTS = (
    "config",
    "core",
    "evaluators",
    "scripts",
    "ui",
    "utils",
)

# Modules that intentionally retain emoji as data (never rendered to users):
#   ui/icons.py — EMOJI_TO_ICON is a legacy-emoji → SVG-icon migration map whose
#   keys are the historical emoji characters. These are dictionary keys, not
#   rendered output, and the registry is covered by tests/test_ui_icons.py and
#   consumed by ui.components._strip_emoji. Excluding it keeps the policy focused
#   on user-visible text rather than migration metadata.
RUNTIME_EMOJI_EXEMPT = frozenset({"ui/icons.py"})

USER_DOCUMENTATION = (
    "ARCHITECTURE.md",
    "README.md",
    "core/mqc_readme.md",
    *tuple(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in sorted((PROJECT_ROOT / "docs").rglob("*.md"))
    ),
)

USER_VISIBLE_SHELL_SCRIPTS = tuple(
    str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    for path in sorted((PROJECT_ROOT / "scripts").rglob("*.sh"))
)

REPOSITORY_VISIBLE_FILES = tuple(
    str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    for path in sorted((PROJECT_ROOT / ".github").rglob("*"))
    if path.is_file() and path.suffix.casefold() in {".md", ".yml", ".yaml"}
)

PRODUCTION_COPY_SURFACES = (
    "core/result_comparator.py",
    "ui/batch_test.py",
    "ui/test_panels.py",
    "utils/test_config_manager.py",
)

CORRUPTED_COPY_MARKERS = (
    "[DEBUG]",
    "hYestory",
    "Test Configuration预设",
    "Save Current Configis预设",
    "under载Configure文件",
    "no对比Data",
    "Test Results对比报告",
)


def _emoji_occurrences(relative_paths: tuple[str, ...]) -> list[str]:
    occurrences = []
    for relative_path in relative_paths:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), start=1):
            matches = "".join(EMOJI_PATTERN.findall(line))
            if matches:
                occurrences.append(f"{relative_path}:{line_number}: {matches}")
    return occurrences


def test_primary_surfaces_have_no_emoji():
    occurrences = _emoji_occurrences(PRIMARY_SURFACES)

    assert not occurrences, "Emoji found in primary UI sources:\n" + "\n".join(
        occurrences
    )


def test_reporting_surfaces_have_no_emoji():
    occurrences = _emoji_occurrences(REPORTING_SURFACES)

    assert not occurrences, "Emoji found in reporting UI sources:\n" + "\n".join(
        occurrences
    )


def test_runtime_sources_have_no_emoji():
    runtime_paths = ["app.py"]
    for source_root in RUNTIME_SOURCE_ROOTS:
        runtime_paths.extend(
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for path in sorted((PROJECT_ROOT / source_root).rglob("*.py"))
        )
    runtime_paths = [p for p in runtime_paths if p not in RUNTIME_EMOJI_EXEMPT]

    occurrences = _emoji_occurrences(tuple(runtime_paths))

    assert not occurrences, "Emoji found in runtime sources:\n" + "\n".join(occurrences)


def test_user_documentation_has_no_emoji():
    occurrences = _emoji_occurrences(USER_DOCUMENTATION)

    assert not occurrences, "Emoji found in user documentation:\n" + "\n".join(
        occurrences
    )


def test_user_visible_shell_scripts_have_no_emoji():
    occurrences = _emoji_occurrences(USER_VISIBLE_SHELL_SCRIPTS)

    assert not occurrences, "Emoji found in shell-script output:\n" + "\n".join(
        occurrences
    )


def test_repository_visible_files_have_no_emoji():
    occurrences = _emoji_occurrences(REPOSITORY_VISIBLE_FILES)

    assert not occurrences, "Emoji found in repository-visible files:\n" + "\n".join(
        occurrences
    )


def test_production_copy_has_no_debug_or_corrupted_markers():
    combined_source = "\n".join(
        (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for relative_path in PRODUCTION_COPY_SURFACES
    )

    found = [marker for marker in CORRUPTED_COPY_MARKERS if marker in combined_source]
    assert not found, f"Debug or corrupted user-visible copy found: {found}"

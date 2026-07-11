"""Tests for the branch-scoped Ruff gate."""

from scripts.lint_changed import parse_changed_python_files


def test_parse_changed_python_files_keeps_only_python_sources():
    output = "ui/sidebar.py\ndocs/guide.md\ntests/test_ui.py\n\n"

    assert parse_changed_python_files(output) == ["ui/sidebar.py", "tests/test_ui.py"]

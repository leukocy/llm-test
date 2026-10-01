"""Rebuild the first-commit Streamlit control and output inventories.

Run from the repository root with ``python docs/ui_parity/extract_legacy_controls.py``.
The commit ID is immutable; no archived checkout is required.
"""

from __future__ import annotations

import ast
import csv
import json
import re

# Fixed git commands read an immutable local commit.
import subprocess  # nosec B404
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = Path(__file__).resolve().parent
BASELINES = {
    "first_commit": "2e1ff54",
}
UNROUTED_MODULES = {
    "config/auth.py",
    "ui/comparison_page.py",
    "ui/dashboard_components.py",
    "ui/dataset_manager.py",
    "ui/evaluation_dashboard.py",
    "ui/history_browser.py",
    "ui/thinking_components.py",
    "utils/test_config_manager.py",
}
UNROUTED_FUNCTIONS = {
    ("app.py", "render_preset_management"),
    ("app.py", "render_custom_config"),
    ("ui/export.py", "create_excel_download_link"),
    ("ui/onboarding.py", "render_quick_reference"),
    ("ui/static_chart_generator.py", "generate_static_html_report"),
    ("utils/helpers.py", "get_table_download_link"),
    ("utils/helpers.py", "get_log_download_link"),
}
BASELINE_BLOCKED_MODULES = {"ui/quality_reports.py"}
BASELINE_BLOCKED_FUNCTIONS = {
    ("ui/advanced_panels.py", "_render_dataset_status_panel"),
    ("ui/advanced_panels.py", "render_quality_test_panel"),
    ("ui/advanced_panels.py", "render_ab_comparison_panel"),
    ("ui/advanced_panels.py", "_display_ab_comparison_results"),
}
CONTROLS = {
    "button",
    "form_submit_button",
    "download_button",
    "selectbox",
    "multiselect",
    "radio",
    "checkbox",
    "toggle",
    "slider",
    "number_input",
    "text_input",
    "text_area",
    "file_uploader",
    "data_editor",
    "date_input",
    "time_input",
    "color_picker",
    "segmented_control",
    "pills",
    "select_slider",
    "chat_input",
    "tabs",
    "expander",
    "form",
}
OUTPUTS = {
    "dataframe",
    "table",
    "metric",
    "plotly_chart",
    "altair_chart",
    "pyplot",
    "line_chart",
    "bar_chart",
    "area_chart",
    "image",
    "json",
    "code",
    "graphviz_chart",
}
HTML_ACTION_FIELDS = ["id", "source", "reachability", "function", "source_line", "source_url"]
FIELDS = [
    "id",
    "source",
    "reachability",
    "function",
    "container",
    "kind",
    "label_or_data",
    "data_expression",
    "delta_expression",
    "options",
    "options_definition",
    "value_or_default",
    "min_value",
    "max_value",
    "step",
    "index",
    "format",
    "format_func",
    "input_type",
    "file_types",
    "multiple_files",
    "file_name",
    "mime",
    "help",
    "placeholder",
    "expanded",
    "key",
    "disabled",
    "on_change",
    "visible_when",
    "handler",
    "source_url",
]


def git(*args: str) -> str:
    # No shell; arguments come only from fixed commit and repository paths.
    return subprocess.check_output(  # nosec B603, B607
        ["git", *args], cwd=ROOT, text=True
    )


def source_paths(commit: str) -> list[str]:
    ui_paths = {
        path
        for path in git("ls-tree", "-r", "--name-only", commit, "--", "app.py", "ui").splitlines()
        if path.endswith(".py")
    }
    streamlit_paths = {
        line.split(":", 1)[1]
        for line in git(
            "grep", "-l", "-E", "import streamlit|from streamlit", commit, "--", "*.py"
        ).splitlines()
    }
    return sorted(
        (
            path
            for path in ui_paths | streamlit_paths
            if path.endswith(".py") and not path.startswith(("tests/", "scripts/"))
        ),
        key=lambda path: (0 if path == "app.py" or path.startswith("ui/") else 1, path),
    )


def reachability(path: str, function: str) -> str:
    if path in UNROUTED_MODULES or (path, function) in UNROUTED_FUNCTIONS:
        return "source_only_not_routed_from_app"
    if path in BASELINE_BLOCKED_MODULES or (path, function) in BASELINE_BLOCKED_FUNCTIONS:
        return "routed_but_blocked_in_first_commit"
    return "app_routed_source"


def expression(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ast.unparse(node)


def keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def argument(call: ast.Call, name: str, position: int) -> ast.AST | None:
    return keyword(call, name) or (call.args[position] if len(call.args) > position else None)


def enclosing(
    node: ast.AST,
    parents: dict[ast.AST, ast.AST],
    kind: type[ast.AST] | tuple[type[ast.AST], ...],
) -> ast.AST | None:
    parent = parents.get(node)
    while parent is not None:
        if isinstance(parent, kind):
            return parent
        parent = parents.get(parent)
    return None


def is_inside(root: ast.AST, target: ast.AST) -> bool:
    return any(child is target for child in ast.walk(root))


def option_definition(
    option: ast.AST | None, call: ast.Call, function: ast.AST | None, tree: ast.Module
) -> str:
    if not isinstance(option, ast.Name):
        return ""
    scope = function or tree
    assignments = [
        node
        for node in ast.walk(scope)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and node.lineno < call.lineno
        and any(
            isinstance(target, ast.Name) and target.id == option.id
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if not assignments:
        return ""
    assignment = max(assignments, key=lambda item: item.lineno)
    return f"L{assignment.lineno}: {expression(assignment.value)}"


def context(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> tuple[str, str, str, str]:
    function = cast(
        ast.FunctionDef | ast.AsyncFunctionDef | None,
        enclosing(call, parents, (ast.FunctionDef, ast.AsyncFunctionDef)),
    )
    function_name = function.name if function else "<module>"
    container = ""
    guards = []
    handler = expression(keyword(call, "on_click"))
    child: ast.AST = call
    parent = parents.get(child)
    while parent is not None and parent is not function:
        if isinstance(parent, ast.With) and not container:
            context_expr = parent.items[0].context_expr
            container = expression(context_expr)
        if isinstance(parent, ast.If):
            if is_inside(parent.test, child):
                if not handler and parent.body:
                    handler = f"L{parent.body[0].lineno}-L{parent.body[-1].end_lineno}"
            elif any(is_inside(statement, child) for statement in parent.body):
                guards.append(expression(parent.test))
            elif any(is_inside(statement, child) for statement in parent.orelse):
                guards.append(f"not ({expression(parent.test)})")
        child = parent
        parent = parents.get(child)
    if (
        not handler
        and function
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in {"button", "form_submit_button"}
    ):
        assignment = cast(
            ast.Assign | ast.AnnAssign | None,
            enclosing(call, parents, (ast.Assign, ast.AnnAssign)),
        )
        if assignment and assignment.lineno == call.lineno:
            targets = (
                assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            )
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            matching = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.If)
                and node.lineno > call.lineno
                and any(
                    isinstance(part, ast.Name) and part.id in names for part in ast.walk(node.test)
                )
                and node.body
            ]
            if matching:
                condition = min(matching, key=lambda item: item.lineno)
                handler = f"L{condition.body[0].lineno}-L{condition.body[-1].end_lineno}"
    return function_name, container, " && ".join(reversed(guards)), handler


def rows_for(commit: str, kinds: set[str]) -> list[dict[str, str]]:
    rows = []
    full_commit = git("rev-parse", commit).strip()
    for path in source_paths(commit):
        tree = ast.parse(git("show", f"{commit}:{path}"), filename=path)
        parents = {
            child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
        }
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
                continue
            kind = call.func.attr
            if kind not in kinds:
                continue
            if kind == "json" and expression(call.func.value) != "st":
                continue
            function = enclosing(call, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
            name, container, guard, handler = context(call, parents)
            options = (
                argument(call, "options", 1)
                if kind
                in {
                    "selectbox",
                    "multiselect",
                    "radio",
                    "select_slider",
                    "segmented_control",
                    "pills",
                }
                else None
            )
            if kind == "tabs":
                options = call.args[0] if call.args else keyword(call, "tabs")
            if kind == "file_uploader":
                options = argument(call, "type", 1)
            label = argument(call, "label", 0)
            if kind == "tabs":
                label = None
            value_position = {
                "text_input": 1,
                "text_area": 1,
                "checkbox": 1,
                "toggle": 1,
                "number_input": 3,
                "slider": 3,
                "multiselect": 2,
                "date_input": 1,
                "time_input": 1,
            }.get(kind)
            value = keyword(call, "value") or keyword(call, "default")
            if value is None and value_position is not None and len(call.args) > value_position:
                value = call.args[value_position]
            index = keyword(call, "index")
            if index is None and kind in {"selectbox", "radio"} and len(call.args) > 2:
                index = call.args[2]
            row = {
                "source": f"{path}:{call.lineno}",
                "reachability": reachability(path, name),
                "function": name,
                "container": container,
                "kind": kind,
                "label_or_data": expression(label),
                "data_expression": expression(
                    argument(call, "data", 1)
                    if kind == "download_button"
                    else argument(call, "value", 1)
                    if kind == "metric"
                    else None
                ),
                "delta_expression": expression(argument(call, "delta", 2))
                if kind == "metric"
                else "",
                "options": expression(options),
                "options_definition": option_definition(options, call, function, tree),
                "value_or_default": expression(value),
                "min_value": expression(
                    keyword(call, "min_value")
                    or (
                        call.args[1]
                        if kind in {"number_input", "slider"} and len(call.args) > 1
                        else None
                    )
                ),
                "max_value": expression(
                    keyword(call, "max_value")
                    or (
                        call.args[2]
                        if kind in {"number_input", "slider"} and len(call.args) > 2
                        else None
                    )
                ),
                "step": expression(
                    keyword(call, "step")
                    or (
                        call.args[4]
                        if kind in {"number_input", "slider"} and len(call.args) > 4
                        else None
                    )
                ),
                "index": expression(index),
                "format": expression(keyword(call, "format")),
                "format_func": expression(keyword(call, "format_func")),
                "input_type": expression(keyword(call, "type")),
                "file_types": expression(options) if kind == "file_uploader" else "",
                "multiple_files": expression(keyword(call, "accept_multiple_files")),
                "file_name": expression(keyword(call, "file_name")),
                "mime": expression(keyword(call, "mime")),
                "help": expression(keyword(call, "help")),
                "placeholder": expression(keyword(call, "placeholder")),
                "expanded": expression(keyword(call, "expanded")),
                "key": expression(keyword(call, "key")),
                "disabled": expression(keyword(call, "disabled")),
                "on_change": expression(keyword(call, "on_change")),
                "visible_when": guard,
                "handler": handler,
                "source_url": f"https://github.com/leukocy/llm-test/blob/{full_commit}/{path}#L{call.lineno}",
            }
            rows.append(row)
    return sorted(
        rows,
        key=lambda item: (
            0 if item["source"].startswith("app.py:") or item["source"].startswith("ui/") else 1,
            item["source"].rsplit(":", 1)[0],
            int(item["source"].rsplit(":", 1)[1]),
        ),
    )


def write_inventory(name: str, commit: str, category: str, kinds: set[str]) -> int:
    rows = rows_for(commit, kinds)
    prefix = "I"
    category_prefix = "C" if category == "controls" else "O"
    for index, row in enumerate(rows, 1):
        row["id"] = f"{prefix}{category_prefix}-{index:04d}"
    destination = DESTINATION / f"{name}_{category}.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_html_actions(name: str, commit: str) -> int:
    rows: list[dict[str, str]] = []
    full_commit = git("rev-parse", commit).strip()
    for path in source_paths(commit):
        source = git("show", f"{commit}:{path}")
        tree = ast.parse(source, filename=path)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for line_number, line in enumerate(source.splitlines(), 1):
            if not re.search(r"<a\s+", line):
                continue
            containing = [
                node
                for node in functions
                if node.lineno <= line_number <= (node.end_lineno or node.lineno)
            ]
            function = (
                min(
                    containing, key=lambda item: (item.end_lineno or item.lineno) - item.lineno
                ).name
                if containing
                else "<module>"
            )
            rows.append(
                {
                    "id": f"IH-{len(rows) + 1:04d}",
                    "source": f"{path}:{line_number}",
                    "reachability": reachability(path, function),
                    "function": function,
                    "source_line": line.strip(),
                    "source_url": f"https://github.com/leukocy/llm-test/blob/{full_commit}/{path}#L{line_number}",
                }
            )
    with (DESTINATION / f"{name}_html_actions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=HTML_ACTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def assigned_literal(tree: ast.AST, name: str):
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if not assignments:
        raise ValueError(f"Missing original assignment: {name}")
    assignment = min(assignments, key=lambda node: node.lineno)
    if assignment.value is None:
        raise ValueError(f"Missing original value: {name}")
    return ast.literal_eval(assignment.value)


def enum_options(tree: ast.AST, name: str, *, names: bool = False) -> list[str | int]:
    enum = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name
    )
    return [
        node.targets[0].id if names else ast.literal_eval(node.value)
        for node in enum.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.isupper()
    ]


def write_dynamic_choices(name: str, commit: str) -> None:
    settings = ast.parse(git("show", f"{commit}:config/settings.py"))
    sidebar = ast.parse(git("show", f"{commit}:ui/sidebar.py"))
    test_panels = ast.parse(git("show", f"{commit}:ui/test_panels.py"))
    advanced = ast.parse(git("show", f"{commit}:ui/advanced_panels.py"))
    robustness = ast.parse(git("show", f"{commit}:core/robustness_tester.py"))
    logger = ast.parse(git("show", f"{commit}:utils/logger.py"))
    preset_tree = ast.parse(git("show", f"{commit}:utils/test_config_manager.py"))
    preset_function = next(
        node
        for node in preset_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_builtin_presets"
    )
    preset_return = next(node for node in preset_function.body if isinstance(node, ast.Return))
    preset_calls = cast(ast.List, preset_return.value).elts
    builtin_presets = [
        {
            **{
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in cast(ast.Call, call).keywords
            },
            "source": f"utils/test_config_manager.py:{call.lineno}",
        }
        for call in preset_calls
    ]
    test_types = assigned_literal(sidebar, "_test_types")
    optional_types = []
    for node in ast.walk(sidebar):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_test_types"
            and node.func.attr == "append"
        ):
            optional_types.append(ast.literal_eval(node.args[0]))
    payload = {
        "baseline_commit": commit,
        "builtin_providers": assigned_literal(settings, "PROVIDER_OPTIONS"),
        "builtin_models": assigned_literal(settings, "MODEL_OPTIONS"),
        "builtin_presets": builtin_presets,
        "hf_model_mapping": assigned_literal(settings, "HF_MODEL_MAPPING"),
        "tokenizer_download_mapping": assigned_literal(settings, "TOKENIZER_HF_MAPPING"),
        "base_test_types": test_types,
        "conditional_test_types": optional_types,
        "segment_strategies": assigned_literal(test_panels, "segment_presets"),
        "quality_dataset_groups": {
            group: assigned_literal(advanced, group)
            for group in (
                "_BASIC_DATASETS",
                "_REASONING_DATASETS",
                "_CODE_EXTRA_DATASETS",
                "_SPECIAL_DATASETS",
            )
        },
        "perturbation_types": enum_options(robustness, "PerturbationType"),
        "log_levels": enum_options(logger, "LogLevel", names=True),
    }
    (DESTINATION / f"{name}_dynamic_choices.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    for name, commit in BASELINES.items():
        for category, kinds in (("controls", CONTROLS), ("outputs", OUTPUTS)):
            print(name, category, write_inventory(name, commit, category, kinds))
        print(name, "html_actions", write_html_actions(name, commit))
        write_dynamic_choices(name, commit)


if __name__ == "__main__":
    main()

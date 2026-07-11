# Professional UI Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将平台升级为专业浅色技术仪表盘，从所有运行时用户可见内容中移除 Emoji，并用稳定测试类型标识、结构化状态和 Material Symbols 替代原有耦合设计。

**Architecture:** 使用 `config/test_types.py` 集中管理稳定测试类型 ID、显示名称和兼容别名；使用 `ui/design_system.py` 提供统一设计令牌、全局 CSS、状态徽章和 Material 图标；使用字符串兼容的结构化洞察对象承载严重程度。主界面保留模型与全局设置在侧边栏，将各测试参数与主操作移入主工作区卡片，报告与导出层共享无 Emoji 的状态语义。

**Tech Stack:** Python 3.10+、Streamlit 1.59、pytest、Ruff、Streamlit AppTest、Playwright CLI。

**Baseline:** 2026-07-11 隔离分支基线为 657 passed、2 failed、1 skipped。两个既有失败均在 `tests/test_result_comparator.py`，由 `sorted(..., reverse=None)` 引发，与本计划无关；最终验证必须单独报告。

---

### Task 1: Stable Test-Type IDs and Legacy Compatibility

**Files:**
- Create: `config/test_types.py`
- Modify: `config/session_state.py`
- Modify: `core/result_persistence.py`
- Modify: `app.py`
- Modify: `ui/sidebar.py`
- Modify: `ui/test_panels.py`
- Modify: `ui/test_control_panel.py`
- Modify: `ui/test_runner.py`
- Modify: `ui/page_layout.py`
- Modify: `tests/test_ui_navigation_state.py`
- Modify: `tests/test_result_persistence.py`
- Modify: `tests/e2e_app_smoke.py`
- Test: `tests/test_test_types.py`

**Step 1: Write the failing test**

Create tests that define the desired public API:

```python
from config.test_types import TEST_TYPE_IDS, test_type_label, normalize_test_type


def test_normalizes_ids_labels_and_legacy_prefixed_labels():
    assert normalize_test_type("prefill") == "prefill"
    assert normalize_test_type("Prefill Stress Test") == "prefill"
    assert normalize_test_type("legacy-prefix Prefill Stress Test") == "prefill"


def test_labels_are_pure_display_text():
    assert TEST_TYPE_IDS[0] == "concurrency"
    assert test_type_label("concurrency") == "Concurrency Test"
```

Update navigation and persistence tests to expect stable IDs in session state while preserving legacy labels on input.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_test_types.py tests/test_ui_navigation_state.py tests/test_result_persistence.py -v`

Expected: FAIL because `config.test_types` does not exist and current session state stores display labels.

**Step 3: Write minimal implementation**

Implement a frozen `TestTypeSpec` registry with the IDs `concurrency`, `prefill`, `segmented`, `long_context`, `matrix`, `custom`, `all`, `stability`, `batch`, `quality`, `comparison`, and `advanced`. Provide:

```python
def normalize_test_type(value: object, allowed_options: Sequence[str] | None = None) -> str: ...
def test_type_label(value: object) -> str: ...
def test_type_icon(value: object) -> str | None: ...
```

Legacy values are matched by canonical alias or by a display label suffix, so old Emoji-prefixed strings remain readable without retaining Emoji literals. Streamlit selectboxes use IDs as options and `format_func=test_type_label`. Application routing, pending tests, resume logic and report dispatch compare IDs only. Persistence normalizes restored metadata to IDs.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_test_types.py tests/test_ui_navigation_state.py tests/test_result_persistence.py tests/e2e_app_smoke.py -v`

Expected: PASS.

**Step 5: Commit**

Run verification: `git diff --check && python -m pytest tests/test_test_types.py tests/test_ui_navigation_state.py tests/test_result_persistence.py tests/e2e_app_smoke.py -v`

Commit: `Refactor: decouple test type IDs from UI labels`

### Task 2: Design System and Professional Application Shell

**Files:**
- Create: `ui/design_system.py`
- Test: `tests/test_design_system.py`
- Modify: `app.py`
- Modify: `ui/sidebar.py`
- Modify: `ui/page_layout.py`
- Modify: `ui/test_control_panel.py`
- Modify: `ui/test_panels.py`
- Modify: `ui/onboarding.py`

**Step 1: Write the failing test**

```python
from ui.design_system import GLOBAL_CSS, material_icon, status_badge_html


def test_material_icons_use_streamlit_native_syntax():
    assert material_icon("play_arrow") == ":material/play_arrow:"


def test_status_badge_has_text_and_semantic_tone():
    html = status_badge_html("Running")
    assert "Running" in html
    assert "status-running" in html


def test_global_css_defines_dashboard_tokens():
    assert "--surface-primary" in GLOBAL_CSS
    assert "--accent-primary" in GLOBAL_CSS
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_design_system.py -v`

Expected: FAIL because `ui.design_system` does not exist.

**Step 3: Write minimal implementation**

Create global design tokens and CSS for the application background, sidebar, main block container, bordered containers, headings, metrics, inputs, tabs, expanders, alerts, primary/secondary/tertiary buttons and responsive widths. Add helpers for native Material icon syntax, status tone lookup and accessible text badges.

Call `apply_design_system()` once from `app.py`. Remove page-specific dark CSS from `ui/page_layout.py`. Render a compact application header and current model/test summary. Convert test parameter expanders in `ui/test_panels.py` from sidebar containers to main-workspace bordered sections. Render the control panel as a compact status/action card. Use native Material icons on primary run, pause, resume, stop, download, refresh, add and delete actions.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_design_system.py tests/e2e_app_smoke.py -v`

Expected: PASS.

**Step 5: Commit**

Run verification: `git diff --check && ruff check app.py ui/design_system.py ui/sidebar.py ui/page_layout.py ui/test_control_panel.py ui/test_panels.py ui/onboarding.py`

Commit: `Feat: add professional dashboard design system`

### Task 3: Remove Emoji from Primary Interactive Surfaces

**Files:**
- Test: `tests/test_user_visible_text.py`
- Modify: `app.py`
- Modify: `config/auth.py`
- Modify: `ui/sidebar.py`
- Modify: `ui/test_panels.py`
- Modify: `ui/test_control_panel.py`
- Modify: `ui/page_layout.py`
- Modify: `ui/onboarding.py`
- Modify: `ui/test_runner.py`

**Step 1: Write the failing test**

Create a Unicode policy helper covering the relevant Emoji blocks and assert that the primary surface sources contain no matches:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_user_visible_text.py::test_primary_surfaces_have_no_emoji -v`

Expected: FAIL and report the current offending files and characters.

**Step 3: Write minimal implementation**

Remove Emoji from every heading, label, help text, Toast, status message and onboarding string in these files. Replace interactive action symbols with native Material `icon=` arguments. Replace icon-only controls with explicit accessible labels such as `Refresh models` and `Delete preset`. Preserve widget keys and existing behavior.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_user_visible_text.py::test_primary_surfaces_have_no_emoji tests/e2e_app_smoke.py -v`

Expected: PASS.

**Step 5: Commit**

Run verification: `git diff --check && ruff check app.py config/auth.py ui/sidebar.py ui/test_panels.py ui/test_control_panel.py ui/page_layout.py ui/onboarding.py ui/test_runner.py tests/test_user_visible_text.py`

Commit: `Refactor: modernize primary UI content`

### Task 4: Structured Insights and Emoji-Free Reports

**Files:**
- Create: `ui/status.py`
- Test: `tests/test_ui_status.py`
- Modify: `tests/test_insights.py`
- Modify: `ui/insights.py`
- Modify: `ui/markdown_summary.py`
- Modify: `ui/reports.py`
- Modify: `ui/export.py`
- Modify: `ui/quality_reports.py`
- Modify: `ui/evaluation_dashboard.py`
- Modify: `ui/thinking_components.py`
- Modify: `ui/static_chart_generator.py`
- Modify: `ui/styled_tables.py`

**Step 1: Write the failing test**

Define a string-compatible structured insight API:

```python
from ui.status import InsightSeverity, PerformanceInsight


def test_performance_insight_preserves_markdown_and_severity():
    item = PerformanceInsight(InsightSeverity.WARNING, "Capacity", "Near the limit")
    assert str(item) == "**Capacity**: Near the limit"
    assert item.severity is InsightSeverity.WARNING
```

Add grade tests proving that severity, not glyph inspection, determines the grade. Extend `tests/test_user_visible_text.py` with a reporting-source group.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ui_status.py tests/test_insights.py tests/test_user_visible_text.py::test_reporting_surfaces_have_no_emoji -v`

Expected: FAIL because the structured type is missing and reporting files contain Emoji.

**Step 3: Write minimal implementation**

Implement `InsightSeverity` and a `str` subclass `PerformanceInsight` so existing Markdown joins remain compatible. Construct all performance insights with explicit severity. Refactor grading and critical-data short-circuit logic to inspect severity. Remove Emoji from report headings, summaries, charts, quality diagnostics, download links and HTML templates. Use textual status labels and CSS badges in HTML exports.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ui_status.py tests/test_insights.py tests/test_performance_metric_sanitizing.py tests/test_user_visible_text.py::test_reporting_surfaces_have_no_emoji -v`

Expected: PASS.

**Step 5: Commit**

Run verification: `git diff --check && ruff check ui/status.py ui/insights.py ui/markdown_summary.py ui/reports.py ui/export.py ui/quality_reports.py ui/evaluation_dashboard.py ui/thinking_components.py ui/static_chart_generator.py ui/styled_tables.py`

Commit: `Refactor: structure report status semantics`

### Task 5: Remove Emoji from Remaining Runtime Content

**Files:**
- Modify: `tests/test_user_visible_text.py`
- Modify: `ui/advanced_panels.py`
- Modify: `ui/batch_test.py`
- Modify: `ui/comparison_page.py`
- Modify: `ui/dashboard_components.py`
- Modify: `ui/dataset_manager.py`
- Modify: `ui/history_browser.py`
- Modify: `ui/log_viewer.py`
- Modify: `core/benchmark_runner.py`
- Modify: `core/certification_report.py`
- Modify: `core/error_messages.py`
- Modify: `core/models/exec_log.py`
- Modify: `core/quality_evaluator.py`
- Modify: `core/result_comparator.py`
- Modify: `evaluators/base_evaluator.py`
- Modify: `utils/log_server.py`
- Modify: `utils/logger.py`
- Modify: `utils/test_config_manager.py`
- Modify: other runtime files under `app.py`, `config/`, `core/`, `evaluators/`, `ui/`, and `utils/` reported by the scanner

**Step 1: Write the failing test**

Extend the source policy to scan all runtime Python files under `app.py`, `config/`, `core/`, `evaluators/`, `ui/`, and `utils/`, excluding tests, scripts, vendored data and generated files.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_user_visible_text.py::test_runtime_sources_have_no_emoji -v`

Expected: FAIL with the remaining source list.

**Step 3: Write minimal implementation**

Remove remaining Emoji from runtime messages, logs, generated certification text, comparison reports and evaluator feedback. Replace status-icon dictionaries with text/status mappings. Preserve mathematical arrows and ordinary Unicode punctuation that are not Emoji.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_user_visible_text.py -v`

Expected: PASS.

**Step 5: Commit**

Run verification: `git diff --check && ruff check app.py config core evaluators ui utils tests/test_user_visible_text.py`

Commit: `Refactor: remove Emoji from runtime content`

### Task 6: Application and Responsive Visual Verification

**Files:**
- Modify: `tests/e2e_app_smoke.py`
- Modify: `ui/design_system.py`
- Modify: UI files identified during visual review
- Artifacts: `output/playwright/ui-dashboard-desktop.png`
- Artifacts: `output/playwright/ui-dashboard-narrow.png`

**Step 1: Write the failing test**

Add AppTest assertions that the default page renders stable-ID labels, contains no Emoji in rendered Markdown/button text, and exposes the principal start action. Run it before visual fixes.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_e2e_app_smoke_runner.py -v`

Expected: FAIL on any remaining rendered Emoji or missing action label.

**Step 3: Write minimal implementation**

Fix only the rendering issues demonstrated by the test. Start Streamlit on port 8501, then use Playwright CLI to capture a fresh snapshot before interactions. Verify desktop width around 1440 px and narrow width around 768 px. Inspect side navigation, status card, parameter card, primary action, onboarding and report views. Iterate on spacing, wrapping, contrast and responsive rules.

**Step 4: Run tests and visual checks**

Run:

```text
python -m pytest tests/test_e2e_app_smoke_runner.py tests/test_user_visible_text.py -v
python -m streamlit run app.py --server.headless true --server.port 8501
npx --yes --package @playwright/mcp playwright-cli --session llm-ui open http://127.0.0.1:8501 --headed
npx --yes --package @playwright/mcp playwright-cli --session llm-ui snapshot
```

Expected: AppTest passes, browser console has no application errors, screenshots show the professional light dashboard at both widths, and rendered content contains no Emoji.

**Step 5: Commit**

Do not commit screenshots. Run `git diff --check`, confirm `output/playwright/` is absent from the staged set, then commit:

`Fix: polish responsive dashboard layout`

### Task 7: Final Verification and Delivery

**Files:**
- Verify all changed files
- Update: `docs/plans/2026-07-11-professional-ui-redesign-implementation.md` only if actual implementation decisions differ materially

**Step 1: Run static and targeted checks**

Run:

```text
python scripts/lint_changed.py --base master
python -m pytest tests/test_test_types.py tests/test_design_system.py tests/test_ui_status.py tests/test_user_visible_text.py tests/test_ui_navigation_state.py tests/test_result_persistence.py tests/test_e2e_app_smoke_runner.py -v
```

Expected: PASS with no lint errors in branch-changed Python files.

The repository-wide command had 237 pre-existing Ruff violations on `master`
before this work. The feature branch reduces that count to 221, but clearing
unrelated legacy lint debt is outside this UI change. `scripts/lint_changed.py`
is the reproducible commit gate for this branch and must remain green.

**Step 2: Run full regression suite**

Run: `python -m pytest tests/ -v`

Expected: all collected tests pass. The two original `result_comparator`
failures were fixed by treating neutral metric direction as `None` without
passing it to `sorted(reverse=...)`.

**Step 3: Audit requirements and staged content**

Run:

```text
git diff --check
git status --short
git diff --stat master...HEAD
git log --oneline master..HEAD
```

Confirm stable IDs, global design system, primary layout change, structured severity, zero runtime Emoji, Material icons, responsive browser evidence, and isolated commits. Confirm no secrets, logs, screenshots, raw data or original-worktree changes are included.

**Step 4: Final commit if needed**

Only if verification created a legitimate source change, stage that exact file set, rerun its checks, inspect `git diff --cached`, and commit with a `Fix:` or `Chore:` subject describing the verified change.

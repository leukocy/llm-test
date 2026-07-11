import re

files = [
    "ui/charts.py",
    "ui/test_control_panel.py",
    "ui/onboarding.py",
    "ui/static_chart_generator.py",
    "ui/advanced_panels.py",
    "ui/batch_test.py",
    "ui/quality_reports.py",
    "ui/export.py",
    "ui/history_browser.py",
    "ui/evaluation_dashboard.py",
    "ui/log_viewer.py",
    "ui/markdown_summary.py",
    "ui/thinking_components.py",
    "ui/test_runner.py",
    "ui/dashboard_components.py",
    "ui/dataset_manager.py",
    "ui/comparison_page.py",
    "ui/realtime_dashboard.py",
    "core/benchmark_runner.py",
    "core/error_messages.py",
    "core/metrics.py",
]

for f in files:
    try:
        content = open(f, encoding="utf-8").read()
        count = len(re.findall(r"[\u4e00-\u9fff]+", content))
        print(f"{count:4d} | {f}")
    except:
        print(f"  ?? | {f}")

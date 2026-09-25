"""Export tests for offline rendering and spreadsheet data integrity."""

import math
import re
from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
from openpyxl import load_workbook

from ui.export import export_interactive_html, export_to_excel, safe_csv_bytes
from ui.reporting.html_export import build_html_report
from ui.reporting.statistics import build_scientific_summary
from ui.static_chart_generator import StaticChartGenerator
from utils.spreadsheet import safe_download_filename


def test_html_report_is_offline_and_escapes_user_supplied_content():
    rows = pd.DataFrame({"concurrency": [1], "error": [None], "ttft": [0.4], "tps": [22]})
    analysis = build_scientific_summary(rows, "concurrency")
    figure = go.Figure(go.Scatter(x=[1, 2], y=[2, 3]))
    payload = "<img src=x onerror=alert(1)>"

    html = export_interactive_html(
        [figure],
        [pd.DataFrame({"Model": [payload]})],
        ["Finding: " + payload],
        title=payload,
        analysis=analysis,
    )

    assert re.search(r"<script[^>]+src=", html, flags=re.IGNORECASE) is None
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert payload not in html
    assert "Statistical breakdown" in html
    assert "95% Wilson" in html
    assert "report-chart-1" in html
    assert "<strong>Finding</strong>" in html


def test_spreadsheet_exports_keep_numbers_numeric_and_formula_text_inert():
    data = pd.DataFrame({"Prompt": ['=HYPERLINK("x")', " ordinary"], "TTFT": [1.25, 2.5]})
    workbook = export_to_excel({"invalid/name": data})

    assert workbook is not None
    sheet = load_workbook(BytesIO(workbook.getvalue())).active
    assert sheet.title == "invalid_name"
    assert sheet.freeze_panes == "A2"
    assert sheet["A2"].data_type == "s"
    assert sheet["A2"].value.startswith("'=")
    assert sheet["B2"].value == 1.25

    csv = safe_csv_bytes(data).decode("utf-8-sig")
    assert "'=HYPERLINK" in csv
    assert "1.25" in csv


def test_quality_html_accepts_named_tables_and_summary_cards():
    html = build_html_report(
        [],
        [("Outcomes", pd.DataFrame({"Accuracy": [0.8]}))],
        title="Quality report",
        overview_cards=[("Datasets", "1", "One workload")],
    )

    assert "Evaluation summary" in html
    assert "<h3>Outcomes</h3>" in html
    assert "80.0%" in html
    assert "One workload" in html


def test_static_chart_preserves_missing_measurement_as_gap():
    generator = StaticChartGenerator(dpi=70)
    figure = generator.draw_quad_chart(
        [1, 2],
        [{"y_data": [0.5, float("nan")], "title": "TTFT", "y_label": "s", "color": "#087f8c"}],
        "Measurement gaps",
        "Concurrency",
    )

    assert math.isnan(figure.axes[0].lines[0].get_ydata()[1])
    assert generator.save_figure_to_bytes(figure, "png").getvalue().startswith(b"\x89PNG")


def test_download_filename_removes_path_and_control_characters():
    assert safe_download_filename("../model/a\nb.html") == "_model_a_b.html"


def test_html_report_remains_downloadable_with_no_valid_chart_measurements():
    analysis = build_scientific_summary(
        pd.DataFrame({"concurrency": [1], "error": ["timeout"], "ttft": [0]}),
        "concurrency",
    )

    html = export_interactive_html([], [], title="Failed run", analysis=analysis)

    assert "Failed run" in html
    assert "Statistical breakdown" in html
    assert "No successful requests" in html
    assert "<h2>Charts</h2>" not in html

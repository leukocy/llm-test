"""Tests for the shared Streamlit design system."""

import inspect

import pytest


def test_material_icons_use_streamlit_native_syntax():
    from ui.design_system import material_icon

    assert material_icon("play_arrow") == ":material/play_arrow:"
    assert material_icon(" download ") == ":material/download:"


@pytest.mark.parametrize(
    ("status", "css_class"),
    [
        ("Idle", "status-idle"),
        ("Running", "status-running"),
        ("Paused", "status-paused"),
        ("Completed", "status-completed"),
        ("Failed", "status-failed"),
        ("Cancelled", "status-cancelled"),
    ],
)
def test_status_badge_has_text_and_semantic_tone(status, css_class):
    from ui.design_system import status_badge_html

    html = status_badge_html(status)

    assert status in html
    assert css_class in html
    assert 'role="status"' in html


def test_status_badge_escapes_unknown_text():
    from ui.design_system import status_badge_html

    html = status_badge_html('<script>alert("x")</script>')

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "status-idle" in html


def test_global_css_defines_dashboard_tokens_and_responsive_rule():
    from ui.design_system import GLOBAL_CSS

    assert "--surface-primary" in GLOBAL_CSS
    assert "--accent-primary" in GLOBAL_CSS
    assert "--status-danger" in GLOBAL_CSS
    assert "@media (max-width: 900px)" in GLOBAL_CSS
    assert "[data-testid=\"stSidebar\"]" in GLOBAL_CSS


def test_global_font_rule_does_not_override_streamlit_material_icons():
    from ui.design_system import GLOBAL_CSS

    assert 'html, body, [class*="st-"]' not in GLOBAL_CSS


def test_sidebar_model_refresh_action_uses_full_width_layout():
    from ui.sidebar import render_sidebar

    source = inspect.getsource(render_sidebar)
    model_controls = source[
        source.index("# Dynamically fetch model list"):source.index("model_id_custom")
    ]

    assert '"Refresh models"' in model_controls
    assert "use_container_width=True" in model_controls
    assert "st.columns(" not in model_controls


def test_sidebar_latency_probe_uses_full_width_layout():
    from ui.sidebar import render_sidebar

    source = inspect.getsource(render_sidebar)
    calibration_controls = source[
        source.index("# Probe button section"):source.index("# PD separation toggle")
    ]

    assert '"Measure latency"' in calibration_controls
    assert "use_container_width=True" in calibration_controls
    assert "st.columns(" not in calibration_controls

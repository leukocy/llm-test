"""Tests for the shared UI components in ui/components.py."""

from ui.components import (
    SEVERITY,
    SEVERITY_RANK,
    STATUS_MAP,
    insight_item,
    insight_text,
    normalize_status,
    status_badge_for,
    status_icon,
    status_label,
)
from ui.icons import SemanticColor

# ---------------------------------------------------------------------------
# normalize_status — the heart of Structure A unification
# ---------------------------------------------------------------------------


class TestNormalizeStatus:
    def test_lowercase_strings(self):
        assert normalize_status("completed") == "completed"
        assert normalize_status("running") == "running"

    def test_mixed_case_strings(self):
        assert normalize_status("Completed") == "completed"
        assert normalize_status("RUNNING") == "running"
        assert normalize_status("Failed") == "failed"

    def test_title_case_from_teststatus_constants(self):
        """TestStatus class uses Title-Case string values."""
        assert normalize_status("Idle") == "idle"
        assert normalize_status("Paused") == "paused"
        assert normalize_status("Cancelled") == "cancelled"

    def test_alternate_spellings(self):
        assert normalize_status("canceled") == "cancelled"
        assert normalize_status("CANCELED") == "cancelled"

    def test_pending_maps_to_waiting(self):
        assert normalize_status("pending") == "waiting"
        assert normalize_status("queued") == "waiting"
        assert normalize_status("ready") == "waiting"

    def test_boolean_true_is_enabled(self):
        assert normalize_status(True) == "enabled"

    def test_boolean_false_is_disabled(self):
        assert normalize_status(False) == "disabled"

    def test_object_with_value_attribute(self):
        class FakeEnum:
            value = "Completed"

        assert normalize_status(FakeEnum()) == "completed"

    def test_object_with_name_attribute(self):
        class FakeEnum:
            name = "FAILED"

        assert normalize_status(FakeEnum()) == "failed"

    def test_unknown_falls_back_to_idle(self):
        assert normalize_status("totally-unknown-state") == "idle"

    def test_none_falls_back_to_idle(self):
        assert normalize_status(None) == "idle"

    def test_integer_falls_back_to_idle(self):
        assert normalize_status(42) == "idle"


# ---------------------------------------------------------------------------
# STATUS_MAP integrity
# ---------------------------------------------------------------------------


class TestStatusMap:
    def test_all_values_are_3tuples(self):
        for key, value in STATUS_MAP.items():
            assert isinstance(value, tuple)
            assert len(value) == 3
            icon_name, level, label = value
            assert isinstance(icon_name, str)
            assert level in ("success", "info", "warning", "danger", "muted")
            assert isinstance(label, str)

    def test_completed_is_success(self):
        """The most-tested semantic: completed → success level."""
        assert STATUS_MAP["completed"][1] == "success"

    def test_failed_is_danger(self):
        assert STATUS_MAP["failed"][1] == "danger"

    def test_running_is_info(self):
        assert STATUS_MAP["running"][1] == "info"

    def test_paused_is_warning(self):
        assert STATUS_MAP["paused"][1] == "warning"

    def test_covers_all_canonical_states(self):
        canonical = {
            "idle",
            "waiting",
            "running",
            "paused",
            "completed",
            "failed",
            "cancelled",
        }
        assert canonical.issubset(STATUS_MAP.keys())


# ---------------------------------------------------------------------------
# Rendering functions
# ---------------------------------------------------------------------------


class TestStatusIcon:
    def test_returns_svg(self):
        rendered = status_icon("completed")
        assert rendered.startswith("<svg")
        assert rendered.rstrip().endswith("</svg>")

    def test_failed_uses_danger_color(self):
        rendered = status_icon("failed")
        assert SemanticColor.DANGER in rendered

    def test_size_parameter(self):
        rendered = status_icon("running", size=20)
        assert 'width="20"' in rendered


class TestStatusBadge:
    def test_completed_badge(self):
        rendered = status_badge_for("completed")
        assert "ui-badge-success" in rendered
        assert "Completed" in rendered
        assert "<svg" in rendered

    def test_failed_badge(self):
        rendered = status_badge_for("failed")
        assert "ui-badge-danger" in rendered

    def test_custom_text_override(self):
        rendered = status_badge_for("running", text="Custom")
        assert "Custom" in rendered

    def test_unknown_status_badge(self):
        rendered = status_badge_for("???")
        assert "ui-badge-muted" in rendered


class TestStatusLabel:
    def test_returns_human_label(self):
        assert status_label("completed") == "Completed"
        assert status_label("running") == "Running"

    def test_unknown(self):
        assert status_label("???") == "Idle"


# ---------------------------------------------------------------------------
# Insight severity
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_all_four_levels_defined(self):
        assert set(SEVERITY.keys()) == {"positive", "neutral", "warning", "critical"}

    def test_severity_rank_order(self):
        assert SEVERITY_RANK[0] == "critical"
        assert SEVERITY_RANK[-1] == "positive"

    def test_each_severity_has_icon_and_level(self):
        for sev, (icon_name, level) in SEVERITY.items():
            assert isinstance(icon_name, str) and icon_name
            assert level in ("success", "info", "warning", "danger")


class TestInsightItem:
    def test_contains_svg_icon(self):
        rendered = insight_item("warning", "Slow throughput")
        assert "<svg" in rendered
        assert "Slow throughput" in rendered

    def test_critical_uses_danger_color(self):
        rendered = insight_item("critical", "Overload")
        assert SemanticColor.DANGER in rendered

    def test_warning_uses_warning_color(self):
        rendered = insight_item("warning", "Degradation")
        assert SemanticColor.WARNING in rendered

    def test_positive_uses_success_color(self):
        rendered = insight_item("positive", "Strong scaling")
        assert SemanticColor.SUCCESS in rendered

    def test_unknown_severity_defaults_neutral(self):
        rendered = insight_item("bogus", "text")
        assert SemanticColor.INFO in rendered

    def test_escapes_text(self):
        rendered = insight_item("neutral", "<script>alert(1)</script>")
        assert "<script>" not in rendered
        assert "&lt;script&gt;" in rendered


class TestInsightText:
    def test_plain_text_with_tag(self):
        result = insight_text("critical", "System down")
        assert "[CRITICAL]" in result
        assert "System down" in result
        assert "<svg" not in result

    def test_no_html(self):
        result = insight_text("warning", "test")
        assert "<" not in result

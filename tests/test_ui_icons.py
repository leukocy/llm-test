"""Tests for the inline SVG icon primitives in ui/icons.py."""

import pytest

from ui.icons import BADGE_LEVELS, EMOJI_TO_ICON, ICONS, SemanticColor, icon, status_badge

# ---------------------------------------------------------------------------
# ICONS registry integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ICONS))
def test_every_icon_is_valid_svg(name):
    """Each icon's inner markup must render into a well-formed SVG."""
    rendered = icon(name)
    assert rendered.startswith("<svg")
    assert rendered.rstrip().endswith("</svg>")
    assert 'viewBox="0 0 24 24"' in rendered
    # stroke-based (fill none), not raster
    assert 'fill="none"' in rendered


@pytest.mark.parametrize("name", sorted(ICONS))
def test_every_icon_has_nonempty_inner(name):
    assert ICONS[name].strip(), f"icon {name!r} has empty path data"


def test_unknown_icon_raises():
    with pytest.raises(KeyError, match="Unknown icon name"):
        icon("does-not-exist")


# ---------------------------------------------------------------------------
# icon() rendering options
# ---------------------------------------------------------------------------


def test_icon_defaults_to_current_color():
    rendered = icon("rocket")
    assert 'stroke="currentColor"' in rendered
    assert "ui-icon" in rendered


def test_icon_color_override():
    rendered = icon("check", color="#ff0000")
    assert 'stroke="#ff0000"' in rendered
    assert "currentColor" not in rendered


def test_icon_size_applied():
    rendered = icon("zap", size=32)
    assert 'width="32"' in rendered
    assert 'height="32"' in rendered


def test_icon_stroke_width():
    rendered = icon("x", stroke_width=1.5)
    assert 'stroke-width="1.5"' in rendered


def test_icon_vertical_align():
    rendered = icon("check", vertical_align="top")
    assert "vertical-align:top" in rendered


def test_icon_class_name():
    rendered = icon("check", class_name="my-custom")
    assert "my-custom" in rendered


def test_icon_escapes_color():
    """A color containing HTML-special chars must be escaped (no injection)."""
    rendered = icon("check", color='red"><script>')
    assert "<script>" not in rendered
    assert "&quot;" in rendered or "&#x27;" in rendered


# ---------------------------------------------------------------------------
# status_badge()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", list(BADGE_LEVELS))
def test_badge_levels_produce_correct_class(level):
    rendered = status_badge(level, "ok")
    assert f"ui-badge-{level}" in rendered
    assert "ui-badge " in rendered
    assert ">ok</span>" in rendered


def test_badge_unknown_level_falls_back_to_muted():
    rendered = status_badge("nonexistent", "x")
    assert "ui-badge-muted" in rendered


def test_badge_with_icon():
    rendered = status_badge("success", "Passed", icon_name="check")
    assert "<svg" in rendered
    assert ">Passed</span>" in rendered


def test_badge_icon_only():
    rendered = status_badge("danger", icon_name="x")
    assert "<svg" in rendered
    assert "</span>" in rendered


def test_badge_escapes_text():
    rendered = status_badge("info", "<b>bold</b>")
    assert "<b>bold</b>" not in rendered
    assert "&lt;b&gt;" in rendered


# ---------------------------------------------------------------------------
# SemanticColor palette
# ---------------------------------------------------------------------------


def test_semantic_colors_are_hex():
    """Every public color constant must be a valid hex color string."""
    public = [
        v for k, v in vars(SemanticColor).items() if not k.startswith("_") and isinstance(v, str)
    ]
    assert len(public) >= 6
    for color in public:
        assert color.startswith("#") and len(color) == 7


# ---------------------------------------------------------------------------
# EMOJI_TO_ICON coverage — every icon referenced must exist in ICONS
# ---------------------------------------------------------------------------


def test_all_emoji_map_targets_exist():
    """Every value in EMOJI_TO_ICON must be a registered icon name."""
    missing = [v for v in EMOJI_TO_ICON.values() if v not in ICONS]
    assert not missing, f"emoji map points to missing icons: {missing}"


def test_emoji_map_covers_core_emoji():
    """The high-frequency emoji used across the UI must all be mapped."""
    core_emoji = [
        "📊",
        "✅",
        "🚀",
        "⚠️",
        "📈",
        "❌",
        "📥",
        "💡",
        "⚡",
        "🔄",
        "🔍",
        "🎯",
        "⏱️",
        "📝",
        "📄",
        "💾",
        "🗑️",
        "🔬",
        "📉",
        "⚙️",
        "🖥️",
        "🗄️",
        "🧪",
        "🛑",
        "⏸️",
        "▶️",
        "⏹️",
    ]
    missing = [e for e in core_emoji if e not in EMOJI_TO_ICON]
    assert not missing, f"core emoji not mapped to icons: {missing}"


def test_no_emoji_map_keys_collide_to_different_icons():
    """Same base emoji with/without variation selector should map consistently."""
    for emoji_bare, icon_name in EMOJI_TO_ICON.items():
        # The VS16 form (emoji + U+FE0F) should map to the same icon if present
        vs16 = emoji_bare + "\ufe0f"
        if vs16 in EMOJI_TO_ICON:
            assert (
                EMOJI_TO_ICON[vs16] == icon_name
            ), f"{emoji_bare!r} -> {icon_name} but {vs16!r} -> {EMOJI_TO_ICON[vs16]}"

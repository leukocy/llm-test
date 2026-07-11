"""
Shared UI components — replaces duplicated emoji-based UI structures.

This module consolidates the patterns previously repeated across the UI:

* **Status → icon mappings** were copy-pasted in 5 places
  (dashboard_components, history_browser x2, test_control_panel, batch_test)
  with inconsistent keys, values, and icon choices.
* **Section headers** used ``st.header("\U0001F4CA Title")`` with ad-hoc emoji.
* **Insight items** used ``"\u26A0\uFE0F **text**"`` with severity implied by emoji.

All components here render via :mod:`ui.icons` inline SVG (no emoji).
"""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from ui.icons import BADGE_LEVELS, SemanticColor, icon, status_badge

# ---------------------------------------------------------------------------
# Status mapping — single source of truth (Structure A)
# ---------------------------------------------------------------------------
# Semantic key → (icon_name, badge_level, human_label).
# This unifies the 5 historically-inconsistent status dicts. "completed" is
# always success-green, "running" is always info-blue, etc. — no more places
# where completed was \U0001F7E2 in one view and \u2705 in another.
STATUS_MAP: dict[str, tuple[str, str, str]] = {
    "idle": ("circle", "muted", "Idle"),
    "waiting": ("hourglass", "muted", "Waiting"),
    "running": ("activity", "info", "Running"),
    "paused": ("pause", "warning", "Paused"),
    "completed": ("check", "success", "Completed"),
    "failed": ("x", "danger", "Failed"),
    "cancelled": ("square", "muted", "Cancelled"),
    # enabled/disabled for batch_test items
    "enabled": ("check", "success", "Enabled"),
    "disabled": ("pause", "muted", "Disabled"),
}


# Aliases used across the codebase — normalized to canonical semantic keys.
_STATUS_ALIASES: dict[str, str] = {
    # TestStatus constants (ui.test_control_panel) use Title-Case values
    "idle": "idle",
    "running": "running",
    "paused": "paused",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "canceled": "cancelled",  # common alternate spelling
    # dashboard_components used "waiting"
    "waiting": "waiting",
    "pending": "waiting",
    "ready": "waiting",
    "queued": "waiting",
    # batch_test boolean states
    "enabled": "enabled",
    "disabled": "disabled",
    "active": "enabled",
    "inactive": "disabled",
}


def normalize_status(key: Any) -> str:
    """Map any status key (str, enum value, bool) to a canonical semantic key.

    Accepts:
      * Strings in any case ("Completed", "COMPLETED", "completed")
      * Enum members or objects with a ``.value``/``.name`` attribute
        (e.g. ``TestStatus.COMPLETED``)
      * Booleans (``True`` → "enabled", ``False`` → "disabled")

    Falls back to ``"idle"`` for unrecognized input.
    """
    if isinstance(key, bool):
        return "enabled" if key else "disabled"

    # Extract value from enum-like objects
    raw = key
    for attr in ("value", "name"):
        val = getattr(key, attr, None)
        if isinstance(val, str):
            raw = val
            break

    if not isinstance(raw, str):
        return "idle"

    normalized = raw.strip().lower()
    return _STATUS_ALIASES.get(normalized, "idle")


def status_icon(key: Any, *, size: int = 14) -> str:
    """Return an inline SVG icon for a status key (for use in metric values)."""
    sem_key = normalize_status(key)
    icon_name = STATUS_MAP.get(sem_key, STATUS_MAP["idle"])[0]
    # Color follows the badge level's semantic hue
    level = STATUS_MAP.get(sem_key, STATUS_MAP["idle"])[1]
    color = _level_color(level)
    return icon(icon_name, size=size, color=color)


def status_badge_for(key: Any, *, text: str | None = None, size: int = 14) -> str:
    """Return a full HTML badge for a status key.

    If ``text`` is None, uses the canonical label from STATUS_MAP.
    """
    sem_key = normalize_status(key)
    icon_name, level, label = STATUS_MAP.get(sem_key, STATUS_MAP["idle"])
    display = text if text is not None else label
    return status_badge(level, display, icon_name=icon_name, size=size)


def status_label(key: Any) -> str:
    """Return the canonical human-readable label for a status key."""
    sem_key = normalize_status(key)
    return STATUS_MAP.get(sem_key, STATUS_MAP["idle"])[2]


def _level_color(level: str) -> str:
    """Resolve a badge level name to its hex color."""
    return {
        "success": SemanticColor.SUCCESS,
        "info": SemanticColor.INFO,
        "warning": SemanticColor.WARNING,
        "danger": SemanticColor.DANGER,
        "muted": SemanticColor.MUTED,
    }.get(level, SemanticColor.MUTED)


# ---------------------------------------------------------------------------
# Insight severity (Structure D) — replaces emoji prefix convention
# ---------------------------------------------------------------------------
# Maps severity keys to (icon_name, badge_level). Used by insights.py to tag
# insight text, and by get_performance_grade() to count by severity instead of
# emoji substring matching.
SEVERITY: dict[str, tuple[str, str]] = {
    "positive": ("check", "success"),
    "neutral": ("activity", "info"),
    "warning": ("alert-triangle", "warning"),
    "critical": ("alert-octagon", "danger"),
}

# Ordered from most to least severe — used for grading priority.
SEVERITY_RANK = ["critical", "warning", "neutral", "positive"]


def insight_item(severity: str, text: str) -> str:
    """Format an insight line with a severity icon prefix (markdown-safe).

    The returned string is plain markdown: the SVG icon (rendered via
    ``unsafe_allow_html`` by the caller's container) sits before the text.
    Callers that use ``st.markdown(insight)`` should ensure the surrounding
    context renders HTML (Streamlit markdown allows inline HTML by default
    only with unsafe_allow_html; for plain-text contexts use
    :func:`insight_text` instead).
    """
    if severity not in SEVERITY:
        severity = "neutral"
    icon_name, level = SEVERITY[severity]
    color = _level_color(level)
    # Escape the text to prevent markdown/HTML injection from data values
    return f"{icon(icon_name, size=16, color=color)} {escape(text)}"


def insight_text(severity: str, text: str) -> str:
    """Return insight text with a textual severity tag (no HTML/SVG).

    Use this for plain-text contexts (CSV export, markdown summary, etc.)
    where inline SVG would be inappropriate.
    """
    tag = severity.upper()
    return f"[{tag}] {text}"


# ---------------------------------------------------------------------------
# Section headers (replaces st.header("\U0001F4CA Title"))
# ---------------------------------------------------------------------------


def section_header(
    title: str,
    icon_name: str | None = None,
    *,
    level: str = "header",
    color: str | None = None,
) -> None:
    """Render a section header with an optional leading icon.

    Parameters
    ----------
    title:
        The header text (plain text, no emoji needed).
    icon_name:
        Optional icon to precede the title. Renders inline SVG.
    level:
        One of ``"header"``, ``"subheader"``, or ``"markdown"``.
    color:
        Optional icon color override. Defaults to the accent color.
    """
    safe_title = escape(title)
    if icon_name:
        icon_html = icon(icon_name, size=20, color=color or SemanticColor.ACCENT)
        html = f'<div class="ui-section-title">{icon_html} {safe_title}</div>'
        if level == "header":
            st.markdown(f"#### {html}", unsafe_allow_html=True)
        elif level == "subheader":
            st.markdown(f"##### {html}", unsafe_allow_html=True)
        else:
            st.markdown(html, unsafe_allow_html=True)
    else:
        # No icon — use native Streamlit headers for accessibility
        if level == "header":
            st.subheader(title)
        elif level == "subheader":
            st.caption(f"**{title}**")
        else:
            st.markdown(f"**{title}**")


# ---------------------------------------------------------------------------
# Generic labeled metric (replaces st.metric("\u2705 Completed", ...))
# ---------------------------------------------------------------------------


def metric(label: str, value: Any, *args: Any, **kwargs: Any) -> None:
    """Wrapper around ``st.metric`` that strips emoji from labels.

    The label is passed through unchanged except emoji are removed.
    All other arguments (value, delta, help, etc.) are forwarded.
    """
    clean_label = _strip_emoji(label)
    st.metric(clean_label, value, *args, **kwargs)


def _strip_emoji(text: str) -> str:
    """Remove emoji characters from a string (best-effort)."""
    from ui.icons import EMOJI_TO_ICON

    result = text
    for emoji_char in EMOJI_TO_ICON:
        result = result.replace(emoji_char, "")
    # Clean up double spaces left behind
    while "  " in result:
        result = result.replace("  ", " ")
    return result.strip()


__all__ = [
    "SEVERITY",
    "SEVERITY_RANK",
    "STATUS_MAP",
    "BADGE_LEVELS",
    "insight_item",
    "insight_text",
    "metric",
    "normalize_status",
    "section_header",
    "status_badge_for",
    "status_icon",
    "status_label",
]

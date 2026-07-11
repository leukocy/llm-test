"""
Inline SVG icon primitives (Lucide-style), zero external dependencies.

Replaces emoji throughout the UI with crisp, color-inheritable line icons.
All SVG path data is hardcoded here — no fonts, no CDN, no network — so the
platform stays fully usable in Docker/offline deployments.

Usage::

    from ui.icons import icon

    st.markdown(f"{icon('rocket')} Start Test")
    st.markdown(icon('check', color=SemanticColor.SUCCESS), unsafe_allow_html=True)

The rendered SVG uses ``stroke="currentColor"`` by default, so the icon
inherits the surrounding text color. Pass an explicit ``color`` to override.
"""

from __future__ import annotations

from html import escape


# ---------------------------------------------------------------------------
# Semantic color palette
# ---------------------------------------------------------------------------
# Single source of truth for status/grade colors. Merges the Bootstrap-grade
# colors historically used by insights.py with the Tailwind-slate accents used
# by thinking_components.py, so the whole UI speaks one color language.
class SemanticColor:
    """Semantic status colors used by badges, icons, and grades."""

    SUCCESS = "#28a745"  # green  — strong positive / A grade
    INFO = "#17a2b8"  # cyan   — good / B+ grade / informational
    WARNING = "#ffc107"  # amber — warning / B- grade / needs attention
    DANGER = "#dc3545"  # red    — critical / C-D grade / failure
    MUTED = "#6c757d"  # gray   — N/A / neutral / disabled
    ACCENT = "#4a9eff"  # blue   — brand accent / links
    # Per-metric accents (from thinking_components.py metric cards)
    SPEED = "#eab308"  # yellow — TTFT / latency
    COST = "#f97316"  # orange — cost
    BRAIN = "#60a5fa"  # blue   — reasoning
    RATIO = "#a855f7"  # purple — ratios


# Badge level → CSS variable name (resolved in page_layout.py CSS)
BADGE_LEVELS = ("success", "info", "warning", "danger", "muted")


# ---------------------------------------------------------------------------
# Icon registry — Lucide path data (24x24 viewBox, stroke-based)
# ---------------------------------------------------------------------------
# Each value is the inner markup of an <svg> (one or more <path>/<circle>/etc).
# The wrapping <svg> element is produced by icon().
ICONS: dict[str, str] = {
    # --- status / severity ---
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "alert-triangle": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
        '<path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "alert-octagon": (
        '<polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/>'
        '<line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/>'
    ),
    "siren": (
        '<path d="M7 18v-6a5 5 0 1 1 10 0v6"/><path d="M5 21a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-1a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2z"/>'
        '<path d="M21 12h1"/><path d="M18.5 4.5 18 5"/><path d="M2 12h1"/><path d="M12 2v1"/>'
        '<path d="M4.93 4.93l.7.7"/><path d="M13.5 6.5h.01"/>'
    ),
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    # --- performance / test ---
    "activity": '<path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"/>',
    "trending-up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "trending-down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "rocket": (
        '<path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/>'
        '<path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/>'
        '<path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0"/><path d="M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5"/>'
    ),
    "gauge": ('<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>'),
    "timer": '<line x1="10" x2="14" y1="2" y2="2"/><line x1="12" x2="15" y1="14" y2="11"/><circle cx="12" cy="14" r="8"/>',
    "trophy": (
        '<path d="M6 9H4.5a2.5 2.5 0 0 1 0-5H6"/><path d="M18 9h1.5a2.5 2.5 0 0 0 0-5H18"/>'
        '<path d="M4 22h16"/><path d="M10 14.66V17c0 .55-.47.98-.97 1.21C7.85 18.75 7 20.24 7 22"/>'
        '<path d="M14 14.66V17c0 .55.47.98.97 1.21C16.15 18.75 17 20.24 17 22"/>'
        '<path d="M18 2H6v7a6 6 0 0 0 12 0V2Z"/>'
    ),
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
    "ruler": '<path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/>',
    "microscope": '<path d="M6 18h8"/><path d="M3 22h18"/><path d="M14 22a7 7 0 1 0 0-14h-1"/><path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/><path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>',
    "flask-conical": '<path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "chart-bar": '<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
    # --- actions ---
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    "save": '<path d="M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z"/><path d="M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7"/><path d="M7 3v4a1 1 0 0 0 1 1h7"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',
    "refresh-cw": '<path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5v0a5.5 5.5 0 0 1-5.5 5.5H11"/><path d="M15 10l5 5-5 5"/>',
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "minus": '<path d="M5 12h14"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "search-check": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/><path d="m8 11 2 2 4-4"/>',
    "eye": '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/>',
    "pencil-line": '<path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    # --- content / files ---
    "file-text": (
        '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
        '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 9H8"/><path d="M16 13H8"/><path d="M16 17H8"/>'
    ),
    "clipboard-list": (
        '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/>'
        '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/>'
        '<path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>'
    ),
    "scroll": (
        '<path d="M19 17V5a2 2 0 0 0-2-2H4"/>'
        '<path d="M8 21h12a2 2 0 0 0 2-2v-1a1 1 0 0 0-1-1H11a1 1 0 0 0-1 1v1a2 2 0 1 1-4 0V5a2 2 0 1 0-4 0v2a1 1 0 0 0 1 1h3"/>'
    ),
    "lightbulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>',
    "package": '<path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "folder-open": '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
    "book-open": '<path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/>',
    "book": '<path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20"/>',
    "files": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M9 13h6"/><path d="M9 17h3"/>',
    # --- system / config ---
    "settings": (
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>'
        '<circle cx="12" cy="12" r="3"/>'
    ),
    "wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "sliders-horizontal": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="16" x2="16" y1="18" y2="22"/>',
    "plug": '<path d="M12 22v-5"/><path d="M9 8V2"/><path d="M15 8V2"/><path d="M18 8v5a4 4 0 0 1-4 4h-4a4 4 0 0 1-4-4V8Z"/>',
    "monitor": '<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/>',
    "globe": '<circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "lock-keyhole": '<circle cx="7.5" cy="15.5" r=".5" fill="currentColor"/><circle cx="18.5" cy="15.5" r=".5" fill="currentColor"/><circle cx="12" cy="10.5" r=".5" fill="currentColor"/><path d="M12 22V8a5 5 0 0 0-10 0"/><path d="M5 12H2a5 5 0 0 1 5-5"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>',
    "tag": '<path d="M12.586 2.586A2 2 0 0 0 11.172 2H4a2 2 0 0 0-2 2v7.172a2 2 0 0 0 .586 1.414l8.704 8.704a2.426 2.426 0 0 0 3.42 0l6.58-6.58a2.426 2.426 0 0 0 0-3.42z"/><circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/>',
    "square-ruler": '<path d="M14.5 3a9 9 0 0 0-9 9L3 15l6 6 3-2.5a9 9 0 0 0 9-9"/><path d="m15 5 4 4"/>',
    "bolt": '<path d="M9 18h6"/><path d="m10 22 4-7h-3l1-7"/>',
    "radio": '<path d="M4.9 19.1C1 15.2 1 8.8 4.9 4.9"/><path d="M7.8 16.2c-2.3-2.3-2.3-6.1 0-8.5"/><circle cx="12" cy="12" r="2"/><path d="M16.2 7.8c2.3 2.3 2.3 6.1 0 8.5"/><path d="M19.1 4.9C23 8.8 23 15.2 19.1 19.1"/>',
    "archive": '<rect width="20" height="5" x="2" y="3" rx="1"/><path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8"/><path d="M10 12h4"/>',
    # --- playback / status dots ---
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "pause": '<rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/>',
    "square": '<rect width="18" height="18" x="3" y="3" rx="2"/>',
    "skip-forward": '<polygon points="5 4 15 12 5 20 5 4"/><line x1="19" x2="19" y1="5" y2="19"/>',
    "hourglass": '<path d="M5 22h14"/><path d="M5 2h14"/><path d="M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22"/><path d="M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2"/>',
    "circle": '<circle cx="12" cy="12" r="10"/>',
    "circle-dot": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="2" fill="currentColor"/>',
    "square-dashed": '<path d="M5 3a2 2 0 0 0-2 2" stroke-dasharray="4 4"/><path d="M19 3a2 2 0 0 1 2 2" stroke-dasharray="4 4"/><path d="M21 19a2 2 0 0 1-2 2" stroke-dasharray="4 4"/><path d="M5 21a2 2 0 0 1-2-2" stroke-dasharray="4 4"/>',
    "gamepad": '<line x1="6" x2="10" y1="11" y2="11"/><line x1="8" x2="8" y1="9" y2="13"/><line x1="15" x2="15.01" y1="12" y2="12"/><line x1="18" x2="18.01" y1="10" y2="10"/><path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z"/>',
    # --- navigation ---
    "arrow-left": '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "arrow-down": '<path d="M12 5v14"/><path d="m19 12-7 7-7-7"/>',
    "arrow-up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    # --- misc / warehouse ---
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "hand": '<path d="M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2"/><path d="M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2"/><path d="M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8"/><path d="M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>',
    "party-popper": '<path d="M5.8 11.3 2 22l10.7-3.79"/><path d="M4 3h.01"/><path d="M22 8h.01"/><path d="M15 2h.01"/><path d="M22 20h.01"/><path d="m22 2-2.24.75a2.9 2.9 0 0 0-1.96 3.12c.1.86-.57 1.63-1.45 1.63h-.38c-.86 0-1.6.6-1.76 1.44L14 10"/><path d="m22 13-1.3.39c-.85.25-1.4 1.11-1.21 1.97l.03.14a1.5 1.5 0 0 1-1.15 1.79l-.46.12"/><path d="M11 5h.01"/><path d="M8 22h.01"/><path d="M16 17h.01"/>',
    "bot": '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    "shuffle": '<path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22"/><path d="m18 2 4 4-4 4"/><path d="M2 6h1.9c1.5 0 2.9.9 3.6 2.2"/><path d="M22 18h-5.9c-1.3 0-2.6-.7-3.3-1.8l-.5-.8"/><path d="m18 14 4 4-4 4"/>',
    "camera": '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>',
    "inbox": '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "calculator": '<rect width="16" height="20" x="4" y="2" rx="2"/><line x1="8" x2="16" y1="6" y2="6"/><line x1="16" x2="16" y1="14" y2="18"/><path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/><path d="M12 14h.01"/><path d="M8 14h.01"/><path d="M12 18h.01"/><path d="M8 18h.01"/>',
    "dice": '<rect width="12" height="12" x="2" y="10" rx="2" ry="2"/><path d="m17.92 14 3.5-3.5a2.24 2.24 0 0 0 0-3l-5-4.92a2.24 2.24 0 0 0-3 0L10 6"/><path d="M6 18h.01"/><path d="M10 14h.01"/><path d="M15 6h.01"/><path d="M18 9h.01"/>',
    "thermometer": '<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>',
    "coins": '<circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18"/><path d="M7 6h1v4"/><path d="m16.71 13.88.7.71-2.82 2.82"/>',
    "building": '<rect width="16" height="20" x="4" y="2" rx="2" ry="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/>',
    "laptop": '<rect width="18" height="12" x="3" y="4" rx="2" ry="2"/><line x1="2" x2="22" y1="20" y2="20"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "rabbit": (
        '<path d="M13 16a3 3 0 0 1 2.24 5"/><path d="M18 12h.01"/><path d="M18 21h-8a4 4 0 0 1-4-4 7 7 0 0 1 7-7h.2L9.6 6.4a1 1 0 1 1 2.8-2.8L15.8 7h.2c3.3 0 6 2.7 6 6v1a2 2 0 0 1-2 2h-1a3 3 0 0 0-3 3"/>'
        '<path d="M20 8.54V4h2a1 1 0 0 0 0-5"/>'
    ),
    "hand-pointing-up": '<path d="M7 11.5V14m0-2.5v-2a1.5 1.5 0 1 1 3 0V12m0-1.5v-2a1.5 1.5 0 1 1 3 0V12m0-.5v-2.5a1.5 1.5 0 0 1 3 0V14"/><path d="M7 14a2 2 0 0 0 2 2h5.78a2 2 0 0 0 1.95-1.52l1.05-4.05a.8.8 0 0 0-.78-1.01c-.37 0-.73.12-1.02.34L16 9.5"/><path d="M5 20h16"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .962 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.962 0z"/><path d="M20 3v4"/><path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
    "magnifying-glass": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "empty": '<rect width="18" height="18" x="3" y="3" rx="2" stroke-dasharray="3 3"/><path d="M9 9h6v6H9z"/>',
    "run": '<circle cx="13" cy="4" r="1.5" fill="currentColor"/><path d="m5 21 4-7-2-3 5-2 3 4 3 1"/><path d="M9 13l-2-1"/>',
}


# ---------------------------------------------------------------------------
# Emoji → icon name reverse-lookup
# ---------------------------------------------------------------------------
# Maps every emoji historically used in the UI to its replacement icon.
# Keys include both bare emoji and emoji+variation-selector forms.
EMOJI_TO_ICON: dict[str, str] = {
    # status / severity
    "✅": "check",
    "❌": "x",
    "⚠️": "alert-triangle",
    "⚠": "alert-triangle",
    "🛑": "alert-octagon",
    "🚨": "siren",
    "ℹ️": "info",
    "ℹ": "info",
    "🐢": "rabbit",
    "🎢": "trending-down",
    # performance / test
    "📊": "chart-bar",
    "📈": "trending-up",
    "📉": "trending-down",
    "⚡": "zap",
    "🚀": "rocket",
    "🎯": "target",
    "⏱️": "timer",
    "⏱": "timer",
    "🏆": "trophy",
    "🔥": "flame",
    "📏": "ruler",
    "🔬": "microscope",
    "🧪": "flask-conical",
    "⚖️": "scale",
    "⚖": "scale",
    "📐": "square-ruler",
    # actions
    "📥": "download",
    "📤": "upload",
    "💾": "save",
    "🗑️": "trash",
    "🗑": "trash",
    "🔄": "refresh-cw",
    "➕": "plus",
    "➖": "minus",
    "🔍": "search",
    "🔎": "search-check",
    "👀": "eye",
    "📝": "pencil-line",
    # content / files
    "📄": "file-text",
    "📋": "clipboard-list",
    "📜": "scroll",
    "💡": "lightbulb",
    "🧠": "brain",
    "📦": "package",
    "📁": "folder",
    "📂": "folder-open",
    "📚": "book-open",
    "📑": "files",
    "📗": "book",
    "📷": "camera",
    # system / config
    "⚙️": "settings",
    "⚙": "settings",
    "🔧": "wrench",
    "🎛️": "sliders-horizontal",
    "🧮": "calculator",
    "🌡️": "thermometer",
    "🎲": "dice",
    "🔌": "plug",
    "🖥️": "monitor",
    "🖥": "monitor",
    "🌐": "globe",
    "🔒": "lock",
    "🔐": "lock-keyhole",
    "🔗": "link",
    "🛡️": "shield",
    "🛡": "shield",
    "🏷️": "tag",
    "🏷": "tag",
    "🔩": "bolt",
    "📡": "radio",
    "🗂️": "archive",
    "🗂": "archive",
    "💻": "laptop",
    # playback / status dots
    "▶️": "play",
    "▶": "play",
    "⏸️": "pause",
    "⏸": "pause",
    "⏹️": "square",
    "⏹": "square",
    "⏭️": "skip-forward",
    "⏭": "skip-forward",
    "⏳": "hourglass",
    "🟢": "circle",
    "🔵": "circle",
    "🟡": "circle",
    "🔴": "circle",
    "⚪": "circle",
    "🔲": "square-dashed",
    "🎮": "gamepad",
    # navigation
    "⬅️": "arrow-left",
    "⬅": "arrow-left",
    "➡️": "arrow-right",
    "➡": "arrow-right",
    "⬇️": "arrow-down",
    "⬇": "arrow-down",
    # misc
    "🗄️": "database",
    "🗄": "database",
    "👋": "hand",
    "🎉": "party-popper",
    "🤖": "bot",
    "🔀": "shuffle",
    "📭": "inbox",
    "💰": "coins",
    "🏢": "building",
    "❓": "info",
    "🏃": "run",
    "✓": "check",
    "👆": "hand-pointing-up",
    "✨": "sparkles",
    "🕵️": "magnifying-glass",
    "🕵": "magnifying-glass",
}


def icon(
    name: str,
    *,
    size: int = 18,
    color: str | None = None,
    stroke_width: float = 2.0,
    vertical_align: str = "middle",
    class_name: str = "ui-icon",
) -> str:
    """Return an inline SVG string for the named icon.

    Parameters
    ----------
    name:
        Key in :data:`ICONS`. Raises ``KeyError`` for unknown names so typos
        surface immediately rather than silently rendering nothing.
    size:
        Pixel size (width and height).
    color:
        Hex/stroke color. ``None`` (default) → ``currentColor``, so the icon
        inherits the surrounding text color.
    stroke_width:
        SVG stroke width (Lucide default is 2.0).
    vertical_align:
        CSS ``vertical-align`` value — ``"middle"`` keeps icons aligned with
        adjacent text in ``st.metric`` values and markdown headings.
    class_name:
        CSS class attached to the ``<svg>`` element. ``"ui-icon"`` is styled
        in ``page_layout.apply_custom_css``.
    """
    try:
        inner = ICONS[name]
    except KeyError:
        raise KeyError(
            f"Unknown icon name: {name!r}. Add it to ICONS in ui/icons.py."
        ) from None
    stroke = color if color else "currentColor"
    return (
        f'<svg class="{escape(class_name)}" xmlns="http://www.w3.org/2000/svg" '
        f'width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="{escape(stroke)}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:{escape(vertical_align)}">{inner}</svg>'
    )


def status_badge(
    level: str,
    text: str = "",
    *,
    icon_name: str | None = None,
    size: int = 14,
) -> str:
    """Return an HTML badge ``<span>`` with semantic coloring.

    Parameters
    ----------
    level:
        One of :data:`BADGE_LEVELS` (``success``/``info``/``warning``/
        ``danger``/``muted``). Unknown levels fall back to ``muted``.
    text:
        Badge label text. Empty string renders an icon-only badge.
    icon_name:
        Optional icon to prepend. ``None`` renders text only.
    """
    if level not in BADGE_LEVELS:
        level = "muted"
    parts = [f'<span class="ui-badge ui-badge-{level}">']
    if icon_name:
        parts.append(
            icon(icon_name, size=size, color="currentColor", vertical_align="baseline")
        )
    if text:
        parts.append(escape(text))
    parts.append("</span>")
    return "".join(parts)


__all__ = [
    "BADGE_LEVELS",
    "EMOJI_TO_ICON",
    "ICONS",
    "SemanticColor",
    "icon",
    "status_badge",
]

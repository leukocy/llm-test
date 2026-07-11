"""Shared visual language for the Streamlit application."""

from __future__ import annotations

from html import escape

import streamlit as st

from config.test_types import test_type_label

GLOBAL_CSS = """
<style>
:root {
    --app-background: #f4f6fb;
    --surface-primary: #ffffff;
    --surface-secondary: #f8fafc;
    --surface-muted: #eef2f7;
    --border-subtle: #dfe5ee;
    --border-strong: #cbd5e1;
    --text-primary: #182230;
    --text-secondary: #526173;
    --text-muted: #758397;
    --accent-primary: #4f46e5;
    --accent-primary-hover: #4338ca;
    --accent-soft: #eef2ff;
    --status-success: #15803d;
    --status-success-soft: #ecfdf3;
    --status-warning: #b45309;
    --status-warning-soft: #fff7ed;
    --status-danger: #b42318;
    --status-danger-soft: #fff1f0;
    --status-info: #1d4ed8;
    --status-info-soft: #eff6ff;
    --shadow-card: 0 1px 2px rgba(15, 23, 42, 0.04), 0 8px 24px rgba(15, 23, 42, 0.05);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 18px;
}

html, body, .stApp {
    color: var(--text-primary);
    font-family: Inter, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}

.stApp {
    background: var(--app-background);
}

[data-testid="stHeader"] {
    background: rgba(244, 246, 251, 0.88);
    border-bottom: 1px solid rgba(203, 213, 225, 0.7);
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] {
    background: var(--surface-primary);
    border-right: 1px solid var(--border-subtle);
}

[data-testid="stSidebarContent"] {
    padding-top: 1.25rem;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1500px;
    padding: 2.25rem 2.5rem 4rem;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary);
    letter-spacing: -0.018em;
}

h1 {
    border: 0;
    font-size: clamp(2rem, 3vw, 2.65rem);
    font-weight: 720;
}

h2 {
    font-size: 1.5rem;
    font-weight: 680;
}

h3 {
    font-size: 1.15rem;
    font-weight: 650;
}

p, label, [data-testid="stCaptionContainer"] {
    color: var(--text-secondary);
}

hr {
    border-color: var(--border-subtle);
    margin: 1.35rem 0;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface-primary);
    border-color: var(--border-subtle);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
}

[data-testid="stMetric"] {
    min-height: 104px;
    padding: 1rem 1.1rem;
    background: var(--surface-primary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}

[data-testid="stMetricLabel"] {
    color: var(--text-muted);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--text-primary);
    font-weight: 700;
}

.stButton > button,
.stDownloadButton > button {
    min-height: 2.55rem;
    border-radius: var(--radius-sm);
    border-color: var(--border-strong);
    font-weight: 620;
    transition: border-color 120ms ease, background 120ms ease, color 120ms ease, transform 120ms ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: var(--accent-primary);
    color: var(--accent-primary);
    transform: translateY(-1px);
}

.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
    color: #ffffff;
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.22);
}

.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
    color: #ffffff;
    background: var(--accent-primary-hover);
    border-color: var(--accent-primary-hover);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
[data-testid="stNumberInputContainer"] {
    background: var(--surface-primary);
    border-color: var(--border-strong);
    border-radius: var(--radius-sm);
}

div[data-testid="stExpander"] {
    overflow: hidden;
    background: var(--surface-primary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}

div[data-testid="stAlert"] {
    border-radius: var(--radius-md);
    border-width: 1px;
}

.dashboard-hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 1.5rem;
    align-items: end;
    margin-bottom: 1.4rem;
    padding: 1.6rem 1.75rem;
    color: var(--text-primary);
    background: linear-gradient(135deg, #ffffff 0%, #f6f7ff 100%);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-card);
}

.dashboard-eyebrow {
    margin: 0 0 0.45rem;
    color: var(--accent-primary);
    font-size: 0.75rem;
    font-weight: 750;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.dashboard-title {
    margin: 0;
    color: var(--text-primary);
    font-size: clamp(1.65rem, 2.6vw, 2.25rem);
    font-weight: 740;
    letter-spacing: -0.035em;
}

.dashboard-subtitle {
    max-width: 720px;
    margin: 0.55rem 0 0;
    color: var(--text-secondary);
    font-size: 0.98rem;
    line-height: 1.6;
}

.context-summary {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 0.55rem;
}

.context-chip {
    display: inline-flex;
    flex-direction: column;
    min-width: 126px;
    padding: 0.72rem 0.85rem;
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-sm);
}

.context-chip-label {
    color: var(--text-muted);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.context-chip-value {
    max-width: 220px;
    margin-top: 0.2rem;
    overflow: hidden;
    color: var(--text-primary);
    font-size: 0.86rem;
    font-weight: 650;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    min-height: 2rem;
    padding: 0.34rem 0.72rem;
    border: 1px solid transparent;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
}

.status-idle { color: var(--text-secondary); background: var(--surface-muted); border-color: var(--border-subtle); }
.status-running { color: var(--status-info); background: var(--status-info-soft); border-color: #bfdbfe; }
.status-paused { color: var(--status-warning); background: var(--status-warning-soft); border-color: #fed7aa; }
.status-completed { color: var(--status-success); background: var(--status-success-soft); border-color: #bbf7d0; }
.status-failed, .status-cancelled { color: var(--status-danger); background: var(--status-danger-soft); border-color: #fecaca; }

.sidebar-brand {
    margin: 0 0 1rem;
    padding: 0.9rem 0.95rem;
    background: var(--surface-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}

.sidebar-brand-name {
    color: var(--text-primary);
    font-size: 0.95rem;
    font-weight: 740;
}

.sidebar-brand-caption {
    margin-top: 0.2rem;
    color: var(--text-muted);
    font-size: 0.73rem;
}

.welcome-banner {
    margin-bottom: 1rem;
    padding: 1rem 1.15rem;
    background: var(--accent-soft);
    border: 1px solid #c7d2fe;
    border-left: 4px solid var(--accent-primary);
    border-radius: var(--radius-md);
}

.welcome-banner h4 {
    margin: 0 0 0.25rem;
    color: var(--text-primary);
    font-weight: 680;
}

.welcome-banner p {
    margin: 0;
    color: var(--text-secondary);
    font-size: 0.9rem;
}

@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1.35rem 1rem 3rem;
    }

    .dashboard-hero {
        grid-template-columns: 1fr;
        padding: 1.25rem;
    }

    .context-summary {
        justify-content: flex-start;
    }

    .context-chip {
        flex: 1 1 130px;
    }
}
</style>
"""


_STATUS_CLASSES = {
    "idle": "status-idle",
    "running": "status-running",
    "paused": "status-paused",
    "completed": "status-completed",
    "failed": "status-failed",
    "cancelled": "status-cancelled",
}


def material_icon(name: str) -> str:
    """Return Streamlit's native Material Symbol syntax."""

    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("Material icon name cannot be empty")
    return f":material/{cleaned}:"


def status_badge_html(status: object) -> str:
    """Render an accessible text badge for a test status."""

    label = str(status or "Idle").strip() or "Idle"
    css_class = _STATUS_CLASSES.get(label.casefold(), "status-idle")
    return (
        f'<span class="status-badge {css_class}" role="status">'
        f"{escape(label)}</span>"
    )


def apply_design_system() -> None:
    """Inject the shared dashboard CSS once per Streamlit render."""

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar_brand() -> None:
    """Render a compact product mark at the top of the sidebar."""

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-name">LLM Benchmark</div>
            <div class="sidebar-brand-caption">Performance and quality workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_application_header(
    *,
    provider: object,
    model_id: object,
    test_type: object,
) -> None:
    """Render the stable application header and active-context summary."""

    provider_text = escape(str(provider or "Not configured"))
    model_text = escape(str(model_id or "Not configured"))
    test_text = escape(test_type_label(test_type))
    st.markdown(
        f"""
        <section class="dashboard-hero">
            <div>
                <p class="dashboard-eyebrow">Evaluation workspace</p>
                <h1 class="dashboard-title">LLM Benchmark Platform</h1>
                <p class="dashboard-subtitle">
                    Configure a workload, run a benchmark, and review comparable latency,
                    throughput, stability, and quality evidence.
                </p>
            </div>
            <div class="context-summary" aria-label="Active benchmark context">
                <span class="context-chip">
                    <span class="context-chip-label">Provider</span>
                    <span class="context-chip-value">{provider_text}</span>
                </span>
                <span class="context-chip">
                    <span class="context-chip-label">Model</span>
                    <span class="context-chip-value">{model_text}</span>
                </span>
                <span class="context-chip">
                    <span class="context-chip-label">Test</span>
                    <span class="context-chip-value">{test_text}</span>
                </span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

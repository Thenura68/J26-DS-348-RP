"""Spacing for readability on a projector. Colours live in .streamlit/config.toml."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
/* Main area: more breathing room and a comfortable reading width */
[data-testid="stMainBlockContainer"] {
    max-width: 1150px;
    padding-top: 3rem;
    padding-bottom: 4rem;
    padding-left: 3.5rem;
    padding-right: 3.5rem;
}
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] li {
    font-size: 1.05rem;
    line-height: 1.7;
}
[data-testid="stMainBlockContainer"] h1 { margin-bottom: 0.5rem; }
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3 { margin-top: 2rem; margin-bottom: 0.75rem; }
[data-testid="stVerticalBlock"] { gap: 1.25rem; }

/* Left panel: roomier navigation links */
[data-testid="stSidebarNav"] a { padding-top: 0.45rem; padding-bottom: 0.45rem; }
</style>
"""


def apply_style() -> None:
    """Inject the dashboard spacing rules; call once per run from app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)

"""Common page building blocks so every component page looks the same."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from shared.registry import component

# Sections every component page uses (they mirror the progress-presentation requirements).
STANDARD_SECTIONS = (
    "Sub-problem and approach",
    "Prototype and results",
    "User requirements met",
    "Design contribution",
)


def page_header(component_key: str) -> None:
    """Title, owner and branch strip at the top of a component page."""
    info = component(component_key)
    st.title(f"{info.key.upper()} · {info.title}")
    st.caption(f"Owner: {info.owner}  ·  Branch: `{info.branch}`")
    st.write(info.summary)


def placeholder(text: str) -> None:
    """Visible marker for content a member still has to add."""
    st.info(f"To do: {text}", icon=":material/construction:")


def show_image(path: Path, caption: str | None = None) -> None:
    """Show a saved figure, or say which script produces it if it is missing."""
    if path.is_file():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.warning(f"Not found: `{path}`. Run the script that produces it first.", icon=":material/image_not_supported:")


def show_table(frame: pd.DataFrame | None, missing_hint: str) -> None:
    """Show a DataFrame, or a hint when the source file is missing."""
    if frame is None:
        st.warning(f"No data yet. {missing_hint}", icon=":material/table:")
    else:
        st.dataframe(frame, width="stretch", hide_index=True)


def metric_row(metrics: dict[str, str]) -> None:
    """A row of headline numbers, e.g. {"Subjects": "53", "Spike density": "27%"}."""
    for column, (label, value) in zip(st.columns(len(metrics)), metrics.items()):
        column.metric(label, value)

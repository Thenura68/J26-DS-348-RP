"""C2 dashboard page. Owner: Wickramaarachchi. Edit only this folder."""

from __future__ import annotations

import streamlit as st

from shared.data import output_dir
from shared.layout import STANDARD_SECTIONS, page_header, placeholder

KEY = "c2"


def render() -> None:
    page_header(KEY)
    sub_problem, prototype, requirements, design = st.tabs(STANDARD_SECTIONS)
    with sub_problem:
        placeholder("the sub-problem this component solves and how.")
    with prototype:
        placeholder("Feature maps or firing rates from the global SNN, training curves, backbone size.")
        st.caption(f"Results folder: `{output_dir(KEY)}`")
    with requirements:
        placeholder("the user requirements this component meets.")
    with design:
        placeholder("what is new or technically strong in this component.")

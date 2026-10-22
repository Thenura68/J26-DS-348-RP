"""C1 dashboard page. Owner: Rathnayake. Edit only this folder.

The "Prototype and results" tab is a working example of the shared helpers: it loads
figures and tables that the C1 scripts save under outputs/c1/. The other tabs are
placeholders to fill in.
"""

from __future__ import annotations

import streamlit as st

from shared.data import load_csv, output_dir
from shared.layout import STANDARD_SECTIONS, page_header, placeholder, show_image, show_table

KEY = "c1"


def render() -> None:
    page_header(KEY)
    results = output_dir(KEY)
    sub_problem, prototype, requirements, design = st.tabs(STANDARD_SECTIONS)

    with sub_problem:
        placeholder("the sub-problem (EEG to sparse spikes that suit each patient) and the approach.")

    with prototype:
        st.subheader("Encoding demo")
        show_image(results / "encoding_demo" / "02010002_tbr_demo.png",
                   "Original vs decoded EEG and the UP/DOWN spike raster (scripts/c1_encode_demo.py)")
        st.subheader("Calibration length")
        show_image(results / "calibration_length" / "calibration_length.png",
                   "Threshold stability and encoding quality vs calibration length (scripts/c1_calibration_length.py)")
        st.subheader("Encoder comparison")
        show_table(load_csv(str(results / "encoder_comparison" / "summary.csv")),
                   "Run `python scripts/c1_compare_encoders.py`.")

    with requirements:
        placeholder("the user requirements C1 meets.")

    with design:
        placeholder("what is new or technically strong in C1.")

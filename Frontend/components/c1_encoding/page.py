"""C1 dashboard page. Owner: Rathnayake. Edit only this folder.

Figures come from the C1 scripts (the page only loads them):
- outputs/c1/dashboard/        ← scripts/c1_dashboard_figures.py
- outputs/c1/calibration_length/ ← scripts/c1_calibration_length.py
- outputs/c1/encoding_demo/    ← scripts/c1_encode_demo.py
- outputs/c1/encoder_comparison/ ← scripts/c1_compare_encoders.py
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from shared.data import load_csv, load_json, output_dir
from components.c1_encoding.comparison_view import render_comparison_tab
from components.c1_encoding.interactive import render_live_demo, render_output_tab
from shared.layout import STANDARD_SECTIONS, page_header, placeholder, show_image

KEY = "c1"


def _adaptation(results) -> None:
    facts = load_json(str(results / "dashboard" / "facts.json"))
    if facts is None:
        st.warning("Run `python scripts/c1_dashboard_figures.py` to build the figures for this section.")
        return
    steps = facts["steps"]
    first, second = facts["two_patients"][0], facts["two_patients"][1]

    st.header("How the adaptation works")
    st.write(
        "Every person's brain signal is different: some signals change a lot from moment to moment, "
        "others very little. One fixed rule would make some patients produce far too many spikes and "
        "others almost none. So the encoder **learns a threshold for each patient and each electrode** "
        "from a short recording at the start, then uses it for the rest."
    )

    st.subheader("1 · Split the recording in time")
    show_image(results / "dashboard" / "adaptation_split.png")
    st.write(
        f"The first **20%** of the recording (about **{facts['calibration_seconds']:.0f} s**) is the "
        f"**calibration** part: it is used only to learn the threshold. The remaining **80%** "
        f"(about **{facts['evaluation_seconds']:.0f} s**) is encoded with that threshold and never "
        "influences it, just like a real device that must be set up before it is used. "
        "Small gaps are 2-second windows removed because they contained artifacts (blinks, movement)."
    )

    st.subheader("2 – 4 · Learn θ, compare every change with it, emit spikes")
    show_image(results / "dashboard" / "adaptation_steps.png")
    st.markdown(
        f"- **Step 2.** Take the change between every two consecutive samples in the calibration part "
        f"({steps['n_changes']:,} changes for this electrode). Their typical size is the standard deviation "
        f"**std(d) = {steps['std_change']:.3f}**; their average is almost exactly 0.\n"
        f"- The threshold is **θ = mean(d) + k × std(d) = {steps['theta']:.3f}**, with one global "
        f"sensitivity **k = {facts['k']:g}** for everyone.\n"
        "- **Step 3.** Each new change is compared with θ: bigger than +θ (green) or smaller than −θ "
        "(red) counts; anything in between (grey) is ignored.\n"
        f"- **Step 4.** The result is a spike train: **UP** spikes for rises, **DOWN** spikes for falls "
        f"({steps['zoom_up_spikes']} UP and {steps['zoom_down_spikes']} DOWN in this one second)."
    )

    st.subheader("5 · Every patient gets their own θ")
    show_image(results / "dashboard" / "adaptation_two_patients.png")
    st.markdown(
        f"Patient **{second['subject_id']}**'s signal changes much less than patient "
        f"**{first['subject_id']}**'s, so it gets a smaller threshold "
        f"(**{second['theta']:.3f}** vs **{first['theta']:.3f}**). Both then produce a similar amount of "
        f"spikes (**{second['adaptive_spike_rate']:.1%}** and **{first['adaptive_spike_rate']:.1%}**). "
        f"With one fixed threshold for everyone ({facts['fixed_theta_cohort_median']:.3f}), patient "
        f"{second['subject_id']} would be almost silent: **{second['fixed_spike_rate']:.1%}** spikes."
    )


def _why_twenty_percent(results) -> None:
    st.header("Why the first 20%?")
    st.markdown(
        "- **Not the whole recording:** a threshold learned from the whole recording would have seen the "
        "very data it is tested on (data leakage), and a real device cannot see the future.\n"
        "- **Not less:** the threshold is estimated from data, so it needs enough of it.\n"
        "- **The test:** thresholds were learned from only the first 10, 20, 30, 45 or 60 seconds, and each "
        "was used to encode the unseen 80% of all 53 patients."
    )
    summary = load_csv(str(results / "calibration_length" / "summary.csv"))
    meta = load_json(str(results / "calibration_length" / "summary.json"))
    if summary is None or meta is None:
        st.warning("Run `python scripts/c1_calibration_length.py` to produce these results.")
        return
    rows = summary[summary["requested_seconds"] != "all"].copy()
    table = pd.DataFrame({
        "Calibration used": [f"{s:.0f} s" for s in rows["median_used_seconds"]],
        "θ change vs full (~60 s)": [f"{v:.1%}" for v in rows["median_theta_change_vs_full_calibration"]],
        "θ change vs long-run value": [f"{v:.1%}" for v in rows["median_theta_change_vs_long_run"]],
        "Evaluation SNR": [f"{v:.2f} dB" for v in rows["median_snr_db"]],
        "Correlation": [f"{v:.3f}" for v in rows["median_correlation"]],
        "Feature ρ": [f"{v:.3f}" for v in rows["median_feature_spearman"]],
    })
    converged = meta.get("converged_at_seconds")
    highlight = [
        not math.isnan(float(s)) and converged is not None and float(s) == float(converged)
        for s in rows["requested_seconds"]
    ]
    st.dataframe(
        table.style.apply(
            lambda row: ["background-color: #d3f9d8; font-weight: bold" if highlight[row.name] else "" for _ in row],
            axis=1,
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("The last row uses 58 s because some patients have slightly fewer clean calibration windows "
               "after artifact rejection. Green = first length meeting the convergence rule "
               "(θ within 5% of the full value and SNR within 0.1 dB).")
    show_image(results / "calibration_length" / "calibration_length.png")
    if converged is not None:
        st.success(f"The threshold settles by **{float(converged):.0f} s**, so the 20% calibration (~60 s) "
                   f"gives about a **{60 / float(converged):.0f}× safety margin**.", icon=":material/check_circle:")
    st.markdown(
        "- **θ stabilises fast:** it is within 5% of its final value from 20 s onwards.\n"
        "- **Quality hardly depends on the length:** SNR and feature preservation stay the same from 10 s to 60 s.\n"
        "- **Longer calibration does not help further:** the remaining ~8% difference from the long-run value "
        "comes from the signal changing during the recording, not from too little calibration data."
    )


def render() -> None:
    page_header(KEY)
    results = output_dir(KEY)
    sub_problem, prototype, output, comparison, requirements, design = st.tabs(
        [STANDARD_SECTIONS[0], STANDARD_SECTIONS[1], "Output for C2", "Encoder comparison", *STANDARD_SECTIONS[2:]]
    )

    with sub_problem:
        _adaptation(results)
        st.divider()
        _why_twenty_percent(results)

    with prototype:
        render_live_demo()
        st.divider()
        st.header("Saved example")
        show_image(results / "encoding_demo" / "02010002_tbr_demo.png",
                   "Original vs decoded EEG and the UP/DOWN spike raster (scripts/c1_encode_demo.py)")

    with output:
        render_output_tab()

    with comparison:
        render_comparison_tab(results)

    with requirements:
        placeholder("the user requirements C1 meets.")

    with design:
        placeholder("what is new or technically strong in C1.")

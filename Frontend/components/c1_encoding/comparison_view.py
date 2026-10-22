"""Encoder comparison tab: one card per encoder from scripts/c1_compare_encoders.py results."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from shared.data import load_csv, load_json
from shared.layout import show_image

METHODS = {
    "tbr": {
        "name": "TBR", "full": "Threshold-Based Representation", "colour": "#3b5bdb",
        "rule": "Spike when the change between samples exceeds θ", "adaptive": "θ per patient & electrode from std of changes",
        "role": "Current main encoder",
    },
    "aer": {
        "name": "AER", "full": "Fixed-threshold TBR", "colour": "#868e96",
        "rule": "Same rule as TBR, one θ for every patient", "adaptive": "Not adaptive (baseline)",
        "role": "Baseline",
    },
    "step_forward": {
        "name": "Step-Forward", "full": "Closed-loop temporal contrast", "colour": "#2f9e44",
        "rule": "Spike when the signal moves θ away from a tracked baseline", "adaptive": "θ per patient & electrode from std of signal",
        "role": "Compared",
    },
    "bsa": {
        "name": "BSA", "full": "Ben's Spiker Algorithm", "colour": "#ae3ec9",
        "rule": "Fit copies of a smoothing filter to the signal's shape", "adaptive": "Scale per patient & electrode",
        "role": "Compared",
    },
}
# (column, label, format, higher is better)
METRICS = (
    ("median_feature_spearman", "Depression features ρ", "{:.3f}", True),
    ("faa_spearman_mean", "Alpha asymmetry ρ", "{:.3f}", True),
    ("median_snr_db", "SNR", "{:.2f} dB", True),
    ("median_correlation", "Correlation", "{:.3f}", True),
    ("median_spike_density", "Spike density", "{:.1%}", None),
    ("density_cv", "Spread between patients (CV)", "{:.2f}", False),
    ("ops_per_sample", "Operations per sample", "{:.0f}", False),
)
SETTINGS = {
    "selected": "Each method at its own tuned setting",
    "density_0.30": "All methods at ≈30% spike density",
    "density_0.15": "All methods at ≈15% spike density",
}

_CSS = """
<style>
.c1-card {border: 1px solid #e9ecef; border-radius: 14px; padding: 1.1rem 1.2rem 0.9rem; background: #ffffff;
          box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06); border-top: 6px solid var(--accent); height: 100%;}
.c1-card.chosen {box-shadow: 0 0 0 3px #fab005, 0 6px 18px rgba(15, 23, 42, 0.10);}
.c1-card h3 {margin: 0; font-size: 1.35rem; color: #212529;}
.c1-card .full {color: #868e96; font-size: 0.85rem; margin-bottom: 0.55rem;}
.c1-card .chips {margin-bottom: 0.6rem;}
.c1-chip {display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px; font-size: 0.75rem;
          font-weight: 600; margin: 0 0.3rem 0.3rem 0; background: #f1f3f5; color: #495057;}
.c1-chip.role {background: var(--accent); color: #ffffff;}
.c1-chip.pick {background: #fab005; color: #212529;}
.c1-card .hero {font-size: 2.1rem; font-weight: 800; color: var(--accent); line-height: 1.1;}
.c1-card .hero-label {font-size: 0.8rem; color: #868e96; margin-bottom: 0.7rem;}
.c1-row {display: flex; justify-content: space-between; font-size: 0.92rem; padding: 0.22rem 0;
         border-bottom: 1px dashed #f1f3f5;}
.c1-row .v {font-weight: 700; color: #212529;}
.c1-best {color: #2b8a3e; font-size: 0.72rem; font-weight: 700; margin-left: 0.3rem;}
.c1-bar {height: 6px; border-radius: 3px; background: #f1f3f5; margin: 0.15rem 0 0.4rem;}
.c1-bar > div {height: 6px; border-radius: 3px; background: var(--accent);}
.c1-card .rule {font-size: 0.82rem; color: #495057; margin-top: 0.6rem; line-height: 1.45;}
</style>
"""


def _card(key: str, row: pd.Series, best: dict, chosen: bool, rho_scale: tuple[float, float]) -> str:
    meta = METHODS[key]
    rows = []
    for column, label, fmt, _ in METRICS[1:]:
        star = '<span class="c1-best">★ best</span>' if key in best.get(column, set()) else ""
        rows.append(f'<div class="c1-row"><span>{html.escape(label)}</span>'
                    f'<span class="v">{fmt.format(row[column])}{star}</span></div>')
    low, high = rho_scale
    fill = 100 * (row["median_feature_spearman"] - low) / (high - low) if high > low else 100
    hero_star = ' <span class="c1-best">★ best</span>' if key in best.get("median_feature_spearman", set()) else ""
    chips = f'<span class="c1-chip role">{html.escape(meta["role"])}</span>'
    if chosen:
        chips += '<span class="c1-chip pick">Selected by the decision rule</span>'
    return (
        f'<div class="c1-card{" chosen" if chosen else ""}" style="--accent: {meta["colour"]}">'
        f'<h3>{html.escape(meta["name"])}</h3><div class="full">{html.escape(meta["full"])}</div>'
        f'<div class="chips">{chips}</div>'
        f'<div class="hero">{row["median_feature_spearman"]:.3f}{hero_star}</div>'
        f'<div class="hero-label">Depression-related features preserved (median ρ)</div>'
        f'<div class="c1-bar"><div style="width: {max(4, min(100, fill)):.0f}%"></div></div>'
        + "".join(rows)
        + f'<div class="rule"><b>How:</b> {html.escape(meta["rule"])}<br><b>Adaptation:</b> {html.escape(meta["adaptive"])}</div>'
        "</div>"
    )


def render_comparison_tab(results) -> None:
    """Cards for TBR, AER, Step-Forward and BSA, plus the decision, plots and paired tests."""
    folder = results / "encoder_comparison"
    summary = load_csv(str(folder / "summary.csv"))
    meta = load_json(str(folder / "summary.json"))
    if summary is None or meta is None:
        st.warning("Run `python scripts/c1_compare_encoders.py` to produce the comparison.")
        return
    st.header("Encoder comparison")
    st.write("Four encoders were tuned on calibration data only and evaluated on the unseen 80% of all "
             f"{meta.get('subjects', 53)} patients, with the same rules for every method.")
    available = [s for s in SETTINGS if s in set(summary["setting"])]
    setting = st.radio("Compare", available, format_func=SETTINGS.get, horizontal=True, key="c1_cmp_setting")
    rows = summary[summary["setting"] == setting].set_index("method")
    order = [m for m in METHODS if m in rows.index]

    best = {}
    for column, _, _, higher in METRICS:
        if higher is None:
            continue
        values = rows.loc[order, column].round(3)
        target = values.max() if higher else values.min()
        best[column] = {key for key, value in values.items() if value == target}  # ties all get a star
    decision = meta.get("decision", {})
    chosen = decision.get("chosen_method") if setting == decision.get("compared_at") else None
    rho = rows.loc[order, "median_feature_spearman"]
    scale = (float(rho.min()) - 0.02, float(rho.max()))

    st.markdown(_CSS, unsafe_allow_html=True)
    for column, key in zip(st.columns(len(order)), order):
        column.markdown(_card(key, rows.loc[key], best, key == chosen, scale), unsafe_allow_html=True)

    st.write("")
    if decision.get("chosen_method"):
        tied = ", ".join(METHODS[m]["name"] for m in decision.get("tied_within_margin", []))
        st.info(
            f"**Decision rule (fixed before the run), compared at {SETTINGS.get(decision.get('compared_at'), '')}:** "
            f"best depression-feature preservation; methods within {decision.get('rule', {}).get('feature_rho_tie_margin', 0.02)} "
            f"are tied and the cheapest wins. Tied: {tied}. **Selected: {METHODS[decision['chosen_method']]['name']}** "
            f"({decision.get('reason', '')}). TBR remains the main encoder for now; moving the patient-adaptive "
            "calibration onto Step-Forward is the next step.",
            icon=":material/gavel:",
        )

    st.subheader("Details")
    left, right = st.columns(2)
    with left:
        show_image(folder / "feature_preservation.png", "Feature preservation per method and setting")
    with right:
        show_image(folder / "density_by_subject.png", "Spike density across the 53 patients (selected settings)")
    show_image(folder / "snr_vs_density_calibration.png", "Tuning curves on calibration data")

    tests = load_csv(str(folder / "paired_tests.csv"))
    if tests is not None:
        view = tests[tests["setting"] == setting][["comparison", "metric", "median_difference", "p_holm"]].copy()
        view["significant"] = view["p_holm"].map(lambda p: "yes" if pd.notna(p) and p < 0.05 else "no")
        with st.expander("Paired tests against TBR (Wilcoxon, Holm-corrected)"):
            st.dataframe(view, hide_index=True, width="stretch")
            st.caption("median_difference = TBR minus the other method, per patient. For SNR and correlation, "
                       "negative means TBR is lower; for the error metrics, negative means TBR has less error.")
    with st.expander("Settings used by each method"):
        st.json(meta.get("settings", {}), expanded=False)

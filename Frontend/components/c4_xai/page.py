"""C4 dashboard page. Owner: Edirisinghe. Edit only this folder.

Shows saved results only (scripts/c4/run_tsa.py → outputs/c4/tsa/<run>/, LSTM + SHAP baseline →
outputs/c4/baseline/ and outputs/c4/synthetic/). Nothing is computed here. outputs/ is git-ignored,
so ``assets/`` holds a committed copy of the summary results (group maps, figures averaged over
subjects, and aggregate numbers; no per-subject data) used when outputs/c4 is missing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from shared.data import load_json, output_dir
from shared.layout import STANDARD_SECTIONS, metric_row, page_header, show_image, show_table

KEY = "c4"
ASSETS = Path(__file__).parent / "assets"

# Runs of scripts/c4/run_tsa.py shown on the page: folder → label
RUNS = {
    "planted_F3_c1": "Planted signal at F3 (known answer)",
    "planted_O2_c1": "Planted signal at O2 (electrode unrelated to depression)",
    "control_c1": "Control: same labels, nothing planted",
    "real_c1": "Real MDD / HC labels",
}
SHORT = {"planted_F3_c1": "Planted at F3", "planted_O2_c1": "Planted at O2",
         "control_c1": "Control (nothing planted)", "real_c1": "Real labels"}
MAIN = "ns_leak_matched"
VERSION_NAMES = {"ns_leak_matched": "TSA-NS, leak-matched (main)", "s_leak_matched": "TSA-S, leak-matched",
                 "ns_paper": "TSA-NS, paper decay", "s_paper": "TSA-S, paper decay"}


def _path(relative: str) -> Path:
    """A result file from outputs/c4, or its committed copy in assets/ when outputs/c4 lacks it."""
    live = output_dir(KEY) / relative
    return live if live.exists() else ASSETS / relative


def _top3(version: dict) -> dict:
    """The masking row for the top-3 electrodes."""
    return next(r for r in version["masking"] if r["unit"] == "channel" and r["k"] == 3)


def _versions_table(summary: list[dict] | None) -> pd.DataFrame | None:
    if summary is None:
        return None
    rows = []
    for s in sorted(summary, key=lambda s: list(VERSION_NAMES).index(s["version"])):
        r = _top3(s)
        rows.append({"TSA version": VERSION_NAMES[s["version"]],
                     "Faithfulness: top-3 removed": round(r["faithfulness"], 2),
                     "Faithfulness: 3 random removed": round(r["faithfulness_random"], 2),
                     "Stability (target ≥ 0.7)": round(s["stability"]["mean"], 2)})
    return pd.DataFrame(rows)


def _planted_rank(run: str) -> str:
    planted = load_json(str(_path(f"tsa/{run}/planted.json")))
    if planted is None:
        return "–"
    return f"{planted[MAIN]['signed']['rank_of_planted']} / 18"


def render() -> None:
    page_header(KEY)
    sub_problem, prototype, requirements, design = st.tabs(STANDARD_SECTIONS)

    with sub_problem:
        st.subheader("The sub-problem")
        st.write("Clinicians won't trust an unexplained “depressed” prediction. Our SNN is a black box, and "
                 "because the global model learns with STDP instead of gradients, gradient-based explainers "
                 "such as SHAP cannot be used. Explanations also have to be cheap enough for an edge device.")
        st.subheader("How this component solves it: Temporal Spike Attribution (TSA)")
        st.markdown(
            "1. **Run the model once** and record which inputs and hidden neurons spiked, and when.\n"
            "2. **Trace back** from the prediction: for every path electrode → hidden neuron → class, multiply "
            "*spiked × connection weight × next neuron spiked × its vote × confidence*.\n"
            "3. **Add up** the paths per electrode and per moment → a scalp map, a timeline and the top electrodes.\n\n"
            "No gradients, no reference data, one forward pass.")
        st.code("MODMA EEG → C1 spike encoder → STDP global SNN (stand-in) → readout → TSA → scalp map + timeline",
                language=None)
        st.subheader("Baseline")
        st.write("An LSTM on EEG band power, explained with SHAP, serves as a non-spiking reference explainer.")

    with prototype:
        if not (output_dir(KEY) / "tsa").exists():
            st.caption("Showing the saved copy of the results in `assets/` (no `outputs/c4` on this computer).")
        f3 = load_json(str(_path("tsa/planted_F3_c1/xai_summary.json")))
        real = load_json(str(_path("tsa/real_c1/xai_summary.json")))
        shap = load_json(str(_path("baseline/xai_summary.json")))
        real_main = _top3(next(s for s in real if s["version"] == MAIN)) if real else None
        metric_row({
            "Planted F3: rank of F3": _planted_rank("planted_F3_c1"),
            "Planted O2: rank of O2": _planted_rank("planted_O2_c1"),
            "Faithfulness (real data)": (f"{real_main['faithfulness']:.2f} vs {real_main['faithfulness_random']:.2f}"
                                         if real_main else "–"),
            "SHAP baseline stability": f"{shap['stability']['mean']:.2f}" if shap else "–",
        })
        st.caption("Faithfulness: drop in the model's confidence when TSA's top-3 electrodes are removed, "
                   "vs 3 random electrodes. Rank: position of the planted electrode by net evidence (TSA-NS, "
                   "leak-matched).")

        st.subheader("TSA results per run")
        run = st.radio("Choose a run", list(RUNS), format_func=lambda r: SHORT[r], horizontal=True)
        st.info(RUNS[run], icon=":material/science:")
        cv = load_json(str(_path(f"tsa/{run}/cv_summary.json")))
        if cv:
            st.write(f"Model on unseen subjects: subject-level AUC **{cv['subject']['auc']:.2f}** "
                     f"(0.5 = chance). TSA explains whatever the model learned.")
        show_image(_path(f"tsa/{run}/figures/group_scalp_maps.png"),
                   "Average over test windows. Top row: size of credit. Bottom row: net evidence for class 1 "
                   "(red = for, blue = against). Each map has its own colour scale.")
        show_table(_versions_table(load_json(str(_path(f"tsa/{run}/xai_summary.json")))),
                   f"Run `python scripts/c4/run_tsa.py --name {run}` first.")
        # Per-window figures: all local ones, else the one example committed in assets/ (planted F3 run).
        examples = (sorted((output_dir(KEY) / "tsa" / run / "figures").glob("subject_*.png"))
                    or sorted((ASSETS / "tsa" / run / "figures").glob("subject_*.png")))
        if examples:
            st.subheader("One prediction explained")
            pick = st.selectbox("Example window", examples, format_func=lambda p: p.stem.replace("_", " "))
            show_image(pick, "Size map · evidence for/against · top electrodes · attribution over time (0.2 s bins)")

        st.subheader("Baseline: LSTM + SHAP")
        if shap:
            r = next(r for r in shap["masking"] if r["unit"] == "channel" and r["k"] == 3)
            st.write(f"Faithfulness {r['faithfulness']:.2f} vs {r['faithfulness_random']:.2f} random · "
                     f"stability {shap['stability']['mean']:.2f}")
        show_image(_path("synthetic/planted_signal_check.png"),
                   "SHAP planted-signal check: F3 planted vs control")
        if f3 is None:
            st.caption("Planted runs missing: run scripts/c4/run_tsa.py with --plant first.")

    with requirements:
        st.table(pd.DataFrame([
            ["FR-01 Per-prediction attribution (electrodes + time)", "Built: 18 electrodes × 4 ms"],
            ["FR-02 Scalp map at the real electrode positions", "Built"],
            ["FR-03 Attribution timeline", "Built: 0.2 s bins"],
            ["FR-04 No gradient computation", "Built: forward pass only"],
            ["FR-05a Through the global (STDP) backbone", "Built: on the STDP stand-in"],
            ["FR-06 Top contributing electrodes", "Built"],
            ["NFR-01 Readable by clinicians", "Met by design; clinician check pending"],
            ["NFR-02 Response time", "Built: 18 ms per 2 s window (laptop)"],
            ["NFR-04 Readable for colour-blind clinicians", "Met by design; checked by simulation"],
            ["FR-05b Through the patient-adaptive layer", "Milestone 3"],
        ], columns=["Requirement", "Status"]))

    with design:
        st.markdown(
            "- **Leak-matched decay (new):** the original TSA fades a spike at every layer, double-counting time, "
            "so it credits only the last ~0.1 s. Fading once, at the output's leak, fixes this.\n"
            "- **Ground-truth validation:** a planted signal with a control, repeated at an electrode unrelated to "
            "depression (O2), through the full pipeline.\n"
            "- **Signed evidence:** shows *for* vs *against* each class, so busy but irrelevant electrodes are "
            "not mistaken for important ones.\n"
            "- **Verified:** matches the original TSA method; 8 known-answer tests pass.")
        st.subheader("Original (paper) decay vs leak-matched decay, real data")
        show_table(_versions_table(real), "Run `python scripts/c4/run_tsa.py --name real_c1` first.")

"""Project overview page: pipeline, team, shared interface and dataset status."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from snn_depression.common.contracts import CLASS_NAMES, FEATURE_DIM
from shared.data import load_manifest
from shared.registry import COMPONENTS

PIPELINE = """
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica"];
  eeg [label="Preprocessed EEG\\n[B, 128, T]", fillcolor="#f1f3f5"];
  c1 [label="C1 Encoding\\nadaptive TBR spikes\\n[B, T, 256]"];
  c2 [label="C2 Global SNN\\nfeatures [B, T', 256]"];
  c3 [label="C3 Personal adapter\\nclass spikes [B, 3]"];
  c4 [label="C4 XAI\\nscalp maps, timelines"];
  out [label="non_depressed /\\nmoderate / severe", fillcolor="#f1f3f5"];
  eeg -> c1 -> c2 -> c3 -> out; c3 -> c4;
}
"""


def render() -> None:
    st.title("Patient-Adaptive depression prediction using online learning in Spiking Neural Networks for EEG signals")
    st.caption("SLIIT IT4010 · Research project J26-DS-348 · MODMA 128-channel resting EEG")
    st.write(
        "EEG is converted to spikes with patient-adaptive thresholds, a global spiking neural "
        "network learns shared features, a lightweight adapter personalises the prediction for "
        "each patient, and an explainability module shows which electrodes and times drove it."
    )
    st.graphviz_chart(PIPELINE, width="stretch")

    st.subheader("Components")
    st.dataframe(
        pd.DataFrame(
            [{"Component": c.key.upper(), "Title": c.title, "Owner": c.owner, "Branch": c.branch} for c in COMPONENTS]
        ),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Shared interface")
    left, right = st.columns(2)
    left.markdown(
        f"- Spike train (C1 → C2): `[B, T, 256]` float32, 0/1\n"
        f"- Features (C2 → C3): `[B, T', {FEATURE_DIM}]`\n"
        f"- Classes: {', '.join(f'`{name}`' for name in CLASS_NAMES)}"
    )
    right.markdown(
        "- Validation: stratified 5-fold over subjects × 5 repeats\n"
        "- Calibration: first 20% of each recording, label-free\n"
        "- Evaluation: remaining 80%"
    )

    st.subheader("Dataset status")
    manifest = load_manifest()
    if manifest is None:
        st.warning("No preprocessed data found. Run `python scripts/c1_preprocess_modma.py` first.")
        return
    counts = manifest.get("labels", {}).get("class_counts", {})
    columns = st.columns(2 + len(counts))
    columns[0].metric("Subjects", manifest.get("n_subjects", "—"))
    columns[1].metric("Channels", len(manifest.get("channels_kept", [])))
    for column, (name, count) in zip(columns[2:], counts.items()):
        column.metric(name, count)

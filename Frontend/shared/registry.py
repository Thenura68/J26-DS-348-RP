"""Single list of the four components shown in the dashboard (titles, owners, branches)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentInfo:
    key: str           # matches outputs/<key> and Frontend/components/<folder>
    folder: str
    title: str
    owner: str
    branch: str
    icon: str
    summary: str


COMPONENTS: tuple[ComponentInfo, ...] = (
    ComponentInfo(
        key="c1", folder="c1_encoding", owner="Rathnayake", branch="Encoding", icon=":material/bolt:",
        title="Patient-Adaptive EEG-to-Spike Encoding",
        summary="Converts EEG into sparse UP/DOWN spike trains with thresholds calibrated per patient, "
                "and checks that depression-relevant information survives.",
    ),
    ComponentInfo(
        key="c2", folder="c2_global_snn", owner="Wickramaarachchi", branch="Global-Model", icon=":material/hub:",
        title="Global SNN Backbone",
        summary="Learns population-level spatio-temporal features from the spike trains "
                "(LIF reservoir, STDP, intrinsic plasticity, surrogate gradients).",
    ),
    ComponentInfo(
        key="c3", folder="c3_adapter", owner="Jayarathna", branch="Personalized-Adaptive-Layer",
        icon=":material/person:",
        title="Personalised SNN Adapter",
        summary="A lightweight SNN classification head that adapts to each new patient and predicts "
                "non_depressed / moderate / severe.",
    ),
    ComponentInfo(
        key="c4", folder="c4_xai", owner="Edirisinghe", branch="Explainable-AI", icon=":material/insights:",
        title="Explainable AI: Temporal Spike Attribution",
        summary="Explains predictions by attributing them to electrodes and time (scalp maps, timelines).",
    ),
)


def component(key: str) -> ComponentInfo:
    """Look up a component by key ("c1" … "c4")."""
    for info in COMPONENTS:
        if info.key == key:
            return info
    raise KeyError(f"Unknown component {key!r}")

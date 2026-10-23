"""C3 dashboard page. Owner: Jayarathna. Edit only this folder.

Personalised SNN adapter, Milestones 1–2, on SYNTHETIC data (no real EEG yet).
All model logic stays in the backend (snn_depression.c3_adapter). The "Train adapter" button
runs the same steps as scripts/c3_toy_train.py through backend calls only; the result is cached
with st.cache_resource, so it trains once per setting (~15 s on CPU) and is instant afterwards,
including after switching pages.
"""

from __future__ import annotations

import time

import altair as alt
import pandas as pd
import streamlit as st
import torch

from shared.layout import STANDARD_SECTIONS, metric_row, page_header
from snn_depression.c3_adapter.adapter import PersonalSNNAdapter
from snn_depression.c3_adapter.synthetic import (
    ToyBackbone,
    make_synthetic_dataset,
    subject_train_test_split,
)
from snn_depression.c3_adapter.training import compute_features, train_adapter
from snn_depression.common.config import load_config
from snn_depression.common.contracts import CLASS_NAMES
from snn_depression.common.seed import set_seed
from snn_depression.evaluation.profiling import count_parameters, time_call

KEY = "c3"
# One colour per class, in CLASS_NAMES order: (non_depressed, moderate, severe).
CLASS_COLOURS = ("#2E9E5B", "#E0A100", "#D64545")
CLASS_SCALE = alt.Scale(domain=list(CLASS_NAMES), range=list(CLASS_COLOURS))
LATENCY_TARGET_MS = 200
PROFILE_TARGET_KB = 48


# ---------------------------------------------------------------------------------------------
# Experiment (backend calls only)
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _run_experiment(data_seed: int, train_seed: int, epochs: int) -> dict:
    """Synthetic data → frozen ToyBackbone → train the adapter → everything the page shows.

    Same order as scripts/c3_toy_train.py, so data seed 0, training seed 42 and 25 epochs
    reproduce that script's result. Cached per (data_seed, train_seed, epochs).
    """
    cfg = load_config("c3_adapter")
    syn = cfg["synthetic"]
    data = make_synthetic_dataset(syn, seed=data_seed)
    train_idx, test_idx = subject_train_test_split(
        data.subject_ids, data.labels, syn["test_subjects_per_class"], seed=data_seed
    )
    train, test = data.subset(train_idx), data.subset(test_idx)
    backbone = ToyBackbone(syn["n_channels"], seed=data_seed, **cfg["toy_backbone"])
    backbone_before = {k: v.clone() for k, v in backbone.state_dict().items()}
    set_seed(train_seed)
    adapter = PersonalSNNAdapter.from_config(cfg)

    start = time.perf_counter()
    history = train_adapter(
        backbone, adapter, train, test, {**cfg["training"], "epochs": epochs}, seed=train_seed
    )
    train_seconds = time.perf_counter() - start

    features = compute_features(backbone, test.spikes, batch_size=256)
    pred, silent = adapter.predict(features)
    with torch.no_grad():
        spike_record, mem_record = adapter._simulate(features)
    one_window = features[:1]
    latency = time_call(lambda: adapter.predict(one_window), n_runs=30, warmup=3)

    return {
        "settings": {"data_seed": data_seed, "train_seed": train_seed, "epochs": epochs},
        "history": history,
        "train_seconds": train_seconds,
        "n_train_windows": len(train),
        "n_train_subjects": len(train.subject_ids.unique()),
        "labels": test.labels.numpy(),
        "subjects": test.subject_ids.numpy(),
        "pred": pred.numpy(),
        "silent": silent.numpy(),
        "spikes": spike_record.numpy(),  # [N, T', 3]
        "membrane": mem_record.numpy(),  # [N, T', 3]
        "out_thresholds": adapter.lif2.threshold.detach().tolist(),
        "threshold_floor": adapter.threshold_min,
        "backbone_identical": all(
            torch.equal(v, backbone_before[k]) for k, v in backbone.state_dict().items()
        ),
        "backbone_frozen": all(not p.requires_grad for p in backbone.parameters()),
        "backbone_eval": not backbone.training,
        "n_adapter": count_parameters(adapter),
        "n_backbone": count_parameters(backbone),
        "latency": latency,
        "n_steps": syn["n_steps"],
        "n_channels": syn["n_channels"],
    }


# ---------------------------------------------------------------------------------------------
# Controls and banner
# ---------------------------------------------------------------------------------------------
def _controls() -> tuple[int, int, int] | None:
    """Data seed, training seed, epochs and the Train button, at the top of this page only."""
    cfg = load_config("c3_adapter")
    last = st.session_state.get("c3_run")
    data_seed, train_seed, epochs = last or (
        cfg["synthetic"]["seed"], cfg["training"]["seed"], cfg["training"]["epochs"]
    )
    with st.expander("Experiment settings", expanded=last is None, icon=":material/tune:"):
        with st.form("c3_form", border=False):
            left, middle, right = st.columns(3)
            new_data_seed = left.number_input(
                "Data seed", 0, 999, int(data_seed), key="c3_data_seed",
                help="Which synthetic subjects are generated and which are held out for testing.",
            )
            new_train_seed = middle.number_input(
                "Training seed", 0, 999, int(train_seed), key="c3_train_seed",
                help="The adapter's starting weights and the order of training batches.",
            )
            new_epochs = right.slider("Epochs", 5, 40, int(epochs), key="c3_epochs")
            if st.form_submit_button("Train adapter", type="primary", icon=":material/play_arrow:",
                                     key="c3_train"):
                st.session_state["c3_run"] = (int(new_data_seed), int(new_train_seed), int(new_epochs))
    return st.session_state.get("c3_run")


def _banner() -> None:
    st.info(
        "**Synthetic data only.** The adapter is trained and tested on generated spike trains "
        "with one label per subject (like MODMA), not on real EEG, so these numbers say nothing "
        "about accuracy on real patients. **Test results are on held-out subjects** never seen in "
        "training. **Label-free patient calibration is Milestone 4 and is not shown here.**",
        icon=":material/science:",
    )


def _needs_training() -> None:
    st.info("Open **Experiment settings** above and press **Train adapter** (about 15 s on CPU).",
            icon=":material/play_circle:")


def _legend() -> None:
    chips = "".join(
        f'<span style="display:inline-block;width:0.9em;height:0.9em;background:{colour};'
        f'border-radius:2px;margin:0 0.35em 0 1em;vertical-align:-0.1em"></span>{name}'
        for name, colour in zip(CLASS_NAMES, CLASS_COLOURS)
    )
    st.markdown(f"<div style='font-size:0.95rem'>{chips}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------
# Section: How it works  (tab "Sub-problem and approach")
# ---------------------------------------------------------------------------------------------
def _how_it_works() -> None:
    st.header("The sub-problem")
    st.write(
        "EEG differs a lot from person to person, so one global model fits some patients well and "
        "others badly. C3 adds a **small trainable SNN layer** on top of a **frozen** feature "
        "backbone (C2's global SNN, or C3's own fallback STDP backbone). Only this layer is trained "
        "or adapted, so personalisation is cheap enough for a wearable device and never overwrites "
        "the population knowledge in the backbone."
    )

    st.header("How it works")
    st.graphviz_chart(
        """
        digraph {
          rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];
          spikes [label="Spike train\\n[B, T, C_spk]\\nfrom C1", fillcolor="#f1f3f5"];
          backbone [label="Frozen backbone\\nC2 global SNN or\\nC3 fallback STDP+SNN", fillcolor="#e9ecef"];
          adapter [label="PersonalSNNAdapter\\nLinear 256→64 → LIF(64)\\nLinear 64→3 → LIF(3)",
                   fillcolor="#dbe4ff", color="#3b5bdb", penwidth=2];
          decision [label="Severity decision\\nspike_counts [B, 3]", fillcolor="#f1f3f5"];
          xai [label="C4 explanations\\nspike_record [B, T′, 3]", fillcolor="#f1f3f5"];
          spikes -> backbone; backbone -> adapter [label=" features\\n [B, T′, 256]", fontsize=10];
          adapter -> decision; adapter -> xai;
        }
        """,
        width="stretch",
    )
    st.caption("Only the highlighted adapter has trainable parameters. Both backbones return "
               "[B, T′, 256], so the adapter works unchanged on either.")

    adapter = PersonalSNNAdapter.from_config(load_config("c3_adapter"))
    layers = pd.DataFrame({
        "Layer": ["fc1", "lif1", "fc2", "lif2"],
        "What it does": ["Linear 256 → 64 (compress the backbone features)",
                         "64 LIF neurons, own β and V_th per neuron",
                         "Linear 64 → 3",
                         f"3 LIF output neurons: {', '.join(CLASS_NAMES)}"],
        "Parameters": [count_parameters(adapter.fc1), count_parameters(adapter.lif1),
                       count_parameters(adapter.fc2), count_parameters(adapter.lif2)],
    })
    st.dataframe(layers, width="stretch", hide_index=True)
    st.caption(f"Total: {count_parameters(adapter):,} parameters.")

    st.subheader("Each LIF neuron, at every time step")
    st.latex(r"U[t] = \beta\,U[t-1] + I[t] - S[t-1]\,V_{th}, \qquad S[t] = \Theta\big(U[t] - V_{th}\big)")
    st.markdown(
        "- **U[t]** is the membrane potential: a *state*, reset to zero for every EEG window, never saved.\n"
        "- **β** (decay) and **V_th** (threshold) are *learned*, one value per neuron.\n"
        "- Training uses a surrogate (atan) gradient, because the spike step Θ has no usable derivative.\n"
        "- **Decision:** the output neuron that fires most over the window gives the class."
    )


# ---------------------------------------------------------------------------------------------
# Section: Patient view  (tab "Prototype and results")
# ---------------------------------------------------------------------------------------------
def _patient_view(r: dict) -> None:
    st.header("Patient view")
    st.write("Pick a held-out test subject: the adapter never saw any of their windows in training.")
    subjects = sorted(set(r["subjects"].tolist()))
    true_of = {s: int(r["labels"][r["subjects"] == s][0]) for s in subjects}
    subject = st.selectbox(
        "Test subject", subjects, key="c3_subject",
        format_func=lambda s: f"Subject {s}  ·  true class: {CLASS_NAMES[true_of[s]]}",
    )
    idx = (r["subjects"] == subject).nonzero()[0]  # this subject's windows, in recording order
    true_class = true_of[subject]
    preds = r["pred"][idx]
    votes = pd.Series(preds).value_counts()
    majority = int(votes.idxmax())

    metric_row({
        "True class": CLASS_NAMES[true_class],
        "Windows correct": f"{int((preds == true_class).sum())} / {len(idx)}",
        "Majority vote": CLASS_NAMES[majority],
        "Silent windows": str(int(r["silent"][idx].sum())),
    })

    st.subheader("Prediction for every window")
    strip = pd.DataFrame({
        "window": range(1, len(idx) + 1),
        "predicted": [CLASS_NAMES[p] for p in preds],
        "row": "windows",  # constant y field: one thin strip, no legend filling the chart
    })
    chart = alt.Chart(strip).mark_bar().encode(
        x=alt.X("window:O", title="Window (recording order)"),
        y=alt.Y("row:N", axis=None),
        color=alt.Color("predicted:N", scale=CLASS_SCALE, legend=None),
        tooltip=["window", "predicted"],
    ).properties(height=70)
    st.altair_chart(chart, width="stretch")
    _legend()

    st.subheader("Inside one window")
    window = st.slider("Window", 1, len(idx), 1, key="c3_window")
    w = int(idx[window - 1])
    _window_detail(r, w)


def _window_detail(r: dict, w: int) -> None:
    spikes, membrane = r["spikes"][w], r["membrane"][w]  # [T', 3]
    counts = spikes.sum(axis=0).astype(int)
    pred, true_class = int(r["pred"][w]), int(r["labels"][w])
    outcome = "correct" if pred == true_class else f"wrong (true: {CLASS_NAMES[true_class]})"
    if r["silent"][w]:
        st.warning(f"Silent window: no output spikes. Decided by the membrane tie-break → "
                   f"**{CLASS_NAMES[pred]}** ({outcome}).", icon=":material/volume_off:")
    else:
        st.write(f"Prediction: **{CLASS_NAMES[pred]}** ({outcome}). Output spikes: "
                 + ", ".join(f"{name} {c}" for name, c in zip(CLASS_NAMES, counts)) + ".")

    steps = range(1, spikes.shape[0] + 1)
    trace = pd.DataFrame([
        {"step": t, "neuron": name, "membrane": float(membrane[t - 1, k]), "spike": int(spikes[t - 1, k])}
        for k, name in enumerate(CLASS_NAMES) for t in steps
    ])
    thresholds = pd.DataFrame({"neuron": list(CLASS_NAMES), "threshold": r["out_thresholds"]})

    left, right = st.columns([3, 2])
    with left:
        lines = alt.Chart(trace).mark_line(strokeWidth=1.5).encode(
            x=alt.X("step:Q", title="Time step"),
            y=alt.Y("membrane:Q", title="Membrane U[t]"),
            color=alt.Color("neuron:N", scale=CLASS_SCALE, legend=alt.Legend(title=None, orient="bottom")),
        )
        rules = alt.Chart(thresholds).mark_rule(strokeDash=[5, 4]).encode(
            y="threshold:Q", color=alt.Color("neuron:N", scale=CLASS_SCALE, legend=None),
        )
        st.altair_chart((lines + rules).properties(height=260, title="Output membranes (dashed = V_th)"),
                        width="stretch")
    with right:
        fired = trace[trace["spike"] == 1]
        raster = alt.Chart(fired).mark_tick(thickness=2, size=22).encode(
            x=alt.X("step:Q", title="Time step", scale=alt.Scale(domain=[1, spikes.shape[0]])),
            y=alt.Y("neuron:N", title=None, sort=list(CLASS_NAMES)),
            color=alt.Color("neuron:N", scale=CLASS_SCALE, legend=None),
        ).properties(height=160, title="Output spikes")
        st.altair_chart(raster, width="stretch")
        bars = alt.Chart(pd.DataFrame({"neuron": list(CLASS_NAMES), "spikes": counts})).mark_bar().encode(
            x=alt.X("spikes:Q", title="Spike count"),
            y=alt.Y("neuron:N", title=None, sort=list(CLASS_NAMES)),
            color=alt.Color("neuron:N", scale=CLASS_SCALE, legend=None),
        ).properties(height=100)
        st.altair_chart(bars, width="stretch")


# ---------------------------------------------------------------------------------------------
# Section: Training  (tab "Prototype and results")
# ---------------------------------------------------------------------------------------------
def _training(r: dict) -> None:
    h = r["history"]
    final = h["test"][-1]
    st.header("Training")
    st.write(f"{r['n_train_windows']} windows from {r['n_train_subjects']} training subjects; "
             f"tested on {final['n']} windows from {len(set(r['subjects'].tolist()))} held-out subjects.")
    metric_row({
        "Test accuracy": f"{final['accuracy']:.3f}",
        "Test macro-F1": f"{final['macro_f1']:.3f}",
        "Chance": f"{h['chance']:.3f}",
        "Silent test windows": f"{h['n_silent_test']} / {final['n']}",
        "Training time": f"{r['train_seconds']:.0f} s",
    })

    epochs = list(range(1, len(h["test"]) + 1))
    loss = pd.DataFrame({"epoch": [0] + epochs, "loss": h["loss"]})
    scores = pd.DataFrame(
        [{"epoch": e, "metric": "train accuracy", "value": m["accuracy"]} for e, m in zip(epochs, h["train"])]
        + [{"epoch": e, "metric": "test accuracy", "value": m["accuracy"]} for e, m in zip(epochs, h["test"])]
        + [{"epoch": e, "metric": "test macro-F1", "value": m["macro_f1"]} for e, m in zip(epochs, h["test"])]
    )
    left, right = st.columns(2)
    with left:
        st.altair_chart(
            alt.Chart(loss).mark_line(point=True).encode(
                x=alt.X("epoch:Q", title="Epoch (0 = before training)"), y=alt.Y("loss:Q", title="Loss"),
            ).properties(height=260, title="Training loss (mse_count_loss)"),
            width="stretch",
        )
    with right:
        chance = alt.Chart(pd.DataFrame({"chance": [h["chance"]]})).mark_rule(
            strokeDash=[4, 4], color="#868e96").encode(y="chance:Q")
        lines = alt.Chart(scores).mark_line(point=True).encode(
            x=alt.X("epoch:Q", title="Epoch"),
            y=alt.Y("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            # Neutral colours: red/amber/green are reserved for the three classes.
            color=alt.Color("metric:N", legend=alt.Legend(title=None, orient="bottom"),
                            scale=alt.Scale(domain=["train accuracy", "test accuracy", "test macro-F1"],
                                            range=["#adb5bd", "#3b5bdb", "#7048e8"])),
        )
        st.altair_chart((lines + chance).properties(height=260, title="Accuracy (dashed = chance)"),
                        width="stretch")

    left, right = st.columns(2)
    with left:
        cells = pd.DataFrame([
            {"actual": CLASS_NAMES[i], "predicted": CLASS_NAMES[j], "windows": n}
            for i, row in enumerate(final["confusion_matrix"]) for j, n in enumerate(row)
        ])
        base = alt.Chart(cells).encode(
            x=alt.X("predicted:N", sort=list(CLASS_NAMES), title="Predicted", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("actual:N", sort=list(CLASS_NAMES), title="True class"),
        )
        heat = base.mark_rect().encode(color=alt.Color("windows:Q", scale=alt.Scale(scheme="blues"), legend=None))
        text = base.mark_text(fontSize=15).encode(
            text="windows:Q",
            color=alt.condition(alt.datum.windows > max(final["n"] / 6, 1), alt.value("white"), alt.value("black")),
        )
        st.altair_chart((heat + text).properties(height=260, title="Confusion matrix (test windows)"),
                        width="stretch")
    with right:
        vth = pd.DataFrame({"epoch": epochs, "lowest": h["min_threshold"]})
        floor = alt.Chart(pd.DataFrame({"floor": [r["threshold_floor"]]})).mark_rule(
            strokeDash=[4, 4], color="#495057").encode(y="floor:Q")
        st.altair_chart(
            (alt.Chart(vth).mark_line(point=True).encode(
                x=alt.X("epoch:Q", title="Epoch"), y=alt.Y("lowest:Q", title="Lowest V_th"),
            ) + floor).properties(height=260, title="Lowest threshold (dashed = 0.1 floor)"),
            width="stretch",
        )
    st.caption("Training pushes some thresholds down; the clamp holds them at the floor so no neuron "
               "fires on every step.")


# ---------------------------------------------------------------------------------------------
# Section: Frozen backbone & efficiency  (tab "User requirements met")
# ---------------------------------------------------------------------------------------------
def _frozen_and_efficiency(r: dict) -> None:
    st.header("Frozen backbone & efficiency")
    profile_kb = r["n_adapter"] * 2 / 1024  # raw float16 parameters, before encryption overhead
    latency_ms = r["latency"]["median_ms"]
    frozen_ok = r["backbone_identical"] and r["backbone_frozen"] and r["backbone_eval"]
    metric_row({
        "Adapter parameters": f"{r['n_adapter']:,}",
        "Profile size (float16)": f"{profile_kb:.1f} KB",
        "Latency per window": f"{latency_ms:.1f} ms",
        "Backbone after training": "bit-identical" if frozen_ok else "CHANGED",
    })

    st.subheader("The backbone stays frozen")
    st.markdown(
        "- `freeze()` switches off gradients for every backbone parameter and puts it in evaluation mode.\n"
        "- Its features are computed once without gradients (`torch.no_grad()`).\n"
        "- The optimiser receives **only the adapter's parameters**, with a check that refuses backbone ones.\n"
        f"- After this run, every backbone tensor was compared with its copy from before training: "
        f"**{'all identical' if r['backbone_identical'] else 'differences found'}**."
    )

    ratio = r["n_adapter"] / (r["n_adapter"] + r["n_backbone"])
    needed = r["n_adapter"] / 0.05 - r["n_adapter"]
    rows = [
        ("Only the adapter learns; backbone untouched", "bit-identical backbone",
         "bit-identical" if frozen_ok else "changed", "Met" if frozen_ok else "Not met"),
        ("Inference latency on CPU", f"< {LATENCY_TARGET_MS} ms per window",
         f"{latency_ms:.1f} ms (median, T′ = {r['n_steps']})",
         "Met" if latency_ms < LATENCY_TARGET_MS else "Not met"),
        ("Patient profile size", f"≈ {PROFILE_TARGET_KB} KB",
         f"{profile_kb:.1f} KB raw float16", "Met" if profile_kb <= PROFILE_TARGET_KB else "Not met"),
        ("Trainable ratio", "< 5%",
         f"{ratio:.1%} with the {r['n_backbone']:,}-parameter toy backbone",
         f"Needs the real backbone (> {needed:,.0f} parameters)"),
        ("Patient calibration time", "≤ 2 min on CPU", "not shown", "Milestone 4"),
        ("Macro-F1 on real EEG", "≥ 0.88", "synthetic data only", "Milestone 7 (MODMA)"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Requirement", "Target", "This demo", "Status"]),
                 width="stretch", hide_index=True)
    st.caption("Latency: median of 30 timed predictions of one window on this machine's CPU. "
               "The toy backbone is deliberately tiny, so its trainable ratio says nothing about the "
               "real target.")


# ---------------------------------------------------------------------------------------------
# Section: Design contribution
# ---------------------------------------------------------------------------------------------
def _design() -> None:
    st.header("Design contribution")
    st.markdown(
        "- **Per-neuron β and V_th.** Each of the 67 LIF neurons has its own decay and threshold "
        "(134 extra parameters), giving Milestone 4's label-free calibration one adjustment per neuron.\n"
        "- **Backbone-independent.** The adapter only depends on the `[B, T′, 256]` feature contract, "
        "so it runs unchanged on C2's backbone or on C3's own fallback STDP backbone.\n"
        "- **Explainable by design.** Besides the decision it returns the full output spike record "
        "`[B, T′, 3]`, which C4 uses to attribute the decision to electrodes and time."
    )
    st.subheader("Three decisions made in Milestone 2")
    decisions = pd.DataFrame([
        ("Which loss trains the adapter",
         "snnTorch mse_count_loss: true class fires on 80% of steps, others on 20%",
         "Works for any window length; keeps output neurons neither silent nor saturated; its target "
         "rates are what Milestone 4 calibration will match."),
        ("snnTorch does not keep V_th positive",
         "Clamp V_th ≥ 0.1 and β to [0, 1] after every update",
         "Without it one neuron's V_th went negative (−0.097) and fired on every step. The saved "
         "profile keeps real, readable values."),
        ("A silent output layer defaulted to the first class",
         "Ties (incl. all-zero counts) go to the neuron closest to firing; silent windows are flagged",
         "Removes the bias towards the first class while keeping the interface contract unchanged."),
    ], columns=["Issue", "Decision", "Why"])
    st.dataframe(decisions, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------------------------
def render() -> None:
    page_header(KEY)
    _banner()
    run = _controls()
    results = None
    if run is not None:
        with st.spinner("Training the adapter on synthetic data (about 15 s on CPU)…"):
            results = _run_experiment(*run)
        s = results["settings"]
        st.caption(f"Showing: data seed {s['data_seed']} · training seed {s['train_seed']} · "
                   f"{s['epochs']} epochs")

    sub_problem, prototype, requirements, design = st.tabs(STANDARD_SECTIONS)
    with sub_problem:
        _how_it_works()
    with prototype:
        if results is None:
            _needs_training()
        else:
            _patient_view(results)
            st.divider()
            _training(results)
    with requirements:
        if results is None:
            _needs_training()
        else:
            _frozen_and_efficiency(results)
    with design:
        _design()

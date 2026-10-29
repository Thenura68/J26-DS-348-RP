"""Interactive C1 demo: run the encoding pipeline live for one patient.

Unlike the rest of the dashboard, this section computes: it runs the real C1 code
(calibration → encoder → decoder → fidelity) on the chosen patient's preprocessed EEG,
so what is shown is exactly what is handed to Component 2. Heavy steps are cached per
patient, so moving the sliders afterwards is instant.
"""

from __future__ import annotations

import io
import json
import time
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from snn_depression.common.config import load_config
from snn_depression.common.contracts import CLASS_NAMES, check_spike_train
from snn_depression.c1_encoding.calibration import ThresholdProfile
from snn_depression.c1_encoding.fidelity import signal_integrity, spectral_fidelity
from snn_depression.c1_encoding.handover import PROFILE_DIR, SPIKE_DIR
from snn_depression.c1_encoding.main_encoder import (
    DISPLAY_NAMES,
    decode,
    encode,
    fit_profile,
    main_k,
    main_method,
    make_encoder,
)
from shared.data import load_manifest, preprocessed_dir

UP, DOWN, THETA = "#2f9e44", "#e03131", "#f08c00"
STEPS = (
    "Load the EEG",
    "Calculate the 128 thresholds from the first 20%",
    "Encode the whole recording",
    "Check fidelity",
)


@lru_cache(maxsize=1)
def _config() -> dict:
    return load_config("c1_encoding")


def _method() -> str:
    """C1's main encoder (configs/c1_encoding.yaml → encoder.method)."""
    return main_method(_config())


def _k_main() -> float:
    """The k the system uses (selected on calibration data); the slider is a demo only."""
    return main_k(_config())


@st.cache_data(show_spinner=False)
def _load_eeg(subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    folder = preprocessed_dir()
    return (np.load(folder / f"{subject_id}_calibration.npy"), np.load(folder / f"{subject_id}_evaluation.npy"))


@st.cache_data(show_spinner=False)
def _fit_profile(subject_id: str, channel_names: tuple[str, ...], calibration_seconds: float, unit: str) -> dict:
    calibration, _ = _load_eeg(subject_id)
    profile = fit_profile(calibration, list(channel_names), subject_id=subject_id,
                          calibration_seconds=calibration_seconds, unit=unit, config=_config())
    return profile.to_dict()


@st.cache_data(show_spinner=False)
def _encode_whole(subject_id: str, profile_dict: dict) -> np.ndarray:
    calibration, evaluation = _load_eeg(subject_id)
    encoder = make_encoder(ThresholdProfile.from_dict(profile_dict))
    spikes = encoder.encode(torch.from_numpy(np.concatenate([calibration, evaluation])))
    check_spike_train(spikes, n_channels=256)
    return spikes.numpy().astype(np.uint8)


@st.cache_data(show_spinner=False)
def _fidelity(subject_id: str, profile_dict: dict, fs: float, channel_names: tuple[str, ...]) -> dict:
    """Decode the last 80% from its spikes and compare with the original EEG."""
    _, evaluation = _load_eeg(subject_id)
    n_calibration = int(profile_dict["calibration_window_count"])
    spikes = torch.from_numpy(_encode_whole(subject_id, profile_dict)[n_calibration:].astype(np.float32))
    eeg = torch.from_numpy(evaluation)
    thresholds = torch.tensor(profile_dict["thresholds"], dtype=torch.float64)
    rebuilt = decode(spikes, thresholds, eeg[:, :, 0].double(), profile_dict["encoder"], _config(), fs)
    fidelity_cfg = _config()["fidelity"]
    spectral = spectral_fidelity(
        eeg, rebuilt, fs, list(channel_names),
        bands={b: tuple(e) for b, e in fidelity_cfg["bands"].items()},
        asymmetry_pairs={p: list(v) for p, v in fidelity_cfg["asymmetry_pairs"].items()},
        segment_sec=float(fidelity_cfg["welch_segment_sec"]),
    )
    return {"integrity": signal_integrity(eeg, rebuilt), "spectral": spectral}


@st.cache_data(show_spinner=False)
def _electrode_xy(channel_names: tuple[str, ...], montage: str) -> np.ndarray:
    import mne

    positions = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    return np.array([positions[name][:2] for name in channel_names])


def _patient_picker(manifest: dict) -> dict:
    rows = manifest["subjects"]
    labels = {
        r["subject_id"]: f"{r['subject_id']} · {r['group']} · PHQ-9 {r['phq9']} · "
                         f"{CLASS_NAMES[r['severity_target']] if r.get('severity_target') is not None else '—'}"
        for r in rows
    }
    subject_id = st.selectbox("Patient", list(labels), format_func=labels.get, key="c1_live_patient")
    return next(r for r in rows if r["subject_id"] == subject_id)


def _run(row: dict, manifest: dict) -> None:
    subject_id = row["subject_id"]
    names = tuple(manifest["channels_kept"])
    fs = float(row["fs"])
    timings = {}
    with st.status(f"Running C1 for patient {subject_id}…", expanded=True) as status:
        start = time.perf_counter()
        st.write(f"**1 · {STEPS[0]}**")
        calibration, evaluation = _load_eeg(subject_id)
        timings[0] = time.perf_counter() - start
        st.write(f"✅ first 20%: `{list(calibration.shape)}` · last 80%: `{list(evaluation.shape)}` "
                 f"(windows × electrodes × samples) · {timings[0]:.1f} s")

        start = time.perf_counter()
        st.write(f"**2 · {STEPS[1]}**")
        profile = _fit_profile(subject_id, names, float(row["calibration_duration_sec"]), manifest["unit"])
        timings[1] = time.perf_counter() - start
        saved_path = PROFILE_DIR / f"{subject_id}.json"
        saved_note = ""
        if saved_path.is_file():
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            same = (np.allclose(saved["thresholds"], profile["thresholds"]) and saved["k"] == profile["k"]
                    and saved.get("encoder", "tbr") == profile["encoder"])
            saved_note = " · matches the saved profile ✅" if same else " · ⚠️ differs from the saved profile (re-run c1_calibrate.py)"
        st.write(f"✅ 128 thresholds, θ = mean(d) + {profile['k']:g} × std(d) of the changes, from {calibration.shape[0]} calibration windows "
                 f"only · {timings[1]:.1f} s{saved_note}")

        start = time.perf_counter()
        st.write(f"**3 · {STEPS[2]}**")
        spikes = _encode_whole(subject_id, profile)
        st.session_state[f"c1_live_{subject_id}"] = {"profile": profile, "spikes": spikes}
        timings[2] = time.perf_counter() - start
        st.write(f"✅ {DISPLAY_NAMES[profile['encoder']]} spikes `{list(spikes.shape)}` = {calibration.shape[0]} + {evaluation.shape[0]} windows × 500 steps × "
                 f"256 channels · contract check passed · {timings[2]:.1f} s")

        start = time.perf_counter()
        st.write(f"**4 · {STEPS[3]}**")
        fidelity = _fidelity(subject_id, profile, fs, names)
        st.session_state[f"c1_live_{subject_id}"]["fidelity"] = fidelity
        timings[3] = time.perf_counter() - start
        integrity = fidelity["integrity"]
        st.write(f"✅ on the unseen last 80%: SNR {integrity['snr_db']:.2f} dB · correlation {integrity['correlation']:.3f} "
                 f"· {timings[3]:.1f} s")
        status.update(label=f"C1 finished for patient {subject_id} in {sum(timings.values()):.1f} s",
                      state="complete", expanded=False)


def _results(row: dict, manifest: dict, live: dict) -> None:
    names = manifest["channels_kept"]
    profile, spikes = live["profile"], live["spikes"]
    thresholds = np.asarray(profile["thresholds"])
    total = int(spikes.sum())
    density = total / (spikes.size / 2)

    st.subheader("Results for this patient")
    columns = st.columns(5)
    columns[0].metric("Thresholds", "128")
    columns[1].metric("Median θ", f"{np.median(thresholds):.3f}")
    columns[2].metric("θ range", f"{thresholds.min():.2f} – {thresholds.max():.2f}")
    columns[3].metric("Spikes", f"{total:,}")
    columns[4].metric("Spike density", f"{density:.1%}", help="Share of electrode × time slots with a spike")

    head, table = st.columns([1.3, 1])
    with head:
        xy = _electrode_xy(tuple(names), str(manifest["channel_montage"]))
        fig, axis = plt.subplots(figsize=(5.2, 4.6))
        points = axis.scatter(xy[:, 0], xy[:, 1], c=thresholds, cmap="viridis", s=70, edgecolor="white", linewidth=0.5)
        axis.add_patch(plt.Circle((0, 0), np.abs(xy).max() * 1.08, fill=False, color="#495057", linewidth=1.5))
        axis.plot([-0.012, 0, 0.012], [np.abs(xy).max() * 1.08, np.abs(xy).max() * 1.22, np.abs(xy).max() * 1.08],
                  color="#495057", linewidth=1.5)
        axis.set_aspect("equal")
        axis.axis("off")
        fig.colorbar(points, ax=axis, shrink=0.8, label="threshold θ")
        axis.set_title("Thresholds on the head (nose at the top)", fontsize=11)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with table:
        st.dataframe(
            pd.DataFrame({"Electrode": names, "θ": np.round(thresholds, 4),
                          "std of changes": np.round(profile["derivative_std"], 4)}),
            hide_index=True, height=380, width="stretch",
        )

    spectral = live["fidelity"]["spectral"]
    st.markdown("**Fidelity on the unseen 80%**")
    integrity = live["fidelity"]["integrity"]
    columns = st.columns(3 + len(spectral["alpha_asymmetry"]))
    columns[0].metric("SNR", f"{integrity['snr_db']:.2f} dB")
    columns[1].metric("Correlation", f"{integrity['correlation']:.3f}")
    columns[2].metric("Relative band-power error", f"{np.median(list(spectral['relative_power_abs_error'].values())):.3f}")
    for column, (pair, values) in zip(columns[3:], spectral["alpha_asymmetry"].items()):
        column.metric(f"Alpha asymmetry {pair.replace('_', '/')}", f"{values['reconstructed']:+.3f}",
                      delta=f"original {values['original']:+.3f}", delta_color="off")


def _interactive_views(row: dict, manifest: dict, live: dict) -> None:
    names = manifest["channels_kept"]
    fs = float(row["fs"])
    calibration, evaluation = _load_eeg(row["subject_id"])
    eeg = np.concatenate([calibration, evaluation])
    spikes = live["spikes"]
    thresholds = np.asarray(live["profile"]["thresholds"])
    n_cal = int(live["profile"]["calibration_window_count"])

    st.subheader("Explore the spikes")
    left, middle, right = st.columns([1, 1.4, 1.6])
    electrode = left.selectbox("Electrode", names, key="c1_live_electrode")
    window = middle.slider("Window", 0, spikes.shape[0] - 1, n_cal, key="c1_live_window",
                           help=f"Windows 0–{n_cal - 1} are the first 20%, {n_cal}–{spikes.shape[0] - 1} the last 80%")
    start_ms, stop_ms = right.slider("Time range in the window (ms)", 0, 2000, (0, 600), step=20, key="c1_live_range")
    part = "first 20% (calibration)" if window < n_cal else "last 80% (evaluation)"
    st.caption(f"Window {window} is in the **{part}** · electrode {electrode} has θ = {thresholds[names.index(electrode)]:.3f}")

    c = names.index(electrode)
    lo, hi = int(start_ms * fs / 1000), max(int(start_ms * fs / 1000) + 2, int(stop_ms * fs / 1000))
    x = eeg[window, c, lo:hi]
    d = np.diff(eeg[window, c], prepend=eeg[window, c, 0])[lo:hi]
    up, down = spikes[window, lo:hi, c] > 0, spikes[window, lo:hi, 128 + c] > 0
    t = (np.arange(lo, hi) / fs) * 1000
    theta = thresholds[c]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 0.5]})
    axes[0].plot(t, x, color="black", linewidth=1.4)
    axes[0].set_ylabel("EEG")
    axes[1].bar(t, d, width=1000 / fs * 0.9, color=[UP if v > theta else DOWN if v < -theta else "#adb5bd" for v in d])
    for value in (theta, -theta):
        axes[1].axhline(value, color=THETA, linestyle="--", linewidth=1.8)
    axes[1].set_ylabel("change d (±θ)")
    axes[2].vlines(t[up], 0.55, 0.95, color=UP, linewidth=2)
    axes[2].vlines(t[down], 0.05, 0.45, color=DOWN, linewidth=2)
    axes[2].set_yticks([0.25, 0.75], ["DOWN", "UP"])
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Time in the window (ms)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    st.caption("The full 256 × 500 spike grid of each window is in the **Output for C2** tab.")


def _spike_grid(spikes: np.ndarray, window: int, highlight: int | None, label: str) -> None:
    grid = spikes[window].T  # [256, 500]
    image = np.ones((256, grid.shape[1], 3))
    image[:128][grid[:128] > 0] = np.array([47, 158, 68]) / 255
    image[128:][grid[128:] > 0] = np.array([224, 49, 49]) / 255
    fig, axis = plt.subplots(figsize=(11, 4.6))
    axis.imshow(image, aspect="auto", interpolation="nearest", extent=[0, 2000, 256, 0])
    axis.axhline(128, color="black", linewidth=1)
    if highlight is not None:
        axis.axhline(highlight + 0.5, color=THETA, linewidth=0.8, alpha=0.8)
        axis.axhline(128 + highlight + 0.5, color=THETA, linewidth=0.8, alpha=0.8)
    axis.set_yticks([64, 192], ["UP  E1–E128", "DOWN  E1–E128"])
    axis.set_xlabel("Time (ms)")
    axis.set_title(f"Window {window}: {grid.mean():.1%} of the 256 × 500 slots are spikes{label}", fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _output_for_c2(row: dict, manifest: dict, live: dict) -> None:
    subject_id = row["subject_id"]
    spikes = live["spikes"]
    n_cal = int(live["profile"]["calibration_window_count"])
    st.header(f"Output ready for Component 2 · patient {subject_id}")
    columns = st.columns(4)
    columns[0].metric("Shape", f"[{spikes.shape[0]}, {spikes.shape[1]}, {spikes.shape[2]}]")
    columns[1].metric("First 20% (calibration)", f"windows 0–{n_cal - 1}")
    columns[2].metric("Last 80% (evaluation)", f"windows {n_cal}–{spikes.shape[0] - 1}")
    columns[3].metric("Values", "0 / 1")
    st.markdown(
        f"- Layout `[windows, time steps, channels]`; channels **0–127 = UP** (E1–E128), **128–255 = DOWN**.\n"
        f"- Thresholds came only from windows 0–{n_cal - 1}. C2/C3 train on whole recordings of training "
        f"patients and score test patients on windows {n_cal} onwards only."
    )
    names = manifest["channels_kept"]
    left, right = st.columns([2, 1])
    window = left.slider("Window to display", 0, spikes.shape[0] - 1, n_cal, key="c1_out_window")
    electrode = right.selectbox("Highlight electrode", ["none", *names], key="c1_out_electrode")
    part = "first 20%" if window < n_cal else "last 80%"
    _spike_grid(spikes, window, None if electrode == "none" else names.index(electrode),
                f" · {part}" + ("" if electrode == "none" else f" · orange lines = {electrode}"))
    per_window = spikes.reshape(spikes.shape[0], -1).mean(axis=1)
    fig, axis = plt.subplots(figsize=(11, 2.6))
    axis.bar(np.arange(len(per_window)), per_window, color=["#3b5bdb" if i < n_cal else "#868e96" for i in range(len(per_window))])
    axis.axvline(n_cal - 0.5, color="black", linewidth=1.2)
    axis.set_xlabel("Window (blue = first 20%, grey = last 80%)")
    axis.set_ylabel("spike share")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    exported = SPIKE_DIR / f"{subject_id}.npz"
    if exported.is_file():
        data, source = exported.read_bytes(), "the exported file from scripts/c1_export_spikes.py"
    else:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, spikes=spikes, n_calibration_windows=np.int64(n_cal),
                            k=np.float64(live["profile"]["k"]), encoder=np.array(live["profile"]["encoder"]))
        data, source = buffer.getvalue(), "generated now (no exported file found)"
    st.download_button(f"Download {subject_id}.npz ({len(data) / 1e6:.1f} MB)", data, file_name=f"{subject_id}.npz",
                       mime="application/octet-stream", icon=":material/download:", key="c1_live_download")
    st.caption(f"Source: {source}. Load it with `snn_depression.c1_encoding.handover.load_exported_spikes`.")
    st.code(
        "from snn_depression.c1_encoding.handover import load_exported_spikes\n"
        f"recording = load_exported_spikes(\"{subject_id}\")\n"
        f"recording.spikes        # {list(spikes.shape)} float32\n"
        f"recording.calibration   # first {n_cal} windows\n"
        f"recording.evaluation    # remaining {spikes.shape[0] - n_cal} windows",
        language="python",
    )


@st.cache_data(show_spinner=False)
def _density_for(subject_id: str, k: float, profile_dict: dict) -> float:
    """Spike density on the last 80% if the thresholds used multiplier k (demo only)."""
    _, evaluation = _load_eeg(subject_id)
    mean, std = np.asarray(profile_dict["mean_derivative"]), np.asarray(profile_dict["derivative_std"])
    theta = torch.from_numpy(np.maximum(mean + k * std, float(_config()["calibration"]["minimum_threshold"]))).float()
    spikes = encode(torch.from_numpy(evaluation), theta, profile_dict["encoder"])
    return float(spikes.sum()) / (spikes.numel() / 2)


def _k_demo(row: dict, live: dict) -> None:
    profile = live["profile"]
    k_main = float(profile["k"])
    st.subheader("Try the sensitivity k (demonstration only)")
    st.caption(f"The system always uses **k = {k_main:g}**, chosen on calibration data. Moving this slider does not "
               "change the output above; it only shows how k changes the number of spikes.")
    mean, std = np.asarray(profile["mean_derivative"]), np.asarray(profile["derivative_std"])
    minimum = float(_config()["calibration"]["minimum_threshold"])
    _, evaluation = _load_eeg(row["subject_id"])
    subject_id = row["subject_id"]

    k = st.slider("k", 0.25, 4.0, k_main, 0.25, key="c1_live_k")
    current, main = _density_for(subject_id, k, profile), _density_for(subject_id, k_main, profile)
    columns = st.columns(3)
    columns[0].metric("Spike density at this k", f"{current:.1%}", delta=f"{(current - main) * 100:+.1f} points vs k = {k_main:g}")
    columns[1].metric("Median θ at this k", f"{np.median(np.maximum(mean + k * std, minimum)):.3f}")
    columns[2].metric("Spikes on the last 80%", f"{int(current * evaluation.size):,}")

    grid = np.round(np.arange(0.5, 4.01, 0.5), 2)
    curve = [_density_for(subject_id, float(value), profile) for value in grid]
    fig, axis = plt.subplots(figsize=(8, 3))
    axis.plot(grid, curve, marker="o", color="#3b5bdb")
    axis.axvline(k_main, color="#495057", linestyle=":", label=f"system k = {k_main:g}")
    axis.scatter([k], [current], color=THETA, s=120, zorder=5, label=f"slider k = {k:g}")
    axis.set_xlabel("k")
    axis.set_ylabel("spike density")
    axis.legend()
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Smaller k → more spikes (more noise kept); larger k → fewer spikes (detail lost).")


def render_live_demo() -> None:
    """Interactive section: pick a patient, run C1 step by step, explore and download the output."""
    manifest = load_manifest()
    if manifest is None:
        st.warning("No preprocessed data found. Run `python scripts/c1_preprocess_modma.py` first.")
        return
    st.header("Run the encoder live")
    st.write(f"Choose a patient and run the four C1 steps on their real EEG with the main encoder, "
             f"**{DISPLAY_NAMES[_method()]}** (k = {_k_main():g}).")
    row = _patient_picker(manifest)
    if st.button("Run encoding", type="primary", icon=":material/play_arrow:", key="c1_live_run"):
        _run(row, manifest)
    live = st.session_state.get(f"c1_live_{row['subject_id']}")
    if not live or "fidelity" not in live:
        st.info("Press **Run encoding** to process this patient.", icon=":material/info:")
        return
    st.success(f"Steps completed for patient {row['subject_id']}: " + " → ".join(f"✅ {s}" for s in STEPS))
    _results(row, manifest, live)
    st.divider()
    _interactive_views(row, manifest, live)
    st.divider()
    _k_demo(row, live)


def render_output_tab() -> None:
    """The spike tensor for the patient selected in the Prototype tab, ready for Component 2."""
    manifest = load_manifest()
    subject_id = st.session_state.get("c1_live_patient")
    live = st.session_state.get(f"c1_live_{subject_id}") if subject_id else None
    if manifest is None or not live or "fidelity" not in live:
        st.info("Choose a patient and press **Run encoding** in the **Prototype and results** tab; "
                "its output appears here.", icon=":material/info:")
        return
    row = next(r for r in manifest["subjects"] if r["subject_id"] == subject_id)
    _output_for_c2(row, manifest, live)

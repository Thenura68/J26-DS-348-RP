"""Explanatory figures for the C1 dashboard page (Frontend/components/c1_encoding).

Builds three figures from the real preprocessed data and saves them, plus the numbers
quoted on the page, under outputs/c1/dashboard/:

1. adaptation_split.png: one patient's whole recording, first 20% = calibration.
2. adaptation_steps.png: how theta is computed from the calibration changes and how
   changes become UP/DOWN spikes.
3. adaptation_two_patients.png: two patients get different thresholds and similar
   spike rates; one fixed threshold for both does not.

Settings: configs/c1_encoding.yaml → dashboard. Usage: python scripts/c1_dashboard_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.config import load_config
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.c1_encoding.calibration import ThresholdSettings, fit_thresholds
from snn_depression.c1_encoding.encoder import tbr_encode_signed

CALIBRATION = "#3b5bdb"
EVALUATION = "#868e96"
UP, DOWN, THETA = "#2f9e44", "#e03131", "#f08c00"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


def _subject(source_dir: Path, rows: dict, subject_id: str, channel: int):
    row = rows[subject_id]
    calibration = np.load(source_dir / f"{subject_id}_calibration.npy")
    evaluation = np.load(source_dir / f"{subject_id}_evaluation.npy")
    return row, calibration, evaluation


def figure_split(path: Path, row, calibration, evaluation, channel: int, channel_name: str, fs: float) -> None:
    """Whole recording on a time axis; windows sit at their true positions (gaps = rejected)."""
    window_sec = calibration.shape[-1] / fs
    split_sec = row["split_sample"] / fs
    duration = row["duration_sec"]
    fig, axis = plt.subplots(figsize=(14, 4.2))
    axis.axvspan(0, split_sec, color=CALIBRATION, alpha=0.10)
    axis.axvspan(split_sec, duration, color=EVALUATION, alpha=0.08)
    for part, windows, indices, offset, colour in (
        ("calibration", calibration, row["calibration_source_window_indices"], 0.0, CALIBRATION),
        ("evaluation", evaluation, row["evaluation_source_window_indices"], split_sec, EVALUATION),
    ):
        for window, index in zip(windows, indices):
            start = offset + index * window_sec
            time = start + np.arange(window.shape[-1])[::4] / fs
            axis.plot(time, window[channel, ::4], color=colour, linewidth=0.5)
    axis.axvline(split_sec, color="black", linewidth=1.5)
    top = float(np.abs(np.concatenate([calibration[:, channel], evaluation[:, channel]])).max())
    axis.set_ylim(-1.1 * top, 1.9 * top)
    axis.text(split_sec / 2, 1.5 * top, f"CALIBRATION\nfirst 20% · {split_sec:.0f} s\n→ learn this patient's θ",
              ha="center", va="center", color=CALIBRATION, fontweight="bold")
    axis.text((split_sec + duration) / 2, 1.5 * top,
              f"EVALUATION\nlast 80% · {duration - split_sec:.0f} s\n→ encode with that θ (never used to set it)",
              ha="center", va="center", color="#495057", fontweight="bold")
    axis.set_xlim(0, duration)
    axis.set_xlabel("Time in the recording (s)")
    axis.set_ylabel(f"{channel_name} (z-score)")
    axis.set_title(f"Step 1 · Split each patient's recording in time (patient {row['subject_id']}, electrode {channel_name})")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def figure_steps(path: Path, calibration, channel: int, channel_name: str, theta: float, fs: float,
                 zoom_seconds: float) -> dict:
    """Histogram of calibration changes with ±theta, and a close-up of changes → spikes."""
    changes = np.diff(calibration[:, channel].astype(np.float64), axis=-1).ravel()
    mean, std = float(changes.mean()), float(changes.std())
    window = calibration[0:1, channel:channel + 1].astype(np.float64)
    n = int(round(zoom_seconds * fs))
    signed = tbr_encode_signed(torch.from_numpy(window), torch.tensor([theta]))[0, 0].numpy()[:n]
    x = window[0, 0, :n]
    d = np.r_[0.0, np.diff(x)]
    time = np.arange(n) / fs * 1000

    fig = plt.figure(figsize=(14, 8.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.45], height_ratios=[1, 1, 0.55], hspace=0.55, wspace=0.25)
    hist = fig.add_subplot(grid[0:2, 0])
    formula = fig.add_subplot(grid[2, 0])
    formula.axis("off")
    bins = np.linspace(-4 * std, 4 * std, 81)
    counts, edges, patches = hist.hist(changes, bins=bins, color=CALIBRATION, alpha=0.75)
    for patch, left in zip(patches, edges[:-1]):
        if left >= theta:
            patch.set_color(UP)
        elif left + (edges[1] - edges[0]) <= -theta:
            patch.set_color(DOWN)
    for value in (theta, -theta):
        hist.axvline(value, color=THETA, linewidth=2.5, linestyle="--")
    hist.set_title("Step 2 · Measure this patient's typical change")
    hist.set_xlabel("Change between consecutive samples, d = x[t] − x[t−1]")
    hist.set_ylabel("How often (calibration data)")
    formula.text(0.0, 0.95,
                 f"{changes.size:,} changes from the first 20%\n"
                 f"mean(d) = {mean:+.4f} (≈ 0)    std(d) = {std:.3f}\n"
                 f"θ = mean(d) + k × std(d) = {theta:.3f}   (k = 1.0)",
                 transform=formula.transAxes, va="top", fontsize=12, family="monospace",
                 bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#ced4da"})
    hist.text(theta, counts.max() * 0.55, "  UP spike\n  (d > +θ)", color=UP, fontweight="bold")
    hist.text(-theta, counts.max() * 0.55, "DOWN spike  \n(d < −θ)  ", color=DOWN, fontweight="bold", ha="right")

    signal = fig.add_subplot(grid[0, 1])
    signal.plot(time, x, color="black", linewidth=1.6)
    signal.set_title(f"Step 3 · Compare every change with θ (first {zoom_seconds:.0f} s, {channel_name})")
    signal.set_ylabel("EEG")
    change = fig.add_subplot(grid[1, 1], sharex=signal)
    colours = [UP if v > theta else DOWN if v < -theta else "#adb5bd" for v in d]
    change.bar(time, d, width=1000 / fs * 0.9, color=colours)
    for value in (theta, -theta):
        change.axhline(value, color=THETA, linewidth=2, linestyle="--")
    change.text(time[-1], theta, " +θ", color=THETA, va="bottom", ha="right", fontweight="bold")
    change.text(time[-1], -theta, " −θ", color=THETA, va="top", ha="right", fontweight="bold")
    change.set_ylabel("change d")
    raster = fig.add_subplot(grid[2, 1], sharex=signal)
    up_times, down_times = time[signed > 0], time[signed < 0]
    raster.vlines(up_times, 0.55, 0.95, color=UP, linewidth=2)
    raster.vlines(down_times, 0.05, 0.45, color=DOWN, linewidth=2)
    raster.set_yticks([0.25, 0.75], ["DOWN", "UP"])
    raster.set_ylim(0, 1)
    raster.set_xlabel("Time (ms)")
    raster.set_title("Step 4 · Output spikes (one UP and one DOWN channel per electrode)")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"mean_change": mean, "std_change": std, "theta": theta, "n_changes": int(changes.size),
            "zoom_up_spikes": int((signed > 0).sum()), "zoom_down_spikes": int((signed < 0).sum())}


def figure_two_patients(path: Path, patients: list[tuple[str, np.ndarray, np.ndarray, float]], channel: int,
                        channel_name: str, fixed_theta: float) -> list[dict]:
    """Each patient's change distribution with their own theta vs one shared fixed theta."""
    fig, axes = plt.subplots(1, len(patients), figsize=(14, 4.8), sharey=False)
    facts = []
    for axis, (subject_id, calibration, evaluation, theta) in zip(axes, patients):
        changes = np.diff(calibration[:, channel].astype(np.float64), axis=-1).ravel()
        eval_changes = np.diff(evaluation[:, channel].astype(np.float64), axis=-1).ravel()
        adaptive = float((np.abs(eval_changes) > theta).mean())
        fixed = float((np.abs(eval_changes) > fixed_theta).mean())
        span = max(4 * changes.std(), 1.3 * fixed_theta)
        axis.hist(changes, bins=np.linspace(-span, span, 81), color=CALIBRATION, alpha=0.7)
        for value in (theta, -theta):
            axis.axvline(value, color=THETA, linewidth=2.5, linestyle="--")
        for value in (fixed_theta, -fixed_theta):
            axis.axvline(value, color="#495057", linewidth=1.8, linestyle=":")
        axis.set_title(f"Patient {subject_id}")
        axis.set_xlabel(f"Change d ({channel_name})")
        axis.text(0.02, 0.97,
                  f"Own θ = {theta:.3f}  → {adaptive:.1%} spikes\nFixed θ = {fixed_theta:.3f} → {fixed:.1%} spikes",
                  transform=axis.transAxes, va="top", fontsize=12,
                  bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "#ced4da"})
        facts.append({"subject_id": subject_id, "theta": theta, "adaptive_spike_rate": adaptive,
                      "fixed_spike_rate": fixed})
    axes[0].set_ylabel("How often")
    fig.suptitle("Step 5 · Every patient gets their own θ  (orange dashed = own θ, grey dotted = one fixed θ for all)",
                 fontweight="bold", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return facts


def run() -> None:
    config = load_config("c1_encoding")
    settings = config["dashboard"]
    source_dir = INTERIM_DIR / "modma_preprocessed"
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = {r["subject_id"]: r for r in manifest["subjects"]}
    names = list(manifest["channels_kept"])
    channel_name = str(settings["channel"])
    channel = names.index(channel_name)
    k = float(settings["k"])
    tbr = ThresholdSettings.from_config(config["calibration"])
    output_dir = OUTPUT_ROOT / "c1" / str(settings["output_subdirectory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    main_id, other_id = str(settings["subject_id"]), str(settings["comparison_subject_id"])
    row, calibration, evaluation = _subject(source_dir, rows, main_id, channel)
    fs = float(row["fs"])
    theta = float(fit_thresholds(calibration, k, tbr)[channel])
    figure_split(output_dir / "adaptation_split.png", row, calibration, evaluation, channel, channel_name, fs)
    steps = figure_steps(output_dir / "adaptation_steps.png", calibration, channel, channel_name, theta, fs,
                         float(settings["zoom_seconds"]))

    all_thetas = [fit_thresholds(np.load(source_dir / f"{sid}_calibration.npy"), k, tbr)[channel] for sid in rows]
    fixed_theta = float(np.median(all_thetas))
    patients = []
    for sid in (main_id, other_id):
        _, cal, ev = _subject(source_dir, rows, sid, channel)
        patients.append((sid, cal, ev, float(fit_thresholds(cal, k, tbr)[channel])))
    two = figure_two_patients(output_dir / "adaptation_two_patients.png", patients, channel, channel_name, fixed_theta)

    facts = {
        "subject_id": main_id, "channel": channel_name, "k": k,
        "calibration_seconds": row["split_sample"] / fs,
        "evaluation_seconds": row["duration_sec"] - row["split_sample"] / fs,
        "calibration_windows": len(row["calibration_source_window_indices"]),
        "evaluation_windows": len(row["evaluation_source_window_indices"]),
        "steps": steps,
        "fixed_theta_cohort_median": fixed_theta,
        "two_patients": two,
    }
    (output_dir / "facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts, indent=2))
    print(f"Figures saved under {output_dir}")


if __name__ == "__main__":
    run()

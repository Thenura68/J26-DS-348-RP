"""Demonstrate C1 TBR encoding and decoding on one held-out subject."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.config import load_config
from snn_depression.common.contracts import check_spike_train
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.calibration import (
    ThresholdSettings,
    compute_threshold_profile,
)
from snn_depression.c1_encoding.decoder import leak_from_cutoff, reconstruct
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder
from snn_depression.c1_encoding.fidelity import signal_integrity


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("dataset_complete"):
        raise ValueError(f"Preprocessed dataset is incomplete: {path}")
    return manifest, {
        str(row["subject_id"]): row for row in manifest["subjects"]
    }


def _contiguous_window_positions(
    source_indices: list[int], required_count: int
) -> list[int]:
    """Find positions of a contiguous sequence of retained source windows."""
    if required_count < 1:
        raise ValueError("contiguous_windows_for_overlay must be positive")
    for start in range(len(source_indices) - required_count + 1):
        candidate = source_indices[start : start + required_count]
        if len(candidate) == required_count and all(
            right == left + 1
            for left, right in zip(candidate, candidate[1:])
        ):
            return list(range(start, start + required_count))
    raise ValueError(
        f"No contiguous sequence of {required_count} retained evaluation windows"
    )


def _save_demo_plot(
    *,
    subject_id: str,
    channel_names: list[str],
    channel_indices: list[int],
    eeg: np.ndarray,
    reconstructed: np.ndarray,
    dual_binary: np.ndarray,
    fs: float,
    overlay_samples: int,
    plot_path: Path,
) -> None:
    """Plot three-channel overlays, a UP/DOWN raster, and spike density."""
    time = np.arange(overlay_samples, dtype=np.float64) / fs
    fig, axes = plt.subplots(
        len(channel_indices) + 2,
        1,
        figsize=(12, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [1] * len(channel_indices) + [1.2, 0.8]},
    )
    for axis_index, channel_index in enumerate(channel_indices):
        channel_name = channel_names[channel_index]
        axis = axes[axis_index]
        axis.plot(
            time,
            eeg[channel_index, :overlay_samples],
            label="preprocessed input",
            linewidth=0.9,
        )
        axis.plot(
            time,
            reconstructed[channel_index, :overlay_samples],
            label="decoded",
            linewidth=0.9,
            alpha=0.85,
        )
        axis.set_ylabel(channel_name)
        if axis_index == 0:
            axis.legend(loc="upper right")

    eeg_channel_count = len(channel_names)
    raster = np.concatenate(
        (
            dual_binary[:overlay_samples, channel_indices],
            dual_binary[
                :overlay_samples,
                [eeg_channel_count + index for index in channel_indices],
            ],
        ),
        axis=1,
    )
    raster_axis = axes[len(channel_indices)]
    raster_axis.imshow(
        raster.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        extent=(0.0, overlay_samples / fs, -0.5, raster.shape[1] - 0.5),
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
    )
    raster_axis.set_yticks(range(raster.shape[1]))
    raster_axis.set_yticklabels(
        [f"UP {channel_names[i]}" for i in channel_indices]
        + [f"DOWN {channel_names[i]}" for i in channel_indices]
    )
    raster_axis.set_ylabel("Spikes")

    density_axis = axes[-1]
    density_axis.plot(
        time,
        dual_binary[:overlay_samples].mean(axis=1),
        linewidth=0.9,
    )
    density_axis.set_ylabel("Spike\ndensity")
    density_axis.set_xlabel("Time (s)")
    fig.suptitle(
        f"Subject {subject_id}: TBR reconstruction and spike activity "
        f"({overlay_samples / fs:.1f} s)"
    )
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def run() -> None:
    """Run the configured calibration-fit/evaluation-window demonstration."""
    config = load_config("c1_encoding")
    set_seed(int(config["project"]["seed"]))
    source_dir = INTERIM_DIR / "modma_preprocessed"
    manifest, rows_by_id = _load_manifest(source_dir / "manifest.json")
    demo = config["encoding_demo"]
    subject_id = str(demo["subject_id"])
    if subject_id not in rows_by_id:
        raise ValueError(f"Subject {subject_id} is absent from the manifest")
    row = rows_by_id[subject_id]
    channel_names = list(manifest["channels_kept"])
    calibration = np.load(source_dir / f"{subject_id}_calibration.npy")
    evaluation = np.load(source_dir / f"{subject_id}_evaluation.npy")
    if not np.isfinite(calibration).all() or not np.isfinite(evaluation).all():
        raise ValueError(f"Subject {subject_id} arrays contain NaN or Inf")

    phase1_dir = OUTPUT_ROOT / "c1" / str(
        config["calibration"]["output_subdirectory"]
    )
    with (phase1_dir / "phase1_summary.json").open(encoding="utf-8") as handle:
        phase1 = json.load(handle)
    with (phase1_dir / "profiles" / f"{subject_id}.json").open(
        encoding="utf-8"
    ) as handle:
        saved_profile = json.load(handle)
    fitted_profile = compute_threshold_profile(
        calibration,
        channel_names,
        subject_id=subject_id,
        k=float(phase1["selected_global_k"]),
        settings=ThresholdSettings.from_config(config["calibration"]),
        calibration_seconds=float(row["calibration_duration_sec"]),
        unit=str(manifest["unit"]),
    )
    encoder = AdaptiveThresholdEncoder.from_profile(
        fitted_profile,
        polarity_mode=str(config["encoder"]["polarity_mode"]),
    )
    if not np.allclose(
        fitted_profile.thresholds,
        np.asarray(saved_profile["thresholds"], dtype=np.float64),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError(
            "Phase 2 fitted thresholds do not match the Phase 1 saved profile"
        )

    source_indices = [
        int(index) for index in row["evaluation_source_window_indices"]
    ]
    needed_windows = int(demo["contiguous_windows_for_overlay"])
    positions = _contiguous_window_positions(source_indices, needed_windows)
    evaluation = evaluation[positions]
    eeg = torch.from_numpy(np.asarray(evaluation, dtype=np.float32))
    dual_binary = encoder.encode(eeg)
    check_spike_train(dual_binary, n_channels=2 * len(channel_names))
    fs = float(row["fs"])
    decoder_leak = leak_from_cutoff(config["decoder"]["leak_cutoff_hz"], fs)
    decoded = reconstruct(
        dual_binary,
        encoder.thresholds,
        eeg[:, :, 0],
        polarity_mode="dual_binary",
        leak=decoder_leak,
    )

    overlay_seconds = float(demo["overlay_seconds"])
    overlay_samples = int(round(overlay_seconds * fs))
    input_signal = eeg.numpy().transpose(1, 0, 2).reshape(
        len(channel_names), -1
    )
    decoded_signal = decoded.numpy().transpose(1, 0, 2).reshape(
        len(channel_names), -1
    )
    binary_stream = dual_binary.numpy().reshape(
        -1, dual_binary.shape[-1]
    )
    if input_signal.shape[-1] < overlay_samples:
        raise ValueError(
            f"Selected windows do not cover {overlay_seconds} seconds"
        )
    channel_indices = [int(index) for index in demo["channel_indices"]]
    if len(channel_indices) != 3 or any(
        index < 0 or index >= len(channel_names) for index in channel_indices
    ):
        raise ValueError("encoding_demo.channel_indices must select three channels")

    selected_input = input_signal[channel_indices, :overlay_samples]
    selected_decoded = decoded_signal[channel_indices, :overlay_samples]
    metrics = signal_integrity(
        torch.from_numpy(selected_input), torch.from_numpy(selected_decoded)
    )
    sample_binary = binary_stream[:overlay_samples]
    metrics.update(
        {
            "spike_density": float(np.count_nonzero(sample_binary) / sample_binary.size),
            "spikes_per_second_per_eeg_channel": float(
                np.count_nonzero(sample_binary)
                / (overlay_seconds * len(channel_names))
            ),
            "overlay_seconds": overlay_seconds,
            "subject_id": subject_id,
            "group": str(row["group"]),
            "threshold_k": float(phase1["selected_global_k"]),
            "decoder_leak": decoder_leak,
            "unit": str(manifest["unit"]),
            "contract_dtype": str(dual_binary.dtype),
            "side_information": (
                "true first sample for every evaluation window and EEG channel"
            ),
        }
    )

    output_dir = OUTPUT_ROOT / "c1" / "encoding_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{subject_id}_tbr_demo.png"
    _save_demo_plot(
        subject_id=subject_id,
        channel_names=channel_names,
        channel_indices=channel_indices,
        eeg=input_signal,
        reconstructed=decoded_signal,
        dual_binary=binary_stream,
        fs=fs,
        overlay_samples=overlay_samples,
        plot_path=plot_path,
    )
    metrics_path = output_dir / f"{subject_id}_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Subject: {subject_id} ({row['group']})")
    print(f"Threshold k: {phase1['selected_global_k']}")
    print(
        f"SNR: {metrics['snr_db']:.4f} dB | "
        f"RMSE: {metrics['rmse']:.6g} {metrics['unit']} | "
        f"correlation: {metrics['correlation']:.6f}"
    )
    print(
        f"Spike density: {metrics['spike_density']:.6f} | "
        f"spikes/s/EEG channel: "
        f"{metrics['spikes_per_second_per_eeg_channel']:.4f}"
    )
    print(f"Saved plot: {plot_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    run()

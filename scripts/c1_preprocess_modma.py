#!/usr/bin/env python3
"""Run the MODMA preprocessing pipeline and write per-subject outputs.

The script keeps the shared training/test rules intact: each subject is processed as a
stand-alone recording, windows are saved in chronological order, and outputs are written to
``data/interim/modma_preprocessed``. QC plots are saved under ``outputs/c1`` to keep them
out of Git-tracked output directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import welch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.config import load_config
from snn_depression.common.contracts import check_eeg
from snn_depression.common.labels import subject_targets
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.data.modma import list_subjects, load_subject
from snn_depression.data.preprocessing import prepare_subject_windows
from snn_depression.data.splits import leave_one_subject_out


def git_commit_hash() -> str:
    """Return the current repo commit hash when available."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def git_worktree_is_dirty() -> bool | None:
    """Report whether the repository has uncommitted changes."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(status.strip())


def _window_seconds(config: dict) -> float:
    return float(config["data"]["window_sec"])


def _overlap(config: dict) -> float:
    return float(config["data"].get("window_overlap", 0.0))


def write_qc_plot(
    subject_id: str,
    eeg: np.ndarray,
    channel_names: list[str],
    fs: float,
    result: object,
    plot_path: Path,
    output_unit_label: str,
) -> None:
    """Save raw/filtered waveforms and power spectra for the first retained channel."""
    if eeg.size == 0:
        raise ValueError(f"Subject {subject_id} has no EEG samples")
    if result.calibration_windows.shape[0] == 0:
        raise ValueError(f"Subject {subject_id} has no calibration windows for QC plotting")

    channel_name = result.channel_names[0]
    raw_channel_index = channel_names.index(channel_name)
    first_kept_index = result.qc["calibration_window_indices"][0]
    window = result.calibration_windows[0, 0]
    window_samples = int(window.size)
    overlap = float(result.qc["window_overlap"])
    step_samples = max(1, int(round(window_samples * (1.0 - overlap))))
    sample_start = first_kept_index * step_samples
    raw_window = eeg[
        raw_channel_index, sample_start : sample_start + window_samples
    ]
    if raw_window.size != window_samples:
        raise ValueError(
            f"QC window for subject {subject_id} exceeds raw recording bounds"
        )

    frequencies_raw, power_raw = welch(raw_window, fs=fs, nperseg=window_samples)
    frequencies_filtered, power_filtered = welch(window, fs=fs, nperseg=window_samples)
    sample_seconds = window_samples / fs
    time = np.arange(window_samples) / fs
    raw_unit_label = "native MAT units (unconfirmed)"

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes[0, 0].plot(time, raw_window)
    axes[0, 0].set_title(f"Raw input — {channel_name}, {sample_seconds:.1f} s")
    axes[0, 0].set_ylabel(raw_unit_label)
    axes[0, 1].plot(time, window)
    axes[0, 1].set_title(f"Filtered output — {channel_name}, {sample_seconds:.1f} s")
    axes[0, 1].set_ylabel(output_unit_label)

    axes[1, 0].semilogy(frequencies_raw, power_raw)
    axes[1, 0].set_title("Raw power spectrum")
    axes[1, 0].set_ylabel(f"Power ({raw_unit_label}²/Hz)")
    axes[1, 1].semilogy(frequencies_filtered, power_filtered)
    axes[1, 1].set_title("Filtered power spectrum")
    axes[1, 1].set_ylabel(f"Power ({output_unit_label}²/Hz)")
    for axis in axes[1]:
        axis.set_xlabel("Frequency (Hz)")
        axis.set_xlim(0, min(60.0, fs / 2))
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 1].set_xlabel("Time (s)")
    fig.suptitle(f"Subject {subject_id} preprocessing QC")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def process_subject(subject_id: str, config: dict, output_dir: Path, qc_dir: Path, plot_dir: Path) -> dict:
    """Process one MODMA subject and save subject-specific outputs."""
    subject = load_subject(subject_id)
    if subject.phq9 is None:
        raise ValueError(f"Subject {subject_id} has no PHQ-9 score")
    result = prepare_subject_windows(
        subject.eeg,
        subject.channel_names,
        subject.fs,
        config,
        configured_baseline_sec=float(config["encoder"]["baseline_sec"]),
    )
    calibration_windows = np.asarray(result.calibration_windows, dtype=np.float32)
    evaluation_windows = np.asarray(result.evaluation_windows, dtype=np.float32)
    windows = np.concatenate((calibration_windows, evaluation_windows), axis=0)
    check_eeg(torch.from_numpy(calibration_windows))
    check_eeg(torch.from_numpy(evaluation_windows))
    check_eeg(torch.from_numpy(windows))

    np.save(output_dir / f"{subject_id}.npy", windows)
    np.save(output_dir / f"{subject_id}_calibration.npy", calibration_windows)
    np.save(output_dir / f"{subject_id}_evaluation.npy", evaluation_windows)

    diagnosis_target, severity_target = subject_targets(subject.group, subject.phq9)
    output_unit = result.qc["output_units"]
    calibration_windows_before = result.qc["calibration_windows_before_rejection"]
    evaluation_windows_before = result.qc["evaluation_windows_before_rejection"]
    windows_before = calibration_windows_before + evaluation_windows_before
    windows_rejected = (
        result.qc["calibration_windows_rejected"]
        + result.qc["evaluation_windows_rejected"]
    )
    qc_payload = {
        "subject_id": subject.subject_id,
        "group": subject.group,
        "phq9": subject.phq9,
        "diagnosis_target": diagnosis_target,
        "severity_target": severity_target,
        "age": subject.meta.get("age"),
        "sex": subject.meta.get("gender"),
        "n_windows": int(windows.shape[0]),
        "channels_kept": list(result.channel_names),
        "channels_dropped": [result.qc["reference_row_dropped"]],
        "channels_flagged": result.qc["channels_flagged_on_calibration_only"],
        "channels_interpolated": result.qc["channels_interpolated"],
        "windows_before_rejection": int(windows_before),
        "windows_rejected": int(windows_rejected),
        "calibration_windows_kept": int(calibration_windows.shape[0]),
        "evaluation_windows_kept": int(evaluation_windows.shape[0]),
        "split_sample": result.qc["split_sample"],
        "calibration_duration_sec": result.qc["calibration_duration_sec"],
        "evaluation_duration_sec": result.qc["evaluation_duration_sec"],
        "c1_baseline_sec": result.qc["c1_baseline_sec"],
        "artifact_threshold": result.qc["artifact_threshold"],
        "artifact_threshold_units": result.qc["artifact_threshold_units"],
        "fs": float(subject.fs),
        "reference_method": result.qc["reference_method"],
        "notch_frequency_hz": result.qc.get("notch_frequency_hz"),
        "amplitude_units": output_unit,
        "percent_windows_rejected": (
            100.0 * windows_rejected / windows_before
            if windows_before
            else 0.0
        ),
    }
    with (qc_dir / f"{subject_id}_qc.json").open("w", encoding="utf-8") as handle:
        json.dump(qc_payload, handle, indent=2)

    plot_path = plot_dir / f"{subject_id}_raw_vs_filtered.png"
    write_qc_plot(
        subject_id,
        subject.eeg,
        subject.channel_names,
        subject.fs,
        result,
        plot_path,
        output_unit,
    )

    return {
        "subject_id": subject.subject_id,
        "group": subject.group,
        "phq9": int(subject.phq9),
        "diagnosis_target": diagnosis_target,
        "severity_target": severity_target,
        "age": subject.meta.get("age"),
        "sex": subject.meta.get("gender"),
        "n_windows": int(windows.shape[0]),
        "n_calibration_windows": int(calibration_windows.shape[0]),
        "n_evaluation_windows": int(evaluation_windows.shape[0]),
        "calibration_source_window_indices": list(
            result.qc["calibration_window_indices"]
        ),
        "evaluation_source_window_indices": list(
            result.qc["evaluation_window_indices"]
        ),
        "windows_before_rejection": int(windows_before),
        "windows_rejected": int(windows_rejected),
        "percent_windows_rejected": qc_payload["percent_windows_rejected"],
        "channels_kept": list(result.channel_names),
        "channels_dropped": [result.qc["reference_row_dropped"]],
        "channels_interpolated": list(result.qc["channels_interpolated"]),
        "artifacts_removed": {
            "channels_flagged": result.qc["channels_flagged_on_calibration_only"],
            "channels_interpolated": list(result.qc["channels_interpolated"]),
            "calibration_windows_rejected": int(result.qc["calibration_windows_rejected"]),
            "evaluation_windows_rejected": int(result.qc["evaluation_windows_rejected"]),
        },
        "fs": float(subject.fs),
        "duration_sec": float(subject.eeg.shape[-1] / subject.fs),
        "split_sample": int(result.qc["split_sample"]),
        "calibration_duration_sec": result.qc["calibration_duration_sec"],
        "evaluation_duration_sec": result.qc["evaluation_duration_sec"],
        "c1_baseline_sec": result.qc["c1_baseline_sec"],
        "artifact_threshold": result.qc["artifact_threshold"],
        "artifact_threshold_units": result.qc["artifact_threshold_units"],
        "unit": qc_payload["amplitude_units"],
        "output_npy": str((output_dir / f"{subject_id}.npy").relative_to(REPO_ROOT)),
        "calibration_npy": str(
            (output_dir / f"{subject_id}_calibration.npy").relative_to(REPO_ROOT)
        ),
        "evaluation_npy": str(
            (output_dir / f"{subject_id}_evaluation.npy").relative_to(REPO_ROOT)
        ),
    }


def write_metadata_csv(rows: list[dict], path: Path) -> None:
    """Write per-subject processing metadata to CSV."""
    fieldnames = [
        "subject_id",
        "group",
        "phq9",
        "diagnosis_target",
        "severity_target",
        "age",
        "sex",
        "n_windows",
        "n_calibration_windows",
        "n_evaluation_windows",
        "windows_before_rejection",
        "windows_rejected",
        "percent_windows_rejected",
        "channels_kept",
        "channels_dropped",
        "channels_interpolated",
        "artifacts_removed",
        "fs",
        "duration_sec",
        "split_sample",
        "calibration_duration_sec",
        "evaluation_duration_sec",
        "c1_baseline_sec",
        "artifact_threshold",
        "artifact_threshold_units",
        "unit",
        "output_npy",
        "calibration_npy",
        "evaluation_npy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "subject_id": row["subject_id"],
                "group": row["group"],
                "phq9": row["phq9"],
                "diagnosis_target": row["diagnosis_target"],
                "severity_target": row["severity_target"],
                "age": row["age"],
                "sex": row["sex"],
                "n_windows": row["n_windows"],
                "n_calibration_windows": row["n_calibration_windows"],
                "n_evaluation_windows": row["n_evaluation_windows"],
                "windows_before_rejection": row["windows_before_rejection"],
                "windows_rejected": row["windows_rejected"],
                "percent_windows_rejected": row["percent_windows_rejected"],
                "channels_kept": ";".join(row["channels_kept"]),
                "channels_dropped": ";".join(row["channels_dropped"]),
                "channels_interpolated": ";".join(row["channels_interpolated"]),
                "artifacts_removed": json.dumps(row["artifacts_removed"], sort_keys=True),
                "fs": row["fs"],
                "duration_sec": row["duration_sec"],
                "split_sample": row["split_sample"],
                "calibration_duration_sec": row["calibration_duration_sec"],
                "evaluation_duration_sec": row["evaluation_duration_sec"],
                "c1_baseline_sec": row["c1_baseline_sec"],
                "artifact_threshold": row["artifact_threshold"],
                "artifact_threshold_units": row["artifact_threshold_units"],
                "unit": row["unit"],
                "output_npy": row["output_npy"],
                "calibration_npy": row["calibration_npy"],
                "evaluation_npy": row["evaluation_npy"],
            })


def build_fold_summaries(
    subject_rows: list[dict],
) -> list[dict]:
    """Summarize leave-one-subject-out folds without splitting any subject's windows."""
    subject_ids = [str(row["subject_id"]) for row in subject_rows]
    if len(subject_ids) < 2:
        raise ValueError("At least two subjects are required to summarize subject-wise folds")
    if len(set(subject_ids)) != len(subject_ids):
        raise ValueError("Subject metadata contains duplicate subject IDs")
    row_by_id = {str(row["subject_id"]): row for row in subject_rows}
    groups = [str(row["group"]) for row in subject_rows]
    fold_summaries = []

    for fold_number, (test_subject, train_indices, test_indices) in enumerate(
        leave_one_subject_out(subject_ids), start=1
    ):
        test_id = str(test_subject)
        train_ids = [subject_ids[index] for index in train_indices]
        test_ids = [subject_ids[index] for index in test_indices]
        if test_ids != [test_id] or test_id in train_ids:
            raise ValueError(f"Subject-wise split leakage detected in fold {fold_number}")

        test_row = row_by_id[test_id]
        calibration_count = int(test_row["n_calibration_windows"])
        evaluation_count = int(test_row["n_evaluation_windows"])
        if calibration_count < 1 or evaluation_count < 1:
            raise ValueError(
                f"Subject {test_id} has an empty calibration or evaluation partition"
            )
        train_group_counts = dict(Counter(groups[index] for index in train_indices))
        fold_summaries.append(
            {
                "fold": fold_number,
                "test_subject_id": test_id,
                "test_group": test_row["group"],
                "training_subject_ids": train_ids,
                "training_group_counts": train_group_counts,
                "test_windows": int(test_row["n_windows"]),
                "calibration_window_count": calibration_count,
                "evaluation_window_count": evaluation_count,
                "calibration_window_source_indices": [
                    int(test_row["calibration_source_window_indices"][0]),
                    int(test_row["calibration_source_window_indices"][-1]),
                ],
                "evaluation_window_source_indices": [
                    int(test_row["evaluation_source_window_indices"][0]),
                    int(test_row["evaluation_source_window_indices"][-1]),
                ],
                "split_sample": int(test_row["split_sample"]),
                "c1_baseline_sec": float(test_row["c1_baseline_sec"]),
            }
        )
    return fold_summaries


def write_manifest(
    config: dict,
    subject_rows: list[dict],
    output_dir: Path,
    expected_subject_ids: list[str],
) -> None:
    """Write a JSON manifest and README for the processed dataset."""
    processed_ids = {str(row["subject_id"]) for row in subject_rows}
    dataset_complete = processed_ids == set(expected_subject_ids)
    group_counts = dict(Counter(row["group"] for row in subject_rows))
    calibration_fraction = float(config.get("split", {}).get("calibration_fraction", 0.2))
    folds = (
        build_fold_summaries(subject_rows)
        if dataset_complete
        else []
    )
    (output_dir / "folds.json").write_text(
        json.dumps(
            {
                "strategy": "leave_one_subject_out",
                "dataset_complete": dataset_complete,
                "calibration_fraction": calibration_fraction,
                "folds": folds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    scaling_settings = config["preprocessing"].get("amplitude_scaling", {})
    scaling_method = scaling_settings.get("method", "zscore")
    output_unit = (
        subject_rows[0]["unit"]
        if subject_rows
        else "not available (no subjects processed)"
    )
    source_unit = (
        scaling_settings.get("verified_input_unit")
        if scaling_method == "microvolts"
        else "unconfirmed native MAT values; no unit conversion applied"
    )
    reference_method = "average"
    reference_description = (
        "the 129th acquisition-reference row was dropped, then CAR was applied "
        "to E1-E128"
    )
    manifest = {
        "dataset": "MODMA EEG_128channels_resting_lanzhou_2015",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(),
        "git_worktree_dirty": git_worktree_is_dirty(),
        "n_subjects": len(subject_rows),
        "expected_subjects": len(expected_subject_ids),
        "dataset_complete": dataset_complete,
        "group_counts": group_counts,
        "channels_kept": list(dict.fromkeys(
            channel for row in subject_rows for channel in row["channels_kept"]
        )),
        "channels_dropped": list(dict.fromkeys(
            channel for row in subject_rows for channel in row["channels_dropped"]
        )),
        "channels_interpolated": list(dict.fromkeys(
            channel
            for row in subject_rows
            for channel in row["channels_interpolated"]
        )),
        "unit": output_unit,
        "source_unit": source_unit,
        "reference": reference_description,
        "channel_montage": config["preprocessing"].get("montage", "GSN-HydroCel-128"),
        "subject_wise_split": {
            "strategy": "leave_one_subject_out",
            "folds_file": "folds.json",
            "n_folds": len(folds),
            "calibration_fraction": calibration_fraction,
            "split_before_processing": True,
        },
        "filter_settings": {
            "low_hz": config["data"]["bandpass_hz"][0],
            "high_hz": config["data"]["bandpass_hz"][1],
            "filter_order": config["data"].get("filter_order", 4),
            "notch_enabled": config["preprocessing"]["notch"].get("enabled", True),
            "notch_hz": (
                config["preprocessing"]["notch"].get("frequency_hz")
                if config["preprocessing"]["notch"].get("enabled", True)
                else None
            ),
            "window_sec": _window_seconds(config),
            "window_overlap": _overlap(config),
            "reference_method": reference_method,
            "filter_padding_sec": config["preprocessing"].get("filter_padding_sec", 2.0),
            "scaling_method": scaling_method,
        },
        "artifact_rules": {
            "flat_std_threshold": config["preprocessing"]["artifact"].get("flat_std_threshold"),
            "max_std_ratio": config["preprocessing"]["artifact"].get("max_std_ratio"),
            "peak_to_peak_mad_multiplier": config["preprocessing"]["artifact"].get("peak_to_peak_mad_multiplier"),
            "ica_enabled": config["preprocessing"]["artifact"].get("ica", {}).get("enabled", False),
            "threshold_fit_partition": "calibration only",
        },
        "outputs": {
            "combined_array_shape": "[N, C, W]",
            "calibration_array": "<subject_id>_calibration.npy",
            "evaluation_array": "<subject_id>_evaluation.npy",
            "array_dtype": "float32",
            "metadata_csv": "metadata.csv",
            "qc_directory": "qc/",
        },
        "subjects": subject_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = f"""# MODMA preprocessing outputs

This directory contains leakage-safe MODMA EEG preprocessing outputs. Calibration and
evaluation were split on raw sample indices before fitting any data-dependent preprocessing
statistics. Outputs are `{output_unit}` `[N, C, W]` float32 windows.

- Dataset: {manifest['dataset']}
- Subjects processed: {manifest['n_subjects']}
- Complete dataset export: {manifest['dataset_complete']}
- Group counts: {manifest['group_counts']}
- Unit: {manifest['unit']}
- Reference: {manifest['reference']}
- Filter settings: {manifest['filter_settings']}
- Artifact rules: {manifest['artifact_rules']}
- Window length: {manifest['filter_settings']['window_sec']} s
- Window overlap: {manifest['filter_settings']['window_overlap']}
- Git commit: {manifest['git_commit']}

Files:
- `metadata.csv` — subject/demographic data, window rejection rate, channel/artifact summary, sampling rate, duration and units
- `manifest.json` — JSON manifest for the run
- `folds.json` — subject-wise folds and per-fold MDD/HC training counts; empty unless all subjects are processed
- `<subject_id>_calibration.npy` — calibration windows only
- `<subject_id>_evaluation.npy` — evaluation windows only
- `<subject_id>.npy` — combined chronological windows for compatibility
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process MODMA EEG subject recordings and save QC outputs.")
    parser.add_argument("--subject", help="Optional single subject ID to process; defaults to every subject.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on the number of subjects to process.")
    parser.add_argument("--config", default="c1_encoding", help="YAML config name under configs/.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)
    config = load_config(args.config)

    output_dir = INTERIM_DIR / "modma_preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = OUTPUT_ROOT / "c1"
    plot_dir.mkdir(parents=True, exist_ok=True)

    expected_subject_ids = list_subjects()
    subject_ids = [args.subject] if args.subject else expected_subject_ids.copy()
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be a positive integer")
        subject_ids = subject_ids[: args.limit]

    rows: list[dict] = []
    for subject_id in tqdm(subject_ids, desc="Processing MODMA subjects"):
        row = process_subject(subject_id, config, output_dir, qc_dir, plot_dir)
        rows.append(row)
        print(f"Processed {subject_id}: {row['n_windows']} windows, channels={len(row['channels_kept'])}")

    write_metadata_csv(rows, output_dir / "metadata.csv")
    write_manifest(config, rows, output_dir, expected_subject_ids)

    print(f"Saved {len(rows)} processed subjects to {output_dir}")
    print(f"QC plots: {plot_dir}")


if __name__ == "__main__":
    main()

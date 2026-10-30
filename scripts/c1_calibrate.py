"""Fit C1 TBR thresholds using only each subject's calibration windows."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.config import load_config
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.calibration import (
    ThresholdProfile,
    ThresholdSettings,
    channel_region_map,
    check_calibration_windows_inside_split,
    compute_threshold_profile,
    sweep_calibration_k,
)
from snn_depression.c1_encoding.comparison import build_methods, select_setting, sweep_method
from snn_depression.c1_encoding.decoder import leak_from_cutoff
from snn_depression.c1_encoding.main_encoder import DISPLAY_NAMES, k_candidates, main_k, main_method


def _index_ranges(indices: list[int]) -> str:
    """Compact sorted integer indices into inclusive ranges for the audit printout."""
    if not indices:
        return "(none)"
    ordered = sorted(set(int(index) for index in indices))
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for index in ordered[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous))
        start = previous = index
    ranges.append((start, previous))
    return ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in ranges
    )


def _read_inputs(
    manifest_path: Path, array_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Read manifest and calibration arrays, rejecting incomplete/inconsistent exports."""
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("dataset_complete"):
        raise ValueError(f"Preprocessing manifest is incomplete: {manifest_path}")
    if not manifest.get("subjects"):
        raise ValueError("Preprocessing manifest contains no subjects")
    channels = manifest.get("channels_kept")
    if not channels:
        raise ValueError("Preprocessing manifest does not list retained channels")
    unit = manifest.get("unit")
    if not unit:
        raise ValueError("Preprocessing manifest does not specify the output unit")

    calibration_by_subject: dict[str, np.ndarray] = {}
    for row in manifest["subjects"]:
        subject_id = str(row["subject_id"])
        path = array_dir / f"{subject_id}_calibration.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Missing calibration array for {subject_id}: {path}")
        windows = np.load(path, mmap_mode="r")
        if windows.ndim != 3 or windows.shape[1] != len(channels):
            raise ValueError(
                f"Invalid calibration shape for {subject_id}: {windows.shape}; "
                f"expected [N, {len(channels)}, T]"
            )
        if windows.shape[0] != int(row["n_calibration_windows"]):
            raise ValueError(
                f"Manifest/array calibration count mismatch for {subject_id}"
            )
        calibration_by_subject[subject_id] = np.asarray(windows)
    return manifest, list(manifest["subjects"]), calibration_by_subject


def _profile_from_row(
    row: dict[str, Any],
    windows: np.ndarray,
    channel_names: list[str],
    config: dict[str, Any],
    k: float,
    unit: str,
    method: str,
) -> ThresholdProfile:
    """Create one final profile (for the main encoder) after checking calibration windows precede the split."""
    data = config["data"]
    check_calibration_windows_inside_split(
        row.get("calibration_source_window_indices", []),
        split_sample=int(row["split_sample"]),
        fs=float(row["fs"]),
        window_sec=float(data["window_sec"]),
        window_overlap=float(data.get("window_overlap", 0.0)),
    )
    return compute_threshold_profile(
        windows,
        channel_names,
        subject_id=str(row["subject_id"]),
        k=k,
        settings=ThresholdSettings.from_config(config["calibration"]),
        calibration_seconds=float(row["calibration_duration_sec"]),
        unit=unit,
        encoder=method,
    )


def _sweep_main_encoder(config: dict[str, Any], calibration_by_subject, decoder_leak: float, fs: float):
    """Sweep k for the main encoder on calibration windows; same rule for every encoder.

    TBR keeps its own sweep (leaky decoder). Step-Forward is scored with the encoder
    comparison's interface (plain integration decoder). Rows: k, median/mean subject SNR,
    median spike density, subjects scored.
    """
    calibration_config = config["calibration"]
    method = main_method(config)
    if method == "tbr":
        return sweep_calibration_k(
            calibration_by_subject,
            k_candidates=k_candidates(config, method),
            settings=ThresholdSettings.from_config(calibration_config),
            selection_metric=str(calibration_config["selection_metric"]),
            snr_tolerance_db=float(calibration_config["snr_tolerance_db"]),
            decoder_leak=decoder_leak,
        )
    encoder = build_methods(
        fs=fs,
        tbr_settings=ThresholdSettings.from_config(calibration_config),
        decoder_leak=decoder_leak,
        step_forward_minimum_threshold=float(config["step_forward"]["minimum_threshold"]),
        bsa_scale_percentile=float(config["bsa"]["scale_percentile"]),
        bsa_minimum_scale=float(config["bsa"]["minimum_scale"]),
    )["step_forward_changes"]
    rows = sweep_method(encoder, {"k": k_candidates(config, method)}, calibration_by_subject)
    best = select_setting(rows, str(calibration_config["selection_metric"]), float(calibration_config["snr_tolerance_db"]))
    curve = [
        {"k": float(row["params"]["k"]), "median_subject_snr_db": row["median_subject_snr_db"],
         "mean_subject_snr_db": row["mean_subject_snr_db"], "median_spike_density": row["median_spike_density"],
         "subjects_scored": row["subjects_scored"]}
        for row in rows
    ]
    return float(best["params"]["k"]), curve


def _representative_subjects(
    rows: list[dict[str, Any]], count: int
) -> list[str]:
    """Choose representatives spanning available groups, then fill deterministically."""
    if count < 1:
        raise ValueError("report_subject_count must be positive")
    selected: list[str] = []
    seen_groups: set[str] = set()
    for row in rows:
        group = str(row["group"])
        if group not in seen_groups:
            selected.append(str(row["subject_id"]))
            seen_groups.add(group)
            if len(selected) == count:
                return selected
    for row in rows:
        subject_id = str(row["subject_id"])
        if subject_id not in selected:
            selected.append(subject_id)
            if len(selected) == count:
                break
    return selected


def _write_profile(profile: ThresholdProfile, profile_dir: Path) -> None:
    """Persist a human-readable JSON profile and numeric NPZ arrays."""
    (profile_dir / f"{profile.subject_id}.json").write_text(
        json.dumps(profile.to_dict(), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        profile_dir / f"{profile.subject_id}.npz",
        thresholds=profile.thresholds,
        mean_derivative=profile.mean_derivative,
        derivative_std=profile.derivative_std,
        channel_names=np.asarray(profile.channel_names),
    )


def _write_k_curve(curve: list[dict[str, Any]], output_dir: Path) -> None:
    """Write the calibration-only SNR-vs-k table and plot."""
    csv_path = output_dir / "snr_vs_k.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "k",
                "median_subject_snr_db",
                "mean_subject_snr_db",
                "median_spike_density",
                "subjects_scored",
            ],
        )
        writer.writeheader()
        writer.writerows(curve)
    (output_dir / "snr_vs_k.json").write_text(
        json.dumps(curve, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    k_values = [row["k"] for row in curve]
    snr_values = [row["median_subject_snr_db"] for row in curve]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(k_values, snr_values, marker="o")
    axis.set_xlabel("k")
    axis.set_ylabel("Median subject calibration SNR (dB)")
    axis.set_title("Calibration-only TBR k sweep")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "snr_vs_k.png", dpi=180)
    plt.close(fig)


def _write_channel_reports(
    profiles: list[ThresholdProfile],
    rows_by_id: dict[str, dict[str, Any]],
    regions: dict[str, list[str]],
    representative_ids: list[str],
    output_dir: Path,
) -> None:
    """Write per-channel values and representative regional summaries."""
    channel_path = output_dir / "per_channel_thresholds.csv"
    with channel_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject_id",
                "group",
                "channel",
                "threshold",
                "mean_derivative",
                "derivative_std",
                "unit",
            ],
        )
        writer.writeheader()
        for profile in profiles:
            group = str(rows_by_id[profile.subject_id]["group"])
            for index, channel in enumerate(profile.channel_names):
                writer.writerow(
                    {
                        "subject_id": profile.subject_id,
                        "group": group,
                        "channel": channel,
                        "threshold": float(profile.thresholds[index]),
                        "mean_derivative": float(profile.mean_derivative[index]),
                        "derivative_std": float(profile.derivative_std[index]),
                        "unit": profile.unit,
                    }
                )

    profile_by_id = {profile.subject_id: profile for profile in profiles}
    summary_path = output_dir / "representative_thresholds_by_region.csv"
    summary_rows: list[dict[str, Any]] = []
    for subject_id in representative_ids:
        profile = profile_by_id[subject_id]
        row = rows_by_id[subject_id]
        for region, channels in regions.items():
            indices = [
                index
                for index, name in enumerate(profile.channel_names)
                if name in channels
            ]
            if not indices:
                continue
            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "group": row["group"],
                    "region": region,
                    "channel_count": len(indices),
                    "median_threshold": float(
                        np.median(profile.thresholds[indices])
                    ),
                    "mean_derivative": float(
                        np.mean(profile.mean_derivative[indices])
                    ),
                    "unit": profile.unit,
                }
            )
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0]) if summary_rows else [],
        )
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    print("\nRepresentative subject thresholds by coarse montage region")
    print("subject_id  group  region     n_channels  median_threshold  mean_derivative  unit")
    for row in summary_rows:
        print(
            f"{row['subject_id']}  {row['group']:<4}  {row['region']:<9}  "
            f"{row['channel_count']:>10}  {row['median_threshold']:>16.6g}  "
            f"{row['mean_derivative']:>15.6g}  {row['unit']}"
        )


def run() -> None:
    """Execute Phase 1 using only the manifest's calibration arrays."""
    config = load_config("c1_encoding")
    set_seed(int(config["project"]["seed"]))
    source_dir = INTERIM_DIR / "modma_preprocessed"
    manifest_path = source_dir / "manifest.json"
    manifest, rows, calibration_by_subject = _read_inputs(
        manifest_path, source_dir
    )
    calibration_config = config["calibration"]
    unit = str(manifest["unit"])
    channel_names = list(manifest["channels_kept"])

    # The exported arrays are separate by partition; these indices are local
    # to their partition and retained after calibration-fitted artifact rejection.
    check_count = int(calibration_config["index_check_subject_count"])
    print("Calibration/evaluation retained source-window index audit:")
    for row in rows[:check_count]:
        calibration_indices = row.get("calibration_source_window_indices", [])
        evaluation_indices = row.get("evaluation_source_window_indices", [])
        print(
            f"  {row['subject_id']}: calibration="
            f"[{_index_ranges(calibration_indices)}], evaluation="
            f"[{_index_ranges(evaluation_indices)}]"
        )

    sampling_rates = {float(row["fs"]) for row in rows}
    if len(sampling_rates) != 1:
        raise ValueError(f"Expected one sampling rate, got {sorted(sampling_rates)}")
    fs = sampling_rates.pop()
    method = main_method(config)
    leak_cutoff_hz = config["decoder"]["leak_cutoff_hz"]
    decoder_leak = leak_from_cutoff(leak_cutoff_hz, fs)
    print(f"\nMain encoder: {DISPLAY_NAMES[method]} ({method}); sweeping k on calibration windows only")
    best_k, curve = _sweep_main_encoder(config, calibration_by_subject, decoder_leak, fs)
    if best_k != main_k(config):
        print(f"WARNING: selected k = {best_k:g} differs from configs/c1_encoding.yaml encoder.k = "
              f"{main_k(config):g}; update encoder.k so the dashboard and handover agree.")
    output_dir = OUTPUT_ROOT / "c1" / str(
        calibration_config["output_subdirectory"]
    )
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    _write_k_curve(curve, output_dir)

    profile_by_id: dict[str, ThresholdProfile] = {}
    rows_by_id = {str(row["subject_id"]): row for row in rows}
    for row in rows:
        subject_id = str(row["subject_id"])
        profile = _profile_from_row(
            row,
            calibration_by_subject[subject_id],
            channel_names,
            config,
            best_k,
            unit,
            method,
        )
        _write_profile(profile, profile_dir)
        profile_by_id[subject_id] = profile

    report_count = int(calibration_config["report_subject_count"])
    representative_ids = _representative_subjects(rows, report_count)
    region_config = calibration_config["channel_regions"]
    regions = channel_region_map(
        channel_names,
        montage_name=str(manifest["channel_montage"]),
        anterior_y_min_m=float(region_config["anterior_y_min_m"]),
        posterior_y_max_m=float(region_config["posterior_y_max_m"]),
    )
    profiles = list(profile_by_id.values())
    summary = {
        "main_encoder": method,
        "selected_global_k": best_k,
        "selection_metric": calibration_config["selection_metric"],
        "snr_tolerance_db": float(calibration_config["snr_tolerance_db"]),
        "decoder_leak_cutoff_hz": leak_cutoff_hz if method == "tbr" else None,
        "decoder_leak": decoder_leak if method == "tbr" else 1.0,
        "selection_data": "calibration windows only",
        "include_mean": bool(calibration_config["include_mean"]),
        "unit": unit,
        "window_sec": float(config["data"]["window_sec"]),
        "window_overlap": float(config["data"].get("window_overlap", 0.0)),
        "subject_count": len(rows),
        "subjects": [
            {
                "subject_id": profile.subject_id,
                "group": rows_by_id[profile.subject_id]["group"],
                "calibration_seconds": profile.calibration_seconds,
                "calibration_window_count": profile.calibration_window_count,
                "flagged_channels": profile.flagged_channels,
            }
            for profile in profiles
        ],
        "snr_vs_k": curve,
    }
    (output_dir / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(
        f"\nSelected global k: {best_k:g} "
        f"for {DISPLAY_NAMES[method]} ({calibration_config['selection_metric']})"
    )
    print("Calibration-only SNR-vs-k:")
    for point in curve:
        print(
            f"  k={point['k']:g}: median={point['median_subject_snr_db']:.4f} dB; "
            f"mean={point['mean_subject_snr_db']:.4f} dB; "
            f"density={point['median_spike_density']:.3f}"
        )
    _write_channel_reports(
        profiles,
        rows_by_id,
        regions,
        representative_ids,
        output_dir,
    )
    print(f"\nProfiles and reports saved under {output_dir}")


if __name__ == "__main__":
    run()

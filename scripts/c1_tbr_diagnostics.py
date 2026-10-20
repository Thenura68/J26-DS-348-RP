"""Diagnose TBR drift and compare Step-Forward using an unnormalized variant.

The variant is rebuilt under DATA_ROOT from the MODMA source recordings using
the existing shared preprocessing helpers, but leaves out calibration Z-scoring.
It remains local data and must not be committed.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
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
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.decoder import reconstruct
from snn_depression.c1_encoding.encoder import tbr_encode_signed
from snn_depression.c1_encoding.step_forward import (
    calibration_std_thresholds,
    pooled_calibration_std_threshold,
)
from snn_depression.data.modma import load_subject
from snn_depression.data.preprocessing import (
    _filter_chunk_with_padding,
    _interpolate_bad_channels,
    flag_bad_channels,
    make_windows,
    rereference,
)


def _load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("dataset_complete"):
        raise ValueError(f"Preprocessed manifest is incomplete: {path}")
    rows = manifest.get("subjects")
    if not rows:
        raise ValueError("Preprocessed manifest has no subject rows")
    return manifest, rows


def _prepare_unnormalized_subject(
    row: dict[str, Any],
    *,
    config: dict[str, Any],
    channel_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one subject's unnormalized windows without changing shared code.

    Outputs are calibration/evaluation windows [N, C, W] in filtered,
    CAR-referenced native MAT units. The source amplitude unit is unconfirmed.
    The raw-time split is made before channel checks/statistics, and the artifact
    limit is fit from calibration windows only.
    """
    subject_id = str(row["subject_id"])
    subject = load_subject(subject_id)
    raw = np.asarray(subject.eeg, dtype=np.float64)
    split_fraction = float(config["split"]["calibration_fraction"])
    split_sample = int(np.floor(raw.shape[1] * split_fraction))
    if not 0 < split_sample < raw.shape[1]:
        raise ValueError(f"Invalid calibration boundary for {subject_id}")
    calibration_raw = raw[:, :split_sample].copy()
    evaluation_raw = raw[:, split_sample:].copy()
    if channel_names != [f"E{i}" for i in range(1, 129)]:
        raise ValueError("Unnormalized MODMA variant expects E1-E128 channel order")
    scalp_names = channel_names
    calibration = calibration_raw[:128]
    evaluation = evaluation_raw[:128]

    artifact = config["preprocessing"]["artifact"]
    bad_channels = flag_bad_channels(
        calibration,
        scalp_names,
        flat_std_threshold=float(artifact["flat_std_threshold"]),
        max_std_ratio=float(artifact["max_std_ratio"]),
        variance_mad_multiplier=float(artifact["variance_mad_multiplier"]),
    )
    bad_names = list(bad_channels)
    montage = str(config["preprocessing"]["montage"])
    calibration = _interpolate_bad_channels(
        calibration, scalp_names, bad_names, subject.fs, montage
    )
    evaluation = _interpolate_bad_channels(
        evaluation, scalp_names, bad_names, subject.fs, montage
    )
    calibration = rereference(calibration, method="average")
    evaluation = rereference(evaluation, method="average")
    calibration = _filter_chunk_with_padding(
        calibration, subject.fs, config
    )
    evaluation = _filter_chunk_with_padding(
        evaluation, subject.fs, config
    )
    window_sec = float(config["data"]["window_sec"])
    overlap = float(config["data"].get("window_overlap", 0.0))
    calibration_windows = make_windows(
        calibration, subject.fs, window_sec, overlap
    )
    evaluation_windows = make_windows(
        evaluation, subject.fs, window_sec, overlap
    )
    if not calibration_windows.shape[0] or not evaluation_windows.shape[0]:
        raise ValueError(f"Subject {subject_id} has an empty window partition")

    ptp_multiplier = float(artifact["peak_to_peak_mad_multiplier"])
    calibration_ptp = np.ptp(calibration_windows, axis=-1).max(axis=1)
    median = float(np.median(calibration_ptp))
    mad = float(np.median(np.abs(calibration_ptp - median)))
    artifact_limit = median + ptp_multiplier * 1.4826 * mad
    calibration_keep = calibration_ptp <= artifact_limit
    evaluation_ptp = np.ptp(evaluation_windows, axis=-1).max(axis=1)
    evaluation_keep = evaluation_ptp <= artifact_limit
    if not calibration_keep.any() or not evaluation_keep.any():
        raise ValueError(
            f"Unnormalized artifact threshold rejects a complete partition for {subject_id}"
        )
    calibration_kept = np.asarray(
        calibration_windows[calibration_keep], dtype=np.float32
    )
    evaluation_kept = np.asarray(
        evaluation_windows[evaluation_keep], dtype=np.float32
    )
    metadata = {
        "subject_id": subject_id,
        "group": str(row["group"]),
        "fs": float(subject.fs),
        "unit": "native MAT values (unverified); no normalization or conversion",
        "split_sample": split_sample,
        "calibration_duration_sec": split_sample / subject.fs,
        "n_calibration_windows": int(calibration_kept.shape[0]),
        "n_evaluation_windows": int(evaluation_kept.shape[0]),
        "calibration_windows_rejected": int((~calibration_keep).sum()),
        "evaluation_windows_rejected": int((~evaluation_keep).sum()),
        "bad_channels_interpolated": bad_names,
        "artifact_threshold": float(artifact_limit),
    }
    return calibration_kept, evaluation_kept, metadata


def _ensure_unnormalized_variant(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    selected_ids: set[str],
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Rebuild and save all unnormalized calibration arrays and cohort evals."""
    output_dir = INTERIM_DIR / str(
        config["diagnostics"]["unnormalized_variant_subdirectory"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_names = list(manifest["channels_kept"])
    output_subjects = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    existing_path = output_dir / "manifest.json"
    existing_by_id: dict[str, dict[str, Any]] = {}
    if existing_path.is_file():
        with existing_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_by_id = {
            str(item["subject_id"]): item for item in existing.get("rows", [])
        }
    for row in rows:
        subject_id = str(row["subject_id"])
        cal_path = output_dir / f"{subject_id}_calibration.npy"
        eval_path = output_dir / f"{subject_id}_evaluation.npy"
        metadata = existing_by_id.get(subject_id)
        if (
            metadata is None
            or not cal_path.is_file()
            or (subject_id in selected_ids and not eval_path.is_file())
        ):
            calibration, evaluation, metadata = _prepare_unnormalized_subject(
                row, config=config, channel_names=channel_names
            )
            np.save(cal_path, calibration)
            if subject_id in selected_ids:
                np.save(eval_path, evaluation)
        rows_by_id[subject_id] = metadata
        output_subjects.append(metadata)
        print(
            f"Unnormalized {subject_id}: "
            f"cal={'ready' if cal_path.is_file() else 'missing'}, "
            f"eval={'ready' if eval_path.is_file() else 'not cached'}"
        )
    variant_manifest = {
        "dataset": manifest["dataset"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": "same split/filter/reference/artifact rules; calibration z-score omitted",
        "source_amplitude_unit": "unverified native MAT values",
        "channels": channel_names,
        "sampling_rate_hz": config["data"]["sampling_rate_hz"],
        "window_sec": config["data"]["window_sec"],
        "window_overlap": config["data"].get("window_overlap", 0.0),
        "calibration_fraction": config["split"]["calibration_fraction"],
        "rows": output_subjects,
    }
    existing_path.write_text(
        json.dumps(variant_manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_dir, rows_by_id


def _select_subjects(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> set[str]:
    """Select configured numbers from MDD and HC deterministically."""
    diagnostics = config["diagnostics"]
    selected: list[str] = []
    for group, count_key in (
        ("MDD", "mdd_subject_count"),
        ("HC", "hc_subject_count"),
    ):
        group_rows = [row for row in rows if str(row["group"]).upper() == group]
        count = int(diagnostics[count_key])
        if len(group_rows) < count:
            raise ValueError(f"Need {count} {group} subjects, found {len(group_rows)}")
        selected.extend(str(row["subject_id"]) for row in group_rows[:count])
    if len(selected) != int(diagnostics["subject_count"]):
        raise ValueError("Configured diagnostic group counts do not sum to subject_count")
    return set(selected)


def encode_signed(windows: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """NumPy wrapper around the package TBR encoder: [N, C, T] -> int8 [N, C, T]."""
    values = torch.from_numpy(np.asarray(windows, dtype=np.float64))
    return tbr_encode_signed(values, np.asarray(thresholds, dtype=np.float64)).numpy()


def reconstruct_signed(
    spikes: np.ndarray, thresholds: np.ndarray, initial_values: np.ndarray
) -> np.ndarray:
    """NumPy wrapper around the standard (non-leaky) package TBR decoder."""
    return reconstruct(
        torch.from_numpy(np.asarray(spikes, dtype=np.int8)),
        np.asarray(thresholds, dtype=np.float64),
        np.asarray(initial_values, dtype=np.float64),
        polarity_mode="signed",
    ).numpy()

def _metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Compute SNR, RMSE, and correlation over matching non-empty arrays."""
    x = np.asarray(original, dtype=np.float64)
    r = np.asarray(reconstructed, dtype=np.float64)
    if x.shape != r.shape or x.size == 0:
        raise ValueError("Metric arrays must have matching, non-empty shapes")
    if not np.isfinite(x).all() or not np.isfinite(r).all():
        raise ValueError("Metric arrays contain NaN or Inf")
    error = x - r
    signal_power = float(np.square(x).sum())
    error_power = float(np.square(error).sum())
    if error_power == 0:
        snr = float("inf") if signal_power > 0 else float("nan")
    elif signal_power == 0:
        snr = float("-inf")
    else:
        snr = float(10.0 * np.log10(signal_power / error_power))
    x_flat, r_flat = x.ravel(), r.ravel()
    correlation = (
        float(np.corrcoef(x_flat, r_flat)[0, 1])
        if np.std(x_flat) > 0 and np.std(r_flat) > 0
        else float("nan")
    )
    return {
        "snr_db": snr,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "correlation": correlation,
    }


def _tbr_thresholds(
    windows: np.ndarray,
    k: float,
    *,
    include_mean: bool,
    minimum_threshold: float,
) -> np.ndarray:
    differences = np.diff(np.asarray(windows, dtype=np.float64), axis=-1)
    derivative_mean = differences.mean(axis=(0, 2))
    derivative_std = differences.std(axis=(0, 2))
    raw_threshold = k * derivative_std
    if include_mean:
        raw_threshold = raw_threshold + derivative_mean
    return np.maximum(raw_threshold, minimum_threshold)


def _tbr_scalar_threshold(
    windows: np.ndarray,
    k: float,
    *,
    include_mean: bool,
    minimum_threshold: float,
) -> float:
    differences = np.diff(np.asarray(windows, dtype=np.float64), axis=-1).ravel()
    threshold = k * float(differences.std())
    if include_mean:
        threshold += float(differences.mean())
    return max(threshold, minimum_threshold)


def _tbr_result(
    windows: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    spikes = encode_signed(windows, thresholds)
    reconstructed = reconstruct_signed(
        spikes, thresholds, np.asarray(windows)[:, :, 0]
    )
    metrics = _metrics(windows, reconstructed)
    metrics["spike_density"] = float(np.count_nonzero(spikes) / spikes.size)
    return reconstructed, metrics


def _sf_signed_numpy(windows: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Numpy Step-Forward recurrence matching step_forward_signed."""
    values = np.asarray(windows, dtype=np.float64)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if threshold_values.ndim == 1:
        threshold_values = threshold_values[None, :, None]
    elif threshold_values.ndim == 2:
        threshold_values = threshold_values[:, :, None]
    spikes = np.zeros(values.shape, dtype=np.int8)
    baseline = values[..., 0].copy()
    for time_index in range(1, values.shape[-1]):
        upward = values[..., time_index] > baseline + threshold_values.squeeze(-1)
        downward = values[..., time_index] < baseline - threshold_values.squeeze(-1)
        events = upward.astype(np.int8) - downward.astype(np.int8)
        spikes[..., time_index] = events
        baseline += events * threshold_values.squeeze(-1)
    return spikes


def _sf_result(
    windows: np.ndarray, thresholds: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    signed = _sf_signed_numpy(windows, thresholds)
    reconstructed = reconstruct_signed(
        signed, np.asarray(thresholds), np.asarray(windows)[:, :, 0]
    )
    metrics = _metrics(windows, reconstructed)
    metrics["spike_density"] = float(np.count_nonzero(signed) / signed.size)
    return reconstructed, metrics


def _calibration_curves(
    rows: list[dict[str, Any]],
    calibration_dir: Path,
    *,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sweep TBR and SF parameters on calibration arrays only."""
    tbr_ks = [float(value) for value in config["calibration"]["k_candidates"]]
    sf_ks = [float(value) for value in config["step_forward"]["k_candidates"]]
    tbr_points: list[dict[str, Any]] = []
    sf_points: list[dict[str, Any]] = []
    for k in tbr_ks:
        subject_metrics = []
        for row in rows:
            subject_id = str(row["subject_id"])
            calibration = np.load(
                calibration_dir / f"{subject_id}_calibration.npy",
                mmap_mode="r",
            )
            thresholds = _tbr_thresholds(
                calibration,
                k,
                include_mean=bool(config["calibration"]["include_mean"]),
                minimum_threshold=float(
                    config["calibration"]["minimum_threshold"]
                ),
            )
            _, metrics = _tbr_result(calibration, thresholds)
            subject_metrics.append(metrics)
        tbr_points.append(
            {
                "k": k,
                "median_snr_db": float(
                    np.median([metric["snr_db"] for metric in subject_metrics])
                ),
                "median_spike_density": float(
                    np.median([metric["spike_density"] for metric in subject_metrics])
                ),
                "mean_snr_db": float(
                    np.mean([metric["snr_db"] for metric in subject_metrics])
                ),
            }
        )

    minimum_threshold = float(config["step_forward"]["minimum_threshold"])
    for k_sf in sf_ks:
        adaptive_metrics = []
        global_metrics = []
        calibration_by_subject: dict[str, np.ndarray] = {}
        for row in rows:
            subject_id = str(row["subject_id"])
            calibration = np.load(
                calibration_dir / f"{subject_id}_calibration.npy",
                mmap_mode="r",
            )
            calibration_by_subject[subject_id] = calibration
            adaptive_thresholds = calibration_std_thresholds(
                calibration, k_sf, minimum_threshold
            )
            _, metrics = _sf_result(calibration, adaptive_thresholds)
            adaptive_metrics.append(metrics)
        global_threshold = pooled_calibration_std_threshold(
            calibration_by_subject, k_sf, minimum_threshold
        )
        for row in rows:
            calibration = calibration_by_subject[str(row["subject_id"])]
            _, metrics = _sf_result(
                calibration,
                np.full(calibration.shape[1], global_threshold),
            )
            global_metrics.append(metrics)
        sf_points.append(
            {
                "k_sf": k_sf,
                "adaptive_median_snr_db": float(
                    np.median([metric["snr_db"] for metric in adaptive_metrics])
                ),
                "adaptive_median_spike_density": float(
                    np.median(
                        [metric["spike_density"] for metric in adaptive_metrics]
                    )
                ),
                "global_median_snr_db": float(
                    np.median([metric["snr_db"] for metric in global_metrics])
                ),
                "global_median_spike_density": float(
                    np.median([metric["spike_density"] for metric in global_metrics])
                ),
            }
        )
    return tbr_points, sf_points


def _choose_parameters(
    tbr_curve: list[dict[str, Any]],
    sf_curve: list[dict[str, Any]],
    target_density: float,
) -> dict[str, float]:
    """Choose parameters from calibration curve only."""
    tbr_best = max(tbr_curve, key=lambda point: point["median_snr_db"])
    tbr_matched = min(
        tbr_curve,
        key=lambda point: abs(point["median_spike_density"] - target_density),
    )
    sf_best = max(
        sf_curve,
        key=lambda point: point["adaptive_median_snr_db"],
    )
    sf_matched = min(
        sf_curve,
        key=lambda point: abs(
            point["adaptive_median_spike_density"] - target_density
        ),
    )
    sf_global_best = max(sf_curve, key=lambda point: point["global_median_snr_db"])
    return {
        "tbr_best_snr_k": float(tbr_best["k"]),
        "tbr_matched_density_k": float(tbr_matched["k"]),
        "tbr_matched_calibration_density": float(tbr_matched["median_spike_density"]),
        "sf_best_snr_k": float(sf_best["k_sf"]),
        "sf_matched_density_k": float(sf_matched["k_sf"]),
        "sf_matched_calibration_density": float(
            sf_matched["adaptive_median_spike_density"]
        ),
        "sf_global_best_snr_k": float(sf_global_best["k_sf"]),
    }


def _density_target_table(
    tbr_curve: list[dict[str, Any]],
    sf_curve: list[dict[str, Any]],
    targets: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> list[dict[str, float | str]]:
    """Select the closest calibration-only density candidate for each target."""
    rows: list[dict[str, float | str]] = []
    for target in targets:
        tbr = min(
            tbr_curve,
            key=lambda point: abs(point["median_spike_density"] - target),
        )
        sf = min(
            sf_curve,
            key=lambda point: abs(point["adaptive_median_spike_density"] - target),
        )
        rows.extend(
            [
                {
                    "target_density": target,
                    "method": "TBR",
                    "k": float(tbr["k"]),
                    "calibration_median_density": float(
                        tbr["median_spike_density"]
                    ),
                    "calibration_median_snr_db": float(tbr["median_snr_db"]),
                },
                {
                    "target_density": target,
                    "method": "SF adaptive",
                    "k": float(sf["k_sf"]),
                    "calibration_median_density": float(
                        sf["adaptive_median_spike_density"]
                    ),
                    "calibration_median_snr_db": float(
                        sf["adaptive_median_snr_db"]
                    ),
                },
            ]
        )
    return rows


def _evaluation_diagnostics(
    subject_rows: list[dict[str, Any]],
    selected_ids: set[str],
    calibration_dir: Path,
    output_dir: Path,
    *,
    config: dict[str, Any],
    choices: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce per-subject evaluation metrics, drift summaries, and diagnostics."""
    tbr_ks = [float(value) for value in config["calibration"]["k_candidates"]]
    sf_ks = [float(value) for value in config["step_forward"]["k_candidates"]]
    selected_rows = [row for row in subject_rows if str(row["subject_id"]) in selected_ids]
    results: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    error_plot_payload: tuple[str, np.ndarray, np.ndarray] | None = None
    curve_eval: list[dict[str, Any]] = []
    pooled_calibrations = {
        str(other["subject_id"]): np.load(
            calibration_dir / f"{other['subject_id']}_calibration.npy",
            mmap_mode="r",
        )
        for other in subject_rows
    }
    sf_global_threshold = pooled_calibration_std_threshold(
        pooled_calibrations,
        choices["sf_global_best_snr_k"],
        float(config["step_forward"]["minimum_threshold"]),
    )
    tbr_global_threshold = _tbr_scalar_threshold_from_pool(
        subject_rows,
        calibration_dir,
        choices["tbr_best_snr_k"],
        bool(config["calibration"]["include_mean"]),
        float(config["calibration"]["minimum_threshold"]),
    )
    for row in selected_rows:
        subject_id = str(row["subject_id"])
        group = str(row["group"])
        calibration = np.load(
            calibration_dir / f"{subject_id}_calibration.npy",
            mmap_mode="r",
        )
        evaluation = np.load(
            calibration_dir / f"{subject_id}_evaluation.npy",
            mmap_mode="r",
        )
        tbr_k = choices["tbr_best_snr_k"]
        tbr_thresholds = _tbr_thresholds(
            calibration,
            tbr_k,
            include_mean=bool(config["calibration"]["include_mean"]),
            minimum_threshold=float(config["calibration"]["minimum_threshold"]),
        )
        tbr_recon, tbr_metrics = _tbr_result(evaluation, tbr_thresholds)
        results.append(
            {
                "subject_id": subject_id,
                "group": group,
                "method": "TBR adaptive per-channel (calibration-best-SNR k)",
                "parameter": tbr_k,
                **tbr_metrics,
            }
        )

        sf_k = choices["sf_best_snr_k"]
        sf_thresholds = calibration_std_thresholds(
            calibration,
            sf_k,
            float(config["step_forward"]["minimum_threshold"]),
        )
        _, sf_metrics = _sf_result(evaluation, sf_thresholds)
        results.append(
            {
                "subject_id": subject_id,
                "group": group,
                "method": "SF adaptive per-channel (calibration-best-SNR k)",
                "parameter": sf_k,
                **sf_metrics,
            }
        )

        _, sf_global_metrics = _sf_result(
            evaluation,
            np.full(evaluation.shape[1], sf_global_threshold),
        )
        results.append(
            {
                "subject_id": subject_id,
                "group": group,
                "method": "SF global fixed (calibration-best-SNR k)",
                "parameter": sf_global_threshold,
                **sf_global_metrics,
            }
        )

        tbr_scalar = _tbr_scalar_threshold(
            calibration,
            tbr_k,
            include_mean=bool(config["calibration"]["include_mean"]),
            minimum_threshold=float(config["calibration"]["minimum_threshold"]),
        )
        _, tbr_fixed_metrics = _tbr_result(
            evaluation, np.full(evaluation.shape[1], tbr_scalar)
        )
        results.append(
            {
                "subject_id": subject_id,
                "group": group,
                "method": "TBR subject-shared scalar threshold",
                "parameter": tbr_scalar,
                **tbr_fixed_metrics,
            }
        )

        _, tbr_global_metrics = _tbr_result(
            evaluation, np.full(evaluation.shape[1], tbr_global_threshold)
        )
        results.append(
            {
                "subject_id": subject_id,
                "group": group,
                "method": "TBR global fixed (calibration pooled)",
                "parameter": tbr_global_threshold,
                **tbr_global_metrics,
            }
        )

        region_n = int(config["diagnostics"]["samples_per_position_region"])
        if region_n * 3 > evaluation.shape[-1]:
            raise ValueError("Window is too short for configured position regions")
        segment_snr = {}
        for region, bounds in {
            "first_50": (0, region_n),
            "middle_50": (
                (evaluation.shape[-1] - region_n) // 2,
                (evaluation.shape[-1] + region_n) // 2,
            ),
            "last_50": (evaluation.shape[-1] - region_n, evaluation.shape[-1]),
        }.items():
            start, stop = bounds
            segment_snr[region] = _metrics(
                evaluation[..., start:stop],
                tbr_recon[..., start:stop],
            )["snr_db"]
        residual = np.asarray(evaluation, dtype=np.float64) - tbr_recon
        drift_rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                **{f"snr_{key}_db": value for key, value in segment_snr.items()},
                "error_std_sample_50": float(residual[:, :, 50].std()),
                "error_std_sample_250": float(residual[:, :, 250].std()),
                "error_std_sample_499": float(residual[:, :, 499].std()),
                "median_reconstructed_to_original_ptp": float(
                    np.median(
                        np.ptp(tbr_recon, axis=-1)
                        / np.maximum(np.ptp(evaluation, axis=-1), 1e-12)
                    )
                ),
                "fraction_abs_diff_over_threshold": float(
                    np.mean(
                        np.abs(np.diff(evaluation, axis=-1))
                        > tbr_thresholds[None, :, None]
                    )
                ),
            }
        )

        for k in tbr_ks:
            thresholds = _tbr_thresholds(
                calibration,
                k,
                include_mean=bool(config["calibration"]["include_mean"]),
                minimum_threshold=float(config["calibration"]["minimum_threshold"]),
            )
            _, metrics = _tbr_result(evaluation, thresholds)
            curve_eval.append(
                {
                    "method": "TBR",
                    "subject_id": subject_id,
                    "k": k,
                    "snr_db": metrics["snr_db"],
                    "spike_density": metrics["spike_density"],
                }
            )
        for k_sf in sf_ks:
            thresholds = calibration_std_thresholds(
                calibration,
                k_sf,
                float(config["step_forward"]["minimum_threshold"]),
            )
            _, metrics = _sf_result(evaluation, thresholds)
            curve_eval.append(
                {
                    "method": "SF",
                    "subject_id": subject_id,
                    "k": k_sf,
                    "snr_db": metrics["snr_db"],
                    "spike_density": metrics["spike_density"],
                }
            )

        matched_tbr_thresholds = _tbr_thresholds(
            calibration,
            choices["tbr_matched_density_k"],
            include_mean=bool(config["calibration"]["include_mean"]),
            minimum_threshold=float(config["calibration"]["minimum_threshold"]),
        )
        _, matched_tbr_metrics = _tbr_result(evaluation, matched_tbr_thresholds)
        matched_sf_thresholds = calibration_std_thresholds(
            calibration,
            choices["sf_matched_density_k"],
            float(config["step_forward"]["minimum_threshold"]),
        )
        _, matched_sf_metrics = _sf_result(evaluation, matched_sf_thresholds)
        for method, parameter, metrics in (
            (
                "TBR matched calibration density",
                choices["tbr_matched_density_k"],
                matched_tbr_metrics,
            ),
            (
                "SF matched calibration density",
                choices["sf_matched_density_k"],
                matched_sf_metrics,
            ),
        ):
            results.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "method": method,
                    "parameter": parameter,
                    **metrics,
                }
            )

        for interval in config["diagnostics"]["reanchoring_intervals"]:
            sf_signed, anchored = _reanchor_numpy(
                evaluation, sf_thresholds, int(interval)
            )
            anchored_metrics = _metrics(evaluation, anchored)
            count = int(np.ceil((evaluation.shape[-1] - 1) / int(interval)))
            channels = int(evaluation.shape[1])
            results.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "method": f"SF reanchored every {interval} samples",
                    "parameter": int(interval),
                    **anchored_metrics,
                    "spike_density": float(np.count_nonzero(sf_signed) / sf_signed.size),
                    "side_information_values_per_window": count * channels,
                    "side_information_bits_per_window": (
                        count
                        * channels
                        * int(config["diagnostics"]["side_information_bits_per_sample"])
                    ),
                }
            )
        if error_plot_payload is None:
            windows_to_plot = min(
                int(config["diagnostics"]["error_plot_window_count"]),
                evaluation.shape[0],
            )
            channel_index = int(config["diagnostics"]["error_plot_channel_index"])
            error_plot_payload = (
                subject_id,
                evaluation[:windows_to_plot, channel_index].copy(),
                tbr_recon[:windows_to_plot, channel_index].copy(),
            )

    _write_csv(output_dir / "evaluation_metrics_by_subject.csv", results)
    _write_csv(output_dir / "tbr_drift_by_subject.csv", drift_rows)
    _write_csv(output_dir / "evaluation_snr_density_curve.csv", curve_eval)
    _plot_drift_errors(error_plot_payload, output_dir, config)
    _plot_snr_density(curve_eval, output_dir)
    _plot_position_snr(drift_rows, output_dir)
    return results, drift_rows


def _tbr_scalar_threshold_from_pool(
    rows: list[dict[str, Any]],
    calibration_dir: Path,
    k: float,
    include_mean: bool,
    minimum_threshold: float,
) -> float:
    count = 0
    sum_values = 0.0
    sum_squares = 0.0
    for row in rows:
        windows = np.asarray(
            np.load(
                calibration_dir / f"{row['subject_id']}_calibration.npy",
                mmap_mode="r",
            ),
            dtype=np.float64,
        )
        differences = np.diff(windows, axis=-1).ravel()
        count += differences.size
        sum_values += float(differences.sum())
        sum_squares += float(np.square(differences).sum())
    mean = sum_values / count
    std = np.sqrt(max(sum_squares / count - mean * mean, 0.0))
    threshold = float(k) * float(std)
    if include_mean:
        threshold += mean
    return max(threshold, minimum_threshold)


def _reanchor_numpy(
    windows: np.ndarray, thresholds: np.ndarray, interval: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(windows, dtype=np.float64)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    if threshold_values.ndim == 1:
        threshold_view = threshold_values[None, :, None]
    elif threshold_values.ndim == 2:
        threshold_view = threshold_values[:, :, None]
    else:
        raise ValueError("thresholds must have shape [C] or [B, C]")
    signed = np.zeros(values.shape, dtype=np.int8)
    baseline = values[..., 0].copy()
    for time_index in range(1, values.shape[-1]):
        if time_index % interval == 0:
            baseline = values[..., time_index].copy()
            continue
        upward = values[..., time_index] > baseline + threshold_view.squeeze(-1)
        downward = values[..., time_index] < baseline - threshold_view.squeeze(-1)
        events = upward.astype(np.int8) - downward.astype(np.int8)
        signed[..., time_index] = events
        baseline += events * threshold_view.squeeze(-1)
    reconstructed = np.empty_like(values)
    for start in range(0, values.shape[-1], interval):
        stop = min(start + interval, values.shape[-1])
        reconstructed[..., start] = values[..., start]
        if stop > start + 1:
            reconstructed[..., start + 1 : stop] = (
                values[..., start, None]
                + np.cumsum(
                    signed[..., start + 1 : stop]
                    * threshold_view,
                    axis=-1,
                )
            )
    return signed, reconstructed


def _plot_drift_errors(
    payload: tuple[str, np.ndarray, np.ndarray] | None,
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    if payload is None:
        raise ValueError("No evaluation windows available for error plot")
    subject_id, original, reconstructed = payload
    fs = float(config["data"]["sampling_rate_hz"])
    times = np.arange(original.shape[-1]) / fs
    fig, axes = plt.subplots(original.shape[0], 1, figsize=(11, 7), sharex=True)
    axes = np.atleast_1d(axes)
    for index, axis in enumerate(axes):
        error = original[index] - reconstructed[index]
        axis.plot(times, error, linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.5)
        axis.set_ylabel(f"window {index + 1}\ne[t]")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"TBR reconstruction residuals — {subject_id}, E1")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject_id}_tbr_error_drift.png", dpi=180)
    plt.close(fig)


def _plot_snr_density(curve: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    for method in ("TBR", "SF"):
        points = [row for row in curve if row["method"] == method]
        grouped: dict[float, list[dict[str, Any]]] = {}
        for point in points:
            grouped.setdefault(float(point["k"]), []).append(point)
        densities = [
            float(np.median([row["spike_density"] for row in values]))
            for values in grouped.values()
        ]
        snrs = [
            float(np.median([row["snr_db"] for row in values]))
            for values in grouped.values()
        ]
        axis.plot(densities, snrs, marker="o", label=method)
        for k, density, snr in zip(grouped, densities, snrs, strict=True):
            axis.annotate(f"k={k:g}", (density, snr), fontsize=7)
    axis.set_xlabel("Median evaluation spike density (5 diagnostic subjects)")
    axis.set_ylabel("Median evaluation SNR (dB)")
    axis.set_title("Evaluation sparsity–fidelity curve (parameters fit on calibration)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "snr_vs_spike_density.png", dpi=180)
    plt.close(fig)


def _plot_position_snr(
    drift_rows: list[dict[str, Any]], output_dir: Path
) -> None:
    positions = ("first_50", "middle_50", "last_50")
    fig, axis = plt.subplots(figsize=(8, 5))
    x_values = np.arange(len(positions))
    for row in drift_rows:
        axis.plot(
            x_values,
            [row[f"snr_{position}_db"] for position in positions],
            marker="o",
            label=f"{row['subject_id']} ({row['group']})",
        )
    axis.set_xticks(x_values, ["First 50", "Middle 50", "Last 50"])
    axis.set_ylabel("TBR evaluation SNR (dB)")
    axis.set_title("TBR SNR by position within evaluation windows")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "tbr_snr_by_window_position.png", dpi=180)
    plt.close(fig)


def _pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for point in points:
        dominated = any(
            other["median_spike_density"] <= point["median_spike_density"]
            and other["median_snr_db"] >= point["median_snr_db"]
            and (
                other["median_spike_density"] < point["median_spike_density"]
                or other["median_snr_db"] > point["median_snr_db"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: point["median_spike_density"])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty results CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    selected_ids: set[str],
    tbr_curve: list[dict[str, Any]],
    sf_curve: list[dict[str, Any]],
    choices: dict[str, float],
    results: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
) -> None:
    tbr_pareto = _pareto_frontier(tbr_curve)
    sf_pareto = _pareto_frontier(
        [
            {
                "k": point["k_sf"],
                "median_snr_db": point["adaptive_median_snr_db"],
                "median_spike_density": point["adaptive_median_spike_density"],
            }
            for point in sf_curve
        ]
    )
    _write_csv(output_dir / "tbr_calibration_snr_density.csv", tbr_curve)
    _write_csv(output_dir / "sf_calibration_snr_density.csv", sf_curve)
    matched_rows = [
        result
        for result in results
        if result["method"] in {
            "TBR matched calibration density",
            "SF matched calibration density",
        }
    ]
    best_rows = [
        result
        for result in results
        if result["method"] in {
            "TBR adaptive per-channel (calibration-best-SNR k)",
            "SF adaptive per-channel (calibration-best-SNR k)",
            "TBR global fixed (calibration pooled)",
            "SF global fixed (calibration-best-SNR k)",
        }
    ]
    _write_csv(output_dir / "matched_density_and_best_snr.csv", matched_rows + best_rows)
    density_targets = _density_target_table(tbr_curve, sf_curve)
    _write_csv(output_dir / "calibration_density_targets.csv", density_targets)
    with (output_dir / "evaluation_snr_density_curve.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        evaluation_curve_rows = list(csv.DictReader(handle))
    evaluation_points: list[dict[str, Any]] = []
    for method in ("TBR", "SF"):
        method_rows = [
            row for row in evaluation_curve_rows if row["method"] == method
        ]
        for k in sorted({float(row["k"]) for row in method_rows}):
            points = [row for row in method_rows if float(row["k"]) == k]
            evaluation_points.append(
                {
                    "method": method,
                    "k": k,
                    "median_snr_db": float(
                        np.median([float(point["snr_db"]) for point in points])
                    ),
                    "median_spike_density": float(
                        np.median(
                            [float(point["spike_density"]) for point in points]
                        )
                    ),
                    "subjects_scored": len(points),
                }
            )
    evaluation_pareto = {
        method: _pareto_frontier(
            [
                {
                    key: point[key]
                    for key in (
                        "k",
                        "median_snr_db",
                        "median_spike_density",
                    )
                }
                for point in evaluation_points
                if point["method"] == method
            ]
        )
        for method in ("TBR", "SF")
    }
    _write_csv(
        output_dir / "evaluation_pareto_curve.csv",
        [
            {"method": method, **point}
            for method, points in evaluation_pareto.items()
            for point in points
        ],
    )
    summary = {
        "diagnostic_subject_ids": sorted(selected_ids),
        "diagnostic_groups": {
            "MDD": sum(
                str(row["subject_id"]) in selected_ids and row["group"] == "MDD"
                for row in rows
            ),
            "HC": sum(
                str(row["subject_id"]) in selected_ids and row["group"] == "HC"
                for row in rows
            ),
        },
        "parameter_selection": "calibration-only",
        "choices": choices,
        "calibration_density_targets": density_targets,
        "tbr_calibration_pareto_frontier": tbr_pareto,
        "sf_calibration_pareto_frontier": sf_pareto,
        "evaluation_tbr_pareto_frontier": evaluation_pareto["TBR"],
        "evaluation_sf_pareto_frontier": evaluation_pareto["SF"],
        "tbr_evaluation_drift": drift_rows,
        "reanchoring_results": [
            result for result in results if result["method"].startswith("SF reanchored")
        ],
        "snr_definition": "10*log10(sum(x**2)/sum((x-reconstruction)**2))",
        "spike_density_definition": "fraction of nonzero signed event samples",
        "unnormalized_variant": (
            "CAR + filtering + calibration-only artifact rule; no Z-score; "
            "native source unit unverified"
        ),
    }
    (output_dir / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _print_sine_sanity(config: dict[str, Any]) -> None:
    sine_config = config["diagnostics"]["sine"]
    fs = float(sine_config["sampling_rate_hz"])
    duration = float(sine_config["duration_sec"])
    time = np.arange(int(round(fs * duration)), dtype=np.float32) / fs
    signal = float(sine_config["amplitude"]) * np.sin(
        2.0 * np.pi * float(sine_config["frequency_hz"]) * time
    )
    windows = signal[None, None, :]
    signed = encode_signed(
        windows,
        np.asarray([float(sine_config["threshold"])], dtype=np.float64),
    )
    reconstructed = reconstruct_signed(
        signed,
        np.asarray([float(sine_config["threshold"])]),
        windows[:, :, 0],
    )
    metrics = _metrics(windows, reconstructed)
    print(
        "Synthetic sine TBR sanity: "
        f"f={sine_config['frequency_hz']} Hz, "
        f"threshold={sine_config['threshold']}, "
        f"SNR={metrics['snr_db']:.3f} dB, "
        f"RMSE={metrics['rmse']:.6f}, "
        f"density={np.count_nonzero(signed) / signed.size:.4f}"
    )


def run() -> None:
    config = load_config("c1_encoding")
    set_seed(int(config["project"]["seed"]))
    source_dir = INTERIM_DIR / "modma_preprocessed"
    manifest, rows = _load_manifest(source_dir / "manifest.json")
    selected_ids = _select_subjects(rows, config)
    all_subject_ids = {str(row["subject_id"]) for row in rows}
    output_dir = OUTPUT_ROOT / "c1" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_dir = INTERIM_DIR / str(
        config["diagnostics"]["unnormalized_variant_subdirectory"]
    )
    variant_manifest_path = variant_dir / "manifest.json"
    if variant_manifest_path.is_file():
        with variant_manifest_path.open(encoding="utf-8") as handle:
            variant_manifest = json.load(handle)
        if (
            variant_manifest.get("dataset") != manifest["dataset"]
            or variant_manifest.get("channels") != manifest["channels_kept"]
            or float(variant_manifest.get("window_sec", -1))
            != float(config["data"]["window_sec"])
            or float(variant_manifest.get("window_overlap", -1))
            != float(config["data"].get("window_overlap", 0.0))
            or float(variant_manifest.get("calibration_fraction", -1))
            != float(config["split"]["calibration_fraction"])
        ):
            raise ValueError(
                "Existing unnormalized variant does not match the requested "
                "dataset/configuration; inspect it before rebuilding: "
                f"{variant_manifest_path}"
            )
    variant_dir, variant_rows_by_id = _ensure_unnormalized_variant(
        rows, manifest=manifest, config=config, selected_ids=all_subject_ids
    )
    tbr_curve, sf_curve = _calibration_curves(
        rows, variant_dir, config=config
    )
    choices = _choose_parameters(
        tbr_curve,
        sf_curve,
        float(config["step_forward"]["matched_spike_density"]),
    )
    results, drift_rows = _evaluation_diagnostics(
        rows,
        selected_ids,
        variant_dir,
        output_dir,
        config=config,
        choices=choices,
    )
    _print_sine_sanity(config)
    _write_summary(
        output_dir,
        rows=rows,
        selected_ids=selected_ids,
        tbr_curve=tbr_curve,
        sf_curve=sf_curve,
        choices=choices,
        results=results,
        drift_rows=drift_rows,
    )
    (variant_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "variant": "No calibration Z-score; all preprocessing windows saved",
                "unit": "native MAT values (unverified)",
                "channels": list(manifest["channels_kept"]),
                "window_sec": config["data"]["window_sec"],
                "window_overlap": config["data"].get("window_overlap", 0.0),
                "calibration_fraction": config["split"]["calibration_fraction"],
                "subjects": list(variant_rows_by_id.values()),
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSelected diagnostic subjects: {sorted(selected_ids)}")
    print(f"Calibration-chosen parameter values: {choices}")
    print("\nCalibration Pareto points — TBR:")
    for point in _pareto_frontier(tbr_curve):
        print(
            f"  k={point['k']:g}: density={point['median_spike_density']:.4f}, "
            f"median SNR={point['median_snr_db']:.3f} dB"
        )
    print("\nEvaluation drift diagnostics:")
    for row in drift_rows:
        print(
            f"  {row['subject_id']} {row['group']}: SNR first/middle/last="
            f"{row['snr_first_50_db']:.2f}/{row['snr_middle_50_db']:.2f}/"
            f"{row['snr_last_50_db']:.2f} dB; error std @50/250/499="
            f"{row['error_std_sample_50']:.4g}/"
            f"{row['error_std_sample_250']:.4g}/"
            f"{row['error_std_sample_499']:.4g}"
        )
    print(f"\nReports saved under {output_dir}")
    print(f"Unnormalized local data variant saved under {variant_dir}")


if __name__ == "__main__":
    run()
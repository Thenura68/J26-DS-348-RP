"""Patient-adaptive TBR calibration: learn per-subject thresholds and the global k.

Only calibration windows (the chronologically first 20% of a subject's
recording) enter this module. For one subject and channel c:

    d          = within-window first differences of the calibration EEG
    theta_c    = k * std(d_c)  [+ mean(d_c) if include_mean]
    theta_c    = max(theta_c, minimum_threshold)

``std(d_c)`` adapts the threshold to the subject; ``k`` is one global
sensitivity multiplier chosen by ``sweep_calibration_k`` from calibration-only
reconstruction fidelity. Encoding, decoding and the metrics are imported from
``encoder``, ``decoder`` and ``fidelity`` so there is one implementation of each.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from snn_depression.c1_encoding.decoder import reconstruct
from snn_depression.c1_encoding.encoder import tbr_encode_signed
from snn_depression.c1_encoding.fidelity import snr_db, spike_density

SELECTION_METRICS = ("median_subject_snr_db", "sparsest_within_snr_tolerance")


@dataclass(frozen=True)
class ThresholdSettings:
    """Threshold-rule options shared by every subject (everything except k)."""

    include_mean: bool
    minimum_threshold: float
    minimum_signal_std: float
    minimum_derivative_std: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.minimum_threshold) or self.minimum_threshold <= 0:
            raise ValueError("minimum_threshold must be positive and finite")
        for name in ("minimum_signal_std", "minimum_derivative_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")

    @classmethod
    def from_config(cls, calibration_config: Mapping[str, Any]) -> ThresholdSettings:
        """Read the settings from the ``calibration`` block of c1_encoding.yaml."""
        return cls(
            include_mean=bool(calibration_config["include_mean"]),
            minimum_threshold=float(calibration_config["minimum_threshold"]),
            minimum_signal_std=float(calibration_config["minimum_signal_std"]),
            minimum_derivative_std=float(calibration_config["minimum_derivative_std"]),
        )


@dataclass(frozen=True)
class ThresholdProfile:
    """Per-subject threshold profile; thresholds are in the EEG data units.

    ``mean_derivative`` and ``derivative_std`` pool within-window differences
    only; no difference is taken across window edges. Arrays are [C].
    """

    subject_id: str
    channel_names: tuple[str, ...]
    thresholds: np.ndarray
    mean_derivative: np.ndarray
    derivative_std: np.ndarray
    k: float
    include_mean: bool
    calibration_seconds: float
    calibration_window_count: int
    unit: str
    flagged_channels: dict[str, list[str]]
    encoder: str = "tbr"  # which encoder applies these thresholds: "tbr" or "step_forward"

    _ARRAY_FIELDS = ("thresholds", "mean_derivative", "derivative_std")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible profile including all per-channel statistics."""
        payload = asdict(self)
        payload["channel_names"] = list(self.channel_names)
        for field_name in self._ARRAY_FIELDS:
            payload[field_name] = getattr(self, field_name).tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ThresholdProfile:
        """Rebuild a profile saved with ``to_dict`` (e.g. a profile JSON file).

        Keys that are not profile fields (e.g. from older profile versions) are ignored;
        fields missing from older files (``encoder``) take their default.
        """
        values = {
            field.name: payload.get(field.name, field.default) if field.default is not MISSING else payload[field.name]
            for field in fields(cls)
        }
        values["channel_names"] = tuple(values["channel_names"])
        for field_name in cls._ARRAY_FIELDS:
            values[field_name] = np.asarray(values[field_name], dtype=np.float64)
        values["flagged_channels"] = {
            reason: list(names) for reason, names in values["flagged_channels"].items()
        }
        return cls(**values)


def _validate_windows(
    windows: np.ndarray, channel_count: int | None = None
) -> np.ndarray:
    """Validate calibration windows [N, C, T] and return them as float64."""
    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected calibration windows [N, C, T], got {values.shape}")
    if values.shape[0] == 0 or values.shape[2] < 2:
        raise ValueError("Calibration requires at least one window with two samples")
    if channel_count is not None and values.shape[1] != channel_count:
        raise ValueError(
            f"Channel count mismatch: windows have {values.shape[1]} channels, "
            f"expected {channel_count}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Calibration windows contain NaN or Inf")
    return values


def _derivative_statistics(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-channel (signal std, mean diff, std diff) of windows [N, C, T]."""
    differences = np.diff(values, axis=-1)
    return (
        values.std(axis=(0, 2)),
        differences.mean(axis=(0, 2)),
        differences.std(axis=(0, 2)),
    )


def _raw_thresholds(
    mean_derivative: np.ndarray,
    derivative_std: np.ndarray,
    k: float,
    settings: ThresholdSettings,
) -> np.ndarray:
    """Threshold rule before clamping: k*std(d) [+ mean(d)]."""
    if not np.isfinite(k) or k <= 0:
        raise ValueError(f"k must be positive and finite, got {k}")
    raw = k * derivative_std
    return raw + mean_derivative if settings.include_mean else raw


def fit_thresholds(
    calibration_windows: np.ndarray, k: float, settings: ThresholdSettings
) -> np.ndarray:
    """Return per-channel thresholds [C] fitted on calibration windows [N, C, T]."""
    values = _validate_windows(calibration_windows)
    _, mean_derivative, derivative_std = _derivative_statistics(values)
    raw = _raw_thresholds(mean_derivative, derivative_std, k, settings)
    return np.maximum(raw, settings.minimum_threshold)


def compute_threshold_profile(
    calibration_windows: np.ndarray,
    channel_names: Sequence[str],
    *,
    subject_id: str,
    k: float,
    settings: ThresholdSettings,
    calibration_seconds: float,
    unit: str,
    encoder: str = "tbr",
) -> ThresholdProfile:
    """Fit a subject's per-channel adaptive thresholds from calibration windows only.

    The rule (theta_c = mean(d_c) + k * std(d_c) of the changes) is the same for every
    encoder; ``encoder`` records which one applies them ("tbr" or "step_forward").

    Input has shape [N_cal, C, T]. Pass the profile to
    ``AdaptiveThresholdEncoder.from_profile`` to encode that subject's EEG.
    """
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("Channel names must be unique")
    values = _validate_windows(calibration_windows, len(channel_names))
    if not np.isfinite(calibration_seconds) or calibration_seconds <= 0:
        raise ValueError("calibration_seconds must be positive and finite")
    if not unit:
        raise ValueError("The preprocessed data unit must be recorded")

    signal_std, mean_derivative, derivative_std = _derivative_statistics(values)
    raw = _raw_thresholds(mean_derivative, derivative_std, k, settings)
    flagged_indices = {
        "flat_or_near_zero_signal_variance": signal_std <= settings.minimum_signal_std,
        "near_zero_derivative_variance": derivative_std <= settings.minimum_derivative_std,
        "threshold_clamped": raw < settings.minimum_threshold,
    }
    return ThresholdProfile(
        subject_id=str(subject_id),
        channel_names=tuple(channel_names),
        thresholds=np.maximum(raw, settings.minimum_threshold),
        mean_derivative=mean_derivative,
        derivative_std=derivative_std,
        k=float(k),
        include_mean=settings.include_mean,
        calibration_seconds=float(calibration_seconds),
        calibration_window_count=int(values.shape[0]),
        unit=str(unit),
        flagged_channels={
            reason: [str(channel_names[i]) for i in np.flatnonzero(mask)]
            for reason, mask in flagged_indices.items()
        },
        encoder=str(encoder),
    )


def _select_k(
    curve: list[dict[str, Any]], selection_metric: str, snr_tolerance_db: float
) -> float:
    best_snr = max(row["median_subject_snr_db"] for row in curve)
    if selection_metric == "median_subject_snr_db":
        best = max(curve, key=lambda row: row["median_subject_snr_db"])
    else:
        eligible = [
            row for row in curve
            if row["median_subject_snr_db"] >= best_snr - snr_tolerance_db
        ]
        best = min(eligible, key=lambda row: (row["median_spike_density"], -row["k"]))
    return float(best["k"])


def sweep_calibration_k(
    calibration_windows_by_subject: Mapping[str, np.ndarray],
    *,
    k_candidates: Sequence[float],
    settings: ThresholdSettings,
    selection_metric: str,
    snr_tolerance_db: float = 0.0,
    decoder_leak: float = 1.0,
) -> tuple[float, list[dict[str, Any]]]:
    """Select one global k from calibration-only reconstruction fidelity.

    For every candidate k and subject: fit thresholds on that subject's
    calibration windows, encode them with TBR, decode with
    ``reconstruct(..., leak=decoder_leak)``, and score SNR and spike density.
    Evaluation arrays are deliberately not accepted. Under cross-validation,
    pass only the training-fold subjects.

    ``selection_metric``:
        - ``"median_subject_snr_db"``: the k with the highest median SNR.
        - ``"sparsest_within_snr_tolerance"``: among k whose median SNR is within
          ``snr_tolerance_db`` of the best, the one with the lowest median spike
          density (fewer spikes = less energy on the edge device).

    Returns (selected k, curve rows with k, median/mean subject SNR (dB),
    median spike density and the number of subjects scored).
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection_metric {selection_metric!r}; "
            f"expected one of {SELECTION_METRICS}"
        )
    if not np.isfinite(snr_tolerance_db) or snr_tolerance_db < 0:
        raise ValueError("snr_tolerance_db must be non-negative and finite")
    if not calibration_windows_by_subject:
        raise ValueError("At least one subject is required for the k sweep")
    candidates = [float(value) for value in k_candidates]
    if not candidates or any(not np.isfinite(v) or v <= 0 for v in candidates):
        raise ValueError("k_candidates must contain positive finite values")

    prepared = []
    for subject_id, windows in calibration_windows_by_subject.items():
        try:
            values = _validate_windows(windows)
        except ValueError as exc:
            raise ValueError(
                f"Invalid calibration windows for subject {subject_id}: {exc}"
            ) from exc
        _, mean_derivative, derivative_std = _derivative_statistics(values)
        prepared.append((torch.from_numpy(values), mean_derivative, derivative_std))

    curve: list[dict[str, Any]] = []
    for k in candidates:
        scores: list[float] = []
        densities: list[float] = []
        for eeg, mean_derivative, derivative_std in prepared:
            thresholds = torch.from_numpy(
                np.maximum(
                    _raw_thresholds(mean_derivative, derivative_std, k, settings),
                    settings.minimum_threshold,
                )
            )
            spikes = tbr_encode_signed(eeg, thresholds)
            decoded = reconstruct(
                spikes,
                thresholds,
                eeg[:, :, 0],
                polarity_mode="signed",
                leak=decoder_leak,
            )
            scores.append(snr_db(eeg, decoded))
            densities.append(spike_density(spikes))
        finite_scores = np.asarray([s for s in scores if np.isfinite(s)])
        if finite_scores.size == 0:
            raise ValueError(f"No finite calibration SNR values for k={k}")
        curve.append(
            {
                "k": k,
                "median_subject_snr_db": float(np.median(finite_scores)),
                "mean_subject_snr_db": float(np.mean(finite_scores)),
                "median_spike_density": float(np.median(densities)),
                "subjects_scored": int(finite_scores.size),
            }
        )
    return _select_k(curve, selection_metric, float(snr_tolerance_db)), curve


def check_calibration_windows_inside_split(
    calibration_source_indices: Sequence[int],
    *,
    split_sample: int,
    fs: float,
    window_sec: float,
    window_overlap: float,
) -> None:
    """Raise if any calibration window extends past the calibration/evaluation split.

    Calibration window ``i`` covers raw samples ``[i*step, i*step + W)`` with
    ``W = window_sec*fs`` and ``step = W*(1 - window_overlap)``; it must end at
    or before ``split_sample``. Evaluation windows are cut from the samples after
    the split, so a passing check means no raw sample is shared between the two
    partitions. This guards the split-before-windowing order of preprocessing.
    """
    indices = np.asarray(calibration_source_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0):
        raise ValueError("Calibration source indices must be a non-negative vector")
    if split_sample < 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("split_sample and fs are invalid")
    if not np.isfinite(window_sec) or window_sec <= 0:
        raise ValueError("window_sec must be positive and finite")
    if not np.isfinite(window_overlap) or not 0.0 <= window_overlap < 1.0:
        raise ValueError("window_overlap must be in [0, 1)")

    window_samples = int(round(window_sec * fs))
    step_samples = max(1, int(round(window_samples * (1.0 - window_overlap))))
    crossing = indices[indices * step_samples + window_samples > split_sample]
    if crossing.size:
        raise ValueError(
            f"Calibration windows {crossing.tolist()} extend past split sample "
            f"{split_sample}; calibration and evaluation would share samples"
        )


def channel_region_map(
    channel_names: Sequence[str],
    *,
    montage_name: str,
    anterior_y_min_m: float,
    posterior_y_max_m: float,
) -> dict[str, list[str]]:
    """Group montage channels coarsely as anterior, central, or posterior.

    Used for reporting thresholds by scalp region, not for encoding.
    """
    import mne

    if anterior_y_min_m <= posterior_y_max_m:
        raise ValueError("Anterior y threshold must exceed posterior y threshold")
    montage = mne.channels.make_standard_montage(montage_name)
    positions = montage.get_positions()["ch_pos"]
    missing = sorted(set(channel_names).difference(positions))
    if missing:
        raise ValueError(
            f"Montage {montage_name!r} is missing channels: {missing}"
        )
    regions: dict[str, list[str]] = {
        "anterior": [],
        "central": [],
        "posterior": [],
    }
    for name in channel_names:
        y_coordinate = float(positions[name][1])
        if y_coordinate >= anterior_y_min_m:
            regions["anterior"].append(name)
        elif y_coordinate <= posterior_y_max_m:
            regions["posterior"].append(name)
        else:
            regions["central"].append(name)
    return regions

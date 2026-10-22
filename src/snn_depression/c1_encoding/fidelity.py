"""Fidelity check: does TBR keep the EEG information used for depression detection?

Two levels of comparison between original EEG and its reconstruction from spikes:

1. Waveform: SNR, RMSE, Pearson correlation (``signal_integrity``). These pool
   all elements of the inputs (any matching shape, e.g. EEG [B, C, T]).
2. Depression-relevant spectral features (``spectral_fidelity``): per-channel
   band power (delta/theta/alpha/beta/gamma, Welch PSD), relative band power,
   and frontal alpha asymmetry. A reconstruction can have modest SNR yet keep
   these features, or vice versa, so both are reported.

Group-level checks (whether MDD-vs-HC differences survive) use ``cohens_d`` on
per-subject features; see ``scripts/c1_fidelity_report.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from scipy.integrate import trapezoid
from scipy.signal import welch
from scipy.stats import spearmanr
from torch import Tensor

Bands = Mapping[str, Sequence[float]]

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _pair(original: Tensor, reconstructed: Tensor) -> tuple[Tensor, Tensor]:
    source = torch.as_tensor(original, dtype=torch.float64)
    estimate = torch.as_tensor(reconstructed, dtype=torch.float64, device=source.device)
    if source.shape != estimate.shape or source.numel() == 0:
        raise ValueError("Metric inputs must have matching, non-empty shapes")
    if not torch.isfinite(source).all() or not torch.isfinite(estimate).all():
        raise ValueError("Metric inputs contain NaN or Inf")
    return source, estimate


def snr_db(original: Tensor, reconstructed: Tensor) -> float:
    """Return SNR = 10*log10(sum(s**2) / sum((s-r)**2)) in dB."""
    source, estimate = _pair(original, reconstructed)
    signal_power = float(source.square().sum())
    error_power = float((source - estimate).square().sum())
    if error_power == 0:
        return math.inf if signal_power > 0 else math.nan
    if signal_power == 0:
        return -math.inf
    return 10.0 * math.log10(signal_power / error_power)


def rmse(original: Tensor, reconstructed: Tensor) -> float:
    """Return the root-mean-square reconstruction error (data units)."""
    source, estimate = _pair(original, reconstructed)
    return float((source - estimate).square().mean().sqrt())


def correlation(original: Tensor, reconstructed: Tensor) -> float:
    """Return Pearson correlation of the flattened inputs (NaN if either is constant)."""
    source, estimate = _pair(original, reconstructed)
    source = source.flatten() - source.mean()
    estimate = estimate.flatten() - estimate.mean()
    denominator = float(source.norm() * estimate.norm())
    if denominator == 0:
        return math.nan
    return float((source * estimate).sum()) / denominator


def spike_density(signed_spikes: Tensor) -> float:
    """Fraction of (channel, time) slots that carry a spike in signed [B, C, T] events.

    For dual-binary [B, T, 2C] input use ``count_nonzero / (numel / 2)`` instead;
    this function counts one slot per EEG channel and sample.
    """
    spikes = torch.as_tensor(signed_spikes)
    if spikes.numel() == 0:
        raise ValueError("Spike train must be non-empty")
    return float(torch.count_nonzero(spikes)) / spikes.numel()


def signal_integrity(original: Tensor, reconstructed: Tensor) -> dict[str, float]:
    """Return {"correlation", "rmse", "snr_db"} for one session or batch."""
    return {
        "correlation": correlation(original, reconstructed),
        "rmse": rmse(original, reconstructed),
        "snr_db": snr_db(original, reconstructed),
    }


def _as_windows(eeg: Tensor | np.ndarray) -> np.ndarray:
    """Return EEG as float64 [N, C, T]; accepts [C, T] or [N, C, T]."""
    values = (
        eeg.detach().cpu().double().numpy()
        if isinstance(eeg, Tensor)
        else np.asarray(eeg, dtype=np.float64)
    )
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3 or values.shape[-1] < 2:
        raise ValueError(f"Expected EEG [N, C, T] or [C, T], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("EEG contains NaN or Inf")
    return values


def band_powers(
    eeg: Tensor | np.ndarray,
    fs: float,
    bands: Bands = DEFAULT_BANDS,
    *,
    segment_sec: float = 1.0,
) -> np.ndarray:
    """Absolute band power per channel, [C, n_bands], in data units squared.

    Welch PSD of every window (Hann segments of ``segment_sec``, 50% overlap),
    averaged over windows, then integrated over each band ``[low, high)``.
    Band order follows ``bands``.
    """
    values = _as_windows(eeg)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    nperseg = min(values.shape[-1], max(2, int(round(segment_sec * fs))))
    freqs, psd = welch(values, fs=fs, nperseg=nperseg, axis=-1)
    psd = psd.mean(axis=0)
    powers = []
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() < 2:
            raise ValueError(
                f"Band {name!r} [{low}, {high}) covers fewer than two PSD bins "
                f"(resolution {fs / nperseg:g} Hz); lengthen segment_sec"
            )
        powers.append(trapezoid(psd[:, mask], freqs[mask], axis=-1))
    return np.stack(powers, axis=-1)


def relative_band_powers(powers: np.ndarray) -> np.ndarray:
    """Normalise band powers [..., n_bands] so each row sums to 1."""
    total = powers.sum(axis=-1, keepdims=True)
    return powers / np.where(total > 0, total, 1.0)


def frontal_alpha_asymmetry(
    alpha_power: np.ndarray,
    channel_names: Sequence[str],
    left: str,
    right: str,
) -> float:
    """Return ln(alpha_right) - ln(alpha_left) for one electrode pair (e.g. F4 - F3)."""
    index = {name: i for i, name in enumerate(channel_names)}
    missing = [name for name in (left, right) if name not in index]
    if missing:
        raise ValueError(f"Asymmetry channels not found: {missing}")
    left_power, right_power = alpha_power[index[left]], alpha_power[index[right]]
    if left_power <= 0 or right_power <= 0:
        return math.nan
    return float(np.log(right_power) - np.log(left_power))


def spectral_fidelity(
    original: Tensor | np.ndarray,
    reconstructed: Tensor | np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    *,
    bands: Bands = DEFAULT_BANDS,
    asymmetry_pairs: Mapping[str, Sequence[str]] | None = None,
    segment_sec: float = 1.0,
) -> dict[str, Any]:
    """Compare depression-relevant spectral features of original vs reconstructed EEG.

    Inputs are EEG [N, C, T] (or [C, T]) from one session. Returns:
        band_power_bias_db:      {band: median over channels of 10*log10(P_rec / P_orig)}
        band_power_abs_error_db: {band: median over channels of |10*log10(P_rec / P_orig)|}
        relative_power_abs_error:{band: median over channels of |rel_rec - rel_orig|}
        alpha_asymmetry:         {pair: {"original": ..., "reconstructed": ...}}
                                 (only if ``asymmetry_pairs`` and an "alpha" band exist)
    """
    original_powers = band_powers(original, fs, bands, segment_sec=segment_sec)
    reconstructed_powers = band_powers(reconstructed, fs, bands, segment_sec=segment_sec)
    if original_powers.shape[0] != len(channel_names):
        raise ValueError("channel_names must match the EEG channel axis")
    tiny = np.finfo(np.float64).tiny
    ratio_db = 10.0 * np.log10(
        np.maximum(reconstructed_powers, tiny) / np.maximum(original_powers, tiny)
    )
    relative_error = np.abs(
        relative_band_powers(reconstructed_powers) - relative_band_powers(original_powers)
    )
    names = list(bands)
    report: dict[str, Any] = {
        "band_power_bias_db": {b: float(np.median(ratio_db[:, i])) for i, b in enumerate(names)},
        "band_power_abs_error_db": {
            b: float(np.median(np.abs(ratio_db[:, i]))) for i, b in enumerate(names)
        },
        "relative_power_abs_error": {
            b: float(np.median(relative_error[:, i])) for i, b in enumerate(names)
        },
    }
    if asymmetry_pairs and "alpha" in bands:
        alpha = names.index("alpha")
        report["alpha_asymmetry"] = {
            pair: {
                "original": frontal_alpha_asymmetry(
                    original_powers[:, alpha], channel_names, left, right
                ),
                "reconstructed": frontal_alpha_asymmetry(
                    reconstructed_powers[:, alpha], channel_names, left, right
                ),
            }
            for pair, (left, right) in asymmetry_pairs.items()
        }
    return report


def passes_thresholds(
    report: Mapping[str, Any],
    *,
    min_correlation: float | None,
    max_band_power_abs_error_db: float | None,
) -> bool | None:
    """Pass/fail for one session; None when no threshold is configured.

    ``report`` needs "correlation" and, if the band threshold is set,
    "band_power_abs_error_db" (every band must be within the limit).
    """
    checks = []
    if min_correlation is not None:
        checks.append(report["correlation"] >= min_correlation)
    if max_band_power_abs_error_db is not None:
        checks.append(
            all(
                error <= max_band_power_abs_error_db
                for error in report["band_power_abs_error_db"].values()
            )
        )
    return all(checks) if checks else None


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Standardised mean difference (mean_a - mean_b) / pooled SD; NaN if undefined."""
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return math.nan
    pooled = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (
        a.size + b.size - 2
    )
    return float((a.mean() - b.mean()) / math.sqrt(pooled)) if pooled > 0 else math.nan


def subject_spectral_features(
    powers: np.ndarray,
    channel_names: Sequence[str],
    band_names: Sequence[str],
    regions: Mapping[str, Sequence[str]],
    asymmetry_pairs: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Per-subject features from band powers [C, n_bands]: regional relative power and FAA.

    Returns ``{region}_{band}_rel`` (mean relative band power over the region's
    channels) and ``faa_{pair}`` (ln alpha_right - ln alpha_left) for each pair.
    """
    relative = relative_band_powers(powers)
    index = {name: i for i, name in enumerate(channel_names)}
    features: dict[str, float] = {}
    for region, members in regions.items():
        rows = [index[name] for name in members]
        for band_index, band in enumerate(band_names):
            features[f"{region}_{band}_rel"] = float(relative[rows, band_index].mean())
    alpha = list(band_names).index("alpha")
    for pair, (left, right) in asymmetry_pairs.items():
        features[f"faa_{pair}"] = frontal_alpha_asymmetry(powers[:, alpha], channel_names, left, right)
    return features


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation over finite pairs; NaN if fewer than 3 or constant."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return math.nan
    return float(spearmanr(x[keep], y[keep]).statistic)


def feature_preservation(
    subjects: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, float]],
    reconstructed: Sequence[Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Compare each per-subject feature before and after encoding, across subjects.

    ``subjects`` are manifest rows (``diagnosis_target``, ``phq9``). For each feature:
    cross-subject Spearman (original vs reconstructed), MDD-vs-HC Cohen's d for both,
    and Spearman with PHQ-9 for both.
    """
    is_mdd = np.array([s["diagnosis_target"] == 1 for s in subjects])
    phq9 = [float(s["phq9"]) if s["phq9"] is not None else math.nan for s in subjects]
    rows = []
    for name in original[0]:
        orig = np.array([f[name] for f in original])
        rec = np.array([f[name] for f in reconstructed])
        rows.append(
            {
                "feature": name,
                "spearman_original_vs_reconstructed": spearman(orig, rec),
                "cohens_d_mdd_vs_hc_original": cohens_d(orig[is_mdd], orig[~is_mdd]),
                "cohens_d_mdd_vs_hc_reconstructed": cohens_d(rec[is_mdd], rec[~is_mdd]),
                "spearman_phq9_original": spearman(orig, phq9),
                "spearman_phq9_reconstructed": spearman(rec, phq9),
            }
        )
    return rows

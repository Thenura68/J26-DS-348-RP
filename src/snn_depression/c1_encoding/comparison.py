"""Fair comparison of spike encoders: TBR (main), AER, Step-Forward and BSA.

Every method is wrapped in the same four steps, so the comparison treats them alike:

    fit(calibration windows [N, C, T], params)  -> per-subject state (first 20% only)
    encode(eeg [B, C, T], state)                -> contract spikes [B, T, 2C]
    decode(spikes, state, first samples [B, C]) -> reconstruction [B, C, T]
    ops_per_sample(params)                      -> rough arithmetic cost per EEG sample

| Method       | Per-subject calibration           | Global parameter(s) swept       |
|--------------|-----------------------------------|---------------------------------|
| step_forward_changes | theta_c = mean(d) + k * std(d) (main encoder) | k            |
| tbr          | theta_c = mean(d) + k * std(d)    | k                               |
| aer          | none (one fixed theta)            | theta                           |
| step_forward_fixed | none (one fixed theta for SF)  | theta                           |
| step_forward | theta_c = k_sf * std(x)           | k_sf                            |
| bsa          | scale_c = percentile(|x|)         | filter length, cutoff, threshold|

All tuning uses calibration windows only and the same selection rule as the TBR k
sweep. The TBR, AER and Step-Forward code is imported unchanged from ``encoder``,
``calibration``, ``decoder`` and ``step_forward``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.stats import wilcoxon
from torch import Tensor

from snn_depression.c1_encoding.bsa import (
    BSASettings,
    bsa_decode,
    bsa_encode,
    fit_bsa_profile,
)
from snn_depression.c1_encoding.calibration import ThresholdSettings, fit_thresholds
from snn_depression.c1_encoding.decoder import reconstruct
from snn_depression.c1_encoding.encoder import tbr_encode_signed, to_dual_binary
from snn_depression.c1_encoding.fidelity import snr_db
from snn_depression.c1_encoding.step_forward import calibration_std_thresholds, step_forward

SELECTION_METRICS = ("median_subject_snr_db", "sparsest_within_snr_tolerance")


@dataclass(frozen=True)
class EncoderMethod:
    """One encoding method behind the common fit/encode/decode interface."""

    name: str
    fit: Callable[[np.ndarray, Mapping[str, Any]], Any]
    encode: Callable[[Tensor, Any], Tensor]
    decode: Callable[[Tensor, Any, Tensor], Tensor]
    ops_per_sample: Callable[[Mapping[str, Any]], float]


def spike_density(spikes: Tensor) -> float:
    """Spikes per EEG channel-sample for contract spikes [B, T, 2C]: count / (B * T * C)."""
    return float(torch.count_nonzero(spikes)) / (spikes.numel() / 2)


def build_methods(
    *,
    fs: float,
    tbr_settings: ThresholdSettings,
    decoder_leak: float,
    step_forward_minimum_threshold: float,
    bsa_scale_percentile: float,
    bsa_minimum_scale: float,
) -> dict[str, EncoderMethod]:
    """Create the four methods with their fixed (non-swept) settings."""

    def tbr_fit(windows, params):
        return torch.from_numpy(fit_thresholds(windows, float(params["k"]), tbr_settings))

    def threshold_encode(eeg, thresholds):
        return to_dual_binary(tbr_encode_signed(eeg, thresholds.to(eeg.dtype)))

    def leaky_decode(spikes, thresholds, initial):
        return reconstruct(spikes, thresholds, initial, leak=decoder_leak)

    def aer_fit(windows, params):
        return torch.full((np.shape(windows)[1],), float(params["theta"]), dtype=torch.float64)

    def sf_fit(windows, params):
        return torch.from_numpy(
            calibration_std_thresholds(windows, float(params["k_sf"]), step_forward_minimum_threshold)
        )

    def sf_encode(eeg, thresholds):
        return step_forward(eeg, thresholds.to(eeg.dtype), polarity_mode="dual_binary")

    def sf_decode(spikes, thresholds, initial):
        # SF events move a tracked baseline by +/-theta, so plain integration recovers it.
        return reconstruct(spikes, thresholds, initial, leak=1.0)

    def sf_changes_fit(windows, params):
        # The main encoder: TBR's patient-adaptive rule (mean + k * std of the changes).
        return torch.from_numpy(fit_thresholds(windows, float(params["k"]), tbr_settings))

    def bsa_fit(windows, params):
        settings = BSASettings(
            filter_length=int(params["filter_length"]),
            cutoff_hz=float(params["cutoff_hz"]),
            threshold=float(params["threshold"]),
            scale_percentile=bsa_scale_percentile,
            minimum_scale=bsa_minimum_scale,
        )
        names = [f"ch{i}" for i in range(np.shape(windows)[1])]
        return fit_bsa_profile(windows, names, subject_id="subject", settings=settings, fs=fs, unit="data")

    return {
        "tbr": EncoderMethod("tbr", tbr_fit, threshold_encode, leaky_decode, lambda p: 3.0),
        "aer": EncoderMethod("aer", aer_fit, threshold_encode, leaky_decode, lambda p: 3.0),
        "step_forward_changes": EncoderMethod("step_forward_changes", sf_changes_fit, sf_encode, sf_decode, lambda p: 4.0),
        "step_forward": EncoderMethod("step_forward", sf_fit, sf_encode, sf_decode, lambda p: 4.0),
        "step_forward_fixed": EncoderMethod("step_forward_fixed", aer_fit, sf_encode, sf_decode, lambda p: 4.0),
        "bsa": EncoderMethod(
            "bsa",
            bsa_fit,
            lambda eeg, profile: bsa_encode(eeg, profile),
            lambda spikes, profile, initial: bsa_decode(spikes, profile),
            # per EEG sample, two parts (POS/NEG), each ~4 operations per filter tap
            lambda p: 2.0 * 4.0 * float(p["filter_length"]),
        ),
    }


def parameter_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a {name: values} grid, in a stable order."""
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*(grid[n] for n in names))]


def evaluate_on_windows(
    method: EncoderMethod, windows: np.ndarray, state: Any
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode and decode windows [N, C, T]; return (eeg float32, spikes, reconstruction)."""
    eeg = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    spikes = method.encode(eeg, state)
    rebuilt = method.decode(spikes, state, eeg[:, :, 0].to(torch.float64))
    return eeg, spikes, rebuilt


def sweep_method(
    method: EncoderMethod,
    grid: Mapping[str, Sequence[Any]],
    calibration_windows_by_subject: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Score every parameter setting on calibration windows only (median subject SNR, density)."""
    curve = []
    for params in parameter_grid(grid):
        scores, densities = [], []
        for windows in calibration_windows_by_subject.values():
            eeg, spikes, rebuilt = evaluate_on_windows(method, windows, method.fit(windows, params))
            scores.append(snr_db(eeg, rebuilt))
            densities.append(spike_density(spikes))
        finite = [s for s in scores if math.isfinite(s)]
        if not finite:
            raise ValueError(f"No finite calibration SNR for {method.name} {params}")
        curve.append(
            {
                "params": params,
                "median_subject_snr_db": float(np.median(finite)),
                "mean_subject_snr_db": float(np.mean(finite)),
                "median_spike_density": float(np.median(densities)),
                "subjects_scored": len(finite),
            }
        )
    return curve


def select_setting(curve: Sequence[Mapping[str, Any]], selection_metric: str, snr_tolerance_db: float) -> dict:
    """Same rule as the TBR k sweep: best SNR, or fewest spikes within a tolerance of it."""
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"Unsupported selection_metric {selection_metric!r}")
    best_snr = max(row["median_subject_snr_db"] for row in curve)
    if selection_metric == "median_subject_snr_db":
        return dict(max(curve, key=lambda row: row["median_subject_snr_db"]))
    eligible = [r for r in curve if r["median_subject_snr_db"] >= best_snr - snr_tolerance_db]
    return dict(min(eligible, key=lambda row: row["median_spike_density"]))


def matched_setting(curve: Sequence[Mapping[str, Any]], target_density: float) -> dict:
    """Setting whose calibration density is closest to ``target_density`` (ties: higher SNR)."""
    return dict(
        min(
            curve,
            key=lambda row: (abs(row["median_spike_density"] - target_density), -row["median_subject_snr_db"]),
        )
    )


def paired_comparison(
    reference: Mapping[str, float], other: Mapping[str, float]
) -> dict[str, float]:
    """Wilcoxon signed-rank test over subjects present in both, plus the median difference."""
    subjects = [s for s in reference if s in other and math.isfinite(reference[s]) and math.isfinite(other[s])]
    a = np.array([reference[s] for s in subjects])
    b = np.array([other[s] for s in subjects])
    if len(subjects) < 6 or np.allclose(a, b):
        return {"n": len(subjects), "median_difference": float(np.median(a - b)) if subjects else math.nan, "p": math.nan}
    return {"n": len(subjects), "median_difference": float(np.median(a - b)), "p": float(wilcoxon(a, b).pvalue)}


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are left as NaN)."""
    indexed = [(p, i) for i, p in enumerate(p_values) if math.isfinite(p)]
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, (len(indexed) - rank) * p))
        adjusted[i] = running
    return adjusted


def choose_encoder(
    candidates: Sequence[Mapping[str, Any]], *, feature_rho_tie_margin: float
) -> dict[str, Any]:
    """Pre-registered decision rule over one row per method (at the matched density).

    1. highest median feature Spearman (depression-relevant information);
    2. among methods within ``feature_rho_tie_margin`` of the best: lowest ops per sample;
    3. then lowest between-subject spike-density CV;
    4. then highest median SNR.
    Returns the chosen row plus the tied set, so the reason is recorded.
    """
    best_rho = max(row["median_feature_spearman"] for row in candidates)
    tied = [r for r in candidates if r["median_feature_spearman"] >= best_rho - feature_rho_tie_margin]
    chosen = min(tied, key=lambda r: (r["ops_per_sample"], r["density_cv"], -r["median_snr_db"]))
    return {
        "chosen_method": chosen["method"],
        "best_median_feature_spearman": best_rho,
        "tied_within_margin": [r["method"] for r in tied],
        "reason": (
            "highest feature preservation"
            if len(tied) == 1
            else "tied on feature preservation; lowest computational cost among tied methods"
        ),
    }

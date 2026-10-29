"""C1's main encoder in one place.

The main encoder is set in configs/c1_encoding.yaml → ``encoder.method``:

- ``step_forward`` (main, chosen by the encoder comparison): patient-adaptive
  Step-Forward. Thresholds use the patient-adaptive rule
  theta_c = mean(d_c) + k * std(d_c) of the changes in the patient's first 20%
  (``calibration.compute_threshold_profile``); Step-Forward applies them against a
  tracked baseline, so small changes accumulate instead of being dropped.
- ``tbr`` (previous main, kept as a baseline): the same thresholds applied to each
  change by Threshold-Based Representation.

Both give the same contract output [B, T, 2C] (UP then DOWN channels). Every C1 script
and the dashboard go through this module, so switching encoder is a config change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from torch import Tensor

from snn_depression.c1_encoding.calibration import ThresholdProfile, ThresholdSettings, compute_threshold_profile
from snn_depression.c1_encoding.decoder import leak_from_cutoff, reconstruct
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder, to_dual_binary, tbr_encode_signed
from snn_depression.c1_encoding.step_forward import AdaptiveStepForwardEncoder, step_forward

MAIN_ENCODERS = ("step_forward", "tbr")
DISPLAY_NAMES = {"step_forward": "Adaptive Step-Forward", "tbr": "Adaptive TBR"}


def main_method(config: Mapping[str, Any]) -> str:
    """The configured main encoder ("step_forward" or "tbr")."""
    method = str(config["encoder"]["method"])
    if method not in MAIN_ENCODERS:
        raise ValueError(f"encoder.method must be one of {MAIN_ENCODERS}, got {method!r}")
    return method


def main_k(config: Mapping[str, Any]) -> float:
    """The threshold multiplier k used by the main encoder (selected by scripts/c1_calibrate.py)."""
    return float(config["encoder"]["k"])


def k_candidates(config: Mapping[str, Any], method: str) -> list[float]:
    """k values the calibration sweep tries for ``method``."""
    if method == "step_forward":
        return [float(k) for k in config["encoder"]["k_candidates"]]
    return [float(k) for k in config["calibration"]["k_candidates"]]


def fit_profile(
    calibration_windows: np.ndarray,
    channel_names: Sequence[str],
    *,
    subject_id: str,
    calibration_seconds: float,
    unit: str,
    config: Mapping[str, Any],
    k: float | None = None,
    method: str | None = None,
) -> ThresholdProfile:
    """Fit a patient's profile for the main encoder from calibration windows [N_cal, C, T] only."""
    method = method or main_method(config)
    return compute_threshold_profile(
        calibration_windows,
        channel_names,
        subject_id=subject_id,
        k=main_k(config) if k is None else float(k),
        settings=ThresholdSettings.from_config(config["calibration"]),
        calibration_seconds=calibration_seconds,
        unit=unit,
        encoder=method,
    )


def make_encoder(profile: ThresholdProfile) -> AdaptiveStepForwardEncoder | AdaptiveThresholdEncoder:
    """The encoder object a profile was fitted for (its ``encoder`` field)."""
    if profile.encoder == "step_forward":
        return AdaptiveStepForwardEncoder.from_profile(profile)
    if profile.encoder == "tbr":
        return AdaptiveThresholdEncoder.from_profile(profile)
    raise ValueError(f"Unknown encoder {profile.encoder!r} in the profile of {profile.subject_id}")


def encode(eeg: Tensor, thresholds: Tensor, method: str) -> Tensor:
    """Encode EEG [B, C, T] with per-channel thresholds [C]; returns contract spikes [B, T, 2C]."""
    if method == "step_forward":
        return step_forward(eeg, thresholds.to(eeg.dtype))
    if method == "tbr":
        return to_dual_binary(tbr_encode_signed(eeg, thresholds.to(eeg.dtype)))
    raise ValueError(f"Unknown encoder {method!r}")


def decode(
    spikes: Tensor, thresholds: Tensor, initial: Tensor, method: str, config: Mapping[str, Any], fs: float
) -> Tensor:
    """Rebuild EEG [B, C, T] from contract spikes, for the fidelity check.

    Step-Forward: the baseline moves by +/-theta at each spike, so integrating the spikes
    from the first sample recovers it exactly; no leak is needed. TBR: integration with
    the configured leak (decoder.leak_cutoff_hz) to remove drift.
    """
    if method == "step_forward":
        return reconstruct(spikes, thresholds, initial, leak=1.0)
    if method == "tbr":
        return reconstruct(spikes, thresholds, initial, leak=leak_from_cutoff(config["decoder"]["leak_cutoff_hz"], fs))
    raise ValueError(f"Unknown encoder {method!r}")

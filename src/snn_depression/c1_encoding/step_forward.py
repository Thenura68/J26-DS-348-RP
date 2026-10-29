"""Step-Forward (SF) temporal-contrast encoding: C1's main encoder.

SF keeps a baseline per channel and emits UP/DOWN when the signal moves more than theta
away from it, moving the baseline by theta each time. Small changes accumulate against
the baseline instead of being dropped, so reconstruction does not drift.

The main encoder is ``AdaptiveStepForwardEncoder``: its thresholds come from the
patient-adaptive rule in ``calibration.compute_threshold_profile`` (theta_c =
mean(d_c) + k * std(d_c) of the changes in the patient's first 20%), the same rule TBR
uses. ``calibration_std_thresholds`` (k * std of the signal) is kept for the earlier
encoder comparison and diagnostics.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from snn_depression.common.contracts import check_eeg, check_spike_train
from snn_depression.c1_encoding.encoder import PolarityMode, threshold_view, to_dual_binary, validate_polarity_mode

if TYPE_CHECKING:
    from snn_depression.c1_encoding.calibration import ThresholdProfile


def step_forward_signed(eeg: Tensor, thresholds: Tensor) -> Tensor:
    """Encode EEG [B, C, T] into strict one-step {-1, 0, +1} SF events.

    Each channel's baseline starts at its first sample. At each later sample,
    emit +1 if the value exceeds baseline plus threshold, or -1 if it falls
    below baseline minus threshold. Advance the baseline by one threshold per
    emitted event. A large excursion can therefore require later time steps to
    catch up; at most one spike per channel/time step is emitted.
    """
    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG [B, C, T], got {tuple(eeg.shape)}")
    if eeg.shape[0] == 0 or eeg.shape[1] == 0 or eeg.shape[2] == 0:
        raise ValueError("Step-Forward requires non-empty batch, channel, and time axes")
    if not torch.isfinite(eeg).all():
        raise ValueError("EEG contains NaN or Inf")
    threshold_values = torch.as_tensor(
        thresholds, dtype=eeg.dtype, device=eeg.device
    )
    if threshold_values.ndim == 1:
        if threshold_values.shape[0] != eeg.shape[1]:
            raise ValueError(
                f"Expected thresholds [{eeg.shape[1]}], "
                f"got {tuple(threshold_values.shape)}"
            )
        threshold_values = threshold_values[None, :]
    elif threshold_values.shape != eeg.shape[:2]:
        raise ValueError(
            f"Thresholds must be [C] or [B, C], got {tuple(threshold_values.shape)}"
        )
    if not torch.isfinite(threshold_values).all() or (threshold_values <= 0).any():
        raise ValueError("Thresholds must be positive and finite")

    spikes = torch.zeros_like(eeg, dtype=torch.int8)
    baseline = eeg[..., 0].clone()
    for time_index in range(1, eeg.shape[-1]):
        current = eeg[..., time_index]
        upward = current > baseline + threshold_values
        downward = current < baseline - threshold_values
        events = upward.to(torch.int8) - downward.to(torch.int8)
        spikes[..., time_index] = events
        baseline = baseline + events.to(eeg.dtype) * threshold_values
    return spikes


def step_forward(
    eeg: Tensor,
    thresholds: Tensor,
    *,
    polarity_mode: str = "dual_binary",
) -> Tensor:
    """Return SF spikes from EEG [B, C, T] in signed or dual-binary format.

    Dual-binary output is contract-compatible float32 [B, T, 2C]. Signed
    output is internal {-1, 0, +1} [B, C, T] and does not satisfy the shared
    spike-train contract.
    """
    if polarity_mode not in {"dual_binary", "signed"}:
        raise ValueError(
            f"polarity_mode must be 'dual_binary' or 'signed', got {polarity_mode!r}"
        )
    signed = step_forward_signed(eeg, thresholds)
    if polarity_mode == "signed":
        return signed
    binary = to_dual_binary(signed)
    check_spike_train(binary, n_channels=2 * eeg.shape[1])
    return binary


def calibration_std_thresholds(
    calibration_windows: np.ndarray,
    k_sf: float,
    minimum_threshold: float,
) -> np.ndarray:
    """Fit SF thresholds k_sf*std(x) per EEG channel from [N, C, T] only."""
    values = np.asarray(calibration_windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[-1] == 0:
        raise ValueError(
            f"Expected non-empty calibration windows [N, C, T], got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Calibration windows contain NaN or Inf")
    if not np.isfinite(k_sf) or k_sf <= 0:
        raise ValueError("k_sf must be positive and finite")
    if not np.isfinite(minimum_threshold) or minimum_threshold <= 0:
        raise ValueError("minimum_threshold must be positive and finite")
    channel_std = values.std(axis=(0, 2))
    return np.maximum(float(k_sf) * channel_std, float(minimum_threshold))


def pooled_calibration_std_threshold(
    calibration_windows_by_subject: dict[str, np.ndarray],
    k_sf: float,
    minimum_threshold: float,
) -> float:
    """Fit one fixed SF threshold from pooled calibration windows only."""
    if not calibration_windows_by_subject:
        raise ValueError("At least one subject is required")
    if not np.isfinite(k_sf) or k_sf <= 0:
        raise ValueError("k_sf must be positive and finite")
    if not np.isfinite(minimum_threshold) or minimum_threshold <= 0:
        raise ValueError("minimum_threshold must be positive and finite")
    total_count = 0
    total_sum = 0.0
    total_square_sum = 0.0
    for subject_id, windows in calibration_windows_by_subject.items():
        values = np.asarray(windows, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError(
                f"Invalid calibration windows for {subject_id}: {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Calibration windows for {subject_id} contain NaN or Inf"
            )
        total_count += int(values.size)
        total_sum += float(values.sum())
        total_square_sum += float(np.square(values).sum())
    mean = total_sum / total_count
    variance = max(total_square_sum / total_count - mean * mean, 0.0)
    return max(float(k_sf) * float(np.sqrt(variance)), float(minimum_threshold))


def encode_reanchored(
    eeg: Tensor, thresholds: Tensor, anchor_interval: int
) -> tuple[Tensor, Tensor]:
    """Return SF events and reconstruction with true-value anchors every N samples.

    The anchor values are extra side information at times ``N, 2N, ...``.
    Returns signed events [B, C, T] and re-anchored decoded EEG [B, C, T].
    """
    if anchor_interval < 1:
        raise ValueError("anchor_interval must be positive")
    threshold_values = torch.as_tensor(
        thresholds, dtype=eeg.dtype, device=eeg.device
    )
    if threshold_values.ndim == 1:
        threshold_values = threshold_values[None, :]
    if threshold_values.shape not in (
        (1, eeg.shape[1]),
        (eeg.shape[0], eeg.shape[1]),
    ):
        raise ValueError("Thresholds must be [C] or [B, C]")
    if not torch.isfinite(threshold_values).all() or (threshold_values <= 0).any():
        raise ValueError("Thresholds must be positive and finite")
    signed = torch.zeros_like(eeg, dtype=torch.int8)
    baseline = eeg[..., 0].clone()
    for time_index in range(1, eeg.shape[-1]):
        if time_index % anchor_interval == 0:
            baseline = eeg[..., time_index].clone()
            continue
        current = eeg[..., time_index]
        upward = current > baseline + threshold_values
        downward = current < baseline - threshold_values
        events = upward.to(torch.int8) - downward.to(torch.int8)
        signed[..., time_index] = events
        baseline = baseline + events.to(eeg.dtype) * threshold_values
    decoded = torch.empty_like(eeg)
    for start in range(0, eeg.shape[-1], anchor_interval):
        stop = min(start + anchor_interval, eeg.shape[-1])
        decoded[..., start] = eeg[..., start]
        if stop > start + 1:
            increments = (
                signed[..., start + 1 : stop].to(eeg.dtype)
                * threshold_values[..., None]
            )
            decoded[..., start + 1 : stop] = (
                eeg[..., start, None]
                + torch.cumsum(increments, dim=-1)
            )
    return signed, decoded

class AdaptiveStepForwardEncoder:
    """Patient-adaptive Step-Forward encoder: one threshold per channel for one subject.

    Build it with ``from_profile`` from a profile fitted on the subject's calibration
    windows only (``calibration.compute_threshold_profile``). ``encode`` accepts EEG
    [B, C, T] float32 and returns dual-binary contract spikes [B, T, 2C] float32, or
    internal signed int8 spikes [B, C, T] with ``polarity_mode="signed"``.
    """

    def __init__(
        self,
        thresholds: np.ndarray | Tensor,
        *,
        channel_names: Sequence[str],
        subject_id: str,
        polarity_mode: PolarityMode = "dual_binary",
    ) -> None:
        self.channel_names = tuple(channel_names)
        self.subject_id = str(subject_id)
        self.polarity_mode = validate_polarity_mode(polarity_mode)
        values = torch.as_tensor(thresholds, dtype=torch.float32)
        if values.ndim != 1 or values.shape[0] != len(self.channel_names):
            raise ValueError(
                f"Expected one threshold per channel [{len(self.channel_names)}], got {tuple(values.shape)}"
            )
        threshold_view(values, 1, len(self.channel_names))
        self._thresholds = values

    @classmethod
    def from_profile(
        cls, profile: ThresholdProfile, *, polarity_mode: PolarityMode = "dual_binary"
    ) -> AdaptiveStepForwardEncoder:
        """Create the encoder for the subject described by a fitted profile."""
        return cls(profile.thresholds, channel_names=profile.channel_names,
                   subject_id=profile.subject_id, polarity_mode=polarity_mode)

    @property
    def thresholds(self) -> Tensor:
        """Per-channel threshold vector [C] (float32, EEG data units)."""
        return self._thresholds

    def encode(self, eeg: Tensor) -> Tensor:
        """Encode EEG [B, C, T] as dual-binary [B, T, 2C] or signed [B, C, T]."""
        check_eeg(eeg, n_channels=len(self.channel_names))
        return step_forward(eeg, self._thresholds.to(eeg.device), polarity_mode=self.polarity_mode)

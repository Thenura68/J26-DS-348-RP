"""Signed threshold-based representation (TBR) EEG spike encoders.

TBR emits a spike when the first difference ``x[t] - x[t-1]`` of a channel
crosses that channel's threshold: ``+1`` (UP) above ``+theta``, ``-1`` (DOWN)
below ``-theta``, otherwise ``0``. The first sample of every window is silent
because the decoder receives it as side information.

Thresholds are made patient-adaptive by ``calibration.compute_threshold_profile``,
which fits one threshold per channel on a subject's calibration windows only.
This module only *applies* thresholds; it never sees evaluation statistics.

``tbr_encode_signed`` is the single TBR implementation in the package; the
encoders, the calibration k sweep and the diagnostics all call it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor

from snn_depression.common.contracts import check_eeg, check_spike_train

if TYPE_CHECKING:
    from snn_depression.c1_encoding.calibration import ThresholdProfile

PolarityMode = Literal["dual_binary", "signed"]


def validate_polarity_mode(mode: str) -> PolarityMode:
    """Return ``mode`` if it is 'dual_binary' or 'signed', else raise."""
    if mode not in ("dual_binary", "signed"):
        raise ValueError(
            f"polarity_mode must be 'dual_binary' or 'signed', got {mode!r}"
        )
    return mode  # type: ignore[return-value]


def validate_signed(spikes: Tensor) -> None:
    """Check internal signed spikes: shape [B, C, T], values in {-1, 0, +1}."""
    if spikes.ndim != 3:
        raise ValueError(
            f"Expected signed spikes [B, C, T], got {tuple(spikes.shape)}"
        )
    if not torch.isfinite(spikes).all():
        raise ValueError("Signed spike train contains NaN or Inf")
    if not ((spikes == -1) | (spikes == 0) | (spikes == 1)).all():
        raise ValueError("Signed spikes must contain only -1, 0, and +1")


def to_dual_binary(signed_spikes: Tensor) -> Tensor:
    """Split signed spikes [B, C, T] into contract output [B, T, 2C].

    Channel order is all UP channels followed by all DOWN channels. The
    contract boundary representation is float32; uint8 is reserved for storage.
    """
    validate_signed(signed_spikes)
    up = (signed_spikes > 0).permute(0, 2, 1)
    down = (signed_spikes < 0).permute(0, 2, 1)
    binary = torch.cat((up, down), dim=-1).to(dtype=torch.float32)
    check_spike_train(binary, n_channels=2 * signed_spikes.shape[1])
    return binary


def from_dual_binary(binary_spikes: Tensor) -> Tensor:
    """Recombine contract spikes [B, T, 2C] into signed float32 spikes [B, C, T]."""
    check_spike_train(binary_spikes)
    if binary_spikes.shape[2] == 0 or binary_spikes.shape[2] % 2 != 0:
        raise ValueError(
            "Dual-binary spike channel count must be positive and even"
        )
    channel_count = binary_spikes.shape[2] // 2
    up = binary_spikes[..., :channel_count]
    down = binary_spikes[..., channel_count:]
    if ((up != 0) & (down != 0)).any():
        raise ValueError("A channel cannot emit UP and DOWN spikes at one time")
    return (up - down).permute(0, 2, 1)


def threshold_view(thresholds: Tensor, batch: int, channels: int) -> Tensor:
    """Validate thresholds [C] or [B, C] and return them broadcastable as [B|1, C, 1]."""
    if thresholds.shape not in ((channels,), (batch, channels)):
        raise ValueError(
            f"Thresholds must be [{channels}] or [{batch}, {channels}], "
            f"got {tuple(thresholds.shape)}"
        )
    if not torch.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError("Thresholds must be positive and finite")
    return thresholds[None, :, None] if thresholds.ndim == 1 else thresholds[:, :, None]


def tbr_encode_signed(eeg: Tensor, thresholds: Tensor | np.ndarray) -> Tensor:
    """Standard TBR on EEG [B, C, T]; returns signed int8 spikes [B, C, T].

    Strict comparisons: a difference exactly equal to ``theta`` emits nothing.
    A difference several thresholds large still emits a single +/-1 spike;
    that magnitude loss is inherent to TBR. Differences are taken inside each
    window only, so no spike is produced across a window boundary.
    """
    if eeg.ndim != 3:
        raise ValueError(f"Expected EEG [B, C, T], got {tuple(eeg.shape)}")
    theta = threshold_view(
        torch.as_tensor(thresholds, dtype=eeg.dtype, device=eeg.device),
        eeg.shape[0],
        eeg.shape[1],
    )
    differences = eeg[..., 1:] - eeg[..., :-1]
    signed = torch.zeros(eeg.shape, dtype=torch.int8, device=eeg.device)
    signed[..., 1:] = (differences > theta).to(torch.int8) - (
        differences < -theta
    ).to(torch.int8)
    return signed


def _format_output(signed: Tensor, polarity_mode: PolarityMode) -> Tensor:
    return signed if polarity_mode == "signed" else to_dual_binary(signed)


class AdaptiveThresholdEncoder:
    """Patient-adaptive TBR encoder: one threshold per channel for one subject.

    Build it from a fitted profile (``AdaptiveThresholdEncoder.from_profile``),
    where the profile comes from ``calibration.compute_threshold_profile`` on
    that subject's calibration windows only, or pass a saved threshold vector.

    ``encode`` accepts EEG [B, C, T] float32. By default it returns dual-binary
    contract spikes [B, T, 2C] float32. ``polarity_mode="signed"`` returns
    internal int8 {-1, 0, +1} spikes [B, C, T], which do not satisfy the
    shared spike-train contract.
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
                f"Expected one threshold per channel [{len(self.channel_names)}], "
                f"got {tuple(values.shape)}"
            )
        threshold_view(values, 1, len(self.channel_names))
        self._thresholds = values

    @classmethod
    def from_profile(
        cls,
        profile: ThresholdProfile,
        *,
        polarity_mode: PolarityMode = "dual_binary",
    ) -> AdaptiveThresholdEncoder:
        """Create the encoder for the subject described by a fitted profile."""
        return cls(
            profile.thresholds,
            channel_names=profile.channel_names,
            subject_id=profile.subject_id,
            polarity_mode=polarity_mode,
        )

    @property
    def thresholds(self) -> Tensor:
        """Per-channel threshold vector [C] (float32, EEG data units)."""
        return self._thresholds

    def encode(self, eeg: Tensor) -> Tensor:
        """Encode EEG [B, C, T] as dual-binary [B, T, 2C] or signed [B, C, T]."""
        check_eeg(eeg, n_channels=len(self.channel_names))
        signed = tbr_encode_signed(eeg, self._thresholds.to(eeg.device))
        return _format_output(signed, self.polarity_mode)


class FixedThresholdEncoder:
    """TBR encoder using one scalar threshold for every channel and subject.

    Non-adaptive baseline for comparison against ``AdaptiveThresholdEncoder``.
    """

    def __init__(
        self,
        threshold: float,
        polarity_mode: PolarityMode = "dual_binary",
    ) -> None:
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive and finite")
        self.threshold = float(threshold)
        self.polarity_mode = validate_polarity_mode(polarity_mode)

    def encode(self, eeg: Tensor) -> Tensor:
        """Encode EEG [B, C, T] with one threshold; output as configured polarity."""
        check_eeg(eeg)
        thresholds = torch.full(
            (eeg.shape[1],), self.threshold, dtype=eeg.dtype, device=eeg.device
        )
        return _format_output(tbr_encode_signed(eeg, thresholds), self.polarity_mode)

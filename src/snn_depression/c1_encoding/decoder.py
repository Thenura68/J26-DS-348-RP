"""Paired decoder for waveform reconstruction (stub)."""

from __future__ import annotations

from torch import Tensor


def reconstruct(spikes: Tensor, thresholds: Tensor) -> Tensor:
    """spikes [B, T, C_spk] → reconstructed EEG [B, C_eeg, S]."""
    raise NotImplementedError

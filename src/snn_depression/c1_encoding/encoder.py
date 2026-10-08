"""Patient-adaptive spike encoder (stub)."""

from __future__ import annotations

from torch import Tensor


class AdaptiveThresholdEncoder:
    """Delta/threshold encoder whose per-channel threshold θ is derived from a patient baseline."""

    def fit_baseline(self, baseline_eeg: Tensor) -> "AdaptiveThresholdEncoder":
        """baseline_eeg: [C_eeg, S] — derive per-channel thresholds."""
        raise NotImplementedError

    def encode(self, eeg: Tensor) -> Tensor:
        """eeg: [B, C_eeg, S] → spikes [B, T, C_spk] (contract boundary 2)."""
        raise NotImplementedError

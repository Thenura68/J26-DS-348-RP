"""Fallback STDP + SNN feature backbone (stub).

Used when the C2 global backbone cannot be connected (shape/time-resolution mismatch, or
not ready). Implements the same FeatureBackbone contract, so the adapter is unchanged.
"""

from __future__ import annotations

from torch import Tensor

from ..common.contracts import FeatureBackbone


class STDPFallbackBackbone(FeatureBackbone):
    def __init__(self, n_input_channels: int, out_features: int = 256):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.out_features = out_features

    def stdp_fit(self, spikes: Tensor, epochs: int = 1) -> "STDPFallbackBackbone":
        """Unsupervised STDP training on spike trains [B, T, C_spk] (no labels)."""
        raise NotImplementedError

    def extract(self, spikes: Tensor) -> Tensor:
        """spikes [B, T, C_spk] → features [B, T', 256]."""
        raise NotImplementedError

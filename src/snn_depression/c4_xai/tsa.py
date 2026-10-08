"""Temporal Spike Attribution across backbone + personal adapter (stub)."""

from __future__ import annotations

from torch import Tensor, nn


class TemporalSpikeAttribution:
    def __init__(self, backbone: nn.Module, adapter: nn.Module):
        self.backbone = backbone
        self.adapter = adapter

    def attribute(self, spikes: Tensor) -> Tensor:
        """spikes [B, T, C_spk] → attribution [B, n_classes, C_eeg, T]."""
        raise NotImplementedError

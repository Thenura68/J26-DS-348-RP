"""Global (population-level) SNN backbone — primary feature path (stub)."""

from __future__ import annotations

from torch import Tensor

from ..common.contracts import FeatureBackbone


class GlobalSNNBackbone(FeatureBackbone):
    """Pre-trained population SNN. Must return features [B, T', 256]."""

    def __init__(self, n_input_channels: int):
        super().__init__()
        self.n_input_channels = n_input_channels

    def extract(self, spikes: Tensor) -> Tensor:
        raise NotImplementedError

"""Personal SNN adapter classification layer (stub).

Planned structure (configs/c3_adapter.yaml):
    Linear 256 → 64  (feature compression)
    LIF output layer, 3 neurons [mild, moderate, severe], learnable β and V_th
    Surrogate-gradient training of adapter parameters only (< 5% of total)
"""

from __future__ import annotations

from torch import Tensor, nn


class PersonalSNNAdapter(nn.Module):
    def __init__(self, in_features: int = 256, hidden_features: int = 64, n_classes: int = 3):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_classes = n_classes

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """features [B, T', 256] → (spike_counts [B, 3], spike_record [B, T', 3])."""
        raise NotImplementedError

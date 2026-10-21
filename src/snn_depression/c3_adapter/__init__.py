"""Component 3 — Personalized SNN Adapter Classification Layer + Fallback Backbone.
Owner: Jayarathna P.D.T.M (IT23241596).

Core:     PersonalSNNAdapter   features [B, T', 256] → severity spike counts [B, 3]
Fallback: STDPFallbackBackbone spikes [B, T, C_spk] → features [B, T', 256]
          (same FeatureBackbone contract as C2, so the adapter runs without C2)
"""

from .adapter import PersonalSNNAdapter, decide
from .synthetic import (
    SyntheticDataset,
    ToyBackbone,
    make_synthetic_dataset,
    subject_train_test_split,
)
from .training import build_optimizer, evaluate, make_loss, train_adapter

__all__ = [
    "PersonalSNNAdapter",
    "SyntheticDataset",
    "ToyBackbone",
    "build_optimizer",
    "decide",
    "evaluate",
    "make_loss",
    "make_synthetic_dataset",
    "subject_train_test_split",
    "train_adapter",
]

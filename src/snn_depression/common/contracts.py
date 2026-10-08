"""Tensor contracts between components (see docs/interface_contract.md).

Every boundary tensor is batch-first. The checks here are cheap and should be called at
component boundaries so that a shape mismatch fails loudly at the point it happens,
instead of surfacing later as a silent accuracy drop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

CONTRACT_VERSION = "0.1"

FEATURE_DIM = 256
N_CLASSES = 3
CLASS_NAMES = ("mild", "moderate", "severe")


class ContractError(ValueError):
    """Raised when a tensor crossing a component boundary breaks the interface contract."""


def _check_common(x: Tensor, name: str, ndim: int) -> None:
    if not isinstance(x, Tensor):
        raise ContractError(f"{name}: expected torch.Tensor, got {type(x).__name__}")
    if x.ndim != ndim:
        raise ContractError(f"{name}: expected {ndim} dims, got shape {tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise ContractError(f"{name}: expected float32, got {x.dtype}")
    if not torch.isfinite(x).all():
        raise ContractError(f"{name}: contains NaN or Inf")


def check_eeg(x: Tensor, n_channels: int | None = None) -> None:
    """Preprocessed EEG: [B, C_eeg, S]."""
    _check_common(x, "eeg", 3)
    if n_channels is not None and x.shape[1] != n_channels:
        raise ContractError(f"eeg: expected {n_channels} channels, got {x.shape[1]}")


def check_spike_train(x: Tensor, n_channels: int | None = None) -> None:
    """Spike train from C1: [B, T, C_spk] with values in {0, 1}."""
    _check_common(x, "spike_train", 3)
    if not ((x == 0) | (x == 1)).all():
        raise ContractError("spike_train: values must be 0 or 1 (split signed spikes into ON/OFF channels)")
    if n_channels is not None and x.shape[2] != n_channels:
        raise ContractError(f"spike_train: expected {n_channels} channels, got {x.shape[2]}")


def check_features(x: Tensor) -> None:
    """Backbone features: [B, T', FEATURE_DIM]. T' may differ from the spike train's T."""
    _check_common(x, "features", 3)
    if x.shape[2] != FEATURE_DIM:
        raise ContractError(f"features: last dim must be {FEATURE_DIM}, got {x.shape[2]}")


def check_adapter_output(spike_counts: Tensor, spike_record: Tensor) -> None:
    """Adapter output: spike_counts [B, N_CLASSES], spike_record [B, T', N_CLASSES]."""
    _check_common(spike_counts, "spike_counts", 2)
    _check_common(spike_record, "spike_record", 3)
    if spike_counts.shape[1] != N_CLASSES or spike_record.shape[2] != N_CLASSES:
        raise ContractError(f"adapter output: class dimension must be {N_CLASSES}")
    if spike_counts.shape[0] != spike_record.shape[0]:
        raise ContractError("adapter output: batch size mismatch between counts and record")


class FeatureBackbone(nn.Module, ABC):
    """Base class for every feature backbone (C2 global SNN and C3 fallback STDP+SNN).

    Subclasses implement ``extract``; ``forward`` validates input and output against the
    contract, so the adapter can be mounted on either backbone without changes.
    """

    feature_dim: int = FEATURE_DIM

    @abstractmethod
    def extract(self, spikes: Tensor) -> Tensor:
        """Map a spike train [B, T, C_spk] to features [B, T', FEATURE_DIM]."""

    def forward(self, spikes: Tensor) -> Tensor:
        check_spike_train(spikes)
        features = self.extract(spikes)
        check_features(features)
        return features

    def freeze(self) -> "FeatureBackbone":
        """Freeze all parameters (used during per-patient adapter calibration)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

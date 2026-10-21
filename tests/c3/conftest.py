"""Shared fixtures for Component 3 tests (synthetic tensors only — no MODMA data)."""

import pytest
import torch
from torch import nn

from snn_depression.c3_adapter import PersonalSNNAdapter
from snn_depression.common.config import load_config
from snn_depression.common.contracts import FEATURE_DIM, FeatureBackbone
from snn_depression.common.seed import set_seed


class DummyBackbone(FeatureBackbone):
    """Stand-in for C2 / the fallback backbone: spikes [B, T, C_spk] → features [B, T, 256].

    Sigmoid keeps the features in (0, 1), like firing rates.
    """

    def __init__(self, n_input_channels: int = 16):
        super().__init__()
        self.proj = nn.Linear(n_input_channels, FEATURE_DIM)

    def extract(self, spikes: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(spikes))


@pytest.fixture(autouse=True)
def _seed():
    set_seed(42)


@pytest.fixture
def adapter() -> PersonalSNNAdapter:
    return PersonalSNNAdapter()


@pytest.fixture
def make_backbone():
    """Factory: ``make_backbone(n_input_channels)`` → a fresh, NOT yet frozen DummyBackbone."""
    return DummyBackbone


@pytest.fixture
def random_spikes():
    """Factory: ``random_spikes(B, T, C)`` → binary spike train [B, T, C] float32."""

    def _make(batch: int = 4, n_steps: int = 50, n_channels: int = 16, rate: float = 0.3):
        return (torch.rand(batch, n_steps, n_channels) < rate).float()

    return _make


@pytest.fixture(scope="session")
def toy_config() -> dict:
    """The real c3 config, with the synthetic data shrunk so one training run takes ~2 s.

    Shorter windows (60 steps instead of 100) carry less evidence per window, so the class effect
    is raised from 0.08 to 0.10 to stay learnable in 12 epochs. Shared across tests: copy it
    (copy.deepcopy) before changing anything.
    """
    cfg = load_config("c3_adapter")
    cfg["synthetic"].update(windows_per_subject=10, n_steps=60, effect=0.10)
    cfg["training"].update(epochs=12)
    return cfg

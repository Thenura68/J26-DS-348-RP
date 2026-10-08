import pytest
import torch

from snn_depression.common.contracts import (
    FEATURE_DIM,
    ContractError,
    FeatureBackbone,
    check_adapter_output,
    check_features,
    check_spike_train,
)


class _ToyBackbone(FeatureBackbone):
    """Minimal backbone used only to test the contract machinery."""

    def __init__(self, n_in: int, out_dim: int = FEATURE_DIM):
        super().__init__()
        self.proj = torch.nn.Linear(n_in, out_dim)

    def extract(self, spikes):
        return self.proj(spikes)


def test_valid_spike_train_passes():
    check_spike_train((torch.rand(2, 50, 8) > 0.5).float())


def test_non_binary_spikes_rejected():
    with pytest.raises(ContractError):
        check_spike_train(torch.rand(2, 50, 8))


def test_wrong_feature_dim_rejected():
    with pytest.raises(ContractError):
        check_features(torch.zeros(2, 50, 128))


def test_nan_features_rejected():
    x = torch.zeros(2, 50, FEATURE_DIM)
    x[0, 0, 0] = float("nan")
    with pytest.raises(ContractError):
        check_features(x)


def test_backbone_forward_enforces_contract():
    spikes = (torch.rand(4, 30, 8) > 0.7).float()
    out = _ToyBackbone(8)(spikes)
    assert out.shape == (4, 30, FEATURE_DIM)


def test_backbone_with_wrong_output_fails():
    spikes = (torch.rand(4, 30, 8) > 0.7).float()
    with pytest.raises(ContractError):
        _ToyBackbone(8, out_dim=100)(spikes)


def test_freeze_disables_grads():
    bb = _ToyBackbone(8).freeze()
    assert all(not p.requires_grad for p in bb.parameters())


def test_adapter_output_shapes():
    check_adapter_output(torch.zeros(4, 3), torch.zeros(4, 30, 3))
    with pytest.raises(ContractError):
        check_adapter_output(torch.zeros(4, 2), torch.zeros(4, 30, 2))

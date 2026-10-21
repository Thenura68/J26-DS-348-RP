# Unit tests for unsupervised STDP feature extraction layer

from __future__ import annotations

import torch
import pytest

from snn_depression.c2_global_snn.stdp import STDPLayer


def test_stdp_layer_forward():
    in_features = 16
    out_features = 8
    batch = 2
    n_steps = 20

    layer = STDPLayer(in_features=in_features, out_features=out_features)
    spikes_in = (torch.rand(batch, n_steps, in_features) > 0.8).float()

    out = layer(spikes_in, learn=False)
    assert out.shape == (batch, n_steps, out_features)
    assert ((out == 0) | (out == 1)).all()


def test_stdp_learning_update():
    in_features = 4
    out_features = 2
    batch = 1

    layer = STDPLayer(in_features=in_features, out_features=out_features, lr_stdp=0.1)
    layer.train()

    # Create pre-then-post spike sequence to elicit LTP
    spikes_in = torch.zeros(batch, 5, in_features)
    spikes_in[:, 0, 0] = 1.0  # pre-spike at t=0
    spikes_in[:, 1, :] = 1.0  # strong input at t=1 causing post-spike

    w_before = layer.weight.clone()
    _ = layer(spikes_in, learn=True)
    w_after = layer.weight

    # Weights should be normalized and bounded
    assert (w_after >= layer.w_min).all()
    assert (w_after <= layer.w_max).all()


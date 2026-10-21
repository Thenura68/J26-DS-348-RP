# Unit tests for GlobalSNNBackbone and adaptation layer integration

from __future__ import annotations

import torch
import pytest

from snn_depression.c2_global_snn.backbone import GlobalSNNBackbone
from snn_depression.c3_adapter.adapter import PersonalSNNAdapter
from snn_depression.common.contracts import FEATURE_DIM, check_features


def test_global_snn_contract_mumtaz_channels():
    batch = 3
    n_steps = 50
    c_spk = 38  # MUMTAZ 19 EEG * 2
    backbone = GlobalSNNBackbone(n_input_channels=c_spk, reservoir_size=64)

    spikes = (torch.rand(batch, n_steps, c_spk) > 0.85).float()
    features = backbone(spikes)

    # Must satisfy FeatureBackbone contract
    check_features(features)
    assert features.shape == (batch, n_steps, FEATURE_DIM)
    assert features.dtype == torch.float32


def test_global_snn_contract_modma_channels():
    batch = 2
    n_steps = 40
    c_spk = 256  # MODMA 128 EEG * 2
    backbone = GlobalSNNBackbone(n_input_channels=c_spk, reservoir_size=128)

    spikes = (torch.rand(batch, n_steps, c_spk) > 0.9).float()
    features = backbone(spikes)

    check_features(features)
    assert features.shape == (batch, n_steps, FEATURE_DIM)


def test_forward_classify_and_synops():
    batch = 4
    n_steps = 30
    c_spk = 38
    backbone = GlobalSNNBackbone(n_input_channels=c_spk, reservoir_size=64, n_classes=2)

    spikes = (torch.rand(batch, n_steps, c_spk) > 0.85).float()
    logits, preds, aux = backbone.forward_classify(spikes)

    assert logits.shape == (batch, 2)
    assert preds.shape == (batch,)
    assert "confidence_pct" in aux
    assert aux["confidence_pct"].shape == (batch,)
    assert (aux["confidence_pct"] >= 0.0).all() and (aux["confidence_pct"] <= 100.0).all()
    assert "total_synops" in aux and aux["total_synops"] > 0
    assert "spike_sparsity" in aux and 0.0 <= aux["spike_sparsity"] <= 1.0


def test_c4_trace_extraction():
    batch = 2
    n_steps = 25
    c_spk = 38
    backbone = GlobalSNNBackbone(n_input_channels=c_spk, reservoir_size=32)

    spikes = (torch.rand(batch, n_steps, c_spk) > 0.85).float()
    trace_data = backbone.extract_with_traces(spikes)

    assert "features" in trace_data
    assert "res_spikes" in trace_data
    assert "res_mems" in trace_data
    assert "weights" in trace_data
    assert "w_in" in trace_data["weights"]
    assert "w_rec" in trace_data["weights"]


@pytest.mark.parametrize("mode", [
    "reservoir_lif",
    "reservoir_alif",
    "reservoir_alif_stdp",
    "full_engine",
])
def test_ablation_modes(mode):
    backbone = GlobalSNNBackbone(n_input_channels=38, reservoir_size=32, ablation_mode=mode)
    spikes = (torch.rand(2, 20, 38) > 0.85).float()
    features = backbone(spikes)
    check_features(features)


def test_c3_adapter_integration():
    # Verify that PersonalSNNAdapter directly mounts onto GlobalSNNBackbone
    batch = 4
    n_steps = 50
    c_spk = 38

    # Backbone
    backbone = GlobalSNNBackbone(n_input_channels=c_spk, reservoir_size=64)
    # Adapter head
    adapter = PersonalSNNAdapter(in_features=FEATURE_DIM, hidden_features=32, n_classes=3)

    spikes = (torch.rand(batch, n_steps, c_spk) > 0.85).float()

    # Global model extracts features
    features = backbone(spikes)  # [B, T', 256]
    # consumes features and outputs predictions
    spike_counts, spike_record = adapter(features)
    pred_classes, silent_flags = adapter.predict(features)

    assert spike_counts.shape == (batch, 3)
    assert spike_record.shape == (batch, n_steps, 3)
    assert pred_classes.shape == (batch,)
    assert silent_flags.shape == (batch,)


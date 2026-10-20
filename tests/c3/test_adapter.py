"""Milestone 1 — PersonalSNNAdapter: shapes, contract, gradients, frozen backbone."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from snn_depression.c3_adapter import PersonalSNNAdapter
from snn_depression.common.config import load_config
from snn_depression.common.contracts import (
    FEATURE_DIM,
    N_CLASSES,
    ContractError,
    check_adapter_output,
)
from snn_depression.evaluation.profiling import count_parameters, trainable_ratio

HIDDEN = 64
# Linear 256→64 + Linear 64→3, plus a per-neuron β and V_th in each LIF layer.
LINEAR_PARAMS = FEATURE_DIM * HIDDEN + HIDDEN + HIDDEN * N_CLASSES + N_CLASSES  # 16,643
PER_NEURON_PARAMS = 2 * (HIDDEN + N_CLASSES)  # 134
EXPECTED_PARAMS = LINEAR_PARAMS + PER_NEURON_PARAMS  # 16,777


# --- 1–3: shapes, contract, output values -------------------------------------------------


@pytest.mark.parametrize("batch, n_steps", [(1, 1), (4, 7), (8, 50), (2, 200)])
def test_output_shapes_for_any_time_length(adapter, batch, n_steps):
    counts, record = adapter(torch.rand(batch, n_steps, FEATURE_DIM))
    assert counts.shape == (batch, N_CLASSES)
    assert record.shape == (batch, n_steps, N_CLASSES)


def test_output_satisfies_contract(adapter):
    counts, record = adapter(torch.rand(4, 30, FEATURE_DIM))
    check_adapter_output(counts, record)
    assert counts.dtype == record.dtype == torch.float32


def test_record_is_binary_and_counts_are_its_time_sum(adapter):
    n_steps = 40
    counts, record = adapter(torch.rand(4, n_steps, FEATURE_DIM) * 3)
    assert ((record == 0) | (record == 1)).all()
    assert torch.equal(counts, record.sum(dim=1))
    assert (counts >= 0).all() and (counts <= n_steps).all()


# --- 4: accepts every feature type allowed by the contract ---------------------------------


@pytest.mark.parametrize(
    "make_features",
    [
        lambda: (torch.rand(4, 30, FEATURE_DIM) < 0.2).float(),  # binary spikes
        lambda: torch.rand(4, 30, FEATURE_DIM),  # firing rates in [0, 1]
        lambda: torch.randn(4, 30, FEATURE_DIM) * 5.0,  # membrane-like, signed, large
    ],
    ids=["spikes", "rates", "membrane"],
)
def test_accepts_spikes_rates_and_membrane_values(adapter, make_features):
    counts, record = adapter(make_features())
    check_adapter_output(counts, record)
    assert ((record == 0) | (record == 1)).all()


# --- 5: bad inputs fail loudly ---------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        torch.rand(4, 30, 128),  # wrong feature dim
        torch.rand(4, FEATURE_DIM),  # missing time axis
        torch.rand(4, 0, FEATURE_DIM),  # T' = 0
        torch.rand(4, 30, FEATURE_DIM, dtype=torch.float64),  # wrong dtype
    ],
    ids=["dim128", "2d", "empty_time", "float64"],
)
def test_bad_input_raises_contract_error(adapter, bad):
    with pytest.raises(ContractError):
        adapter(bad)


def test_nan_input_raises_contract_error(adapter):
    x = torch.rand(2, 10, FEATURE_DIM)
    x[1, 3, 7] = float("nan")
    with pytest.raises(ContractError):
        adapter(x)


@pytest.mark.parametrize(
    "kwargs", [{"in_features": 128}, {"n_classes": 2}, {"surrogate_name": "foo"}]
)
def test_constructor_rejects_contract_breaking_sizes(kwargs):
    with pytest.raises(ValueError):
        PersonalSNNAdapter(**kwargs)


# --- 6: parameter count, learnable β / V_th --------------------------------------------------


def test_parameter_count_with_per_neuron_beta_and_threshold(adapter):
    assert count_parameters(adapter) == EXPECTED_PARAMS
    params = dict(adapter.named_parameters())
    for layer, size in [("lif1", HIDDEN), ("lif2", N_CLASSES)]:
        assert params[f"{layer}.beta"].shape == (size,)
        assert params[f"{layer}.threshold"].shape == (size,)


def test_beta_and_threshold_are_buffers_when_not_learnable():
    adapter = PersonalSNNAdapter(learn_beta=False, learn_threshold=False)
    assert count_parameters(adapter) == LINEAR_PARAMS
    names = {n for n, _ in adapter.named_parameters()}
    assert not any(n.endswith(("beta", "threshold")) for n in names)
    # Still saved with the model, just not trained.
    assert {"lif1.beta", "lif1.threshold"} <= set(adapter.state_dict())


def test_membrane_state_is_not_saved_in_state_dict(adapter):
    adapter(torch.rand(5, 10, FEATURE_DIM))
    # U[t] is state, not a parameter: it must not end up in a patient profile.
    assert not any(k.endswith("mem") for k in adapter.state_dict())


# --- 7–8: gradients reach only the adapter; frozen backbone untouched ------------------------


def _frozen_pipeline_loss(backbone, adapter, spikes):
    labels = torch.tensor([0, 1, 2, 1])
    counts, _ = adapter(backbone(spikes))
    # Spike counts used as logits — only to get a gradient here; the real loss is Milestone 2.
    return F.cross_entropy(counts, labels)


def test_gradients_reach_every_adapter_parameter_and_no_backbone_parameter(
    adapter, make_backbone, random_spikes
):
    backbone = make_backbone(16).freeze()
    loss = _frozen_pipeline_loss(backbone, adapter, random_spikes(4, 50, 16))
    loss.backward()

    for name, p in adapter.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all zeros"
    for name, p in backbone.named_parameters():
        assert not p.requires_grad, f"backbone {name} is trainable"
        assert p.grad is None, f"backbone {name} received a gradient"


def test_optimiser_step_changes_adapter_but_not_frozen_backbone(
    adapter, make_backbone, random_spikes
):
    backbone = make_backbone(16).freeze()
    backbone_before = {k: v.clone() for k, v in backbone.state_dict().items()}
    adapter_before = {k: v.clone() for k, v in adapter.state_dict().items()}

    # The optimiser deliberately gets the backbone parameters too: freezing alone must protect
    # them, without relying on everyone remembering to pass only adapter.parameters().
    opt = torch.optim.Adam(itertools.chain(backbone.parameters(), adapter.parameters()), lr=1e-2)
    opt.zero_grad()
    _frozen_pipeline_loss(backbone, adapter, random_spikes(4, 50, 16)).backward()
    opt.step()

    assert not backbone.training
    for k, v in backbone.state_dict().items():
        assert torch.equal(v, backbone_before[k]), f"frozen backbone {k} changed"
    changed = [k for k, v in adapter.state_dict().items() if not torch.equal(v, adapter_before[k])]
    assert {"fc1.weight", "fc2.weight", "lif1.beta", "lif1.threshold"} <= set(changed)


# --- 9: no state leaks between calls or between samples --------------------------------------


def test_repeated_calls_give_identical_output(adapter):
    x = torch.rand(4, 30, FEATURE_DIM) * 2
    counts_a, record_a = adapter(x)
    counts_b, record_b = adapter(x)
    assert torch.equal(record_a, record_b) and torch.equal(counts_a, counts_b)


def test_samples_in_a_batch_do_not_affect_each_other(adapter):
    shared = torch.rand(1, 30, FEATURE_DIM) * 2
    # Same batch size both times (same matmul kernels) — only the other samples differ.
    _, record_a = adapter(torch.cat([shared, torch.rand(3, 30, FEATURE_DIM) * 2]))
    _, record_b = adapter(torch.cat([shared, torch.randn(3, 30, FEATURE_DIM) * 5]))
    assert torch.equal(record_a[0], record_b[0])


# --- 10: config and trainable-ratio target ---------------------------------------------------


def test_from_config_reads_yaml_values():
    cfg = load_config("c3_adapter")
    adapter = PersonalSNNAdapter.from_config(cfg)
    a = cfg["adapter"]
    assert adapter.hidden_features == a["hidden_features"]
    hidden, n_classes = a["hidden_features"], a["n_classes"]
    assert torch.allclose(adapter.lif1.beta, torch.full((hidden,), a["beta_init"]))
    assert torch.allclose(adapter.lif2.threshold, torch.full((n_classes,), a["threshold_init"]))
    assert count_parameters(adapter) == EXPECTED_PARAMS


def test_from_config_rejects_unsupported_neuron():
    cfg = load_config("c3_adapter")
    cfg["adapter"]["neuron"] = "izhikevich"
    with pytest.raises(ValueError):
        PersonalSNNAdapter.from_config(cfg)


def test_trainable_ratio_below_target_with_large_enough_backbone(adapter, make_backbone):
    # Linear(1290 → 256) ≈ 330k params: the "backbone ≥ ~320k params" condition in CLAUDE.md.
    backbone = make_backbone(1290).freeze()
    max_ratio = load_config("c3_adapter")["calibration"]["max_trainable_ratio"]
    assert trainable_ratio(adapter, backbone) < max_ratio

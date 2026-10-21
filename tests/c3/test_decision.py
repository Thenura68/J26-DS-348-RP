"""Milestone 2 — silent-output rule: count first, membrane tie-break, silent flag (decision c)."""

import torch

from snn_depression.c3_adapter import PersonalSNNAdapter, decide
from snn_depression.common.contracts import FEATURE_DIM, check_adapter_output


def test_unique_max_count_is_plain_argmax():
    counts = torch.tensor([[5.0, 1.0, 0.0], [0.0, 2.0, 7.0], [1.0, 3.0, 2.0]])
    mem_max = torch.tensor([[0.0, 9.0, 9.0], [9.0, 0.0, 0.0], [9.0, 0.0, 9.0]])
    pred, silent = decide(counts, mem_max)
    # The membrane is ignored when one class clearly has the most spikes.
    assert pred.tolist() == counts.argmax(dim=1).tolist() == [0, 2, 1]
    assert not silent.any()


def test_silent_window_is_flagged_and_decided_by_membrane():
    pred, silent = decide(torch.zeros(1, 3), torch.tensor([[0.1, 0.7, 0.3]]))
    # Plain argmax would say 0 ("mild"); the neuron closest to firing (moderate) wins instead.
    assert pred.tolist() == [1]
    assert silent.tolist() == [True]


def test_tie_break_only_considers_classes_with_the_top_count():
    pred, _ = decide(torch.tensor([[2.0, 2.0, 1.0]]), torch.tensor([[0.1, 0.3, 0.9]]))
    assert pred.tolist() == [1]  # class 2 has the highest membrane but fewer spikes


def test_exact_tie_goes_to_the_lowest_index():
    pred, _ = decide(torch.zeros(2, 3), torch.full((2, 3), 0.5))
    assert pred.tolist() == [0, 0]


def test_silent_flag_only_when_no_output_spikes():
    _, silent = decide(torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), torch.zeros(2, 3))
    assert silent.tolist() == [False, True]


def test_rule_is_deterministic():
    counts = torch.randint(0, 3, (64, 3)).float()
    mem_max = torch.randn(64, 3)
    first, second = decide(counts, mem_max), decide(counts, mem_max)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_all_silent_adapter_flags_every_window_and_stays_contract_valid():
    # V_th far above any reachable membrane: no neuron in the adapter can fire.
    adapter = PersonalSNNAdapter(threshold_init=1000.0)
    features = torch.rand(8, 30, FEATURE_DIM)
    counts, record = adapter(features)
    check_adapter_output(counts, record)  # unchanged contract even when silent
    assert counts.sum() == 0

    pred, silent = adapter.predict(features)
    assert silent.all()
    assert pred.dtype == torch.int64 and ((pred >= 0) & (pred <= 2)).all()
    pred_again, _ = adapter.predict(features)
    assert torch.equal(pred, pred_again)


def test_predict_agrees_with_forward_counts(adapter):
    features = torch.rand(32, 40, FEATURE_DIM) * 3
    counts, _ = adapter(features)
    pred, silent = adapter.predict(features)
    top = counts.amax(dim=1, keepdim=True)
    unique_max = (counts == top).sum(dim=1) == 1
    assert torch.equal(pred[unique_max], counts.argmax(dim=1)[unique_max])
    assert torch.equal(silent, counts.sum(dim=1) == 0)

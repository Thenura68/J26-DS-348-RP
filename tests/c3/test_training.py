"""Milestone 2 — training on a frozen backbone: learning, isolation, V_th floor, reproducibility.

Training runs once per module (``trained`` fixture) on the small ``toy_config`` (~2 s).
"""

import copy
import itertools
from types import SimpleNamespace

import pytest
import torch

from snn_depression.c3_adapter import (
    PersonalSNNAdapter,
    ToyBackbone,
    build_optimizer,
    make_synthetic_dataset,
    subject_train_test_split,
    train_adapter,
)
from snn_depression.c3_adapter.training import check_optimizer_excludes
from snn_depression.common.seed import set_seed


def _run(cfg: dict, seed: int = 42) -> SimpleNamespace:
    """Data → frozen ToyBackbone → adapter → train_adapter, exactly as the script does."""
    syn = cfg["synthetic"]
    data = make_synthetic_dataset(syn, seed=syn["seed"])
    train_idx, test_idx = subject_train_test_split(
        data.subject_ids, data.labels, syn["test_subjects_per_class"], seed=syn["seed"]
    )
    backbone = ToyBackbone(syn["n_channels"], seed=syn["seed"], **cfg["toy_backbone"])
    backbone_before = {k: v.clone() for k, v in backbone.state_dict().items()}
    set_seed(seed)
    adapter = PersonalSNNAdapter.from_config(cfg)
    history = train_adapter(
        backbone, adapter, data.subset(train_idx), data.subset(test_idx), cfg["training"], seed=seed
    )
    return SimpleNamespace(
        backbone=backbone, backbone_before=backbone_before, adapter=adapter, history=history
    )


@pytest.fixture(scope="module")
def trained(toy_config):
    return _run(toy_config)


def test_history_has_one_entry_per_epoch(trained, toy_config):
    epochs = toy_config["training"]["epochs"]
    h = trained.history
    assert len(h["loss"]) == epochs + 1  # + the loss before any update
    assert len(h["train"]) == len(h["test"]) == len(h["min_threshold"]) == epochs


def test_loss_after_training_is_clearly_lower_than_at_start(trained):
    h = trained.history
    # Same data, same evaluation, weights before vs after training.
    assert h["train"][-1]["loss"] < 0.25 * h["loss"][0]
    assert h["loss"][-1] < h["loss"][1]  # running training loss also fell


def test_test_accuracy_is_clearly_above_chance(trained):
    final = trained.history["test"][-1]
    chance = trained.history["chance"]
    assert chance == pytest.approx(1 / 3)
    # Held-out SUBJECTS, so this is generalisation, not memorisation of seen windows.
    assert final["accuracy"] >= 0.50
    assert final["macro_f1"] >= 0.45


def test_backbone_is_bit_identical_and_in_eval_mode_after_training(trained):
    assert not trained.backbone.training
    assert all(not p.requires_grad for p in trained.backbone.parameters())
    for k, v in trained.backbone.state_dict().items():
        assert torch.equal(v, trained.backbone_before[k]), f"backbone {k} changed during training"


def test_optimizer_contains_only_adapter_parameters(toy_config, make_backbone):
    adapter = PersonalSNNAdapter.from_config(toy_config)
    backbone = make_backbone(16)
    opt = build_optimizer(adapter, toy_config["training"])
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids == {id(p) for p in adapter.parameters()}
    assert not opt_ids & {id(p) for p in backbone.parameters()}
    check_optimizer_excludes(opt, backbone)  # does not raise


def test_optimizer_check_rejects_backbone_parameters(toy_config, make_backbone):
    adapter = PersonalSNNAdapter.from_config(toy_config)
    backbone = make_backbone(16)
    bad = torch.optim.Adam(itertools.chain(adapter.parameters(), backbone.parameters()))
    with pytest.raises(ValueError):
        check_optimizer_excludes(bad, backbone)


def test_threshold_never_drops_below_the_floor(trained):
    floor = trained.adapter.threshold_min
    assert floor > 0
    assert min(trained.history["min_threshold"]) >= floor
    for lif in (trained.adapter.lif1, trained.adapter.lif2):
        assert (lif.threshold >= floor).all()
        assert ((lif.beta >= 0) & (lif.beta <= 1)).all()


def test_constrain_parameters_clamps_only_out_of_range_values():
    adapter = PersonalSNNAdapter(threshold_min=0.1)
    with torch.no_grad():
        adapter.lif1.threshold[:3] = torch.tensor([-0.5, 0.0, 0.7])
        adapter.lif1.beta[:3] = torch.tensor([1.5, -0.2, 0.6])
    adapter.constrain_parameters()
    assert adapter.lif1.threshold[:3].tolist() == pytest.approx([0.1, 0.1, 0.7])
    assert adapter.lif1.beta[:3].tolist() == pytest.approx([1.0, 0.0, 0.6])


def test_threshold_min_must_be_positive_and_not_above_the_initial_threshold():
    with pytest.raises(ValueError):
        PersonalSNNAdapter(threshold_min=0.0)
    with pytest.raises(ValueError):
        PersonalSNNAdapter(threshold_init=1.0, threshold_min=1.5)


def test_silent_test_windows_are_counted(trained):
    final = trained.history["test"][-1]
    assert trained.history["n_silent_test"] == final["n_silent"]
    assert 0 <= final["n_silent"] <= final["n"]


def test_same_seed_gives_the_same_result(toy_config):
    cfg = copy.deepcopy(toy_config)
    cfg["training"]["epochs"] = 2
    first, second = _run(cfg, seed=7), _run(cfg, seed=7)
    assert first.history["loss"] == second.history["loss"]
    assert first.history["test"] == second.history["test"]
    for k, v in first.adapter.state_dict().items():
        assert torch.equal(v, second.adapter.state_dict()[k])

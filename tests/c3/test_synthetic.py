"""Milestone 2 — synthetic dataset, subject-wise split and toy backbone."""

import pytest
import torch

from snn_depression.c3_adapter import (
    SyntheticDataset,
    ToyBackbone,
    make_synthetic_dataset,
    subject_train_test_split,
)
from snn_depression.common.contracts import FEATURE_DIM, N_CLASSES, check_spike_train


@pytest.fixture(scope="module")
def syn(toy_config):
    return toy_config["synthetic"]


@pytest.fixture(scope="module")
def data(syn) -> SyntheticDataset:
    return make_synthetic_dataset(syn, seed=syn["seed"])


def test_shapes_dtypes_and_binary_spikes(syn, data):
    n = N_CLASSES * syn["n_subjects_per_class"] * syn["windows_per_subject"]
    assert data.spikes.shape == (n, syn["n_steps"], syn["n_channels"])
    assert data.labels.shape == data.subject_ids.shape == (n,)
    assert data.labels.dtype == data.subject_ids.dtype == torch.int64
    check_spike_train(data.spikes)  # float32, finite, values in {0, 1}


def test_classes_are_balanced_in_windows_and_subjects(syn, data):
    windows_per_class = data.labels.bincount(minlength=N_CLASSES)
    assert (windows_per_class == windows_per_class[0]).all()
    for k in range(N_CLASSES):
        assert len(data.subject_ids[data.labels == k].unique()) == syn["n_subjects_per_class"]


def test_every_subject_has_exactly_one_label(data):
    for s in data.subject_ids.unique():
        assert len(data.labels[data.subject_ids == s].unique()) == 1


def test_split_is_by_subject_stratified_and_complete(syn, data):
    train_idx, test_idx = subject_train_test_split(
        data.subject_ids, data.labels, syn["test_subjects_per_class"], seed=0
    )
    train_subj = set(data.subject_ids[train_idx].tolist())
    test_subj = set(data.subject_ids[test_idx].tolist())
    assert not train_subj & test_subj, "a subject appears in both train and test"
    # Every window is used exactly once.
    assert sorted(torch.cat([train_idx, test_idx]).tolist()) == list(range(len(data)))
    for k in range(N_CLASSES):
        held_out = data.subject_ids[test_idx][data.labels[test_idx] == k].unique()
        assert len(held_out) == syn["test_subjects_per_class"]


def test_split_rejects_a_subject_with_two_labels():
    subject_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    labels = torch.tensor([0, 1, 0, 0, 1, 1, 2, 2])  # subject 0 has labels 0 and 1
    with pytest.raises(ValueError):
        subject_train_test_split(subject_ids, labels, 1, seed=0)


def test_generator_is_deterministic_and_seed_dependent(syn, data):
    again = make_synthetic_dataset(syn, seed=syn["seed"])
    other = make_synthetic_dataset(syn, seed=syn["seed"] + 1)
    assert torch.equal(data.spikes, again.spikes)
    assert not torch.equal(data.spikes, other.spikes)


def test_each_class_fires_faster_in_its_own_channel_group_and_segment(syn, data):
    """The classes really differ: class k has the highest rate in (channel group k, segment k)."""
    group = torch.arange(syn["n_channels"]) * N_CLASSES // syn["n_channels"]
    segment = torch.arange(syn["n_steps"]) * N_CLASSES // syn["n_steps"]
    for k in range(N_CLASSES):
        cell = data.spikes[:, segment == k][:, :, group == k].mean(dim=(1, 2))  # [N]
        rate_per_class = torch.stack([cell[data.labels == c].mean() for c in range(N_CLASSES)])
        assert rate_per_class.argmax() == k


def test_toy_backbone_returns_contract_features_deterministically(syn, data):
    spikes = data.spikes[:16]
    features = ToyBackbone(syn["n_channels"], seed=0)(spikes)  # forward() checks the contract
    assert features.shape == (16, syn["n_steps"], FEATURE_DIM)
    assert ((features == 0) | (features == 1)).all()
    assert torch.equal(features, ToyBackbone(syn["n_channels"], seed=0)(spikes))
    # Almost no dead features: at least 95% of feature neurons fire somewhere in the dataset.
    all_features = ToyBackbone(syn["n_channels"], seed=0)(data.spikes)
    assert (all_features.sum(dim=(0, 1)) > 0).float().mean() >= 0.95

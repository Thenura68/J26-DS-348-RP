"""Synthetic 3-class spike data and a toy frozen backbone, for Milestone 2 (no real EEG).

The data is built to behave like MODMA in the ways that matter for evaluation:
- one severity label per SUBJECT, many windows per subject, windows in recording order;
- subject-to-subject variability, so a model must generalise to unseen subjects;
- class differences that are real but small and noisy (learnable, not trivially separable).

Nothing here says anything about real EEG accuracy. It only shows the training machinery works.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from ..common.contracts import FEATURE_DIM, N_CLASSES, FeatureBackbone


@dataclass(frozen=True)
class SyntheticDataset:
    """spikes [N, T, C_spk] float32 in {0, 1}; labels [N] int64; subject_ids [N] int64.

    Windows are stored subject by subject, each subject's windows in chronological order, so
    ``data.splits.calibration_split`` can be applied per subject later (Milestone 4).
    """

    spikes: Tensor
    labels: Tensor
    subject_ids: Tensor

    def __len__(self) -> int:
        return self.labels.shape[0]

    def subset(self, idx: Tensor) -> "SyntheticDataset":
        return SyntheticDataset(self.spikes[idx], self.labels[idx], self.subject_ids[idx])


def make_synthetic_dataset(params: dict, seed: int) -> SyntheticDataset:
    """Generate the toy dataset from the ``synthetic`` config section.

    Signal: the channels are split into 3 groups and the window into 3 time segments
    (first / middle / last third). Class k fires faster in channel group k during segment k.
    That gives each class a channel signature AND a temporal one, which C4 can attribute later.

    Overlap (why it is not trivial):
    - every subject has its own random gain per channel (log-normal, ``subject_jitter``), applied
      to the whole window. A subject with naturally high rates in group j can resemble class j;
      this inter-subject variability is exactly the problem the personal adapter exists for;
    - every subject expresses the class signal with a different strength (same log-normal spread);
    - every window gets extra rate noise per channel (``window_noise``);
    - spikes are Bernoulli draws, so counts are noisy even at a fixed rate.
    """
    n_subj_per_class = params["n_subjects_per_class"]
    n_windows = params["windows_per_subject"]
    n_steps = params["n_steps"]
    n_channels = params["n_channels"]
    base_rate = params["base_rate"]
    effect = params["effect"]
    jitter = params["subject_jitter"]
    window_noise = params["window_noise"]
    if n_channels < N_CLASSES or n_steps < N_CLASSES:
        raise ValueError("need at least one channel and one time step per class")

    # Own generator, so the data never depends on (or disturbs) the global RNG state.
    g = torch.Generator().manual_seed(seed)

    group = torch.arange(n_channels) * N_CLASSES // n_channels  # [C] → 0, 1, 2
    segment = torch.arange(n_steps) * N_CLASSES // n_steps  # [T] → 0, 1, 2
    # templates[k] marks (time step, channel) cells where class k fires faster: [3, T, C]
    templates = torch.stack(
        [(segment[:, None] == k) & (group[None, :] == k) for k in range(N_CLASSES)]
    ).float()

    n_subjects = N_CLASSES * n_subj_per_class
    subject_label = torch.arange(N_CLASSES).repeat_interleave(n_subj_per_class)  # [S]
    channel_gain = torch.exp(jitter * torch.randn(n_subjects, n_channels, generator=g))  # [S, C]
    signal_gain = torch.exp(jitter * torch.randn(n_subjects, generator=g))  # [S]

    # Per-subject rate map [S, T, C]: baseline × channel gain, plus the class template.
    subject_rate = (
        base_rate * channel_gain[:, None, :]
        + effect * signal_gain[:, None, None] * templates[subject_label]
    )
    # Per-window noise, constant over the window for each channel: [S, W, 1, C]
    noise = window_noise * torch.randn(n_subjects, n_windows, 1, n_channels, generator=g)
    rate = (subject_rate[:, None] + noise).clamp(0.0, 1.0)  # [S, W, T, C]
    spikes = torch.bernoulli(rate, generator=g).reshape(-1, n_steps, n_channels)

    labels = subject_label.repeat_interleave(n_windows)
    subject_ids = torch.arange(n_subjects).repeat_interleave(n_windows)
    return SyntheticDataset(spikes.float(), labels.long(), subject_ids.long())


def subject_train_test_split(
    subject_ids: Tensor, labels: Tensor, test_subjects_per_class: int, seed: int
) -> tuple[Tensor, Tensor]:
    """Split WINDOW indices into (train, test) by subject, stratified by class.

    Splitting by window would put windows of the same person in train and test, so the model could
    recognise the person instead of the severity (leakage). Here a subject is entirely in one set.
    Returns sorted int64 index tensors, so chronological order within a subject is preserved.
    """
    ids = subject_ids.numpy()
    lab = labels.numpy()
    rng = np.random.default_rng(seed)
    test_subjects = []
    for k in range(N_CLASSES):
        subjects_k = []
        for s in np.unique(ids):
            subject_labels = np.unique(lab[ids == s])
            if len(subject_labels) != 1:
                raise ValueError(f"subject {s} has several labels {subject_labels}; expected one")
            if subject_labels[0] == k:
                subjects_k.append(s)
        if len(subjects_k) <= test_subjects_per_class:
            raise ValueError(f"class {k}: {len(subjects_k)} subjects, cannot hold out "
                             f"{test_subjects_per_class} and still train")
        test_subjects.extend(rng.choice(subjects_k, size=test_subjects_per_class, replace=False))

    test_mask = np.isin(ids, test_subjects)
    train_idx = torch.from_numpy(np.flatnonzero(~test_mask))
    test_idx = torch.from_numpy(np.flatnonzero(test_mask))
    assert not set(ids[train_idx.numpy()]) & set(ids[test_idx.numpy()])
    return train_idx, test_idx


class ToyBackbone(FeatureBackbone):
    """Stand-in for C2 / the fallback backbone: spikes [B, T, C_spk] → spike features [B, T, 256].

    A fixed random LIF reservoir: a random projection C_spk → 256 drives 256 LIF neurons, whose
    output spikes are the features. It is spiking because both real backbones are (C2 is a LIF
    reservoir; the fallback backbone is a layer of STDP-trained LIF neurons). A first version
    (random projection + sigmoid) produced dense values near 0.5 whose class-related variation
    was ~10× smaller than the fixed offsets between neurons; a fresh adapter could not learn
    from it.
    It is never trained: the training loop freezes it, as it will freeze the real backbone.
    Its weights come from ``seed``, so every run sees the same backbone.
    """

    def __init__(
        self,
        n_input_channels: int,
        gain: float = 2.0,
        bias: float = -0.1,
        beta: float = 0.9,
        threshold: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.beta = beta
        self.threshold = threshold
        self.proj = nn.Linear(n_input_channels, FEATURE_DIM)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            # Gain 2 / sqrt(C) gives almost every feature neuron enough drive to fire (0 of 256
            # silent on the default data, 7 on the short test windows);
            # a small negative mean bias keeps the features sparse, like a real spiking layer.
            weight = torch.randn(FEATURE_DIM, n_input_channels, generator=g)
            self.proj.weight.copy_(gain * weight / n_input_channels**0.5)
            self.proj.bias.copy_(bias + 0.1 * torch.randn(FEATURE_DIM, generator=g))

    def extract(self, spikes: Tensor) -> Tensor:
        current = self.proj(spikes)  # [B, T, 256]
        mem = torch.zeros_like(current[:, 0])
        out = []
        for t in range(spikes.shape[1]):
            mem = self.beta * mem + current[:, t]
            spk = (mem > self.threshold).float()
            mem = mem - spk * self.threshold  # subtract reset, as in the adapter
            out.append(spk)
        return torch.stack(out, dim=1)

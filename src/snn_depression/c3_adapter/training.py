"""Supervised training of the adapter on a FROZEN backbone (Milestone 2; reused by Milestone 4).

Only the adapter learns. The backbone is frozen with ``FeatureBackbone.freeze()``, runs under
``torch.no_grad()``, and none of its parameters is ever given to the optimiser.
"""

from __future__ import annotations

import snntorch.functional as SF
import torch
from torch import Tensor

from ..common.contracts import N_CLASSES, FeatureBackbone
from ..common.seed import set_seed
from ..evaluation.metrics import classification_metrics
from .adapter import PersonalSNNAdapter, decide
from .synthetic import SyntheticDataset


def make_loss(cfg: dict):
    """Loss from the ``training`` config section. Called on time-first spikes [T', B, 3].

    mse_count_loss (snnTorch): target spike count = correct_rate × T' for the true class and
    incorrect_rate × T' for the others, compared with MSE. Chosen over cross-entropy on counts
    (identical to snnTorch's ce_count_loss) and ce_rate_loss because:
    - targets are fractions of T', so they mean the same for 2 s or 10 s windows;
    - a non-zero incorrect_rate keeps every output neuron mid-range (neither silent nor saturated),
      which means fewer silent windows and fewer spikes (energy) than "fire as much as possible";
    - the target rates are the firing-rate targets Milestone 4's label-free calibration matches.
    Its loss value grows with T', but Adam scales each step by the gradient's running magnitude,
    so the step size is largely unaffected.
    """
    if cfg["loss"] != "mse_count":
        raise ValueError(f"unsupported loss {cfg['loss']!r}; supported: 'mse_count'")
    return SF.mse_count_loss(correct_rate=cfg["correct_rate"], incorrect_rate=cfg["incorrect_rate"])


def build_optimizer(adapter: PersonalSNNAdapter, cfg: dict) -> torch.optim.Optimizer:
    """Adam over the adapter's parameters ONLY (weights, β, V_th)."""
    return torch.optim.Adam(adapter.parameters(), lr=cfg["lr"])


def check_optimizer_excludes(optimizer: torch.optim.Optimizer, backbone: FeatureBackbone) -> None:
    """Raise if any backbone parameter is in the optimiser (it must stay frozen)."""
    backbone_ids = {id(p) for p in backbone.parameters()}
    for group in optimizer.param_groups:
        if any(id(p) in backbone_ids for p in group["params"]):
            raise ValueError("optimiser contains backbone parameters; only the adapter may train")


def min_threshold(adapter: PersonalSNNAdapter) -> float:
    """Smallest V_th across both LIF layers."""
    return min(adapter.lif1.threshold.min().item(), adapter.lif2.threshold.min().item())


@torch.no_grad()
def compute_features(backbone: FeatureBackbone, spikes: Tensor, batch_size: int) -> Tensor:
    """Run the frozen backbone once: spikes [N, T, C_spk] → features [N, T', 256].

    The backbone is frozen and deterministic, so its features never change during training.
    Computing them once (under no_grad: no autograd graph, less memory) gives exactly the same
    result as re-running it on every batch of every epoch, and is far cheaper. For real data
    that does not fit in memory this becomes a per-batch computation (Milestone 7).
    """
    return torch.cat(
        [backbone(spikes[i : i + batch_size]) for i in range(0, spikes.shape[0], batch_size)]
    )


@torch.no_grad()
def evaluate(
    adapter: PersonalSNNAdapter, features: Tensor, labels: Tensor, loss_fn, batch_size: int
) -> dict:
    """features [N, T', 256], labels [N] → shared metrics (accuracy, macro-F1, confusion matrix, n)
    + mean loss + silent-window count.

    Predictions use the adapter's tie-break / silent rule (``decide`` in adapter.py), so the
    reported accuracy is what ``adapter.predict`` would give.
    """
    adapter.eval()
    preds, silent, loss_sum = [], [], 0.0
    for start in range(0, len(labels), batch_size):
        y = labels[start : start + batch_size]
        # One simulation gives both the loss (from spikes) and the prediction (spikes + membrane).
        spike_record, mem_record = adapter._simulate(features[start : start + batch_size])
        loss_sum += loss_fn(spike_record.transpose(0, 1), y).item() * len(y)
        p, s = decide(spike_record.sum(dim=1), mem_record.amax(dim=1))
        preds.append(p)
        silent.append(s)
    metrics = classification_metrics(labels.numpy(), torch.cat(preds).numpy())
    metrics["loss"] = loss_sum / len(labels)
    metrics["n_silent"] = int(torch.cat(silent).sum())
    return metrics


def train_adapter(
    backbone: FeatureBackbone,
    adapter: PersonalSNNAdapter,
    train_data: SyntheticDataset,
    test_data: SyntheticDataset,
    cfg: dict,
    seed: int | None = None,
) -> dict:
    """Train the adapter on ``train_data`` with the backbone frozen; evaluate after every epoch.

    cfg: the ``training`` config section (epochs, lr, batch_size, seed, loss, correct_rate,
    incorrect_rate). ``seed`` overrides cfg["seed"]. It seeds the batch shuffling;
    build the adapter after ``set_seed`` as well for a fully reproducible run.

    Returns a history dict:
      loss          [epochs + 1]  index 0 = train loss before any update, then mean loss per epoch
      train, test   [epochs]      metrics dicts from ``evaluate`` after each epoch
      min_threshold [epochs]      smallest V_th seen after any update in that epoch
      n_silent_test int           silent test windows after the last epoch
      seed, chance
    """
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed)
    batch_size = cfg["batch_size"]
    eval_batch = max(batch_size, 256)  # no gradients in evaluation, so larger batches are fine

    backbone.freeze()  # requires_grad=False + eval mode
    optimizer = build_optimizer(adapter, cfg)
    check_optimizer_excludes(optimizer, backbone)
    loss_fn = make_loss(cfg)
    shuffle_gen = torch.Generator().manual_seed(seed)

    train_x = compute_features(backbone, train_data.spikes, eval_batch)
    test_x = compute_features(backbone, test_data.spikes, eval_batch)
    train_y, test_y = train_data.labels, test_data.labels

    adapter.constrain_parameters()  # start inside the valid range
    history = {
        "seed": seed,
        "chance": 1.0 / N_CLASSES,
        "loss": [evaluate(adapter, train_x, train_y, loss_fn, eval_batch)["loss"]],
        "train": [],
        "test": [],
        "min_threshold": [],
    }

    for _ in range(cfg["epochs"]):
        adapter.train()
        order = torch.randperm(len(train_y), generator=shuffle_gen)
        loss_sum, lowest_vth = 0.0, float("inf")
        for start in range(0, len(train_y), batch_size):
            idx = order[start : start + batch_size]
            _, spike_record = adapter(train_x[idx])
            # snnTorch losses expect time-first spikes [T', B, 3]; the boundary stays batch-first.
            loss: Tensor = loss_fn(spike_record.transpose(0, 1), train_y[idx])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            adapter.constrain_parameters()  # decision (b): keep V_th ≥ threshold_min, β in [0, 1]

            loss_sum += loss.item() * len(idx)
            lowest_vth = min(lowest_vth, min_threshold(adapter))

        history["loss"].append(loss_sum / len(train_y))
        history["min_threshold"].append(lowest_vth)
        history["train"].append(evaluate(adapter, train_x, train_y, loss_fn, eval_batch))
        history["test"].append(evaluate(adapter, test_x, test_y, loss_fn, eval_batch))

    history["n_silent_test"] = history["test"][-1]["n_silent"] if history["test"] else 0
    return history

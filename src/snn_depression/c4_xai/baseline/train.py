"""Training and evaluation of the LSTM baseline for one subject-wise fold.

Training subjects contribute all their windows. Validation and test subjects contribute
only their evaluation windows (after the first 20% calibration portion), matching the
team's calibration/evaluation rule so results stay comparable with C3.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, recall_score,
                             roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader

from .dataset import WindowDataset
from .features import feature_spec, load_subject_features


def make_datasets(cfg: dict, fold: dict, subjects: pd.DataFrame, fdir: Path) -> dict:
    b = cfg["baseline"]
    spec = feature_spec(cfg)
    labels = dict(zip(subjects.subject_id, subjects.label))
    n_cal = dict(zip(subjects.subject_id, subjects.n_calibration_windows))

    def arrays(split: str, evaluation_only: bool) -> dict[str, np.ndarray]:
        out = {}
        for sid in fold[split]:
            a = load_subject_features(fdir, sid, spec)
            out[sid] = a[:, n_cal[sid]:] if evaluation_only else a
        return out

    train = arrays("train", False)
    # Standardise with training-subject statistics only (no test leakage)
    allx = np.concatenate(list(train.values()), axis=1)
    mean, std = allx.mean(1), allx.std(1) + 1e-6
    seq = b["seq_windows"]
    return {
        "train": WindowDataset(train, labels, seq, b["train_stride_windows"], mean, std),
        "val": WindowDataset(arrays("val", True), labels, seq, b["eval_stride_windows"], mean, std),
        "test": WindowDataset(arrays("test", True), labels, seq, b["eval_stride_windows"], mean, std),
    }


@torch.no_grad()
def predict(model: nn.Module, dataset: WindowDataset, device, batch_size: int = 256) -> pd.DataFrame:
    """Sequence-level predictions: subject_id, label, prob_0..prob_K-1."""
    model.eval()
    probs, labels, subj = [], [], []
    for x, y, si in DataLoader(dataset, batch_size=batch_size):
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        labels.append(y.numpy())
        subj += [dataset.subjects[i] for i in si.tolist()]
    probs = np.concatenate(probs)
    df = pd.DataFrame(probs, columns=[f"prob_{k}" for k in range(probs.shape[1])])
    df.insert(0, "label", np.concatenate(labels))
    df.insert(0, "subject_id", subj)
    return df


def classification_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Binary metrics incl. balanced accuracy, sensitivity, specificity and AUC.

    (The shared evaluation.metrics is fixed to the 3 severity classes.)
    """
    pred = probs.argmax(1)
    out = {"accuracy": accuracy_score(labels, pred),
           "balanced_accuracy": balanced_accuracy_score(labels, pred),
           "macro_f1": f1_score(labels, pred, average="macro")}
    if probs.shape[1] == 2:
        out["sensitivity"] = recall_score(labels, pred, pos_label=1, zero_division=0)
        out["specificity"] = recall_score(labels, pred, pos_label=0, zero_division=0)
        if len(np.unique(labels)) == 2:
            out["auc"] = roc_auc_score(labels, probs[:, 1])
    return {k: float(v) for k, v in out.items()}


def evaluate(pred_df: pd.DataFrame) -> dict:
    """Sequence-level and subject-level (mean probability per subject) metrics."""
    pcols = [c for c in pred_df.columns if c.startswith("prob_")]
    window = classification_metrics(pred_df.label.values, pred_df[pcols].values)
    subj = pred_df.groupby("subject_id")[["label", *pcols]].mean()
    subject = classification_metrics(subj.label.values.astype(int), subj[pcols].values)
    return {"window": window, "subject": subject}


def augment(x: torch.Tensor, t: dict, n_bands: int) -> torch.Tensor:
    """Gaussian noise + whole-electrode dropout, against memorising individual subjects."""
    if t["aug_noise_std"] > 0:
        x = x + t["aug_noise_std"] * torch.randn_like(x)
    if t["aug_channel_dropout"] > 0:
        b, steps, f = x.shape
        keep = torch.rand(b, 1, f // n_bands, 1, device=x.device) >= t["aug_channel_dropout"]
        x = (x.view(b, steps, f // n_bands, n_bands) * keep).view(b, steps, f)
    return x


def train_fold(model: nn.Module, datasets: dict, cfg: dict, device, log=print) -> pd.DataFrame:
    """Train with class weights and early stopping on validation subject-level balanced accuracy."""
    t = cfg["baseline"]["train"]
    n_bands = feature_spec(cfg).n_bands
    n_classes = model.head[-1].out_features
    counts = np.bincount(datasets["train"].window_labels(), minlength=n_classes)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    loader = DataLoader(datasets["train"], batch_size=t["batch_size"], shuffle=True, drop_last=True)

    best_score, best_state, bad_epochs, history = -np.inf, None, 0, []
    for epoch in range(1, t["max_epochs"] + 1):
        model.train()
        total = 0.0
        for x, y, _ in loader:
            x, y = augment(x.to(device), t, n_bands), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        train_loss = total / max(1, len(loader) * t["batch_size"])

        val = evaluate(predict(model, datasets["val"], device))
        score = val["subject"]["balanced_accuracy"] + 0.1 * val["window"]["balanced_accuracy"]
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_window_bacc": val["window"]["balanced_accuracy"],
                        "val_subject_bacc": val["subject"]["balanced_accuracy"]})
        log(f"  epoch {epoch:3d}  loss {train_loss:.4f}  "
            f"val window bAcc {val['window']['balanced_accuracy']:.3f}  "
            f"val subject bAcc {val['subject']['balanced_accuracy']:.3f}")
        if score > best_score:
            best_score, best_state, bad_epochs = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= t["patience"]:
                log(f"  early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return pd.DataFrame(history)

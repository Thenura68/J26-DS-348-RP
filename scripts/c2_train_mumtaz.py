from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.c2_global_snn.backbone import GlobalSNNBackbone
from snn_depression.common.config import load_config
from snn_depression.common.paths import DATA_ROOT, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.data.mumtaz import (
    build_mumtaz_dataset,
    list_mumtaz_recordings,
    load_mumtaz_recording,
    preprocess_mumtaz_recording,
    encode_mumtaz_windows,
)
from snn_depression.data.splits import stratified_subject_kfold


def get_or_build_mumtaz_dataset(condition: str = "EC", k: float = 1.0) -> dict[str, Any]:
    # Load cached preprocessed & TBR encoded MUMTAZ dataset or build and cache it
    interim_dir = DATA_ROOT / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    cache_path = interim_dir / f"mumtaz_encoded_{condition}_k{k:.1f}.pt"

    if cache_path.exists():
        print(f"Loading pre-encoded MUMTAZ dataset from cache: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print(f"Preprocessing and encoding MUMTAZ ({condition}) dataset with Adaptive TBR...")
    dataset = build_mumtaz_dataset(condition=condition, k=k)
    torch.save(dataset, cache_path)
    print(f"Saved encoded MUMTAZ dataset to cache: {cache_path}")
    return dataset


def train_one_fold(
    train_spikes: list[torch.Tensor],
    train_labels: list[int],
    test_spikes: list[torch.Tensor],
    test_labels: list[int],
    ablation_mode: str,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    reservoir_size: int = 128,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, Any], GlobalSNNBackbone]:
    # Train and evaluate backbone on one cross-validation fold
    # Concatenate windows for training
    x_train = torch.cat(train_spikes, dim=0).to(device)  # [N_train, T, 38]
    y_train = torch.tensor(
        [label for spk, label in zip(train_spikes, train_labels) for _ in range(len(spk))],
        dtype=torch.long,
        device=device,
    )

    x_test = torch.cat(test_spikes, dim=0).to(device)  # [N_test, T, 38]
    y_test = torch.tensor(
        [label for spk, label in zip(test_spikes, test_labels) for _ in range(len(spk))],
        dtype=torch.long,
        device=device,
    )

    c_spk = x_train.shape[2]
    backbone = GlobalSNNBackbone(
        n_input_channels=c_spk,
        reservoir_size=reservoir_size,
        ablation_mode=ablation_mode,
        n_classes=2,
    ).to(device)

    # Unsupervised STDP pre-training
    if backbone.stdp_layer is not None:
        backbone.eval()
        for i in range(0, min(len(x_train), 64), batch_size):
            batch_spk = x_train[i : i + batch_size]
            backbone.fit_stdp(batch_spk)

    # Supervised training with SuperSpike surrogate gradients
    optimizer = torch.optim.Adam(
        [p for p in backbone.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    n_samples = len(x_train)
    backbone.train()

    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            bx = x_train[batch_idx]
            by = y_train[batch_idx]

            optimizer.zero_grad()
            logits, _, aux = backbone.forward_classify(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            # Enforce parameters within physiological bounds
            backbone.constrain_parameters()

            # Apply Intrinsic Plasticity update
            if backbone.use_ip:
                window_duration = bx.shape[1] * (backbone.dt_ms / 1000.0)
                backbone.apply_intrinsic_plasticity(aux["res_spikes"], window_duration)

    # Evaluation on held-out test subjects
    backbone.eval()
    test_preds = []
    test_confs = []
    total_synops = 0
    total_sparsity = 0.0

    with torch.no_grad():
        for i in range(0, len(x_test), batch_size):
            bx = x_test[i : i + batch_size]
            _, preds, aux = backbone.forward_classify(bx)
            test_preds.append(preds.cpu())
            test_confs.append(aux["confidence_pct"].cpu())
            total_synops += aux.get("total_synops", 0)
            total_sparsity += aux.get("spike_sparsity", 0.0) * len(bx)

    y_pred_np = torch.cat(test_preds).numpy()
    y_test_np = y_test.cpu().numpy()
    conf_np = torch.cat(test_confs).numpy()

    acc = float(accuracy_score(y_test_np, y_pred_np))
    recall = float(recall_score(y_test_np, y_pred_np, average="binary", zero_division=0))
    f1 = float(f1_score(y_test_np, y_pred_np, average="macro", zero_division=0))
    cm = confusion_matrix(y_test_np, y_pred_np).tolist()

    metrics = {
        "accuracy": acc,
        "recall_sensitivity": recall,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "mean_confidence": float(np.mean(conf_np)),
        "mean_synops_per_window": float(total_synops / max(1, len(x_test))),
        "spike_sparsity": float(total_sparsity / max(1, len(x_test))),
        "test_windows": len(x_test),
    }
    return metrics, backbone


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate C2 on MUMTAZ dataset")
    parser.add_argument("--condition", default="EC", choices=["EC", "EO", "ALL"], help="EEG condition")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--reservoir-size", type=int, default=128, help="Reservoir neuron count")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit subjects for rapid benchmarking")
    parser.add_argument(
        "--ablation",
        default="full_engine",
        choices=["all", "reservoir_lif", "reservoir_alif", "reservoir_alif_stdp", "full_engine"],
        help="Ablation mode",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting C2 Training on MUMTAZ (Device: {device}, Condition: {args.condition}, Epochs: {args.epochs}, Batch: {args.batch_size})")

    # Load or build cached MUMTAZ dataset
    dataset = get_or_build_mumtaz_dataset(condition=args.condition, k=1.0)
    subjects = dataset["subjects"]
    labels = dataset["labels"]

    if args.max_subjects is not None and args.max_subjects < len(subjects):
        subjects = subjects[:args.max_subjects]

    sub_labels = [labels[s] for s in subjects]
    print(f"Cohort: {len(subjects)} subjects ({sub_labels.count(1)} MDD, {sub_labels.count(0)} HC)")

    # Stratified 5-Fold cross-validation
    folds = stratified_subject_kfold(subjects, sub_labels, n_splits=5, n_repeats=1, seed=args.seed)

    modes_to_test = (
        ["reservoir_lif", "reservoir_alif", "reservoir_alif_stdp", "full_engine"]
        if args.ablation == "all"
        else [args.ablation]
    )

    results_by_mode: dict[str, list[dict[str, Any]]] = {m: [] for m in modes_to_test}
    best_backbone = None
    best_f1 = -1.0

    out_dir = OUTPUT_ROOT / "c2"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in modes_to_test:
        print(f"\n{'='*25} Running Ablation: {mode} {'='*25}")
        fold_metrics = []

        for f_idx, fold in enumerate(folds, 1):
            train_sub_ids = fold["train_subject_ids"]
            test_sub_ids = fold["test_subject_ids"]

            train_spk = [dataset["all_spikes"][sid] for sid in train_sub_ids if sid in dataset["all_spikes"]]
            train_lbl = [labels[sid] for sid in train_sub_ids if sid in dataset["all_spikes"]]

            # Test subjects use only their evaluation windows 
            test_spk = [dataset["eval_spikes"][sid] for sid in test_sub_ids if sid in dataset["eval_spikes"]]
            test_lbl = [labels[sid] for sid in test_sub_ids if sid in dataset["eval_spikes"]]

            res, model = train_one_fold(
                train_spikes=train_spk,
                train_labels=train_lbl,
                test_spikes=test_spk,
                test_labels=test_lbl,
                ablation_mode=mode,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                reservoir_size=args.reservoir_size,
                device=device,
            )
            fold_metrics.append(res)
            print(
                f"Fold {f_idx}/5 -> Acc: {res['accuracy']*100:.2f}%, "
                f"Recall: {res['recall_sensitivity']*100:.2f}%, "
                f"Macro-F1: {res['macro_f1']:.3f}, "
                f"SynOps/win: {res['mean_synops_per_window']:.0f}, "
                f"Sparsity: {res['spike_sparsity']*100:.1f}%"
            )

            if res["macro_f1"] > best_f1:
                best_f1 = res["macro_f1"]
                best_backbone = model

        results_by_mode[mode] = fold_metrics

    # Print Benchmark Summary Table
    print("MUMTAZ C2 BENCHMARK SUMMARY (Stratified 5-Fold Cross Validation)")
    print(f"{'Ablation Mode':<25} | {'Accuracy':<10} | {'Recall':<10} | {'Macro-F1':<10} | {'SynOps/win':<12} | {'Sparsity':<10}")

    summary_export = {}
    for mode, m_list in results_by_mode.items():
        mean_acc = np.mean([m["accuracy"] for m in m_list])
        mean_rec = np.mean([m["recall_sensitivity"] for m in m_list])
        mean_f1 = np.mean([m["macro_f1"] for m in m_list])
        mean_synops = np.mean([m["mean_synops_per_window"] for m in m_list])
        mean_spar = np.mean([m["spike_sparsity"] for m in m_list])

        print(f"{mode:<25} | {mean_acc*100:>8.2f}% | {mean_rec*100:>8.2f}% | {mean_f1:>10.3f} | {mean_synops:>12.0f} | {mean_spar*100:>8.1f}%")

        summary_export[mode] = {
            "mean_accuracy": float(mean_acc),
            "mean_recall": float(mean_rec),
            "mean_macro_f1": float(mean_f1),
            "mean_synops_per_window": float(mean_synops),
            "mean_spike_sparsity": float(mean_spar),
            "folds": m_list,
        }

    # Save results and best checkpoint
    report_file = out_dir / f"mumtaz_benchmark_{args.condition}_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(summary_export, f, indent=2)
    print(f"\nFull benchmark report saved to: {report_file}")

    if best_backbone is not None:
        ckpt_file = out_dir / f"global_backbone_mumtaz_best.pt"
        torch.save(best_backbone.state_dict(), ckpt_file)
        print(f"Best model checkpoint saved to: {ckpt_file} (Peak Macro-F1: {best_f1:.3f})")


if __name__ == "__main__":
    main()

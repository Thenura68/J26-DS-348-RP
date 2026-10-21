"""Train the C4 LSTM baseline on every subject-wise fold.

    python scripts/c4/train_lstm.py [--folds 0 1] [--set baseline.model.hidden=32]

Writes outputs/c4/<output_subdir>/fold<k>/{model.pt, history.csv, test_predictions.csv,
metrics.json} and cv_summary.json (mean ± std over folds). Checkpoints are never committed.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from snn_depression.c4_xai.baseline.features import (CLASS_NAMES, feature_spec, features_dir,
                                                     read_subjects)
from snn_depression.c4_xai.baseline.lstm import build_model
from snn_depression.c4_xai.baseline.train import evaluate, make_datasets, predict, train_fold
from snn_depression.c4_xai.baseline.utils import apply_overrides, get_device
from snn_depression.common.config import load_config
from snn_depression.common.paths import component_output_dir
from snn_depression.common.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c4_xai")
    ap.add_argument("--folds", type=int, nargs="*", help="fold indices (default: all)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)

    fdir = features_dir(cfg)
    subjects = read_subjects(fdir)
    folds = json.load(open(fdir / "splits.json"))["folds"]
    folds = [f for f in folds if args.folds is None or f["fold"] in args.folds]
    device = get_device(cfg["baseline"]["train"]["device"])
    n_features = feature_spec(cfg).n_features
    out_root = component_output_dir("c4") / cfg["baseline"]["output_subdir"]

    results = []
    for fold in folds:
        k = fold["fold"]
        print(f"Fold {k}: {len(fold['train'])} train / {len(fold['val'])} val / "
              f"{len(fold['test'])} test subjects (device {device})")
        set_seed(cfg["project"]["seed"] + k)
        datasets = make_datasets(cfg, fold, subjects, fdir)
        model = build_model(cfg, n_features, len(CLASS_NAMES)).to(device)
        history = train_fold(model, datasets, cfg, device)

        out = out_root / f"fold{k}"
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "fold": fold,
                    "n_features": n_features, "n_classes": len(CLASS_NAMES)}, out / "model.pt")
        history.to_csv(out / "history.csv", index=False)
        preds = predict(model, datasets["test"], device)
        preds.to_csv(out / "test_predictions.csv", index=False)
        metrics = evaluate(preds)
        json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
        print(f"  TEST subject: " + "  ".join(f"{m} {v:.3f}" for m, v in metrics["subject"].items()))
        results.append(metrics)

    if len(results) > 1:
        summary = {lvl: {m: {"mean": float(np.mean([r[lvl][m] for r in results])),
                             "std": float(np.std([r[lvl][m] for r in results]))}
                         for m in results[0][lvl]}
                   for lvl in ("window", "subject")}
        json.dump(summary, open(out_root / "cv_summary.json", "w"), indent=2)
        print("\nCross-validation (mean ± std over folds)")
        for lvl, ms in summary.items():
            print(f"  {lvl:8s} " + "  ".join(f"{m} {v['mean']:.3f}±{v['std']:.3f}" for m, v in ms.items()))


if __name__ == "__main__":
    main()

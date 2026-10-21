"""SHAP attributions + explanation figures for held-out test subjects of every fold.

    python scripts/c4/explain_shap.py [--folds 0]

Writes outputs/c4/<output_subdir>/fold<k>/shap_attributions.npz  (attr [N, F, T], x [N, T, F],
subject_id, label, target, prob, background), fold<k>/figures/subject_<id>.png and
figures/group_scalp_maps.png.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from snn_depression.c4_xai.aggregate import channel_scores
from snn_depression.c4_xai.baseline.features import (CLASS_NAMES, feature_spec, features_dir,
                                                     read_subjects)
from snn_depression.c4_xai.baseline.lstm import load_model
from snn_depression.c4_xai.baseline.shap_explainer import LSTMShapExplainer, make_predict_fn
from snn_depression.c4_xai.baseline.train import make_datasets
from snn_depression.c4_xai.baseline.utils import apply_overrides
from snn_depression.c4_xai.scalp_map import explanation_figure, plot_scalp_map
from snn_depression.common.config import load_config
from snn_depression.common.paths import component_output_dir
from snn_depression.common.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c4_xai")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    sc = cfg["shap"]
    spec = feature_spec(cfg)
    fdir = features_dir(cfg)
    subjects = read_subjects(fdir)
    out_root = component_output_dir("c4") / cfg["baseline"]["output_subdir"]

    fold_dirs = sorted(out_root.glob("fold*"))
    if args.folds is not None:
        fold_dirs = [d for d in fold_dirs if int(d.name[4:]) in args.folds]
    if not fold_dirs:
        raise SystemExit("No trained folds - run scripts/c4/train_lstm.py first")

    all_attr, all_target = [], []
    for fdir_k in fold_dirs:
        model, ckpt = load_model(fdir_k / "model.pt")
        fold = ckpt["fold"]
        seed = cfg["project"]["seed"] + fold["fold"]
        set_seed(seed)
        rng = np.random.default_rng(seed)
        ds = make_datasets(cfg, fold, subjects, fdir)

        bg_idx = rng.choice(len(ds["train"]), min(sc["n_background"], len(ds["train"])), replace=False)
        background = np.stack([ds["train"].window(i) for i in bg_idx])
        explainer = LSTMShapExplainer(model, background, sc["nsamples"])

        test, sel = ds["test"], []
        for sid in fold["test"]:            # evenly spaced sequences per held-out subject
            idx = [i for i, s in enumerate(test.window_subjects()) if s == sid]
            if idx:
                n = min(sc["windows_per_subject"], len(idx))
                sel += [idx[j] for j in np.linspace(0, len(idx) - 1, n).astype(int)]
        x = np.stack([test.window(i) for i in sel])
        sid = np.array([test.window_subjects()[i] for i in sel])
        label = np.array([test.labels[s] for s in sid])
        probs = make_predict_fn(model)(x)
        target = probs.argmax(1)           # explain what the model actually predicted

        attr = np.concatenate([explainer(x[i:i + 8], target[i:i + 8])
                               for i in tqdm(range(0, len(x), 8), desc=f"SHAP {fdir_k.name}")])
        np.savez_compressed(fdir_k / "shap_attributions.npz", attr=attr, x=x, subject_id=sid,
                            label=label, target=target, prob=probs[np.arange(len(x)), target],
                            background=background)

        fig_dir = fdir_k / "figures"
        fig_dir.mkdir(exist_ok=True)
        for s in np.unique(sid):
            rows = np.where(sid == s)[0]
            i = int(rows[len(rows) // 2])
            header = (f"Subject {s} (true {CLASS_NAMES[label[i]]}) - predicted {CLASS_NAMES[target[i]]} "
                      f"({probs[i, target[i]]:.0%}) - LSTM + SHAP baseline")
            fig = explanation_figure(attr[i], spec.channel_names, spec.band_names, spec.step_s, header)
            fig.savefig(fig_dir / f"subject_{s}.png", dpi=110)
            plt.close(fig)
        all_attr.append(attr)
        all_target.append(target)

    attr, target = np.concatenate(all_attr), np.concatenate(all_target)
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(5 * len(CLASS_NAMES), 4.5))
    for c, ax in enumerate(axes):
        mask = target == c
        if not mask.any():
            ax.set_axis_off()
            continue
        s = np.stack([channel_scores(a, "abs", spec.n_bands) for a in attr[mask]])
        plot_scalp_map((s / s.sum(1, keepdims=True)).mean(0), spec.channel_names,
                       f"Predicted {CLASS_NAMES[c]} (n={mask.sum()})", ax=ax)
    fig.suptitle("LSTM + SHAP: mean attribution across held-out subjects")
    (out_root / "figures").mkdir(exist_ok=True)
    fig.savefig(out_root / "figures" / "group_scalp_maps.png", dpi=120, bbox_inches="tight")
    print(f"Saved attributions and figures under {out_root}")


if __name__ == "__main__":
    main()

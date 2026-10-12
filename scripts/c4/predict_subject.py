"""Predict (and optionally explain) ONE MODMA subject with the fold model that never saw them.

    python scripts/c4/predict_subject.py 02010005 --explain

Uses the subject's evaluation windows (after the 20% calibration portion), cut into
sequences; each sequence gets P(MDD) and the subject's result is the mean.
With --explain, SHAP attributions are averaged over the subject's sequences into one figure.
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from snn_depression.c4_xai.baseline.features import (CLASS_NAMES, feature_spec, features_dir,
                                                     read_subjects)
from snn_depression.c4_xai.baseline.lstm import load_model
from snn_depression.c4_xai.baseline.shap_explainer import LSTMShapExplainer, make_predict_fn
from snn_depression.c4_xai.baseline.train import make_datasets
from snn_depression.c4_xai.baseline.utils import apply_overrides
from snn_depression.c4_xai.scalp_map import explanation_figure
from snn_depression.common.config import load_config
from snn_depression.common.paths import component_output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="MODMA subject id, e.g. 02010005")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--config", default="c4_xai")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    sid = args.subject.zfill(8)

    spec = feature_spec(cfg)
    fdir = features_dir(cfg)
    subjects = read_subjects(fdir)
    out_root = component_output_dir("c4") / cfg["baseline"]["output_subdir"]
    folds = json.load(open(fdir / "splits.json"))["folds"]
    fold = next((f for f in folds if sid in f["test"]), None)
    if fold is None:
        raise SystemExit(f"Subject {sid} is not in the dataset")
    row = subjects[subjects.subject_id == sid].iloc[0]
    truth = CLASS_NAMES[int(row.label)]
    print(f"Subject {sid}: true {truth} (PHQ-9 {row.phq9}); fold {fold['fold']} model never saw them")

    model, _ = load_model(out_root / f"fold{fold['fold']}" / "model.pt")
    ds = make_datasets(cfg, fold, subjects, fdir)
    test = ds["test"]
    idx = [i for i, s in enumerate(test.window_subjects()) if s == sid]
    x = np.stack([test.window(i) for i in idx])
    p = make_predict_fn(model)(x)
    overall = p.mean(0)
    verdict = CLASS_NAMES[int(overall.argmax())]
    print(f"{len(p)} sequences of {cfg['baseline']['seq_windows'] * spec.step_s:g} s. P(MDD):")
    print("  " + "  ".join(f"{v:.2f}" for v in p[:, 1]))
    print(f"\nRESULT: {verdict} (P(MDD) = {overall[1]:.2f})   truth: {truth} -> "
          f"{'CORRECT' if verdict == truth else 'WRONG'}")

    if args.explain:
        rng = np.random.default_rng(cfg["project"]["seed"])
        bg = np.stack([ds["train"].window(i) for i in
                       rng.choice(len(ds["train"]), cfg["shap"]["n_background"], replace=False)])
        target = int(overall.argmax())
        attr = LSTMShapExplainer(model, bg, cfg["shap"]["nsamples"])(x, [target] * len(x)).mean(0)
        fig = explanation_figure(attr, spec.channel_names, spec.band_names, spec.step_s,
                                 f"{sid}: predicted {verdict} (P(MDD) {overall[1]:.0%}), truth {truth}"
                                 " - mean SHAP over all sequences")
        path = out_root / "figures" / f"subject_{sid}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"Explanation figure: {path}")


if __name__ == "__main__":
    main()

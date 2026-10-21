"""Score saved SHAP attributions: faithfulness, sufficiency, stability, consistency, concordance.

    python scripts/c4/evaluate_xai.py

Writes outputs/c4/<output_subdir>/xai_masking.csv, xai_per_window.csv and xai_summary.json
(means and one-sided Wilcoxon tests of top-k against random-k masking).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tqdm import tqdm

from snn_depression.c4_xai.aggregate import channel_scores
from snn_depression.c4_xai.baseline.features import feature_spec
from snn_depression.c4_xai.baseline.lstm import load_model
from snn_depression.c4_xai.baseline.shap_explainer import LSTMShapExplainer, make_predict_fn
from snn_depression.c4_xai.baseline.utils import apply_overrides
from snn_depression.c4_xai.metrics import (cross_window_consistency, frontal_concordance,
                                           masking_curves, stability)
from snn_depression.c4_xai.montage import region_channels
from snn_depression.common.config import load_config
from snn_depression.common.paths import component_output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c4_xai")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    ec = cfg["evaluation"]
    spec = feature_spec(cfg)
    nb = spec.n_bands
    out_root = component_output_dir("c4") / cfg["baseline"]["output_subdir"]
    frontal = spec.local_index(region_channels(("F3", "F4")))
    rng = np.random.default_rng(cfg["project"]["seed"])

    fold_dirs = [d for d in sorted(out_root.glob("fold*")) if (d / "shap_attributions.npz").exists()]
    if not fold_dirs:
        raise SystemExit("No shap_attributions.npz - run scripts/c4/explain_shap.py first")
    n_stab = max(1, ec["stability_max_windows"] // len(fold_dirs))

    masking, per_window, consistency = [], [], []
    for fdir_k in fold_dirs:
        d = np.load(fdir_k / "shap_attributions.npz")
        model, _ = load_model(fdir_k / "model.pt")
        predict_fn = make_predict_fn(model)
        # Same background as the saved explanations, so stability re-runs the same explainer
        explainer = LSTMShapExplainer(model, d["background"], cfg["shap"]["nsamples"])
        stab_idx = set(rng.choice(len(d["x"]), min(n_stab, len(d["x"])), replace=False).tolist())

        for i in tqdm(range(len(d["x"])), desc=f"metrics {fdir_k.name}"):
            x, a, t = d["x"][i], d["attr"][i], int(d["target"][i])
            common = {"fold": fdir_k.name, "window": i, "subject_id": str(d["subject_id"][i]),
                      "label": int(d["label"][i]), "target": t, "correct": int(d["label"][i]) == t}
            for row in masking_curves(predict_fn, x, a, t, ec["top_k_channels"], "channel",
                                      n_bands=nb, n_random=ec["n_random_masks"], rng=rng):
                masking.append({**common, **row})
            for row in masking_curves(predict_fn, x, a, t, ec["top_k_time_bins"], "time", 1,
                                      n_bands=nb, n_random=ec["n_random_masks"], rng=rng):
                masking.append({**common, **row})
            w = {**common, **frontal_concordance(a, frontal, nb, ec["concordance_top_k"])}
            if i in stab_idx:
                w["stability"] = stability(explainer, x, t, ec["stability_repeats"],
                                           ec["stability_noise_std"], nb, rng)
            per_window.append(w)

        for s in np.unique(d["subject_id"]):
            consistency.append(cross_window_consistency(
                [channel_scores(a, "abs", nb) for a in d["attr"][d["subject_id"] == s]]))

    masking, per_window = pd.DataFrame(masking), pd.DataFrame(per_window)
    masking.to_csv(out_root / "xai_masking.csv", index=False)
    per_window.to_csv(out_root / "xai_per_window.csv", index=False)

    summary = {"frontal_channels": [spec.channel_names[i] for i in frontal], "masking": [],
               "concordance": {}}
    for (unit, k), g in masking.groupby(["unit", "k"]):
        summary["masking"].append({
            "unit": unit, "k": int(k), "n": len(g),
            "faithfulness": g.faithfulness.mean(), "faithfulness_random": g.faithfulness_random.mean(),
            "faithfulness_p": wilcoxon(g.faithfulness, g.faithfulness_random, alternative="greater").pvalue,
            "sufficiency": g.sufficiency.mean(), "sufficiency_random": g.sufficiency_random.mean(),
            "sufficiency_p": wilcoxon(g.sufficiency, g.sufficiency_random, alternative="greater").pvalue,
        })
    stab = per_window["stability"].dropna() if "stability" in per_window else pd.Series(dtype=float)
    summary["stability"] = {"n": len(stab), "mean": stab.mean(), "frac_ge_0.7": (stab >= 0.7).mean()}
    summary["consistency"] = {"n_subjects": len(consistency), "mean": float(np.nanmean(consistency))}
    for t, g in per_window.groupby("target"):
        summary["concordance"][f"predicted_class_{t}"] = {
            "n": len(g), "mean_frontal_ratio": g.frontal_ratio.mean(),
            "frac_frontal_ratio_gt_1": (g.frontal_ratio > 1).mean(),
            f"frac_frontal_in_top{ec['concordance_top_k']}": g.frontal_in_top_k.mean()}
    json.dump(summary, open(out_root / "xai_summary.json", "w"), indent=2, default=float)

    print(pd.DataFrame(summary["masking"]).round(3).to_string(index=False))
    print("stability:", {k: round(float(v), 3) for k, v in summary["stability"].items()})
    print("cross-window consistency:", round(summary["consistency"]["mean"], 3))
    print("concordance:", json.dumps(summary["concordance"], default=lambda v: round(float(v), 3)))


if __name__ == "__main__":
    main()

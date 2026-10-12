"""Did the explainer find the planted signal? Compares a planted run against its control.

    python scripts/c4/check_synthetic.py alpha_F3_x1 control

For each run: classification result, the planted electrode's rank, how often it is the top-1
electrode, the alpha band's share and the planted feature's (electrode × alpha) share, plus
mean scalp maps side by side (outputs/c4/synthetic/planted_signal_check.png).
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from snn_depression.c4_xai.aggregate import channel_band_scores
from snn_depression.c4_xai.baseline.features import feature_spec
from snn_depression.c4_xai.montage import channel_index, display_name
from snn_depression.c4_xai.scalp_map import plot_band_shares, plot_scalp_map
from snn_depression.common.config import load_config
from snn_depression.common.paths import PROCESSED_DIR, component_output_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    spec = feature_spec(load_config("c4_xai"))
    nb, alpha = spec.n_bands, spec.band_names.index("alpha")
    out_dir = component_output_dir("c4") / "synthetic"

    fig, axes = plt.subplots(2, len(args.runs), figsize=(6 * len(args.runs), 9), squeeze=False)
    for col, name in enumerate(args.runs):
        info = json.load(open(PROCESSED_DIR / "c4_synthetic" / name / "splits.json"))["synthetic"]
        planted = spec.local_index([channel_index(info["electrode"])])[0]
        run = out_dir / name
        cv = json.load(open(run / "cv_summary.json"))["subject"]
        attrs = np.concatenate([np.load(f)["attr"] for f in sorted(run.glob("fold*/shap_attributions.npz"))])

        cb = np.stack([channel_band_scores(a, nb, "abs") for a in attrs])        # [N, C, B]
        cb = cb / cb.sum(axis=(1, 2), keepdims=True)
        elec, bands = cb.sum(2).mean(0), cb.sum(1).mean(0)
        top1 = (cb.sum(2).argmax(1) == planted).mean()
        rank = int((elec > elec[planted]).sum()) + 1

        print(f"\n=== {name} (planted: {info['electrode']} alpha, strength {info['strength']})")
        print(f"  model: subject bAcc {cv['balanced_accuracy']['mean']:.2f} ± "
              f"{cv['balanced_accuracy']['std']:.2f}   AUC {cv['auc']['mean']:.2f}")
        print(f"  {info['electrode']} rank {rank}/{spec.n_channels}; top electrode in "
              f"{top1 * 100:.0f}% of {len(attrs)} windows (chance {100 / spec.n_channels:.0f}%)")
        print(f"  alpha share {bands[alpha] * 100:.1f}% (fair 20%); {info['electrode']}-alpha share "
              f"{cb[:, planted, alpha].mean() * 100:.1f}% (fair {100 / spec.n_features:.1f}%)")
        print("  top 3:", ", ".join(f"{display_name(spec.channel_names[i])} {elec[i] * 100:.1f}%"
                                    for i in np.argsort(-elec)[:3]))

        plot_scalp_map(elec, spec.channel_names,
                       f"{name}\nbAcc {cv['balanced_accuracy']['mean']:.2f}, "
                       f"{info['electrode']} rank {rank}", ax=axes[0, col])
        plot_band_shares(bands, spec.band_names, axes[1, col])

    fig.suptitle("Planted-signal check: mean SHAP attribution over all explained windows")
    fig.tight_layout()
    fig.savefig(out_dir / "planted_signal_check.png", dpi=100)
    print(f"\nFigure: {out_dir / 'planted_signal_check.png'}")


if __name__ == "__main__":
    main()

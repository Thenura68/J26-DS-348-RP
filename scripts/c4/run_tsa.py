"""TSA on MODMA: practice STDP SNN, subject-wise CV, explanation metrics and planted-signal check.

Pipeline (C1 and C2 are still stubs, so their C4 stand-ins are used):
    C1 preprocessed windows → temporary_encoder (adaptive delta, ON/OFF) → PracticeBackbone (STDP)
    → PracticeReadout (HC/MDD) → TemporalSpikeAttribution in 4 versions (S/NS × paper/leak_matched)

    python scripts/c4/run_tsa.py --name real
    python scripts/c4/run_tsa.py --name planted_F3 --plant F3
    python scripts/c4/run_tsa.py --name control --plant F3 --strength 0
    python scripts/c4/run_tsa.py --name planted_F3_all --plant F3 --plant-where all

Planted mode: real labels are replaced by a random permutation (no real signal left) and a
9-11 Hz alpha wave is added at one electrode for every subject labelled 1, so the correct
explanation is known. ``--plant-where eval`` (default) adds it only after the calibration
portion; with ``all`` the patient-adaptive thresholds absorb most of it (see the C4 README).

Masking metrics set spikes to 0. Explanations use ``shap.windows_per_subject`` windows per test
subject, the same number as the SHAP baseline. Results go to outputs/c4/tsa/<name>/.
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tqdm import tqdm

from snn_depression.c4_xai.aggregate import channel_scores, time_scores
from snn_depression.c4_xai.baseline.features import feature_spec, subject_folds
from snn_depression.c4_xai.baseline.utils import apply_overrides
from snn_depression.c4_xai.metrics import frontal_concordance, masking_curves
from snn_depression.c4_xai.montage import channel_index, region_channels
from snn_depression.c4_xai.practice_snn import PracticeBackbone, PracticeReadout
from snn_depression.c4_xai.scalp_map import plot_scalp_map, plot_timeline, plot_top_channels
from snn_depression.c4_xai.temporary_encoder import downsample, encode, fit_thresholds
from snn_depression.c4_xai.tsa import TemporalSpikeAttribution
from snn_depression.common.config import load_config
from snn_depression.common.paths import INTERIM_DIR, component_output_dir
from snn_depression.common.seed import set_seed

C1_DIR = INTERIM_DIR / "modma_preprocessed"
FS = 250.0
DOWNSAMPLE = 2
STEP_S = DOWNSAMPLE / FS                 # one SNN time step = 8 ms
TIME_BIN = 25                            # time-masking unit: 25 steps = 0.2 s
VERSIONS = [("s", "paper"), ("s", "leak_matched"), ("ns", "paper"), ("ns", "leak_matched")]
MAIN_VERSION = ("ns", "leak_matched")    # used for the per-subject figures


# ── data ─────────────────────────────────────────────────────────────────────────────

def plant_alpha(windows: np.ndarray, ch: int, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Add a 9-11 Hz sine of the given RMS to electrode ``ch`` of windows [N, C, S]."""
    out = windows.copy()
    t = np.arange(windows.shape[-1]) / FS
    freq = rng.uniform(9, 11)
    for i in range(len(out)):
        out[i, ch] += rms * np.sqrt(2) * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return out


def load_subjects(spec, args, label_seed: int, k: float) -> tuple[dict, np.ndarray, np.ndarray]:
    """Encode every subject's evaluation windows. Returns ({id: data}, ids, labels)."""
    meta = pd.read_csv(C1_DIR / "metadata.csv", dtype={"subject_id": str})
    ids = meta.subject_id.to_numpy()
    labels = meta.diagnosis_target.to_numpy().astype(int)
    rng = np.random.default_rng(label_seed)
    if args.plant:
        labels = rng.permutation(labels)       # random labels, same class balance
        plant_ch = spec.local_index([channel_index(args.plant)])[0]
    data = {}
    for sid, y in tqdm(list(zip(ids, labels)), desc="encode"):
        cal = np.load(C1_DIR / f"{sid}_calibration.npy")[:, spec.channels]
        ev = np.load(C1_DIR / f"{sid}_evaluation.npy")[:, spec.channels]
        if args.plant and y == 1 and args.strength > 0:
            rms = args.strength * cal[:, plant_ch].std()
            ev = plant_alpha(ev, plant_ch, rms, rng)
            if args.plant_where == "all":
                cal = plant_alpha(cal, plant_ch, rms, rng)
        thr = fit_thresholds(downsample(cal, DOWNSAMPLE), k)
        eeg = downsample(ev, DOWNSAMPLE).astype(np.float32)          # kept for stability re-encoding
        data[sid] = {"eeg": eeg, "thr": thr, "spikes": encode(eeg, thr), "label": int(y)}
    return data, ids, labels


def to_interleaved(x: np.ndarray) -> np.ndarray:
    """[..., T, 2C] ON-first (contract) → electrode-major [ON_0, OFF_0, ON_1, …] for metrics.py."""
    c = x.shape[-1] // 2
    return np.stack([x[..., :c], x[..., c:]], -1).reshape(*x.shape[:-1], 2 * c)


def from_interleaved(x: np.ndarray) -> np.ndarray:
    c = x.shape[-1] // 2
    xr = x.reshape(*x.shape[:-1], c, 2)
    return np.concatenate([xr[..., 0], xr[..., 1]], -1)


# ── model ────────────────────────────────────────────────────────────────────────────

def hidden_counts(bb: PracticeBackbone, x: torch.Tensor, chunk: int = 256) -> np.ndarray:
    """Hidden spike counts per window [N, 256] — the readout's input."""
    return torch.cat([bb(c).sum(1) for c in x.split(chunk)]).numpy()


def make_predict(bb: PracticeBackbone, ro: PracticeReadout, chunk: int = 256):
    """Spikes [N, T, 2C] (contract order) → class probabilities [N, K] as numpy."""
    def predict(x) -> np.ndarray:
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return torch.cat([ro(bb(c))["probs"] for c in x.split(chunk)]).numpy()
    return predict


def train_fold(data: dict, train_ids, epochs: int, seed: int):
    x = torch.cat([data[s]["spikes"] for s in train_ids])
    y = np.concatenate([[data[s]["label"]] * len(data[s]["spikes"]) for s in train_ids])
    g = torch.Generator().manual_seed(seed)
    bb = PracticeBackbone(n_inputs=x.shape[-1], seed=seed)
    bb.train_stdp(x[torch.randperm(len(x), generator=g)].split(64), epochs=epochs)
    ro = PracticeReadout().fit(hidden_counts(bb, x), y)
    return bb, ro


def example_figure(attr: np.ndarray, names: list[str], header: str):
    """One explained window, attr [C, T] for the predicted class: size map, signed map
    (red = for the prediction, blue = against), top electrodes, timeline in 0.2 s bins."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    plot_scalp_map(channel_scores(attr, "abs"), names, "Size of attribution", ax=axes[0])
    plot_scalp_map(channel_scores(attr, "signed"), names, "Evidence for (+) / against (−) prediction",
                   ax=axes[1], signed=True)
    plot_top_channels(channel_scores(attr, "signed"), names, axes[2],
                      title="Top electrodes (net evidence for prediction)")
    plot_timeline(time_scores(attr, TIME_BIN, "abs"), TIME_BIN * STEP_S, axes[3])
    fig.suptitle(header)
    fig.tight_layout()
    return fig


# ── explanation metrics ──────────────────────────────────────────────────────────────

def tsa_stability(tsa: TemporalSpikeAttribution, eeg: np.ndarray, thr: np.ndarray, target: int,
                  repeats: int, noise_std: float, rng: np.random.Generator) -> float:
    """Noise is added to the EEG (spikes must stay 0/1), then re-encoded and re-explained.
    Mean correlation of electrode scores with the clean window's."""
    batch = np.stack([eeg] + [eeg + rng.normal(0, noise_std, eeg.shape).astype(np.float32)
                              for _ in range(repeats)])
    attr = tsa.attribute(encode(batch, thr))[:, target].numpy()          # [R+1, C, T]
    ref = channel_scores(attr[0], "signed")
    return float(np.mean([np.corrcoef(ref, channel_scores(a, "signed"))[0, 1] for a in attr[1:]]))


def summarise(masking: pd.DataFrame, per_window: pd.DataFrame, top_k: int) -> list[dict]:
    """Per TSA version: means and one-sided Wilcoxon tests (top-k vs random-k), as for SHAP."""
    out = []
    for version, g_mask in masking.groupby("version"):
        rows = []
        for (unit, k), g in g_mask.groupby(["unit", "k"]):
            rows.append({
                "unit": unit, "k": int(k), "n": len(g),
                "faithfulness": g.faithfulness.mean(), "faithfulness_random": g.faithfulness_random.mean(),
                "faithfulness_p": wilcoxon(g.faithfulness, g.faithfulness_random,
                                           alternative="greater").pvalue,
                "sufficiency": g.sufficiency.mean(), "sufficiency_random": g.sufficiency_random.mean(),
                "sufficiency_p": wilcoxon(g.sufficiency, g.sufficiency_random,
                                          alternative="greater").pvalue})
        w = per_window[per_window.version == version]
        stab = w["stability"].dropna()
        out.append({"version": version, "masking": rows,
                    "stability": {"n": len(stab), "mean": stab.mean(), "frac_ge_0.7": (stab >= 0.7).mean()},
                    "concordance": {"mean_frontal_ratio": w.frontal_ratio.mean(),
                                    f"frac_frontal_in_top{top_k}": w.frontal_in_top_k.mean()}})
    return out


# ── main ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name → outputs/c4/tsa/<name>/")
    ap.add_argument("--plant", default=None, help="electrode for the planted alpha, e.g. F3")
    ap.add_argument("--strength", type=float, default=1.0, help="planted alpha RMS ÷ electrode SD")
    ap.add_argument("--plant-where", choices=["eval", "all"], default="eval")
    ap.add_argument("--k", type=float, default=2.0, help="encoder threshold multiplier")
    ap.add_argument("--epochs", type=int, default=1, help="STDP passes over the training windows")
    ap.add_argument("--stability-windows", type=int, default=2, help="per test subject")
    ap.add_argument("--config", default="c4_xai")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    seed, ec = cfg["project"]["seed"], cfg["evaluation"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    spec = feature_spec(cfg)
    names = spec.channel_names
    frontal = spec.local_index(region_channels(("F3", "F4")))
    out_dir = component_output_dir("c4") / "tsa" / args.name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    data, ids, labels = load_subjects(spec, args, cfg["synthetic"]["label_seed"], args.k)
    folds = subject_folds(ids, labels, cfg["baseline"]["n_folds"], cfg["baseline"]["val_fraction"], seed)

    preds, masking, per_window = [], [], []
    group = {v: [] for v in VERSIONS}         # (label, class-1 attribution [C, T]) per explained window
    for f in folds:
        bb, ro = train_fold(data, f["train"] + f["val"], args.epochs, seed + f["fold"])
        predict = make_predict(bb, ro)
        predict_int = lambda x: predict(from_interleaved(np.asarray(x)))  # noqa: E731
        tsas = {v: TemporalSpikeAttribution(bb, ro, variant=v[0], decay=v[1]) for v in VERSIONS}

        example_done = False
        for sid in tqdm(f["test"], desc=f"fold {f['fold']}"):
            d = data[sid]
            p = predict(d["spikes"])[:, 1]
            preds += [{"fold": f["fold"], "subject_id": sid, "label": d["label"], "p_mdd": pi} for pi in p]

            idx = np.linspace(0, len(d["spikes"]) - 1, cfg["shap"]["windows_per_subject"]).astype(int)
            x = d["spikes"][idx]
            targets = predict(x).argmax(1)
            for v, tsa in tsas.items():
                attr = tsa.attribute(x).numpy()                               # [n, K, C, T]
                for j, (i, t) in enumerate(zip(idx, targets)):
                    a = attr[j, t]
                    group[v].append((d["label"], attr[j, 1]))
                    common = {"version": f"{v[0]}_{v[1]}", "fold": f["fold"], "subject_id": sid,
                              "window": int(i), "label": d["label"], "target": int(t)}
                    x_int, a_int = to_interleaved(x[j].numpy()), np.repeat(a / 2, 2, axis=0)
                    for unit, ks, bin_len in (("channel", ec["top_k_channels"], 1),
                                              ("time", ec["top_k_time_bins"], TIME_BIN)):
                        for row in masking_curves(predict_int, x_int, a_int, int(t), ks, unit,
                                                  bin_len, 2, ec["n_random_masks"], rng):
                            masking.append({**common, **row})
                    w = {**common, **frontal_concordance(a, frontal, 1, ec["concordance_top_k"])}
                    if j < args.stability_windows:
                        w["stability"] = tsa_stability(tsa, d["eeg"][i], d["thr"], int(t),
                                                       ec["stability_repeats"], ec["stability_noise_std"], rng)
                    per_window.append(w)

            # one example figure per fold: the first label-1 subject with a window predicted 1
            if not example_done and d["label"] == 1 and (targets == 1).any():
                j = int(np.argmax(targets == 1))
                a = tsas[MAIN_VERSION].attribute(x[j:j + 1])[0, 1].numpy()
                fig = example_figure(a, names, f"TSA-NS leak-matched · subject {sid} · window "
                                     f"{idx[j]} · label 1 · predicted 1 · fold {f['fold']}")
                fig.savefig(out_dir / "figures" / f"subject_{sid}.png", dpi=120)
                plt.close(fig)
                example_done = True

    # model accuracy (window- and subject-level)
    preds = pd.DataFrame(preds)
    preds.to_csv(out_dir / "test_predictions.csv", index=False)
    subj = preds.groupby("subject_id").agg(label=("label", "first"), p_mdd=("p_mdd", "mean"))
    cv = {"window": {"balanced_accuracy": balanced_accuracy_score(preds.label, preds.p_mdd > .5),
                     "auc": roc_auc_score(preds.label, preds.p_mdd)},
          "subject": {"balanced_accuracy": balanced_accuracy_score(subj.label, subj.p_mdd > .5),
                      "auc": roc_auc_score(subj.label, subj.p_mdd), "n": len(subj)}}
    json.dump(cv, open(out_dir / "cv_summary.json", "w"), indent=2, default=float)

    # explanation metrics
    masking, per_window = pd.DataFrame(masking), pd.DataFrame(per_window)
    masking.to_csv(out_dir / "xai_masking.csv", index=False)
    per_window.to_csv(out_dir / "xai_per_window.csv", index=False)
    summary = summarise(masking, per_window, ec["concordance_top_k"])
    json.dump(summary, open(out_dir / "xai_summary.json", "w"), indent=2, default=float)

    # group scalp maps for class 1 — top row: size of credit over all explained windows;
    # bottom row: net evidence for class 1 over label-1 windows — and the planted check
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    planted = {"electrode": args.plant, "strength": args.strength, "where": args.plant_where}
    for col, v in enumerate(VERSIONS):
        label = f"TSA-{v[0].upper()} {v[1]}"
        size = np.mean([channel_scores(a, "abs") for _, a in group[v]], axis=0)
        net = np.mean([channel_scores(a, "signed") for y, a in group[v] if y == 1], axis=0)
        plot_scalp_map(size, names, f"{label}\nsize of credit", ax=axes[0, col])
        plot_scalp_map(net, names, f"{label}\nnet evidence for class 1 (label-1 windows)",
                       ax=axes[1, col], signed=True)
        if args.plant:
            # Two readings of class-1 attribution over label-1 windows: "abs" = size of credit
            # (busy electrodes whose +/− path credit cancels can rank high), "signed" = net push
            # towards class 1 (the planted question). Both are reported.
            target = spec.local_index([channel_index(args.plant)])[0]
            planted[f"{v[0]}_{v[1]}"] = {}
            for mode in ("abs", "signed"):
                pos = np.mean([channel_scores(a, mode) for y, a in group[v] if y == 1], axis=0)
                order = list(np.argsort(-pos))
                planted[f"{v[0]}_{v[1]}"][mode] = {"rank_of_planted": order.index(target) + 1,
                                                   "top3": [names[i] for i in order[:3]]}
    fig.suptitle(f"{args.name}: TSA attribution for class 1 over test windows")
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "group_scalp_maps.png", dpi=120)
    plt.close(fig)
    if args.plant:
        rate = {y: float(np.mean([d["spikes"][..., [target, target + len(names)]].mean().item()
                                  for d in data.values() if d["label"] == y])) for y in (0, 1)}
        planted["planted_electrode_spike_rate"] = {"label_0": rate[0], "label_1": rate[1]}
        json.dump(planted, open(out_dir / "planted.json", "w"), indent=2, default=float)

    # report
    print(f"\n=== {args.name} ({(time.time() - t_start) / 60:.1f} min) ===")
    print("model:", json.dumps(cv, default=lambda v: round(float(v), 3)))
    for s in summary:
        ch3 = next(r for r in s["masking"] if r["unit"] == "channel" and r["k"] == 3)
        print(f"{s['version']:16s} faithfulness top-3 {ch3['faithfulness']:.3f} vs random "
              f"{ch3['faithfulness_random']:.3f} (p={ch3['faithfulness_p']:.1e}) | sufficiency "
              f"{ch3['sufficiency']:.3f} vs {ch3['sufficiency_random']:.3f} | stability "
              f"{s['stability']['mean']:.2f}")
    if args.plant:
        print("planted:", json.dumps(planted, default=lambda v: round(float(v), 3)))
    shap_path = component_output_dir("c4") / cfg["baseline"]["output_subdir"] / "xai_summary.json"
    if shap_path.exists():
        shap = json.load(open(shap_path))
        ch3 = next(r for r in shap["masking"] if r["unit"] == "channel" and r["k"] == 3)
        print(f"{'SHAP (LSTM)':16s} faithfulness top-3 {ch3['faithfulness']:.3f} vs random "
              f"{ch3['faithfulness_random']:.3f} | stability {shap['stability']['mean']:.2f}"
              "  (different model: compare explainers, not models)")
    print("saved to", out_dir)


if __name__ == "__main__":
    main()

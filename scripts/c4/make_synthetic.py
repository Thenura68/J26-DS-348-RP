"""Planted-signal check: build a dataset whose correct explanation is known.

Real MODMA recordings get RANDOM labels (the real MDD/HC labels are discarded, so no real
signal remains). For every subject labelled 1, an alpha oscillation (9-11 Hz, slowly waxing
and waning) is added to one electrode of the raw EEG, raising that electrode's alpha power by
``strength`` × its own. The recording then goes through C1's preprocessing and the C4
band-power step unchanged. With --strength 0 the same labels get no signal (control).

    python scripts/c4/make_synthetic.py --name alpha_F3_x1
    python scripts/c4/make_synthetic.py --name control --strength 0

Then run the baseline on it:
    S="baseline.features_subdir=c4_synthetic/alpha_F3_x1 baseline.output_subdir=synthetic/alpha_F3_x1"
    python scripts/c4/train_lstm.py --set $S   (and explain_shap.py / evaluate_xai.py)
"""

from __future__ import annotations

import argparse
import json

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

from snn_depression.c4_xai.baseline.features import subject_folds, window_band_power
from snn_depression.c4_xai.montage import channel_index
from snn_depression.common.config import load_config
from snn_depression.common.labels import subject_targets
from snn_depression.common.paths import PROCESSED_DIR
from snn_depression.data.modma import list_subjects, load_subject
from snn_depression.data.preprocessing import prepare_subject_windows


def planted_alpha(n: int, fs: float, rms: float, rng: np.random.Generator) -> np.ndarray:
    """Alpha-like oscillation of the given RMS: 9-11 Hz carrier, slow amplitude envelope."""
    t = np.arange(n) / fs
    envelope = 1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 2 * np.pi))
    wave = envelope * np.sin(2 * np.pi * rng.uniform(9, 11) * t + rng.uniform(0, 2 * np.pi))
    return wave * rms / wave.std()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--strength", type=float, default=None, help="default: synthetic.strength")
    ap.add_argument("--electrode", default=None, help="default: synthetic.electrode")
    args = ap.parse_args()
    cfg = load_config("c4_xai")
    c1_cfg = load_config("c1_encoding")
    syn = cfg["synthetic"]
    strength = syn["strength"] if args.strength is None else args.strength
    electrode = args.electrode or syn["electrode"]
    rng = np.random.default_rng(syn["label_seed"])

    ids = list_subjects()
    labels = np.zeros(len(ids), dtype=int)
    labels[rng.permutation(len(ids))[: len(ids) // 2]] = 1
    out = PROCESSED_DIR / "c4_synthetic" / args.name
    out.mkdir(parents=True, exist_ok=True)
    ch = channel_index(electrode)

    rows = []
    for sid, label in tqdm(list(zip(ids, labels)), desc=args.name):
        rec = load_subject(sid)
        eeg = rec.eeg.copy()
        if label == 1 and strength > 0:
            own = mne.filter.filter_data(eeg[ch], rec.fs, 8, 13, verbose=False)
            eeg[ch] += planted_alpha(eeg.shape[1], rec.fs, np.sqrt(strength) * own.std(), rng)
        split = prepare_subject_windows(eeg, rec.channel_names, rec.fs, c1_cfg)
        np.save(out / f"{sid}_bp.npy", window_band_power(split.windows, rec.fs,
                                                          cfg["baseline"]["band_power_segment_sec"]))
        rows.append({"subject_id": sid, "group": rec.group, "phq9": rec.phq9, "label": int(label),
                     "real_label": subject_targets(rec.group, rec.phq9)[0],
                     "n_windows": len(split.windows),
                     "n_calibration_windows": len(split.calibration_windows)})

    subjects = pd.DataFrame(rows)
    subjects.to_csv(out / "subjects.csv", index=False)
    b = cfg["baseline"]
    folds = subject_folds(subjects.subject_id, subjects.label, b["n_folds"], b["val_fraction"],
                          cfg["project"]["seed"])
    json.dump({"seed": cfg["project"]["seed"], "folds": folds,
               "synthetic": {"electrode": electrode, "strength": strength,
                             "label_seed": syn["label_seed"]}},
              open(out / "splits.json", "w"), indent=2)
    print(f"{labels.sum()} planted / {len(labels) - labels.sum()} clean subjects -> {out}  "
          f"(agreement with real labels {(subjects.label == subjects.real_label).mean():.2f})")


if __name__ == "__main__":
    main()

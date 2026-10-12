"""Band-power features for the C4 LSTM baseline, from C1's preprocessed MODMA windows.

Run C1 first:   python scripts/c1_preprocess_modma.py
Then:           python scripts/c4/make_baseline_features.py

Reads   data/interim/modma_preprocessed/<id>.npy  [N, 128, W] + metadata.csv
Writes  data/processed/c4_baseline/<id>_bp.npy     [N, 128, 5] log10 band power
        data/processed/c4_baseline/subjects.csv    id, label (HC 0 / MDD 1), PHQ-9, window counts
        data/processed/c4_baseline/splits.json     subject-wise folds
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from snn_depression.c4_xai.baseline.features import features_dir, subject_folds, window_band_power
from snn_depression.c4_xai.baseline.utils import apply_overrides
from snn_depression.common.config import load_config
from snn_depression.common.paths import INTERIM_DIR

C1_DIR = INTERIM_DIR / "modma_preprocessed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="c4_xai")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)

    meta_path = C1_DIR / "metadata.csv"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found - run scripts/c1_preprocess_modma.py first")
    meta = pd.read_csv(meta_path, dtype={"subject_id": str})
    out = features_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    fs = float(cfg["data"]["sampling_rate_hz"])
    for sid in tqdm(meta.subject_id, desc="band power"):
        windows = np.load(C1_DIR / f"{sid}.npy")                      # calibration then evaluation
        np.save(out / f"{sid}_bp.npy",
                window_band_power(windows, fs, cfg["baseline"]["band_power_segment_sec"]))

    subjects = meta[["subject_id", "group", "phq9", "diagnosis_target",
                     "n_windows", "n_calibration_windows"]].rename(columns={"diagnosis_target": "label"})
    subjects.to_csv(out / "subjects.csv", index=False)
    b = cfg["baseline"]
    folds = subject_folds(subjects.subject_id, subjects.label, b["n_folds"], b["val_fraction"],
                          cfg["project"]["seed"])
    with open(out / "splits.json", "w") as f:
        json.dump({"seed": cfg["project"]["seed"], "folds": folds}, f, indent=2)
    print(subjects.groupby("group").size().to_string())
    print(f"Wrote features for {len(subjects)} subjects and {len(folds)} folds to {out}")


if __name__ == "__main__":
    main()

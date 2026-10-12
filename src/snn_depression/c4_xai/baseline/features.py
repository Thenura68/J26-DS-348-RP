"""Band-power features from C1's preprocessed windows, and the FeatureSpec describing them.

One time step = one C1 window (``data.window_sec``, 2 s). Each window becomes the log10
power in five bands for every electrode. Features are channel-major
(feature f = channel × n_bands + band), so attributions reshape to [C, B, T].

Note: C1 z-scores each channel separately, so band power is relative within an electrode;
absolute power differences between electrodes (e.g. F3 vs F4 alpha) are reduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold, train_test_split

from snn_depression.common.paths import PROCESSED_DIR

from ..montage import CHANNEL_NAMES, EGI_TO_1020, channel_index

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
CLASS_NAMES = ("HC", "MDD")


@dataclass
class FeatureSpec:
    channels: np.ndarray        # indices into E1-E128
    band_names: list[str]
    step_s: float               # seconds per time step (one C1 window)

    @property
    def n_bands(self) -> int:
        return len(self.band_names)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_features(self) -> int:
        return self.n_channels * self.n_bands

    @property
    def channel_names(self) -> list[str]:
        return [CHANNEL_NAMES[i] for i in self.channels]

    def local_index(self, global_idx) -> np.ndarray:
        """E1-E128 indices → positions in this channel subset (absent ones dropped)."""
        pos = {g: i for i, g in enumerate(self.channels)}
        return np.array([pos[g] for g in global_idx if g in pos], dtype=int)


def feature_spec(cfg: dict) -> FeatureSpec:
    ch = cfg["baseline"]["channels"]
    if ch == "all":
        channels = np.arange(len(CHANNEL_NAMES))
    elif ch == "1020":
        channels = np.array(sorted(channel_index(n) for n in EGI_TO_1020))
    else:
        channels = np.array([channel_index(n) for n in ch])
    return FeatureSpec(channels, list(BANDS), float(cfg["data"]["window_sec"]))


def features_dir(cfg: dict) -> Path:
    return PROCESSED_DIR / cfg["baseline"]["features_subdir"]


def window_band_power(windows: np.ndarray, fs: float, segment_sec: float = 1.0) -> np.ndarray:
    """windows [N, C, W] → log10 band power [N, C, B] (Welch, ``segment_sec`` segments)."""
    freqs, psd = welch(windows, fs=fs, nperseg=int(segment_sec * fs), axis=-1)
    bp = np.stack([psd[..., (freqs >= lo) & (freqs < hi)].mean(-1) for lo, hi in BANDS.values()], -1)
    return np.log10(bp + 1e-30).astype(np.float32)


def read_subjects(fdir: Path) -> pd.DataFrame:
    """subjects.csv written by scripts/c4/make_baseline_features.py (IDs kept as strings)."""
    return pd.read_csv(fdir / "subjects.csv", dtype={"subject_id": str})


def load_subject_features(fdir: Path, subject_id: str, spec: FeatureSpec) -> np.ndarray:
    """[F, N_windows] float32 for one subject, restricted to the spec's channels."""
    bp = np.load(fdir / f"{subject_id}_bp.npy")[:, spec.channels]       # [N, C, B]
    return bp.reshape(bp.shape[0], -1).T.astype(np.float32)


def subject_folds(subject_ids, labels, n_folds: int = 5, val_fraction: float = 0.15,
                  seed: int = 42) -> list[dict]:
    """Subject-wise stratified folds: no subject is in more than one of train/val/test."""
    subject_ids, labels = np.asarray(subject_ids), np.asarray(labels)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for k, (trval, test) in enumerate(skf.split(subject_ids, labels)):
        tr, val = train_test_split(trval, test_size=val_fraction, stratify=labels[trval],
                                   random_state=seed + k)
        folds.append({"fold": k,
                      "train": sorted(subject_ids[tr].tolist()),
                      "val": sorted(subject_ids[val].tolist()),
                      "test": sorted(subject_ids[test].tolist())})
    return folds

"""Reduce an attribution map [F, T] to electrode scores, band scores and a timeline.

Features are channel-major: F = channels × bands (bands = 1 when the model sees one value
per electrode, e.g. spikes or raw EEG), so attr.reshape(C, B, T) recovers the electrodes.

``signed`` keeps the direction of evidence (positive = supports the explained class) and is
used to rank features for masking, so masking the top features removes evidence *for* the
prediction. ``abs`` gives importance regardless of direction and is used for display.
"""

from __future__ import annotations

import numpy as np


def _prep(attr: np.ndarray, mode: str) -> np.ndarray:
    return attr if mode == "signed" else np.abs(attr)


def channel_scores(attr: np.ndarray, mode: str = "signed", n_bands: int = 1) -> np.ndarray:
    """attr [F, T] → [C], summed over each electrode's bands and over time."""
    a = _prep(attr, mode)
    return a.reshape(-1, n_bands, a.shape[-1]).sum(axis=(1, 2))


def channel_band_scores(attr: np.ndarray, n_bands: int, mode: str = "abs") -> np.ndarray:
    """attr [F, T] → [C, B], summed over time."""
    a = _prep(attr, mode)
    return a.reshape(-1, n_bands, a.shape[-1]).sum(axis=2)


def time_scores(attr: np.ndarray, bin_len: int = 1, mode: str = "signed") -> np.ndarray:
    """attr [F, T] → [n_bins], summed over features within bins of ``bin_len`` steps."""
    a = _prep(attr, mode).sum(axis=0)
    n_bins = len(a) // bin_len
    return a[: n_bins * bin_len].reshape(n_bins, bin_len).sum(axis=1)


def normalise_share(scores: np.ndarray) -> np.ndarray:
    """Non-negative shares summing to 1 (for "% of attribution" displays)."""
    s = np.abs(scores)
    return s / (s.sum() + 1e-12)

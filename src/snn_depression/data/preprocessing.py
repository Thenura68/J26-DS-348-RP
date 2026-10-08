"""EEG preprocessing shared by all components."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    eeg: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last axis.

    eeg: [..., S] (e.g. [C, S]); returns the same shape.
    """
    if not 0 < low < high < fs / 2:
        raise ValueError(f"need 0 < low < high < fs/2, got low={low}, high={high}, fs={fs}")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, eeg, axis=-1)


def make_windows(
    eeg: np.ndarray, fs: float, window_sec: float, overlap: float = 0.0
) -> np.ndarray:
    """Split a continuous recording [C, S] into windows [N, C, W], in chronological order.

    The trailing remainder shorter than one window is dropped.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    win = int(round(window_sec * fs))
    step = max(1, int(round(win * (1.0 - overlap))))
    n_samples = eeg.shape[-1]
    if n_samples < win:
        return np.empty((0, eeg.shape[0], win), dtype=eeg.dtype)
    starts = range(0, n_samples - win + 1, step)
    return np.stack([eeg[:, s : s + win] for s in starts])

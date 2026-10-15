"""Temporary replacement for C1's spike encoder — swap in ``c1_encoding`` when it is ready.

Follows C1's planned method (``configs/c1_encoding.yaml``: adaptive delta, MAD, k = 1.0) so
TSA can be developed before C1's encoder exists. Output matches contract boundary 2:
spikes ``[B, T, 2 × C_eeg]`` float32, ordered ``[ON_ch0..ON_chN, OFF_ch0..OFF_chN]``.

Delta encoding, per electrode: keep a reference level; when the signal rises more than θ above
it emit an ON spike and raise the reference by θ, when it falls more than θ below emit an OFF
spike and lower it by θ. θ = k × MAD of the sample-to-sample change, measured only on the
patient's calibration portion (first 20 % of the recording), so thresholds are patient-adaptive.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def downsample(eeg: np.ndarray, factor: int = 2) -> np.ndarray:
    """Average non-overlapping blocks of ``factor`` samples (250 Hz → 125 Hz for factor 2).

    eeg [N, C, S] → [N, C, S // factor]. Keeps content up to fs_new / 2 (62.5 Hz), above
    the 45 Hz upper edge of C1's band-pass.
    """
    n, c, s = eeg.shape
    s = s - s % factor
    return eeg[..., :s].reshape(n, c, s // factor, factor).mean(-1)


def fit_thresholds(calibration: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Per-electrode thresholds θ [C] from calibration windows [N, C, S].

    θ_c = k × MAD(Δx_c), where Δx is the change between consecutive samples. MAD (median
    absolute deviation) is used instead of std so a few artifact spikes don't inflate θ.
    """
    diff = np.diff(calibration, axis=-1)                       # [N, C, S-1]
    diff = diff.transpose(1, 0, 2).reshape(calibration.shape[1], -1)   # [C, N·(S-1)]
    med = np.median(diff, axis=1, keepdims=True)
    mad = np.median(np.abs(diff - med), axis=1)
    return (k * np.maximum(mad, 1e-8)).astype(np.float32)


def encode(eeg: np.ndarray | Tensor, thresholds: np.ndarray | Tensor) -> Tensor:
    """Delta-encode EEG windows into ON/OFF spike trains.

    Args:
        eeg: [B, C, S] windows (each window encoded independently; reference starts at its
            first sample).
        thresholds: [C] per-electrode θ from :func:`fit_thresholds`.
    Returns:
        spikes [B, S, 2C] float32 of 0/1, ON channels first then OFF (batch-first, contract
        boundary 2). At most one spike per channel per time step.
    """
    x = torch.as_tensor(eeg, dtype=torch.float32)
    theta = torch.as_tensor(thresholds, dtype=torch.float32)   # [C]
    b, c, s = x.shape
    ref = x[:, :, 0].clone()                                   # [B, C]
    on = torch.zeros(b, s, c)
    off = torch.zeros(b, s, c)
    for t in range(1, s):
        up = x[:, :, t] - ref >= theta
        down = ref - x[:, :, t] >= theta
        on[:, t] = up.float()
        off[:, t] = down.float()
        ref = ref + theta * up - theta * down
    return torch.cat([on, off], dim=-1)

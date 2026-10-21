"""Sequences of consecutive windows over per-subject [F, N] feature arrays."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Yields (x [T, F] float32, label, subject index) for sequences of ``seq_len`` steps.

    ``mean``/``std`` [F] standardise features (training-subject statistics only).
    """

    def __init__(self, arrays: dict[str, np.ndarray], labels: dict[str, int], seq_len: int,
                 stride: int, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.arrays = arrays
        self.labels = labels
        self.seq_len = seq_len
        self.mean, self.std = mean, std
        self.subjects = list(arrays)
        self.index = [(sid, start)
                      for sid, a in arrays.items()
                      for start in range(0, a.shape[1] - seq_len + 1, stride)]

    def __len__(self) -> int:
        return len(self.index)

    def window(self, i: int) -> np.ndarray:
        sid, start = self.index[i]
        x = np.array(self.arrays[sid][:, start:start + self.seq_len].T, dtype=np.float32)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return x

    def __getitem__(self, i: int):
        sid, _ = self.index[i]
        return torch.from_numpy(self.window(i)), self.labels[sid], self.subjects.index(sid)

    def window_labels(self) -> np.ndarray:
        return np.array([self.labels[sid] for sid, _ in self.index])

    def window_subjects(self) -> list[str]:
        return [sid for sid, _ in self.index]

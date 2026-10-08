"""Subject-wise splitting. No subject may appear in both training and testing."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np


def leave_one_subject_out(subject_ids: Sequence) -> Iterator[tuple[object, np.ndarray, np.ndarray]]:
    """Yield (test_subject, train_indices, test_indices) for every subject.

    subject_ids: one entry per sample/window.
    """
    ids = np.asarray(subject_ids)
    for subject in np.unique(ids):
        test_mask = ids == subject
        yield subject, np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def calibration_split(n_windows: int, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split of ONE subject's windows into (calibration, evaluation) indices.

    The first ``fraction`` of windows calibrate the personal adapter; the rest are used
    only for evaluation. Windows must already be in recording order.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    n_cal = max(1, int(np.floor(n_windows * fraction)))
    if n_cal >= n_windows:
        raise ValueError(f"not enough windows ({n_windows}) for a calibration/evaluation split")
    idx = np.arange(n_windows)
    return idx[:n_cal], idx[n_cal:]

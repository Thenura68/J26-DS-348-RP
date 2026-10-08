"""Signal-integrity metrics: correlation, RMSE, SNR (stub)."""

from __future__ import annotations

from torch import Tensor


def signal_integrity(original: Tensor, reconstructed: Tensor) -> dict:
    """Return {"correlation", "rmse", "snr_db"} for one session."""
    raise NotImplementedError

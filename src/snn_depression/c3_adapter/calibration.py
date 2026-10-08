"""Per-patient calibration: freeze the backbone, fine-tune only the adapter (stub)."""

from __future__ import annotations

from torch import Tensor

from ..common.contracts import FeatureBackbone
from .adapter import PersonalSNNAdapter


def calibrate(
    backbone: FeatureBackbone,
    adapter: PersonalSNNAdapter,
    calib_spikes: Tensor,
    calib_labels: Tensor,
    max_epochs: int = 50,
    lr: float = 1e-3,
) -> dict:
    """Train the adapter on the patient's calibration windows (first 20% of the recording).

    Returns a log dict: epochs run, wall-clock seconds, final loss, trainable ratio.
    """
    raise NotImplementedError

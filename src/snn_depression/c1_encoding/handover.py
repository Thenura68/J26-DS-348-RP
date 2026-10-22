"""C1 → C2 handover: encode a patient's whole recording with thresholds from its first 20%.

The thresholds come from the patient's saved calibration profile, which
``scripts/c1_calibrate.py`` fits on the calibration windows only (the first 20% of the
recording). The whole recording (calibration windows followed by evaluation windows, in
time order) is then encoded with those thresholds.

Every result carries ``n_calibration_windows``: windows ``[:n]`` are the first 20%
and windows ``[n:]`` the last 80%. Components that score test patients must use only
the windows from ``n`` on for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from snn_depression.common.contracts import check_spike_train
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.c1_encoding.calibration import ThresholdProfile
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder

PREPROCESSED_DIR = INTERIM_DIR / "modma_preprocessed"
PROFILE_DIR = OUTPUT_ROOT / "c1" / "calibration" / "profiles"
SPIKE_DIR = OUTPUT_ROOT / "c1" / "spikes"


@dataclass(frozen=True)
class RecordingSpikes:
    """A patient's whole recording as contract spikes.

    ``spikes``: [N_windows, T, 256] float32 in {0, 1}, windows in time order.
    ``n_calibration_windows``: windows [:n] are the first 20%, [n:] the last 80%.
    """

    subject_id: str
    spikes: Tensor
    n_calibration_windows: int
    k: float

    @property
    def calibration(self) -> Tensor:
        """Spikes of the first 20% (the part the thresholds were fitted on)."""
        return self.spikes[: self.n_calibration_windows]

    @property
    def evaluation(self) -> Tensor:
        """Spikes of the last 80% (never used to set the thresholds)."""
        return self.spikes[self.n_calibration_windows :]


def load_profile(subject_id: str, profile_dir: Path = PROFILE_DIR) -> ThresholdProfile:
    """Load a patient's saved threshold profile (written by scripts/c1_calibrate.py)."""
    path = Path(profile_dir) / f"{subject_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No threshold profile for {subject_id} at {path}; run scripts/c1_calibrate.py")
    return ThresholdProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))


def encode_recording(
    subject_id: str,
    *,
    preprocessed_dir: Path = PREPROCESSED_DIR,
    profile_dir: Path = PROFILE_DIR,
) -> RecordingSpikes:
    """Encode a patient's whole recording with the thresholds from its first 20%.

    Reads ``<id>_calibration.npy`` and ``<id>_evaluation.npy`` (EEG windows [N, 128, T]),
    concatenates them in time order and encodes them with the patient's saved profile.
    Returns spikes [N_cal + N_eval, T, 256] plus the calibration/evaluation boundary.
    """
    preprocessed_dir = Path(preprocessed_dir)
    calibration = np.load(preprocessed_dir / f"{subject_id}_calibration.npy")
    evaluation = np.load(preprocessed_dir / f"{subject_id}_evaluation.npy")
    profile = load_profile(subject_id, profile_dir)
    if profile.calibration_window_count != calibration.shape[0]:
        raise ValueError(
            f"Profile for {subject_id} was fitted on {profile.calibration_window_count} calibration windows, "
            f"but the data has {calibration.shape[0]}; re-run scripts/c1_calibrate.py"
        )
    eeg = torch.from_numpy(np.concatenate([calibration, evaluation]).astype(np.float32, copy=False))
    spikes = AdaptiveThresholdEncoder.from_profile(profile).encode(eeg)
    check_spike_train(spikes, n_channels=2 * len(profile.channel_names))
    return RecordingSpikes(subject_id, spikes, int(calibration.shape[0]), float(profile.k))


def save_recording_spikes(recording: RecordingSpikes, spike_dir: Path = SPIKE_DIR) -> Path:
    """Save as compressed ``<id>.npz`` (uint8 spikes + boundary + k); returns the path."""
    spike_dir = Path(spike_dir)
    spike_dir.mkdir(parents=True, exist_ok=True)
    path = spike_dir / f"{recording.subject_id}.npz"
    np.savez_compressed(
        path,
        spikes=recording.spikes.numpy().astype(np.uint8),
        n_calibration_windows=np.int64(recording.n_calibration_windows),
        k=np.float64(recording.k),
    )
    return path


def load_exported_spikes(subject_id: str, spike_dir: Path = SPIKE_DIR) -> RecordingSpikes:
    """Load spikes saved by ``scripts/c1_export_spikes.py`` as float32 contract tensors."""
    path = Path(spike_dir) / f"{subject_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No exported spikes for {subject_id} at {path}; run scripts/c1_export_spikes.py")
    with np.load(path) as data:
        spikes = torch.from_numpy(data["spikes"].astype(np.float32))
        recording = RecordingSpikes(subject_id, spikes, int(data["n_calibration_windows"]), float(data["k"]))
    check_spike_train(recording.spikes)
    return recording

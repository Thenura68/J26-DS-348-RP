"""MODMA dataset loading (shared).

TODO (team): implement once the dataset is downloaded and the file format is confirmed.
Everyone should load data through this module so that channel order, labels and
subject IDs are identical across components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SubjectRecording:
    subject_id: str
    eeg: np.ndarray                 # [C, S] raw (unfiltered) EEG in microvolts
    fs: float                       # sampling rate (Hz)
    channel_names: list[str]
    phq9: int | None                # None for subjects without a score
    group: str                      # "MDD" or "HC"
    meta: dict = field(default_factory=dict)  # age, gender, ...


def list_subjects(montage: str = "eeg_128ch_resting") -> list[str]:
    """Return all subject IDs available for a montage."""
    raise NotImplementedError("Implement after MODMA download — see data/README.md")


def load_subject(subject_id: str, montage: str = "eeg_128ch_resting") -> SubjectRecording:
    """Load one subject's resting-state recording plus PHQ-9 label."""
    raise NotImplementedError("Implement after MODMA download — see data/README.md")

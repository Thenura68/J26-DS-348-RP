"""Electrode layout for the EGI HydroCel GSN-128 net used by MODMA.

MODMA channels are named E1-E128, not 10-20 names. EGI_TO_1020 gives EGI's published
nearest 10-20 equivalents (F3 = E24 and F4 = E124 were checked against MNE's template
positions), so scalp maps carry the labels clinicians expect and the clinical
concordance check can target F3/F4. Cz is the recording reference and is dropped by C1.
"""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np

MONTAGE = "GSN-HydroCel-128"
CHANNEL_NAMES = [f"E{i}" for i in range(1, 129)]

EGI_TO_1020 = {
    "Fp1": "E22", "Fp2": "E9", "Fz": "E11",
    "F3": "E24", "F4": "E124", "F7": "E33", "F8": "E122",
    "C3": "E36", "C4": "E104",
    "T7": "E45", "T8": "E108",
    "P3": "E52", "P4": "E92", "Pz": "E62", "P7": "E58", "P8": "E96",
    "O1": "E70", "O2": "E83",
}
_1020_FROM_EGI = {v: k for k, v in EGI_TO_1020.items()}


def make_info(ch_names: Sequence[str] = CHANNEL_NAMES, sfreq: float = 250.0) -> mne.Info:
    """MNE Info with HydroCel-128 positions for the given channel subset."""
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage(MONTAGE))
    return info


def channel_index(name: str) -> int:
    """Index into E1-E128 for an E-number or a 10-20 name (e.g. "F3" → 23)."""
    return CHANNEL_NAMES.index(EGI_TO_1020.get(name, name))


def channel_positions() -> np.ndarray:
    """[128, 3] electrode positions in metres."""
    return np.array([ch["loc"][:3] for ch in make_info()["chs"]])


def region_channels(centers: Sequence[str] = ("F3", "F4"), n_neighbors: int = 6) -> np.ndarray:
    """Indices (into E1-E128) of each centre electrode plus its nearest neighbours."""
    pos = channel_positions()
    idx: set[int] = set()
    for c in centers:
        d = np.linalg.norm(pos - pos[channel_index(c)], axis=1)
        idx.update(np.argsort(d)[: n_neighbors + 1].tolist())
    return np.array(sorted(idx))


def display_name(name: str) -> str:
    """"E24" → "E24 (F3)"; electrodes without a 10-20 equivalent keep their E-number."""
    return f"{name} ({_1020_FROM_EGI[name]})" if name in _1020_FROM_EGI else name


def topomap_labels(ch_names: Sequence[str] = CHANNEL_NAMES) -> list[str]:
    """plot_topomap labels: the 10-20 name at its equivalent electrode, blank elsewhere."""
    return [_1020_FROM_EGI.get(n, "") for n in ch_names]

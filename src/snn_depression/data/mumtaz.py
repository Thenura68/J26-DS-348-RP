from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import torch
from torch import Tensor

from snn_depression.c1_encoding.calibration import (
    ThresholdProfile,
    ThresholdSettings,
    compute_threshold_profile,
)
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder
from snn_depression.common.contracts import check_spike_train
from snn_depression.common.paths import DATA_ROOT
from snn_depression.data.preprocessing import (
    SubjectWindowSplit,
    bandpass_filter,
    calibration_artifact_limit,
    make_windows,
    within_artifact_limit,
)
from snn_depression.data.splits import stratified_subject_kfold

MUMTAZ_CHANNELS_19 = (
    "Fp1", "F3", "C3", "P3", "O1",
    "F7", "T3", "T5", "Fz", "Fp2",
    "F4", "C4", "P4", "O2", "F8",
    "T4", "T6", "Cz", "Pz",
)

_CHANNEL_MAP = {
    f"EEG {ch}-LE": ch for ch in MUMTAZ_CHANNELS_19
}


@dataclass
class MUMTAZRecording:
    # One subject recording from MUMTAZ dataset

    subject_id: str
    eeg: np.ndarray 
    fs: float
    channel_names: list[str]
    group: str  # 'MDD' or 'HC'
    diagnosis_target: int  # 1 = MDD, 0 = HC
    condition: str  
    file_path: Path
    meta: dict[str, Any] = field(default_factory=dict)


def _find_mumtaz_root(base_dir: Path | None = None) -> Path:
    candidates = (
        (base_dir or DATA_ROOT) / "MUMTAZ",
        (base_dir or DATA_ROOT) / "raw" / "mumtaz",
        (base_dir or DATA_ROOT),
    )
    for c in candidates:
        if c.is_dir() and any(c.glob("*.edf")):
            return c
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not locate MUMTAZ EDF files. Looked in: {tried}")


def parse_mumtaz_filename(filename: str) -> dict[str, Any] | None:
    # Extract group, subject_id, and condition from filename

    stem = Path(filename).stem
    # Match MDD
    m_mdd = re.search(r"MDD\s+S(\d+)\s+([A-Z0-9]+)", stem, re.IGNORECASE)
    if m_mdd:
        sub_num = int(m_mdd.group(1))
        cond = m_mdd.group(2).upper()
        return {
            "group": "MDD",
            "subject_id": f"MDD_S{sub_num:02d}",
            "condition": cond,
            "diagnosis_target": 1,
        }
    # Match HC / H
    m_hc = re.search(r"(?:^|\D)H\s+S(\d+)\s+([A-Z0-9]+)", stem, re.IGNORECASE)
    if m_hc:
        sub_num = int(m_hc.group(1))
        cond = m_hc.group(2).upper()
        return {
            "group": "HC",
            "subject_id": f"HC_S{sub_num:02d}",
            "condition": cond,
            "diagnosis_target": 0,
        }
    return None


def list_mumtaz_recordings(
    data_dir: Path | None = None,
    condition: str = "EC",
) -> list[dict[str, Any]]:
    root = _find_mumtaz_root(data_dir)
    recordings = []
    condition = condition.upper()
    for p in sorted(root.glob("*.edf")):
        info = parse_mumtaz_filename(p.name)
        if info is None:
            continue
        if condition != "ALL" and info["condition"] != condition:
            continue
        info["file_path"] = p
        recordings.append(info)
    return recordings


def load_mumtaz_recording(file_path: Path | str) -> MUMTAZRecording:
    # Load one MUMTAZ EDF file, selecting the 19 standard 10-20 channels.
    p = Path(file_path)
    info = parse_mumtaz_filename(p.name)
    if info is None:
        raise ValueError(f"Could not parse MUMTAZ subject and condition from {p.name}")

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose=False)
    # Match the 19 standard channels
    available_chs = raw.ch_names
    picked_chs = []
    std_names = []
    for raw_name in available_chs:
        clean = raw_name.strip()
        if clean in _CHANNEL_MAP:
            picked_chs.append(raw_name)
            std_names.append(_CHANNEL_MAP[clean])
        else:
            # Fallback check without prefix/suffix
            sub = clean.replace("EEG ", "").replace("-LE", "").strip()
            if sub in MUMTAZ_CHANNELS_19:
                picked_chs.append(raw_name)
                std_names.append(sub)

    if len(picked_chs) != 19:
        raise ValueError(
            f"Expected 19 standard 10-20 channels, found {len(picked_chs)} in {p.name}: {picked_chs}"
        )

    raw.pick(picked_chs)
    data = raw.get_data().astype(np.float64)  # [19, S]

    return MUMTAZRecording(
        subject_id=info["subject_id"],
        eeg=data,
        fs=float(raw.info["sfreq"]),
        channel_names=std_names,
        group=info["group"],
        diagnosis_target=info["diagnosis_target"],
        condition=info["condition"],
        file_path=p,
        meta={"n_samples": data.shape[1], "duration_sec": data.shape[1] / raw.info["sfreq"]},
    )


def preprocess_mumtaz_recording(
    recording: MUMTAZRecording,
    window_sec: float = 2.0,
    window_overlap: float = 0.0,
    calibration_fraction: float = 0.2,
    bandpass_hz: tuple[float, float] = (0.5, 45.0),
    notch_hz: float = 50.0,
    ptp_mad_multiplier: float = 10.0,
) -> SubjectWindowSplit:
    
    raw_data = recording.eeg  # [19, S]
    fs = recording.fs
    n_samples = raw_data.shape[1]

    # Bandpass filter
    filtered = bandpass_filter(raw_data, fs=fs, low=bandpass_hz[0], high=bandpass_hz[1], order=4)

    # Chronological partition
    split_sample = int(round(n_samples * calibration_fraction))
    if split_sample <= 0 or split_sample >= n_samples:
        raise ValueError(f"Invalid split_sample {split_sample} for total {n_samples}")

    cal_data = filtered[:, :split_sample]
    eval_data = filtered[:, split_sample:]

    # Z-score fitted on calibration portion only
    mean = cal_data.mean(axis=-1, keepdims=True)
    std = cal_data.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    cal_scaled = (cal_data - mean) / std
    eval_scaled = (eval_data - mean) / std

    # Windowing
    cal_windows = make_windows(cal_scaled, fs=fs, window_sec=window_sec, overlap=window_overlap)
    eval_windows = make_windows(eval_scaled, fs=fs, window_sec=window_sec, overlap=window_overlap)

    if cal_windows.shape[0] == 0 or eval_windows.shape[0] == 0:
        raise ValueError(f"Subject {recording.subject_id} has insufficient samples for windows")

    # Peak-to-peak artifact rejection fitted from calibration windows only
    artifact_limit = calibration_artifact_limit(cal_windows, ptp_mad_multiplier)
    cal_keep = within_artifact_limit(cal_windows, artifact_limit)
    eval_keep = within_artifact_limit(eval_windows, artifact_limit)

    # Fallback if strict artifact filter rejected all windows
    if not cal_keep.any():
        cal_keep[:] = True
    if not eval_keep.any():
        eval_keep[:] = True

    cal_kept = np.asarray(cal_windows[cal_keep], dtype=np.float32)
    eval_kept = np.asarray(eval_windows[eval_keep], dtype=np.float32)

    qc = {
        "subject_id": recording.subject_id,
        "group": recording.group,
        "condition": recording.condition,
        "fs": fs,
        "n_samples": n_samples,
        "cal_windows": int(cal_kept.shape[0]),
        "eval_windows": int(eval_kept.shape[0]),
        "artifact_limit": float(artifact_limit),
    }

    return SubjectWindowSplit(
        calibration_windows=cal_kept,
        evaluation_windows=eval_kept,
        channel_names=list(recording.channel_names),
        qc=qc,
    )


def encode_mumtaz_windows(
    split: SubjectWindowSplit,
    subject_id: str,
    k: float = 1.0,
    settings: ThresholdSettings | None = None,
) -> tuple[Tensor, Tensor, ThresholdProfile]:
    # Fit adaptive TBR profile on calibration windows and encode both sets to dual-binary spikes.

    # cal_spikes: [N_cal, T, 38] float32 {0, 1}
    # eval_spikes: [N_eval, T, 38] float32 {0, 1}
    # profile: fitted ThresholdProfile
    
    if settings is None:
        settings = ThresholdSettings(
            include_mean=True,
            minimum_threshold=1.0e-6,
            minimum_signal_std=1.0e-8,
            minimum_derivative_std=1.0e-8,
        )

    cal_sec = split.calibration_windows.shape[0] * (split.calibration_windows.shape[2] / 256.0)
    profile = compute_threshold_profile(
        split.calibration_windows,
        split.channel_names,
        subject_id=subject_id,
        k=k,
        settings=settings,
        calibration_seconds=cal_sec,
        unit="zscore",
    )

    encoder = AdaptiveThresholdEncoder.from_profile(profile, polarity_mode="dual_binary")

    cal_spikes = encoder.encode(torch.from_numpy(split.calibration_windows).float())
    eval_spikes = encoder.encode(torch.from_numpy(split.evaluation_windows).float())

    check_spike_train(cal_spikes)
    check_spike_train(eval_spikes)

    return cal_spikes, eval_spikes, profile


def build_mumtaz_dataset(
    data_dir: Path | None = None,
    condition: str = "EC",
    k: float = 1.0,
    window_sec: float = 2.0,
) -> dict[str, Any]:
    # Load, preprocess, and encode all subjects from MUMTAZ into a unified cohort dictionary.
    # 'subjects': list of subject_ids
    # 'labels': dict of subject_id -> diagnosis_target (1=MDD, 0=HC)
    # 'cal_spikes': dict of subject_id -> Tensor [N_cal, T, 38]
    # 'eval_spikes': dict of subject_id -> Tensor [N_eval, T, 38]
    # 'all_spikes': dict of subject_id -> Tensor [N_total, T, 38]
    # 'channel_names': tuple of 19 channel names
    # 'spike_channels': 38 (dual binary)
    
    recordings = list_mumtaz_recordings(data_dir=data_dir, condition=condition)
    if not recordings:
        raise ValueError(f"No recordings found for condition {condition}")

    subject_data: dict[str, dict[str, Any]] = {}
    labels: dict[str, int] = {}
    cal_spikes_dict: dict[str, Tensor] = {}
    eval_spikes_dict: dict[str, Tensor] = {}
    all_spikes_dict: dict[str, Tensor] = {}

    settings = ThresholdSettings(
        include_mean=True,
        minimum_threshold=1.0e-6,
        minimum_signal_std=1.0e-8,
        minimum_derivative_std=1.0e-8,
    )

    for rec_info in recordings:
        rec = load_mumtaz_recording(rec_info["file_path"])
        sub_id = rec.subject_id
        # Avoid duplicate files for same subject in same condition
        if sub_id in subject_data:
            continue

        split = preprocess_mumtaz_recording(rec, window_sec=window_sec)
        cal_spk, eval_spk, profile = encode_mumtaz_windows(split, sub_id, k=k, settings=settings)

        all_spk = torch.cat((cal_spk, eval_spk), dim=0)

        subject_data[sub_id] = {"qc": split.qc, "profile": profile}
        labels[sub_id] = rec.diagnosis_target
        cal_spikes_dict[sub_id] = cal_spk
        eval_spikes_dict[sub_id] = eval_spk
        all_spikes_dict[sub_id] = all_spk

    subjects = sorted(labels.keys())
    return {
        "subjects": subjects,
        "labels": labels,
        "cal_spikes": cal_spikes_dict,
        "eval_spikes": eval_spikes_dict,
        "all_spikes": all_spikes_dict,
        "channel_names": MUMTAZ_CHANNELS_19,
        "spike_channels": 2 * len(MUMTAZ_CHANNELS_19),
        "condition": condition,
    }


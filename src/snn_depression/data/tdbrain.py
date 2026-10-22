from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mne
import numpy as np
import openpyxl
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

TDBRAIN_CHANNELS_26 = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4", "T7", "C3", "Cz", "C4", "T8",
    "CP3", "CPz", "CP4", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
)


@dataclass
class TDBRAINRecording:

    subject_id: str
    eeg: np.ndarray  
    fs: float
    channel_names: list[str]
    group: str  
    diagnosis_target: int  
    condition: str  
    file_path: Path
    meta: dict[str, Any] = field(default_factory=dict)


def _find_tdbrain_root(base_dir: Path | None = None) -> Path:
    """Locate the TDBRAIN directory under DATA_ROOT."""
    candidates = (
        (base_dir or DATA_ROOT) / "TDBRAIN",
        (base_dir or DATA_ROOT) / "raw" / "tdbrain",
        (base_dir or DATA_ROOT),
    )
    for c in candidates:
        if c.is_dir() and (c / "TDBRAIN_participants_V3.xlsx").exists():
            return c
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not locate TDBRAIN folder with metadata sheet. Looked in: {tried}")


def load_tdbrain_participants(excel_path: Path | str) -> dict[str, dict[str, Any]]:
    wb = openpyxl.load_workbook(str(excel_path), read_only=True)
    sheet = wb.active
    rows = sheet.iter_rows(values_only=True)
    header = next(rows, None)
    if header is None:
        raise ValueError(f"TDBRAIN participants spreadsheet is empty: {excel_path}")

    id_col = header.index("TDBRAIN_ID")
    status_col = header.index("formal_status")
    ind_col = header.index("indication")
    age_col = header.index("age") if "age" in header else None
    gender_col = header.index("gender") if "gender" in header else None

    participants: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r[id_col] is None:
            continue
        sid = str(r[id_col]).strip()
        status = str(r[status_col]).strip().upper() if r[status_col] is not None else ""
        ind = str(r[ind_col]).strip().upper() if r[ind_col] is not None else ""

        if status == "MDD" or ind == "MDD":
            group = "MDD"
            target = 1
        elif status == "HEALTHY" or ind == "HEALTHY":
            group = "HEALTHY"
            target = 0
        else:
            continue  

        participants[sid] = {
            "subject_id": sid,
            "group": group,
            "diagnosis_target": target,
            "age": r[age_col] if age_col is not None else None,
            "gender": r[gender_col] if gender_col is not None else None,
        }
    wb.close()
    return participants


def list_tdbrain_recordings(
    data_dir: Path | None = None,
    condition: str = "restEC",
) -> list[dict[str, Any]]:
    """List all available TDBRAIN recordings with matching MDD or HEALTHY ground-truth labels."""
    root = _find_tdbrain_root(data_dir)
    excel_path = root / "TDBRAIN_participants_V3.xlsx"
    participants = load_tdbrain_participants(excel_path)

    dataset_dir = root / "TDBRAIN_Dataset_V3_1"
    if not dataset_dir.is_dir():
        dataset_dir = root

    recordings = []
    condition = condition.lower()

    for sid, info in sorted(participants.items()):
        sub_dir = dataset_dir / sid
        if not sub_dir.is_dir():
            continue

        # Find matching condition BDF file
        bdf_candidates = list(sub_dir.rglob(f"*{condition}*_eeg.bdf"))
        if not bdf_candidates:
            # Fallback pattern
            bdf_candidates = list(sub_dir.rglob(f"*{condition}*.bdf"))
        if not bdf_candidates:
            continue

        rec_entry = dict(info)
        rec_entry["file_path"] = bdf_candidates[0]
        rec_entry["condition"] = condition
        recordings.append(rec_entry)

    return recordings


def load_tdbrain_recording(
    file_path: Path | str,
    target_fs: float = 250.0,
    participant_info: dict[str, Any] | None = None,
) -> TDBRAINRecording:
    # Load one TDBRAIN BDF recording, select 26 scalp channels, and resample to 250 Hz
    p = Path(file_path)
    raw = mne.io.read_raw_bdf(str(p), preload=True, verbose=False)

    # Pick the 26 standard channels
    available = raw.ch_names
    picked = [ch for ch in TDBRAIN_CHANNELS_26 if ch in available]
    if len(picked) != 26:
        raise ValueError(
            f"Expected 26 standard scalp channels in {p.name}, but found {len(picked)}: {picked}"
        )
    raw.pick(picked)

    # Resample to target_fs (250 Hz) to unify with MODMA
    if raw.info["sfreq"] != target_fs:
        raw.resample(target_fs, verbose=False)

    data = raw.get_data().astype(np.float64)  # [26, S]

    # Resolve metadata
    if participant_info is not None:
        sid = participant_info["subject_id"]
        group = participant_info["group"]
        target = participant_info["diagnosis_target"]
    else:
        # Infer subject id from path
        sid = p.parent.parent.parent.name if p.parent.parent.parent.name.startswith("sub-") else p.stem.split("_")[0]
        group = "UNKNOWN"
        target = -1

    return TDBRAINRecording(
        subject_id=sid,
        eeg=data,
        fs=target_fs,
        channel_names=list(TDBRAIN_CHANNELS_26),
        group=group,
        diagnosis_target=target,
        condition="restEC",
        file_path=p,
        meta={"n_samples": data.shape[1], "duration_sec": data.shape[1] / target_fs},
    )


def preprocess_tdbrain_recording(
    recording: TDBRAINRecording,
    window_sec: float = 2.0,
    window_overlap: float = 0.0,
    calibration_fraction: float = 0.2,
    bandpass_hz: tuple[float, float] = (0.5, 45.0),
    ptp_mad_multiplier: float = 10.0,
) -> SubjectWindowSplit:

    raw_data = recording.eeg  # [26, S]
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

    # Windowing (500 samples per window at 250 Hz)
    cal_windows = make_windows(cal_scaled, fs=fs, window_sec=window_sec, overlap=window_overlap)
    eval_windows = make_windows(eval_scaled, fs=fs, window_sec=window_sec, overlap=window_overlap)

    if cal_windows.shape[0] == 0 or eval_windows.shape[0] == 0:
        raise ValueError(f"Subject {recording.subject_id} has insufficient samples for windows")

    # Peak-to-peak artifact rejection fitted from calibration windows only
    artifact_limit = calibration_artifact_limit(cal_windows, ptp_mad_multiplier)
    cal_keep = within_artifact_limit(cal_windows, artifact_limit)
    eval_keep = within_artifact_limit(eval_windows, artifact_limit)

    if not cal_keep.any():
        cal_keep[:] = True
    if not eval_keep.any():
        eval_keep[:] = True

    cal_kept = np.asarray(cal_windows[cal_keep], dtype=np.float32)
    eval_kept = np.asarray(eval_windows[eval_keep], dtype=np.float32)

    qc = {
        "subject_id": recording.subject_id,
        "group": recording.group,
        "fs": fs,
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


def encode_tdbrain_windows(
    split: SubjectWindowSplit,
    subject_id: str,
    k: float = 1.0,
    settings: ThresholdSettings | None = None,
) -> tuple[Tensor, Tensor, ThresholdProfile]:
    # Fit adaptive TBR profile on calibration windows and encode both sets to dual-binary spikes

    if settings is None:
        settings = ThresholdSettings(
            include_mean=True,
            minimum_threshold=1.0e-6,
            minimum_signal_std=1.0e-8,
            minimum_derivative_std=1.0e-8,
        )

    cal_sec = split.calibration_windows.shape[0] * (split.calibration_windows.shape[2] / 250.0)
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


def build_tdbrain_dataset(
    data_dir: Path | None = None,
    condition: str = "restEC",
    k: float = 1.0,
    window_sec: float = 2.0,
    max_subjects: int | None = None,
) -> dict[str, Any]:
    recordings = list_tdbrain_recordings(data_dir=data_dir, condition=condition)
    if not recordings:
        raise ValueError(f"No recordings found for condition {condition} in TDBRAIN")

    if max_subjects is not None and max_subjects < len(recordings):
        recordings = recordings[:max_subjects]

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
        sid = rec_info["subject_id"]
        if sid in subject_data:
            continue

        rec = load_tdbrain_recording(rec_info["file_path"], target_fs=250.0, participant_info=rec_info)
        split = preprocess_tdbrain_recording(rec, window_sec=window_sec)
        cal_spk, eval_spk, profile = encode_tdbrain_windows(split, sid, k=k, settings=settings)

        all_spk = torch.cat((cal_spk, eval_spk), dim=0)

        subject_data[sid] = {"qc": split.qc, "profile": profile}
        labels[sid] = rec.diagnosis_target
        cal_spikes_dict[sid] = cal_spk
        eval_spikes_dict[sid] = eval_spk
        all_spikes_dict[sid] = all_spk

    subjects = sorted(labels.keys())
    return {
        "subjects": subjects,
        "labels": labels,
        "cal_spikes": cal_spikes_dict,
        "eval_spikes": eval_spikes_dict,
        "all_spikes": all_spikes_dict,
        "channel_names": TDBRAIN_CHANNELS_26,
        "spike_channels": 2 * len(TDBRAIN_CHANNELS_26),
        "condition": condition,
    }


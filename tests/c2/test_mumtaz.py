# Unit tests for MUMTAZ dataset loading and equivalent encoding pipeline

from __future__ import annotations

import numpy as np
import torch
import pytest

from snn_depression.common.contracts import check_spike_train
from snn_depression.data.mumtaz import (
    MUMTAZRecording,
    parse_mumtaz_filename,
    preprocess_mumtaz_recording,
    encode_mumtaz_windows,
)


def test_parse_mumtaz_filename():
    info_mdd = parse_mumtaz_filename("MDD S1 EC.edf")
    assert info_mdd == {
        "group": "MDD",
        "subject_id": "MDD_S01",
        "condition": "EC",
        "diagnosis_target": 1,
    }

    info_hc = parse_mumtaz_filename("H S15 EO.edf")
    assert info_hc == {
        "group": "HC",
        "subject_id": "HC_S15",
        "condition": "EO",
        "diagnosis_target": 0,
    }

    info_prefix = parse_mumtaz_filename("6921143_H S15 EO.edf")
    assert info_prefix["subject_id"] == "HC_S15"


def test_mumtaz_preprocessing_and_encoding():
    # Create synthetic 19-channel EEG recording
    fs = 256.0
    duration_sec = 20.0
    n_samples = int(fs * duration_sec)
    channel_names = [f"Ch{i}" for i in range(19)]
    rng = np.random.default_rng(42)
    # 10 Hz sine wave + noise simulating resting EEG
    t = np.linspace(0, duration_sec, n_samples)
    synthetic_eeg = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.normal(size=(19, n_samples))

    recording = MUMTAZRecording(
        subject_id="MDD_S01",
        eeg=synthetic_eeg,
        fs=fs,
        channel_names=channel_names,
        group="MDD",
        diagnosis_target=1,
        condition="EC",
        file_path="dummy.edf",
    )

    split = preprocess_mumtaz_recording(recording, window_sec=2.0)
    assert split.calibration_windows.ndim == 3
    assert split.evaluation_windows.ndim == 3
    assert split.calibration_windows.shape[1] == 19
    assert split.calibration_windows.shape[2] == int(fs * 2.0)

    cal_spk, eval_spk, profile = encode_mumtaz_windows(split, "MDD_S01", k=1.0)
    assert cal_spk.shape[2] == 38  # 19 * 2 (dual binary)
    assert eval_spk.shape[2] == 38

    check_spike_train(cal_spk)
    check_spike_train(eval_spk)


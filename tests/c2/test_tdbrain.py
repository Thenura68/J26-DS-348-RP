# Unit tests for TDBRAIN dataset loading, 26-channel preprocessing, and TBR encoding

from __future__ import annotations

import numpy as np
import torch
import pytest

from snn_depression.c2_global_snn.backbone import GlobalSNNBackbone
from snn_depression.common.contracts import check_features, check_spike_train
from snn_depression.data.tdbrain import (
    TDBRAIN_CHANNELS_26,
    TDBRAINRecording,
    encode_tdbrain_windows,
    preprocess_tdbrain_recording,
)


def test_tdbrain_channel_count():
    assert len(TDBRAIN_CHANNELS_26) == 26
    assert len(TDBRAIN_CHANNELS_26) * 2 == 52


def test_tdbrain_preprocessing_and_encoding():
    fs = 250.0
    duration_sec = 20.0
    n_samples = int(fs * duration_sec)
    channel_names = list(TDBRAIN_CHANNELS_26)
    rng = np.random.default_rng(42)

    # 10 Hz alpha rhythm + noise
    t = np.linspace(0, duration_sec, n_samples)
    synthetic_eeg = np.sin(2 * np.pi * 10 * t) + 0.2 * rng.normal(size=(26, n_samples))

    recording = TDBRAINRecording(
        subject_id="sub-test01",
        eeg=synthetic_eeg,
        fs=fs,
        channel_names=channel_names,
        group="MDD",
        diagnosis_target=1,
        condition="restEC",
        file_path="dummy.bdf",
    )

    split = preprocess_tdbrain_recording(recording, window_sec=2.0)
    assert split.calibration_windows.ndim == 3
    assert split.evaluation_windows.ndim == 3
    assert split.calibration_windows.shape[1] == 26
    assert split.calibration_windows.shape[2] == int(fs * 2.0)  # 500 samples

    cal_spk, eval_spk, profile = encode_tdbrain_windows(split, "sub-test01", k=1.0)
    assert cal_spk.shape[1] == 500
    assert cal_spk.shape[2] == 52  # 26 * 2 (dual binary)
    assert eval_spk.shape[2] == 52

    check_spike_train(cal_spk)
    check_spike_train(eval_spk)


def test_global_backbone_with_tdbrain_spikes():
    # Verify GlobalSNNBackbone seamlessly ingests TDBRAIN [B, 500, 52] spikes and returns [B, 500, 256]
    batch = 3
    n_steps = 500
    c_spk = 52  # 26 * 2

    backbone = GlobalSNNBackbone(n_input_channels=c_spk, reservoir_size=64)
    spikes = (torch.rand(batch, n_steps, c_spk) > 0.85).float()

    check_spike_train(spikes)
    features = backbone(spikes)

    check_features(features)
    assert features.shape == (batch, n_steps, 256)


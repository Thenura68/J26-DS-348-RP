import json

import numpy as np
import pytest
import torch

from snn_depression.common.contracts import check_spike_train
from snn_depression.c1_encoding.calibration import ThresholdSettings, compute_threshold_profile
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder
from snn_depression.c1_encoding.handover import (
    encode_recording,
    load_exported_spikes,
    load_profile,
    save_recording_spikes,
)

NAMES = ["E1", "E2"]


def _setup(tmp_path, n_cal=3, n_eval=5):
    rng = np.random.default_rng(0)
    calibration = rng.normal(size=(n_cal, 2, 50)).astype(np.float32)
    evaluation = (3.0 * rng.normal(size=(n_eval, 2, 50))).astype(np.float32)
    data, profiles = tmp_path / "data", tmp_path / "profiles"
    data.mkdir(), profiles.mkdir()
    np.save(data / "s1_calibration.npy", calibration)
    np.save(data / "s1_evaluation.npy", evaluation)
    profile = compute_threshold_profile(
        calibration, NAMES, subject_id="s1", k=1.0,
        settings=ThresholdSettings(True, 1e-6, 1e-8, 1e-8), calibration_seconds=6.0, unit="z",
    )
    (profiles / "s1.json").write_text(json.dumps(profile.to_dict()), encoding="utf-8")
    return data, profiles, calibration, evaluation, profile


def test_whole_recording_is_encoded_with_thresholds_from_the_first_part(tmp_path):
    data, profiles, calibration, evaluation, profile = _setup(tmp_path)

    recording = encode_recording("s1", preprocessed_dir=data, profile_dir=profiles)

    encoder = AdaptiveThresholdEncoder.from_profile(profile)
    expected = encoder.encode(torch.from_numpy(np.concatenate([calibration, evaluation])))
    assert torch.equal(recording.spikes, expected)
    assert recording.spikes.shape == (8, 50, 4)
    assert recording.n_calibration_windows == 3
    assert torch.equal(recording.calibration, expected[:3])
    assert torch.equal(recording.evaluation, expected[3:])
    check_spike_train(recording.spikes, n_channels=4)


def test_thresholds_do_not_depend_on_the_evaluation_part(tmp_path):
    data, profiles, _, evaluation, _ = _setup(tmp_path)
    before = load_profile("s1", profiles).thresholds.copy()

    np.save(data / "s1_evaluation.npy", 100.0 * evaluation)
    encode_recording("s1", preprocessed_dir=data, profile_dir=profiles)

    assert np.array_equal(load_profile("s1", profiles).thresholds, before)


def test_stale_profile_is_rejected(tmp_path):
    data, profiles, calibration, _, _ = _setup(tmp_path)
    np.save(data / "s1_calibration.npy", np.concatenate([calibration, calibration[:1]]))

    with pytest.raises(ValueError, match="re-run scripts/c1_calibrate.py"):
        encode_recording("s1", preprocessed_dir=data, profile_dir=profiles)


def test_export_round_trip_keeps_spikes_and_boundary(tmp_path):
    data, profiles, *_ = _setup(tmp_path)
    recording = encode_recording("s1", preprocessed_dir=data, profile_dir=profiles)

    path = save_recording_spikes(recording, tmp_path / "spikes")
    loaded = load_exported_spikes("s1", tmp_path / "spikes")

    assert path.name == "s1.npz"
    assert loaded.spikes.dtype == torch.float32
    assert torch.equal(loaded.spikes, recording.spikes)
    assert loaded.n_calibration_windows == 3 and loaded.k == 1.0


def test_missing_profile_explains_what_to_run(tmp_path):
    data, *_ = _setup(tmp_path)
    with pytest.raises(FileNotFoundError, match="c1_calibrate.py"):
        encode_recording("s1", preprocessed_dir=data, profile_dir=tmp_path / "nowhere")

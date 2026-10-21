import numpy as np
import pytest
import torch

from snn_depression.common.config import load_config
from snn_depression.common.contracts import check_eeg
from snn_depression.common.labels import subject_targets
from snn_depression.data.preprocessing import (
    _filter_chunk_with_padding,
    calibration_artifact_limit,
    make_windows,
    notch_filter,
    prepare_subject_windows,
    rereference,
    within_artifact_limit,
    zscore_from_calibration,
)
from snn_depression.data.splits import build_fold_summaries, subject_folds


def _tone_amplitude(signal: np.ndarray, fs: float, frequency: float) -> float:
    time = np.arange(signal.size) / fs
    basis = np.exp(-2j * np.pi * frequency * time)
    return float(2 * np.abs(np.dot(signal, basis)) / signal.size)


def test_notch_removes_50_hz_and_preserves_10_hz():
    fs = 250
    time = np.arange(fs * 8) / fs
    signal = np.sin(2 * np.pi * 10 * time) + 0.6 * np.sin(2 * np.pi * 50 * time)
    filtered = notch_filter(signal[None, :], fs, frequency=50.0)[0]
    interior = slice(fs, -fs)

    assert _tone_amplitude(filtered[interior], fs, 50.0) < 0.03
    assert np.isclose(
        _tone_amplitude(filtered[interior], fs, 10.0),
        _tone_amplitude(signal[interior], fs, 10.0),
        rtol=0.03,
    )


def test_average_rereferencing_preserves_shape_and_zeroes_mean():
    eeg = np.arange(60, dtype=float).reshape(3, 20)
    referenced = rereference(eeg)

    assert referenced.shape == eeg.shape
    assert np.allclose(referenced.mean(axis=0), 0.0)


def test_windows_stay_in_chronological_order():
    eeg = np.arange(40, dtype=float)[None, :]
    windows = make_windows(eeg, fs=10, window_sec=1, overlap=0.5)

    assert windows.shape == (7, 1, 10)
    assert windows[:, 0, 0].tolist() == [0, 5, 10, 15, 20, 25, 30]


def test_artifact_limit_from_calibration_rejects_an_injected_evaluation_spike():
    rng = np.random.default_rng(3)
    calibration = rng.normal(size=(20, 2, 100))
    evaluation = rng.normal(size=(5, 2, 100))
    evaluation[2, 1, 50] = 100.0

    limit = calibration_artifact_limit(calibration, mad_multiplier=10.0)
    keep = within_artifact_limit(evaluation, limit)

    amplitudes = np.ptp(calibration, axis=-1).max(axis=1)
    median = np.median(amplitudes)
    assert limit == median + 10.0 * 1.4826 * np.median(np.abs(amplitudes - median))
    assert keep.tolist() == [True, True, False, True, True]
    assert within_artifact_limit(calibration, limit).all()


def test_zscore_uses_calibration_statistics_only():
    calibration = np.linspace(-1.0, 1.0, 20)[None, :]
    evaluation = np.linspace(10.0, 20.0, 80)[None, :]

    calibration_scaled, evaluation_scaled = zscore_from_calibration(
        calibration, evaluation, ["E1"]
    )
    _, changed_evaluation_scaled = zscore_from_calibration(
        calibration, 2.0 * evaluation, ["E1"]
    )

    assert np.isclose(calibration_scaled.mean(), 0.0, atol=1e-12)
    assert np.isclose(calibration_scaled.std(), 1.0)
    assert evaluation_scaled.mean() > 10.0
    assert np.allclose(changed_evaluation_scaled, 2.0 * evaluation_scaled + calibration.mean() / calibration.std())


def test_zscore_rejects_a_flat_calibration_channel():
    with pytest.raises(ValueError, match="zero-variance"):
        zscore_from_calibration(np.zeros((2, 10)) + [[0.0], [1.0]], np.ones((2, 10)), ["E1", "E2"])


def test_configured_pipeline_preserves_channels_and_native_amplitudes():
    fs = 250
    time = np.arange(fs * 6) / fs
    rng = np.random.default_rng(42)
    eeg = rng.normal(0.0, 3.0, size=(129, time.size))
    eeg[0] = 0.0
    eeg[128] = 123.0
    config = load_config("c1_encoding")
    config["split"]["calibration_fraction"] = 0.5
    config["data"]["window_sec"] = 1.0
    config["preprocessing"]["artifact"]["max_std_ratio"] = 1000.0
    config["preprocessing"]["artifact"]["variance_mad_multiplier"] = 1000.0

    result = prepare_subject_windows(
        eeg,
        [*(f"E{index}" for index in range(1, 129)), "Cz"],
        fs,
        config,
    )

    assert result.windows.shape[1:] == (128, fs)
    assert result.windows.dtype == np.float32
    assert result.channel_names == [f"E{index}" for index in range(1, 129)]
    assert result.qc["reference_row_dropped"] == "Cz"
    assert result.qc["channels_flagged_on_calibration_only"] == {"E1": ["flat"]}
    assert result.qc["channels_interpolated"] == ["E1"]
    assert result.qc["reference_method"] == "average"
    assert result.qc["c1_baseline_sec"] == 3.0
    assert result.qc["scaling_method"] == "zscore"
    check_eeg(torch.from_numpy(result.windows), n_channels=128)


def test_calibration_preprocessing_does_not_depend_on_evaluation_values():
    fs = 250
    rng = np.random.default_rng(21)
    eeg = rng.normal(size=(129, fs * 10))
    eeg[128] = 0.0
    channel_names = [*(f"E{index}" for index in range(1, 129)), "E129"]
    config = load_config("c1_encoding")
    config["split"]["calibration_fraction"] = 0.5
    config["data"]["window_sec"] = 1.0
    config["preprocessing"]["artifact"]["max_std_ratio"] = 1000.0
    config["preprocessing"]["artifact"]["variance_mad_multiplier"] = 1000.0

    first = prepare_subject_windows(eeg, channel_names, fs, config)
    changed_evaluation = eeg.copy()
    changed_evaluation[:128, fs * 5 :] *= 1.01
    second = prepare_subject_windows(changed_evaluation, channel_names, fs, config)

    assert np.allclose(first.calibration_windows, second.calibration_windows)
    assert first.qc["artifact_threshold"] == second.qc["artifact_threshold"]
    assert first.qc["split_sample"] == fs * 5
    assert first.qc["calibration_duration_sec"] == 5.0
    assert first.qc["evaluation_duration_sec"] == 5.0


def test_filter_chunk_with_padding_preserves_length_and_finiteness():
    fs = 250
    config = load_config("c1_encoding")
    eeg = np.random.default_rng(8).normal(size=(2, fs * 4))

    filtered = _filter_chunk_with_padding(eeg, fs, config)

    assert filtered.shape == eeg.shape
    assert np.isfinite(filtered).all()
    assert config["preprocessing"]["filter_padding_sec"] == 2.0


def test_fold_summary_keeps_subjects_separate_and_counts_test_windows():
    rows = [
        {
            "subject_id": subject_id,
            "group": group,
            "severity_target": label,
            "n_calibration_windows": 2,
            "n_evaluation_windows": 8,
        }
        for subject_id, group, label in [
            ("a", "MDD", 1), ("b", "HC", 0), ("c", "MDD", 2),
            ("d", "MDD", 1), ("e", "HC", 0), ("f", "MDD", 2),
        ]
    ]
    split = {"strategy": "stratified_subject_kfold", "n_splits": 2, "n_repeats": 1, "seed": 0}

    folds = build_fold_summaries(
        rows, subject_folds([r["subject_id"] for r in rows], [r["severity_target"] for r in rows], split)
    )

    assert len(folds) == 2
    for fold in folds:
        assert not set(fold["test_subject_ids"]) & set(fold["train_subject_ids"])
        assert fold["test_class_counts"] == {0: 1, 1: 1, 2: 1}
        assert all(t["evaluation_window_count"] == 8 for t in fold["test_subjects"])


def test_healthy_control_gets_diagnosis_zero_and_non_depressed_class():
    assert subject_targets("HC", 3) == (0, 0)
    assert subject_targets("MDD", 12) == (1, 1)
    assert subject_targets("MDD", 21) == (1, 2)

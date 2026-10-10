import numpy as np
import torch

from snn_depression.common.config import load_config
from snn_depression.common.contracts import check_eeg
from snn_depression.common.labels import subject_targets
from snn_depression.data.preprocessing import (
    _filter_chunk_with_padding,
    make_windows,
    notch_filter,
    normalize_from_calibration,
    prepare_subject_windows,
    preprocess_eeg,
    reject_bad_segments,
    rereference,
)
from scripts.c1_preprocess_modma import build_fold_summaries


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


def test_segment_rejection_flags_an_injected_spike():
    windows = np.zeros((5, 2, 100), dtype=float)
    windows[2, 1, 50] = 100.0

    kept, rejected, threshold = reject_bad_segments(
        windows, peak_to_peak_threshold=10.0
    )

    assert rejected == [2]
    assert threshold == 10.0
    assert kept.shape == (4, 2, 100)


def test_calibration_normalization_uses_only_initial_samples():
    calibration = np.linspace(-1.0, 1.0, 20)
    evaluation = np.linspace(10.0, 20.0, 80)
    eeg = np.concatenate((calibration, evaluation))[None, :]

    normalized = normalize_from_calibration(eeg, calibration_fraction=0.2)

    assert np.isclose(normalized[0, :20].mean(), 0.0, atol=1e-12)
    assert np.isclose(normalized[0, :20].std(), 1.0)
    assert normalized[0, 20:].mean() > 10.0


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

    result = preprocess_eeg(
        eeg,
        [*(f"E{index}" for index in range(1, 129)), "Cz"],
        fs,
        config,
    )

    assert result.windows.shape[1:] == (128, fs)
    assert result.windows.dtype == np.float32
    assert result.channel_names == [f"E{index}" for index in range(1, 129)]
    assert result.qc["channels_dropped_by_config"] == ["Cz"]
    assert result.qc["channels_dropped_as_bad"] == []
    assert result.qc["channels_flagged"] == {"E1": ["flat"]}
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


def test_fold_summary_keeps_subjects_separate_and_splits_calibration_chronologically():
    rows = [
        {
            "subject_id": "a",
            "group": "MDD",
            "n_windows": 10,
            "n_calibration_windows": 2,
            "n_evaluation_windows": 8,
            "calibration_source_window_indices": [0, 1],
            "evaluation_source_window_indices": list(range(8)),
            "split_sample": 100,
            "c1_baseline_sec": 20.0,
        },
        {
            "subject_id": "b",
            "group": "HC",
            "n_windows": 10,
            "n_calibration_windows": 2,
            "n_evaluation_windows": 8,
            "calibration_source_window_indices": [0, 1],
            "evaluation_source_window_indices": list(range(8)),
            "split_sample": 100,
            "c1_baseline_sec": 20.0,
        },
        {
            "subject_id": "c",
            "group": "MDD",
            "n_windows": 10,
            "n_calibration_windows": 2,
            "n_evaluation_windows": 8,
            "calibration_source_window_indices": [0, 1],
            "evaluation_source_window_indices": list(range(8)),
            "split_sample": 100,
            "c1_baseline_sec": 20.0,
        },
    ]

    folds = build_fold_summaries(rows)

    assert len(folds) == 3
    for fold in folds:
        assert fold["test_subject_id"] not in fold["training_subject_ids"]
        assert len(fold["training_subject_ids"]) == 2
        assert fold["calibration_window_source_indices"] == [0, 1]
        assert fold["evaluation_window_source_indices"] == [0, 7]
    assert folds[0]["training_group_counts"] == {"HC": 1, "MDD": 1}


def test_healthy_control_has_diagnosis_class_zero_and_no_severity_target():
    assert subject_targets("HC", 3) == (0, None)
    assert subject_targets("MDD", 7) == (1, 0)
    assert subject_targets("MDD", 3) == (1, None)

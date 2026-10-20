import json

import numpy as np
import pytest

from snn_depression.c1_encoding.calibration import (
    ThresholdProfile,
    ThresholdSettings,
    check_calibration_windows_inside_split,
    compute_threshold_profile,
    fit_thresholds,
    sweep_calibration_k,
)
from snn_depression.c1_encoding.decoder import leak_from_cutoff


def _settings(*, include_mean=True, minimum_threshold=1e-9, minimum_std=1e-9):
    return ThresholdSettings(
        include_mean=include_mean,
        minimum_threshold=minimum_threshold,
        minimum_signal_std=minimum_std,
        minimum_derivative_std=minimum_std,
    )


def _profile(windows, channel_names, *, k=1.0, settings=None):
    return compute_threshold_profile(
        windows,
        channel_names,
        subject_id="synthetic",
        k=k,
        settings=settings or _settings(),
        calibration_seconds=4.0,
        unit="synthetic units",
    )


def test_threshold_pooling_does_not_difference_across_window_edges():
    windows = np.array(
        [
            [[0.0, 1.0, 3.0]],
            [[10.0, 12.0, 13.0]],
        ]
    )

    profile = _profile(windows, ["E1"], k=2.0)

    assert np.isclose(profile.mean_derivative[0], 1.5)
    assert np.isclose(profile.derivative_std[0], 0.5)
    assert np.isclose(profile.thresholds[0], 2.5)


def test_flat_channel_uses_minimum_threshold_and_is_reported():
    windows = np.zeros((2, 2, 8), dtype=np.float32)
    windows[:, 1] = np.arange(8, dtype=np.float32)

    profile = _profile(
        windows,
        ["flat", "varying"],
        settings=_settings(include_mean=False, minimum_threshold=0.01, minimum_std=1e-8),
    )

    assert profile.thresholds[0] == 0.01
    assert profile.flagged_channels["flat_or_near_zero_signal_variance"] == ["flat"]
    assert profile.flagged_channels["near_zero_derivative_variance"] == [
        "flat",
        "varying",
    ]
    assert profile.flagged_channels["threshold_clamped"] == ["flat", "varying"]


def test_thresholds_adapt_to_each_subject_amplitude():
    rng = np.random.default_rng(0)
    quiet = rng.standard_normal((4, 2, 100))
    loud = 10.0 * quiet

    quiet_thresholds = fit_thresholds(quiet, 1.0, _settings(include_mean=False))
    loud_thresholds = fit_thresholds(loud, 1.0, _settings(include_mean=False))

    assert np.allclose(loud_thresholds, 10.0 * quiet_thresholds)


def test_profile_matches_fit_thresholds_and_round_trips_through_json():
    rng = np.random.default_rng(1)
    windows = rng.standard_normal((3, 2, 20))

    profile = _profile(windows, ["E1", "E2"], k=0.5)
    restored = ThresholdProfile.from_dict(json.loads(json.dumps(profile.to_dict())))

    assert np.allclose(profile.thresholds, fit_thresholds(windows, 0.5, _settings()))
    assert restored.channel_names == ("E1", "E2")
    assert np.array_equal(restored.thresholds, profile.thresholds)
    assert restored.flagged_channels == profile.flagged_channels


def test_duplicate_channel_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        _profile(np.zeros((1, 2, 4)), ["E1", "E1"])


def _random_walk_subjects():
    rng = np.random.default_rng(2)
    return {
        name: np.cumsum(rng.standard_normal((3, 2, 200)), axis=-1)
        for name in ("a", "b", "c")
    }


def test_calibration_k_sweep_reports_snr_and_density_per_k():
    best_k, curve = sweep_calibration_k(
        _random_walk_subjects(),
        k_candidates=[0.5, 1.0, 2.0],
        settings=_settings(include_mean=False),
        selection_metric="median_subject_snr_db",
    )

    assert best_k in (0.5, 1.0, 2.0)
    assert [point["k"] for point in curve] == [0.5, 1.0, 2.0]
    assert all(point["subjects_scored"] == 3 for point in curve)
    densities = [point["median_spike_density"] for point in curve]
    assert densities == sorted(densities, reverse=True)
    assert best_k == max(curve, key=lambda p: p["median_subject_snr_db"])["k"]


def test_sparsest_selection_trades_tiny_snr_loss_for_fewer_spikes():
    subjects = _random_walk_subjects()
    kwargs = dict(
        k_candidates=[0.1, 0.3, 0.5, 1.0, 2.0],
        settings=_settings(include_mean=False),
        decoder_leak=leak_from_cutoff(1.0, 250.0),
    )

    best_snr_k, curve = sweep_calibration_k(
        subjects, selection_metric="median_subject_snr_db", **kwargs
    )
    huge_tolerance_k, _ = sweep_calibration_k(
        subjects,
        selection_metric="sparsest_within_snr_tolerance",
        snr_tolerance_db=1e6,
        **kwargs,
    )
    zero_tolerance_k, _ = sweep_calibration_k(
        subjects,
        selection_metric="sparsest_within_snr_tolerance",
        snr_tolerance_db=0.0,
        **kwargs,
    )

    assert huge_tolerance_k == 2.0
    assert zero_tolerance_k == best_snr_k


def test_unknown_selection_metric_is_rejected():
    with pytest.raises(ValueError, match="selection_metric"):
        sweep_calibration_k(
            _random_walk_subjects(),
            k_candidates=[1.0],
            settings=_settings(),
            selection_metric="best_accuracy",
        )


def test_calibration_windows_inside_split_pass():
    # 1 s windows at 10 Hz with 50% overlap: window 18 covers samples 90-99.
    check_calibration_windows_inside_split(
        list(range(19)),
        split_sample=100,
        fs=10.0,
        window_sec=1.0,
        window_overlap=0.5,
    )


def test_calibration_window_crossing_split_is_rejected():
    with pytest.raises(ValueError, match=r"\[19\]"):
        check_calibration_windows_inside_split(
            [18, 19],
            split_sample=100,
            fs=10.0,
            window_sec=1.0,
            window_overlap=0.5,
        )

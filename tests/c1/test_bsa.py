import json
import math

import numpy as np
import pytest
import torch

from snn_depression.common.contracts import check_spike_train
from snn_depression.c1_encoding.bsa import (
    BSAProfile,
    BSASettings,
    _bsa_core,
    bsa_decode,
    bsa_encode,
    bsa_spike_density,
    design_bsa_filter,
    fit_bsa_profile,
    sweep_bsa_settings,
)
from snn_depression.c1_encoding.fidelity import snr_db

FS = 250.0


def _profile(windows, *, settings=None, names=None):
    names = names or [f"E{i + 1}" for i in range(windows.shape[1])]
    return fit_bsa_profile(
        windows,
        names,
        subject_id="synthetic",
        settings=settings or BSASettings(filter_length=15, cutoff_hz=30.0, threshold=0.0),
        fs=FS,
        unit="synthetic units",
    )


def _sine(frequency_hz=10.0, amplitude=1.0, windows=2, channels=1):
    time = np.arange(500) / FS
    wave = amplitude * np.sin(2 * np.pi * frequency_hz * time)
    return np.tile(wave, (windows, channels, 1))


def test_filter_is_symmetric_unit_gain_lowpass():
    taps = design_bsa_filter(25, 15.0, FS)

    assert taps.shape == (25,)
    assert np.isclose(taps.sum(), 1.0)
    assert np.allclose(taps, taps[::-1])
    assert taps.argmax() == 12
    with pytest.raises(ValueError):
        design_bsa_filter(25, 130.0, FS)
    with pytest.raises(ValueError):
        design_bsa_filter(1, 15.0, FS)


def _reference_bsa(signal, taps, threshold):
    """Literal per-channel BSA loop (Schrauwen & Van Campenhout, 2003) for comparison."""
    s = signal.copy()
    spikes = np.zeros_like(s, dtype=bool)
    for i in range(s.size):
        end = min(i + taps.size, s.size)
        f = taps[: end - i]
        if np.abs(s[i:end] - f).sum() <= np.abs(s[i:end]).sum() - threshold:
            spikes[i] = True
            s[i:end] -= f
    return spikes


def test_vectorised_core_matches_literal_bsa_loop():
    rng = np.random.default_rng(0)
    signal = rng.uniform(0.0, 1.0, size=(2, 3, 60))
    taps = design_bsa_filter(9, 30.0, FS)

    fast = _bsa_core(torch.from_numpy(signal), torch.from_numpy(taps), 0.05).numpy()

    for b in range(2):
        for c in range(3):
            assert np.array_equal(fast[b, c], _reference_bsa(signal[b, c], taps, 0.05))


def test_zero_signal_gives_no_spikes_and_zero_reconstruction():
    eeg = np.zeros((2, 2, 100))
    profile = _profile(np.ones((1, 2, 100)))

    spikes = bsa_encode(torch.from_numpy(eeg.astype(np.float32)), profile)

    assert torch.count_nonzero(spikes) == 0
    assert torch.equal(bsa_decode(spikes, profile), torch.zeros((2, 2, 100), dtype=torch.float64))


def test_output_follows_the_spike_train_contract():
    eeg = _sine(channels=3)
    spikes = bsa_encode(torch.from_numpy(eeg.astype(np.float32)), _profile(eeg))

    assert spikes.shape == (2, 500, 6)
    assert spikes.dtype == torch.float32
    check_spike_train(spikes, n_channels=6)


def test_positive_and_negative_halves_go_to_separate_channels():
    pulse = np.zeros((1, 1, 200))
    pulse[0, 0, 80:120] = 1.0
    profile = _profile(np.concatenate((pulse, -pulse)))

    up = bsa_encode(torch.from_numpy(pulse.astype(np.float32)), profile)
    down = bsa_encode(torch.from_numpy((-pulse).astype(np.float32)), profile)

    assert up[..., 0].sum() > 0 and up[..., 1].sum() == 0
    assert down[..., 1].sum() > 0 and down[..., 0].sum() == 0


def test_sine_is_reconstructed_well():
    eeg = _sine(frequency_hz=10.0)
    profile = _profile(eeg)

    spikes = bsa_encode(torch.from_numpy(eeg.astype(np.float32)), profile)
    rebuilt = bsa_decode(spikes, profile)

    assert snr_db(torch.from_numpy(eeg), rebuilt) > 8.0


def test_scale_comes_from_calibration_windows_only():
    calibration = _sine(amplitude=2.0)
    profile = _profile(calibration)
    loud_evaluation = _sine(amplitude=50.0)

    assert np.isclose(profile.scale[0], np.percentile(np.abs(calibration), 99.0))
    spikes = bsa_encode(torch.from_numpy(loud_evaluation.astype(np.float32)), profile)
    assert np.isclose(profile.scale[0], np.percentile(np.abs(calibration), 99.0))
    assert bsa_spike_density(spikes) > 0


def test_flat_calibration_channel_is_clamped_and_reported():
    windows = np.zeros((2, 2, 100))
    windows[:, 1] = _sine(windows=2)[:, 0, :100]

    profile = _profile(windows)

    assert profile.scale[0] == profile.settings.minimum_scale
    assert profile.clamped_channels == ["E1"]


def test_higher_threshold_never_adds_spikes():
    rng = np.random.default_rng(1)
    eeg = np.cumsum(rng.normal(size=(2, 2, 300)), axis=-1) * 0.05
    counts = []
    for threshold in (0.0, 0.1, 0.3):
        profile = _profile(eeg, settings=BSASettings(15, 30.0, threshold))
        counts.append(int(bsa_encode(torch.from_numpy(eeg.astype(np.float32)), profile).sum()))

    assert counts[0] >= counts[1] >= counts[2]


def test_profile_round_trips_through_json():
    profile = _profile(_sine(channels=2))

    restored = BSAProfile.from_dict(json.loads(json.dumps(profile.to_dict())))

    assert restored.settings == profile.settings
    assert np.array_equal(restored.scale, profile.scale)
    assert restored.channel_names == profile.channel_names


def test_sweep_scores_every_setting_and_picks_one_of_them():
    rng = np.random.default_rng(2)
    subjects = {
        name: np.cumsum(rng.normal(size=(2, 2, 200)), axis=-1) * 0.05 for name in ("a", "b")
    }

    best, curve = sweep_bsa_settings(
        subjects,
        fs=FS,
        filter_lengths=[9, 15],
        cutoffs_hz=[30.0],
        thresholds=[0.0, 0.2],
        scale_percentile=99.0,
        minimum_scale=1e-6,
        selection_metric="sparsest_within_snr_tolerance",
        snr_tolerance_db=0.5,
    )

    assert len(curve) == 4
    assert all(row["subjects_scored"] == 2 and math.isfinite(row["median_subject_snr_db"]) for row in curve)
    assert (best.filter_length, best.cutoff_hz, best.threshold) in {
        (r["filter_length"], r["cutoff_hz"], r["threshold"]) for r in curve
    }
    with pytest.raises(ValueError, match="selection_metric"):
        sweep_bsa_settings(
            subjects, fs=FS, filter_lengths=[9], cutoffs_hz=[30.0], thresholds=[0.0],
            scale_percentile=99.0, minimum_scale=1e-6, selection_metric="best",
        )

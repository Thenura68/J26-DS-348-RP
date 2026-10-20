import math

import numpy as np
import pytest
import torch

from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.decoder import leak_from_cutoff, reconstruct
from snn_depression.c1_encoding.encoder import tbr_encode_signed
from snn_depression.c1_encoding.fidelity import (
    DEFAULT_BANDS,
    band_powers,
    cohens_d,
    frontal_alpha_asymmetry,
    passes_thresholds,
    relative_band_powers,
    spectral_fidelity,
)

FS = 250.0


def _sine_windows(frequency_hz, *, amplitudes=(1.0,), windows=4, seconds=2.0, noise=0.0):
    set_seed(0)
    time = np.arange(int(FS * seconds)) / FS
    wave = np.sin(2 * np.pi * frequency_hz * time)
    eeg = np.stack([a * wave for a in amplitudes])[None].repeat(windows, axis=0)
    return eeg + noise * np.random.default_rng(0).standard_normal(eeg.shape)


def test_alpha_sine_power_lands_in_alpha_band():
    powers = band_powers(_sine_windows(10.0), FS)
    relative = relative_band_powers(powers)

    assert powers.shape == (1, len(DEFAULT_BANDS))
    assert relative[0, list(DEFAULT_BANDS).index("alpha")] > 0.95
    assert np.isclose(relative.sum(), 1.0)


def test_identical_signals_have_zero_spectral_error():
    eeg = _sine_windows(10.0, amplitudes=(1.0, 0.5), noise=0.2)

    report = spectral_fidelity(eeg, eeg.copy(), FS, ["E1", "E2"])

    assert all(v == 0 for v in report["band_power_abs_error_db"].values())
    assert all(v == 0 for v in report["relative_power_abs_error"].values())


def test_uniform_gain_shows_as_bias_but_keeps_relative_power():
    eeg = _sine_windows(10.0, noise=0.2)

    report = spectral_fidelity(eeg, 2.0 * eeg, FS, ["E1"])

    for band in DEFAULT_BANDS:
        assert report["band_power_bias_db"][band] == pytest.approx(10 * math.log10(4.0))
        assert report["relative_power_abs_error"][band] == pytest.approx(0.0, abs=1e-12)


def test_frontal_alpha_asymmetry_is_log_right_minus_left():
    eeg = _sine_windows(10.0, amplitudes=(1.0, 2.0))
    alpha = band_powers(eeg, FS)[:, list(DEFAULT_BANDS).index("alpha")]

    faa = frontal_alpha_asymmetry(alpha, ["F3", "F4"], "F3", "F4")
    report = spectral_fidelity(eeg, eeg, FS, ["F3", "F4"], asymmetry_pairs={"F3_F4": ["F3", "F4"]})

    assert faa == pytest.approx(math.log(4.0), rel=1e-6)
    assert report["alpha_asymmetry"]["F3_F4"]["original"] == pytest.approx(faa)
    with pytest.raises(ValueError, match="not found"):
        frontal_alpha_asymmetry(alpha, ["F3", "F4"], "F3", "F8")


def test_band_narrower_than_psd_resolution_is_rejected():
    with pytest.raises(ValueError, match="fewer than two PSD bins"):
        band_powers(_sine_windows(10.0), FS, {"narrow": (10.0, 10.5)})


def test_tbr_round_trip_keeps_alpha_dominance_and_asymmetry_sign():
    eeg = _sine_windows(10.0, amplitudes=(1.0, 2.0), noise=0.3)
    tensor = torch.from_numpy(eeg)
    theta = (tensor[..., 1:] - tensor[..., :-1]).std(dim=(0, 2))

    spikes = tbr_encode_signed(tensor, theta)
    decoded = reconstruct(
        spikes, theta, tensor[:, :, 0], polarity_mode="signed", leak=leak_from_cutoff(1.0, FS)
    )
    report = spectral_fidelity(
        eeg, decoded, FS, ["F3", "F4"], asymmetry_pairs={"F3_F4": ["F3", "F4"]}
    )
    relative = relative_band_powers(band_powers(decoded, FS))

    assert (relative[:, list(DEFAULT_BANDS).index("alpha")] > 0.5).all()
    assert report["alpha_asymmetry"]["F3_F4"]["reconstructed"] > 0


def test_passes_thresholds():
    report = {"correlation": 0.6, "band_power_abs_error_db": {"alpha": 2.0, "beta": 4.0}}

    assert passes_thresholds(report, min_correlation=None, max_band_power_abs_error_db=None) is None
    assert passes_thresholds(report, min_correlation=0.5, max_band_power_abs_error_db=None)
    assert not passes_thresholds(report, min_correlation=0.5, max_band_power_abs_error_db=3.0)


def test_cohens_d_known_value_and_degenerate_cases():
    assert cohens_d([2.0, 4.0], [0.0, 2.0]) == pytest.approx(2.0 / math.sqrt(2.0))
    assert math.isnan(cohens_d([1.0], [0.0, 2.0]))
    assert math.isnan(cohens_d([1.0, 1.0], [1.0, 1.0]))

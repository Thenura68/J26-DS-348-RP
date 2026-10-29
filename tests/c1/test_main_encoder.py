import json

import numpy as np
import pytest
import torch

from snn_depression.common.config import load_config
from snn_depression.common.contracts import check_spike_train
from snn_depression.c1_encoding.calibration import ThresholdProfile, ThresholdSettings, compute_threshold_profile
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder
from snn_depression.c1_encoding.main_encoder import (
    decode,
    encode,
    fit_profile,
    k_candidates,
    main_k,
    main_method,
    make_encoder,
)
from snn_depression.c1_encoding.step_forward import AdaptiveStepForwardEncoder, step_forward

SETTINGS = ThresholdSettings(include_mean=True, minimum_threshold=1e-6, minimum_signal_std=1e-8, minimum_derivative_std=1e-8)


def _eeg(windows=3, channels=2, samples=200, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(samples) / 250.0
    return (np.sin(2 * np.pi * 10 * time) + 0.3 * rng.normal(size=(windows, channels, samples))).astype(np.float32)


def _profile(windows, encoder="step_forward", k=2.0):
    return compute_threshold_profile(windows, ["E1", "E2"], subject_id="s1", k=k, settings=SETTINGS,
                                     calibration_seconds=6.0, unit="z", encoder=encoder)


def test_config_selects_adaptive_step_forward_as_main_encoder():
    config = load_config("c1_encoding")
    assert main_method(config) == "step_forward"
    assert main_k(config) in k_candidates(config, "step_forward")


def test_step_forward_uses_the_same_patient_adaptive_thresholds_as_tbr():
    windows = _eeg()
    sf = _profile(windows, "step_forward")
    tbr = _profile(windows, "tbr")

    assert np.array_equal(sf.thresholds, tbr.thresholds)
    assert sf.encoder == "step_forward" and tbr.encoder == "tbr"


def test_make_encoder_follows_the_profile():
    windows = _eeg()
    assert isinstance(make_encoder(_profile(windows, "step_forward")), AdaptiveStepForwardEncoder)
    assert isinstance(make_encoder(_profile(windows, "tbr")), AdaptiveThresholdEncoder)


def test_adaptive_step_forward_matches_step_forward_and_meets_the_contract():
    windows = _eeg()
    profile = _profile(windows)
    eeg = torch.from_numpy(windows)

    spikes = make_encoder(profile).encode(eeg)

    check_spike_train(spikes, n_channels=4)
    assert spikes.shape == (3, 200, 4)
    assert torch.equal(spikes, step_forward(eeg, torch.from_numpy(profile.thresholds).float()))
    assert torch.equal(spikes, encode(eeg, torch.from_numpy(profile.thresholds), "step_forward"))


def test_step_forward_decoding_tracks_the_signal_within_about_one_threshold():
    windows = _eeg(samples=400)
    profile = _profile(windows)
    eeg = torch.from_numpy(windows)
    theta = torch.from_numpy(profile.thresholds)

    rebuilt = decode(make_encoder(profile).encode(eeg), theta, eeg[:, :, 0].double(), "step_forward",
                     load_config("c1_encoding"), 250.0)

    error = (eeg.double() - rebuilt).abs()
    assert torch.quantile(error.flatten(), 0.95) <= 1.5 * theta.max()


def test_fit_profile_uses_the_configured_encoder_and_k():
    config = load_config("c1_encoding")
    profile = fit_profile(_eeg(), ["E1", "E2"], subject_id="s1", calibration_seconds=6.0, unit="z", config=config)

    assert profile.encoder == main_method(config)
    assert profile.k == main_k(config)


def test_old_profile_files_without_encoder_still_load_as_tbr():
    payload = _profile(_eeg(), "tbr").to_dict()
    payload.pop("encoder")

    restored = ThresholdProfile.from_dict(json.loads(json.dumps(payload)))

    assert restored.encoder == "tbr"


def test_unknown_encoder_is_rejected():
    profile = _profile(_eeg(), "step_forward")
    broken = ThresholdProfile.from_dict({**profile.to_dict(), "encoder": "rate"})
    with pytest.raises(ValueError, match="Unknown encoder"):
        make_encoder(broken)

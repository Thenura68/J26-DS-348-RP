import math

import numpy as np
import pytest
import torch

from snn_depression.common.contracts import ContractError, check_spike_train
from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.calibration import (
    ThresholdSettings,
    compute_threshold_profile,
)
from snn_depression.c1_encoding.decoder import leak_from_cutoff, reconstruct
from snn_depression.c1_encoding.encoder import (
    AdaptiveThresholdEncoder,
    FixedThresholdEncoder,
    from_dual_binary,
    tbr_encode_signed,
    to_dual_binary,
)
from snn_depression.c1_encoding.fidelity import (
    correlation,
    signal_integrity,
    snr_db,
    spike_density,
)


def _adaptive_encoder(thresholds, *, polarity_mode="dual_binary", channel_names=("E1",)):
    return AdaptiveThresholdEncoder(
        np.asarray(thresholds, dtype=np.float64),
        channel_names=channel_names,
        subject_id="synthetic",
        polarity_mode=polarity_mode,
    )


def test_exact_threshold_step_emits_no_spike():
    eeg = torch.tensor([[[0.0, 1.0]]], dtype=torch.float32)

    spikes = _adaptive_encoder([1.0]).encode(eeg)

    assert torch.count_nonzero(spikes) == 0


def test_difference_larger_than_threshold_emits_one_positive_spike():
    eeg = torch.tensor([[[0.0, 1.5, 1.5]]], dtype=torch.float32)

    assert tbr_encode_signed(eeg, torch.tensor([1.0])).tolist() == [[[0, 1, 0]]]


def test_tbr_constant_signal_has_no_spikes():
    eeg = torch.full((3, 2, 7), 4.25, dtype=torch.float32)

    assert not tbr_encode_signed(eeg, torch.tensor([0.1, 0.1])).any()


def test_per_sample_thresholds_broadcast_over_batch():
    eeg = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]]], dtype=torch.float32)

    signed = tbr_encode_signed(eeg, torch.tensor([[1.0], [3.0]]))

    assert signed[:, 0, 1].tolist() == [1, 0]


@pytest.mark.parametrize(
    ("step", "expected_reconstruction_error"),
    [(1.5, 0.5), (3.0, 2.0)],
)
def test_one_spike_decoder_documents_large_step_loss(
    step, expected_reconstruction_error
):
    eeg = torch.tensor([[[0.0, step]]], dtype=torch.float32)
    encoder = _adaptive_encoder([1.0])

    spikes = encoder.encode(eeg)
    decoded = reconstruct(spikes, torch.tensor([1.0]), eeg[:, :, 0])

    assert spikes[0, 1, 0].item() == 1.0
    assert decoded[0, 0, 1].item() == 1.0
    assert np.isclose((eeg - decoded)[0, 0, 1].item(), expected_reconstruction_error)


def test_encoder_decoder_residual_identity_for_arbitrary_signal():
    eeg = torch.tensor(
        [
            [[0.0, 0.5, 2.0, 0.0], [1.0, 3.0, 3.5, -1.0]],
            [[2.0, 0.0, 1.0, 4.0], [-1.0, -2.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    thresholds = torch.tensor([1.0, 1.5])
    encoder = _adaptive_encoder(thresholds.numpy(), channel_names=("E1", "E2"))

    dual_binary = encoder.encode(eeg)
    signed = from_dual_binary(dual_binary)
    decoded = reconstruct(dual_binary, thresholds, eeg[:, :, 0])
    differences = eeg[..., 1:] - eeg[..., :-1]
    accumulated_step_error = torch.cumsum(
        differences - thresholds[None, :, None] * signed[..., 1:],
        dim=-1,
    )
    expected_residual = torch.cat(
        (torch.zeros_like(eeg[..., :1]), accumulated_step_error),
        dim=-1,
    )

    assert torch.allclose(eeg - decoded, expected_residual)


def test_constant_signal_has_no_spikes_and_reconstructs_from_initial_value():
    eeg = torch.full((2, 2, 8), 4.25, dtype=torch.float32)
    encoder = _adaptive_encoder([0.1, 0.2], channel_names=("E1", "E2"))

    spikes = encoder.encode(eeg)
    decoded = reconstruct(spikes, torch.tensor([0.1, 0.2]), eeg[:, :, 0])

    assert torch.count_nonzero(spikes) == 0
    assert torch.equal(decoded, eeg)


def test_leaky_decoder_matches_recursive_definition():
    signed = torch.tensor([[[0, 1, 1, 0, -1]]], dtype=torch.int8)
    leak = 0.5

    decoded = reconstruct(
        signed, torch.tensor([2.0]), torch.tensor([[4.0]]), polarity_mode="signed", leak=leak
    )

    expected, current = [4.0], 4.0
    for spike in signed[0, 0, 1:].tolist():
        current = leak * current + 2.0 * spike
        expected.append(current)
    assert decoded[0, 0].tolist() == pytest.approx(expected)


def test_leak_of_one_is_standard_decoding_and_cutoff_mapping():
    signed = torch.tensor([[[0, 1, -1, 1]]], dtype=torch.int8)
    args = (signed, torch.tensor([1.0]), torch.tensor([[0.0]]))

    standard = reconstruct(*args, polarity_mode="signed")
    leak_one = reconstruct(*args, polarity_mode="signed", leak=leak_from_cutoff(None, 250.0))

    assert torch.equal(standard, leak_one)
    assert leak_from_cutoff(1.0, 250.0) == pytest.approx(math.exp(-2 * math.pi / 250))
    with pytest.raises(ValueError):
        reconstruct(*args, polarity_mode="signed", leak=1.5)


def test_leaky_decoder_removes_drift_on_band_limited_signal():
    set_seed(0)
    fs, seconds = 250.0, 8.0
    time = torch.arange(int(fs * seconds), dtype=torch.float64) / fs
    noise = 0.3 * torch.randn(time.shape, dtype=torch.float64)
    eeg = (torch.sin(2 * math.pi * 10 * time) + noise)[None, None, :]
    theta = torch.tensor([0.5 * float((eeg[..., 1:] - eeg[..., :-1]).std())])

    spikes = tbr_encode_signed(eeg, theta)
    standard = reconstruct(spikes, theta, eeg[:, :, 0], polarity_mode="signed")
    leaky = reconstruct(
        spikes, theta, eeg[:, :, 0], polarity_mode="signed", leak=leak_from_cutoff(1.0, fs)
    )

    assert snr_db(eeg, leaky) > snr_db(eeg, standard)


def test_dual_binary_conversion_round_trips_and_passes_contract():
    signed = torch.tensor([[[0, 1, -1], [-1, 0, 1]]], dtype=torch.int8)

    binary = to_dual_binary(signed)

    assert binary.dtype == torch.float32
    assert binary.shape == (1, 3, 4)
    check_spike_train(binary, n_channels=4)
    assert torch.equal(from_dual_binary(binary), signed.to(torch.float32))


def test_encoder_from_calibration_profile_uses_profile_thresholds():
    calibration = np.array(
        [
            [[0.0, 1.0, 0.0, 2.0], [0.0, 2.0, 0.0, 4.0]],
            [[0.0, -1.0, 0.0, -2.0], [0.0, -2.0, 0.0, -4.0]],
        ],
        dtype=np.float32,
    )
    profile = compute_threshold_profile(
        calibration,
        ["E1", "E2"],
        subject_id="synthetic",
        k=1.0,
        settings=ThresholdSettings(
            include_mean=False,
            minimum_threshold=1e-6,
            minimum_signal_std=1e-8,
            minimum_derivative_std=1e-8,
        ),
        calibration_seconds=2.0,
        unit="synthetic units",
    )

    encoder = AdaptiveThresholdEncoder.from_profile(profile)
    spikes = encoder.encode(torch.zeros((1, 2, 5), dtype=torch.float32))

    assert encoder.subject_id == "synthetic"
    assert encoder.channel_names == ("E1", "E2")
    assert torch.allclose(encoder.thresholds, torch.from_numpy(profile.thresholds).float())
    check_spike_train(spikes, n_channels=4)


def test_threshold_vector_must_match_channels():
    with pytest.raises(ValueError, match="one threshold per channel"):
        _adaptive_encoder([1.0, 2.0], channel_names=("E1",))
    with pytest.raises(ValueError, match="positive"):
        _adaptive_encoder([0.0])


def test_supplied_per_channel_thresholds_and_fixed_encoder():
    eeg = torch.tensor([[[0.0, 2.0], [0.0, 2.0]]], dtype=torch.float32)
    adaptive = _adaptive_encoder([1.0, 2.0], channel_names=("E1", "E2"))
    fixed = FixedThresholdEncoder(1.0)

    adaptive_spikes = adaptive.encode(eeg)
    fixed_spikes = fixed.encode(eeg)

    assert from_dual_binary(adaptive_spikes)[0, :, 1].tolist() == [1.0, 0.0]
    assert from_dual_binary(fixed_spikes)[0, :, 1].tolist() == [1.0, 1.0]


def test_signed_mode_is_internal_and_does_not_pass_contract():
    eeg = torch.tensor([[[0.0, 2.0]]], dtype=torch.float32)

    signed = _adaptive_encoder([1.0], polarity_mode="signed").encode(eeg)

    assert signed.shape == (1, 1, 2)
    assert signed.dtype == torch.int8
    with pytest.raises(ContractError):
        check_spike_train(signed)


def test_fixed_seed_gives_identical_encoded_output():
    set_seed(42)
    eeg_a = torch.randn(2, 2, 5, dtype=torch.float32)
    output_a = FixedThresholdEncoder(0.5).encode(eeg_a)

    set_seed(42)
    eeg_b = torch.randn(2, 2, 5, dtype=torch.float32)
    output_b = FixedThresholdEncoder(0.5).encode(eeg_b)

    assert torch.equal(eeg_a, eeg_b)
    assert torch.equal(output_a, output_b)


def test_fidelity_metrics_on_known_values():
    original = torch.tensor([3.0, 4.0])
    reconstructed = torch.tensor([3.0, 3.0])

    assert snr_db(original, reconstructed) == pytest.approx(10 * math.log10(25.0))
    assert snr_db(original, original) == math.inf
    assert correlation(original, -original) == pytest.approx(-1.0)
    assert math.isnan(correlation(original, torch.ones(2)))
    assert set(signal_integrity(original, reconstructed)) == {"correlation", "rmse", "snr_db"}
    assert spike_density(torch.tensor([[[0, 1, -1, 0]]])) == 0.5

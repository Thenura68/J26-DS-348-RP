import numpy as np
import torch

from snn_depression.common.contracts import check_spike_train
from snn_depression.c1_encoding.decoder import reconstruct
from snn_depression.c1_encoding.encoder import (
    FixedThresholdEncoder,
    from_dual_binary,
)
from snn_depression.c1_encoding.step_forward import (
    calibration_std_thresholds,
    encode_reanchored,
    pooled_calibration_std_threshold,
    step_forward,
    step_forward_signed,
)


def test_step_forward_starts_at_x0_and_steps_baseline_once_per_sample():
    signal = torch.tensor([[[0.0, 3.0, 3.0, -3.0]]], dtype=torch.float32)

    spikes = step_forward_signed(signal, torch.tensor([1.0]))

    assert spikes.tolist() == [[[0, 1, 1, -1]]]


def test_step_forward_dual_binary_has_contract_shape_and_values():
    signal = torch.tensor([[[0.0, 2.0, -2.0]]], dtype=torch.float32)

    spikes = step_forward(signal, torch.tensor([1.0]))

    assert spikes.shape == (1, 3, 2)
    check_spike_train(spikes, n_channels=2)
    assert from_dual_binary(spikes).tolist() == [[[0.0, 1.0, -1.0]]]


def test_sf_decoder_is_cumulative_and_reconstructs_tracked_baseline():
    signal = torch.tensor([[[0.0, 3.0, 3.0, -3.0]]], dtype=torch.float32)
    thresholds = torch.tensor([1.0])

    spikes = step_forward(signal, thresholds)
    decoded = reconstruct(
        spikes, thresholds, signal[:, :, 0], polarity_mode="dual_binary"
    )

    assert decoded.tolist() == [[[0.0, 1.0, 2.0, 1.0]]]


def test_sf_thresholds_are_patient_channel_calibration_std_only():
    calibration = np.array(
        [
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0]],
            [[-2.0, 0.0, 2.0], [-4.0, 0.0, 4.0]],
        ],
        dtype=np.float32,
    )

    thresholds = calibration_std_thresholds(
        calibration, k_sf=0.5, minimum_threshold=0.01
    )

    assert thresholds.shape == (2,)
    assert thresholds[1] > thresholds[0]


def test_global_sf_threshold_uses_all_calibration_subjects():
    calibration = {
        "a": np.array([[[0.0, 2.0]]], dtype=np.float32),
        "b": np.array([[[0.0, 4.0]]], dtype=np.float32),
    }

    threshold = pooled_calibration_std_threshold(
        calibration, k_sf=0.5, minimum_threshold=0.01
    )

    assert threshold > 0.5


def test_reanchoring_recovers_true_samples_and_preserves_window_shape():
    signal = torch.tensor(
        [[[0.0, 2.5, 2.5, 2.5, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    thresholds = torch.tensor([1.0])

    signed, decoded = encode_reanchored(signal, thresholds, anchor_interval=3)

    assert signed.shape == signal.shape
    assert decoded.shape == signal.shape
    assert torch.equal(decoded[..., 3], signal[..., 3])


def test_tiny_threshold_synthetic_sine_sanity():
    fs = 250.0
    time = torch.arange(int(fs * 2.0), dtype=torch.float32) / fs
    sine = torch.sin(2.0 * torch.pi * 10.0 * time)[None, None, :]
    threshold = 0.001

    spikes = FixedThresholdEncoder(threshold).encode(sine)
    decoded = reconstruct(
        spikes,
        torch.tensor([threshold]),
        sine[:, :, 0],
        polarity_mode="dual_binary",
    )
    error = sine - decoded
    snr_db = 10.0 * torch.log10(sine.square().sum() / error.square().sum())

    assert torch.count_nonzero(spikes) > 0
    assert torch.isfinite(decoded).all()
    assert torch.isfinite(snr_db)
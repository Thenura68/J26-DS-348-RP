import math

import numpy as np
import pytest
import torch

from snn_depression.common.contracts import check_spike_train
from snn_depression.c1_encoding.calibration import ThresholdSettings
from snn_depression.c1_encoding.comparison import (
    build_methods,
    choose_encoder,
    evaluate_on_windows,
    holm_correction,
    matched_setting,
    paired_comparison,
    parameter_grid,
    select_setting,
    sweep_method,
)

FS = 250.0
PARAMS = {
    "tbr": {"k": 1.0},
    "aer": {"theta": 0.2},
    "step_forward": {"k_sf": 0.5},
    "bsa": {"filter_length": 9, "cutoff_hz": 30.0, "threshold": 0.0},
}


def _methods():
    return build_methods(
        fs=FS,
        tbr_settings=ThresholdSettings(
            include_mean=True, minimum_threshold=1e-6, minimum_signal_std=1e-8, minimum_derivative_std=1e-8
        ),
        decoder_leak=0.975,
        step_forward_minimum_threshold=1e-8,
        bsa_scale_percentile=99.0,
        bsa_minimum_scale=1e-6,
    )


def _eeg(windows=3, channels=2, samples=200, seed=0):
    rng = np.random.default_rng(seed)
    time = np.arange(samples) / FS
    signal = np.sin(2 * np.pi * 10 * time) + 0.3 * rng.normal(size=(windows, channels, samples))
    return signal.astype(np.float32)


@pytest.mark.parametrize("name", list(PARAMS))
def test_every_method_meets_the_contract_and_reconstructs(name):
    method = _methods()[name]
    windows = _eeg()

    eeg, spikes, rebuilt = evaluate_on_windows(method, windows, method.fit(windows, PARAMS[name]))

    check_spike_train(spikes, n_channels=4)
    assert spikes.shape == (3, 200, 4)
    assert rebuilt.shape == eeg.shape
    assert torch.isfinite(rebuilt).all()
    assert method.ops_per_sample(PARAMS[name]) > 0


def test_aer_uses_the_same_threshold_for_every_subject_and_tbr_adapts():
    methods = _methods()
    quiet, loud = _eeg(seed=1), 10.0 * _eeg(seed=1)

    assert torch.equal(methods["aer"].fit(quiet, {"theta": 0.2}), methods["aer"].fit(loud, {"theta": 0.2}))
    assert torch.allclose(methods["tbr"].fit(loud, {"k": 1.0}), 10.0 * methods["tbr"].fit(quiet, {"k": 1.0}))


def test_parameter_grid_and_sweep_cover_every_setting():
    grid = {"filter_length": [9, 15], "cutoff_hz": [30.0], "threshold": [0.0, 0.1]}
    assert len(parameter_grid(grid)) == 4

    curve = sweep_method(_methods()["bsa"], grid, {"a": _eeg(seed=2), "b": _eeg(seed=3)})

    assert [row["params"] for row in curve] == parameter_grid(grid)
    assert all(row["subjects_scored"] == 2 for row in curve)


def test_selection_rule_and_density_matching():
    curve = [
        {"params": {"k": 0.5}, "median_subject_snr_db": 2.0, "median_spike_density": 0.6},
        {"params": {"k": 1.0}, "median_subject_snr_db": 1.8, "median_spike_density": 0.3},
        {"params": {"k": 2.0}, "median_subject_snr_db": 0.5, "median_spike_density": 0.05},
    ]

    assert select_setting(curve, "median_subject_snr_db", 0.0)["params"] == {"k": 0.5}
    assert select_setting(curve, "sparsest_within_snr_tolerance", 0.5)["params"] == {"k": 1.0}
    assert matched_setting(curve, 0.1)["params"] == {"k": 2.0}
    with pytest.raises(ValueError):
        select_setting(curve, "best", 0.0)


def test_holm_correction_and_paired_comparison():
    assert holm_correction([0.01, 0.04, math.nan]) == pytest.approx([0.02, 0.04, math.nan], nan_ok=True)

    reference = {f"s{i}": float(i) + 1.0 for i in range(10)}
    other = {f"s{i}": float(i) for i in range(10)}
    result = paired_comparison(reference, other)

    assert result["n"] == 10 and result["median_difference"] == pytest.approx(1.0) and result["p"] < 0.01


def test_decision_rule_breaks_ties_by_cost():
    rows = [
        {"method": "tbr", "median_feature_spearman": 0.95, "ops_per_sample": 3, "density_cv": 0.2, "median_snr_db": 2.0},
        {"method": "bsa", "median_feature_spearman": 0.96, "ops_per_sample": 120, "density_cv": 0.1, "median_snr_db": 5.0},
        {"method": "aer", "median_feature_spearman": 0.80, "ops_per_sample": 3, "density_cv": 0.4, "median_snr_db": 1.0},
    ]

    tied = choose_encoder(rows, feature_rho_tie_margin=0.02)
    clear = choose_encoder(rows, feature_rho_tie_margin=0.0)

    assert tied["chosen_method"] == "tbr" and set(tied["tied_within_margin"]) == {"tbr", "bsa"}
    assert clear["chosen_method"] == "bsa"

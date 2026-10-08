import numpy as np
import pytest

from snn_depression.common.labels import phq9_to_severity
from snn_depression.data.preprocessing import bandpass_filter, make_windows
from snn_depression.data.splits import calibration_split, leave_one_subject_out


def test_phq9_mapping():
    assert phq9_to_severity(3) is None
    assert phq9_to_severity(5) == 0
    assert phq9_to_severity(14) == 1
    assert phq9_to_severity(27) == 2
    with pytest.raises(ValueError):
        phq9_to_severity(30)


def test_bandpass_removes_out_of_band_tone():
    fs = 250
    t = np.arange(0, 4, 1 / fs)
    in_band = np.sin(2 * np.pi * 10 * t)
    out_band = np.sin(2 * np.pi * 60 * t)
    filtered = bandpass_filter((in_band + out_band)[None, :], fs)[0]
    mid = slice(fs, -fs)  # ignore edges
    assert np.std(filtered[mid] - in_band[mid]) < 0.1


def test_windows_are_chronological():
    eeg = np.arange(1000, dtype=float)[None, :].repeat(4, axis=0)  # [C=4, S=1000]
    w = make_windows(eeg, fs=250, window_sec=1.0)
    assert w.shape == (4, 4, 250)
    assert w[1, 0, 0] == 250


def test_leave_one_subject_out_has_no_leakage():
    ids = ["a", "a", "b", "b", "c"]
    for subject, train, test in leave_one_subject_out(ids):
        assert set(np.asarray(ids)[test]) == {subject}
        assert subject not in set(np.asarray(ids)[train])


def test_calibration_split_is_first_20_percent():
    cal, ev = calibration_split(100, 0.2)
    assert cal.tolist() == list(range(20))
    assert ev[0] == 20 and len(ev) == 80

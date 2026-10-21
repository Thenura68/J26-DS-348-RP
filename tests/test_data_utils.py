from collections import Counter

import numpy as np
import pytest

from snn_depression.common.labels import (
    label_disagrees_with_group,
    phq9_to_severity,
    subject_targets,
)
from snn_depression.data.preprocessing import bandpass_filter, make_windows
from snn_depression.data.splits import (
    build_fold_summaries,
    calibration_split,
    leave_one_subject_out,
    stratified_subject_kfold,
    subject_folds,
)


def test_phq9_mapping_uses_cutoffs_10_and_20():
    assert [phq9_to_severity(s) for s in (0, 5, 9, 10, 14, 19, 20, 27)] == [0, 0, 0, 1, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        phq9_to_severity(30)
    with pytest.raises(ValueError):
        phq9_to_severity(-1)


def test_subject_targets_give_diagnosis_and_three_class_label():
    assert subject_targets("HC", 3) == (0, 0)
    assert subject_targets("MDD", 15) == (1, 1)
    assert subject_targets("mdd", 23) == (1, 2)
    with pytest.raises(ValueError):
        subject_targets("other", 3)


def test_label_disagreement_flags_scores_that_contradict_the_diagnosis():
    assert not label_disagrees_with_group("HC", 5)
    assert not label_disagrees_with_group("MDD", 11)
    assert label_disagrees_with_group("HC", 12)
    assert label_disagrees_with_group("MDD", 7)


def _cohort():
    ids = [f"s{i:02d}" for i in range(53)]
    labels = [0] * 29 + [1] * 16 + [2] * 8
    return ids, labels


def test_stratified_folds_test_every_subject_once_per_repeat_without_leakage():
    ids, labels = _cohort()
    folds = stratified_subject_kfold(ids, labels, n_splits=5, n_repeats=3, seed=7)

    assert len(folds) == 15
    for repeat in (1, 2, 3):
        tested = Counter(s for f in folds if f["repeat"] == repeat for s in f["test_subject_ids"])
        assert set(tested) == set(ids) and set(tested.values()) == {1}
    for fold in folds:
        assert not set(fold["train_subject_ids"]) & set(fold["test_subject_ids"])
        assert set(fold["train_subject_ids"]) | set(fold["test_subject_ids"]) == set(ids)


def test_stratified_folds_keep_every_class_in_every_test_fold():
    ids, labels = _cohort()
    label_by_id = dict(zip(ids, labels))

    for fold in stratified_subject_kfold(ids, labels, n_splits=5, n_repeats=2):
        counts = Counter(label_by_id[s] for s in fold["test_subject_ids"])
        assert set(counts) == {0, 1, 2}
        assert 5 <= counts[0] <= 6 and 3 <= counts[1] <= 4 and 1 <= counts[2] <= 2


def test_stratified_folds_are_reproducible_and_independent_of_input_order():
    ids, labels = _cohort()
    first = stratified_subject_kfold(ids, labels, n_repeats=2, seed=42)
    reversed_input = stratified_subject_kfold(ids[::-1], labels[::-1], n_repeats=2, seed=42)
    other_seed = stratified_subject_kfold(ids, labels, n_repeats=2, seed=43)

    assert first == reversed_input
    assert first[0]["test_subject_ids"] != other_seed[0]["test_subject_ids"]
    assert first[0]["test_subject_ids"] != first[5]["test_subject_ids"]


def test_stratified_folds_reject_small_classes_and_window_level_input():
    with pytest.raises(ValueError, match="at least n_splits"):
        stratified_subject_kfold([f"s{i}" for i in range(13)], [0] * 10 + [1] * 3, n_splits=5)
    with pytest.raises(ValueError, match="unique"):
        stratified_subject_kfold(["a", "a", "b"], [0, 0, 1], n_splits=2)


def test_subject_folds_follow_the_configured_strategy():
    ids, labels = _cohort()
    stratified = subject_folds(ids, labels, {"strategy": "stratified_subject_kfold", "n_splits": 5, "n_repeats": 2})
    loso = subject_folds(ids, labels, {"strategy": "leave_one_subject_out"})

    assert len(stratified) == 10
    assert len(loso) == 53 and all(len(f["test_subject_ids"]) == 1 for f in loso)
    with pytest.raises(ValueError, match="Unknown split strategy"):
        subject_folds(ids, labels, {"strategy": "random_windows"})


def test_fold_summaries_count_classes_and_reject_leakage():
    ids, labels = _cohort()
    rows = [
        {"subject_id": s, "group": "HC" if y == 0 else "MDD", "severity_target": y,
         "n_calibration_windows": 29, "n_evaluation_windows": 116}
        for s, y in zip(ids, labels)
    ]
    folds = stratified_subject_kfold(ids, labels, n_splits=5)
    summaries = build_fold_summaries(rows, folds)

    assert sum(sum(s["test_class_counts"].values()) for s in summaries) == 53
    assert summaries[0]["test_subjects"][0]["evaluation_window_count"] == 116
    leaky = [dict(folds[0], train_subject_ids=folds[0]["train_subject_ids"] + folds[0]["test_subject_ids"][:1])]
    with pytest.raises(ValueError, match="leakage"):
        build_fold_summaries(rows, leaky)


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

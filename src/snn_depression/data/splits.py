"""Subject-wise splitting. No subject may appear in both training and testing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

FOLD_STRATEGIES = ("stratified_subject_kfold", "leave_one_subject_out")


def leave_one_subject_out(subject_ids: Sequence) -> Iterator[tuple[object, np.ndarray, np.ndarray]]:
    """Yield (test_subject, train_indices, test_indices) for every subject.

    subject_ids: one entry per sample/window.
    """
    ids = np.asarray(subject_ids)
    for subject in np.unique(ids):
        test_mask = ids == subject
        yield subject, np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def stratified_subject_kfold(
    subject_ids: Sequence[str],
    labels: Sequence[int],
    *,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Stratified k-fold over SUBJECTS, optionally repeated with seeds seed, seed+1, ...

    Input has one entry per subject (one label each), not per window, so all of a
    subject's windows land in the same fold and every fold keeps the class ratio.
    Subjects are sorted first, so the result does not depend on input order.

    Returns one dict per (repeat, fold) with ``repeat``, ``fold`` (both 1-based),
    ``seed``, ``train_subject_ids`` and ``test_subject_ids``. Within each repeat every
    subject is tested exactly once.
    """
    ids = [str(subject) for subject in subject_ids]
    if len(ids) != len(labels):
        raise ValueError(f"{len(ids)} subject ids but {len(labels)} labels")
    if len(set(ids)) != len(ids):
        raise ValueError("Subject ids must be unique: pass one entry per subject, not per window")
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be >= 2 and n_repeats >= 1")
    label_by_id = dict(zip(ids, (int(label) for label in labels), strict=True))
    counts = Counter(label_by_id.values())
    too_small = {label: count for label, count in counts.items() if count < n_splits}
    if too_small:
        raise ValueError(
            f"Every class needs at least n_splits={n_splits} subjects; too small: {too_small}"
        )

    ordered = sorted(ids)
    ordered_labels = np.array([label_by_id[subject] for subject in ordered])
    folds = []
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(ordered, ordered_labels), start=1):
            folds.append(
                {
                    "repeat": repeat + 1,
                    "fold": fold,
                    "seed": seed + repeat,
                    "train_subject_ids": [ordered[i] for i in train],
                    "test_subject_ids": [ordered[i] for i in test],
                }
            )
    return folds


def subject_folds(
    subject_ids: Sequence[str], labels: Sequence[int], split_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Build folds from the ``split`` block of configs/common.yaml (same dict format for both strategies)."""
    strategy = split_config.get("strategy", "stratified_subject_kfold")
    if strategy == "stratified_subject_kfold":
        return stratified_subject_kfold(
            subject_ids,
            labels,
            n_splits=int(split_config.get("n_splits", 5)),
            n_repeats=int(split_config.get("n_repeats", 1)),
            seed=int(split_config.get("seed", 42)),
        )
    if strategy == "leave_one_subject_out":
        ids = [str(subject) for subject in subject_ids]
        return [
            {
                "repeat": 1,
                "fold": fold,
                "seed": None,
                "train_subject_ids": [ids[i] for i in train],
                "test_subject_ids": [ids[i] for i in test],
            }
            for fold, (_, train, test) in enumerate(leave_one_subject_out(ids), start=1)
        ]
    raise ValueError(f"Unknown split strategy {strategy!r}; expected one of {FOLD_STRATEGIES}")


def build_fold_summaries(
    subject_rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    *,
    label_key: str = "severity_target",
) -> list[dict[str, Any]]:
    """Check folds for leakage and describe them for ``folds.json``.

    ``subject_rows`` are manifest rows (subject_id, group, label_key, window counts).
    Raises if a subject is in both train and test, if a fold misses a subject, or if a
    subject is not tested exactly once per repeat. Each summary adds per-class subject
    counts for train and test plus the test subjects' calibration/evaluation window counts.
    """
    rows = {str(row["subject_id"]): row for row in subject_rows}
    if len(rows) != len(subject_rows):
        raise ValueError("Subject metadata contains duplicate subject IDs")
    tested: dict[int, Counter] = {}
    summaries = []
    for fold in folds:
        train, test = list(fold["train_subject_ids"]), list(fold["test_subject_ids"])
        label = f"repeat {fold['repeat']} fold {fold['fold']}"
        if set(train) & set(test):
            raise ValueError(f"Subject-wise split leakage detected in {label}")
        if set(train) | set(test) != set(rows):
            raise ValueError(f"{label} does not cover exactly the processed subjects")
        tested.setdefault(fold["repeat"], Counter()).update(test)
        for subject in test:
            row = rows[subject]
            if int(row["n_calibration_windows"]) < 1 or int(row["n_evaluation_windows"]) < 1:
                raise ValueError(f"Subject {subject} has an empty calibration or evaluation partition")
        summaries.append(
            {
                **fold,
                "train_class_counts": dict(sorted(Counter(rows[s][label_key] for s in train).items())),
                "test_class_counts": dict(sorted(Counter(rows[s][label_key] for s in test).items())),
                "train_group_counts": dict(sorted(Counter(rows[s]["group"] for s in train).items())),
                "test_subjects": [
                    {
                        "subject_id": subject,
                        "group": rows[subject]["group"],
                        label_key: rows[subject][label_key],
                        "calibration_window_count": int(rows[subject]["n_calibration_windows"]),
                        "evaluation_window_count": int(rows[subject]["n_evaluation_windows"]),
                    }
                    for subject in test
                ],
            }
        )
    for repeat, counts in tested.items():
        if set(counts) != set(rows) or any(count != 1 for count in counts.values()):
            raise ValueError(f"Repeat {repeat}: every subject must be tested exactly once")
    return summaries


def calibration_split(n_windows: int, fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split of ONE subject's windows into (calibration, evaluation) indices.

    The first ``fraction`` of windows calibrate the personal adapter; the rest are used
    only for evaluation. Windows must already be in recording order.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    n_cal = max(1, int(np.floor(n_windows * fraction)))
    if n_cal >= n_windows:
        raise ValueError(f"not enough windows ({n_windows}) for a calibration/evaluation split")
    idx = np.arange(n_windows)
    return idx[:n_cal], idx[n_cal:]

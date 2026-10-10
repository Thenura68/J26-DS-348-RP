"""PHQ-9 score → depression severity label."""

from __future__ import annotations

from .contracts import CLASS_NAMES

# Inclusive PHQ-9 ranges. Keep in sync with configs/common.yaml → labels.phq9_bins.
PHQ9_BINS: dict[str, tuple[int, int]] = {
    "mild": (5, 9),
    "moderate": (10, 14),
    "severe": (15, 27),
}


def phq9_to_severity(score: int, bins: dict[str, tuple[int, int]] | None = None) -> int | None:
    """Return the class index (0=mild, 1=moderate, 2=severe), or None if below the mild range.

    What to do with ``None`` (exclude the subject, or treat as a 'minimal' class) is set by
    ``labels.below_mild`` in configs/common.yaml.
    """
    if not 0 <= score <= 27:
        raise ValueError(f"PHQ-9 score must be in 0..27, got {score}")
    bins = bins or PHQ9_BINS
    for idx, name in enumerate(CLASS_NAMES):
        low, high = bins[name]
        if low <= score <= high:
            return idx
    return None


def subject_targets(group: str, phq9: int) -> tuple[int, int | None]:
    """Return independent diagnosis and MDD-severity targets.

    Diagnosis target is 0 for healthy control and 1 for MDD. Severity target is
    the PHQ-9 class index (0=mild, 1=moderate, 2=severe) for MDD only; it is
    ``None`` for healthy controls and for MDD scores below the mild threshold.
    """
    normalized_group = group.strip().upper()
    if normalized_group == "HC":
        if not 0 <= phq9 <= 27:
            raise ValueError(f"PHQ-9 score must be in 0..27, got {phq9}")
        return 0, None
    if normalized_group != "MDD":
        raise ValueError(f"Unknown MODMA group {group!r}; expected 'MDD' or 'HC'")
    return 1, phq9_to_severity(phq9)

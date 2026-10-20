"""PHQ-9 score → 3-class depression label, plus the MDD/HC diagnosis label."""

from __future__ import annotations

from .contracts import CLASS_NAMES

# Inclusive PHQ-9 ranges. Keep in sync with configs/common.yaml → labels.phq9_bins.
# 10 and 20 are the published PHQ-9 cut-offs (Kroenke et al., 2001): 10–19 spans
# "moderate" and "moderately severe", 20–27 is "severe". On MODMA these ranges agree
# with the psychiatrist-confirmed groups (every HC scores 0–5, every MDD patient 11–24).
PHQ9_BINS: dict[str, tuple[int, int]] = {
    "non_depressed": (0, 9),
    "moderate": (10, 19),
    "severe": (20, 27),
}


def phq9_to_severity(score: int, bins: dict[str, tuple[int, int]] | None = None) -> int:
    """Return the class index (0=non_depressed, 1=moderate, 2=severe) for a PHQ-9 score 0–27."""
    if not 0 <= score <= 27:
        raise ValueError(f"PHQ-9 score must be in 0..27, got {score}")
    bins = bins or PHQ9_BINS
    for idx, name in enumerate(CLASS_NAMES):
        low, high = bins[name]
        if low <= score <= high:
            return idx
    raise ValueError(f"PHQ-9 score {score} is not covered by the configured bins {bins}")


def subject_targets(group: str, phq9: int) -> tuple[int, int]:
    """Return (diagnosis target, severity target) for one subject.

    Diagnosis target comes from the clinical group: 0 for healthy control, 1 for MDD.
    Severity target is the 3-class PHQ-9 label from ``phq9_to_severity`` and is defined
    for every subject, healthy controls included.
    """
    normalized_group = group.strip().upper()
    if normalized_group not in {"HC", "MDD"}:
        raise ValueError(f"Unknown MODMA group {group!r}; expected 'MDD' or 'HC'")
    return int(normalized_group == "MDD"), phq9_to_severity(phq9)


def label_disagrees_with_group(group: str, phq9: int) -> bool:
    """True if the PHQ-9 class contradicts the diagnosis (HC scored ≥10, or MDD scored ≤9)."""
    diagnosis, severity = subject_targets(group, phq9)
    return (diagnosis == 0) != (severity == 0)

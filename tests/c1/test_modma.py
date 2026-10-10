import pytest

import snn_depression.data.modma as modma


def test_loader_raises_when_phq9_is_missing(monkeypatch, tmp_path):
    subject_id = "02010002"

    def subject_index():
        return (
            tmp_path,
            {subject_id: tmp_path / "02010002rest.mat"},
            {subject_id: {"type": "MDD", "PHQ-9": None}},
        )

    monkeypatch.setattr(modma, "_subject_index", subject_index)

    with pytest.raises(ValueError, match="has no PHQ-9 score"):
        modma.load_subject(subject_id)

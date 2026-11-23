"""Patient leakage and deterministic fold checks; execute on the HPC."""

from __future__ import annotations

from dataclasses import replace

import pytest

from eeg_compare.data.config import SplitConfig
from eeg_compare.data.recordings import Recording
from eeg_compare.data.splits import class_coverage, make_splits, split_for_patient


def _cohort(recording: Recording) -> list[Recording]:
    return [
        replace(
            recording,
            recording_id=f"record-{index}",
            patient_id=f"patient-{index // 2}",
            edf_sha256=f"content-{index}",
            official_split="eval" if index >= 8 else "train",
        )
        for index in range(10)
    ]


def test_assignments_are_patient_disjoint_and_order_invariant(recording: Recording) -> None:
    cohort = _cohort(recording)
    config = SplitConfig(n_folds=2)
    assignments = make_splits(cohort, config)
    assert make_splits(cohort[::-1], config) == assignments
    assert len(assignments) == 5
    assert [sum(row.validation_fold == fold for row in assignments) for fold in range(2)] == [2, 2]
    evaluation = next(row for row in assignments if row.official_split == "eval")
    assert evaluation.validation_fold == -1
    assert all(split_for_patient(evaluation, fold) == "eval" for fold in range(2))


def test_official_patient_overlap_and_duplicate_content_fail(recording: Recording) -> None:
    cohort = _cohort(recording)
    leaked = [*cohort[:-1], replace(cohort[-1], patient_id=cohort[0].patient_id)]
    with pytest.raises(ValueError, match="overlaps"):
        make_splits(leaked, SplitConfig(n_folds=2))
    duplicated = [*cohort[:-1], replace(cohort[-1], edf_sha256=cohort[0].edf_sha256)]
    with pytest.raises(ValueError, match="duplicate"):
        make_splits(duplicated, SplitConfig(n_folds=2))


def test_missing_class_coverage_is_reported_without_changing_folds(recording: Recording) -> None:
    assignments = make_splits(_cohort(recording), SplitConfig(n_folds=2))
    counts = {row.patient_id: {6: 3} for row in assignments}
    report = class_coverage(assignments, counts, SplitConfig(n_folds=2))
    assert len(report) == 6
    assert all(row["missing_classes"] == ["spsw", "gped", "pled", "eyem", "artf"] for row in report)
    with pytest.raises(ValueError, match="fewer"):
        make_splits(_cohort(recording), SplitConfig(n_folds=5))

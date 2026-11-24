"""Patient leakage and deterministic fold checks; execute on the HPC."""

from __future__ import annotations

from dataclasses import replace

import pytest

from eeg_compare.data.config import SplitConfig
from eeg_compare.data.recordings import Recording, RecordingIdentity, cluster_of
from eeg_compare.data.splits import class_coverage, make_splits, split_for_cluster


def _cohort(recording: Recording) -> list[Recording]:
    return [
        replace(
            recording,
            recording_id=f"record-{index}",
            patient_id=f"patient-{index // 2}",
            cluster_id=f"patient:patient-{index // 2}",
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
    assert all(split_for_cluster(evaluation, fold) == "eval" for fold in range(2))


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
    counts = {row.cluster_id: {6: 3} for row in assignments}
    report = class_coverage(assignments, counts, SplitConfig(n_folds=2))
    assert len(report) == 6
    assert all(row["missing_classes"] == ["spsw", "gped", "pled", "eyem", "artf"] for row in report)
    with pytest.raises(ValueError, match="fewer"):
        make_splits(_cohort(recording), SplitConfig(n_folds=5))


def _patients(recording: Recording, count: int) -> list[Recording]:
    return [
        replace(
            recording,
            recording_id=f"record-{index}",
            patient_id=f"patient-{index}",
            cluster_id=f"patient:patient-{index}",
            edf_sha256=f"content-{index}",
        )
        for index in range(count)
    ]


def test_stratified_folds_balance_the_patients_of_each_class(recording: Recording) -> None:
    cohort = _patients(recording, 20)
    presence = {
        row.cluster_id: {1, 6} if index < 5 else {2, 6} if index < 15 else {6} for index, row in enumerate(cohort)
    }
    config = SplitConfig(n_folds=5, stratify="class_presence")
    assignments = make_splits(cohort, config, presence)
    assert make_splits(cohort[::-1], config, presence) == assignments
    for fold in range(5):
        members = [row.cluster_id for row in assignments if row.validation_fold == fold]
        assert len(members) == 4
        assert sum(1 in presence[cluster] for cluster in members) == 1
        assert sum(2 in presence[cluster] for cluster in members) == 2
    with pytest.raises(ValueError, match="needs the classes"):
        make_splits(cohort, config)
    report = class_coverage(
        assignments, {cluster: dict.fromkeys(codes, 2) for cluster, codes in presence.items()}, config
    )
    validation = [row for row in report if row["split"] == "validation"]
    assert all(
        row["clusters"] == {"spsw": 1, "gped": 2, "pled": 0, "eyem": 0, "artf": 0, "bckg": 4} for row in validation
    )
    assert all(row["n_clusters"] == 4 for row in validation)


def test_patients_without_a_class_still_fill_the_folds(recording: Recording) -> None:
    cohort = _patients(recording, 6)
    config = SplitConfig(n_folds=3, stratify="class_presence")
    assignments = make_splits(cohort, config, {})
    assert sorted(sum(row.validation_fold == fold for row in assignments) for fold in range(3)) == [2, 2, 2]


def test_session_clusters_stay_in_evaluation_and_never_train(recording: Recording) -> None:
    cohort = _patients(recording, 4)
    sessions = [
        replace(
            recording,
            recording_id=f"eval-{index}",
            patient_id="",
            session_id=f"{index:03d}",
            cluster_id=f"session:{index:03d}",
            cluster_kind="session",
            edf_sha256=f"eval-content-{index}",
            official_split="eval",
        )
        for index in range(2)
    ]
    assignments = make_splits([*cohort, *sessions], SplitConfig(n_folds=2))
    evaluation = [row for row in assignments if row.official_split == "eval"]
    assert [(row.cluster_id, row.cluster_kind, row.validation_fold) for row in evaluation] == [
        ("session:000", "session", -1),
        ("session:001", "session", -1),
    ]
    assert all(row.cluster_kind == "patient" for row in assignments if row.official_split == "train")
    with pytest.raises(ValueError, match="must be patients"):
        make_splits([*cohort, replace(sessions[0], official_split="train")], SplitConfig(n_folds=2))
    with pytest.raises(ValueError, match="no cluster identity"):
        make_splits([*cohort, replace(sessions[0], cluster_id="")], SplitConfig(n_folds=2))


def test_cluster_of_uses_the_declared_evaluation_unit() -> None:
    known = RecordingIdentity("eval/a.edf", "0007", "032", "fixture")
    hidden = RecordingIdentity("eval/b.edf", "", "033", "fixture")
    assert cluster_of(known, "train", "session") == ("patient:0007", "patient")
    assert cluster_of(known, "eval", "patient") == ("patient:0007", "patient")
    assert cluster_of(hidden, "eval", "session") == ("session:033", "session")
    for identity, split, kind in ((hidden, "eval", "patient"), (hidden, "train", "session")):
        with pytest.raises(ValueError, match="needs a nonempty"):
            cluster_of(identity, split, kind)  # type: ignore[arg-type]


def test_duplicate_content_error_names_the_files_and_their_scope(recording: Recording) -> None:
    cohort = _cohort(recording)
    same_cluster = replace(cohort[1], edf_path="train/copy.edf", edf_sha256=cohort[0].edf_sha256)
    other_patient = replace(cohort[2], edf_path="train/other.edf", edf_sha256="shared", rec_sha256="rec-a")
    twin = replace(cohort[4], edf_path="train/twin.edf", edf_sha256="shared", rec_sha256="rec-b")
    leaked = replace(cohort[9], edf_path="eval/leak.edf", edf_sha256=cohort[6].edf_sha256)
    rows = [cohort[0], same_cluster, other_patient, cohort[3], twin, cohort[5], cohort[6], cohort[7], cohort[8], leaked]
    with pytest.raises(ValueError) as error:
        make_splits(rows, SplitConfig(n_folds=2))
    message = str(error.value)
    assert message.startswith("duplicate EDF content in the corpus: 3 groups")
    assert "[one cluster; same annotations] train/copy.edf = train/example.edf" in message
    assert "[different train clusters; different annotations] train/other.edf = train/twin.edf" in message
    assert "[train and evaluation; same annotations] eval/leak.edf = train/example.edf" in message
    with pytest.raises(ValueError, match="duplicate recording identity"):
        make_splits([cohort[0], replace(cohort[1], recording_id=cohort[0].recording_id)], SplitConfig(n_folds=2))

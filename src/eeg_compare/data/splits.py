"""Patient-level development folds that preserve the official evaluation set."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from eeg_compare.data.annotations import CLASSES
from eeg_compare.data.config import SplitConfig
from eeg_compare.data.provenance import content_hash
from eeg_compare.data.recordings import Recording


@dataclass(frozen=True)
class PatientSplit:
    """One patient assignment; eval patients use validation_fold=-1."""

    patient_id: str
    official_split: Literal["train", "eval"]
    validation_fold: int


def make_splits(recordings: Sequence[Recording], config: SplitConfig) -> list[PatientSplit]:
    """Hash-shuffle patients then deal them across folds, independent of input order.

    Raises on official patient leakage, duplicate EDF content or insufficient
    training patients. Folds balance patient counts, not event labels; coverage is
    reported separately so folds are never selected using evaluation labels.
    """
    patients: dict[str, set[str]] = defaultdict(set)
    seen_content: set[str] = set()
    seen_records: set[str] = set()
    for recording in recordings:
        if recording.recording_id in seen_records or recording.edf_sha256 in seen_content:
            raise ValueError("duplicate recording identity or EDF content in the corpus")
        seen_records.add(recording.recording_id)
        seen_content.add(recording.edf_sha256)
        patients[recording.patient_id].add(recording.official_split)
    if any(len(splits) != 1 for splits in patients.values()):
        raise ValueError("patient identity overlaps the official train/eval split; repair the identity map or corpus")
    training = [patient for patient, splits in patients.items() if splits == {"train"}]
    if len(training) < config.n_folds:
        raise ValueError("fewer training patients than development folds")
    training.sort(key=lambda patient: (content_hash([config.seed, patient]), patient))
    folds = {patient: index % config.n_folds for index, patient in enumerate(training)}
    return [
        PatientSplit(patient, "train" if splits == {"train"} else "eval", folds.get(patient, -1))
        for patient, splits in sorted(patients.items())
    ]


def split_for_patient(assignment: PatientSplit, validation_fold: int) -> Literal["train", "validation", "eval"]:
    """Resolve a fold-specific partition without moving official evaluation patients."""
    if assignment.official_split == "eval":
        return "eval"
    return "validation" if assignment.validation_fold == validation_fold else "train"


def class_coverage(
    assignments: Sequence[PatientSplit], counts: Mapping[str, Mapping[int, int]], config: SplitConfig
) -> list[dict[str, object]]:
    """Report event counts and missing classes per fold/partition; do not repair them silently."""
    reports: list[dict[str, object]] = []
    for fold in range(config.n_folds):
        for split in ("train", "validation", "eval"):
            totals = {code: 0 for code in CLASSES}
            for assignment in assignments:
                if split_for_patient(assignment, fold) == split:
                    for code, count in counts.get(assignment.patient_id, {}).items():
                        totals[code] += count
            reports.append(
                {
                    "validation_fold": fold,
                    "split": split,
                    "counts": {CLASSES[code]: count for code, count in totals.items()},
                    "missing_classes": [CLASSES[code] for code, count in totals.items() if not count],
                }
            )
    return reports

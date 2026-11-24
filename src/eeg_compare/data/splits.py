"""Cluster-level development folds that preserve the official evaluation set."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from eeg_compare.data.annotations import CLASSES
from eeg_compare.data.config import SplitConfig
from eeg_compare.data.provenance import content_hash
from eeg_compare.data.recordings import ClusterKind, Recording


@dataclass(frozen=True)
class ClusterSplit:
    """One cluster assignment; evaluation clusters use validation_fold=-1.

    A training cluster is a patient. An evaluation cluster is a patient or a
    session, as the configuration declares.
    """

    cluster_id: str
    cluster_kind: ClusterKind
    official_split: Literal["train", "eval"]
    validation_fold: int


def _stratified_folds(
    order: Sequence[str], presence: Mapping[str, Collection[int]], config: SplitConfig
) -> dict[str, int]:
    """Iterative stratification of clusters by the classes that each one holds.

    The class with the fewest unassigned clusters goes first. Each of its clusters
    goes to the fold that still wants most clusters of that class, then to the fold
    that wants most clusters in all, then by a seeded hash. Clusters without a class
    fill the folds that want most clusters. ``order`` fixes the visiting order, so
    the result does not depend on the input order of the recordings.
    """
    n_folds = config.n_folds
    held = {cluster: frozenset(presence.get(cluster, ())) for cluster in order}
    codes = sorted({code for classes in held.values() for code in classes})
    want_total = [len(order) / n_folds] * n_folds
    want = {code: [sum(code in held[cluster] for cluster in order) / n_folds] * n_folds for code in codes}
    folds: dict[str, int] = {}

    def place(cluster: str, code: int | None) -> None:
        def priority(fold: int) -> tuple[float, float, str]:
            wanted = want[code][fold] if code is not None else 0.0
            return wanted, want_total[fold], content_hash([config.seed, cluster, fold])

        fold = max(range(n_folds), key=priority)
        folds[cluster] = fold
        want_total[fold] -= 1
        for other in held[cluster]:
            want[other][fold] -= 1

    remaining = list(order)
    while remaining:
        left = {code: sum(code in held[cluster] for cluster in remaining) for code in codes}
        live = [code for code in codes if left[code]]
        if not live:
            for cluster in remaining:
                place(cluster, None)
            break
        rarest = min(live, key=lambda code: (left[code], code))
        for cluster in [cluster for cluster in remaining if rarest in held[cluster]]:
            place(cluster, rarest)
        remaining = [cluster for cluster in remaining if rarest not in held[cluster]]
    return folds


def make_splits(
    recordings: Sequence[Recording],
    config: SplitConfig,
    class_presence: Mapping[str, Collection[int]] | None = None,
) -> list[ClusterSplit]:
    """Assign training patients to folds, independent of the input order.

    ``class_presence`` maps a cluster ID to the class codes of its kept event
    examples. ``stratify="class_presence"`` requires it and balances the number
    of patients per class over the folds; fold sizes are then close to equal but
    not guaranteed within one. ``stratify="none"`` deals hash-ordered patients, so
    fold sizes differ by at most one and labels are not used.

    Raises on duplicate EDF content, a cluster or a known patient in both official
    splits, an inconsistent cluster kind, a training cluster that is not a patient,
    or fewer training patients than folds.
    """
    kinds: dict[str, set[str]] = defaultdict(set)
    official: dict[str, set[str]] = defaultdict(set)
    patients: dict[str, set[str]] = defaultdict(set)
    seen_content: set[str] = set()
    seen_records: set[str] = set()
    for recording in recordings:
        if recording.recording_id in seen_records or recording.edf_sha256 in seen_content:
            raise ValueError("duplicate recording identity or EDF content in the corpus")
        seen_records.add(recording.recording_id)
        seen_content.add(recording.edf_sha256)
        if not recording.cluster_id:
            raise ValueError("a recording has no cluster identity")
        kinds[recording.cluster_id].add(recording.cluster_kind)
        official[recording.cluster_id].add(recording.official_split)
        if recording.patient_id:
            patients[recording.patient_id].add(recording.official_split)
    if any(len(splits) != 1 for splits in official.values()) or any(len(splits) != 1 for splits in patients.values()):
        raise ValueError("patient identity overlaps the official train/eval split; repair the identity map or corpus")
    if any(len(values) != 1 for values in kinds.values()):
        raise ValueError("a cluster has more than one kind")
    training = [cluster for cluster, splits in official.items() if splits == {"train"}]
    if any(kinds[cluster] != {"patient"} for cluster in training):
        raise ValueError("training clusters must be patients")
    if len(training) < config.n_folds:
        raise ValueError("fewer training patients than development folds")
    training.sort(key=lambda cluster: (content_hash([config.seed, cluster]), cluster))
    if config.stratify == "class_presence":
        if class_presence is None:
            raise ValueError("stratify=class_presence needs the classes that each training patient holds")
        folds = _stratified_folds(training, class_presence, config)
    else:
        folds = {cluster: index % config.n_folds for index, cluster in enumerate(training)}
    result = []
    for cluster, splits in sorted(official.items()):
        kind: ClusterKind = "patient" if kinds[cluster] == {"patient"} else "session"
        result.append(ClusterSplit(cluster, kind, "train" if splits == {"train"} else "eval", folds.get(cluster, -1)))
    return result


def split_for_cluster(assignment: ClusterSplit, validation_fold: int) -> Literal["train", "validation", "eval"]:
    """Resolve a fold-specific partition without moving official evaluation clusters."""
    if assignment.official_split == "eval":
        return "eval"
    return "validation" if assignment.validation_fold == validation_fold else "train"


def class_coverage(
    assignments: Sequence[ClusterSplit], counts: Mapping[str, Mapping[int, int]], config: SplitConfig
) -> list[dict[str, object]]:
    """Report example and cluster counts per class, fold and partition; do not repair them silently.

    ``counts`` maps a cluster ID to its kept event examples per class code.
    ``clusters`` is the number of clusters of the partition that hold the class,
    which is the effective sample size of a patient-level resampling.
    """
    reports: list[dict[str, object]] = []
    for fold in range(config.n_folds):
        for split in ("train", "validation", "eval"):
            totals = {code: 0 for code in CLASSES}
            holders = {code: 0 for code in CLASSES}
            members = 0
            for assignment in assignments:
                if split_for_cluster(assignment, fold) == split:
                    members += 1
                    for code, count in counts.get(assignment.cluster_id, {}).items():
                        totals[code] += count
                        holders[code] += count > 0
            reports.append(
                {
                    "validation_fold": fold,
                    "split": split,
                    "n_clusters": members,
                    "counts": {CLASSES[code]: count for code, count in totals.items()},
                    "clusters": {CLASSES[code]: count for code, count in holders.items()},
                    "missing_classes": [CLASSES[code] for code, count in totals.items() if not count],
                }
            )
    return reports

"""Identical EDF files: one recording, the union of its annotation files, and one cluster.

A recording belongs to one person. When two folders hold EDF files with identical
bytes, the two folder identities name the same person, or one of them is wrong.
In both cases the folders must never be on two sides of a split, and the signal
must not count twice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from eeg_compare.data.provenance import content_hash
from eeg_compare.data.recordings import Recording


def duplicate_groups(recordings: Sequence[Recording]) -> list[list[Recording]]:
    """Groups of recordings with identical EDF bytes; members and groups are sorted by path."""
    by_content: dict[str, list[Recording]] = defaultdict(list)
    for recording in recordings:
        by_content[recording.edf_sha256].append(recording)
    groups = [sorted(group, key=lambda row: row.edf_path) for group in by_content.values() if len(group) > 1]
    return sorted(groups, key=lambda group: group[0].edf_path)


def describe_duplicates(groups: Sequence[Sequence[Recording]], limit: int = 20) -> str:
    """Name the files of every group of identical EDF content, with what the group joins.

    A group inside one cluster repeats one recording. A group across training
    patients says that two patient IDs hold the same signal. A group across the
    official splits says that a training file is also an evaluation file, which
    is a leak. The annotation files of a group can still differ.
    """
    lines = [f"duplicate EDF content in the corpus: {len(groups)} groups of identical files"]
    for group in sorted(groups, key=lambda rows: min(row.edf_path for row in rows))[:limit]:
        if len({row.official_split for row in group}) > 1:
            scope = "train and evaluation"
        elif len({row.cluster_id for row in group}) > 1:
            scope = f"different {group[0].official_split} clusters"
        else:
            scope = "one cluster"
        marks = "same annotations" if len({row.rec_sha256 for row in group}) == 1 else "different annotations"
        lines.append(f"  [{scope}; {marks}] " + " = ".join(sorted(row.edf_path for row in group)))
    if len(groups) > limit:
        lines.append(f"  ... and {len(groups) - limit} more groups")
    return "\n".join(lines)


def merge_duplicates(recordings: Sequence[Recording]) -> tuple[list[Recording], list[dict[str, Any]]]:
    """Merge every group of identical EDF files into one recording and one cluster.

    The first path of a group stays as the recording. Its copies leave the
    inventory; their EDF paths, annotation paths and annotation hashes are kept on
    the recording, so the build reads the annotation rows of all copies. The
    recording gets a new ID that binds those annotation files.

    The clusters that a group joins become one cluster, and the join is
    transitive. Every recording of the joined clusters, also one without a copy,
    gets the merged cluster ID: the kind, then the member IDs in sorted order with
    ``+`` between them (``patient:a+b``). The ``patient_id`` of a recording stays
    the identity of its own folder.

    Returns
    -------
    tuple[list[Recording], list[dict[str, Any]]]
        The remaining recordings in the input order, and one report row per
        group: the official split, the merged cluster, the clusters it joined, the
        EDF and annotation paths (the kept file first), and whether the annotation
        files are identical.

    Raises
    ------
    ValueError
        If a group joins the training and the evaluation set, joins clusters of
        two kinds, or its files were inspected with different geometry, or if a
        merged cluster ID equals another cluster ID.
    """
    groups = duplicate_groups(recordings)
    parent: dict[str, str] = {}

    def find(cluster: str) -> str:
        while parent.get(cluster, cluster) != cluster:
            cluster = parent[cluster]
        return cluster

    for group in groups:
        paths = " = ".join(row.edf_path for row in group)
        if len({row.official_split for row in group}) > 1:
            raise ValueError(
                f"identical EDF content in the training and the evaluation set ({paths}); "
                "this is a leak between the official sets and has no merge rule"
            )
        if len({row.cluster_kind for row in group}) > 1:
            raise ValueError(f"identical EDF content joins clusters of two kinds ({paths})")
        geometry = {
            (row.sfreq, row.n_samples, row.source_channels, row.channel_matrix, row.missing_channels) for row in group
        }
        if len(geometry) > 1:
            raise ValueError(f"identical EDF files were inspected with different geometry ({paths})")
        for row in group[1:]:
            first, second = find(group[0].cluster_id), find(row.cluster_id)
            if first != second:
                low, high = sorted((first, second))
                parent[high] = low

    members: dict[str, set[str]] = defaultdict(set)
    for recording in recordings:
        members[find(recording.cluster_id)].add(recording.cluster_id)
    names: dict[str, str] = {}
    for root, clusters in members.items():
        if len(clusters) == 1:
            names[root] = root
        else:
            kind = root.split(":", 1)[0]
            names[root] = f"{kind}:" + "+".join(sorted(cluster.split(":", 1)[-1] for cluster in clusters))
    if len(set(names.values())) != len(names):
        raise ValueError("a merged cluster ID equals another cluster ID")

    copies = {group[0].recording_id: group[1:] for group in groups}
    dropped = {row.recording_id for group in groups for row in group[1:]}
    merged = []
    for recording in recordings:
        if recording.recording_id in dropped:
            continue
        others = copies.get(recording.recording_id, [])
        if others:
            recording = replace(
                recording,
                recording_id=content_hash(
                    [
                        recording.edf_path,
                        recording.edf_sha256,
                        recording.rec_sha256,
                        [[row.edf_path, row.rec_sha256] for row in others],
                    ]
                ),
                duplicate_edf_paths=tuple(row.edf_path for row in others),
                duplicate_rec_paths=tuple(row.rec_path for row in others),
                duplicate_rec_sha256=tuple(row.rec_sha256 for row in others),
            )
        merged.append(replace(recording, cluster_id=names[find(recording.cluster_id)]))
    report: list[dict[str, Any]] = [
        {
            "official_split": group[0].official_split,
            "cluster_id": names[find(group[0].cluster_id)],
            "joined_clusters": sorted({row.cluster_id for row in group}),
            "edf_paths": [row.edf_path for row in group],
            "rec_paths": [row.rec_path for row in group],
            "same_annotation_file": len({row.rec_sha256 for row in group}) == 1,
        }
        for group in groups
    ]
    return merged, report

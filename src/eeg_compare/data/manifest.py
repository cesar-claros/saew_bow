"""Versioned, content-checked TUEV inventories and per-recording example shards."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import orjson
import polars as pl
from tqdm import tqdm

from eeg_compare.data.annotations import CLASSES, Annotation, parse_annotations
from eeg_compare.data.config import PipelineConfig
from eeg_compare.data.duplicates import describe_duplicates, duplicate_groups, merge_duplicates
from eeg_compare.data.examples import Example, make_examples
from eeg_compare.data.provenance import (
    content_hash,
    file_hash,
    relative_path,
    resolve_path,
    runtime_fingerprint,
    write_json,
)
from eeg_compare.data.recordings import Recording, RecordingIdentity, cluster_of, inspect_recording
from eeg_compare.data.splits import ClusterSplit, class_coverage, make_splits

SCHEMA_VERSION = 2
RECORDING_SCHEMA = {
    "recording_id": pl.String,
    "edf_path": pl.String,
    "rec_path": pl.String,
    "patient_id": pl.String,
    "session_id": pl.String,
    "identity_source": pl.String,
    "cluster_id": pl.String,
    "cluster_kind": pl.String,
    "official_split": pl.String,
    "sfreq": pl.Float64,
    "n_samples": pl.Int64,
    "source_channels": pl.List(pl.String),
    "channel_matrix": pl.List(pl.List(pl.Float64)),
    "missing_channels": pl.List(pl.String),
    "edf_sha256": pl.String,
    "rec_sha256": pl.String,
    "duplicate_edf_paths": pl.List(pl.String),
    "duplicate_rec_paths": pl.List(pl.String),
    "duplicate_rec_sha256": pl.List(pl.String),
}
EXAMPLE_SCHEMA = {
    "example_id": pl.String,
    "recording_id": pl.String,
    "patient_id": pl.String,
    "cluster_id": pl.String,
    "official_split": pl.String,
    "kind": pl.String,
    "source_example_id": pl.String,
    "annotation_sources": pl.List(pl.Int64),
    "annotation_lines": pl.List(pl.Int64),
    "annotation_channels": pl.List(pl.Int64),
    "annotation_start_seconds": pl.Float64,
    "annotation_stop_seconds": pl.Float64,
    "class_code": pl.Int64,
    "label": pl.String,
    "target": pl.Int64,
    "start_sample": pl.Int64,
    "n_samples": pl.Int64,
    "pad_left": pl.Int64,
    "pad_right": pl.Int64,
    "onset_offset_samples": pl.Float64,
    "context_class_codes": pl.List(pl.Int64),
    "conflicting_annotation": pl.Boolean,
    "duplicate_rows": pl.Int64,
    "row_weight": pl.Int64,
    "overlap_group_id": pl.String,
    "status": pl.String,
    "exclusion_reasons": pl.List(pl.String),
}
SPLIT_SCHEMA = {
    "cluster_id": pl.String,
    "cluster_kind": pl.String,
    "official_split": pl.String,
    "validation_fold": pl.Int64,
}


@dataclass(frozen=True)
class CorpusLayout:
    """Physical dataset location and the two official split subdirectories."""

    root: Path
    version: str
    train_subdir: str = "train"
    eval_subdir: str = "eval"


def read_identities(path: Path) -> dict[str, RecordingIdentity]:
    """Read explicit patient and session identities from CSV, preserving leading zeros.

    Required columns are edf_path, patient_id, session_id and identity_source.
    A row needs its evidence and at least one of the two IDs: the patient ID is
    empty where the corpus hides the patient, and the manifest build then needs
    the session ID for a declared session cluster. Missing/duplicate identities
    raise ``ValueError``; filenames are never interpreted as identities.
    """
    expected = {"edf_path", "patient_id", "session_id", "identity_source"}
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != 4 or set(reader.fieldnames) != expected:
            raise ValueError(f"identity CSV needs exactly these columns: {sorted(expected)}")
        for number, row in enumerate(reader, start=2):
            if set(row) != expected or any(value is None for value in row.values()):
                raise ValueError(f"identity CSV line {number}: malformed fields")
            values = {key: value.strip() for key, value in row.items()}
            values["edf_path"] = relative_path(values["edf_path"])
            if not values["identity_source"] or not (values["patient_id"] or values["session_id"]):
                raise ValueError(
                    f"identity CSV line {number}: identity_source and a patient_id or session_id are required"
                )
            identity = RecordingIdentity(**values)
            if identity.edf_path in result:
                raise ValueError("identity CSV contains a duplicate recording path")
            result[identity.edf_path] = identity
    return result


def discover_recordings(layout: CorpusLayout) -> list[tuple[str, Literal["train", "eval"]]]:
    """Inventory both official split trees; reject missing or overlapping directories."""
    if not layout.version.strip():
        raise ValueError("an explicit corpus version is required")
    root = layout.root.resolve()
    train = resolve_path(root, layout.train_subdir)
    evaluation = resolve_path(root, layout.eval_subdir)
    if train.is_relative_to(evaluation) or evaluation.is_relative_to(train):
        raise ValueError("official split directories must be disjoint")
    result: list[tuple[str, Literal["train", "eval"]]] = []
    for split, directory in (("train", train), ("eval", evaluation)):
        if not directory.is_dir():
            raise ValueError(f"official {split} directory does not exist")
        paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".edf")
        if not paths:
            raise ValueError(f"official {split} directory has no EDF files")
        for path in paths:
            name = path.relative_to(root).as_posix()
            resolve_path(root, name)
            result.append((name, "train" if split == "train" else "eval"))
    return result


def _tally_conflicts(tally: Counter[tuple[str, int, str]], examples: list[Example]) -> None:
    """Add the event examples of one recording to the conflict tally, in intervals and source rows.

    An example is eligible when no rule other than the conflict rule excludes it,
    so the tally gives the kept counts under both conflict policies from one build.
    """
    for example in examples:
        if example.kind != "event":
            continue
        eligible = not set(example.exclusion_reasons) - {"conflicting_annotation"}
        for field, flag in (
            ("all", True),
            ("conflicting", example.conflicting_annotation),
            ("kept_if_keep", eligible),
            ("kept_if_drop", eligible and not example.conflicting_annotation),
        ):
            if flag:
                tally[(example.official_split, example.class_code, f"{field}_intervals")] += 1
                tally[(example.official_split, example.class_code, f"{field}_rows")] += example.row_weight


def _conflict_report(tally: Counter[tuple[str, int, str]]) -> list[dict[str, object]]:
    """One row per official split and class; ``intervals`` are examples of the configured unit."""
    fields = [
        f"{field}_{measure}"
        for field in ("all", "conflicting", "kept_if_keep", "kept_if_drop")
        for measure in ("intervals", "rows")
    ]
    return [
        {"official_split": split, "label": label, **{field: tally[(split, code, field)] for field in fields}}
        for split in ("train", "eval")
        for code, label in CLASSES.items()
    ]


def class_presence(example_paths: list[Path]) -> dict[str, frozenset[int]]:
    """The class codes of the kept event examples of every cluster, read from the example shards."""
    frame = (
        pl.scan_parquet(example_paths)
        .filter((pl.col("status") == "kept") & (pl.col("kind") == "event"))
        .select("cluster_id", "class_code")
        .unique()
        .collect()
    )
    presence: dict[str, set[int]] = defaultdict(set)
    for cluster, code in frame.iter_rows():
        presence[cluster].add(code)
    return {cluster: frozenset(codes) for cluster, codes in presence.items()}


def _read_annotations(root: Path, recording: Recording) -> list[Annotation]:
    """The rows of the recording's annotation file, then the rows of the files of its identical copies.

    ``source`` is 0 for the recording's own file and k for the k-th copy. A file
    whose content changed since the inspection raises.
    """
    files = [
        (recording.rec_path, recording.rec_sha256),
        *zip(recording.duplicate_rec_paths, recording.duplicate_rec_sha256, strict=True),
    ]
    rows: list[Annotation] = []
    for source, (path, expected) in enumerate(files):
        data = resolve_path(root, path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("annotations changed while constructing the manifest")
        rows.extend(replace(row, source=source) for row in parse_annotations(data.decode("utf-8-sig")))
    return rows


def _describe_merge(entry: dict[str, Any], annotations: list[Annotation], examples: list[Example]) -> None:
    """Add to a duplicate-group report how the annotation files of the merged recording relate."""
    sources = sorted({row.source for row in annotations} | {0})
    events = [example for example in examples if example.kind == "event"]
    entry["rows_per_file"] = [sum(row.source == source for row in annotations) for source in sources]
    entry["classes_per_file"] = [
        sorted({CLASSES[row.class_code] for row in annotations if row.source == source}) for source in sources
    ]
    entry["intervals"] = len(events)
    entry["intervals_marked_in_several_files"] = sum(len(set(example.annotation_sources)) > 1 for example in events)
    entry["conflicting_intervals"] = sum(example.conflicting_annotation for example in events)


def build_manifest(
    layout: CorpusLayout, identities_path: Path, config: PipelineConfig, output_dir: Path
) -> dict[str, Any]:
    """Create an immutable manifest directory on the HPC, without reading signal samples.

    EDF/REC content is hashed; headers and annotations are validated. All
    recordings are inspected first. EDF files with identical bytes then follow
    ``[splits] duplicate_content``: ``error`` stops before the output directory
    exists, and ``merge`` keeps one recording per group with the annotation rows
    of all copies. Example shards are written one recording at a time, including
    exclusions. metadata.json is written last as the completion marker. A failure
    after the inspection leaves an incomplete output directory, which readers
    refuse. Existing output directories are never reused.
    """
    discovered = discover_recordings(layout)
    identities_hash = file_hash(identities_path)
    identities = read_identities(identities_path)
    paths = {path for path, _ in discovered}
    if paths != set(identities):
        raise ValueError(
            f"identity coverage differs from corpus: {len(paths - identities.keys())} missing, "
            f"{len(identities.keys() - paths)} unused"
        )
    eval_cluster = config.splits.eval_cluster
    patient_partitions: dict[str, set[str]] = defaultdict(set)
    for path, split in discovered:
        cluster_of(identities[path], split, eval_cluster)
        if identities[path].patient_id:
            patient_partitions[identities[path].patient_id].add(split)
    if any(len(partitions) != 1 for partitions in patient_partitions.values()):
        raise ValueError("patient identity overlaps official train/eval partitions")
    recordings = [
        inspect_recording(layout.root, identities[name], split, config.preprocessing, eval_cluster)
        for name, split in tqdm(discovered, desc="Inspecting TUEV recordings", unit="recording")
    ]
    duplicate_report: list[dict[str, Any]] = []
    groups = duplicate_groups(recordings)
    if groups and config.splits.duplicate_content == "error":
        raise ValueError(
            describe_duplicates(groups)
            + '\nSet [splits] duplicate_content = "merge" to keep one recording per group, with the annotation rows '
            "of all copies, and to join the clusters of each group."
        )
    if groups:
        recordings, duplicate_report = merge_duplicates(recordings)
    merged_by_path = {entry["edf_paths"][0]: entry for entry in duplicate_report}
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "examples").mkdir()
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    statistics: Counter[str] = Counter()
    conflicts: Counter[tuple[str, int, str]] = Counter()
    artifacts = {}
    statistics["duplicate_content_groups"] = len(groups)
    statistics["merged_duplicate_files"] = sum(len(group) - 1 for group in groups)
    for recording in tqdm(recordings, desc="Building TUEV example manifest", unit="recording"):
        annotations = _read_annotations(layout.root, recording)
        examples = make_examples(recording, annotations, config)
        shard = Path("examples") / f"{recording.recording_id}.parquet"
        pl.DataFrame([asdict(example) for example in examples], schema=EXAMPLE_SCHEMA).write_parquet(output_dir / shard)
        artifacts[shard.as_posix()] = file_hash(output_dir / shard)
        if recording.edf_path in merged_by_path:
            _describe_merge(merged_by_path[recording.edf_path], annotations, examples)
        statistics["source_annotation_rows"] += len(annotations)
        statistics["recordings_with_missing_channels"] += bool(recording.missing_channels)
        _tally_conflicts(conflicts, examples)
        kept_events = 0
        for example in examples:
            if example.kind == "displaced":
                statistics["displaced"] += 1
                continue
            statistics[example.status] += 1
            statistics["duplicate_annotation_rows"] += example.duplicate_rows
            statistics["conflicting_examples"] += example.conflicting_annotation
            for reason in example.exclusion_reasons:
                statistics[f"excluded:{reason}"] += 1
            if example.status == "kept":
                kept_events += 1
                statistics["kept_rows"] += example.row_weight
                counts[recording.cluster_id][example.class_code] += 1
        if config.examples.displaced_controls and kept_events and all(row.kind == "event" for row in examples):
            statistics["recordings_without_displaced_context"] += 1
    presence = {cluster: frozenset(code for code, count in codes.items() if count) for cluster, codes in counts.items()}
    assignments = make_splits(recordings, config.splits, presence)
    pl.DataFrame([asdict(recording) for recording in recordings], schema=RECORDING_SCHEMA).write_parquet(
        output_dir / "recordings.parquet"
    )
    pl.DataFrame([asdict(assignment) for assignment in assignments], schema=SPLIT_SCHEMA).write_parquet(
        output_dir / "splits.parquet"
    )
    for name in ("recordings.parquet", "splits.parquet"):
        artifacts[name] = file_hash(output_dir / name)
    runtime = runtime_fingerprint()
    if file_hash(identities_path) != identities_hash:
        raise ValueError("identity mapping changed while constructing the manifest")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "corpus_version": layout.version,
        "layout": {"train_subdir": layout.train_subdir, "eval_subdir": layout.eval_subdir},
        "identity_mapping_sha256": identities_hash,
        "identity_policy": "Explicit identity map; evidence supplied by its author, not independently verified by filenames.",
        "cluster_policy": {
            "train": "patient",
            "eval": eval_cluster,
            "disjointness": (
                "Checked on the supplied patient IDs of both official splits."
                if eval_cluster == "patient"
                else "Evaluation clusters are sessions without a patient identity. Disjointness from the training "
                "patients rests on the corpus documentation; sessions are not verified independent patients."
            ),
        },
        "config": config.to_dict(),
        "config_hash": content_hash(config.to_dict()),
        "runtime": runtime,
        "preprocessing_hash": content_hash([asdict(config.preprocessing), runtime]),
        "artifact_hashes": artifacts,
        "statistics": dict(statistics),
        "recording_count": len(recordings),
        "cluster_count": len(assignments),
        "training_patient_count": sum(assignment.official_split == "train" for assignment in assignments),
        "fold_sizes": [
            sum(assignment.validation_fold == fold for assignment in assignments)
            for fold in range(config.splits.n_folds)
        ],
        "label_map": {str(code - 1): label for code, label in CLASSES.items()},
        "class_coverage": class_coverage(assignments, counts, config.splits),
        "conflict_report": _conflict_report(conflicts),
        "duplicate_content": {"policy": config.splits.duplicate_content, "groups": duplicate_report},
        "signal_validation": "Pending HPC signal checks; build reads headers and annotations only.",
    }
    write_json(output_dir / "metadata.json", metadata)
    return metadata


@dataclass(frozen=True)
class Manifest:
    """Validated artifacts and their portable configuration; signals remain lazy."""

    directory: Path
    config: PipelineConfig
    recordings: tuple[Recording, ...]
    assignments: tuple[ClusterSplit, ...]
    example_paths: tuple[Path, ...]
    metadata: dict[str, Any]

    def scan_examples(self) -> pl.LazyFrame:
        """Scan only the declared shards; untracked files cannot enter a benchmark."""
        return pl.scan_parquet(list(self.example_paths))

    def require_class_coverage(self, validation_fold: int) -> None:
        """Fail before a six-class benchmark if training or validation lacks a class."""
        if not 0 <= validation_fold < self.config.splits.n_folds:
            raise ValueError("invalid validation fold")
        missing = [
            row
            for row in self.metadata["class_coverage"]
            if row["validation_fold"] == validation_fold and row["split"] != "eval" and row["missing_classes"]
        ]
        if missing:
            raise ValueError("training or validation lacks classes; inspect class_coverage before benchmarking")


def load_manifest(directory: Path) -> Manifest:
    """Verify completed artifact hashes and deterministic cluster assignments.

    Does not load EDF samples. Incomplete or modified artifacts raise; the runtime
    fingerprint is checked separately by the signal dataset before preprocessing.
    """
    metadata = orjson.loads((directory / "metadata.json").read_bytes())
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported manifest schema")
    config = PipelineConfig.from_dict(metadata["config"])
    if content_hash(config.to_dict()) != metadata["config_hash"]:
        raise ValueError("manifest configuration hash mismatch")
    if content_hash([asdict(config.preprocessing), metadata["runtime"]]) != metadata["preprocessing_hash"]:
        raise ValueError("manifest preprocessing hash mismatch")
    artifacts = metadata["artifact_hashes"]
    if not {"recordings.parquet", "splits.parquet"}.issubset(artifacts):
        raise ValueError("manifest is missing required artifacts")
    for name, expected in artifacts.items():
        if file_hash(resolve_path(directory, name)) != expected:
            raise ValueError("manifest artifact hash mismatch")
    recordings = []
    for row in pl.read_parquet(directory / "recordings.parquet").iter_rows(named=True):
        row["source_channels"] = tuple(row["source_channels"])
        row["channel_matrix"] = tuple(tuple(weights) for weights in row["channel_matrix"])
        for key in ("missing_channels", "duplicate_edf_paths", "duplicate_rec_paths", "duplicate_rec_sha256"):
            row[key] = tuple(row[key])
        recordings.append(Recording(**row))
    assignments = tuple(
        ClusterSplit(**row) for row in pl.read_parquet(directory / "splits.parquet").iter_rows(named=True)
    )
    shards = tuple(resolve_path(directory, name) for name in sorted(artifacts) if name.startswith("examples/"))
    expected_shards = {f"examples/{recording.recording_id}.parquet" for recording in recordings}
    if {name for name in artifacts if name.startswith("examples/")} != expected_shards:
        raise ValueError("example shards do not match the recording inventory")
    presence = class_presence(list(shards)) if config.splits.stratify == "class_presence" else None
    if list(assignments) != make_splits(recordings, config.splits, presence):
        raise ValueError("cluster assignments do not match the declared deterministic split")
    return Manifest(directory, config, tuple(recordings), assignments, shards, metadata)

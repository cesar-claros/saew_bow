"""Versioned, content-checked TUEV inventories and per-recording example shards."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import orjson
import polars as pl
from tqdm import tqdm

from eeg_compare.data.annotations import CLASSES, parse_annotations
from eeg_compare.data.config import PipelineConfig
from eeg_compare.data.examples import make_examples
from eeg_compare.data.provenance import (
    content_hash,
    file_hash,
    relative_path,
    resolve_path,
    runtime_fingerprint,
    write_json,
)
from eeg_compare.data.recordings import Recording, RecordingIdentity, inspect_recording
from eeg_compare.data.splits import PatientSplit, class_coverage, make_splits

SCHEMA_VERSION = 1
RECORDING_SCHEMA = {
    "recording_id": pl.String,
    "edf_path": pl.String,
    "rec_path": pl.String,
    "patient_id": pl.String,
    "session_id": pl.String,
    "identity_source": pl.String,
    "official_split": pl.String,
    "sfreq": pl.Float64,
    "n_samples": pl.Int64,
    "source_channels": pl.List(pl.String),
    "channel_matrix": pl.List(pl.List(pl.Float64)),
    "missing_channels": pl.List(pl.String),
    "edf_sha256": pl.String,
    "rec_sha256": pl.String,
}
EXAMPLE_SCHEMA = {
    "example_id": pl.String,
    "recording_id": pl.String,
    "patient_id": pl.String,
    "official_split": pl.String,
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
    "overlap_group_id": pl.String,
    "status": pl.String,
    "exclusion_reasons": pl.List(pl.String),
}
SPLIT_SCHEMA = {"patient_id": pl.String, "official_split": pl.String, "validation_fold": pl.Int64}


@dataclass(frozen=True)
class CorpusLayout:
    """Physical dataset location and the two official split subdirectories."""

    root: Path
    version: str
    train_subdir: str = "train"
    eval_subdir: str = "eval"


def read_identities(path: Path) -> dict[str, RecordingIdentity]:
    """Read explicit patient identities from CSV, preserving leading zeros.

    Required columns are edf_path, patient_id, session_id and identity_source.
    Session IDs may be empty. Missing/duplicate identities raise ``ValueError``;
    filenames are never interpreted as patient identities.
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
            if not values["patient_id"] or not values["identity_source"]:
                raise ValueError(f"identity CSV line {number}: patient_id and identity_source are required")
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


def build_manifest(
    layout: CorpusLayout, identities_path: Path, config: PipelineConfig, output_dir: Path
) -> dict[str, Any]:
    """Create an immutable manifest directory on the HPC, without reading signal samples.

    EDF/REC content is hashed; headers and annotations are validated. Example
    shards are written one recording at a time, including exclusions. metadata.json
    is written last as the completion marker. Failures leave an incomplete output
    directory, which readers refuse. Existing output directories are never reused.
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
    patient_partitions: dict[str, set[str]] = defaultdict(set)
    for path, split in discovered:
        patient_partitions[identities[path].patient_id].add(split)
    if any(len(partitions) != 1 for partitions in patient_partitions.values()):
        raise ValueError("patient identity overlaps official train/eval partitions")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "examples").mkdir()
    recordings = []
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    statistics: Counter[str] = Counter()
    artifacts = {}
    for name, split in tqdm(discovered, desc="Building TUEV example manifest", unit="recording"):
        recording = inspect_recording(layout.root, identities[name], split, config.preprocessing)
        annotation_bytes = resolve_path(layout.root, recording.rec_path).read_bytes()
        if hashlib.sha256(annotation_bytes).hexdigest() != recording.rec_sha256:
            raise ValueError("annotations changed while constructing the manifest")
        annotations = parse_annotations(annotation_bytes.decode("utf-8-sig"))
        examples = make_examples(recording, annotations, config)
        shard = Path("examples") / f"{recording.recording_id}.parquet"
        pl.DataFrame([asdict(example) for example in examples], schema=EXAMPLE_SCHEMA).write_parquet(output_dir / shard)
        artifacts[shard.as_posix()] = file_hash(output_dir / shard)
        recordings.append(recording)
        statistics["source_annotation_rows"] += len(annotations)
        statistics["recordings_with_missing_channels"] += bool(recording.missing_channels)
        for example in examples:
            statistics[example.status] += 1
            statistics["duplicate_annotation_rows"] += example.duplicate_rows
            statistics["conflicting_examples"] += example.conflicting_annotation
            for reason in example.exclusion_reasons:
                statistics[f"excluded:{reason}"] += 1
            if example.status == "kept":
                counts[recording.patient_id][example.class_code] += 1
    assignments = make_splits(recordings, config.splits)
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
        "identity_policy": "Explicit patient map; evidence supplied by its author, not independently verified by filenames.",
        "config": config.to_dict(),
        "config_hash": content_hash(config.to_dict()),
        "runtime": runtime,
        "preprocessing_hash": content_hash([asdict(config.preprocessing), runtime]),
        "artifact_hashes": artifacts,
        "statistics": dict(statistics),
        "recording_count": len(recordings),
        "patient_count": len(assignments),
        "label_map": {str(code - 1): label for code, label in CLASSES.items()},
        "class_coverage": class_coverage(assignments, counts, config.splits),
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
    assignments: tuple[PatientSplit, ...]
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
    """Verify completed artifact hashes and deterministic patient assignments.

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
        row["missing_channels"] = tuple(row["missing_channels"])
        recordings.append(Recording(**row))
    assignments = tuple(
        PatientSplit(**row) for row in pl.read_parquet(directory / "splits.parquet").iter_rows(named=True)
    )
    if list(assignments) != make_splits(recordings, config.splits):
        raise ValueError("patient assignments do not match the declared deterministic split")
    shards = tuple(resolve_path(directory, name) for name in sorted(artifacts) if name.startswith("examples/"))
    expected_shards = {f"examples/{recording.recording_id}.parquet" for recording in recordings}
    if {name for name in artifacts if name.startswith("examples/")} != expected_shards:
        raise ValueError("example shards do not match the recording inventory")
    return Manifest(directory, config, tuple(recordings), assignments, shards, metadata)

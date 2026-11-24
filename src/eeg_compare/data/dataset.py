"""Lazy, reproducible signal access with bounded per-worker memory and optional disk caching."""

from __future__ import annotations

import fcntl
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import orjson
import polars as pl
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from eeg_compare.data.examples import Example
from eeg_compare.data.manifest import Manifest, load_manifest
from eeg_compare.data.provenance import content_hash, file_hash, resolve_path, runtime_fingerprint, write_json
from eeg_compare.data.recordings import Recording, normalize_signal, read_recording, resampled_length
from eeg_compare.data.splits import split_for_cluster


@dataclass(frozen=True)
class ReaderOptions:
    """Per-worker cache limits and source verification; no disk cache by default."""

    cache_dir: Path | None = None
    memory_cache_records: int = 1
    verify_sources: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.memory_cache_records, bool) or not isinstance(self.memory_cache_records, int):
            raise ValueError("memory_cache_records must be an integer")
        if self.memory_cache_records < 0 or not isinstance(self.verify_sources, bool):
            raise ValueError("invalid reader options")


@dataclass(frozen=True)
class Partition:
    """A fold-specific selection; evaluation access must be explicit.

    ``kind="displaced"`` selects the record-only control contexts of the same
    clusters. The two kinds never mix in one dataset.
    """

    split: Literal["train", "validation", "eval"] = "train"
    validation_fold: int = 0
    allow_eval: bool = False
    kind: Literal["event", "displaced"] = "event"

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "eval"):
            raise ValueError("invalid dataset split")
        if self.kind not in ("event", "displaced"):
            raise ValueError("kind must be event or displaced")
        if not isinstance(self.allow_eval, bool):
            raise ValueError("allow_eval must be a boolean")
        if self.split == "eval" and not self.allow_eval:
            raise ValueError("evaluation access requires allow_eval=True")
        if (
            isinstance(self.validation_fold, bool)
            or not isinstance(self.validation_fold, int)
            or self.validation_fold < 0
        ):
            raise ValueError("validation_fold must be a nonnegative integer")


@dataclass(frozen=True)
class SignalExample:
    """Signal plus masks; padded samples and flat channels are never marked valid."""

    signal: NDArray[np.float32]
    valid_samples: NDArray[np.bool_]
    valid_channels: NDArray[np.bool_]


class TUEVReader:
    """Read a completed manifest's recordings on the HPC.

    Whole-record filtering gives consistent boundaries and a single resampling
    phase. Memory holds at most the configured number of processed recordings,
    plus one recording under construction. Optional disk entries are keyed by
    source content, preprocessing and runtime; checksums and POSIX file locks
    prevent corrupt or partially written entries from being consumed by workers.
    """

    def __init__(self, root: Path, manifest: Manifest, options: ReaderOptions | None = None) -> None:
        self.root = root
        self.manifest = manifest
        self.options = options or ReaderOptions()
        if content_hash(manifest.config.to_dict()) != manifest.metadata["config_hash"]:
            raise ValueError("reader configuration differs from the completed manifest")
        if (
            content_hash([asdict(manifest.config.preprocessing), manifest.metadata["runtime"]])
            != manifest.metadata["preprocessing_hash"]
        ):
            raise ValueError("reader preprocessing differs from the completed manifest")
        if runtime_fingerprint() != manifest.metadata["runtime"]:
            raise ValueError("data-pipeline code or libraries differ from the manifest; rebuild it in this environment")
        self.recordings = {recording.recording_id: recording for recording in manifest.recordings}
        self._memory: OrderedDict[str, NDArray[np.float32]] = OrderedDict()
        self._verified: dict[str, tuple[int, ...]] = {}
        self._pid = os.getpid()

    def _verify_sources(self, recording: Recording) -> None:
        if not self.options.verify_sources:
            return
        names = (recording.edf_path, recording.rec_path, *recording.duplicate_rec_paths)
        digests = (recording.edf_sha256, recording.rec_sha256, *recording.duplicate_rec_sha256)
        paths = [resolve_path(self.root, name) for name in names]
        stats = [path.stat() for path in paths]
        signature = tuple(value for stat in stats for value in (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        if self._verified.get(recording.recording_id) != signature:
            for path, expected in zip(paths, digests, strict=True):
                if file_hash(path) != expected:
                    raise ValueError("source EDF or annotations changed after manifest construction")
            self._verified[recording.recording_id] = signature

    def _validate_signal(self, data: NDArray[np.float32], recording: Recording) -> None:
        config = self.manifest.config.preprocessing
        expected = (len(config.channels), resampled_length(recording.n_samples, recording.sfreq, config.target_sfreq))
        if data.shape != expected or data.dtype != np.float32 or not np.isfinite(data).all():
            raise ValueError("cached or preprocessed recording has invalid shape, dtype or values")

    def _disk_recording(self, recording: Recording) -> NDArray[np.float32]:
        directory = self.options.cache_dir
        if directory is None:
            return read_recording(self.root, recording, self.manifest.config.preprocessing)
        directory.mkdir(parents=True, exist_ok=True)
        key = content_hash([recording.edf_sha256, self.manifest.metadata["preprocessing_hash"]])
        signal_path, marker = directory / f"{key}.npy", directory / f"{key}.json"
        with (directory / f"{key}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if marker.exists():
                    details = orjson.loads(marker.read_bytes())
                    if details["key"] != key or file_hash(signal_path) != details["sha256"]:
                        raise ValueError("recording cache checksum mismatch")
                else:
                    data = read_recording(self.root, recording, self.manifest.config.preprocessing)
                    self._validate_signal(data, recording)
                    temporary = directory / f"{key}.{uuid4().hex}.tmp"
                    with temporary.open("xb") as handle:
                        np.save(handle, data, allow_pickle=False)
                    checksum = file_hash(temporary)
                    temporary.replace(signal_path)
                    temporary_marker = directory / f"{key}.{uuid4().hex}.json.tmp"
                    write_json(temporary_marker, {"key": key, "sha256": checksum})
                    temporary_marker.replace(marker)
                result = np.load(signal_path, mmap_mode="r", allow_pickle=False)
                self._validate_signal(result, recording)
                return result
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def recording(self, recording_id: str) -> NDArray[np.float32]:
        """Return immutable preprocessed data, checking input hashes once per unchanged worker source."""
        if os.getpid() != self._pid:
            self._memory.clear()
            self._verified.clear()
            self._pid = os.getpid()
        recording = self.recordings[recording_id]
        self._verify_sources(recording)
        if recording_id in self._memory:
            self._memory.move_to_end(recording_id)
            return self._memory[recording_id]
        data = self._disk_recording(recording)
        self._validate_signal(data, recording)
        data.setflags(write=False)
        if self.options.memory_cache_records:
            self._memory[recording_id] = data
            while len(self._memory) > self.options.memory_cache_records:
                self._memory.popitem(last=False)
        return data

    def example(self, example: Example) -> SignalExample:
        """Extract real samples, normalize if requested, then apply zero padding and masks."""
        if example.status != "kept":
            raise ValueError("cannot load an excluded example")
        recording = self.recordings[example.recording_id]
        if (example.patient_id, example.cluster_id, example.official_split) != (
            recording.patient_id,
            recording.cluster_id,
            recording.official_split,
        ):
            raise ValueError("example identity does not match its recording")
        data = self.recording(example.recording_id)
        first, last = max(0, example.start_sample), min(data.shape[-1], example.start_sample + example.n_samples)
        if first >= last:
            raise ValueError("example has no real samples")
        left, right = first - example.start_sample, example.start_sample + example.n_samples - last
        if (left, right) != (example.pad_left, example.pad_right):
            raise ValueError("example padding differs from the manifest")
        values = np.array(data[:, first:last], dtype=np.float64, copy=True)
        valid_channels = np.ptp(values, axis=-1) > 0
        if self.manifest.config.preprocessing.normalization_scope == "example":
            values = normalize_signal(values, self.manifest.config.preprocessing)
        valid_channels &= np.ptp(values, axis=-1) > 0
        signal = np.zeros((data.shape[0], example.n_samples), dtype=np.float32)
        signal[:, left : left + last - first] = values
        valid_samples = np.zeros(example.n_samples, dtype=np.bool_)
        valid_samples[left : left + last - first] = True
        return SignalExample(signal, valid_samples, valid_channels)


class TUEVDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Map-style dataset with tensor signals, targets, stable IDs, row weights and validity masks.

    Only the selected partition's metadata is collected, for one example kind.
    Partitions follow cluster IDs: patients in training data, and patients or
    sessions in evaluation data as the manifest declares. Waveforms remain lazy;
    the reader and cache are worker-local. No patient balancing is implied by
    this dataset: the later training adapter must preserve its sampler explicitly.
    """

    def __init__(
        self, root: Path, manifest_dir: Path, partition: Partition | None = None, options: ReaderOptions | None = None
    ) -> None:
        self.manifest = load_manifest(manifest_dir)
        self.partition = partition or Partition()
        if self.partition.validation_fold >= self.manifest.config.splits.n_folds:
            raise ValueError("validation fold exceeds the manifest's fold count")
        clusters = [
            assignment.cluster_id
            for assignment in self.manifest.assignments
            if split_for_cluster(assignment, self.partition.validation_fold) == self.partition.split
        ]
        self.examples = (
            self.manifest.scan_examples()
            .filter(
                (pl.col("status") == "kept")
                & (pl.col("kind") == self.partition.kind)
                & pl.col("cluster_id").is_in(clusters)
            )
            .sort("recording_id", "start_sample", "example_id")
            .collect(engine="streaming")
        )
        if self.examples.is_empty():
            raise ValueError("selected partition has no retained examples")
        if self.examples["example_id"].n_unique() != self.examples.height:
            raise ValueError("duplicate example IDs in manifest")
        self.reader = TUEVReader(root, self.manifest, options)

    def __len__(self) -> int:
        return self.examples.height

    def example_metadata(self, index: int) -> Example:
        """Return the source annotation, timing and grouping information for an example."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.examples.row(index, named=True)
        for key in (
            "annotation_sources",
            "annotation_lines",
            "annotation_channels",
            "context_class_codes",
            "exclusion_reasons",
        ):
            row[key] = tuple(row[key])
        return Example(**row)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.example_metadata(index)
        example = self.reader.example(row)
        return {
            "signal": torch.from_numpy(example.signal),
            "target": torch.tensor(row.target, dtype=torch.long),
            "example_id": row.example_id,
            "row_weight": torch.tensor(row.row_weight, dtype=torch.long),
            "valid_samples": torch.from_numpy(example.valid_samples),
            "valid_channels": torch.from_numpy(example.valid_channels),
        }

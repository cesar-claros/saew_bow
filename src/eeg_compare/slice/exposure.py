"""Exposure plans: stored lists of training units with a quota per patient (slice protocol, section 3).

A plan fixes the balance between patients once, in a table, and not in a
training loop. Every arm of a fold reads the same rows, so a learner that makes
no draws and a learner that shuffles see the same data.

Two sources:

- ``whole_recordings``: 5 s intervals on a fixed grid of every training
  recording, with no use of the marks; a quota of intervals per patient.
- ``annotated_contexts``: the kept training events with their contexts; a quota
  of events per patient and class.

The units of a patient are ranked by a hash of the seed and the unit, and the
first units up to the quota are kept. So a draw has no repeats, the choice of
one unit does not depend on other units, and a larger quota holds the units of
a smaller one. The plan of a fold holds the chosen units of the training
patients of that fold only.

This module is not in ``eeg_compare.data`` on purpose: the reader refuses a
manifest when a module of that package changes, and a plan does not change how
a manifest is made.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from eeg_compare.data.manifest import Manifest
from eeg_compare.data.provenance import content_hash, file_hash, write_json
from eeg_compare.data.recordings import resampled_length

__all__ = [
    "PLAN_SCHEMA",
    "SOURCES",
    "ExposureConfig",
    "build_plans",
    "check_plans",
    "context_units",
    "draw",
    "fold_units",
    "interval_units",
    "load_plans",
    "pooled_counts",
    "write_plans",
]

SCHEMA_VERSION = 1
SOURCES = ("whole_recordings", "annotated_contexts")
PLAN_SCHEMA = {
    "fold": pl.Int64,
    "source": pl.String,
    "unit_id": pl.String,
    "recording_id": pl.String,
    "cluster_id": pl.String,
    "label": pl.String,
    "start_sample": pl.Int64,
    "n_samples": pl.Int64,
    "rank": pl.Int64,
}
Unit = dict[str, Any]


@dataclass(frozen=True)
class ExposureConfig:
    """The quotas and the seed of a plan.

    Parameters
    ----------
    interval_seconds : float
        Length of a unit of the whole-recording source; the context length of the study is 5 s.
    minutes_per_patient : float
        Quota of the whole-recording source.
    events_per_patient_class : int
        Quota of the annotated-context source.
    seed : int
        Seed of the ranking.
    """

    interval_seconds: float = 5.0
    minutes_per_patient: float = 5.0
    events_per_patient_class: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.interval_seconds > 0 or not self.minutes_per_patient > 0:
            raise ValueError("interval_seconds and minutes_per_patient must be positive")
        count = self.events_per_patient_class
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("events_per_patient_class must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if self.intervals_per_patient < 1:
            raise ValueError("the quota of the whole-recording source holds no interval")

    @property
    def intervals_per_patient(self) -> int:
        """Intervals that the quota of the whole-recording source allows per patient."""
        return int(self.minutes_per_patient * 60.0 / self.interval_seconds + 1e-9)


def _interval_samples(manifest: Manifest, config: ExposureConfig) -> int:
    samples = config.interval_seconds * manifest.config.preprocessing.target_sfreq
    if abs(samples - round(samples)) > 1e-9:
        raise ValueError("an interval must hold a whole number of samples")
    return int(round(samples))


def interval_units(manifest: Manifest, config: ExposureConfig) -> list[Unit]:
    """Every interval of the fixed grid of every training recording.

    The grid of a recording starts at its first sample. An interval lies fully
    inside its recording, so the last samples of a recording that do not fill an
    interval belong to no unit. The marks are not read.
    """
    length = _interval_samples(manifest, config)
    target = manifest.config.preprocessing.target_sfreq
    units: list[Unit] = []
    for recording in manifest.recordings:
        if recording.official_split != "train" or recording.missing_channels:
            continue
        total = resampled_length(recording.n_samples, recording.sfreq, target)
        for index in range(total // length):
            units.append(
                {
                    "unit_id": f"{recording.recording_id}:{index * length}",
                    "recording_id": recording.recording_id,
                    "cluster_id": recording.cluster_id,
                    "label": None,
                    "start_sample": index * length,
                    "n_samples": length,
                }
            )
    return units


def context_units(manifest: Manifest) -> list[Unit]:
    """Every kept event of the training set, with its context; the unit is the example.

    Raises
    ------
    ValueError
        If a kept event has zero padding. A context of a plan holds real samples
        only, so the manifest must drop the events at the edge of a recording.
    """
    frame = (
        manifest.scan_examples()
        .filter((pl.col("official_split") == "train") & (pl.col("kind") == "event") & (pl.col("status") == "kept"))
        .select(
            "example_id", "recording_id", "cluster_id", "label", "start_sample", "n_samples", "pad_left", "pad_right"
        )
        .collect()
    )
    if frame.filter((pl.col("pad_left") > 0) | (pl.col("pad_right") > 0)).height:
        raise ValueError("a kept event has zero padding; the plans need the edge policy 'drop'")
    return [
        {
            "unit_id": row["example_id"],
            "recording_id": row["recording_id"],
            "cluster_id": row["cluster_id"],
            "label": row["label"],
            "start_sample": row["start_sample"],
            "n_samples": row["n_samples"],
        }
        for row in frame.to_dicts()
    ]


def draw(units: Sequence[Unit], group: Sequence[str], quota: int, seed: int) -> list[Unit]:
    """Keep at most ``quota`` units of every group, by the rank of a hash of the seed and the unit.

    Parameters
    ----------
    units : Sequence[dict]
        Units with a ``unit_id`` and the fields of ``group``.
    group : Sequence[str]
        Fields that define a group, for example the patient, or the patient and the class.
    quota : int
        Largest number of units of a group.
    seed : int
        Seed of the ranking.

    Returns
    -------
    list[dict]
        The kept units with their ``rank`` inside the group, in the order of the
        groups and then of the rank. A larger quota keeps the same units and adds more.

    Raises
    ------
    ValueError
        If the quota is not positive or a unit appears twice.
    """
    if quota < 1:
        raise ValueError("the quota must be positive")
    if len({unit["unit_id"] for unit in units}) != len(units):
        raise ValueError("a unit appears twice")
    groups: dict[tuple[Any, ...], list[Unit]] = defaultdict(list)
    for unit in units:
        groups[tuple(unit[name] for name in group)].append(unit)
    kept: list[Unit] = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda unit: (content_hash([seed, unit["unit_id"]]), unit["unit_id"]))
        kept.extend({**unit, "rank": rank} for rank, unit in enumerate(ranked[:quota]))
    return kept


def build_plans(manifest: Manifest, config: ExposureConfig) -> pl.DataFrame:
    """The plans of every fold and both sources, as one table with the columns of ``PLAN_SCHEMA``."""
    chosen = {
        SOURCES[0]: draw(interval_units(manifest, config), ("cluster_id",), config.intervals_per_patient, config.seed),
        SOURCES[1]: draw(
            context_units(manifest), ("cluster_id", "label"), config.events_per_patient_class, config.seed
        ),
    }
    rows: list[Unit] = []
    for fold in range(manifest.config.splits.n_folds):
        training = {
            split.cluster_id
            for split in manifest.assignments
            if split.official_split == "train" and split.validation_fold != fold
        }
        for source in SOURCES:
            rows.extend(
                {"fold": fold, "source": source, **unit} for unit in chosen[source] if unit["cluster_id"] in training
            )
    return pl.DataFrame(rows, schema=PLAN_SCHEMA)


def _count_row(fold: int | str, source: str, per_patient: Counter[str], samples: int, sfreq: float) -> dict[str, Any]:
    """Units, patients, effective patients (one over the sum of the squared shares), largest patient and hours."""
    total = sum(per_patient.values())
    return {
        "fold": fold,
        "source": source,
        "units": total,
        "patients": len(per_patient),
        "effective_patients": total**2 / sum(value**2 for value in per_patient.values()),
        "largest_patient": max(per_patient.values()),
        "hours": samples / sfreq / 3600.0,
    }


def pooled_counts(plan: pl.DataFrame, sfreq: float) -> list[dict[str, Any]]:
    """The counts of every source over all folds together, with the fold named ``all``.

    A unit that several folds hold counts once. These are the counts of the
    draw before it is cut by fold, so they compare with the counts of the
    training set that the quotas were chosen on.
    """
    rows = []
    for source in SOURCES:
        units: dict[str, tuple[str, int]] = {}
        selected = plan.filter(pl.col("source") == source).select("unit_id", "cluster_id", "n_samples")
        for unit_id, cluster_id, n_samples in selected.iter_rows():
            units[unit_id] = (cluster_id, n_samples)
        if units:
            per_patient = Counter(cluster_id for cluster_id, _ in units.values())
            samples = sum(n_samples for _, n_samples in units.values())
            rows.append(_count_row("all", source, per_patient, samples, sfreq))
    return rows


def check_plans(plan: pl.DataFrame, manifest: Manifest, config: ExposureConfig) -> list[dict[str, Any]]:
    """Verify a plan against its manifest and its quotas, and count it.

    Returns
    -------
    list[dict]
        One row per fold and source: units, patients, the effective number of
        patients (one over the sum of the squared shares), the largest number of
        units of one patient, and the signal in hours.

    Raises
    ------
    ValueError
        If a unit belongs to a patient outside the training part of its fold or
        to an evaluation recording, a unit repeats, a quota is exceeded, an
        interval is off the grid or outside its recording, or a context is not a
        kept training event.
    """
    splits = {split.cluster_id: split for split in manifest.assignments}
    recordings = {recording.recording_id: recording for recording in manifest.recordings}
    target = manifest.config.preprocessing.target_sfreq
    length = _interval_samples(manifest, config)
    events = {unit["unit_id"]: unit for unit in context_units(manifest)}
    quotas = {SOURCES[0]: config.intervals_per_patient, SOURCES[1]: config.events_per_patient_class}
    seen: set[tuple[int, str, str]] = set()
    counts: dict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    quota_counts: Counter[tuple[int, str, str, str | None]] = Counter()
    samples: Counter[tuple[int, str]] = Counter()
    for row in plan.to_dicts():
        fold, source = row["fold"], row["source"]
        if source not in SOURCES or not 0 <= fold < manifest.config.splits.n_folds:
            raise ValueError("the plan holds an unknown source or fold")
        split = splits.get(row["cluster_id"])
        recording = recordings.get(row["recording_id"])
        if split is None or split.official_split != "train" or split.validation_fold == fold:
            raise ValueError("the plan holds a patient outside the training part of its fold")
        if recording is None or recording.official_split != "train" or recording.cluster_id != row["cluster_id"]:
            raise ValueError("the plan holds a recording that is not a training recording of its patient")
        if (fold, source, row["unit_id"]) in seen:
            raise ValueError("a unit repeats inside a fold and source")
        seen.add((fold, source, row["unit_id"]))
        if source == SOURCES[0]:
            total = resampled_length(recording.n_samples, recording.sfreq, target)
            inside = 0 <= row["start_sample"] and row["start_sample"] + row["n_samples"] <= total
            if row["n_samples"] != length or row["start_sample"] % length or not inside:
                raise ValueError("an interval is off the grid or outside its recording")
            key = (fold, source, row["cluster_id"], None)
        else:
            event = events.get(row["unit_id"])
            same = event is not None and all(
                event[name] == row[name] for name in ("recording_id", "label", "start_sample")
            )
            if not same:
                raise ValueError("a context of the plan is not a kept training event of the manifest")
            key = (fold, source, row["cluster_id"], row["label"])
        quota_counts[key] += 1
        if quota_counts[key] > quotas[source]:
            raise ValueError("a patient exceeds a quota")
        counts[(fold, source)][row["cluster_id"]] += 1
        samples[(fold, source)] += row["n_samples"]
    return [
        _count_row(fold, source, counts[(fold, source)], samples[(fold, source)], target)
        for fold, source in sorted(counts)
    ]


def write_plans(manifest: Manifest, config: ExposureConfig, output: Path) -> dict[str, Any]:
    """Build, check and store the plans of a manifest in a new directory.

    The directory gets ``plan.parquet`` and ``metadata.json`` (the configuration,
    the hashes of the manifest and of the plan, the counts of ``check_plans``
    and those of ``pooled_counts``).

    Raises
    ------
    ValueError
        If the directory exists.
    """
    if output.exists():
        raise ValueError(f"{output} exists; a plan is never overwritten")
    plan = build_plans(manifest, config)
    summary = check_plans(plan, manifest, config)
    output.mkdir(parents=True)
    plan.write_parquet(output / "plan.parquet")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "manifest_config_hash": manifest.metadata["config_hash"],
        "manifest_metadata_hash": file_hash(manifest.directory / "metadata.json"),
        "plan_hash": file_hash(output / "plan.parquet"),
        "summary": summary,
        "pooled": pooled_counts(plan, manifest.config.preprocessing.target_sfreq),
    }
    write_json(output / "metadata.json", metadata)
    return metadata


def load_plans(directory: Path, manifest: Manifest) -> tuple[pl.DataFrame, ExposureConfig]:
    """Read stored plans and verify them against the manifest and against a new build.

    Raises
    ------
    ValueError
        If the schema, a hash, or the manifest differs, or the stored plan is not
        the plan that the configuration gives for this manifest.
    """
    metadata = orjson.loads((directory / "metadata.json").read_bytes())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported exposure plan schema")
    if metadata["manifest_config_hash"] != manifest.metadata["config_hash"] or metadata[
        "manifest_metadata_hash"
    ] != file_hash(manifest.directory / "metadata.json"):
        raise ValueError("the plan belongs to another manifest")
    if file_hash(directory / "plan.parquet") != metadata["plan_hash"]:
        raise ValueError("the stored plan changed after it was written")
    config = ExposureConfig(**metadata["config"])
    plan = pl.read_parquet(directory / "plan.parquet")
    if not plan.equals(build_plans(manifest, config)):
        raise ValueError("the stored plan differs from a new build with its configuration")
    check_plans(plan, manifest, config)
    return plan, config


def fold_units(plan: pl.DataFrame, fold: int, source: str) -> pl.DataFrame:
    """The rows of one fold and one source, in the order of the plan."""
    if source not in SOURCES:
        raise ValueError(f"the source must be one of {SOURCES}, got {source!r}")
    return plan.filter((pl.col("fold") == fold) & (pl.col("source") == source))

"""The data of the slice: the examples, the plan intervals and their signals, from the training set only.

The examples are the kept events of the training patients. Each has one
displaced control, a context of the same recording far from every mark. The
plan intervals are the units of the whole-recording source over all folds; a
fold uses the rows of its own plan.

All signals are read in one pass, one recording at a time. The reader of this
module refuses a recording of the evaluation set (gate 2 of the protocol).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray

from eeg_compare.data.manifest import Manifest
from eeg_compare.slice.exposure import SOURCES
from eeg_compare.slice.scaling import Scales, fit_profile, profile_factors
from eeg_compare.slice.signals import RecordingSource, load_units

__all__ = [
    "CLIP_SHARE_MAX",
    "Study",
    "TrainingReader",
    "clip_shares",
    "event_table",
    "fold_interval_rows",
    "fold_parts",
    "load_study",
    "pooled_intervals",
]

CLIP_SHARE_MAX = 0.001


class TrainingReader:
    """Gives the recordings of the training set and refuses all others.

    Parameters
    ----------
    reader : RecordingSource
        The reader of the manifest.
    manifest : Manifest
        Its manifest.
    """

    def __init__(self, reader: RecordingSource, manifest: Manifest) -> None:
        self._reader = reader
        self._training = {row.recording_id for row in manifest.recordings if row.official_split == "train"}
        self.read: list[str] = []

    def recording(self, recording_id: str) -> NDArray[np.float32]:
        """The signal of a training recording.

        Raises
        ------
        ValueError
            If the recording is not in the training set.
        """
        if recording_id not in self._training:
            raise ValueError("the slice reads recordings of the training set only")
        self.read.append(recording_id)
        return self._reader.recording(recording_id)


@dataclass(frozen=True)
class Study:
    """The examples of the slice with their signals.

    Parameters
    ----------
    events : polars.DataFrame
        One row per kept training event, in a fixed order: ``example_id``,
        ``recording_id``, ``cluster_id``, ``label``, ``target``, ``row_weight``,
        ``fold`` (the fold that holds its patient out), ``start_sample``,
        ``control_start`` and ``n_samples``.
    intervals : polars.DataFrame
        The plan intervals of the whole-recording source over all folds, each once.
    plan : polars.DataFrame
        The exposure plan.
    contexts : torch.Tensor
        ``(n, C, T)`` event contexts, in the order of ``events``.
    controls : torch.Tensor
        ``(n, C, T)`` displaced control of each event.
    interval_signals : torch.Tensor
        ``(N, C, T)`` signals of ``intervals``.
    labels : tuple[str, ...]
        Class names by target.
    n_folds : int
        Folds of the manifest.
    sfreq : float
        Sampling rate of the signals in Hz.
    event_samples : tuple[int, int]
        Half-open sample range of the labeled second inside a context.
    recordings_read : int
        Recordings that the load read; all are training recordings.
    """

    events: pl.DataFrame
    intervals: pl.DataFrame
    plan: pl.DataFrame
    contexts: torch.Tensor
    controls: torch.Tensor
    interval_signals: torch.Tensor
    labels: tuple[str, ...]
    n_folds: int
    sfreq: float
    event_samples: tuple[int, int]
    recordings_read: int

    @property
    def n_classes(self) -> int:
        """Number of classes."""
        return len(self.labels)


def event_table(manifest: Manifest) -> pl.DataFrame:
    """The kept training events with their displaced control and their fold, in a fixed order.

    Raises
    ------
    ValueError
        If an event has zero padding, no control or two controls, a control lies in
        another recording, or a patient has no fold.
    """
    frame = (
        manifest.scan_examples()
        .filter((pl.col("official_split") == "train") & (pl.col("status") == "kept"))
        .select(
            "example_id",
            "kind",
            "source_example_id",
            "recording_id",
            "cluster_id",
            "label",
            "target",
            "row_weight",
            "start_sample",
            "n_samples",
            "pad_left",
            "pad_right",
        )
        .collect()
    )
    if frame.filter((pl.col("pad_left") > 0) | (pl.col("pad_right") > 0)).height:
        raise ValueError("a kept example has zero padding; the slice needs the edge policy 'drop'")
    events = frame.filter(pl.col("kind") == "event").sort("recording_id", "start_sample", "example_id")
    controls = frame.filter(pl.col("kind") == "displaced").select(
        pl.col("source_example_id").alias("example_id"),
        pl.col("start_sample").alias("control_start"),
        pl.col("recording_id").alias("control_recording"),
    )
    if controls["example_id"].n_unique() != controls.height:
        raise ValueError("an event has two displaced controls")
    folds = pl.DataFrame(
        [
            {"cluster_id": row.cluster_id, "fold": row.validation_fold}
            for row in manifest.assignments
            if row.official_split == "train"
        ],
        schema={"cluster_id": pl.String, "fold": pl.Int64},
    )
    table = (
        events.with_row_index("order")
        .join(controls, on="example_id", how="left")
        .join(folds, on="cluster_id", how="left")
        .sort("order")
    )
    if table["control_start"].null_count():
        raise ValueError("an event has no displaced control; the slice needs a manifest with displaced controls")
    if table.filter(pl.col("control_recording") != pl.col("recording_id")).height:
        raise ValueError("a displaced control lies in another recording than its event")
    if table["fold"].null_count() or table["example_id"].n_unique() != table.height:
        raise ValueError("an event has no fold, or an event appears twice")
    return table.select(
        "example_id",
        "recording_id",
        "cluster_id",
        "label",
        "target",
        "row_weight",
        "fold",
        "start_sample",
        "control_start",
        "n_samples",
    )


def pooled_intervals(plan: pl.DataFrame) -> pl.DataFrame:
    """The units of the whole-recording source over all folds, each once, by recording and start."""
    return (
        plan.filter(pl.col("source") == SOURCES[0])
        .select("unit_id", "recording_id", "cluster_id", "start_sample", "n_samples")
        .unique(subset="unit_id", maintain_order=True)
        .sort("recording_id", "start_sample")
    )


def load_study(manifest: Manifest, plan: pl.DataFrame, reader: RecordingSource) -> Study:
    """Read the contexts, the controls and the plan intervals of the training set.

    Parameters
    ----------
    manifest : Manifest
        The manifest of the study.
    plan : polars.DataFrame
        Its verified exposure plan (``load_plans``).
    reader : RecordingSource
        The reader of the manifest. It is wrapped, so that an evaluation recording raises.

    Raises
    ------
    ValueError
        If the classes are not numbered from zero without a gap, the contexts
        and the intervals differ in length, or a context is not the length that
        the manifest states.
    """
    events = event_table(manifest)
    intervals = pooled_intervals(plan)
    names = events.select("target", "label").unique().sort("target")
    if names["target"].to_list() != list(range(names.height)):
        raise ValueError("the targets of the kept events must be 0 to n - 1 with one label each")
    if (
        set(events["n_samples"].to_list()) != set(intervals["n_samples"].to_list())
        or events["n_samples"].n_unique() != 1
    ):
        raise ValueError("the contexts and the plan intervals must have one length")
    sfreq = manifest.config.preprocessing.target_sfreq
    before = manifest.config.examples.before_seconds
    event_samples = (round(before * sfreq), round((before + 1.0) * sfreq))
    if (
        events["n_samples"][0] != manifest.config.examples.sample_count(sfreq)
        or event_samples[1] > events["n_samples"][0]
    ):
        raise ValueError("a context does not have the length of the manifest, or holds no labeled second")
    units = pl.concat(
        [
            events.select("recording_id", "start_sample", "n_samples"),
            events.select("recording_id", pl.col("control_start").alias("start_sample"), "n_samples"),
            intervals.select("recording_id", "start_sample", "n_samples"),
        ]
    )
    guarded = TrainingReader(reader, manifest)
    signals = load_units(guarded, units)
    count = events.height
    return Study(
        events=events,
        intervals=intervals,
        plan=plan,
        contexts=signals[:count],
        controls=signals[count : 2 * count],
        interval_signals=signals[2 * count :],
        labels=tuple(names["label"].to_list()),
        n_folds=manifest.config.splits.n_folds,
        sfreq=float(sfreq),
        event_samples=event_samples,
        recordings_read=len(set(guarded.read)),
    )


def fold_parts(study: Study, fold: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The rows of ``events`` that train and that validate in a fold.

    Raises
    ------
    ValueError
        If the fold does not exist, a part is empty, or a patient is in both parts.
    """
    if not 0 <= fold < study.n_folds:
        raise ValueError(f"fold {fold} does not exist")
    held_out = study.events["fold"].to_numpy() == fold
    train, validation = np.flatnonzero(~held_out), np.flatnonzero(held_out)
    patients = study.events["cluster_id"].to_numpy()
    if not train.size or not validation.size:
        raise ValueError(f"fold {fold} has an empty part")
    if set(patients[train].tolist()) & set(patients[validation].tolist()):
        raise ValueError(f"a patient is in both parts of fold {fold}")
    return train, validation


def fold_interval_rows(study: Study, fold: int) -> NDArray[np.int64]:
    """The rows of ``intervals`` that the plan of a fold holds, in the order of ``intervals``.

    Raises
    ------
    ValueError
        If the plan of the fold is empty or holds a patient that the fold holds out.
    """
    chosen = study.plan.filter((pl.col("fold") == fold) & (pl.col("source") == SOURCES[0]))
    position = {unit_id: row for row, unit_id in enumerate(study.intervals["unit_id"].to_list())}
    rows = np.asarray(sorted(position[unit_id] for unit_id in chosen["unit_id"].to_list()), dtype=np.int64)
    if not rows.size:
        raise ValueError(f"the plan of fold {fold} holds no interval")
    held_out = set(study.events.filter(pl.col("fold") == fold)["cluster_id"].to_list())
    if held_out & set(chosen["cluster_id"].to_list()):
        raise ValueError(f"the plan of fold {fold} holds a patient that the fold holds out")
    return rows


def clip_shares(
    study: Study, manifest: Manifest, scales: Scales, clip: float, cap: float = 3.0, chunk: int = 1024
) -> list[dict[str, Any]]:
    """The share of plan samples at the clip, by fold, for the rule of the ``profile`` input.

    The ``profile`` input is computed from the primary input. It differs from a
    second read of the recording only for a sample at the clip in a channel
    whose factor is below 1 (protocol, section 16.1). If the share of such
    samples is above ``CLIP_SHARE_MAX``, the comparison of the two input levels
    is reported as approximate.

    Returns
    -------
    list[dict]
        Per fold: ``share_at_clip``, ``share_at_clip_factor_below_one`` and ``approximate``.
    """
    signals = study.interval_signals
    at_clip = torch.cat(
        [(signals[first : first + chunk].abs() >= clip).sum(dim=-1) for first in range(0, signals.shape[0], chunk)]
    ).numpy()
    recordings = study.intervals["recording_id"].to_list()
    fold_of = {row.cluster_id: row.validation_fold for row in manifest.assignments if row.official_split == "train"}
    out = []
    for fold in range(study.n_folds):
        training = [
            row.recording_id
            for row in manifest.recordings
            if row.official_split == "train" and not row.missing_channels and fold_of[row.cluster_id] != fold
        ]
        factors = profile_factors(scales, fit_profile(scales, training, cap), training)
        rows = fold_interval_rows(study, fold)
        below = np.stack([factors[recordings[row]] < 1.0 for row in rows.tolist()])
        samples = float(rows.size * signals.shape[1] * signals.shape[2])
        share = float((at_clip[rows] * below).sum()) / samples
        out.append(
            {
                "fold": fold,
                "share_at_clip": float(at_clip[rows].sum()) / samples,
                "share_at_clip_factor_below_one": share,
                "approximate": share > CLIP_SHARE_MAX,
            }
        )
    return out

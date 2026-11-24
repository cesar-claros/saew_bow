"""Deterministic onset contexts with annotation, overlap and exclusion provenance."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import accumulate
from typing import Literal

from eeg_compare.data.annotations import CLASSES, MONTAGE, Annotation
from eeg_compare.data.config import PipelineConfig
from eeg_compare.data.provenance import content_hash
from eeg_compare.data.recordings import Recording, resampled_length

BACKGROUND = next(code for code, label in CLASSES.items() if label == "bckg")
CONFLICT_REASONS = ("conflicting_annotation", "background_with_other_class")


def conflict_exclusion(policy: str, class_code: int, partners: Sequence[int]) -> str | None:
    """The reason for which a conflict rule excludes an interval, or ``None`` when it keeps it.

    ``partners`` are the other classes whose annotated intervals overlap the
    interval on any channel. ``keep`` excludes nothing and ``drop`` excludes every
    interval with a partner. ``background_yields`` reads a background mark as
    "no event on this channel": a background interval with a partner is excluded
    as ``background_with_other_class``, an interval of another class stays when
    background is its only partner, and two classes other than background still
    exclude each other.

    Raises
    ------
    ValueError
        For an unknown policy.
    """
    if policy not in ("keep", "drop", "background_yields"):
        raise ValueError(f"unknown conflict policy: {policy}")
    if policy == "keep" or not partners:
        return None
    if policy == "drop":
        return "conflicting_annotation"
    if class_code == BACKGROUND:
        return "background_with_other_class"
    return "conflicting_annotation" if any(code != BACKGROUND for code in partners) else None


@dataclass(frozen=True)
class Example:
    """One classified context; target labels are zero-based, source codes one-based.

    ``kind="event"`` is a context around an annotated interval. ``row_weight`` is
    the number of its source annotation rows, exact duplicates included, so a
    row-weighted score counts each row as the published row-level protocol does.

    ``kind="displaced"`` is a record-only control: a context of the same recording
    away from every mark. It carries the class of ``source_example_id`` and that
    example's ``row_weight``, has no annotation rows or channels, and its
    annotation times are the pseudo-onset that keeps the event geometry.

    ``annotation_sources`` and ``annotation_lines`` are parallel: row i of the
    example is line ``annotation_lines[i]`` of annotation file
    ``annotation_sources[i]`` (0 is the file of the recording; k is the file of
    its k-th identical copy).

    ``conflict_class_codes`` are the other classes whose annotated intervals
    overlap the interval of the example on any channel, whatever the conflict
    rule; ``conflicting_annotation`` says that there is one. The configured rule
    then decides the exclusion (``conflict_exclusion``).
    """

    example_id: str
    recording_id: str
    patient_id: str
    cluster_id: str
    official_split: Literal["train", "eval"]
    kind: Literal["event", "displaced"]
    source_example_id: str
    annotation_sources: tuple[int, ...]
    annotation_lines: tuple[int, ...]
    annotation_channels: tuple[int, ...]
    annotation_start_seconds: float
    annotation_stop_seconds: float
    class_code: int
    label: str
    target: int
    start_sample: int
    n_samples: int
    pad_left: int
    pad_right: int
    onset_offset_samples: float
    context_class_codes: tuple[int, ...]
    conflict_class_codes: tuple[int, ...]
    conflicting_annotation: bool
    duplicate_rows: int
    row_weight: int
    overlap_group_id: str
    status: Literal["kept", "excluded"]
    exclusion_reasons: tuple[str, ...]


class _IntervalLabels:
    """Per-class prefix maxima answer overlap queries without pairwise scans."""

    def __init__(self, annotations: Sequence[Annotation]) -> None:
        self.index: dict[int, tuple[list[float], list[float]]] = {}
        for code in CLASSES:
            intervals = sorted((row.start_seconds, row.stop_seconds) for row in annotations if row.class_code == code)
            self.index[code] = (
                [start for start, _ in intervals],
                list(accumulate((stop for _, stop in intervals), max)),
            )

    def labels(self, start: float, stop: float) -> tuple[int, ...]:
        result = []
        for code, (starts, max_stops) in self.index.items():
            index = bisect_left(starts, stop) - 1
            if index >= 0 and max_stops[index] > start:
                result.append(code)
        return tuple(result)


def _with_overlap_groups(examples: Sequence[Example], recording_id: str) -> list[Example]:
    """Label connected components of half-open context intervals, in start order."""
    rightmost: int | None = None
    group_id = ""
    grouped = []
    for example in sorted(examples, key=lambda row: (row.start_sample, row.example_id)):
        if rightmost is None or example.start_sample >= rightmost:
            group_id = content_hash([recording_id, "overlap", example.start_sample, example.example_id])
            rightmost = example.start_sample + example.n_samples
        else:
            rightmost = max(rightmost, example.start_sample + example.n_samples)
        grouped.append(replace(example, overlap_group_id=group_id))
    return grouped


def _free_starts(annotations: Sequence[Annotation], sfreq: float, n_samples: int, total: int, gap: int) -> list[range]:
    """Context starts whose every sample is at least ``gap`` samples from every annotation row.

    A context ``[s, s + n_samples)`` is blocked by a mark ``[low, high)`` when
    ``s + n_samples + gap > low`` and ``s < high + gap``. The result is the ordered
    complement of the blocked starts within ``[0, total - n_samples]``.
    """
    last = total - n_samples
    if last < 0:
        return []
    blocked = sorted(
        (
            math.floor(row.start_seconds * sfreq) - n_samples - gap + 1,
            math.ceil(row.stop_seconds * sfreq) + gap,
        )
        for row in annotations
    )
    free = []
    cursor = 0
    for low, high in blocked:
        if low > cursor:
            free.append(range(cursor, min(low, last + 1)))
        cursor = max(cursor, high)
        if cursor > last:
            break
    if cursor <= last:
        free.append(range(cursor, last + 1))
    return [span for span in free if len(span)]


def _displaced(
    recording: Recording,
    annotations: Sequence[Annotation],
    events: Sequence[Example],
    config: PipelineConfig,
    total: int,
) -> list[Example]:
    """One control context per kept event, drawn by a hash of the event ID.

    The draw is uniform over the free starts of the recording, so two events can
    receive overlapping contexts; overlap groups record that. A recording without
    a free start yields no control. Unannotated time stays unlabeled: the class of
    a control is the class of its source event, for the record-only control alone.
    """
    sfreq = config.preprocessing.target_sfreq
    gap_seconds = config.examples.displaced_gap_seconds
    free = _free_starts(annotations, sfreq, config.examples.sample_count(sfreq), total, math.ceil(gap_seconds * sfreq))
    available = sum(len(span) for span in free)
    if not available:
        return []
    result = []
    for event in events:
        if event.status != "kept":
            continue
        offset = int(content_hash([event.example_id, "displaced"])[:15], 16) % available
        for span in free:
            if offset < len(span):
                break
            offset -= len(span)
        start_sample = span[offset]
        onset = start_sample / sfreq + config.examples.before_seconds
        result.append(
            replace(
                event,
                example_id=content_hash([recording.recording_id, "displaced", event.example_id, gap_seconds, sfreq]),
                kind="displaced",
                source_example_id=event.example_id,
                annotation_sources=(),
                annotation_lines=(),
                annotation_channels=(),
                annotation_start_seconds=onset,
                annotation_stop_seconds=onset + event.annotation_stop_seconds - event.annotation_start_seconds,
                start_sample=start_sample,
                pad_left=0,
                pad_right=0,
                onset_offset_samples=onset * sfreq - start_sample,
                context_class_codes=(),
                conflict_class_codes=(),
                conflicting_annotation=False,
                duplicate_rows=0,
                overlap_group_id="",
            )
        )
    return _with_overlap_groups(result, recording.recording_id)


def make_examples(recording: Recording, annotations: Sequence[Annotation], config: PipelineConfig) -> list[Example]:
    """Build contexts and retain every exclusion; never generate unannotated negatives.

    Annotation intervals must lie inside the recording. Context starts round to
    the nearest target sample, with exact half-sample ties toward positive infinity.
    IDs depend on semantic annotation identity and context geometry, not row order
    and not the edge or conflict rule, so one example has one ID in manifests that
    differ in those rules. Overlap groups are connected components of half-open
    context intervals,
    computed within each kind. With ``displaced_controls``, the control contexts
    of the kept events follow the event examples in the result.
    """
    sfreq = config.preprocessing.target_sfreq
    n_samples = config.examples.sample_count(sfreq)
    total = resampled_length(recording.n_samples, recording.sfreq, sfreq)
    groups: dict[tuple[float, float, int, int], list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        if annotation.stop_seconds > recording.duration_seconds + 1e-8:
            raise ValueError(f"annotation line {annotation.line_number} exceeds the recording duration")
        key = (
            annotation.start_seconds,
            annotation.stop_seconds,
            annotation.class_code,
            annotation.channel if config.examples.unit == "annotation" else -1,
        )
        groups[key].append(annotation)
    intervals = _IntervalLabels(annotations)
    result = []
    for key, group in sorted(groups.items()):
        start, stop, code, _ = key
        start_sample = math.floor((start - config.examples.before_seconds) * sfreq + 0.5)
        pad_left, pad_right = max(0, -start_sample), max(0, start_sample + n_samples - total)
        channels = tuple(sorted({annotation.channel for annotation in group}))
        partners = tuple(label for label in intervals.labels(start, stop) if label != code)
        reasons = []
        if recording.missing_channels:
            reasons.append("missing_recording_channels")
        if not any(MONTAGE[channel] in config.preprocessing.channels for channel in channels):
            reasons.append("annotated_channels_excluded")
        if (pad_left or pad_right) and config.examples.edge_policy == "drop":
            reasons.append("context_outside_recording")
        conflict_reason = conflict_exclusion(config.examples.conflict_policy, code, partners)
        if conflict_reason is not None:
            reasons.append(conflict_reason)
        identifier = content_hash([recording.recording_id, key, config.examples.context_identity(), sfreq])
        rows = sorted((annotation.source, annotation.line_number) for annotation in group)
        result.append(
            Example(
                example_id=identifier,
                recording_id=recording.recording_id,
                patient_id=recording.patient_id,
                cluster_id=recording.cluster_id,
                official_split=recording.official_split,
                kind="event",
                source_example_id="",
                annotation_sources=tuple(source for source, _ in rows),
                annotation_lines=tuple(line for _, line in rows),
                annotation_channels=channels,
                annotation_start_seconds=start,
                annotation_stop_seconds=stop,
                class_code=code,
                label=CLASSES[code],
                target=code - 1,
                start_sample=start_sample,
                n_samples=n_samples,
                pad_left=pad_left,
                pad_right=pad_right,
                onset_offset_samples=start * sfreq - start_sample,
                context_class_codes=intervals.labels(start_sample / sfreq, (start_sample + n_samples) / sfreq),
                conflict_class_codes=partners,
                conflicting_annotation=bool(partners),
                duplicate_rows=len(group) - len(channels),
                row_weight=len(group),
                overlap_group_id="",
                status="excluded" if reasons else "kept",
                exclusion_reasons=tuple(reasons),
            )
        )
    events = _with_overlap_groups(result, recording.recording_id)
    if not config.examples.displaced_controls:
        return events
    return [*events, *_displaced(recording, annotations, events, config, total)]

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


@dataclass(frozen=True)
class Example:
    """One classified context; target labels are zero-based, source codes one-based."""

    example_id: str
    recording_id: str
    patient_id: str
    official_split: Literal["train", "eval"]
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
    conflicting_annotation: bool
    duplicate_rows: int
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


def make_examples(recording: Recording, annotations: Sequence[Annotation], config: PipelineConfig) -> list[Example]:
    """Build contexts and retain every exclusion; never generate unannotated negatives.

    Annotation intervals must lie inside the recording. Context starts round to
    the nearest target sample, with exact half-sample ties toward positive infinity.
    IDs depend on semantic annotation identity and context configuration, not row
    order. Overlap groups are connected components of half-open context intervals.
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
        conflict = any(label != code for label in intervals.labels(start, stop))
        reasons = []
        if recording.missing_channels:
            reasons.append("missing_recording_channels")
        if not any(MONTAGE[channel] in config.preprocessing.channels for channel in channels):
            reasons.append("annotated_channels_excluded")
        if (pad_left or pad_right) and config.examples.edge_policy == "drop":
            reasons.append("context_outside_recording")
        if conflict and config.examples.conflict_policy == "drop":
            reasons.append("conflicting_annotation")
        identifier = content_hash([recording.recording_id, key, config.to_dict()["examples"], sfreq])
        result.append(
            Example(
                example_id=identifier,
                recording_id=recording.recording_id,
                patient_id=recording.patient_id,
                official_split=recording.official_split,
                annotation_lines=tuple(sorted(annotation.line_number for annotation in group)),
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
                conflicting_annotation=conflict,
                duplicate_rows=len(group) - len(channels),
                overlap_group_id="",
                status="excluded" if reasons else "kept",
                exclusion_reasons=tuple(reasons),
            )
        )
    rightmost: int | None = None
    group_id = ""
    grouped = []
    for example in sorted(result, key=lambda row: (row.start_sample, row.example_id)):
        if rightmost is None or example.start_sample >= rightmost:
            group_id = content_hash([recording.recording_id, "overlap", example.start_sample, example.example_id])
            rightmost = example.start_sample + example.n_samples
        else:
            rightmost = max(rightmost, example.start_sample + example.n_samples)
        grouped.append(replace(example, overlap_group_id=group_id))
    return grouped

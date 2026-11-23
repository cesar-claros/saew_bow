"""Annotation identity, sample-clock and exclusion-policy checks for the HPC."""

from __future__ import annotations

from dataclasses import replace

import pytest

from eeg_compare.data.annotations import MONTAGE, parse_annotations
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig
from eeg_compare.data.examples import make_examples
from eeg_compare.data.recordings import Recording


@pytest.mark.parametrize(
    "line",
    [
        "0,nan,1,1",
        "0,0,inf,1",
        "0,-1,1,1",
        "0,1,1,1",
        "0,2,1,1",
        "0,0,1,1.5",
        "0.5,0,1,1",
        "22,0,1,1",
        "-1,0,1,1",
        "0,0,1,7",
        "0,0,1",
        "0,word,1,1",
    ],
)
def test_invalid_annotations_fail_with_line_number(line: str) -> None:
    with pytest.raises(ValueError, match="line 2"):
        parse_annotations("\n" + line)


def test_source_line_numbers_and_ear_channels_survive() -> None:
    rows = parse_annotations("\n8,0.25,1.5,1.0\n\n13,2,3,6\n")
    assert [(row.line_number, row.channel, row.class_code) for row in rows] == [(2, 8, 1), (4, 13, 6)]


def test_context_clock_does_not_round_the_annotation(recording: Recording) -> None:
    example = make_examples(recording, parse_annotations("0,2.003,3,1"), PipelineConfig())[0]
    assert example.start_sample == 1
    assert example.n_samples == 1250
    assert example.onset_offset_samples == pytest.approx(499.75)
    assert example.annotation_start_seconds == 2.003


def test_duplicate_and_multichannel_marks_have_traceable_identity(recording: Recording) -> None:
    rows = parse_annotations("0,3,4,1\n0,3,4,1\n1,3,4,1\n0,10,11,6")
    examples = make_examples(recording, rows, PipelineConfig())
    assert len(examples) == 2
    first = examples[0]
    assert first.annotation_lines == (1, 2, 3)
    assert first.annotation_channels == (0, 1)
    assert first.duplicate_rows == 1
    assert first.overlap_group_id != examples[1].overlap_group_id
    assert [row.example_id for row in make_examples(recording, rows[::-1], PipelineConfig())] == [
        row.example_id for row in examples
    ]
    separate = make_examples(recording, rows, PipelineConfig(examples=ExampleConfig(unit="annotation")))
    assert len(separate) == 3


def test_conflicts_use_half_open_annotations_not_contexts(recording: Recording) -> None:
    overlapping = parse_annotations("0,3,4,1\n1,3.5,4.5,2")
    examples = make_examples(recording, overlapping, PipelineConfig())
    assert all(row.exclusion_reasons == ("conflicting_annotation",) for row in examples)
    assert len({row.overlap_group_id for row in examples}) == 1
    retained = make_examples(recording, overlapping, PipelineConfig(examples=ExampleConfig(conflict_policy="keep")))
    assert all(row.status == "kept" and row.conflicting_annotation for row in retained)
    touching = make_examples(recording, parse_annotations("0,3,4,1\n1,4,5,2"), PipelineConfig())
    assert all(not row.conflicting_annotation and row.context_class_codes == (1, 2) for row in touching)


def test_edges_are_excluded_or_padded_without_moving_onset(recording: Recording) -> None:
    rows = parse_annotations("0,0.5,1,1\n0,19,20,6")
    excluded = make_examples(recording, rows, PipelineConfig())
    kept = make_examples(recording, rows, PipelineConfig(examples=ExampleConfig(edge_policy="zero_pad")))
    assert [row.start_sample for row in excluded] == [row.start_sample for row in kept] == [-375, 4250]
    assert [(row.pad_left, row.pad_right) for row in kept] == [(375, 0), (0, 500)]
    assert all(row.status == "excluded" for row in excluded)
    assert all(row.status == "kept" for row in kept)


def test_channel_subsets_never_reinterpret_annotation_indices(recording: Recording) -> None:
    channels = tuple(name for index, name in enumerate(MONTAGE) if index not in (8, 13))
    config = PipelineConfig(preprocessing=PreprocessingConfig(channels=channels))
    examples = make_examples(recording, parse_annotations("8,3,4,1\n21,8,9,6"), config)
    assert examples[0].exclusion_reasons == ("annotated_channels_excluded",)
    assert examples[1].annotation_channels == (21,) and examples[1].status == "kept"
    missing = make_examples(replace(recording, missing_channels=("FP1-F7",)), parse_annotations("0,3,4,1"), config)
    assert "missing_recording_channels" in missing[0].exclusion_reasons


def test_bad_duration_and_fractional_example_length_fail(recording: Recording) -> None:
    with pytest.raises(ValueError, match="duration"):
        make_examples(recording, parse_annotations("0,19,21,1"), PipelineConfig())
    with pytest.raises(ValueError, match="integer number"):
        PipelineConfig(examples=ExampleConfig(after_seconds=0.001))
    assert PreprocessingConfig(highpass_hz=0.0, lowpass_hz=0.0).highpass_hz is None


def test_no_annotations_do_not_become_background_examples(recording: Recording) -> None:
    assert make_examples(recording, [], PipelineConfig()) == []

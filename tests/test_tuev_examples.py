"""Annotation identity, sample-clock and exclusion-policy checks for the HPC."""

from __future__ import annotations

from dataclasses import replace

import pytest

from eeg_compare.data.annotations import MONTAGE, parse_annotations
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig
from eeg_compare.data.examples import _free_starts, conflict_exclusion, make_examples
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


def test_row_weight_counts_every_source_row(recording: Recording) -> None:
    rows = parse_annotations("0,3,4,1\n0,3,4,1\n1,3,4,1\n0,10,11,6")
    merged = make_examples(recording, rows, PipelineConfig())
    assert [(row.kind, row.row_weight, row.source_example_id) for row in merged] == [("event", 3, ""), ("event", 1, "")]
    assert all(row.cluster_id == recording.cluster_id for row in merged)
    separate = make_examples(recording, rows, PipelineConfig(examples=ExampleConfig(unit="annotation")))
    assert sorted(row.row_weight for row in separate) == [1, 1, 2]
    assert sum(row.row_weight for row in separate) == sum(row.row_weight for row in merged) == len(rows)


def test_free_starts_keep_the_gap_on_both_sides() -> None:
    mark = parse_annotations("0,10,11,1")
    assert _free_starts(mark, 250.0, 1250, 10000, 2500) == [range(5250, 8751)]
    later = parse_annotations("0,60,61,1")
    assert _free_starts(later, 250.0, 1250, 30000, 2500) == [range(0, 11251), range(17750, 28751)]
    assert _free_starts(mark, 250.0, 1250, 5000, 2500) == []
    assert _free_starts(mark, 250.0, 1250, 1000, 0) == []
    assert _free_starts([], 250.0, 1250, 5000, 2500) == [range(0, 3751)]


def test_displaced_controls_stay_away_from_every_mark(recording: Recording) -> None:
    long = replace(recording, n_samples=250 * 120)
    rows = parse_annotations("0,60,61,1\n3,60,61,1\n0,90,91,6")
    plain = PipelineConfig()
    config = PipelineConfig(examples=ExampleConfig(displaced_controls=True))
    events = make_examples(long, rows, plain)
    examples = make_examples(long, rows, config)
    assert [row.example_id for row in examples[:2]] == [row.example_id for row in events]
    controls = examples[2:]
    assert [row.kind for row in controls] == ["displaced", "displaced"]
    by_id = {row.example_id: row for row in events}
    for control in controls:
        source = by_id[control.source_example_id]
        assert (control.target, control.label, control.row_weight) == (source.target, source.label, source.row_weight)
        assert control.status == "kept" and control.annotation_lines == () and control.annotation_channels == ()
        assert control.context_class_codes == () and (control.pad_left, control.pad_right) == (0, 0)
        assert control.onset_offset_samples == pytest.approx(500)
        first, last = control.start_sample, control.start_sample + control.n_samples - 1
        assert 0 <= first and last < 250 * 120
        for mark in rows:
            low, high = int(mark.start_seconds * 250), int(mark.stop_seconds * 250)
            assert low - 1 - last >= 2500 or first - high >= 2500
    assert len({row.example_id for row in examples}) == len(examples)
    assert [row.example_id for row in make_examples(long, rows[::-1], config)] == [row.example_id for row in examples]


def test_no_displaced_control_without_free_time(recording: Recording) -> None:
    config = PipelineConfig(examples=ExampleConfig(displaced_controls=True))
    examples = make_examples(recording, parse_annotations("0,9,10,1"), config)
    assert [row.kind for row in examples] == ["event"]
    near = PipelineConfig(examples=ExampleConfig(displaced_controls=True, displaced_gap_seconds=0.0))
    controls = [row for row in make_examples(recording, parse_annotations("0,9,10,1"), near) if row.kind == "displaced"]
    assert len(controls) == 1 and (controls[0].start_sample + 1250 <= 2250 or controls[0].start_sample >= 2500)
    with pytest.raises(ValueError, match="displaced_gap_seconds"):
        ExampleConfig(displaced_gap_seconds=-1.0)


def test_background_yields_to_another_class_on_another_channel(recording: Recording) -> None:
    rows = parse_annotations("1,3,4,5\n0,3,4,6\n2,8,9,2\n3,8.5,9.5,5\n0,13,14,6\n4,13.5,14.5,6")
    built = {
        policy: {
            (row.annotation_start_seconds, row.label): row
            for row in make_examples(recording, rows, PipelineConfig(examples=ExampleConfig(conflict_policy=policy)))
        }
        for policy in ("background_yields", "drop", "keep")
    }
    yields = built["background_yields"]
    artifact, background = yields[(3.0, "artf")], yields[(3.0, "bckg")]
    assert (artifact.status, artifact.exclusion_reasons, artifact.conflict_class_codes) == ("kept", (), (6,))
    assert artifact.conflicting_annotation
    assert (background.status, background.exclusion_reasons) == ("excluded", ("background_with_other_class",))
    assert background.conflict_class_codes == (5,)
    for key in ((8.0, "gped"), (8.5, "artf")):
        assert (yields[key].status, yields[key].exclusion_reasons) == ("excluded", ("conflicting_annotation",))
    for key in ((13.0, "bckg"), (13.5, "bckg")):
        assert yields[key].status == "kept" and yields[key].conflict_class_codes == ()
    assert sorted(key for key, row in built["drop"].items() if row.status == "kept") == [(13.0, "bckg"), (13.5, "bckg")]
    assert all(row.status == "kept" for row in built["keep"].values())
    assert all(
        built[policy][key].example_id == row.example_id
        and built[policy][key].conflict_class_codes == row.conflict_class_codes
        for policy in ("drop", "keep")
        for key, row in yields.items()
    )


def test_conflict_rules_on_single_intervals() -> None:
    assert conflict_exclusion("background_yields", 5, (6,)) is None
    assert conflict_exclusion("background_yields", 6, (5,)) == "background_with_other_class"
    assert conflict_exclusion("background_yields", 2, (5, 6)) == "conflicting_annotation"
    assert conflict_exclusion("background_yields", 6, ()) is None
    assert conflict_exclusion("drop", 5, (6,)) == "conflicting_annotation"
    assert conflict_exclusion("drop", 5, ()) is None and conflict_exclusion("keep", 2, (1, 5)) is None
    with pytest.raises(ValueError, match="unknown conflict policy"):
        conflict_exclusion("first", 5, (6,))
    with pytest.raises(ValueError, match="conflict_policy"):
        ExampleConfig(conflict_policy="first")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="edge_policy"):
        ExampleConfig(edge_policy="wrap")  # type: ignore[arg-type]

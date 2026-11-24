"""Identical EDF files: the merge rule, the merged annotation rows and the manifest; run on the HPC."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from eeg_compare.data.annotations import parse_annotations
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.dataset import TUEVReader
from eeg_compare.data.duplicates import duplicate_groups, merge_duplicates
from eeg_compare.data.examples import make_examples
from eeg_compare.data.manifest import CorpusLayout, build_manifest, load_manifest
from eeg_compare.data.recordings import Recording
from eeg_compare.data.splits import make_splits

_PREPROCESSING = PreprocessingConfig(input_mode="bipolar", highpass_hz=None, notch_hz=())


def _row(recording: Recording, patient: str, name: str, content: str, split: str = "train") -> Recording:
    kind = "patient" if split == "train" else "session"
    return replace(
        recording,
        recording_id=f"id-{name}",
        edf_path=f"{split}/{patient}/{name}.edf",
        rec_path=f"{split}/{patient}/{name}.rec",
        patient_id=patient if split == "train" else "",
        session_id="" if split == "train" else patient,
        cluster_id=f"{kind}:{patient}",
        cluster_kind=kind,  # type: ignore[arg-type]
        official_split=split,  # type: ignore[arg-type]
        edf_sha256=content,
        rec_sha256=f"rec-{name}",
    )


def _cohort(recording: Recording) -> list[Recording]:
    return [
        _row(recording, "a", "a_1", "x"),
        _row(recording, "b", "b_1", "x"),
        _row(recording, "a", "a_2", "y"),
        _row(recording, "b", "b_2", "z"),
        _row(recording, "c", "c_1", "w"),
        _row(recording, "b", "b_3", "v"),
        _row(recording, "d", "d_3", "v"),
        _row(recording, "032", "e_1", "u", split="eval"),
    ]


def test_merge_keeps_one_recording_per_group_and_joins_the_clusters(recording: Recording) -> None:
    rows = _cohort(recording)
    assert [[row.edf_path for row in group] for group in duplicate_groups(rows)] == [
        ["train/a/a_1.edf", "train/b/b_1.edf"],
        ["train/b/b_3.edf", "train/d/d_3.edf"],
    ]
    merged, report = merge_duplicates(rows)
    assert [row.edf_path for row in merged] == [
        "train/a/a_1.edf",
        "train/a/a_2.edf",
        "train/b/b_2.edf",
        "train/c/c_1.edf",
        "train/b/b_3.edf",
        "eval/032/e_1.edf",
    ]
    assert [row.cluster_id for row in merged] == [
        "patient:a+b+d",
        "patient:a+b+d",
        "patient:a+b+d",
        "patient:c",
        "patient:a+b+d",
        "session:032",
    ]
    first, plain, third = merged[0], merged[1], merged[4]
    assert first.duplicate_edf_paths == ("train/b/b_1.edf",) and first.duplicate_rec_paths == ("train/b/b_1.rec",)
    assert first.duplicate_rec_sha256 == ("rec-b_1",) and first.patient_id == "a" and first.rec_sha256 == "rec-a_1"
    assert first.recording_id != "id-a_1" and third.duplicate_edf_paths == ("train/d/d_3.edf",)
    assert plain.recording_id == "id-a_2" and plain.duplicate_edf_paths == ()
    assert merge_duplicates(rows[::-1])[0][-1].recording_id == first.recording_id
    assert report[0] == {
        "official_split": "train",
        "cluster_id": "patient:a+b+d",
        "joined_clusters": ["patient:a", "patient:b"],
        "edf_paths": ["train/a/a_1.edf", "train/b/b_1.edf"],
        "rec_paths": ["train/a/a_1.rec", "train/b/b_1.rec"],
        "same_annotation_file": False,
    }
    assert report[1]["joined_clusters"] == ["patient:b", "patient:d"] and len(report) == 2
    with pytest.raises(ValueError, match="different train clusters; different annotations"):
        make_splits(rows, SplitConfig(n_folds=2))
    assignments = make_splits(merged, SplitConfig(n_folds=2))
    assert sorted(row.cluster_id for row in assignments) == ["patient:a+b+d", "patient:c", "session:032"]
    untouched, nothing = merge_duplicates(merged)
    assert untouched == merged and nothing == []


def test_a_copy_in_the_other_official_set_has_no_merge_rule(recording: Recording) -> None:
    rows = [*_cohort(recording), _row(recording, "033", "e_2", "y", split="eval")]
    with pytest.raises(ValueError, match="leak between the official sets"):
        merge_duplicates(rows)
    mixed = [*_cohort(recording), replace(_row(recording, "f", "f_1", "w"), cluster_kind="session")]
    with pytest.raises(ValueError, match="two kinds"):
        merge_duplicates(mixed)


def test_merged_annotation_rows_keep_their_file_and_meet_the_conflict_rule(recording: Recording) -> None:
    own = parse_annotations("0,3,4,1\n1,3,4,1")
    copy = [replace(row, source=1) for row in parse_annotations("0,3,4,1\n2,3,4,1\n0,10,11,6")]
    first, second = make_examples(recording, [*own, *copy], PipelineConfig())
    assert first.annotation_sources == (0, 0, 1, 1) and first.annotation_lines == (1, 2, 1, 2)
    assert first.annotation_channels == (0, 1, 2) and (first.row_weight, first.duplicate_rows) == (4, 1)
    assert (second.annotation_sources, second.annotation_lines, second.row_weight) == ((1,), (3,), 1)
    assert [row.example_id for row in make_examples(recording, [*copy, *own], PipelineConfig())] == [
        first.example_id,
        second.example_id,
    ]
    clash = [replace(row, source=1) for row in parse_annotations("5,3.5,4.5,2")]
    examples = make_examples(recording, [*own, *clash], PipelineConfig())
    assert all(row.conflicting_annotation and row.status == "excluded" for row in examples)
    kept = make_examples(recording, [*own, *clash], PipelineConfig(examples=ExampleConfig(conflict_policy="keep")))
    assert [row.annotation_sources for row in kept] == [(0, 0), (1,)]


def _signal(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    time = np.arange(recording.n_samples) / recording.sfreq
    return np.tile(np.sin(2 * np.pi * 10 * time), (len(config.channels), 1)).astype(np.float32)


def test_manifest_merges_identical_files_only_when_the_configuration_says_so(
    corpus_files: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, identities = corpus_files
    (root / "train/record-1.edf").write_bytes((root / "train/record-0.edf").read_bytes())
    (root / "train/record-1.rec").write_text("1,3,4,1\n0,5,6,2\n0,15,16,6\n0,7.5,8.5,4")
    layout = CorpusLayout(root, "synthetic")
    refused = PipelineConfig(preprocessing=_PREPROCESSING, splits=SplitConfig(n_folds=2))
    with pytest.raises(ValueError) as error:
        build_manifest(layout, identities, refused, tmp_path / "refused")
    message = str(error.value)
    assert "[different train clusters; different annotations] train/record-0.edf = train/record-1.edf" in message
    assert 'duplicate_content = "merge"' in message and not (tmp_path / "refused").exists()

    config = PipelineConfig(preprocessing=_PREPROCESSING, splits=SplitConfig(n_folds=2, duplicate_content="merge"))
    metadata = build_manifest(layout, identities, config, tmp_path / "manifest")
    assert metadata["recording_count"] == 3 and metadata["cluster_count"] == 3
    assert metadata["training_patient_count"] == 2 and metadata["fold_sizes"] == [1, 1]
    statistics = metadata["statistics"]
    assert (statistics["kept"], statistics["kept_rows"], statistics["source_annotation_rows"]) == (18, 20, 22)
    assert (statistics["excluded:conflicting_annotation"], statistics["duplicate_annotation_rows"]) == (2, 1)
    assert (statistics["duplicate_content_groups"], statistics["merged_duplicate_files"]) == (1, 1)
    assert metadata["duplicate_content"]["policy"] == "merge"
    assert metadata["duplicate_content"]["groups"] == [
        {
            "official_split": "train",
            "cluster_id": "patient:patient-0+patient-1",
            "joined_clusters": ["patient:patient-0", "patient:patient-1"],
            "edf_paths": ["train/record-0.edf", "train/record-1.edf"],
            "rec_paths": ["train/record-0.rec", "train/record-1.rec"],
            "same_annotation_file": False,
            "rows_per_file": [6, 4],
            "classes_per_file": [
                ["artf", "bckg", "eyem", "gped", "pled", "spsw"],
                ["bckg", "eyem", "gped", "spsw"],
            ],
            "intervals": 8,
            "intervals_marked_in_several_files": 2,
            "conflicting_intervals": 2,
        }
    ]

    manifest = load_manifest(tmp_path / "manifest")
    merged = next(row for row in manifest.recordings if row.duplicate_edf_paths)
    assert merged.edf_path == "train/record-0.edf" and merged.duplicate_rec_paths == ("train/record-1.rec",)
    assert merged.cluster_id == "patient:patient-0+patient-1" and merged.patient_id == "patient-0"
    examples = manifest.scan_examples().collect()
    shared = examples.filter(examples["annotation_sources"].list.len() == 2)
    assert sorted(shared["row_weight"]) == [2, 2] and sorted(map(tuple, shared["annotation_sources"])) == [
        (0, 1),
        (0, 1),
    ]
    assert set(examples.filter(examples["recording_id"] == merged.recording_id)["cluster_id"]) == {merged.cluster_id}

    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    reader = TUEVReader(root, manifest)
    assert reader.recording(merged.recording_id).shape == (22, 5000)
    (root / "train/record-1.rec").write_text("0,1,2,1")
    with pytest.raises(ValueError, match="source EDF or annotations changed"):
        reader.recording(merged.recording_id)

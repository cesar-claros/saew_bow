"""Session clusters, stratified folds, conflict counts and displaced controls in the manifest; run on the HPC."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from numpy.typing import NDArray

from eeg_compare.data.annotations import parse_annotations
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.dataset import Partition, TUEVDataset
from eeg_compare.data.examples import make_examples
from eeg_compare.data.manifest import (
    CorpusLayout,
    _conflict_report,
    _tally_conflicts,
    build_manifest,
    class_presence,
    load_manifest,
    read_identities,
)
from eeg_compare.data.recordings import Recording

_PREPROCESSING = PreprocessingConfig(input_mode="bipolar", highpass_hz=None, notch_hz=())


def _signal(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    time = np.arange(recording.n_samples) / recording.sfreq
    return np.tile(np.sin(2 * np.pi * 10 * time), (len(config.channels), 1)).astype(np.float32)


def _build(corpus_files: tuple[Path, Path], directory: Path, config: PipelineConfig) -> dict[str, Any]:
    root, identities = corpus_files
    return build_manifest(CorpusLayout(root, "synthetic"), identities, config, directory)


def test_session_clusters_and_stratified_folds_survive_the_round_trip(
    corpus_files: tuple[Path, Path], tmp_path: Path
) -> None:
    config = PipelineConfig(
        preprocessing=_PREPROCESSING, splits=SplitConfig(n_folds=3, stratify="class_presence", eval_cluster="session")
    )
    metadata = _build(corpus_files, tmp_path / "manifest", config)
    assert metadata["cluster_policy"]["eval"] == "session"
    assert "sessions" in metadata["cluster_policy"]["disjointness"]
    assert metadata["cluster_count"] == 4 and metadata["training_patient_count"] == 3
    assert metadata["fold_sizes"] == [1, 1, 1]
    manifest = load_manifest(tmp_path / "manifest")
    kinds = {row.cluster_id: row.cluster_kind for row in manifest.assignments}
    assert kinds == {
        "patient:patient-0": "patient",
        "patient:patient-1": "patient",
        "patient:patient-2": "patient",
        "session:session-3": "session",
    }
    presence = class_presence(list(manifest.example_paths))
    assert all(codes == frozenset(range(1, 7)) for codes in presence.values()) and len(presence) == 4
    evaluation = next(row for row in manifest.recordings if row.official_split == "eval")
    assert (evaluation.cluster_id, evaluation.cluster_kind) == ("session:session-3", "session")
    coverage = [row for row in metadata["class_coverage"] if row["split"] == "validation"]
    assert all(row["n_clusters"] == 1 and set(row["clusters"].values()) == {1} for row in coverage)


def test_hidden_evaluation_patients_need_the_session_declaration(
    corpus_files: tuple[Path, Path], tmp_path: Path
) -> None:
    root, identities = corpus_files
    hidden = identities.read_text().replace("patient-3", "")
    identities.write_text(hidden)
    assert read_identities(identities)["eval/record-3.edf"].patient_id == ""
    as_patient = PipelineConfig(preprocessing=_PREPROCESSING, splits=SplitConfig(n_folds=2))
    with pytest.raises(ValueError, match="needs a nonempty patient_id"):
        _build(corpus_files, tmp_path / "refused", as_patient)
    assert not (tmp_path / "refused").exists()
    as_session = PipelineConfig(preprocessing=_PREPROCESSING, splits=SplitConfig(n_folds=2, eval_cluster="session"))
    _build(corpus_files, tmp_path / "manifest", as_session)
    manifest = load_manifest(tmp_path / "manifest")
    examples = manifest.scan_examples().filter(pl.col("official_split") == "eval").collect()
    assert set(examples["cluster_id"]) == {"session:session-3"} and set(examples["patient_id"]) == {""}


def test_displaced_controls_form_their_own_dataset(
    corpus_files: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _ = corpus_files
    config = PipelineConfig(
        preprocessing=_PREPROCESSING,
        examples=ExampleConfig(displaced_controls=True, displaced_gap_seconds=0.0),
        splits=SplitConfig(n_folds=3),
    )
    metadata = _build(corpus_files, tmp_path / "manifest", config)
    statistics = metadata["statistics"]
    assert statistics["kept"] == 24 and statistics["displaced"] == 24 and statistics["kept_rows"] == 24
    assert "recordings_without_displaced_context" not in statistics
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    events = TUEVDataset(root, tmp_path / "manifest", Partition("train", 0))
    controls = TUEVDataset(root, tmp_path / "manifest", Partition("train", 0, kind="displaced"))
    assert set(events.examples["kind"]) == {"event"} and set(controls.examples["kind"]) == {"displaced"}
    assert events.examples.height == controls.examples.height == 12
    targets = dict(zip(events.examples["example_id"], events.examples["target"]))
    assert all(
        targets[source] == target
        for source, target in zip(controls.examples["source_example_id"], controls.examples["target"])
    )
    assert set(controls.examples["cluster_id"]) == set(events.examples["cluster_id"])
    item = controls[0]
    signal, weight = item["signal"], item["row_weight"]
    assert isinstance(signal, torch.Tensor) and isinstance(weight, torch.Tensor)
    assert signal.shape == (22, 1250) and int(weight) == 1
    assert all(3500 <= start <= 3750 for start in controls.examples["start_sample"])
    with pytest.raises(ValueError, match="kind"):
        Partition("train", 0, kind="other")  # type: ignore[arg-type]


def test_conflict_report_gives_both_policies_from_one_build(recording: Recording) -> None:
    rows = parse_annotations("0,3,4,1\n1,3,4,1\n2,3.5,4.5,2\n0,10,11,6\n0,0.5,1.5,6\n1,1,2,5")
    reports = []
    for policy in ("drop", "keep", "background_yields"):
        tally: Counter[tuple[str, int, str]] = Counter()
        _tally_conflicts(
            tally, make_examples(recording, rows, PipelineConfig(examples=ExampleConfig(conflict_policy=policy)))
        )
        reports.append(_conflict_report(tally))
    assert reports[0] == reports[1] == reports[2] and len(reports[0]) == 12
    by_label = {row["label"]: row for row in reports[0] if row["official_split"] == "train"}
    spsw, gped, artf, bckg = (by_label[label] for label in ("spsw", "gped", "artf", "bckg"))
    assert (spsw["all_intervals"], spsw["all_rows"], spsw["conflicting_rows"]) == (1, 2, 2)
    assert (spsw["kept_if_keep_intervals"], spsw["kept_if_keep_rows"], spsw["kept_if_drop_rows"]) == (1, 2, 0)
    assert (gped["conflicting_intervals"], gped["kept_if_keep_intervals"], gped["kept_if_drop_intervals"]) == (1, 1, 0)
    assert (bckg["all_intervals"], bckg["conflicting_intervals"]) == (2, 1)
    assert (bckg["kept_if_keep_intervals"], bckg["kept_if_drop_intervals"]) == (1, 1)
    assert (artf["conflicting_intervals"], artf["kept_if_keep_intervals"], artf["kept_if_drop_intervals"]) == (1, 0, 0)
    assert (spsw["kept_if_background_yields_intervals"], gped["kept_if_background_yields_intervals"]) == (0, 0)
    assert (bckg["kept_if_background_yields_intervals"], artf["kept_if_background_yields_intervals"]) == (1, 0)
    assert all(row["all_intervals"] == 0 for row in reports[0] if row["official_split"] == "eval")


def test_conflict_report_counts_what_background_yields_restores(recording: Recording) -> None:
    rows = parse_annotations("1,3,4,5\n2,3,4,5\n0,3,4,6\n0,10,11,6")
    tally: Counter[tuple[str, int, str]] = Counter()
    _tally_conflicts(tally, make_examples(recording, rows, PipelineConfig()))
    by_label = {row["label"]: row for row in _conflict_report(tally) if row["official_split"] == "train"}
    artf, bckg = by_label["artf"], by_label["bckg"]
    assert (artf["kept_if_drop_intervals"], artf["kept_if_background_yields_intervals"]) == (0, 1)
    assert (artf["kept_if_background_yields_rows"], artf["kept_if_keep_rows"]) == (2, 2)
    assert (bckg["all_intervals"], bckg["kept_if_background_yields_intervals"], bckg["kept_if_keep_intervals"]) == (
        2,
        1,
        2,
    )

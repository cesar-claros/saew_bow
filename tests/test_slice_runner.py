"""Step 2 of the slice from a manifest to a stored run and its report, with mocked EDF content; run on the HPC."""

from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch
from numpy.typing import NDArray

from eeg_compare.cells.api import Geometry
from eeg_compare.data.annotations import MONTAGE
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.dataset import TUEVReader
from eeg_compare.data.manifest import CorpusLayout, Manifest, build_manifest, load_manifest
from eeg_compare.data.recordings import Recording
from eeg_compare.slice.exposure import ExposureConfig, build_plans, write_plans
from eeg_compare.slice.report import PROTOCOL_TOKENS, read_run, render
from eeg_compare.slice.runner import ARMS, NULL_ARM, Settings, _valid_start, compare_blocks, fold_blocks, prepare, run
from eeg_compare.slice.scaling import fit_profile, profile_factors
from eeg_compare.slice.study import (
    Study,
    TrainingReader,
    clip_shares,
    event_table,
    fold_interval_rows,
    fold_parts,
    load_study,
    pooled_intervals,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_slice.py"
SMALL = Settings(
    n_atoms=4,
    seeds=(0,),
    forest_seeds=(0,),
    forest_trees=5,
    kmeans_passes=10,
    null_permutations=2,
    n_jobs=1,
    device="cpu",
)
CLIPPED = 10


def _signal(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    """Noise with a 10 Hz burst in each labeled second, on the channels that name the class of the second.

    The marks of the fixture are at 3, 5, 7, 9, 11 and 13 s, one per class in the
    order of the classes. Record 0 also holds some samples at the clip of 50.
    """
    index = int(Path(recording.edf_path).stem.split("-")[1])
    data = np.random.default_rng(index).standard_normal((len(config.channels), recording.n_samples))
    time = np.arange(250) / 250.0
    for label in range(6):
        first = (3 + 2 * label) * 250
        data[label::6, first : first + 250] += 4.0 * np.sin(2 * np.pi * 10.0 * time)
    if index == 0:
        data[0, 100 : 100 + CLIPPED] = 50.0
    return data.astype(np.float32)


@pytest.fixture
def corpus(tmp_path: Path, corpus_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A manifest that the slice can read: scale per recording, a clip, displaced controls and two folds."""
    root, identities = corpus_files
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(
            input_mode="bipolar",
            highpass_hz=None,
            notch_hz=(),
            normalization="mad",
            normalization_scope="recording",
            clip=50.0,
        ),
        examples=ExampleConfig(displaced_controls=True, displaced_gap_seconds=0.0),
        splits=SplitConfig(n_folds=2),
    )
    directory = tmp_path / "slice-manifest"
    build_manifest(CorpusLayout(root, "synthetic"), identities, config, directory)
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    return root, directory


@pytest.fixture
def manifest(corpus: tuple[Path, Path]) -> Manifest:
    return load_manifest(corpus[1])


@pytest.fixture
def study(corpus: tuple[Path, Path], manifest: Manifest) -> Study:
    return load_study(manifest, build_plans(manifest, ExposureConfig()), TUEVReader(corpus[0], manifest))


def _scales(manifest: Manifest) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(7)
    return {
        row.recording_id: np.exp(rng.normal(-11.0, 0.3, len(MONTAGE)))
        for row in manifest.recordings
        if row.official_split == "train"
    }


def test_the_event_table_pairs_every_event_with_its_control_and_its_fold(
    manifest: Manifest, mini_corpus: tuple[Path, Path]
) -> None:
    table = event_table(manifest)
    assert table.height == 18 and table["example_id"].n_unique() == 18
    assert sorted(table["target"].unique().to_list()) == list(range(6)) and set(table["fold"].to_list()) == {0, 1}
    assert table.select(pl.col("cluster_id").n_unique()).item() == 3
    # every patient is held out by one fold, and a control lies after the last mark of its recording
    assert table.group_by("cluster_id").agg(pl.col("fold").n_unique())["fold"].to_list() == [1, 1, 1]
    assert table["control_start"].min() >= 3500 and table["control_start"].max() <= 3750
    by_recording = table.group_by("recording_id", maintain_order=True).agg(pl.col("start_sample"))
    assert all(starts == sorted(starts) for starts in by_recording["start_sample"].to_list())
    with pytest.raises(ValueError, match="no displaced control"):
        event_table(load_manifest(mini_corpus[1]))


def test_the_study_holds_training_signals_in_the_order_of_its_tables(corpus: tuple[Path, Path], study: Study) -> None:
    assert study.contexts.shape == study.controls.shape == (18, 22, 1250)
    assert study.interval_signals.shape == (12, 22, 1250) and study.intervals.height == 12
    assert study.labels == ("spsw", "gped", "pled", "eyem", "artf", "bckg") and study.n_classes == 6
    assert study.recordings_read == 3 and study.n_folds == 2
    assert study.sfreq == 250.0 and study.event_samples == (500, 750)
    study_manifest = load_manifest(corpus[1])
    reader = TUEVReader(corpus[0], study_manifest)
    for row, event in enumerate(study.events.iter_rows(named=True)):
        signal = reader.recording(event["recording_id"])
        first, control = event["start_sample"], event["control_start"]
        assert np.array_equal(study.contexts[row].numpy(), signal[:, first : first + 1250])
        assert np.array_equal(study.controls[row].numpy(), signal[:, control : control + 1250])
    for row, unit in enumerate(study.intervals.iter_rows(named=True)):
        signal = reader.recording(unit["recording_id"])
        assert np.array_equal(study.interval_signals[row].numpy(), signal[:, unit["start_sample"] :][:, :1250])
    # the labeled second of a context holds the burst of its class, and a control holds none
    energy = study.contexts[:, :, 500:750].pow(2).mean(dim=-1)
    targets = torch.tensor(study.events["target"].to_list())
    marked = torch.arange(22)[None, :] % 6 == targets[:, None]
    assert float(energy[marked].min()) > 4.0 and float(energy[~marked].max()) < 3.0
    assert float(study.controls.pow(2).mean(dim=-1).max()) < 3.0
    # the reader of the study refuses the evaluation set
    held_out = next(row.recording_id for row in study_manifest.recordings if row.official_split == "eval")
    guarded = TrainingReader(reader, study_manifest)
    with pytest.raises(ValueError, match="training set only"):
        guarded.recording(held_out)
    assert guarded.read == []


def test_the_parts_and_the_plan_of_a_fold_share_no_patient(study: Study) -> None:
    patients = np.asarray(study.events["cluster_id"].to_list())
    assert pooled_intervals(study.plan).equals(study.intervals)
    seen = []
    for fold in range(2):
        train, validation = fold_parts(study, fold)
        assert train.size + validation.size == 18 and not set(train.tolist()) & set(validation.tolist())
        assert not set(patients[train]) & set(patients[validation])
        rows = fold_interval_rows(study, fold)
        assert rows.tolist() == sorted(rows.tolist()) and rows.size == 4 * len(set(patients[train]))
        assert set(study.intervals["cluster_id"].gather(rows.tolist()).to_list()) == set(patients[train])
        seen.extend(validation.tolist())
    assert sorted(seen) == list(range(18))
    with pytest.raises(ValueError, match="does not exist"):
        fold_parts(study, 2)


def test_the_clip_share_counts_samples_at_the_clip_in_channels_with_a_factor_below_one(
    study: Study, manifest: Manifest
) -> None:
    scales = _scales(manifest)
    found = clip_shares(study, manifest, scales, 50.0)
    assert [row["fold"] for row in found] == [0, 1]
    fold_of = {row.cluster_id: row.validation_fold for row in manifest.assignments if row.official_split == "train"}
    first = next(row for row in manifest.recordings if row.edf_path.endswith("record-0.edf"))
    for row in found:
        training = [
            item.recording_id
            for item in manifest.recordings
            if item.official_split == "train" and fold_of[item.cluster_id] != row["fold"]
        ]
        samples = 4 * len(training) * 22 * 1250
        present = first.recording_id in training
        assert row["share_at_clip"] == pytest.approx(CLIPPED / samples if present else 0.0)
        factors = profile_factors(scales, fit_profile(scales, training), training)
        below = present and bool(factors[first.recording_id][0] < 1.0)
        assert row["share_at_clip_factor_below_one"] == pytest.approx(CLIPPED / samples if below else 0.0)
        assert row["approximate"] is False


def test_a_start_with_a_flat_patch_is_drawn_again() -> None:
    rows = torch.zeros((20, 3, 300))
    rows[0] = torch.randn((3, 300), generator=torch.Generator().manual_seed(0))
    geometry = Geometry(n_atoms=2, atom_len=25, pool_len=50, n_projections=1)
    # one draw in 400 takes both patches from the row with signal
    start, attempts = _valid_start(rows, geometry, seed=0, attempts=100000)
    assert attempts > 1 and start.segments.tolist() == [0, 0] and start.patches.shape == (2, 3, 25)
    assert _valid_start(rows, geometry, seed=0, attempts=100000)[1] == attempts
    with pytest.raises(ValueError, match="no start"):
        _valid_start(torch.zeros((2, 3, 300)), geometry, seed=0, attempts=5)
    with pytest.raises(ValueError, match="expected segments"):
        _valid_start(torch.zeros((2, 3, 10)), geometry, seed=0, attempts=5)


def test_a_fold_gives_one_block_per_arm_head_and_readout_and_repeats(study: Study) -> None:
    prepared = prepare(study, SMALL)
    assert set(prepared.spectral) == {"full", "event", "context", "displaced"}
    assert all(values.shape == (18, 176) for values in prepared.spectral.values())
    assert prepared.context_spectra.shape == (18, 10, 22, 30) and prepared.interval_spectra.shape == (12, 10, 22, 30)
    blocks, record = fold_blocks(study, prepared, SMALL, 0)
    _, validation = fold_parts(study, 0)
    keys = [block.key for block in blocks]
    assert len(set(keys)) == len(keys) and all(block.fold == 0 for block in blocks)
    # five feature arms with four readouts and two heads, the prior, and two permutations of the null
    assert len(blocks) == 5 * 4 * 2 + 1 + 2
    assert {block.arm for block in blocks} == {*ARMS, NULL_ARM}
    for block in blocks:
        assert np.array_equal(block.rows, validation) and block.probabilities.shape == (validation.size, 6)
        assert block.probabilities.dtype == np.float32
        np.testing.assert_allclose(block.probabilities.sum(axis=1), 1.0, atol=1e-5)
    tokens = {(row["arm"], row["readout"]): row for row in record["counts"]}
    for arm in ("spectra_bag", "S1-frozen-data", "S1-frozen-gauss"):
        assert {name: tokens[(arm, name)]["tokens_per_context"] for name in PROTOCOL_TOKENS["single"]} == (
            PROTOCOL_TOKENS["single"]
        )
    assert {name: tokens[("SF-frozen-data", name)]["tokens_per_context"] for name in PROTOCOL_TOKENS["full"]} == (
        PROTOCOL_TOKENS["full"]
    )
    assert all(abs(row["sum_min"] - 1.0) < 1e-9 and abs(row["sum_max"] - 1.0) < 1e-9 for row in record["counts"])
    assert record["start_attempts"] == [1] and record["logistic_passes"] > 0
    assert set(record["seconds"]) == {"logistic", "forest", "null", "spectra_bag", "tokens"}
    # gate 6: a second computation from the signals gives the same predictions
    again, _ = fold_blocks(study, prepare(study, SMALL), SMALL, 0)
    same = compare_blocks(blocks, again)
    assert same["comparable"] and same["equal_predictions"] and same["unequal_predictions"] == 0
    assert same["blocks"] == len(blocks) and same["predictions"] == len(blocks) * validation.size
    assert same["max_probability_difference"] < 1e-5
    assert {row["arm"] for row in same["by_arm"]} == {*ARMS, NULL_ARM}
    assert sum(row["blocks"] for row in same["by_arm"]) == len(blocks)
    # a changed prediction is counted for its arm, and the other arms stay equal
    last = blocks[-1]
    changed = compare_blocks(blocks, [*blocks[:-1], replace(last, probabilities=1.0 - last.probabilities)])
    assert changed["comparable"] and not changed["equal_predictions"]
    assert changed["unequal_predictions"] == validation.size and changed["max_probability_difference"] > 0.1
    by_arm = {row["arm"]: row["unequal_predictions"] for row in changed["by_arm"]}
    assert by_arm.pop(last.arm) == validation.size and not any(by_arm.values())
    # other blocks, or other examples, cannot be compared
    other, _ = fold_blocks(study, prepared, SMALL, 1)
    for different in (other, blocks[:-1]):
        found = compare_blocks(blocks, different)
        assert not found["comparable"] and not found["equal_predictions"] and found["by_arm"] == []


def test_a_run_is_stored_and_read_with_its_gates(study: Study, manifest: Manifest, tmp_path: Path) -> None:
    output = tmp_path / "run"
    report = run(study, manifest, _scales(manifest), SMALL, output, {"name": "test"})
    assert sorted(path.name for path in output.iterdir()) == ["examples.parquet", "predictions.parquet", "report.json"]
    assert report["gate_6"]["equal_predictions"] and report["gate_2"]["recordings_read"] == 3
    assert len(report["gate_6"]["by_arm"]) == 7 and report["gate_6"]["unequal_predictions"] == 0
    assert report["examples"] == 18 and report["patients"] == 3 and report["plan_intervals"] == 12
    assert report["readouts"]["event"] == {"windows": [4, 5], "ranges": [[438, 812]]}
    predictions = pl.read_parquet(output / "predictions.parquet")
    assert predictions.columns[:9] == [
        "arm",
        "seed",
        "head",
        "head_seed",
        "readout",
        "fold",
        "example_id",
        "target",
        "prediction",
    ]
    groups = predictions.group_by("arm", "seed", "head", "head_seed", "readout").agg(
        pl.len().alias("rows"), pl.col("example_id").n_unique().alias("examples")
    )
    assert groups.height == 5 * 4 * 2 + 1 + 2 and set(groups["rows"].to_list()) == {18}
    assert set(groups["examples"].to_list()) == {18}
    truth = dict(zip(study.events["example_id"].to_list(), study.events["target"].to_list(), strict=True))
    assert all(truth[name] == target for name, target in predictions.select("example_id", "target").iter_rows())
    stored = pl.read_parquet(output / "examples.parquet")
    assert stored.columns == ["example_id", "cluster_id", "label", "target", "row_weight", "fold"]
    assert stored["example_id"].to_list() == study.events["example_id"].to_list()

    summary = read_run(output, n_draws=50)
    gates = summary["gates"]
    assert gates["gate_2"]["passes"] and gates["gate_5"]["passes"] and gates["gate_6"]["passes"]
    assert gates["gate_5"]["primary_geometry"] and len(gates["gate_5"]["rows"]) == 16
    assert gates["gate_3"]["permutations"] == 2 and 0.0 <= gates["gate_3"]["balanced_accuracy"] <= 1.0
    scores = {(row["arm"], row["head"], row["readout"]): row for row in summary["scores"]}
    assert len(scores) == 5 * 4 * 2 + 1 and NULL_ARM not in {row["arm"] for row in summary["scores"]}
    assert scores[("prior", "prior", "full")]["balanced_accuracy"] == pytest.approx(1.0 / 6.0)
    # the burst of a class is on its own channels: the spectral control reads it, and not in the control context
    spectral = scores[("spectral", "logistic", "full")]
    assert spectral["balanced_accuracy"] > 0.7
    assert scores[("spectral", "logistic", "displaced")]["balanced_accuracy"] < spectral["balanced_accuracy"]
    assert spectral["interval"]["low"] <= spectral["interval"]["high"] and len(spectral["recall"]) == 6
    assert np.array(spectral["confusion"]).sum() == pytest.approx(18.0)
    assert {(row["first"], row["second"]) for row in summary["pairs"]} >= {("spectral", "spectra_bag")}
    text = render(summary)
    assert "Gates" in text and "spectral" in text and "Paired differences" in text
    with pytest.raises(ValueError, match="never overwritten"):
        run(study, manifest, _scales(manifest), SMALL, output, {})


def test_the_script_runs_and_reads_step_2(
    corpus: tuple[Path, Path], manifest: Manifest, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = SimpleNamespace(**runpy.run_path(str(SCRIPT)))
    root, directory = corpus
    plans = tmp_path / "plans"
    write_plans(manifest, ExposureConfig(), plans)
    scales = tmp_path / "scales.json"
    content = {
        "band_hz": [None, None],
        "measure_1_profile": {"profile_factor_by_channel": {name: 1.0 for name in MONTAGE}},
        "scales": [{"recording_id": name, "scale": value.tolist()} for name, value in _scales(manifest).items()],
    }
    scales.write_text(json.dumps(content), encoding="utf-8")
    output = tmp_path / "run"
    arguments = ["--root", str(root), "--manifest", str(directory), "--plans", str(plans), "--scales", str(scales)]
    options = ["--smoke", "--device", "cpu", "--jobs", "1", "--draws", "20"]
    assert script.main(["run", *arguments, "--output", str(output), *options]) == 0
    printed = capsys.readouterr().out
    assert "Read 3 recordings: 18 events" in printed and "Gates" in printed and "6 determinism  PASS" in printed
    summary = tmp_path / "summary.json"
    assert script.main(["read", "--run", str(output), "--draws", "20", "--output", str(summary)]) == 0
    assert "Balanced accuracy on the full readout" in capsys.readouterr().out
    assert json.loads(summary.read_text())["gates"]["gate_6"]["passes"] is True
    assert script.main(["run", *arguments, "--output", str(output), *options]) == 2
    assert "never overwritten" in capsys.readouterr().err
    assert script.main(["run", *arguments, "--output", str(directory / "run"), *options]) == 2
    assert script.main(["read", "--run", str(tmp_path / "absent")]) == 2

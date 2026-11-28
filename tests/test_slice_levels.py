"""Step 3 of the slice: the levels, the run that continues, and the rules of the read; run on the HPC."""

from __future__ import annotations

import json
import runpy
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from numpy.typing import NDArray

from eeg_compare.data.annotations import MONTAGE
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.dataset import TUEVReader
from eeg_compare.data.manifest import CorpusLayout, Manifest, build_manifest, load_manifest
from eeg_compare.data.recordings import Recording
from eeg_compare.slice.decide import decide, evaluate, render_decisions
from eeg_compare.slice.exposure import ExposureConfig, build_plans, write_plans
from eeg_compare.slice.levels import compare_frames, fold_factors, patient_halves, profile_rows, run_levels
from eeg_compare.slice.report import load_run, read_run, render
from eeg_compare.slice.runner import ARMS, NULL_ARM, Settings
from eeg_compare.slice.signals import load_units
from eeg_compare.slice.study import Study, fold_interval_rows, fold_parts, load_study

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_slice.py"
LABELS = ("spsw", "gped", "pled", "eyem", "artf", "bckg")
NARROW = ExposureConfig(minutes_per_patient=10.0 / 60.0)
SMALL = Settings(
    n_atoms=4,
    seeds=(0,),
    forest_seeds=(0,),
    forest_trees=5,
    kmeans_passes=10,
    null_permutations=2,
    max_passes=3,
    level_atoms=(2, 6),
    n_jobs=1,
    device="cpu",
)
FROZEN = ("S1-frozen-data", "S1-frozen-gauss", "SF-frozen-data")
LEVEL_ARMS = {
    "S1-L1",
    "S1-L2",
    *(f"{arm}@raw" for arm in ("S1-L1", "S1-L2", *FROZEN)),
    *(f"{arm}@K{count}" for arm in ("S1-L1", "S1-L2") for count in (2, 6)),
    *(f"{arm}@P250" for arm in ("S1-L2", *FROZEN)),
    "S1-L2@10min",
    "spectral@profile",
    "SF-frozen-data@profile",
    "S1-L2@profile-raw",
    "S1-L2@half-in",
    "S1-L2@half-out",
}


def _signal(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    """Noise with a 10 Hz burst in each labeled second, on the channels that name the class of the second."""
    index = int(Path(recording.edf_path).stem.split("-")[1])
    data = np.random.default_rng(index).standard_normal((len(config.channels), recording.n_samples))
    time = np.arange(250) / 250.0
    for label in range(6):
        first = (3 + 2 * label) * 250
        data[label::6, first : first + 250] += 4.0 * np.sin(2 * np.pi * 10.0 * time)
    return data.astype(np.float32)


@pytest.fixture
def corpus(
    tmp_path: Path, larger_corpus_files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Six training patients in a manifest that the slice can read, with two folds."""
    root, identities = larger_corpus_files
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
    plans = build_plans(manifest, NARROW), build_plans(manifest, ExposureConfig())
    return load_study(manifest, plans[0], TUEVReader(corpus[0], manifest), plans[1])


def _scales(manifest: Manifest) -> dict[str, NDArray[np.float64]]:
    rng = np.random.default_rng(7)
    return {
        row.recording_id: np.exp(rng.normal(-11.0, 0.3, len(MONTAGE)))
        for row in manifest.recordings
        if row.official_split == "train"
    }


def _numbers(**changes: Any) -> dict[str, Any]:
    differences = {
        "displaced": [0.01, 0.08, -0.05, 0.30, 0.09, 0.20],
        "context": [0.00, 0.02, 0.05, 0.20, 0.05, 0.08],
    }
    effect = [
        {"class": name, "against": against, "difference": value}
        for against, values in differences.items()
        for name, value in zip(LABELS, values, strict=True)
    ]
    numbers = {
        "h": 0.03,
        "learned_minus_frozen": 0.05,
        "bag_minus_learned": -0.02,
        "best_arm": "spectral",
        "cosine_minus_raw": 0.04,
        "profile_minus_primary_spectral": -0.01,
        "record_effect": effect,
        "levels": {
            "K64": [{"head": "logistic", "difference": 0.05}, {"head": "forest", "difference": 0.04}],
            "raw tokens": [{"head": "logistic", "difference": 0.05}, {"head": "forest", "difference": 0.02}],
            "10 minutes": [{"head": "logistic", "difference": 0.09}],
        },
    }
    return {**numbers, **changes}


def test_the_rules_of_the_read_follow_the_protocol() -> None:
    found = evaluate(_numbers())
    assert found["h"] == 0.03 and found["margin"] == 0.03 and found["resolution"] == "ranking possible"
    assert all(found[name] is True for name in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"))
    # a level replaces the primary value only if it is above by more than h on both heads
    assert found["defaults"] == {"K64": True, "raw tokens": False, "10 minutes": False}
    # a class within the margin on the displaced or the context readout gets no waveform claim
    assert found["no_waveform_claim"] == ["gped", "pled", "spsw"]
    wide = evaluate(_numbers(h=0.06))
    assert wide["margin"] == 0.06 and wide["resolution"] == "described comparison"
    assert not wide["S1"] and not wide["S2"] and not wide["S6"] and wide["S3"] and wide["S7"]
    assert wide["no_waveform_claim"] == ["artf", "gped", "pled", "spsw"]
    narrow = evaluate(_numbers(h=0.01))
    assert not narrow["S1"] and narrow["margin"] == 0.02 and not narrow["S7"]
    assert narrow["defaults"]["raw tokens"] is True and narrow["defaults"]["10 minutes"] is False
    assert not evaluate(_numbers(bag_minus_learned=-0.04))["S3"]
    assert not evaluate(_numbers(best_arm="SF-frozen-data"))["S4"]
    assert not evaluate(_numbers(profile_minus_primary_spectral=-0.05))["S7"]
    effect = [dict(row) for row in _numbers()["record_effect"]]
    effect[1]["difference"] = 0.15
    assert not evaluate(_numbers(record_effect=effect))["S5"]


def test_the_profile_input_of_rows_equals_the_reader_of_units() -> None:
    rng = np.random.default_rng(0)
    data = {name: (30.0 * rng.standard_normal((3, 400))).astype(np.float32) for name in ("a", "b")}
    reader = SimpleNamespace(recording=data.__getitem__)
    schema = {"recording_id": pl.String, "start_sample": pl.Int64, "n_samples": pl.Int64}
    units = pl.DataFrame([("b", 10, 100), ("a", 0, 100), ("b", 250, 100)], schema=schema, orient="row")
    factors = {"a": np.array([2.0, 0.5, 1.0]), "b": np.array([0.3, 3.0, 1.7])}
    primary = load_units(reader, units)
    found = profile_rows(primary, units["recording_id"].to_list(), factors, 50.0)
    assert torch.equal(found, load_units(reader, units, factors, clip=50.0))
    assert float(found.abs().max()) == 50.0 and not torch.equal(found, primary)
    with pytest.raises(ValueError, match="one recording per row"):
        profile_rows(primary, ["a"], factors, 50.0)


def test_the_factors_of_a_fold_use_the_profile_of_its_training_patients(manifest: Manifest) -> None:
    scales = _scales(manifest)
    training = [row for row in manifest.recordings if row.official_split == "train"]
    for fold in range(2):
        factors = fold_factors(manifest, scales, fold)
        assert set(factors) == {row.recording_id for row in training}
        assert all(value.shape == (22,) and (value > 0).all() and (value <= 3.0).all() for value in factors.values())
    # the profile of a fold does not move when a recording of its validation patients changes
    fold_of = {row.cluster_id: row.validation_fold for row in manifest.assignments}
    held_out = next(row.recording_id for row in training if fold_of[row.cluster_id] == 0)
    kept = next(row.recording_id for row in training if fold_of[row.cluster_id] != 0)
    changed = {**scales, held_out: scales[held_out] * np.linspace(0.5, 2.0, 22)}
    assert np.array_equal(fold_factors(manifest, changed, 0)[kept], fold_factors(manifest, scales, 0)[kept])
    assert not np.array_equal(fold_factors(manifest, changed, 1)[kept], fold_factors(manifest, scales, 1)[kept])


def test_patient_halves_are_balanced_inside_every_stratum() -> None:
    sizes = {0: 12, 1: 9, 2: 6}
    patients = [f"p{stratum}-{number}" for stratum, size in sizes.items() for number in range(size)]
    strata = [stratum for stratum, size in sizes.items() for _ in range(size)]
    first, second = patient_halves(patients * 2, strata * 2, seed=0)
    assert not first & second and first | second == set(patients)
    for stratum, size in sizes.items():
        members = {name for name, value in zip(patients, strata, strict=True) if value == stratum}
        assert len(first & members) == (size + 1) // 2 and len(second & members) == size // 2
    assert patient_halves(patients[::-1], strata[::-1], seed=0) == (first, second)
    assert patient_halves(patients, strata, seed=1) != (first, second)


def test_two_stored_computations_of_a_fold_are_compared_by_arm() -> None:
    base = pl.DataFrame(
        {
            "arm": ["a", "a", "b", "b"],
            "seed": [0, 0, 0, 0],
            "head": ["logistic"] * 4,
            "head_seed": [0, 0, 0, 0],
            "readout": ["full"] * 4,
            "fold": [0, 0, 0, 0],
            "example_id": ["x", "y", "x", "y"],
            "target": [0, 1, 0, 1],
            "prediction": [0, 1, 1, 1],
            "p0": [0.9, 0.2, 0.4, 0.1],
            "p1": [0.1, 0.8, 0.6, 0.9],
        },
        schema_overrides={"p0": pl.Float32, "p1": pl.Float32},
    )
    same = compare_frames(base, base.clone())
    assert same["comparable"] and same["equal_predictions"] and same["blocks"] == 2 and same["predictions"] == 4
    assert same["unequal_predictions"] == 0 and same["max_probability_difference"] == 0.0
    changed = base.with_columns(
        pl.Series("prediction", [0, 1, 0, 1]), pl.Series("p0", [0.9, 0.2, 0.7, 0.1], dtype=pl.Float32)
    )
    found = compare_frames(base, changed)
    assert found["comparable"] and not found["equal_predictions"] and found["unequal_predictions"] == 1
    by_arm = {row["arm"]: row for row in found["by_arm"]}
    assert by_arm["a"]["unequal_predictions"] == 0 and by_arm["a"]["max_probability_difference"] == 0.0
    assert by_arm["b"]["unequal_predictions"] == 1
    assert by_arm["b"]["max_probability_difference"] == pytest.approx(0.3, abs=1e-6)
    other = base.with_columns(pl.Series("example_id", ["x", "y", "x", "z"]))
    for different in (other, base.head(3)):
        result = compare_frames(base, different)
        assert not result["comparable"] and not result["equal_predictions"] and result["by_arm"] == []


def test_step_3_runs_fold_by_fold_and_a_second_call_continues(study: Study, manifest: Manifest, tmp_path: Path) -> None:
    assert study.intervals.height == 24 and study.contexts.shape == (36, 22, 1250)
    patients = np.asarray(study.events["cluster_id"].to_list())
    for fold in range(2):
        training = len(set(patients[fold_parts(study, fold)[0]].tolist()))
        assert training >= 2 and fold_interval_rows(study, fold).size == 2 * training
        assert fold_interval_rows(study, fold, wide=True).size == 4 * training
    scales = _scales(manifest)
    output = tmp_path / "run"
    messages: list[str] = []
    report = run_levels(study, manifest, scales, SMALL, output, {"name": "test"}, messages.append)
    assert [message.split(":")[0] for message in messages] == ["fold-0", "fold-1", "repeat-0"]
    assert report["step"] == 3 and report["gate_6"]["comparable"] and report["gate_6"]["equal_predictions"]
    assert sorted(path.name for path in output.iterdir()) == [
        "config.json",
        "examples.parquet",
        "folds",
        "predictions.parquet",
        "report.json",
    ]
    assert len(list((output / "folds").iterdir())) == 6
    fits = report["fits"]
    assert len(fits) == 2 and all(len(rows) == 10 for rows in fits)
    assert {row["arm"] for row in fits[0]} == {
        "S1-L1",
        "S1-L2",
        "S1-L1@K2",
        "S1-L2@K2",
        "S1-L1@K6",
        "S1-L2@K6",
        "S1-L2@P250",
        "S1-L2@10min",
        "S1-L2@profile-raw",
        "S1-L2@half",
    }
    assert all(1 <= row["passes"] <= 3 and 0.0 <= row["residual_share"] <= 1.0 for row in fits[0])
    by_arm = {row["arm"]: row for row in fits[0]}
    assert by_arm["S1-L2@10min"]["segments"] == 2 * by_arm["S1-L2"]["segments"]
    assert by_arm["S1-L2@half"]["segments"] < by_arm["S1-L2"]["segments"]
    assert (by_arm["S1-L1@K6"]["atoms"], by_arm["S1-L2@P250"]["pool_len"]) == (6, 250)
    assert all(sum(halves) >= 2 and min(halves) >= 1 for halves in report["halves"])

    predictions = pl.read_parquet(output / "predictions.parquet")
    assert set(predictions["arm"].unique().to_list()) == {*ARMS, NULL_ARM, *LEVEL_ARMS}
    groups = predictions.group_by("arm", "seed", "head", "head_seed", "readout").agg(
        pl.len().alias("rows"), pl.col("example_id").n_unique().alias("examples")
    )
    assert set(groups["rows"].to_list()) == {36} and set(groups["examples"].to_list()) == {36}

    # a complete run is never computed again, and a run that stopped continues with its stored folds
    with pytest.raises(ValueError, match="complete run"):
        run_levels(study, manifest, scales, SMALL, output, {"name": "test"})
    partial = tmp_path / "partial"
    (partial / "folds").mkdir(parents=True)
    for name in ("config.json", "folds/fold-0.json", "folds/fold-0.parquet"):
        shutil.copy(output / name, partial / name)
    messages.clear()
    run_levels(study, manifest, scales, SMALL, partial, {"name": "test"}, messages.append)
    assert [message.split(":")[0] for message in messages] == ["fold-1", "repeat-0"]
    assert pl.read_parquet(partial / "predictions.parquet").equals(predictions)
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(output / "config.json", other / "config.json")
    with pytest.raises(ValueError, match="other settings"):
        run_levels(study, manifest, scales, replace(SMALL, forest_trees=6), other, {"name": "test"})
    with pytest.raises(ValueError, match="other settings"):
        run_levels(study, manifest, scales, SMALL, other, {"name": "another"})
    with pytest.raises(ValueError, match="wide plan"):
        run_levels(replace(study, wide_plan=None), manifest, scales, SMALL, tmp_path / "none", {})

    # the read: gates, the fits, and the numbers and rules of step 3
    data = load_run(output)
    summary = read_run(output, 50, 0, data)
    gates = summary["gates"]
    assert summary["step"] == 3 and len(summary["fits"]) == 20
    assert gates["gate_2"]["passes"] and gates["gate_5"]["passes"] and gates["gate_6"]["passes"]
    decisions = decide(data, 50, 0)
    assert decisions is not None
    numbers, outcomes = decisions["numbers"], decisions["outcomes"]
    assert outcomes["h"] == numbers["m1"]["half_width"] and outcomes["margin"] >= 0.02
    assert all(isinstance(outcomes[name], bool) for name in ("S1", "S2", "S3", "S4", "S5", "S6", "S7"))
    assert set(outcomes["defaults"]) == {"K2", "K6", "raw tokens", "tiling P250", "profile input", "10 minutes"}
    assert [len(numbers[key]) for key in ("m4", "m5", "m6", "m7", "m8", "m9", "m11")] == [8, 10, 8, 6, 2, 2, 8]
    assert numbers["best_arm"] in numbers["primary_scores"] and len(numbers["record_effect"]) == 12
    assert numbers["m1"]["first"] == "S1-L2" and numbers["m1"]["second"] == "S1-frozen-data"
    text = render(summary) + render_decisions(decisions)
    for part in ("Slice, step 3", "Alternating fits", "M1  h =", "Predictions", "Defaults of stage 4", "M11"):
        assert part in text


def test_the_script_runs_step_3_and_prints_its_rules(
    corpus: tuple[Path, Path], manifest: Manifest, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = SimpleNamespace(**runpy.run_path(str(SCRIPT)))
    root, directory = corpus
    plans, wide = tmp_path / "plans", tmp_path / "wide-plans"
    write_plans(manifest, NARROW, plans)
    write_plans(manifest, ExposureConfig(), wide)
    scales = tmp_path / "scales.json"
    content = {
        "band_hz": [None, None],
        "measure_1_profile": {"profile_factor_by_channel": {name: 1.0 for name in MONTAGE}},
        "scales": [{"recording_id": name, "scale": value.tolist()} for name, value in _scales(manifest).items()],
    }
    scales.write_text(json.dumps(content), encoding="utf-8")
    output = tmp_path / "run"
    arguments = ["--root", str(root), "--manifest", str(directory), "--plans", str(plans), "--scales", str(scales)]
    options = ["--wide-plans", str(wide), "--smoke", "--device", "cpu", "--jobs", "1", "--draws", "20"]
    assert script.main(["levels", *arguments, "--output", str(output), *options]) == 0
    printed = capsys.readouterr().out
    for part in ("Read 6 recordings: 36 events", "fold-0:", "repeat-0:", "6 determinism  PASS", "Step 3: measures"):
        assert part in printed
    summary = tmp_path / "summary.json"
    assert script.main(["read", "--run", str(output), "--draws", "20", "--output", str(summary)]) == 0
    assert "Step 3: measures" in capsys.readouterr().out
    assert set(json.loads(summary.read_text())["decisions"]["outcomes"]) >= {"h", "margin", "S1", "S7", "defaults"}
    assert script.main(["levels", *arguments, "--output", str(output), *options]) == 2
    assert "complete run" in capsys.readouterr().err
    assert script.main(["levels", *arguments, "--output", str(wide / "run"), *options]) == 2

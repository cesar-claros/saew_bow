"""The recording test of the slice: an event against its own displaced control; run on the HPC."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest
from numpy.typing import NDArray

from eeg_compare.data.annotations import MONTAGE
from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.dataset import TUEVReader
from eeg_compare.data.manifest import CorpusLayout, Manifest, build_manifest, load_manifest
from eeg_compare.data.provenance import write_json
from eeg_compare.data.recordings import Recording
from eeg_compare.slice.detect import (
    NULL_ARM,
    PAIR_ARMS,
    PAIR_READOUTS,
    evaluate_detection,
    pair_values,
    read_detection,
    render_detection,
    run_detection,
)
from eeg_compare.slice.exposure import ExposureConfig, build_plans, write_plans
from eeg_compare.slice.runner import Settings
from eeg_compare.slice.study import Study, load_study

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_slice.py"
LABELS = ["spsw", "gped", "pled", "eyem", "artf", "bckg"]
SMALL = Settings(
    n_atoms=4,
    seeds=(0,),
    forest_seeds=(0,),
    forest_trees=5,
    kmeans_passes=10,
    null_permutations=2,
    max_passes=3,
    n_jobs=1,
    device="cpu",
)


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
    return load_study(manifest, build_plans(manifest, ExposureConfig()), TUEVReader(corpus[0], manifest))


def _row(arm: str, head: str, label: str, value: float, low: float) -> dict[str, Any]:
    return {
        "arm": arm,
        "head": head,
        "readout": "full",
        "class": label,
        "value": value,
        "low": low,
        "high": value + 0.05,
    }


def _summary(rows: list[dict[str, Any]], null: dict[str, float] | None = None, equal: bool = True) -> dict[str, Any]:
    values = {label: 0.5 for label in LABELS} if null is None else null
    return {
        "labels": LABELS,
        "rows": rows,
        "null": [{"class": label, "value": value} for label, value in values.items()],
        "gate_determinism": {"equal_scores": equal},
    }


def test_a_pair_counts_when_the_event_has_the_higher_score() -> None:
    assert pair_values(np.array([0.9, 0.2, 0.5]), np.array([0.1, 0.7, 0.5])).tolist() == [1.0, 0.0, 0.5]


def test_the_rules_of_the_recording_test() -> None:
    rows = [
        *(_row("spectral", head, "eyem", 0.90, 0.80) for head in ("logistic", "forest")),
        _row("S1-L2", "logistic", "eyem", 0.70, 0.60),
        _row("S1-L2", "forest", "eyem", 0.55, 0.45),
        # a value above the limit does not count when its interval holds 0.5
        *(_row("spectral", head, "gped", 0.62, 0.49) for head in ("logistic", "forest")),
        *(_row("S1-L2", head, "gped", 0.61, 0.52) for head in ("logistic", "forest")),
        *(_row("spectral", head, "pled", 0.58, 0.53) for head in ("logistic", "forest")),
        *(_row("spectral", head, "bckg", 0.52, 0.47) for head in ("logistic", "forest")),
        # another arm does not decide
        *(_row("SF-frozen-data", head, "spsw", 0.95, 0.90) for head in ("logistic", "forest")),
    ]
    found = evaluate_detection(_summary(rows))
    assert found["readable"]["eyem"] == ["spectral"] and found["readable"]["gped"] == ["S1-L2"]
    assert found["readable"]["pled"] == [] and found["readable"]["spsw"] == [] and found["readable"]["bckg"] == []
    assert found["outcome"] == "event_signal" and not found["region_confound"]
    assert found["gate_null"] and found["gate_determinism"]
    without = evaluate_detection(_summary([row for row in rows if row["class"] != "gped"]))
    assert without["outcome"] == "no_event_signal" and without["readable"]["eyem"] == ["spectral"]
    marked = evaluate_detection(
        _summary([*rows, *(_row("S1-L2", head, "bckg", 0.70, 0.62) for head in ("logistic", "forest"))])
    )
    assert marked["region_confound"] and marked["readable"]["bckg"] == ["S1-L2"]
    assert not evaluate_detection(_summary(rows, {**{label: 0.5 for label in LABELS}, "gped": 0.56}))["gate_null"]
    assert not evaluate_detection(_summary(rows, {}))["gate_null"]
    assert not evaluate_detection(_summary(rows, equal=False))["gate_determinism"]


def test_the_measure_is_a_mean_over_patients_and_seeds(tmp_path: Path) -> None:
    # four patients of one class; spectral ranks every pair, S1-L2 ranks some, and its second seed ranks all
    pairs = [("p1", "e1"), ("p1", "e2"), ("p2", "e3"), ("p2", "e4"), ("p3", "e5"), ("p4", "e6")]
    first = {"e1": (0.9, 0.1), "e2": (0.8, 0.2), "e3": (0.7, 0.3), "e4": (0.2, 0.6), "e5": (0.5, 0.5), "e6": (0.1, 0.9)}
    rows = []
    for head in ("logistic", "forest"):
        for _, example in pairs:
            rows.append(("spectral", 0, head, "full", 0, "gped", example, 0.9, 0.1))
            rows.append(("S1-L2", 0, head, "full", 0, "gped", example, *first[example]))
            rows.append(("S1-L2", 1, head, "full", 0, "gped", example, 0.9, 0.1))
            rows.append((NULL_ARM, 0, head, "full", 0, "gped", example, *first[example]))
    names = ["arm", "seed", "head", "readout", "fold", "label", "example_id", "score_event", "score_control"]
    root = tmp_path / "run"
    root.mkdir()
    pl.DataFrame(rows, schema=names, orient="row").write_parquet(root / "pairs.parquet")
    pl.DataFrame({"example_id": [e for _, e in pairs], "cluster_id": [p for p, _ in pairs]}).write_parquet(
        root / "examples.parquet"
    )
    report = {
        "labels": ["gped"],
        "settings": {},
        "gate_determinism": {"equal_scores": True, "largest": 0.0},
        "total_seconds": 1.0,
    }
    write_json(root / "report.json", report)
    summary = read_detection(root, n_draws=200)
    found = {(row["arm"], row["head"]): row for row in summary["rows"]}
    assert found[("spectral", "logistic")]["value"] == 1.0 and found[("spectral", "logistic")]["half_width"] == 0.0
    # patient means of the first seed: 1, 0.5, 0.5, 0; with the second seed: 1, 0.75, 0.75, 0.5
    assert found[("S1-L2", "logistic")]["value"] == pytest.approx(0.75)
    assert (found[("S1-L2", "logistic")]["patients"], found[("S1-L2", "logistic")]["pairs"]) == (4, 6)
    assert 0.5 <= found[("S1-L2", "logistic")]["low"] <= 0.75 <= found[("S1-L2", "logistic")]["high"] <= 1.0
    assert summary["null"][0]["value"] == pytest.approx(0.5) and NULL_ARM not in {row["arm"] for row in summary["rows"]}
    difference = next(row for row in summary["differences"] if row["first"] == "S1-L2/logistic/full")
    assert difference["second"] == "spectral/logistic/full" and difference["difference"] == pytest.approx(-0.25)
    assert difference["low"] <= -0.25 <= difference["high"] <= 0.0
    assert read_detection(root, n_draws=200)["rows"] == summary["rows"]


def test_the_recording_test_reads_an_event_that_its_control_does_not_hold(study: Study, tmp_path: Path) -> None:
    output = tmp_path / "detect"
    report = run_detection(study, SMALL, output, {"name": "test"})
    assert sorted(path.name for path in output.iterdir()) == ["examples.parquet", "pairs.parquet", "report.json"]
    assert report["step"] == "detect" and report["gate_determinism"]["equal_scores"]
    pairs = pl.read_parquet(output / "pairs.parquet")
    assert set(pairs["arm"].unique().to_list()) == {*PAIR_ARMS, NULL_ARM}
    assert set(pairs["readout"].unique().to_list()) == set(PAIR_READOUTS) and set(
        pairs["label"].unique().to_list()
    ) == set(LABELS)
    groups = pairs.group_by("arm", "seed", "head", "readout").agg(
        pl.col("example_id").n_unique().alias("events"), pl.len()
    )
    assert set(groups["events"].to_list()) == {36} and set(groups["len"].to_list()) == {36}
    summary = read_detection(output, n_draws=50)
    rows = {(row["arm"], row["head"], row["readout"], row["class"]): row for row in summary["rows"]}
    assert len(rows) == len(PAIR_ARMS) * 2 * 3 * 6
    # the burst of a class is in its labeled second and not in the control, so the spectral control reads every class
    for label in LABELS:
        found = rows[("spectral", "logistic", "full", label)]
        assert found["value"] >= 0.8 and found["patients"] == 6 and found["pairs"] == 6
        assert rows[("spectral", "logistic", "event", label)]["value"] >= 0.8
    assert len(summary["null"]) == 6 and len(summary["differences"]) == 4 * 2 * 6
    outcomes = evaluate_detection(summary)
    assert set(outcomes) == {"readable", "region_confound", "outcome", "gate_null", "gate_determinism"}
    assert outcomes["gate_determinism"] and outcomes["outcome"] in ("event_signal", "no_event_signal")
    text = render_detection(summary, outcomes)
    for part in ("Recording test", "Readout full", "Readout context", "Paired differences", "Outcome:"):
        assert part in text
    with pytest.raises(ValueError, match="never overwritten"):
        run_detection(study, SMALL, output, {})


def test_the_script_runs_the_recording_test(
    corpus: tuple[Path, Path], manifest: Manifest, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = SimpleNamespace(**runpy.run_path(str(SCRIPT)))
    root, directory = corpus
    plans = tmp_path / "plans"
    write_plans(manifest, ExposureConfig(), plans)
    rng = np.random.default_rng(7)
    scales = tmp_path / "scales.json"
    content = {
        "band_hz": [None, None],
        "measure_1_profile": {"profile_factor_by_channel": {name: 1.0 for name in MONTAGE}},
        "scales": [
            {"recording_id": row.recording_id, "scale": np.exp(rng.normal(-11.0, 0.3, 22)).tolist()}
            for row in manifest.recordings
            if row.official_split == "train"
        ],
    }
    scales.write_text(json.dumps(content), encoding="utf-8")
    output = tmp_path / "detect"
    arguments = ["--root", str(root), "--manifest", str(directory), "--plans", str(plans), "--scales", str(scales)]
    options = ["--smoke", "--device", "cpu", "--jobs", "1", "--draws", "20"]
    assert script.main(["detect", *arguments, "--output", str(output), *options]) == 0
    printed = capsys.readouterr().out
    for part in ("Read 6 recordings: 36 events", "Recording test", "determinism  PASS", "Outcome:"):
        assert part in printed
    summary = tmp_path / "summary.json"
    assert script.main(["read", "--run", str(output), "--draws", "20", "--output", str(summary)]) == 0
    assert "Recording test" in capsys.readouterr().out
    assert json.loads(summary.read_text())["outcomes"]["outcome"] in ("event_signal", "no_event_signal")
    assert script.main(["detect", *arguments, "--output", str(output), *options]) == 2
    assert "never overwritten" in capsys.readouterr().err

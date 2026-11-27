"""Exposure plans of the slice: quotas, folds, storage and the plan script; run on the HPC."""

from __future__ import annotations

import runpy
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from eeg_compare.data.manifest import Manifest, load_manifest
from eeg_compare.data.provenance import file_hash, write_json
from eeg_compare.slice.exposure import (
    PLAN_SCHEMA,
    SOURCES,
    ExposureConfig,
    build_plans,
    check_plans,
    context_units,
    draw,
    fold_units,
    interval_units,
    load_plans,
    pooled_counts,
    write_plans,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_exposure.py"
INTERVALS, CONTEXTS = SOURCES
TWO_INTERVALS = ExposureConfig(minutes_per_patient=10.0 / 60.0)


@pytest.fixture
def manifest(mini_corpus: tuple[Path, Path]) -> Manifest:
    """Three training recordings of 20 s, one per patient with six kept events each, and two folds."""
    return load_manifest(mini_corpus[1])


def _training(manifest: Manifest, fold: int) -> set[str]:
    return {
        row.cluster_id for row in manifest.assignments if row.official_split == "train" and row.validation_fold != fold
    }


def _edited(plan: pl.DataFrame, index: int, **changes: Any) -> pl.DataFrame:
    rows = plan.to_dicts()
    rows[index] = {**rows[index], **changes}
    return pl.DataFrame(rows, schema=PLAN_SCHEMA)


def _first(plan: pl.DataFrame, source: str) -> int:
    return plan["source"].to_list().index(source)


def _synthetic(counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"unit_id": f"{patient}:{index}", "cluster_id": patient, "label": "odd" if index % 2 else "even"}
        for patient, count in counts.items()
        for index in range(count)
    ]


def test_the_configuration_gives_the_quota_in_intervals_and_refuses_bad_values() -> None:
    assert ExposureConfig().intervals_per_patient == 60
    assert ExposureConfig(minutes_per_patient=10.0).intervals_per_patient == 120
    assert TWO_INTERVALS.intervals_per_patient == 2
    bad_values: tuple[dict[str, Any], ...] = (
        {"minutes_per_patient": 0.0},
        {"interval_seconds": -1.0},
        {"minutes_per_patient": 0.05},
        {"events_per_patient_class": 0},
        {"events_per_patient_class": True},
        {"seed": 1.5},
    )
    for bad in bad_values:
        with pytest.raises(ValueError):
            ExposureConfig(**bad)


def test_intervals_lie_on_the_grid_of_the_training_recordings(manifest: Manifest) -> None:
    units = interval_units(manifest, ExposureConfig())
    training = {row.recording_id: row.cluster_id for row in manifest.recordings if row.official_split == "train"}
    starts = [0, 1250, 2500, 3750]
    assert len(training) == 3 and len(units) == 12
    assert {unit["recording_id"] for unit in units} == set(training)
    for recording_id, cluster_id in training.items():
        own = [unit for unit in units if unit["recording_id"] == recording_id]
        assert [unit["start_sample"] for unit in own] == starts
        assert [unit["unit_id"] for unit in own] == [f"{recording_id}:{start}" for start in starts]
        assert all(unit["n_samples"] == 1250 and unit["label"] is None for unit in own)
        assert all(unit["cluster_id"] == cluster_id for unit in own)
    # the samples after the last full interval belong to no unit
    longer = replace(manifest, recordings=tuple(replace(row, n_samples=5600) for row in manifest.recordings))
    assert interval_units(longer, ExposureConfig()) == units
    shorter = replace(manifest, recordings=tuple(replace(row, n_samples=1249) for row in manifest.recordings))
    assert interval_units(shorter, ExposureConfig()) == []
    assert len(interval_units(manifest, ExposureConfig(interval_seconds=2.0))) == 30
    with pytest.raises(ValueError, match="whole number"):
        interval_units(manifest, ExposureConfig(interval_seconds=0.0101))


def test_contexts_are_the_kept_training_events(manifest: Manifest) -> None:
    units = context_units(manifest)
    kept = (
        manifest.scan_examples()
        .filter((pl.col("official_split") == "train") & (pl.col("kind") == "event") & (pl.col("status") == "kept"))
        .collect()
    )
    assert len(units) == kept.height == 18
    assert {unit["unit_id"] for unit in units} == set(kept["example_id"])
    assert all(unit["n_samples"] == 1250 for unit in units)
    assert sorted(Counter(unit["label"] for unit in units).values()) == [3] * 6
    assert sorted(Counter((unit["cluster_id"], unit["label"]) for unit in units).values()) == [1] * 18
    # a context of a plan holds real samples only
    padded = manifest.scan_examples().collect().with_columns(pl.lit(5).alias("pad_left"))
    with pytest.raises(ValueError, match="zero padding"):
        context_units(SimpleNamespace(scan_examples=padded.lazy))  # type: ignore[arg-type]


def test_a_draw_keeps_the_quota_of_every_group_without_repeats() -> None:
    units = _synthetic({"a": 2, "b": 3, "c": 40, "d": 200})
    kept = draw(units, ("cluster_id",), 5, 0)
    assert Counter(unit["cluster_id"] for unit in kept) == {"a": 2, "b": 3, "c": 5, "d": 5}
    assert len({unit["unit_id"] for unit in kept}) == len(kept)
    assert [unit["rank"] for unit in kept if unit["cluster_id"] == "d"] == [0, 1, 2, 3, 4]
    assert [unit["cluster_id"] for unit in kept] == sorted(unit["cluster_id"] for unit in kept)
    # the order of the input and the units of other patients do not change the choice
    assert draw(list(reversed(units)), ("cluster_id",), 5, 0) == kept
    alone = [unit for unit in units if unit["cluster_id"] == "d"]
    assert draw(alone, ("cluster_id",), 5, 0) == [unit for unit in kept if unit["cluster_id"] == "d"]
    # a larger quota keeps the units of a smaller one
    assert [unit for unit in draw(units, ("cluster_id",), 9, 0) if unit["rank"] < 5] == kept
    # another seed gives another choice
    other = draw(units, ("cluster_id",), 5, 1)
    chosen = [{unit["unit_id"] for unit in rows if unit["cluster_id"] == "d"} for rows in (kept, other)]
    assert chosen[0] != chosen[1]
    # a group can be the patient and the class
    pairs = draw(units, ("cluster_id", "label"), 1, 0)
    expected = {(patient, label): 1 for patient in "abcd" for label in ("even", "odd")}
    assert Counter((unit["cluster_id"], unit["label"]) for unit in pairs) == expected
    with pytest.raises(ValueError, match="quota"):
        draw(units, ("cluster_id",), 0, 0)
    with pytest.raises(ValueError, match="twice"):
        draw([*units, units[0]], ("cluster_id",), 5, 0)


def test_a_plan_holds_the_training_patients_of_its_fold_only(manifest: Manifest) -> None:
    plan = build_plans(manifest, TWO_INTERVALS)
    assert plan.columns == list(PLAN_SCHEMA)
    assert all(plan.schema[name] == kind for name, kind in PLAN_SCHEMA.items())
    patients = {row.cluster_id for row in manifest.assignments if row.official_split == "train"}
    evaluation = {row.recording_id for row in manifest.recordings if row.official_split == "eval"}
    assert len(patients) == 3 and len(evaluation) == 1 and not evaluation & set(plan["recording_id"])
    held_out: set[str] = set()
    for fold in range(2):
        training = _training(manifest, fold)
        assert 0 < len(training) < 3
        held_out |= patients - training
        intervals, contexts = fold_units(plan, fold, INTERVALS), fold_units(plan, fold, CONTEXTS)
        assert set(intervals["cluster_id"]) == set(contexts["cluster_id"]) == training
        assert Counter(intervals["cluster_id"]) == {patient: 2 for patient in training}
        assert sorted(intervals["rank"].unique()) == [0, 1]
        assert intervals["label"].null_count() == intervals.height
        assert Counter(contexts["cluster_id"]) == {patient: 6 for patient in training}
        assert contexts["rank"].max() == 0 and contexts["label"].n_unique() == 6
    assert held_out == patients
    # a larger quota holds the units of a smaller one
    full = build_plans(manifest, ExposureConfig())
    assert fold_units(full, 0, INTERVALS).height == 4 * len(_training(manifest, 0))
    assert set(fold_units(plan, 0, INTERVALS)["unit_id"]) < set(fold_units(full, 0, INTERVALS)["unit_id"])
    with pytest.raises(ValueError, match="source"):
        fold_units(plan, 0, "other")


def test_stored_plans_load_only_when_they_match_the_manifest_and_a_new_build(
    manifest: Manifest, tmp_path: Path
) -> None:
    out = tmp_path / "plans"
    metadata = write_plans(manifest, TWO_INTERVALS, out)
    plan, config = load_plans(out, manifest)
    assert config == TWO_INTERVALS and plan.equals(build_plans(manifest, TWO_INTERVALS))
    assert metadata["summary"] == check_plans(plan, manifest, config)
    assert metadata["pooled"] == pooled_counts(plan, 250.0)
    assert sorted(path.name for path in out.iterdir()) == ["metadata.json", "plan.parquet"]
    with pytest.raises(ValueError, match="never overwritten"):
        write_plans(manifest, TWO_INTERVALS, out)

    def copy(name: str, table: pl.DataFrame, **changes: Any) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        table.write_parquet(directory / "plan.parquet")
        content = {**metadata, "plan_hash": file_hash(directory / "plan.parquet"), **changes}
        write_json(directory / "metadata.json", content)
        return directory

    assert load_plans(copy("same", plan), manifest)[0].equals(plan)
    with pytest.raises(ValueError, match="differs from a new build"):
        load_plans(copy("shorter", plan.head(plan.height - 1)), manifest)
    with pytest.raises(ValueError, match="differs from a new build"):
        load_plans(copy("quota", plan, config={**metadata["config"], "minutes_per_patient": 5.0}), manifest)
    with pytest.raises(ValueError, match="another manifest"):
        load_plans(copy("foreign", plan, manifest_config_hash="0" * 64), manifest)
    with pytest.raises(ValueError, match="unsupported"):
        load_plans(copy("old", plan, schema_version=0), manifest)
    with (out / "plan.parquet").open("ab") as handle:
        handle.write(b"modified")
    with pytest.raises(ValueError, match="changed after"):
        load_plans(out, manifest)


def test_the_check_refuses_a_plan_that_breaks_a_rule(manifest: Manifest) -> None:
    config = ExposureConfig()
    plan = build_plans(manifest, config)
    assert check_plans(plan, manifest, config)
    interval, context = _first(plan, INTERVALS), _first(plan, CONTEXTS)
    rows = plan.to_dicts()
    other = next(row for row in rows if row["cluster_id"] != rows[interval]["cluster_id"])
    outside = next(row for row in manifest.recordings if row.official_split == "eval")
    cases = [
        ("outside the training part", _edited(plan, interval, fold=1 - rows[interval]["fold"])),
        (
            "outside the training part",
            _edited(plan, interval, cluster_id=outside.cluster_id, recording_id=outside.recording_id),
        ),
        ("not a training recording of its patient", _edited(plan, interval, recording_id=other["recording_id"])),
        ("not a training recording of its patient", _edited(plan, interval, recording_id="unknown")),
        ("repeats", pl.concat([plan, plan.head(1)])),
        ("off the grid", _edited(plan, interval, start_sample=100)),
        ("off the grid", _edited(plan, interval, start_sample=5000)),
        ("off the grid", _edited(plan, interval, n_samples=2500)),
        ("not a kept training event", _edited(plan, context, unit_id="unknown")),
        ("not a kept training event", _edited(plan, context, start_sample=rows[context]["start_sample"] + 1)),
        ("unknown source or fold", _edited(plan, interval, source="other")),
        ("unknown source or fold", _edited(plan, interval, fold=7)),
    ]
    for message, bad in cases:
        with pytest.raises(ValueError, match=message):
            check_plans(bad, manifest, config)
    with pytest.raises(ValueError, match="exceeds a quota"):
        check_plans(plan, manifest, TWO_INTERVALS)


def test_the_check_counts_units_patients_and_hours(manifest: Manifest) -> None:
    config = ExposureConfig()
    plan = build_plans(manifest, config)
    summary = check_plans(plan, manifest, config)
    keys = [(row["fold"], row["source"]) for row in summary]
    assert keys == [(fold, source) for fold in range(2) for source in sorted(SOURCES)]
    for row in summary:
        patients = len(_training(manifest, row["fold"]))
        each = 4 if row["source"] == INTERVALS else 6
        assert (row["patients"], row["units"], row["largest_patient"]) == (patients, each * patients, each)
        assert row["effective_patients"] == pytest.approx(patients)
        assert row["hours"] == pytest.approx(row["units"] * 5.0 / 3600.0)
    # unequal patients count as fewer patients
    first = fold_units(plan, 0, INTERVALS)["unit_id"][0]
    fewer = plan.filter(~((pl.col("fold") == 0) & (pl.col("unit_id") == first)))
    counts = list(Counter(fold_units(fewer, 0, INTERVALS)["cluster_id"]).values())
    found = next(row for row in check_plans(fewer, manifest, config) if (row["fold"], row["source"]) == (0, INTERVALS))
    assert found["effective_patients"] == pytest.approx(sum(counts) ** 2 / sum(value**2 for value in counts))
    assert found["effective_patients"] < found["patients"] or found["patients"] == 1
    # over all folds together a unit counts once, so the counts are those of the draw before the cut by fold
    pooled = pooled_counts(plan, 250.0)
    assert [(row["fold"], row["source"]) for row in pooled] == [("all", INTERVALS), ("all", CONTEXTS)]
    for row, each in zip(pooled, (4, 6), strict=True):
        assert (row["patients"], row["units"], row["largest_patient"]) == (3, 3 * each, each)
        assert row["effective_patients"] == pytest.approx(3.0)
        assert row["hours"] == pytest.approx(3 * each * 5.0 / 3600.0)
    assert pooled_counts(plan.clear(), 250.0) == []


def test_the_script_writes_and_verifies_plans(
    mini_corpus: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = SimpleNamespace(**runpy.run_path(str(SCRIPT)))
    directory, out = str(mini_corpus[1]), tmp_path / "plans"
    assert script.main(["--manifest", directory, "--output", str(out), "--minutes", "0.2"]) == 0
    written = capsys.readouterr().out
    assert "Wrote" in written and written.count(INTERVALS) == written.count(CONTEXTS) == 3
    assert written.count(" all ") == 2
    assert script.main(["--manifest", directory, "--check", str(out)]) == 0
    assert "Verified" in capsys.readouterr().out
    assert script.main(["--manifest", directory, "--output", str(out)]) == 2
    assert "never overwritten" in capsys.readouterr().err
    assert script.main(["--manifest", directory, "--output", str(mini_corpus[1] / "plans")]) == 2
    assert "outside the manifest" in capsys.readouterr().err
    assert script.main(["--manifest", directory, "--check", str(tmp_path / "absent")]) == 2
    assert script.main(["--manifest", directory, "--output", str(tmp_path / "bad"), "--events", "0"]) == 2
    assert not (tmp_path / "bad").exists()

"""Runs, budget rule and reader of the synthetic benchmark on the small smoke setting; run on the HPC.

The smoke setting checks the wiring. Its numbers decide nothing, so the tests
assert structure and the fixed rules, and only loose bounds on measures.
"""

from __future__ import annotations

import json
import runpy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from eeg_compare.cells.api import CELL_NAMES, CELLS, Geometry, draw_start, start_atoms
from eeg_compare.synthetic.benchmark import (
    KINDS,
    Settings,
    calibrate_budget,
    check_settings,
    fixed_point_atoms,
    levels,
    loss_change,
    oracle_atoms,
    read_root,
    render,
    run_level,
    settings_from_json,
    smoke_settings,
)
from eeg_compare.synthetic.generator import generate

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_synthetic.py"
TINY = smoke_settings()
STATUSES = {"pass", "fail", "incomplete", "not evaluated"}


def test_levels_and_stored_settings(tmp_path: Path) -> None:
    made = levels(TINY)
    assert set(made) == {
        "base",
        "background",
        "polarity",
        "more_atoms",
        "delay_4",
        "delay_8",
        "delay_16",
        "few_10",
        "few_25",
        "few_50",
    }
    assert made["polarity"].n_atoms == 2 and made["more_atoms"].n_atoms == 6 and made["base"].n_atoms is None
    assert made["background"].generator == {"events": False} and made["delay_16"].generator == {"max_delay": 16}
    assert settings_from_json(TINY.as_json()) == TINY and settings_from_json(Settings().as_json()) == Settings()
    root = tmp_path / "root"
    check_settings(root, TINY)
    check_settings(root, TINY)
    with pytest.raises(ValueError, match="another setting"):
        check_settings(root, replace(TINY, n_atoms=4))


def test_oracle_and_fixed_point_dictionaries() -> None:
    run = generate(TINY.generator, seed=0)
    truth = run.truth
    geometry = Geometry(3, 32, 32, 2)
    start = draw_start(run.train, geometry, torch.Generator().manual_seed(0))
    single = oracle_atoms("S1", truth, torch.float32)
    full = oracle_atoms("SF", truth, torch.float32)
    separable = oracle_atoms("SR", truth, torch.float32)
    assert single.waveform is not None and single.waveform.shape == (2, 32) and single.waveform.dtype == torch.float32
    assert full.template is not None and full.template.shape == (2, 4, 32)
    assert separable.pattern is not None and separable.pattern.shape == (2, 4, 1)
    for structure in ("S1", "SF", "SR"):
        initial = start_atoms(structure, start, geometry)  # type: ignore[arg-type]
        planted = fixed_point_atoms(structure, truth, initial)  # type: ignore[arg-type]
        assert planted.n_atoms == 3
        if structure == "SF":
            assert planted.template is not None and initial.template is not None
            assert torch.allclose(planted.template[:2].double(), truth.templates, atol=1e-6)
            assert torch.equal(planted.template[2:], initial.template[2:])
        else:
            assert planted.waveform is not None and initial.waveform is not None
            assert torch.allclose(planted.waveform[:2].double(), truth.waveforms, atol=1e-6)
            assert torch.equal(planted.waveform[2:], initial.waveform[2:])
        if structure == "SR":
            assert planted.pattern is not None and initial.pattern is not None
            assert torch.allclose(planted.pattern[:2, :, 0].double(), truth.patterns, atol=1e-6)
            assert torch.equal(planted.pattern[:, :, 1], initial.pattern[:, :, 1])
            assert torch.equal(planted.pattern[2:], initial.pattern[2:])
    with pytest.raises(ValueError, match="planted templates"):
        fixed_point_atoms("S1", truth, oracle_atoms("S1", replace(truth, waveforms=truth.waveforms[:1]), torch.float32))


def test_loss_change_compares_the_last_two_tenths() -> None:
    assert loss_change([1.0] * 80 + [0.9] * 20, 100) == 0.0
    assert abs(loss_change([1.0] * 90 + [0.5] * 10, 100) - 0.5) < 1e-12
    assert abs(loss_change([2.0] * 40 + [1.0] * 5 + [0.8] * 5 + [9.0] * 50, 50) - 0.2) < 1e-12
    with pytest.raises(ValueError, match="at least 10 steps"):
        loss_change([1.0] * 5, 5)
    with pytest.raises(ValueError, match="trace"):
        loss_change([1.0] * 20, 30)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A root with the base level (with the fixed-point fits) and the background level for both seeds."""
    folder = tmp_path_factory.mktemp("synthetic") / "root"
    check_settings(folder, TINY)
    made = levels(TINY)
    for seed in TINY.seeds:
        run_level(made["base"], seed, TINY, 6, "cpu", fixed_point=True, output=folder / "runs" / "base" / f"seed{seed}")
        run_level(made["background"], seed, TINY, 6, "cpu", output=folder / "runs" / "background" / f"seed{seed}")
    return folder


def test_a_run_writes_one_file_per_cell_with_its_dictionaries(root: Path) -> None:
    output = root / "runs" / "base" / "seed0"
    assert sorted(path.name for path in output.glob("*.json")) == sorted(f"{name}.json" for name in CELL_NAMES)
    records = run_level(levels(TINY)["base"], 0, TINY, 6, "cpu", fixed_point=True, output=output)
    assert len(records) == 9 * 4, "a finished cell must be read from its file"
    for record in records:
        cell = CELLS[record["cell"]]
        assert record["kind"] in KINDS and record["level"] == "base" and record["seed"] == 0
        assert len(record["waveform_recovery"]) == 2 and len(record["decoy_recovery"]) == 4
        assert (record["template_recovery"] is None) == (cell.structure == "S1")
        assert record["detection"]["n_events"] == 16
        assert (record["fit"] is not None) == (record["kind"] in ("learned", "fixed_point"))
        if record["kind"] == "oracle":
            assert record["n_atoms"] == 2 and record["waveform_recovery_mean"] > 0.99
            assert record["detection"]["recall"] >= 0.5, "the planted dictionary must find its strong events"
        else:
            assert record["n_atoms"] == 3
        if record["fit"] is not None:
            assert ("steps" in record["fit"]) == cell.gradient and ("passes" in record["fit"]) != cell.gradient
            assert record["fit"]["seconds"] >= 0
    saved = torch.load(output / "SR-L5.pt", weights_only=True)
    assert set(saved) == {"waveform", "pattern"} and saved["pattern"].shape == (3, 4, 2)
    quiet = json.loads((root / "runs" / "background" / "seed1" / "S1-L3.json").read_text(encoding="utf-8"))
    assert [record["kind"] for record in quiet] == ["learned", "frozen", "oracle"]
    assert all(record["detection"]["n_events"] == 0 and record["detection"]["recall"] is None for record in quiet)


def test_a_repeated_run_gives_the_same_measures(root: Path, tmp_path: Path) -> None:
    repeat = tmp_path / "repeat"
    check_settings(repeat, TINY)
    run_level(levels(TINY)["base"], 0, TINY, 6, "cpu", fixed_point=True, output=repeat / "runs" / "base" / "seed0")
    report = read_root(root, TINY, repeat)
    gate = report["gates"]["6_determinism"]
    assert gate["compared_records"] == 36 and gate["status"] == "pass", gate
    assert gate["largest_difference"] <= 1e-6 and gate["records_of_another_shape"] is False


def test_the_reader_applies_the_fixed_rules(root: Path, tmp_path: Path) -> None:
    report = read_root(root, TINY)
    assert len(report["summary"]) == 2 * 9 * 4 + 2 * 9 * 3
    assert set(report["gates"]) == {"4_signal_level", "5_fixed_point", "range", "6_determinism"}
    assert all(gate["status"] in STATUSES for gate in report["gates"].values())
    assert report["gates"]["6_determinism"]["status"] == "not evaluated"
    for name in ("4_signal_level", "5_fixed_point", "range"):
        assert report["gates"][name]["status"] in {"pass", "fail"}, report["gates"][name]
    assert all(value is not None for value in report["gates"]["4_signal_level"]["oracle_recall"].values())
    assert all(floor is not None and len(floor) == 2 for floor in report["floors"].values())
    assert report["predictions"]["P5"]["status"] in {"holds", "does not hold"}
    assert len(report["predictions"]["P5"]["differences_by_seed"]) == 2
    assert report["predictions"]["P7"]["status"] in {"holds", "does not hold"}
    recovered = [row["recovered"] for row in report["summary"]]
    assert all(count is not None and 0 <= count <= 2 for count in recovered)
    text = render(report)
    assert text.startswith("Gates\n") and "P7" in text and "SR-L5n" in text
    json.dumps(report)
    empty = tmp_path / "empty"
    check_settings(empty, TINY)
    nothing = read_root(empty, TINY)
    assert nothing["summary"] == []
    assert {gate["status"] for gate in nothing["gates"].values()} == {"not evaluated"}
    assert {prediction["status"] for prediction in nothing["predictions"].values()} == {"not evaluated"}
    partial = read_root(root, replace(TINY, seeds=(0, 1, 2)))
    assert partial["gates"]["4_signal_level"]["status"] == "not evaluated"
    assert partial["predictions"]["P5"]["status"] == "not evaluated"


def test_the_budget_rule_reads_one_long_fit() -> None:
    result = calibrate_budget(TINY, device="cpu")
    assert result["checkpoints"] == [10, 20] and result["steps"] in (10, 20)
    assert set(result["cells"]) == {name for name in CELL_NAMES if CELLS[name].gradient}
    for entry in result["cells"].values():
        assert len(entry["step_losses"]) >= 20 and set(entry["loss_change"]) == {"10", "20"}
        assert entry["converged_at"] in (None, 10, 20)
        if entry["converged_at"] is not None:
            assert entry["loss_change"][str(entry["converged_at"])] < 0.01
    slowest = max(entry["converged_at"] or 20 for entry in result["cells"].values())
    assert result["steps"] == slowest


def test_the_commands_of_the_script_share_one_root(tmp_path: Path) -> None:
    script = SimpleNamespace(**runpy.run_path(str(SCRIPT)))
    root, repeat = tmp_path / "smoke", tmp_path / "smoke-repeat"
    assert script.main(["read", "--root", str(root)]) == 2, "a folder without a setting is not a root"
    assert script.main(["run", "--root", str(root), "--smoke", "--levels", "base", "--device", "cpu"]) == 2
    assert script.main(["calibrate", "--root", str(root), "--smoke", "--device", "cpu"]) == 0
    budget = json.loads((root / "budget.json").read_text(encoding="utf-8"))["steps"]
    assert script.main(["calibrate", "--root", str(root), "--device", "cpu"]) == 2, "a root holds one budget"
    common = ["--device", "cpu", "--cells", "S1-L4", "S1-L5", "SR-L5"]
    assert script.main(["run", "--root", str(root), "--levels", "base", "--fixed-point", *common]) == 0
    assert script.main(["run", "--root", str(root), "--levels", "background", *common]) == 0
    assert script.main(["run", "--root", str(root), "--levels", "burst_near_event", *common]) == 2
    assert script.main(["run", "--root", str(root), "--levels", "base", "--level", "8", *common]) == 2
    again = ["run", "--root", str(repeat), "--smoke", "--levels", "base", "--seeds", "0", "--fixed-point"]
    assert script.main([*again, "--budget-from", str(root), *common]) == 0
    assert json.loads((repeat / "budget.json").read_text(encoding="utf-8"))["steps"] == budget
    code = script.main(["read", "--root", str(root), "--determinism-root", str(repeat)])
    assert code in (0, 1)
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    assert report["gates"]["6_determinism"]["status"] == "pass" and (root / "summary.csv").stat().st_size > 0
    assert report["gates"]["4_signal_level"]["status"] == "incomplete", "three of nine cells ran"
    assert report["predictions"]["P5"]["status"] in {"holds", "does not hold"}

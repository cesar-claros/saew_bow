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
    MARGINS,
    PREDICTIONS,
    Settings,
    budget_from_traces,
    calibrate_budget,
    check_settings,
    decide,
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
STRESS_LEVELS = ("amplitude_artifacts", "burst_near_event", "overlap", "delay_16", "few_10")


def test_levels_and_stored_settings(tmp_path: Path) -> None:
    made = levels(TINY)
    assert set(made) == {
        "base",
        "background",
        "amplitude_artifacts",
        "burst_near_event",
        "overlap",
        "polarity",
        "polarity_d1",
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
    assert made["polarity_d1"].n_projections == 1 and made["polarity"].n_projections is None
    assert made["burst_near_event"].generator == {"burst": True} and made["overlap"].generator == {"pairs": True}
    older = Settings().as_json()
    for name in ("pool_len", "amplitude_log_sd", "artifacts_per_segment", "burst", "burst_min_share", "pairs"):
        del older["generator"][name]
    assert settings_from_json(older) == Settings(), "a setting file of an earlier version must still load"
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    (earlier / "settings.json").write_text(json.dumps(older), encoding="utf-8")
    check_settings(earlier, Settings())
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


def test_the_rule_of_a_directional_prediction() -> None:
    assert decide([0.10, 0.08, 0.06, 0.07, -0.01], 0.05) == "holds"
    assert decide([0.10, 0.20, -0.01, -0.01, 0.10], 0.05) == "undecided", "only three of five seeds agree"
    assert decide([0.04, 0.04, 0.04, 0.04, 0.04], 0.05) == "undecided", "the mean is below the margin"
    assert decide([-0.10, -0.08, -0.06, -0.07, 0.01], 0.05) == "opposite"
    assert decide([0.2, 0.2], 0.05) == "holds" and decide([0.3, -0.1], 0.05) == "undecided"
    assert set(PREDICTIONS) == {"P1", "P1s", "P2", "P3", "P4", "P6"} and MARGINS["recall"] == 0.10


def test_the_stress_levels_record_their_layout(root: Path) -> None:
    burst = json.loads((root / "runs" / "burst_near_event" / "seed0" / "S1-L3.json").read_text(encoding="utf-8"))
    assert len(burst[0]["burst_check"]) == 2 and len(burst[0]["burst_check"][0]) == 2
    pairs = json.loads((root / "runs" / "overlap" / "seed0" / "SF-L5.json").read_text(encoding="utf-8"))
    counted = pairs[0]["detection"]["pairs"]
    assert counted["n"] == 8 and abs(counted["both"] + counted["one"] + counted["none"] - 1.0) < 1e-12
    base = json.loads((root / "runs" / "base" / "seed0" / "SF-L5.json").read_text(encoding="utf-8"))
    assert base[0]["burst_check"] is None and base[0]["detection"]["pairs"] is None
    few = json.loads((root / "runs" / "few_10" / "seed0" / "SR-L5.json").read_text(encoding="utf-8"))
    assert few[0]["fit"]["steps"] >= 6 and few[0]["level"] == "few_10"


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
    """A root with the base level (with the fixed-point fits), the background level and five more levels."""
    folder = tmp_path_factory.mktemp("synthetic") / "root"
    check_settings(folder, TINY)
    made = levels(TINY)
    for seed in TINY.seeds:
        run_level(made["base"], seed, TINY, 6, "cpu", fixed_point=True, output=folder / "runs" / "base" / f"seed{seed}")
        run_level(made["background"], seed, TINY, 6, "cpu", output=folder / "runs" / "background" / f"seed{seed}")
        for name in STRESS_LEVELS:
            run_level(made[name], seed, TINY, 6, "cpu", output=folder / "runs" / name / f"seed{seed}")
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
    assert len(report["summary"]) == 2 * 9 * 4 + 2 * 9 * 3 * (1 + len(STRESS_LEVELS))
    assert set(report["gates"]) == {"4_signal_level", "5_fixed_point", "range", "6_determinism"}
    assert all(gate["status"] in STATUSES for gate in report["gates"].values())
    assert report["gates"]["6_determinism"]["status"] == "not evaluated"
    for name in ("4_signal_level", "5_fixed_point", "range"):
        assert report["gates"][name]["status"] in {"pass", "fail"}, report["gates"][name]
    assert all(value is not None for value in report["gates"]["4_signal_level"]["oracle_recall"].values())
    assert all(floor is not None and len(floor) == 2 for floor in report["floors"].values())
    predictions = report["predictions"]
    assert list(predictions) == ["P1", "P1s", "P2", "P3", "P4", "P5", "P6", "P7"]
    assert predictions["P5"]["status"] in {"holds", "does not hold"}
    assert len(predictions["P5"]["differences_by_seed"]) == 2
    assert predictions["P7"]["status"] in {"holds", "does not hold"}
    for name in ("P1", "P1s", "P2", "P3", "P4", "P6"):
        entry = predictions[name]
        assert entry["status"] in {"holds", "opposite", "undecided"}, entry
        assert len(entry["differences_by_seed"]) == 2
        assert entry["status"] == decide(entry["differences_by_seed"], MARGINS[PREDICTIONS[name]["measure"]])
        assert ("recall_differences_by_seed" in entry) == (name != "P2")
        assert ("equal_in_base" in entry) == (name in ("P3", "P4"))
    assert predictions["P3"]["equal_in_base"] in {"holds", "does not hold"}
    high = [
        row["template_recovery"]
        for row in report["summary"]
        if (row["level"], row["cell"], row["kind"]) == ("delay_16", "SF-L5", "learned")
    ]
    low = [
        row["template_recovery"]
        for row in report["summary"]
        if (row["level"], row["cell"], row["kind"]) == ("delay_16", "SR-L5", "learned")
    ]
    assert predictions["P3"]["differences_by_seed"] == [a - b for a, b in zip(high, low)]
    recovered = [row["recovered"] for row in report["summary"]]
    assert all(count is not None and 0 <= count <= 2 for count in recovered)
    by_template = report["by_template"]
    assert set(by_template) == {"base", "background", *STRESS_LEVELS} and set(by_template["base"]) == set(CELL_NAMES)
    entry = by_template["base"]["S1-L1"]["learned"]
    assert entry["seeds"] == 2 and len(entry["waveform_recovery"]) == 2 and len(entry["recall"]) == 2
    assert all(value is not None for value in entry["recall"])
    assert by_template["background"]["S1-L1"]["learned"]["recall"] == [None, None]
    assert set(by_template["base"]["SF-L5"]) == set(KINDS) and set(by_template["background"]["SF-L5"]) == set(KINDS[:3])
    text = render(report)
    assert text.startswith("Gates\n") and "P7" in text and "SR-L5n" in text and "by template" in text
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
    assert result["checkpoints"] == [10, 20] and result["steps"] in (10, 20) and result["seed"] == 0
    assert result["rule"] == {"name": "share of the loss decrease", "share": 0.95, "window": 1}
    assert set(result["cells"]) == {name for name in CELL_NAMES if CELLS[name].gradient}
    for entry in result["cells"].values():
        assert len(entry["step_losses"]) >= 20 and set(entry["share_of_decrease"]) == {"10", "20"}
        assert entry["share_of_decrease"]["20"] == 1.0 and entry["converged_at"] in (10, 20)
        assert entry["seconds"] >= 0
    assert result["steps"] == max(entry["converged_at"] for entry in result["cells"].values())
    traces = {name: entry["step_losses"] for name, entry in result["cells"].items()}
    assert budget_from_traces(traces, TINY)["steps"] == result["steps"]


def test_the_budget_is_the_first_checkpoint_with_the_share_of_the_decrease() -> None:
    settings = replace(TINY, first_steps=100, max_steps=800)
    traces = {
        "fast": [1.0] * 10 + [0.5] * 790,
        "step": [1.0] * 10 + [0.6] * 290 + [0.5] * 500,
        "slow": [1.0] * 10 + [0.6] * 90 + [0.53] * 100 + [0.51] * 200 + [0.5] * 400,
        "flat": [0.7] * 800,
    }
    result = budget_from_traces(traces, settings)
    assert result["checkpoints"] == [100, 200, 400, 800] and result["rule"]["window"] == 10
    cells = result["cells"]
    assert [cells[name]["converged_at"] for name in traces] == [100, 400, 400, 100]
    assert abs(cells["step"]["share_of_decrease"]["200"] - 0.8) < 1e-12
    assert abs(cells["slow"]["share_of_decrease"]["200"] - 0.94) < 1e-12, "a share below 0.95 is not converged"
    assert abs(cells["slow"]["share_of_decrease"]["400"] - 0.98) < 1e-12
    assert cells["flat"]["share_of_decrease"] == {"100": 1.0, "200": 1.0, "400": 1.0, "800": 1.0}
    assert result["steps"] == 400 and result["converged_only_at_the_largest_checkpoint"] == []
    late = budget_from_traces({"late": [1.0] * 10 + [0.9] * 690 + [0.5] * 100}, settings)
    assert late["steps"] == 800 and late["converged_only_at_the_largest_checkpoint"] == ["late"]
    with pytest.raises(ValueError, match="needs 800"):
        budget_from_traces({"short": [1.0] * 500}, settings)
    with pytest.raises(ValueError, match="at least one cell"):
        budget_from_traces({}, settings)


def test_the_commands_of_the_script_share_one_root(tmp_path: Path) -> None:
    script = SimpleNamespace(**runpy.run_path(str(SCRIPT)))
    root, repeat = tmp_path / "smoke", tmp_path / "smoke-repeat"
    assert script.main(["read", "--root", str(root)]) == 2, "a folder without a setting is not a root"
    assert script.main(["run", "--root", str(root), "--smoke", "--levels", "base", "--device", "cpu"]) == 2
    assert script.main(["calibrate", "--root", str(root), "--smoke", "--device", "cpu"]) == 0
    budget = json.loads((root / "budget.json").read_text(encoding="utf-8"))["steps"]
    assert script.main(["calibrate", "--root", str(root), "--device", "cpu"]) == 2, "a root holds one budget"
    reread, other_level = tmp_path / "smoke-from-traces", tmp_path / "smoke-other-level"
    assert script.main(["calibrate", "--root", str(reread), "--smoke", "--traces-from", str(root)]) == 0
    stored = json.loads((reread / "budget.json").read_text(encoding="utf-8"))
    assert stored["steps"] == budget and stored["traces_from"] == str(root)
    wrong = ["calibrate", "--root", str(other_level), "--smoke", "--level", "9", "--traces-from", str(root)]
    assert script.main(wrong) == 2, "the traces of a root with another setting are refused"
    common = ["--device", "cpu", "--cells", "S1-L4", "S1-L5", "SR-L5"]
    assert script.main(["run", "--root", str(root), "--levels", "base", "--fixed-point", *common]) == 0
    assert script.main(["run", "--root", str(root), "--levels", "background", *common]) == 0
    assert script.main(["run", "--root", str(root), "--levels", "no_such_level", *common]) == 2
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

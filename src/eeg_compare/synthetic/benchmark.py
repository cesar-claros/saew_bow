"""Runs of the synthetic benchmark (stage 2 protocol, sections 4 to 8).

A root folder holds one setting: the generator, the geometry and the signal
level. ``run_level`` fits the cells of one level and seed and writes one file
per cell. Each file holds the measures of four dictionaries of that cell: the
learned one, the frozen start, the oracle (the planted dictionary) and, on
request, the fit that starts at the planted dictionary (the fixed-point gate).
``calibrate_budget`` applies the convergence rule of the gradient cells.
``read_root`` applies the fixed rules of the gates and of the predictions that
need the base and the background level only.

The thresholds of the rules are constants of this module. They were fixed in
the protocol before any run.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from eeg_compare.cells.api import (
    CELL_NAMES,
    CELLS,
    Budget,
    FittedCell,
    Geometry,
    draw_start,
    fit_cell,
    frozen_cell,
)
from eeg_compare.cells.api import start_atoms as cell_start
from eeg_compare.cells.common import CellAtoms, Structure
from eeg_compare.device import resolve_device
from eeg_compare.synthetic.generator import GeneratorConfig, SyntheticRun, Truth, generate
from eeg_compare.synthetic.measures import atom_map, decoy_recovery, detection, template_recovery, waveform_recovery

__all__ = [
    "KINDS",
    "Level",
    "Settings",
    "calibrate_budget",
    "check_settings",
    "fixed_point_atoms",
    "levels",
    "load_records",
    "loss_change",
    "oracle_atoms",
    "read_root",
    "render",
    "run_level",
    "settings_from_json",
    "smoke_settings",
]

KINDS = ("learned", "frozen", "oracle", "fixed_point")
RECOVERED = 0.80  # smallest recovery of a recovered template
ABOVE_FLOOR = 0.20  # and its smallest distance above the floor of the template
MAP_FLOOR = 0.50  # smallest recovery that maps an atom to a template
ORACLE_RECALL = 0.90  # gate 4
EQUIVALENCE_MARGIN = 0.05  # P5
LOSS_CHANGE = 0.01  # convergence rule of the gradient budget
DETERMINISM_TOLERANCE = 1e-6  # gate 6
_UNTIMED = ("seconds",)


@dataclass(frozen=True)
class Settings:
    """The setting of a root: generator, geometry, budgets of the rules and seeds.

    Parameters
    ----------
    generator : GeneratorConfig
        The base condition. Its ``level`` is the signal level of the root.
    n_atoms, pool_len, n_projections : int
        ``K``, ``P`` and ``D`` of the cells.
    seeds : tuple[int, ...]
        The seeds of a complete condition.
    max_passes : int
        Largest number of passes of an alternating rule.
    tol : float
        Stopping tolerance of an alternating rule.
    lr : float
        Learning rate of the gradient rule.
    segments_per_step : int
        Segments per optimizer step.
    first_steps, max_steps : int
        First and largest budget of the convergence rule of the gradient cells.
    tolerance_s : float
        Largest distance between a claim and the start of a detected event, in seconds.
    """

    generator: GeneratorConfig = GeneratorConfig()
    n_atoms: int = 8
    pool_len: int = 125
    n_projections: int = 2
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_passes: int = 100
    tol: float = 1e-4
    lr: float = 1e-3
    segments_per_step: int = 2
    first_steps: int = 3000
    max_steps: int = 24000
    tolerance_s: float = 0.048

    def as_json(self) -> dict[str, Any]:
        """The setting as plain JSON values."""
        return json.loads(json.dumps(asdict(self)))


def settings_from_json(value: dict[str, Any]) -> Settings:
    """The setting that ``Settings.as_json`` wrote."""
    generator = dict(value["generator"])
    generator["band"] = tuple(generator["band"])
    rest = {name: entry for name, entry in value.items() if name != "generator"}
    rest["seeds"] = tuple(rest["seeds"])
    return Settings(generator=GeneratorConfig(**generator), **rest)


def smoke_settings() -> Settings:
    """A small setting for a quick check of the wiring on a machine; its results decide nothing.

    It keeps the physical times of the protocol at a lower sampling rate: atoms
    and pooling windows of 0.5 s and segments of 5 s at 64 Hz, with 4 channels,
    2 templates, 3 atoms and 2 seeds.
    """
    generator = GeneratorConfig(
        sfreq=64.0,
        n_times=320,
        n_channels=4,
        atom_len=32,
        n_templates=2,
        pattern_channels=2,
        events_per_segment=2,
        min_gap=32,
        level=12.0,
        n_train=12,
        n_test=8,
        calibration_segments=20,
    )
    return Settings(generator, n_atoms=3, pool_len=32, seeds=(0, 1), max_passes=4, first_steps=10, max_steps=20)


@dataclass(frozen=True)
class Level:
    """One level of a condition: changes of the generator and of the number of atoms."""

    name: str
    condition: str
    generator: dict[str, Any] = field(default_factory=dict)
    n_atoms: int | None = None


def levels(settings: Settings) -> dict[str, Level]:
    """The levels that the generator of this step can make, by name."""
    made = [
        Level("base", "base"),
        Level("background", "background", {"events": False}),
        Level("polarity", "polarity", {"random_sign": True}, n_atoms=settings.generator.n_templates),
        Level("more_atoms", "more_atoms", n_atoms=2 * settings.n_atoms),
    ]
    made += [Level(f"delay_{delay}", "delay", {"max_delay": delay}) for delay in (4, 8, 16)]
    made += [Level(f"few_{count}", "few_examples", {"n_train": count}) for count in (10, 25, 50)]
    return {level.name: level for level in made}


def check_settings(root: Path, settings: Settings) -> None:
    """Write the setting of a new root, or refuse a root that holds another setting.

    Raises
    ------
    ValueError
        If the root holds a different setting.
    """
    path = root / "settings.json"
    wanted = settings.as_json()
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != wanted:
            raise ValueError(f"{root} holds another setting; a root holds one setting, so use a new root")
        return
    root.mkdir(parents=True, exist_ok=True)
    _write_json(path, wanted)


def _write_json(path: Path, value: Any) -> None:
    """Write JSON through a temporary file, so that a stopped run leaves no partial file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def oracle_atoms(structure: Structure, truth: Truth, dtype: torch.dtype) -> CellAtoms:
    """The planted dictionary in a structure: waveforms, planted templates, or waveforms with one pattern each."""
    if structure == "S1":
        return CellAtoms("S1", waveform=truth.waveforms.to(dtype))
    if structure == "SF":
        return CellAtoms("SF", template=truth.templates.to(dtype))
    return CellAtoms("SR", waveform=truth.waveforms.to(dtype), pattern=truth.patterns.to(dtype).unsqueeze(-1))


def fixed_point_atoms(structure: Structure, truth: Truth, start: CellAtoms) -> CellAtoms:
    """The start of the fixed-point gate: the planted dictionary in the first ``G`` atoms of the common start.

    For ``SR`` the first spatial pattern of a planted atom is the planted
    pattern; its other patterns stay those of the common start.

    Raises
    ------
    ValueError
        If the start has fewer atoms than planted templates.
    """
    count = int(truth.waveforms.shape[0])
    if start.n_atoms < count:
        raise ValueError(f"the start has {start.n_atoms} atoms for {count} planted templates")
    if structure == "SF":
        assert start.template is not None
        return CellAtoms("SF", template=torch.cat([truth.templates.to(start.template), start.template[count:]]))
    assert start.waveform is not None
    waveform = torch.cat([truth.waveforms.to(start.waveform), start.waveform[count:]])
    if structure == "S1":
        return CellAtoms("S1", waveform=waveform)
    assert start.pattern is not None
    pattern = start.pattern.clone()
    pattern[:count, :, 0] = truth.patterns.to(pattern)
    return CellAtoms("SR", waveform=waveform, pattern=pattern)


def loss_change(losses: Sequence[float], steps: int) -> float:
    """Relative change of the mean loss from the ninth to the last tenth of the first ``steps`` steps."""
    if steps < 10 or len(losses) < steps:
        raise ValueError(f"need at least 10 steps and a trace of {steps} losses, got {len(losses)}")
    tenth = steps // 10
    earlier = sum(losses[steps - 2 * tenth : steps - tenth]) / tenth
    last = sum(losses[steps - tenth : steps]) / tenth
    return abs(last - earlier) / max(abs(earlier), 1e-300)


def _fit_summary(record: dict[str, Any], seconds: float) -> dict[str, Any]:
    """The short record of a fit: passes or steps, the last values of its trace, and the time."""
    summary: dict[str, Any] = {"seconds": seconds}
    if "passes" in record:
        last = record["history"][-1]
        summary.update(passes=record["passes"], converged=record["converged"], last_change=last["change"])
        summary.update(residual_share=last["residual_share"], empty_atoms=last["empty_atoms"])
    else:
        losses = record["step_losses"]
        tenth = max(len(losses) // 10, 1)
        summary.update(steps=record["steps"], epochs=record["epochs"])
        summary.update(first_loss=sum(losses[:tenth]) / tenth, last_loss=sum(losses[-tenth:]) / tenth)
        summary["loss_change"] = loss_change(losses, len(losses)) if len(losses) >= 10 else None
    return summary


def _evaluate(dictionary: FittedCell, run: SyntheticRun, test: torch.Tensor, tolerance: int) -> dict[str, Any]:
    """Recovery and event detection of one dictionary of a cell."""
    truth = run.truth
    max_shift = (run.config.atom_len - 1) // 2
    atoms = dictionary.atoms
    recovery = waveform_recovery(atoms, truth.waveforms, truth.patterns, max_shift)
    template_of_atom, shift_of_atom = atom_map(recovery, MAP_FLOOR)
    full = None if atoms.structure == "S1" else template_recovery(atoms, truth.templates, max_shift).value.tolist()
    return {
        "waveform_recovery": recovery.value.tolist(),
        "waveform_recovery_mean": float(recovery.value.mean()),
        "template_recovery": full,
        "template_recovery_mean": None if full is None else sum(full) / len(full),
        "decoy_recovery": decoy_recovery(atoms, truth.decoys, truth.patterns, max_shift).tolist(),
        "mapped_atoms": int((template_of_atom >= 0).sum()),
        "detection": detection(dictionary.tokens(test), run.test_events, template_of_atom, shift_of_atom, tolerance),
    }


def _atoms_state(atoms: CellAtoms) -> dict[str, torch.Tensor]:
    """The tensors of a dictionary on the CPU, for a later look at the learned atoms."""
    fields = {"waveform": atoms.waveform, "template": atoms.template, "pattern": atoms.pattern}
    return {name: value.detach().cpu() for name, value in fields.items() if value is not None}


def run_level(
    level: Level,
    seed: int,
    settings: Settings,
    steps: int,
    device: str = "auto",
    cells: Sequence[str] = CELL_NAMES,
    fixed_point: bool = False,
    output: Path | None = None,
) -> list[dict[str, Any]]:
    """Fit and measure the cells of one level and seed.

    Parameters
    ----------
    level : Level
        The level.
    seed : int
        Seed of the data, of the common start and of the batch order.
    settings : Settings
        The setting of the root.
    steps : int
        Optimizer steps of the gradient cells.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.
    cells : Sequence[str]
        Names of the cells to run.
    fixed_point : bool
        Also fit every cell from the planted dictionary.
    output : Path | None
        Folder of this level and seed. A cell with a file there is read and not
        fitted again, unless the file lacks a requested fixed-point fit. The
        learned atoms go to ``<cell>.pt``.

    Returns
    -------
    list[dict[str, Any]]
        One record per cell and dictionary kind.
    """
    config = replace(settings.generator, **level.generator)
    run = generate(config, seed)
    target = resolve_device(device)
    train, test = run.train.to(target), run.test.to(target)
    geometry = Geometry(level.n_atoms or settings.n_atoms, config.atom_len, settings.pool_len, settings.n_projections)
    oracle_geometry = Geometry(config.n_templates, config.atom_len, settings.pool_len, 1)
    start = draw_start(train, geometry, torch.Generator().manual_seed(seed))
    budget = Budget(settings.max_passes, settings.tol, steps, settings.lr, settings.segments_per_step)
    tolerance = int(round(settings.tolerance_s * config.sfreq))
    records: list[dict[str, Any]] = []
    for name in cells:
        path = None if output is None else output / f"{name}.json"
        if path is not None and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if not fixed_point or any(record["kind"] == "fixed_point" for record in saved):
                records.extend(saved)
                continue
        cell = CELLS[name]
        initial = cell_start(cell.structure, start, geometry)
        began = time.perf_counter()
        learned = fit_cell(cell, train, initial, geometry, budget, seed=seed, device=device)
        fits: dict[str, dict[str, Any] | None] = {"learned": _fit_summary(learned.record, time.perf_counter() - began)}
        dictionaries = {
            "learned": learned,
            "frozen": frozen_cell(cell, initial, geometry, device),
            "oracle": frozen_cell(cell, oracle_atoms(cell.structure, run.truth, train.dtype), oracle_geometry, device),
        }
        if fixed_point:
            planted = fixed_point_atoms(cell.structure, run.truth, initial)
            began = time.perf_counter()
            dictionaries["fixed_point"] = fit_cell(cell, train, planted, geometry, budget, seed=seed, device=device)
            fits["fixed_point"] = _fit_summary(dictionaries["fixed_point"].record, time.perf_counter() - began)
        made = []
        for kind, dictionary in dictionaries.items():
            made.append(
                {
                    "level": level.name,
                    "condition": level.condition,
                    "seed": seed,
                    "cell": name,
                    "structure": cell.structure,
                    "rule": cell.rule,
                    "kind": kind,
                    "n_atoms": dictionary.geometry.n_atoms,
                    "signal_level": config.level,
                    "fit": fits.get(kind),
                    **_evaluate(dictionary, run, test, tolerance),
                }
            )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_atoms_state(learned.atoms), path.with_suffix(".pt"))
            _write_json(path, made)
        records.extend(made)
    return records


def calibrate_budget(settings: Settings, device: str = "auto", seed: int = 0) -> dict[str, Any]:
    """The budget of the gradient cells by the convergence rule of the protocol.

    Every gradient cell fits once on the base level with the largest budget. The
    budget of a cell is the first of the doubling checkpoints at which the mean
    loss of the last tenth of the steps differs by less than 1% from the mean of
    the tenth before it. A shorter fit with the same seed takes the same first
    steps, so one long fit gives every checkpoint. The budget of all gradient
    cells is the largest budget of a cell, or the largest checkpoint when a cell
    does not converge.

    Returns
    -------
    dict[str, Any]
        ``steps`` (the budget), ``checkpoints``, and per cell the loss change at
        every checkpoint, the checkpoint of convergence and the loss trace.
    """
    checkpoints = [settings.first_steps]
    while checkpoints[-1] * 2 <= settings.max_steps:
        checkpoints.append(checkpoints[-1] * 2)
    config = settings.generator
    run = generate(config, seed)
    train = run.train.to(resolve_device(device))
    geometry = Geometry(settings.n_atoms, config.atom_len, settings.pool_len, settings.n_projections)
    start = draw_start(train, geometry, torch.Generator().manual_seed(seed))
    budget = Budget(settings.max_passes, settings.tol, checkpoints[-1], settings.lr, settings.segments_per_step)
    cells: dict[str, Any] = {}
    for name in CELL_NAMES:
        cell = CELLS[name]
        if not cell.gradient:
            continue
        began = time.perf_counter()
        fitted = fit_cell(cell, train, cell_start(cell.structure, start, geometry), geometry, budget, seed, device)
        losses = fitted.record["step_losses"]
        changes = {str(steps): loss_change(losses, steps) for steps in checkpoints if steps <= len(losses)}
        converged = [steps for steps in checkpoints if changes.get(str(steps), math.inf) < LOSS_CHANGE]
        cells[name] = {
            "loss_change": changes,
            "converged_at": converged[0] if converged else None,
            "seconds": time.perf_counter() - began,
            "step_losses": list(losses),
        }
    chosen = max((entry["converged_at"] or checkpoints[-1]) for entry in cells.values())
    return {"steps": chosen, "checkpoints": checkpoints, "rule": LOSS_CHANGE, "seed": seed, "cells": cells}


def load_records(root: Path) -> list[dict[str, Any]]:
    """All records of a root, in the order of their files."""
    records: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*/seed*/*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    return records


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _largest_gap(ours: Any, theirs: Any) -> float:
    """Largest absolute difference of the numbers of two JSON values; infinite when their shapes differ."""
    if isinstance(ours, dict) and isinstance(theirs, dict):
        if set(ours) != set(theirs):
            return math.inf
        return max((_largest_gap(ours[key], theirs[key]) for key in ours if key not in _UNTIMED), default=0.0)
    if isinstance(ours, list) and isinstance(theirs, list):
        if len(ours) != len(theirs):
            return math.inf
        return max((_largest_gap(a, b) for a, b in zip(ours, theirs)), default=0.0)
    if isinstance(ours, bool) or isinstance(theirs, bool) or ours is None or theirs is None or isinstance(ours, str):
        return 0.0 if ours == theirs else math.inf
    if isinstance(ours, (int, float)) and isinstance(theirs, (int, float)):
        return abs(float(ours) - float(theirs))
    return math.inf


def read_root(root: Path, settings: Settings, determinism_root: Path | None = None) -> dict[str, Any]:
    """Apply the fixed rules to the records of a root.

    The rules need the base and the background level: the floor of a template
    for a cell is its mean recovery by the learned dictionary of that cell in the
    background level. A rule without its records has the status ``not
    evaluated``; a rule with a part of its records has the status ``incomplete``.

    Parameters
    ----------
    root : Path
        A root with records.
    settings : Settings
        The setting of the root (its seeds and its number of templates).
    determinism_root : Path | None
        A second root with a repeat of some runs, for gate 6.

    Returns
    -------
    dict[str, Any]
        ``summary`` (one row per record), ``floors``, ``gates`` and ``predictions``.
    """
    records = load_records(root)
    index = {(r["level"], r["seed"], r["cell"], r["kind"]): r for r in records}
    seeds = list(settings.seeds)
    n_templates = settings.generator.n_templates

    def values(level: str, cell: str, kind: str) -> list[dict[str, Any]]:
        return [index[key] for seed in seeds if (key := (level, seed, cell, kind)) in index]

    floors: dict[str, list[float] | None] = {}
    for cell in CELL_NAMES:
        found = [record["waveform_recovery"] for record in values("background", cell, "learned")]
        floors[cell] = [sum(column) / len(column) for column in zip(*found)] if len(found) == len(seeds) else None

    def recovered(record: dict[str, Any]) -> int | None:
        floor = floors.get(record["cell"])
        if floor is None:
            return None
        pairs = zip(record["waveform_recovery"], floor)
        return sum(1 for value, low in pairs if value >= RECOVERED and value >= low + ABOVE_FLOOR)

    summary = []
    for record in records:
        found, fit = record["detection"], record["fit"] or {}
        summary.append(
            {
                "level": record["level"],
                "seed": record["seed"],
                "cell": record["cell"],
                "kind": record["kind"],
                "waveform_recovery": record["waveform_recovery_mean"],
                "template_recovery": record["template_recovery_mean"],
                "decoy_recovery": _mean(record["decoy_recovery"]),
                "recovered": recovered(record),
                "recall": found["recall"],
                "precision": found["precision"],
                "f1": found["f1"],
                "claims": found["n_claims"],
                "passes": fit.get("passes"),
                "steps": fit.get("steps"),
                "seconds": fit.get("seconds"),
            }
        )

    def status(parts: Sequence[bool | None]) -> str:
        if not parts or all(part is None for part in parts):
            return "not evaluated"
        if any(part is None for part in parts):
            return "incomplete"
        return "pass" if all(parts) else "fail"

    gates: dict[str, Any] = {}
    oracle: dict[str, float | None] = {}
    for cell in CELL_NAMES:
        recalls = [r["detection"]["recall"] for r in values("base", cell, "oracle")]
        oracle[cell] = _mean(recalls) if len(recalls) == len(seeds) else None
    gates["4_signal_level"] = {
        "status": status([None if value is None else value >= ORACLE_RECALL for value in oracle.values()]),
        "rule": f"mean oracle recall in base at least {ORACLE_RECALL} for every cell",
        "oracle_recall": oracle,
    }
    need_seeds = math.ceil(0.8 * len(seeds))
    kept: dict[str, int | None] = {}
    for cell in CELL_NAMES:
        counts = [recovered(r) for r in values("base", cell, "fixed_point")]
        complete = len(counts) == len(seeds) and all(count is not None for count in counts)
        kept[cell] = sum(1 for count in counts if count is not None and count >= n_templates - 1) if complete else None
    gates["5_fixed_point"] = {
        "status": status([None if value is None else value >= need_seeds for value in kept.values()]),
        "rule": f"from the planted dictionary, at least {n_templates - 1} of {n_templates} templates stay recovered "
        f"in at least {need_seeds} of {len(seeds)} seeds",
        "seeds_that_keep_the_templates": kept,
    }
    counts = [recovered(r) for cell in CELL_NAMES for r in values("base", cell, "learned")]
    complete = len(counts) == len(CELL_NAMES) * len(seeds) and all(count is not None for count in counts)
    mean_recovered = sum(count for count in counts if count is not None) / len(counts) if complete else None
    gates["range"] = {
        "status": status([None if mean_recovered is None else mean_recovered >= n_templates / 2]),
        "rule": f"mean recovered templates in base from the common start at least {n_templates / 2}; "
        "below it the signal level rises one step, once",
        "mean_recovered_templates": mean_recovered,
    }
    if determinism_root is None:
        gates["6_determinism"] = {"status": "not evaluated", "rule": "needs a second root with repeated runs"}
    else:
        repeated = {(r["level"], r["seed"], r["cell"], r["kind"]): r for r in load_records(determinism_root)}
        shared = sorted(set(index) & set(repeated))
        gap = max((_largest_gap(index[key], repeated[key]) for key in shared), default=math.inf)
        gates["6_determinism"] = {
            "status": status([gap <= DETERMINISM_TOLERANCE] if shared else []),
            "rule": f"every measure of a repeated run differs by at most {DETERMINISM_TOLERANCE}",
            "compared_records": len(shared),
            "largest_difference": None if math.isinf(gap) else gap,
            "records_of_another_shape": bool(shared) and math.isinf(gap),
        }

    predictions: dict[str, Any] = {}
    alternating, gradient = values("base", "S1-L4", "learned"), values("base", "S1-L5", "learned")
    if len(alternating) == len(gradient) == len(seeds):
        differences = [a["waveform_recovery_mean"] - g["waveform_recovery_mean"] for a, g in zip(alternating, gradient)]
        mean_difference = sum(differences) / len(differences)
        predictions["P5"] = {
            "status": "holds" if abs(mean_difference) < EQUIVALENCE_MARGIN else "does not hold",
            "rule": f"absolute mean difference of the waveform recovery of S1-L4 and S1-L5 in base below "
            f"{EQUIVALENCE_MARGIN}",
            "differences_by_seed": differences,
            "mean_difference": mean_difference,
        }
    else:
        predictions["P5"] = {"status": "not evaluated"}
    learned_background = [values("background", cell, "learned") for cell in CELL_NAMES]
    if all(len(found) == len(seeds) for found in learned_background):
        largest = max(max(record["waveform_recovery"]) for found in learned_background for record in found)
        predictions["P7"] = {
            "status": "holds" if largest < RECOVERED else "does not hold",
            "rule": f"no learned cell reaches a recovery of {RECOVERED} for a template in background",
            "largest_recovery": largest,
        }
    else:
        predictions["P7"] = {"status": "not evaluated"}
    return {"summary": summary, "floors": floors, "gates": gates, "predictions": predictions}


def render(report: dict[str, Any]) -> str:
    """The text view of a report of ``read_root``: gates, predictions, and the means of the base level by cell."""
    lines = ["Gates"]
    for name, gate in report["gates"].items():
        lines.append(f"  {name}: {gate['status']} ({gate['rule']})")
    lines.append("Predictions")
    for name, prediction in report["predictions"].items():
        lines.append(f"  {name}: {prediction['status']}" + (f" ({prediction['rule']})" if "rule" in prediction else ""))
    lines.append("Base level, mean over seeds: waveform recovery, recovered templates, recall, precision")
    rows = [row for row in report["summary"] if row["level"] == "base"]

    def cell_mean(cell: str, kind: str, column: str) -> str:
        found = [row[column] for row in rows if row["cell"] == cell and row["kind"] == kind and row[column] is not None]
        return "   n/a" if not found else f"{sum(found) / len(found):6.3f}"

    for cell in CELL_NAMES:
        for kind in KINDS:
            if any(row["cell"] == cell and row["kind"] == kind for row in rows):
                columns = ("waveform_recovery", "recovered", "recall", "precision")
                values = " ".join(cell_mean(cell, kind, column) for column in columns)
                lines.append(f"  {cell:7s} {kind:11s} {values}")
    return "\n".join(lines) + "\n"

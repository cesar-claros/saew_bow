"""Synthetic benchmark of stage 2; execute on the HPC.

Protocol: docs/Synthetic Benchmark Protocol.md. A root folder holds one signal
level. The three commands share it.

1. ``calibrate`` sets the budget of the gradient cells by the convergence rule,
   on the base level with seed 0, and writes ``budget.json``.
2. ``run`` fits the nine cells for the given levels and seeds and writes one
   file per level, seed and cell under ``runs/``. A cell with a file is not
   fitted again. ``--fixed-point`` adds the fit from the planted dictionary
   (gate 5; the protocol asks for it on the base level).
3. ``read`` applies the fixed rules and writes ``report.json`` and
   ``summary.csv``. ``--determinism-root`` names a second root with a repeat of
   some runs (gate 6).

Order of step 2 of the protocol, from the repository root with ``PYTHONPATH=src``::

    python scripts/run_synthetic.py calibrate --root artifacts/synthetic-001
    python scripts/run_synthetic.py run --root artifacts/synthetic-001 --levels base --fixed-point
    python scripts/run_synthetic.py run --root artifacts/synthetic-001 --levels background
    python scripts/run_synthetic.py run --root artifacts/synthetic-001-repeat --levels base --seeds 0 --fixed-point --budget-from artifacts/synthetic-001
    python scripts/run_synthetic.py read --root artifacts/synthetic-001 --determinism-root artifacts/synthetic-001-repeat

``--smoke`` makes a new root with a small setting (4 channels, 64 Hz, 2 seeds).
It checks the wiring in about a minute; its results decide nothing.

The levels of this step are base, background, polarity, more_atoms, delay_4,
delay_8, delay_16, few_10, few_25 and few_50. The levels with artifacts, bursts
and event pairs are not built yet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")  # the retained trainer draws one progress bar per pass


def _settings(level: float | None, root: Path, smoke: bool = False) -> Any:
    """The setting of the root: the stored one, or a new one (the protocol setting, or the small smoke setting)."""
    from eeg_compare.synthetic.benchmark import Settings, check_settings, settings_from_json, smoke_settings

    stored = root / "settings.json"
    if stored.exists():
        settings = settings_from_json(json.loads(stored.read_text(encoding="utf-8")))
        if level is not None and float(level) != settings.generator.level:
            raise ValueError(
                f"{root} holds the signal level {settings.generator.level}; a root holds one level, so use a new root"
            )
        return settings
    settings = smoke_settings() if smoke else Settings()
    if level is not None:
        settings = replace(settings, generator=replace(settings.generator, level=float(level)))
    check_settings(root, settings)
    return settings


def _budget(root: Path, steps: int | None, source: Path | None) -> int:
    """The steps of the gradient cells: from the argument, from another root, or from ``budget.json`` of the root."""
    path = root / "budget.json"
    if source is not None:
        steps = int(json.loads((source / "budget.json").read_text(encoding="utf-8"))["steps"])
    if path.exists():
        saved = int(json.loads(path.read_text(encoding="utf-8"))["steps"])
        if steps is not None and steps != saved:
            raise ValueError(f"{root} holds the budget {saved}; a root holds one budget, so use a new root")
        return saved
    if steps is None:
        raise ValueError(f"{root} has no budget.json; run 'calibrate' first, or give --steps or --budget-from")
    path.write_text(json.dumps({"steps": steps, "source": "argument" if source is None else str(source)}) + "\n")
    return steps


def _prepare() -> None:
    """Quiet logs of the retained trainer, and repeatable GPU convolutions."""
    import torch
    from loguru import logger

    logger.remove()
    logger.add(lambda message: sys.stderr.write(message), level="WARNING")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calibrate(args: argparse.Namespace) -> int:
    """Write ``budget.json`` of a root by the convergence rule."""
    from eeg_compare.synthetic.benchmark import calibrate_budget

    settings = _settings(args.level, args.root, args.smoke)
    path = args.root / "budget.json"
    if path.exists():
        raise ValueError(f"{path} exists; a root holds one budget")
    result = calibrate_budget(settings, args.device)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"Budget of the gradient cells: {result['steps']} steps\n")
    for name, entry in result["cells"].items():
        changes = ", ".join(f"{steps}: {value:.4f}" for steps, value in entry["loss_change"].items())
        sys.stdout.write(
            f"  {name}: loss change by checkpoint {changes}; converged at {entry['converged_at']}; "
            f"{entry['seconds']:.0f} s\n"
        )
    return 0


def run(args: argparse.Namespace) -> int:
    """Fit and measure the given levels and seeds."""
    from eeg_compare.cells.api import CELL_NAMES
    from eeg_compare.synthetic.benchmark import levels, run_level

    settings = _settings(args.level, args.root, args.smoke)
    steps = _budget(args.root, args.steps, args.budget_from)
    known = levels(settings)
    unknown = [name for name in args.levels if name not in known] + [c for c in args.cells or [] if c not in CELL_NAMES]
    if unknown:
        raise ValueError(f"unknown levels or cells {unknown}; the levels are {sorted(known)}")
    seeds = args.seeds if args.seeds is not None else list(settings.seeds)
    for name in args.levels:
        for seed in seeds:
            began = time.perf_counter()
            output = args.root / "runs" / name / f"seed{seed}"
            records = run_level(
                known[name], seed, settings, steps, args.device, args.cells or CELL_NAMES, args.fixed_point, output
            )
            learned = [r["waveform_recovery_mean"] for r in records if r["kind"] == "learned"]
            sys.stdout.write(
                f"{name} seed {seed}: {len(records)} records in {time.perf_counter() - began:.0f} s; "
                f"mean waveform recovery of the learned cells {sum(learned) / max(len(learned), 1):.3f}\n"
            )
            sys.stdout.flush()
    return 0


def read(args: argparse.Namespace) -> int:
    """Write the report of a root; return 1 when a gate fails."""
    import polars as pl

    from eeg_compare.synthetic.benchmark import read_root, render

    if not (args.root / "settings.json").exists():
        raise ValueError(f"{args.root} is not a root of the benchmark")
    settings = _settings(None, args.root)
    if args.determinism_root is not None:
        other = json.loads((args.determinism_root / "settings.json").read_text(encoding="utf-8"))
        if other != settings.as_json():
            raise ValueError("the two roots hold different settings")
    report = read_root(args.root, settings, args.determinism_root)
    pl.DataFrame(report["summary"], infer_schema_length=None).write_csv(args.root / "summary.csv")
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(render(report))
    return 1 if any(gate["status"] == "fail" for gate in report["gates"].values()) else 0


def main(argv: list[str] | None = None) -> int:
    """Run one command; return 0 on success, 1 when a gate fails, 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("calibrate", calibrate), ("run", run), ("read", read)):
        command = commands.add_parser(name)
        command.set_defaults(function=function)
        command.add_argument("--root", type=Path, required=True, help="Folder of one signal level")
        if name != "read":
            command.add_argument("--level", type=float, help="Signal level of a new root (default 8)")
            command.add_argument("--device", default="auto", help="cpu, cuda, cuda:N or auto")
            command.add_argument("--smoke", action="store_true", help="New root with the small setting: a wiring check")
    runner = commands.choices["run"]
    runner.add_argument("--levels", nargs="+", required=True, help="Names of the levels")
    runner.add_argument("--seeds", nargs="+", type=int, help="Seeds; default: the five seeds of the setting")
    runner.add_argument("--cells", nargs="+", help="Cells; default: all nine")
    runner.add_argument("--fixed-point", action="store_true", help="Also fit from the planted dictionary")
    runner.add_argument("--steps", type=int, help="Steps of the gradient cells, in place of budget.json")
    runner.add_argument("--budget-from", type=Path, help="Root whose budget.json gives the steps")
    commands.choices["read"].add_argument("--determinism-root", type=Path, help="Root with repeated runs")
    args = parser.parse_args(argv)
    try:
        _prepare()
        return int(args.function(args))
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Synthetic benchmark failed: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

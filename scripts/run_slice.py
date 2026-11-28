"""Run or read steps 2 and 3 of the slice (docs/Slice Protocol.md, section 14); execute on the HPC.

Step 2 (``run``) takes the frozen arms and the controls through the complete
path: signals, tokens, features, two heads and stored out-of-fold predictions.
Step 3 (``levels``) adds the learned arms and the levels. Both read training
recordings only. All commands run from the repository root with ``PYTHONPATH=src``.

Step 2, into a new directory::

    python scripts/run_slice.py run --root /path/to/corpus --manifest artifacts/tuev-manifest-clip50-001 \
        --plans artifacts/tuev-exposure-001 --scales artifacts/tuev-scale-audit-001.json \
        --cache-dir cache/tuev --output artifacts/slice-step2-001

Step 3 needs a second plan with the larger quota (``scripts/make_exposure.py --minutes 10``). It is
stored fold by fold. The same command continues a run that stopped::

    python scripts/run_slice.py levels --root /path/to/corpus --manifest artifacts/tuev-manifest-clip50-001 \
        --plans artifacts/tuev-exposure-001 --wide-plans artifacts/tuev-exposure-10min-001 \
        --scales artifacts/tuev-scale-audit-001.json --cache-dir cache/tuev --output artifacts/slice-step3-001

``--smoke`` checks the wiring of either step with small settings; its scores are not results.

Read a stored run again (NumPy and Polars only)::

    python scripts/run_slice.py read --run artifacts/slice-step3-001
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _run(args: argparse.Namespace) -> None:
    from eeg_compare.data.dataset import ReaderOptions, TUEVReader
    from eeg_compare.data.manifest import load_manifest
    from eeg_compare.data.provenance import file_hash
    from eeg_compare.slice.exposure import load_plans
    from eeg_compare.slice.levels import run_levels
    from eeg_compare.slice.runner import Settings, run, smoke_settings
    from eeg_compare.slice.scaling import read_scales
    from eeg_compare.slice.signals import study_clip
    from eeg_compare.slice.study import load_study

    levels = args.command == "levels"
    inside = [args.manifest, args.plans, *([args.wide_plans] if levels else [])]
    if any(args.output.resolve().is_relative_to(path.resolve()) for path in inside):
        raise ValueError("the run must lie outside the manifest and the plans")
    if not levels and args.output.exists():
        raise ValueError(f"{args.output} exists; a run is never overwritten")
    manifest = load_manifest(args.manifest)
    preprocessing = manifest.config.preprocessing
    study_clip(preprocessing)
    plan, plan_config = load_plans(args.plans, manifest)
    wide_plan, wide_config = load_plans(args.wide_plans, manifest) if levels else (None, None)
    scales = read_scales(args.scales, preprocessing.channels, [preprocessing.highpass_hz, preprocessing.lowpass_hz])
    reader = TUEVReader(args.root, manifest, ReaderOptions(cache_dir=args.cache_dir))
    settings = replace(smoke_settings(args.device) if args.smoke else Settings(device=args.device), n_jobs=args.jobs)
    _say("Reading the signals of the training recordings")
    study = load_study(manifest, plan, reader, wide_plan)
    _say(
        f"Read {study.recordings_read} recordings: {study.events.height} events, their controls and "
        f"{study.intervals.height} plan intervals"
    )
    inputs = {
        "manifest": args.manifest.name,
        "manifest_config_hash": manifest.metadata["config_hash"],
        "plans": args.plans.name,
        "plan_hash": file_hash(args.plans / "plan.parquet"),
        "plan_config": asdict(plan_config),
        "scales": args.scales.name,
        "scales_sha256": file_hash(args.scales),
        "smoke": bool(args.smoke),
    }
    if levels:
        assert wide_config is not None
        inputs["wide_plans"] = args.wide_plans.name
        inputs["wide_plan_hash"] = file_hash(args.wide_plans / "plan.parquet")
        inputs["wide_plan_config"] = asdict(wide_config)
        run_levels(study, manifest, scales, settings, args.output, inputs, _say)
    else:
        run(study, manifest, scales, settings, args.output, inputs)
    _say(f"Wrote {args.output}")


def _read(root: Path, draws: int, seed: int, output: Path | None) -> None:
    from eeg_compare.data.provenance import write_json
    from eeg_compare.slice.decide import decide, render_decisions
    from eeg_compare.slice.report import load_run, read_run, render

    data = load_run(root)
    summary = read_run(root, draws, seed, data)
    decisions = decide(data, draws, seed)
    if output is not None:
        write_json(output, {**summary, "decisions": decisions})
    sys.stdout.write(render(summary))
    if decisions is not None:
        sys.stdout.write("\n" + render_decisions(decisions))


def main(argv: list[str] | None = None) -> int:
    """Run or read a step of the slice; return 0 on success and 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Run step 2 into a new directory, then print its report")
    levels = commands.add_parser("levels", help="Run or continue step 3, then print its report")
    for command in (runner, levels):
        command.add_argument("--root", type=Path, required=True, help="Folder that holds the official directories")
        command.add_argument("--manifest", type=Path, required=True, help="The manifest of the study")
        command.add_argument("--plans", type=Path, required=True, help="Its exposure plans (scripts/make_exposure.py)")
        command.add_argument("--scales", type=Path, required=True, help="The report of scripts/audit_scales.py")
        command.add_argument("--output", type=Path, required=True, help="Directory of the run")
        command.add_argument("--cache-dir", type=Path, default=None, help="Disk cache of the preprocessed recordings")
        command.add_argument("--device", default="auto", help="cpu, cuda, cuda:N or auto")
        command.add_argument("--jobs", type=int, default=-1, help="Processes of the random forest; -1 uses all cores")
        command.add_argument("--smoke", action="store_true", help="Small settings for a check of the wiring")
    levels.add_argument("--wide-plans", type=Path, required=True, help="The plans with the larger quota (10 minutes)")
    reader = commands.add_parser("read", help="Print the report of a stored run")
    reader.add_argument("--run", type=Path, required=True, help="Directory of a run")
    for command in (runner, levels, reader):
        command.add_argument("--draws", type=int, default=2000, help="Bootstrap draws of the report")
        command.add_argument("--seed", type=int, default=0, help="Seed of the bootstrap draws")
    reader.add_argument("--output", type=Path, default=None, help="New JSON file for the summary")
    args = parser.parse_args(argv)
    try:
        if args.command == "read":
            _read(args.run, args.draws, args.seed, args.output)
        else:
            _run(args)
            _read(args.output, args.draws, args.seed, None)
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Slice failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

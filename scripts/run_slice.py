"""Run or read step 2 of the slice (docs/Slice Protocol.md, section 14); execute on the HPC.

Step 2 takes the frozen arms and the controls through the complete path:
signals, tokens, features, two heads and stored out-of-fold predictions. It
reads training recordings only.

Run it, from the repository root with ``PYTHONPATH=src``. The cache directory is
optional; with it a second run does not filter the recordings again::

    python scripts/run_slice.py run --root /path/to/corpus --manifest artifacts/tuev-manifest-clip50-001 \
        --plans artifacts/tuev-exposure-001 --scales artifacts/tuev-scale-audit-001.json \
        --output artifacts/slice-step2-001 --cache-dir cache/tuev-clip50-001

Check the wiring first with small settings (its scores are not results)::

    python scripts/run_slice.py run --smoke ... --output artifacts/slice-step2-smoke

Read a stored run again (NumPy and Polars only)::

    python scripts/run_slice.py read --run artifacts/slice-step2-001
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path


def _run(args: argparse.Namespace) -> None:
    from eeg_compare.data.dataset import ReaderOptions, TUEVReader
    from eeg_compare.data.manifest import load_manifest
    from eeg_compare.data.provenance import file_hash
    from eeg_compare.slice.exposure import load_plans
    from eeg_compare.slice.runner import Settings, run, smoke_settings
    from eeg_compare.slice.scaling import read_scales
    from eeg_compare.slice.signals import study_clip
    from eeg_compare.slice.study import load_study

    for name in ("manifest", "plans"):
        if args.output.resolve().is_relative_to(getattr(args, name).resolve()):
            raise ValueError("the run must lie outside the manifest and the plans")
    if args.output.exists():
        raise ValueError(f"{args.output} exists; a run is never overwritten")
    manifest = load_manifest(args.manifest)
    preprocessing = manifest.config.preprocessing
    study_clip(preprocessing)
    plan, plan_config = load_plans(args.plans, manifest)
    scales = read_scales(args.scales, preprocessing.channels, [preprocessing.highpass_hz, preprocessing.lowpass_hz])
    reader = TUEVReader(args.root, manifest, ReaderOptions(cache_dir=args.cache_dir))
    settings = replace(smoke_settings(args.device) if args.smoke else Settings(device=args.device), n_jobs=args.jobs)
    sys.stdout.write("Reading the signals of the training recordings\n")
    sys.stdout.flush()
    study = load_study(manifest, plan, reader)
    sys.stdout.write(
        f"Read {study.recordings_read} recordings: {study.events.height} events, their controls and "
        f"{study.intervals.height} plan intervals\n"
    )
    sys.stdout.flush()
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
    run(study, manifest, scales, settings, args.output, inputs)
    sys.stdout.write(f"Wrote {args.output}\n")


def _read(root: Path, draws: int, seed: int, output: Path | None) -> None:
    from eeg_compare.data.provenance import write_json
    from eeg_compare.slice.report import read_run, render

    summary = read_run(root, draws, seed)
    if output is not None:
        write_json(output, summary)
    sys.stdout.write(render(summary))


def main(argv: list[str] | None = None) -> int:
    """Run or read step 2; return 0 on success and 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Run step 2 into a new directory, then print its report")
    runner.add_argument("--root", type=Path, required=True, help="Folder that holds the official directories")
    runner.add_argument("--manifest", type=Path, required=True, help="The manifest of the study")
    runner.add_argument("--plans", type=Path, required=True, help="Its exposure plans (scripts/make_exposure.py)")
    runner.add_argument("--scales", type=Path, required=True, help="The report of scripts/audit_scales.py")
    runner.add_argument("--output", type=Path, required=True, help="New directory of the run")
    runner.add_argument("--cache-dir", type=Path, default=None, help="Disk cache of the preprocessed recordings")
    runner.add_argument("--device", default="auto", help="cpu, cuda, cuda:N or auto")
    runner.add_argument("--jobs", type=int, default=-1, help="Processes of the random forest; -1 uses all cores")
    runner.add_argument("--smoke", action="store_true", help="Small settings for a check of the wiring")
    reader = commands.add_parser("read", help="Print the report of a stored run")
    reader.add_argument("--run", type=Path, required=True, help="Directory of a run")
    for command in (runner, reader):
        command.add_argument("--draws", type=int, default=2000, help="Bootstrap draws of the report")
        command.add_argument("--seed", type=int, default=0, help="Seed of the bootstrap draws")
    reader.add_argument("--output", type=Path, default=None, help="New JSON file for the summary")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            _run(args)
            _read(args.output, args.draws, args.seed, None)
        else:
            _read(args.run, args.draws, args.seed, args.output)
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Slice step 2 failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

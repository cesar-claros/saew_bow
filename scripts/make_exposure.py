"""Build or verify the exposure plans of a TUEV manifest; execute on the HPC.

An exposure plan is the stored list of training units of every fold, with a
quota per patient (docs/Slice Protocol.md, section 3). It reads the tables of a
manifest only. It reads no signal and no evaluation recording.

Build the plans of the study, from the repository root with ``PYTHONPATH=src``::

    python scripts/make_exposure.py --manifest artifacts/tuev-manifest-clip50-001 --output artifacts/tuev-exposure-001

Verify stored plans against the manifest and against a new build (gate 4 of the
slice protocol)::

    python scripts/make_exposure.py --manifest artifacts/tuev-manifest-clip50-001 --check artifacts/tuev-exposure-001

The second levels of the protocol are other plans: ``--minutes 10`` and ``--events 50``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def render(summary: list[dict[str, Any]]) -> str:
    """The counts of a plan, one line per fold and source, then one line per source for all folds together."""
    lines = ["fold  source              units  patients  effective  largest   hours"]
    for row in summary:
        lines.append(
            f"{row['fold']:>4}  {row['source']:<18} {row['units']:>6} {row['patients']:>9} "
            f"{row['effective_patients']:>10.1f} {row['largest_patient']:>8} {row['hours']:>7.2f}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write or verify the plans; return 0 on success and 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="A completed manifest directory")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path, help="New directory of the plans")
    target.add_argument("--check", type=Path, help="Existing directory of plans to verify")
    parser.add_argument("--minutes", type=float, default=5.0, help="Signal per patient of the whole-recording source")
    parser.add_argument("--events", type=int, default=20, help="Events per patient and class of the context source")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the ranking")
    args = parser.parse_args(argv)
    try:
        from eeg_compare.data.manifest import load_manifest
        from eeg_compare.slice.exposure import ExposureConfig, check_plans, load_plans, pooled_counts, write_plans

        manifest = load_manifest(args.manifest)
        if args.check is not None:
            plan, config = load_plans(args.check, manifest)
            sfreq = manifest.config.preprocessing.target_sfreq
            summary = [*check_plans(plan, manifest, config), *pooled_counts(plan, sfreq)]
            sys.stdout.write(f"Verified {args.check}: the stored plan equals a new build ({config})\n")
        else:
            if args.output.resolve().is_relative_to(args.manifest.resolve()):
                raise ValueError("the plans must lie outside the manifest directory")
            config = ExposureConfig(
                minutes_per_patient=args.minutes, events_per_patient_class=args.events, seed=args.seed
            )
            metadata = write_plans(manifest, config, args.output)
            summary = [*metadata["summary"], *metadata["pooled"]]
            sys.stdout.write(f"Wrote {args.output} ({config})\n")
    except (OSError, ValueError, KeyError) as error:
        sys.stderr.write(f"Exposure plan failed: {error}\n")
        return 2
    sys.stdout.write(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""HPC commands for manifest construction and a recorded signal smoke check."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from loguru import logger

from eeg_compare.data.config import PipelineConfig
from eeg_compare.data.manifest import CorpusLayout, build_manifest
from eeg_compare.data.provenance import file_hash, write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TUEV data preparation; execute on the HPC.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="Audit headers/annotations and write a new manifest directory")
    build.add_argument("--root", type=Path, required=True, help="Corpus root containing official split directories")
    build.add_argument("--version", required=True, help="Explicit corpus release/version identifier")
    build.add_argument("--identities", type=Path, required=True, help="Audited recording-to-patient CSV")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True, help="New directory, never an existing run")
    build.add_argument("--train-subdir", default="train")
    build.add_argument("--eval-subdir", default="eval")
    check = subcommands.add_parser("check", help="Read selected signal examples and save an HPC smoke-check report")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True, help="New report JSON, outside the immutable manifest")
    check.add_argument("--split", choices=("train", "validation", "eval"), default="train")
    check.add_argument("--fold", type=int, default=0)
    check.add_argument("--limit", type=int, default=8)
    check.add_argument("--cache-dir", type=Path)
    check.add_argument("--allow-eval", action="store_true")
    check.add_argument("--kind", choices=("event", "displaced"), default="event")
    return parser


def _check(args: argparse.Namespace) -> None:
    from eeg_compare.data.dataset import Partition, ReaderOptions, TUEVDataset

    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.output.exists() or args.output.resolve().is_relative_to(args.manifest.resolve()):
        raise ValueError("check output must be a new file outside the completed manifest directory")
    partition = Partition(
        cast(Literal["train", "validation", "eval"], args.split),
        args.fold,
        args.allow_eval,
        cast(Literal["event", "displaced"], args.kind),
    )
    dataset = TUEVDataset(args.root, args.manifest, partition, ReaderOptions(cache_dir=args.cache_dir))
    count = min(args.limit, len(dataset))
    indices = [0] if count == 1 else [index * (len(dataset) - 1) // (count - 1) for index in range(count)]
    checked = []
    for index in indices:
        row = dataset.example_metadata(index)
        signal = dataset.reader.example(row)
        checked.append(
            {
                "example_id": row.example_id,
                "kind": row.kind,
                "shape": list(signal.signal.shape),
                "real_samples": int(signal.valid_samples.sum()),
                "valid_channels": int(signal.valid_channels.sum()),
                "onset_offset_samples": row.onset_offset_samples,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "manifest_sha256": file_hash(args.manifest / "metadata.json"),
            "split": args.split,
            "validation_fold": args.fold,
            "kind": args.kind,
            "checked_examples": checked,
            "scope": "Deterministic signal smoke check only; manual annotation/timing review and full corpus QA remain required.",
        },
    )
    logger.info("Read {} signal examples; saved the HPC smoke-check report", len(checked))


def main(argv: list[str] | None = None) -> int:
    """Run one explicit HPC action; return 2 for invalid inputs without overwriting artifacts."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            metadata = build_manifest(
                CorpusLayout(args.root, args.version, args.train_subdir, args.eval_subdir),
                args.identities,
                PipelineConfig.from_toml(args.config),
                args.output,
            )
            logger.info(
                "Wrote {} recordings and {} cluster assignments ({} training patients; fold sizes {})",
                metadata["recording_count"],
                metadata["cluster_count"],
                metadata["training_patient_count"],
                metadata["fold_sizes"],
            )
            merged = metadata["duplicate_content"]["groups"]
            if merged:
                logger.info(
                    "Merged {} groups of identical EDF files into {} clusters; read duplicate_content in metadata.json",
                    len(merged),
                    len({row["cluster_id"] for row in merged}),
                )
            for row in metadata["conflict_report"]:
                if row["conflicting_intervals"]:
                    logger.info(
                        "Conflicts, {} {}: {} of {} intervals ({} of {} rows)",
                        row["official_split"],
                        row["label"],
                        row["conflicting_intervals"],
                        row["all_intervals"],
                        row["conflicting_rows"],
                        row["all_rows"],
                    )
            if any(row["missing_classes"] for row in metadata["class_coverage"] if row["split"] != "eval"):
                logger.warning("Some development partitions lack classes; inspect class_coverage before benchmarking")
        else:
            _check(args)
    except (ValueError, TypeError, OSError, KeyError) as error:
        logger.error("TUEV preparation failed: {}", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

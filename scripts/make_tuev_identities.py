"""Write the TUEV identity CSV from the folder layout of the corpus; execute on the HPC.

The corpus README (``AAREADME.txt`` of v2.0.1) defines the layout. A training
file lies in the folder of its patient (``train/<patient>/...``). An evaluation
file lies in the folder of its session index (``eval/<index>/...``), and the
corpus gives no patient for it. The earlier inventory of this corpus read the
same folders and found 290 training patients and 80 evaluation sessions.

This is an explicit, separate step. The manifest build never reads an identity
from a path: it needs the CSV, and each row states its evidence in
``identity_source``. Read the printed summary before the build. The output holds
patient identifiers, so write it under the ignored ``data/`` directory.

Standard library only, so the container's interpreter runs it without the package.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
FIELDS = ("edf_path", "patient_id", "session_id", "identity_source")
SOURCES = {
    "train": "patient folder of the training set (corpus AAREADME.txt); scripts/make_tuev_identities.py",
    "eval": "session index folder of the evaluation set (corpus AAREADME.txt); scripts/make_tuev_identities.py",
}
REFERENCE_V2_0_1 = {"train": {"files": 359, "groups": 290}, "eval": {"files": 159, "groups": 80}}


def collect(
    root: Path, train_subdir: str = "train", eval_subdir: str = "eval"
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """Identity rows of every EDF below the two official directories, and a summary for review.

    The group of a file is the folder that holds it: the patient of a training
    file and the session index of an evaluation file. Paths in the rows are
    relative to ``root``, as the manifest build expects.

    Args:
        root: The folder that holds the two official directories (``.../v2.0.1/edf``).
        train_subdir: Name of the training directory below ``root``.
        eval_subdir: Name of the evaluation directory below ``root``.

    Returns:
        The rows (training files first, each split sorted by path), and per
        official split: the files, the groups,
        the files without one ``.rec`` companion, and the files whose name does
        not contain the name of their folder.

    Raises:
        ValueError: An official directory is missing or empty, an EDF lies
            directly in an official directory, or two different folders of one
            split have the same name.
    """
    rows: list[dict[str, str]] = []
    summary: dict[str, dict[str, int]] = {}
    for split, subdir in (("train", train_subdir), ("eval", eval_subdir)):
        directory = root / subdir
        if not directory.is_dir():
            raise ValueError(f"official {split} directory does not exist: {directory}")
        paths = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".edf")
        if not paths:
            raise ValueError(f"official {split} directory has no EDF files: {directory}")
        folders: dict[str, set[Path]] = {}
        without_rec = unnamed = 0
        for path in paths:
            if path.parent == directory:
                raise ValueError(f"{path} lies directly in the {split} directory, so it has no group folder")
            group = path.parent.name
            folders.setdefault(group, set()).add(path.parent)
            companions = [
                other
                for other in path.parent.iterdir()
                if other.is_file() and other.stem == path.stem and other.suffix.lower() == ".rec"
            ]
            without_rec += len(companions) != 1
            unnamed += group not in path.name
            rows.append(
                {
                    "edf_path": path.relative_to(root).as_posix(),
                    "patient_id": group if split == "train" else "",
                    "session_id": group if split == "eval" else "",
                    "identity_source": SOURCES[split],
                }
            )
        repeated = sorted(group for group, parents in folders.items() if len(parents) > 1)
        if repeated:
            raise ValueError(
                f"{split}: different folders have the same name, so the group is ambiguous: {repeated[:5]}"
            )
        summary[split] = {
            "files": len(paths),
            "groups": len(folders),
            "files_without_one_rec": without_rec,
            "files_whose_name_lacks_the_folder_name": unnamed,
        }
    return rows, summary


def main(argv: list[str] | None = None) -> int:
    """Write the CSV and print the summary; return 0 on success, otherwise 1.

    Args:
        argv: CLI arguments; defaults to the process arguments.

    Returns:
        Exit status. An existing output file is refused.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Folder that holds the official directories")
    parser.add_argument("--output", type=Path, required=True, help="New CSV, for example data/tuev_identities.csv")
    parser.add_argument("--train-subdir", default="train")
    parser.add_argument("--eval-subdir", default="eval")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        rows, summary = collect(args.root.resolve(), args.train_subdir, args.eval_subdir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError) as error:
        logger.error("Identity mapping failed: %s", error)
        return 1
    report = {
        "output": str(args.output),
        "rows": len(rows),
        "summary": summary,
        "reference_v2.0.1": REFERENCE_V2_0_1,
        "matches_reference_v2.0.1": all(
            summary[split][key] == value for split, counts in REFERENCE_V2_0_1.items() for key, value in counts.items()
        ),
        "note": "Evaluation rows have a session index and no patient; build with eval_cluster = 'session'.",
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

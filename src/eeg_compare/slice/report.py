"""Read a stored run of the slice: gates, scores and intervals (slice protocol, sections 9, 10 and 12).

This module needs NumPy, Polars and orjson only, so a run that was copied from
the HPC can be read without Torch.

The score of an arm is the mean over its seeds, and over the seeds of the
forest for that head. An interval comes from one set of bootstrap draws over
patients that all arms share, so a difference of two arms is paired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl
from numpy.typing import NDArray

from eeg_compare.slice.evaluate import (
    bootstrap_multiplicities,
    confusion,
    example_weights,
    interval,
    patient_strata,
    recall_draws,
    summarize,
)

__all__ = ["CHANCE_MARGIN", "PREVIEW_PAIRS", "PROTOCOL_TOKENS", "read_run", "render"]

CHANCE_MARGIN = 0.03
NULL_ARM = "spectral-null"
KEYS = ["arm", "head", "readout", "seed", "head_seed"]
PREVIEW_PAIRS = (
    ("spectral", "spectra_bag"),
    ("spectral", "S1-frozen-data"),
    ("spectra_bag", "S1-frozen-data"),
    ("S1-frozen-data", "S1-frozen-gauss"),
    ("S1-frozen-data", "SF-frozen-data"),
)
# Tokens of a context in the primary geometry (protocol, section 7): L = P = 125, 1,250 samples, 22 channels.
PROTOCOL_TOKENS = {
    "single": {"full": 220, "event": 44, "context": 132, "displaced": 220},
    "full": {"full": 10, "event": 2, "context": 6, "displaced": 10},
}
_FULL_ARMS = ("SF-",)


def _gate_5(report: dict[str, Any]) -> dict[str, Any]:
    """The counts of gate 5 against the token numbers of the protocol, when the run has the primary geometry."""
    settings = report["settings"]
    primary = (
        settings["atom_len"] == 125
        and settings["pool_len"] == 125
        and report["context_samples"] == 1250
        and report["channels"] == 22
    )
    rows = []
    passed = True
    for row in report["gate_5"]:
        kind = "full" if row["arm"].startswith(_FULL_ARMS) else "single"
        expected = PROTOCOL_TOKENS[kind][row["readout"]] if primary else None
        sums = abs(row["sum_min"] - 1.0) < 1e-9 and abs(row["sum_max"] - 1.0) < 1e-9
        good = sums and (expected is None or expected == row["tokens_per_context"])
        passed = passed and good
        rows.append({**row, "expected_tokens": expected, "passes": good})
    return {"passes": passed and bool(rows), "primary_geometry": primary, "rows": rows}


def read_run(root: Path, n_draws: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The gates and the scores of a stored run.

    Parameters
    ----------
    root : Path
        Directory of a run (``predictions.parquet``, ``examples.parquet``, ``report.json``).
    n_draws : int
        Bootstrap draws.
    seed : int
        Seed of the draws.

    Returns
    -------
    dict
        ``gates``, ``scores`` (one row per arm, head and readout), ``pairs``
        (paired differences on the full readout), and ``draws`` with the number
        of draws that held no example of a class.

    Raises
    ------
    ValueError
        If a block of predictions does not hold every example once.
    """
    report = orjson.loads((root / "report.json").read_bytes())
    examples = pl.read_parquet(root / "examples.parquet").with_row_index("row")
    predictions = pl.read_parquet(root / "predictions.parquet", columns=[*KEYS, "example_id", "prediction"]).join(
        examples.select("example_id", "row"), on="example_id", how="left"
    )
    n_classes = len(report["labels"])
    count = examples.height
    targets = examples["target"].to_numpy().astype(np.int64)
    patients = np.asarray(examples["cluster_id"].to_list(), dtype=str)
    index, stratum = patient_strata(targets, patients, n_classes)
    multiplicities = bootstrap_multiplicities(stratum, n_draws, seed)
    weightings = {
        "row_weighted": examples["row_weight"].to_numpy().astype(np.float64),
        "patient_weighted": example_weights(targets, patients, n_classes),
    }

    groups: dict[tuple[str, str, str], dict[tuple[int, int], NDArray[np.int64]]] = {}
    for key, frame in predictions.group_by(KEYS, maintain_order=True):
        arm, head, readout, arm_seed, head_seed = key
        ordered = np.full(count, -1, dtype=np.int64)
        if frame["row"].null_count() == 0:
            ordered[frame["row"].to_numpy()] = frame["prediction"].to_numpy()
        if frame.height != count or (ordered < 0).any():
            raise ValueError(f"the predictions of {key} do not hold every example once")
        groups.setdefault((str(arm), str(head), str(readout)), {})[(int(arm_seed), int(head_seed))] = ordered

    scores = []
    draws: dict[tuple[str, str, str], NDArray[np.float64]] = {}
    empty = 0
    for (arm, head, readout), runs in groups.items():
        summaries = {combo: summarize(confusion(targets, found, n_classes)) for combo, found in runs.items()}
        per_class = [recall_draws(targets, found, index, multiplicities, n_classes) for found in runs.values()]
        empty += int(np.isnan(per_class[0]).any(axis=1).sum())
        balanced = np.mean([np.nanmean(values, axis=1) for values in per_class], axis=0)
        draws[(arm, head, readout)] = balanced
        by_seed: dict[int, list[float]] = {}
        for (arm_seed, _), summary in summaries.items():
            by_seed.setdefault(arm_seed, []).append(summary["balanced_accuracy"])
        seed_means = [float(np.mean(values)) for values in by_seed.values()]
        row: dict[str, Any] = {
            "arm": arm,
            "head": head,
            "readout": readout,
            "seeds": len(by_seed),
            "runs": len(runs),
            "balanced_accuracy": float(np.mean([s["balanced_accuracy"] for s in summaries.values()])),
            "interval": interval(balanced),
            "seed_min": min(seed_means),
            "seed_max": max(seed_means),
            "recall": np.mean([s["recall"] for s in summaries.values()], axis=0).tolist(),
            "recall_interval": [interval(np.mean([v[:, c] for v in per_class], axis=0)) for c in range(n_classes)],
            "macro_f1": float(np.mean([s["macro_f1"] for s in summaries.values()])),
            "kappa": float(np.mean([s["kappa"] for s in summaries.values()])),
            "accuracy": float(np.mean([s["accuracy"] for s in summaries.values()])),
            "confusion": np.mean([confusion(targets, found, n_classes) for found in runs.values()], axis=0).tolist(),
        }
        for name, weights in weightings.items():
            row[name] = float(
                np.mean(
                    [
                        summarize(confusion(targets, found, n_classes, weights))["balanced_accuracy"]
                        for found in runs.values()
                    ]
                )
            )
        scores.append(row)

    pairs = []
    for first, second in PREVIEW_PAIRS:
        for head in ("logistic", "forest"):
            a, b = draws.get((first, head, "full")), draws.get((second, head, "full"))
            if a is None or b is None:
                continue
            point = {(row["arm"], row["head"], row["readout"]): row["balanced_accuracy"] for row in scores}
            pairs.append(
                {
                    "first": first,
                    "second": second,
                    "head": head,
                    "difference": point[(first, head, "full")] - point[(second, head, "full")],
                    "interval": interval(a - b),
                }
            )

    null = [row for row in scores if row["arm"] == NULL_ARM]
    chance = 1.0 / n_classes
    gate_3: dict[str, Any] = {"passes": False, "chance": chance, "margin": CHANCE_MARGIN}
    if null:
        gate_3.update(
            {
                "balanced_accuracy": null[0]["balanced_accuracy"],
                "permutations": null[0]["seeds"],
                "smallest": null[0]["seed_min"],
                "largest": null[0]["seed_max"],
                "passes": abs(null[0]["balanced_accuracy"] - chance) <= CHANCE_MARGIN,
            }
        )
    gate_2 = {**report["gate_2"]}
    gate_2["passes"] = bool(gate_2["training_recordings_only"] and gate_2["fold_parts_disjoint"])
    gate_6 = {**report["gate_6"], "passes": bool(report["gate_6"]["equal_predictions"])}
    return {
        "labels": report["labels"],
        "settings": report["settings"],
        "examples": count,
        "patients": int(stratum.size),
        "gates": {"gate_2": gate_2, "gate_3": gate_3, "gate_5": _gate_5(report), "gate_6": gate_6},
        "clip_share": report["clip_share"],
        "logistic": report["logistic"],
        "total_seconds": report["total_seconds"],
        "peak_gpu_bytes": report["peak_gpu_bytes"],
        "scores": [row for row in scores if row["arm"] != NULL_ARM],
        "pairs": pairs,
        "draws": {"count": n_draws, "seed": seed, "draws_without_a_class": empty},
    }


def _line(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True)).rstrip()


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]
    return [_line(header, widths), *(_line(row, widths) for row in rows)]


def render(summary: dict[str, Any]) -> str:
    """The report of a run as text: the gates, the scores on the full readout, the recall by readout, the pairs."""
    gates = summary["gates"]
    lines = [f"Slice, step 2: {summary['examples']} examples of {summary['patients']} patients", "", "Gates"]
    two, three, five, six = gates["gate_2"], gates["gate_3"], gates["gate_5"], gates["gate_6"]
    lines.append(
        f"  2 no leak      {'PASS' if two['passes'] else 'FAIL'}  {two['recordings_read']} training recordings read, "
        "no evaluation recording, no patient in both parts of a fold"
    )
    if "balanced_accuracy" in three:
        lines.append(
            f"  3 null         {'PASS' if three['passes'] else 'FAIL'}  spectral with permuted labels reads "
            f"{three['balanced_accuracy']:.3f} (mean of {three['permutations']}; {three['smallest']:.3f} to "
            f"{three['largest']:.3f}); chance {three['chance']:.3f}, margin {three['margin']:.2f}"
        )
    else:
        lines.append("  3 null         FAIL  the run holds no null arm")
    note = "against the numbers of the protocol" if five["primary_geometry"] else "geometry differs from the protocol"
    lines.append(
        f"  5 counts       {'PASS' if five['passes'] else 'FAIL'}  {len(five['rows'])} arm and readout pairs, {note}"
    )
    lines.append(
        f"  6 determinism  {'PASS' if six['passes'] else 'FAIL'}  {six['blocks']} blocks of fold 0 computed twice, "
        f"largest probability difference {six['max_probability_difference']:.2e}"
    )
    share = max(row["share_at_clip_factor_below_one"] for row in summary["clip_share"])
    state = "approximate" if any(row["approximate"] for row in summary["clip_share"]) else "exact enough"
    lines += [
        "",
        f"Rule of the profile input: largest share of plan samples at the clip in a channel with a factor below 1 "
        f"is {share:.5%} ({state}; limit 0.1%)",
        f"Logistic passes: largest {summary['logistic']['largest_passes']} of {summary['logistic']['limit']}",
        f"Time: {summary['total_seconds'] / 60:.1f} min; peak GPU memory {summary['peak_gpu_bytes'] / 2**30:.2f} GB",
        f"Draws without an example of a class: {summary['draws']['draws_without_a_class']} "
        f"of {summary['draws']['count']} per arm at most",
        "",
        "Balanced accuracy on the full readout (mean over seeds; interval from the bootstrap over patients)",
    ]
    full = [row for row in summary["scores"] if row["readout"] == "full"]
    lines += _table(
        ["arm", "head", "seeds", "BA", "interval", "half", "seed range", "F1", "kappa", "rows", "patients"],
        [
            [
                row["arm"],
                row["head"],
                str(row["seeds"]),
                f"{row['balanced_accuracy']:.3f}",
                f"{row['interval']['low']:.3f} to {row['interval']['high']:.3f}",
                f"{row['interval']['half_width']:.3f}",
                f"{row['seed_min']:.3f} to {row['seed_max']:.3f}",
                f"{row['macro_f1']:.3f}",
                f"{row['kappa']:.3f}",
                f"{row['row_weighted']:.3f}",
                f"{row['patient_weighted']:.3f}",
            ]
            for row in full
        ],
    )
    lines += ["", "Recall by class and readout, logistic head"]
    lines += _table(
        ["arm", "readout", *summary["labels"], "BA"],
        [
            [
                row["arm"],
                row["readout"],
                *(f"{value:.2f}" for value in row["recall"]),
                f"{row['balanced_accuracy']:.3f}",
            ]
            for row in summary["scores"]
            if row["head"] == "logistic"
        ],
    )
    lines += ["", "Paired differences on the full readout (first minus second)"]
    lines += _table(
        ["first", "second", "head", "difference", "interval", "half"],
        [
            [
                row["first"],
                row["second"],
                row["head"],
                f"{row['difference']:+.3f}",
                f"{row['interval']['low']:+.3f} to {row['interval']['high']:+.3f}",
                f"{row['interval']['half_width']:.3f}",
            ]
            for row in summary["pairs"]
        ],
    )
    return "\n".join(lines) + "\n"

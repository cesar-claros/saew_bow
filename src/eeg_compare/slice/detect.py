"""The recording test: tell the context of an event from its own displaced control (protocol, section 19).

The six-class score of the slice comes mostly from the recording. This test
removes the recording. Every event has a control context of the same recording,
far from every mark. For one class at a time, a head learns to tell event
contexts from their controls on the training patients, and gives a score to
both contexts of every event of the validation patients.

The measure of a class is the share of pairs in which the event has the higher
score (a tie counts as a half), taken per patient and then averaged over the
patients that hold the class. Chance is 0.5, and no property of a patient or a
recording can move it, because both contexts of a pair come from one recording.

The first part of this module runs the test and needs Torch. ``read_detection``
and ``evaluate_detection`` need NumPy and Polars only.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl
from numpy.typing import NDArray

from eeg_compare.slice.evaluate import example_weights, interval
from eeg_compare.slice.report import _table

__all__ = [
    "EPILEPTIFORM",
    "FIXED_ARMS",
    "NULL_ARM",
    "NULL_MARGIN",
    "PAIR_ARMS",
    "PAIR_READOUTS",
    "READ_LIMIT",
    "detection_frame",
    "evaluate_detection",
    "pair_values",
    "read_detection",
    "render_detection",
    "run_detection",
]

PAIR_READOUTS = ("full", "event", "context")
PAIR_ARMS = ("spectral", "spectra_bag", "S1-L1", "S1-L2", "S1-frozen-data", "S1-frozen-gauss", "SF-frozen-data")
NULL_ARM = "spectral-null"
FIXED_ARMS = ("spectral", "S1-L2")
EPILEPTIFORM = ("spsw", "gped", "pled")
CONTROL_CLASS = "bckg"
READ_LIMIT = 0.60
NULL_MARGIN = 0.05
HEADS = ("logistic", "forest")
KEYS = ["arm", "seed", "head", "readout", "fold", "label", "example_id"]


def pair_values(event: NDArray[np.float64], control: NDArray[np.float64]) -> NDArray[np.float64]:
    """One value per pair: 1 if the event has the higher score, 0.5 for a tie, else 0."""
    event, control = np.asarray(event, dtype=np.float64), np.asarray(control, dtype=np.float64)
    return np.where(event > control, 1.0, np.where(event == control, 0.5, 0.0))


def _feature_sets(
    study: Any, prepared: Any, control_spectral: dict[str, Any], settings: Any, fold: int
) -> Iterator[Any]:
    """The features of every arm and seed of a fold: ``(arm, seed, {readout: (events, controls)})``."""
    import torch

    from eeg_compare.cells.api import Budget, Geometry, fit_cell, start_atoms
    from eeg_compare.cells.common import CellAtoms, unit_norm
    from eeg_compare.slice.features import cell_tokens, count_features
    from eeg_compare.slice.runner import _seed, _valid_start
    from eeg_compare.slice.spectra import fit_spectra_bag, spectra_tokens
    from eeg_compare.slice.study import fold_interval_rows

    readouts = {name: prepared.readouts[name] for name in PAIR_READOUTS}
    yield "spectral", 0, {name: (prepared.spectral[name], control_spectral[name]) for name in PAIR_READOUTS}

    def counts(tokens: tuple[Any, Any], n_atoms: int) -> dict[str, tuple[Any, Any]]:
        return {
            name: (
                count_features(tokens[0], readout.windows, n_atoms),
                count_features(tokens[1], readout.windows, n_atoms),
            )
            for name, readout in readouts.items()
        }

    n_atoms, atom_len, pool_len = settings.n_atoms, settings.atom_len, settings.pool_len
    rows = torch.from_numpy(fold_interval_rows(study, fold))
    training_spectra = prepared.interval_spectra[rows]
    for seed in settings.seeds:
        model = fit_spectra_bag(training_spectra, n_atoms, _seed("spectra", fold, seed), settings.kmeans_passes)
        tokens = (spectra_tokens(model, prepared.context_spectra), spectra_tokens(model, prepared.control_spectra))
        yield "spectra_bag", seed, counts(tokens, n_atoms)
    del training_spectra
    intervals = study.interval_signals[rows]
    geometry = Geometry(n_atoms, atom_len, pool_len, 1)
    budget = Budget(max_passes=settings.max_passes, tol=settings.tol)
    for seed in settings.seeds:
        start, _ = _valid_start(intervals, geometry, _seed("start", fold, seed), settings.start_attempts)
        gauss = torch.randn((n_atoms, atom_len), generator=torch.Generator().manual_seed(_seed("gauss", fold, seed)))
        dictionaries = {
            "S1-frozen-data": start_atoms("S1", start, geometry),
            "S1-frozen-gauss": CellAtoms("S1", waveform=unit_norm(gauss)),
            "SF-frozen-data": start_atoms("SF", start, geometry),
        }
        for name in ("S1-L1", "S1-L2"):
            fitted = fit_cell(name, intervals, dictionaries["S1-frozen-data"], geometry, budget, device=settings.device)
            assert fitted.atoms.waveform is not None
            dictionaries[name] = CellAtoms("S1", waveform=fitted.atoms.waveform.detach().cpu())
            del fitted
        for arm, atoms in dictionaries.items():
            tokens = (
                cell_tokens(study.contexts, atoms, "cosine", pool_len, settings.device),
                cell_tokens(study.controls, atoms, "cosine", pool_len, settings.device),
            )
            yield arm, seed, counts(tokens, n_atoms)


def detection_frame(
    study: Any, prepared: Any, control_spectral: dict[str, Any], settings: Any, fold: int
) -> pl.DataFrame:
    """The scores of both contexts of every validation event of a fold, for every arm, seed, head and readout.

    A head is fitted for one class at a time, on the events of that class of
    the training patients and on their controls. The null arm is the spectral
    control with the two labels of every training pair in a random order.

    Returns
    -------
    polars.DataFrame
        ``arm``, ``seed``, ``head``, ``readout``, ``fold``, ``label``,
        ``example_id``, ``score_event`` and ``score_control``.
    """
    from eeg_compare.slice.heads import forest_probabilities, logistic_probabilities, standardize
    from eeg_compare.slice.runner import LOGISTIC_PASSES, _seed
    from eeg_compare.slice.study import fold_parts

    train, validation = fold_parts(study, fold)
    targets = study.events["target"].to_numpy().astype(np.int64)
    patients = np.asarray(study.events["cluster_id"].to_list(), dtype=str)
    ids = study.events["example_id"].to_list()
    frames = []

    def add(arm: str, seed: int, head: str, readout: str, label: int, rows: NDArray[np.int64], scores: Any) -> None:
        count = rows.size
        frames.append(
            pl.DataFrame(
                {
                    "arm": [arm] * count,
                    "seed": [seed] * count,
                    "head": [head] * count,
                    "readout": [readout] * count,
                    "fold": [fold] * count,
                    "label": [study.labels[label]] * count,
                    "example_id": [ids[row] for row in rows.tolist()],
                    "score_event": np.asarray(scores[:count], dtype=np.float64),
                    "score_control": np.asarray(scores[count:], dtype=np.float64),
                },
                schema_overrides={"seed": pl.Int64, "fold": pl.Int64},
            )
        )

    for arm, seed, features in _feature_sets(study, prepared, control_spectral, settings, fold):
        for readout, (events, controls) in features.items():
            for label in range(study.n_classes):
                fit_rows, test_rows = train[targets[train] == label], validation[targets[validation] == label]
                if not fit_rows.size or not test_rows.size:
                    continue
                x_train, x_test = standardize(
                    np.vstack([events[fit_rows], controls[fit_rows]]),
                    np.vstack([events[test_rows], controls[test_rows]]),
                )
                y_train = np.repeat(np.array([1, 0], dtype=np.int64), fit_rows.size)
                groups = np.tile(patients[fit_rows], 2)
                weights = example_weights(y_train, groups, 2)
                found, _ = logistic_probabilities(x_train, y_train, weights, x_test, 2, max_iter=LOGISTIC_PASSES)
                add(arm, seed, "logistic", readout, label, test_rows, found[:, 1])
                found = forest_probabilities(
                    x_train, y_train, weights, x_test, 2, 0, trees=settings.forest_trees, n_jobs=settings.n_jobs
                )
                add(arm, seed, "forest", readout, label, test_rows, found[:, 1])
                if arm == "spectral" and readout == "full":
                    for number in range(settings.null_permutations):
                        swap = np.random.default_rng(_seed("pair-null", number, fold, label)).integers(
                            0, 2, fit_rows.size
                        )
                        permuted = np.concatenate([1 - swap, swap]).astype(np.int64)
                        null_weights = example_weights(permuted, groups, 2)
                        found, _ = logistic_probabilities(
                            x_train, permuted, null_weights, x_test, 2, max_iter=LOGISTIC_PASSES
                        )
                        add(NULL_ARM, number, "logistic", readout, label, test_rows, found[:, 1])
    return pl.concat(frames)


def run_detection(study: Any, settings: Any, output: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    """Run the recording test and store it in a new directory.

    The directory gets ``pairs.parquet`` (the scores of both contexts of every
    event), ``examples.parquet`` and ``report.json``. Fold 0 is computed twice,
    and the report says whether the two computations are equal.

    Raises
    ------
    ValueError
        If the directory exists.
    """
    import sklearn
    import torch

    from eeg_compare.data.provenance import write_json
    from eeg_compare.device import resolve_device
    from eeg_compare.slice.features import band_features
    from eeg_compare.slice.runner import prepare

    if output.exists():
        raise ValueError(f"{output} exists; a run is never overwritten")
    began = time.perf_counter()

    def ready() -> tuple[Any, dict[str, Any]]:
        prepared = prepare(study, settings)
        frame = int(round(study.sfreq))
        control = {
            name: band_features(
                study.controls, prepared.readouts[name].ranges, study.sfreq, nperseg=frame, device=settings.device
            )
            for name in PAIR_READOUTS
        }
        return prepared, control

    prepared, control = ready()
    frames, seconds = [], []
    for fold in range(study.n_folds):
        started = time.perf_counter()
        frames.append(detection_frame(study, prepared, control, settings, fold))
        seconds.append(time.perf_counter() - started)
    repeated = detection_frame(study, *ready(), settings, 0)
    scores = ["score_event", "score_control"]
    same = frames[0].select(KEYS).equals(repeated.select(KEYS))
    by_head: dict[str, float] = {}
    if same:
        delta = frames[0].select(
            "head", pl.max_horizontal([(pl.col(name) - repeated[name]).abs() for name in scores]).alias("delta")
        )
        table = delta.group_by("head", maintain_order=True).agg(pl.col("delta").max())
        by_head = {str(row["head"]): float(row["delta"]) for row in table.to_dicts()}
    largest = max(by_head.values(), default=0.0)
    device = resolve_device(settings.device)
    report = {
        "schema_version": 1,
        "step": "detect",
        "settings": asdict(settings),
        "inputs": inputs,
        "labels": list(study.labels),
        "examples": study.events.height,
        "patients": study.events["cluster_id"].n_unique(),
        "gate_determinism": {
            "comparable": bool(same),
            "equal_scores": bool(same and largest == 0.0),
            "largest": largest,
            "by_head": by_head,
        },
        "seconds": seconds,
        "total_seconds": time.perf_counter() - began,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "runtime": {"torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    output.mkdir(parents=True)
    pl.concat(frames).write_parquet(output / "pairs.parquet")
    study.events.select("example_id", "cluster_id", "label", "target", "fold").write_parquet(
        output / "examples.parquet"
    )
    write_json(output / "report.json", report)
    return report


def _patient_means(frame: pl.DataFrame) -> dict[str, float]:
    """The share of correct pairs of every patient, as the mean over the seeds of an arm."""
    value = (
        pl.when(pl.col("score_event") > pl.col("score_control"))
        .then(1.0)
        .when(pl.col("score_event") == pl.col("score_control"))
        .then(0.5)
        .otherwise(0.0)
    )
    table = (
        frame.with_columns(value.alias("value")).group_by("cluster_id").agg(pl.col("value").mean()).sort("cluster_id")
    )
    return dict(zip(table["cluster_id"].to_list(), table["value"].to_list(), strict=True))


def read_detection(root: Path, n_draws: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The measures of a stored recording test.

    Returns
    -------
    dict
        ``rows``: for every arm, head, readout and class, the mean over patients
        of the share of correct pairs, its bootstrap interval over patients, and
        the numbers of patients and pairs. ``differences``: for the fixed pairs
        of arms and of readouts, the paired difference by class. ``null`` and
        ``gate_determinism``.
    """
    report = orjson.loads((root / "report.json").read_bytes())
    examples = pl.read_parquet(root / "examples.parquet").select("example_id", "cluster_id")
    pairs = pl.read_parquet(root / "pairs.parquet").join(examples, on="example_id", how="left")
    labels = report["labels"]
    rng = np.random.default_rng(seed)
    means: dict[tuple[str, str, str, str], dict[str, float]] = {}
    counts: dict[tuple[str, str, str, str], int] = {}
    for key, frame in pairs.group_by(["arm", "head", "readout", "label"], maintain_order=True):
        name = (str(key[0]), str(key[1]), str(key[2]), str(key[3]))
        means[name] = _patient_means(frame)
        counts[name] = frame["example_id"].n_unique()
    draws: dict[str, NDArray[np.int64]] = {}
    for label in labels:
        holders = sorted({patient for key, values in means.items() if key[3] == label for patient in values})
        draws[label] = rng.integers(0, max(len(holders), 1), size=(n_draws, max(len(holders), 1)))

    def sampled(values: dict[str, float], label: str) -> NDArray[np.float64]:
        ordered = np.asarray([values[patient] for patient in sorted(values)], dtype=np.float64)
        return ordered[draws[label]].mean(axis=1)

    rows = []
    for (arm, head, readout, label), values in means.items():
        rows.append(
            {
                "arm": arm,
                "head": head,
                "readout": readout,
                "class": label,
                "value": float(np.mean(list(values.values()))),
                **interval(sampled(values, label)),
                "patients": len(values),
                "pairs": counts[(arm, head, readout, label)],
            }
        )

    def difference(first: tuple[str, str, str], second: tuple[str, str, str], label: str) -> dict[str, Any] | None:
        a, b = means.get((*first, label)), means.get((*second, label))
        if a is None or b is None or set(a) != set(b):
            return None
        delta = {patient: a[patient] - b[patient] for patient in a}
        return {
            "first": "/".join(first),
            "second": "/".join(second),
            "class": label,
            "difference": float(np.mean(list(delta.values()))),
            **interval(sampled(delta, label)),
        }

    differences = []
    for head in HEADS:
        for label in labels:
            for first, second in (
                (("S1-L2", head, "full"), ("spectral", head, "full")),
                (("S1-L2", head, "full"), ("S1-frozen-data", head, "full")),
                (("spectral", head, "event"), ("spectral", head, "context")),
                (("S1-L2", head, "event"), ("S1-L2", head, "context")),
            ):
                found = difference(first, second, label)
                if found is not None:
                    differences.append(found)
    return {
        "labels": labels,
        "settings": report["settings"],
        "rows": [row for row in rows if row["arm"] != NULL_ARM],
        "null": [row for row in rows if row["arm"] == NULL_ARM],
        "differences": differences,
        "gate_determinism": report["gate_determinism"],
        "total_seconds": report["total_seconds"],
    }


def evaluate_detection(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply the rules of the recording test to its measures; no data is read.

    Returns
    -------
    dict
        ``readable``: for every class, whether a fixed arm (``spectral`` or
        ``S1-L2``) reads its events on the ``full`` readout with a value of at
        least ``READ_LIMIT`` and an interval above 0.5, on both heads.
        ``region_confound``: whether the background class is readable in the
        same sense (its events are marked background, so a readable background
        means that marked and unmarked stretches differ). ``outcome``:
        ``event_signal`` if an epileptiform class is readable, else
        ``no_event_signal``. ``gate_null``: the null of every class is within
        ``NULL_MARGIN`` of 0.5.
    """
    rows = {(row["arm"], row["head"], row["readout"], row["class"]): row for row in summary["rows"]}

    def reads(arm: str, label: str) -> bool:
        found = [rows.get((arm, head, "full", label)) for head in HEADS]
        return all(row is not None and row["value"] >= READ_LIMIT and row["low"] > 0.5 for row in found)

    readable = {label: [arm for arm in FIXED_ARMS if reads(arm, label)] for label in summary["labels"]}
    null = {row["class"]: row["value"] for row in summary["null"]}
    return {
        "readable": readable,
        "region_confound": bool(readable.get(CONTROL_CLASS)),
        "outcome": "event_signal" if any(readable.get(label) for label in EPILEPTIFORM) else "no_event_signal",
        "gate_null": bool(null) and all(abs(value - 0.5) <= NULL_MARGIN for value in null.values()),
        "gate_determinism": bool(summary["gate_determinism"]["equal_scores"]),
    }


def render_detection(summary: dict[str, Any], outcomes: dict[str, Any]) -> str:
    """The recording test as text: the gates, the measure by arm and class, and the rules."""
    labels = summary["labels"]
    gate = summary["gate_determinism"]
    null = ", ".join(f"{row['class']} {row['value']:.3f}" for row in summary["null"])
    lines = [
        "Recording test: the share of pairs in which the event has the higher score (mean over patients; chance 0.5)",
        "",
        f"  determinism  {'PASS' if outcomes['gate_determinism'] else 'FAIL'}  largest score difference {gate['largest']:.2e}"
        + "".join(f", {head} {value:.2e}" for head, value in gate.get("by_head", {}).items()),
        f"  null         {'PASS' if outcomes['gate_null'] else 'FAIL'}  {null}",
        "",
        "Patients and pairs by class: "
        + ", ".join(
            f"{row['class']} {row['patients']} and {row['pairs']}"
            for row in summary["rows"]
            if (row["arm"], row["head"], row["readout"]) == ("spectral", "logistic", "full")
        ),
    ]
    for readout in PAIR_READOUTS:
        lines += ["", f"Readout {readout}: value (low to high)"]
        table = []
        for arm in PAIR_ARMS:
            for head in HEADS:
                found = {
                    row["class"]: row
                    for row in summary["rows"]
                    if (row["arm"], row["head"], row["readout"]) == (arm, head, readout)
                }
                if found:
                    table.append(
                        [arm, head]
                        + [
                            f"{found[label]['value']:.2f} ({found[label]['low']:.2f} to {found[label]['high']:.2f})"
                            if label in found
                            else ""
                            for label in labels
                        ]
                    )
        lines += _table(["arm", "head", *labels], table)
    lines += ["", "Paired differences by class (first minus second; arm/head/readout)"]
    lines += _table(
        ["first", "second", "class", "difference", "interval"],
        [
            [
                row["first"],
                row["second"],
                row["class"],
                f"{row['difference']:+.3f}",
                f"{row['low']:+.3f} to {row['high']:+.3f}",
            ]
            for row in summary["differences"]
        ],
    )
    lines += [
        "",
        f"Rule: a fixed arm reads a class at {READ_LIMIT:.2f} or more with an interval above 0.5, on both heads",
    ]
    for label in labels:
        arms = outcomes["readable"].get(label, [])
        lines.append(f"  {label:<5} {'readable by ' + ' and '.join(arms) if arms else 'not readable'}")
    lines.append(
        f"Marked background against unmarked stretches: {'differs' if outcomes['region_confound'] else 'no difference'}"
    )
    lines.append(f"Outcome: {outcomes['outcome']}")
    return "\n".join(lines) + "\n"

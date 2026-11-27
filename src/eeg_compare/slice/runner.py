"""Step 2 of the slice: the frozen arms and the controls, from signals to stored predictions (protocol, section 14).

Arms of this step, all on the primary input with cosine tokens:

================  ===========================================================
Arm               Features
================  ===========================================================
prior             none; the class shares of the training part
spectral          band energies and shares per channel
spectra_bag       counts of k-means tokens of window spectra
S1-frozen-data    counts of tokens of random single-channel plan windows
S1-frozen-gauss   counts of tokens of Gaussian single-channel atoms
SF-frozen-data    counts of tokens of the same random plan windows, all channels
================  ===========================================================

Every fold fits its dictionaries, its k-means models and its heads on its
training patients and predicts its validation patients. The run stores one
prediction per example, arm, seed, head and readout. It also holds three gates:
the null of gate 3 (``spectral`` with permuted training labels, several
permutations), the counts of gate 5 and the repeat of gate 6 (fold 0 is
computed twice from the signals).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import sklearn
import torch
from numpy.typing import NDArray

from eeg_compare.cells.api import Geometry, Start, draw_start, start_atoms
from eeg_compare.cells.common import CellAtoms, unit_norm
from eeg_compare.data.manifest import Manifest
from eeg_compare.data.provenance import content_hash, write_json
from eeg_compare.device import resolve_device
from eeg_compare.slice.features import band_features, cell_tokens, count_features, window_spectra
from eeg_compare.slice.heads import (
    example_weights,
    forest_probabilities,
    logistic_probabilities,
    prior_probabilities,
    standardize,
)
from eeg_compare.slice.readouts import Readout, make_readouts
from eeg_compare.slice.scaling import Scales
from eeg_compare.slice.signals import study_clip
from eeg_compare.slice.spectra import fit_spectra_bag, spectra_tokens
from eeg_compare.slice.study import Study, clip_shares, fold_interval_rows, fold_parts

__all__ = [
    "ARMS",
    "NULL_ARM",
    "SCHEMA_VERSION",
    "Block",
    "Prepared",
    "Settings",
    "compare_blocks",
    "fold_blocks",
    "predictions_frame",
    "prepare",
    "run",
    "smoke_settings",
]

SCHEMA_VERSION = 1
ARMS = ("prior", "spectral", "spectra_bag", "S1-frozen-data", "S1-frozen-gauss", "SF-frozen-data")
NULL_ARM = "spectral-null"
LOGISTIC_PASSES = 2000


@dataclass(frozen=True)
class Settings:
    """The settings of a run of step 2; the defaults are those of the protocol.

    Parameters
    ----------
    n_atoms : int
        Atoms ``K`` of a dictionary and centers of the bag of spectra.
    atom_len : int
        Atom length ``L``.
    pool_len : int
        Pooling window ``P``.
    seeds : tuple[int, ...]
        Seeds of the arms with a random part.
    forest_seeds : tuple[int, ...]
        Seeds of the random forest, for every seed of an arm.
    forest_trees : int
        Trees of the random forest.
    kmeans_passes : int
        Largest number of passes of k-means.
    start_attempts : int
        Draws of a start before the run stops, when a draw holds a flat patch.
    null_permutations : int
        Label permutations of gate 3; the gate reads their mean.
    n_jobs : int
        Processes of the random forest; ``-1`` uses all cores.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.
    """

    n_atoms: int = 32
    atom_len: int = 125
    pool_len: int = 125
    seeds: tuple[int, ...] = (0, 1, 2)
    forest_seeds: tuple[int, ...] = (0, 1, 2)
    forest_trees: int = 500
    kmeans_passes: int = 100
    start_attempts: int = 100
    null_permutations: int = 10
    n_jobs: int = -1
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in (
            "n_atoms",
            "atom_len",
            "pool_len",
            "forest_trees",
            "kmeans_passes",
            "start_attempts",
            "null_permutations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not self.seeds or not self.forest_seeds:
            raise ValueError("a run needs at least one seed and one forest seed")


def smoke_settings(device: str = "auto") -> Settings:
    """Small settings for a check of the wiring; their scores are not results."""
    return Settings(
        n_atoms=8,
        seeds=(0,),
        forest_seeds=(0,),
        forest_trees=20,
        kmeans_passes=20,
        null_permutations=2,
        device=device,
    )


@dataclass(frozen=True)
class Prepared:
    """What does not depend on a fold: the readouts, the spectral features and the window spectra."""

    readouts: dict[str, Readout]
    spectral: dict[str, NDArray[np.float64]]
    context_spectra: torch.Tensor
    control_spectra: torch.Tensor
    interval_spectra: torch.Tensor


@dataclass(frozen=True)
class Block:
    """The predictions of one arm, seed, head and readout for the validation part of one fold.

    Parameters
    ----------
    arm, seed, head, head_seed, readout, fold
        The keys of the block. ``head`` is ``logistic``, ``forest`` or ``prior``.
    rows : numpy.ndarray
        Rows of the event table that the block predicts.
    probabilities : numpy.ndarray
        ``(len(rows), n_classes)`` class probabilities in single precision.
    """

    arm: str
    seed: int
    head: str
    head_seed: int
    readout: str
    fold: int
    rows: NDArray[np.int64]
    probabilities: NDArray[np.float32]

    @property
    def key(self) -> tuple[str, int, str, int, str, int]:
        """The keys of the block."""
        return self.arm, self.seed, self.head, self.head_seed, self.readout, self.fold


def _seed(*parts: object) -> int:
    """A 32-bit seed from the name of a random part, so that no two parts share a stream."""
    return int(content_hash(list(parts))[:8], 16)


def prepare(study: Study, settings: Settings) -> Prepared:
    """The readouts, the spectral features of every readout and the spectra of all windows."""
    readouts = make_readouts(study.contexts.shape[-1], settings.atom_len, settings.pool_len, study.event_samples)
    frame = int(round(study.sfreq))
    spectral = {
        name: band_features(
            study.controls if readout.displaced else study.contexts,
            readout.ranges,
            study.sfreq,
            nperseg=frame,
            device=settings.device,
        )
        for name, readout in readouts.items()
    }
    context_spectra, control_spectra, interval_spectra = (
        window_spectra(rows, settings.atom_len, settings.pool_len, study.sfreq, device=settings.device)
        for rows in (study.contexts, study.controls, study.interval_signals)
    )
    return Prepared(readouts, spectral, context_spectra, control_spectra, interval_spectra)


def _valid_start(rows: torch.Tensor, geometry: Geometry, seed: int, attempts: int) -> tuple[Start, int]:
    """The common start of a fold and seed; a draw with a flat patch is drawn again from the same stream."""
    generator = torch.Generator().manual_seed(seed)
    for attempt in range(1, attempts + 1):
        try:
            return draw_start(rows, geometry, generator), attempt
        except ValueError as error:
            if "flat" not in str(error):
                raise
    raise ValueError(f"no start without a flat patch in {attempts} draws")


def _token_features(
    arm: str,
    tokens: tuple[torch.Tensor, torch.Tensor],
    readouts: dict[str, Readout],
    n_atoms: int,
    counts: list[dict[str, Any]],
) -> dict[str, NDArray[np.float64]]:
    """The count features of every readout, and the record of gate 5 for them."""
    features = {}
    for name, readout in readouts.items():
        chosen = tokens[1] if readout.displaced else tokens[0]
        values = count_features(chosen, readout.windows, n_atoms)
        sums = values.sum(axis=1)
        counts.append(
            {
                "arm": arm,
                "readout": name,
                "tokens_per_context": len(readout.windows) * int(chosen.shape[2]),
                "sum_min": float(sums.min()),
                "sum_max": float(sums.max()),
            }
        )
        features[name] = values
    return features


def fold_blocks(study: Study, prepared: Prepared, settings: Settings, fold: int) -> tuple[list[Block], dict[str, Any]]:
    """All predictions of one fold, and the record of its fits.

    Returns
    -------
    tuple[list[Block], dict]
        The blocks in a fixed order, and a record with ``seconds`` per part,
        ``logistic_passes`` (the largest), ``start_attempts`` and the ``counts``
        of gate 5.
    """
    train, validation = fold_parts(study, fold)
    targets = study.events["target"].to_numpy().astype(np.int64)
    patients = np.asarray(study.events["cluster_id"].to_list(), dtype=str)
    n_classes = study.n_classes
    weights = example_weights(targets[train], patients[train], n_classes)
    blocks: list[Block] = []
    seconds: dict[str, float] = defaultdict(float)
    record: dict[str, Any] = {"fold": fold, "logistic_passes": 0, "start_attempts": [], "counts": []}

    def add(arm: str, seed: int, head: str, head_seed: int, readout: str, probabilities: NDArray[np.float64]) -> None:
        blocks.append(Block(arm, seed, head, head_seed, readout, fold, validation, probabilities.astype(np.float32)))

    def heads(arm: str, seed: int, features: dict[str, NDArray[np.float64]]) -> None:
        for name, values in features.items():
            x_train, x_test = standardize(values[train], values[validation])
            began = time.perf_counter()
            found, passes = logistic_probabilities(
                x_train, targets[train], weights, x_test, n_classes, max_iter=LOGISTIC_PASSES
            )
            seconds["logistic"] += time.perf_counter() - began
            record["logistic_passes"] = max(record["logistic_passes"], passes)
            add(arm, seed, "logistic", 0, name, found)
            began = time.perf_counter()
            for forest_seed in settings.forest_seeds:
                found = forest_probabilities(
                    x_train,
                    targets[train],
                    weights,
                    x_test,
                    n_classes,
                    forest_seed,
                    trees=settings.forest_trees,
                    n_jobs=settings.n_jobs,
                )
                add(arm, seed, "forest", forest_seed, name, found)
            seconds["forest"] += time.perf_counter() - began

    add("prior", 0, "prior", 0, "full", prior_probabilities(targets[train], validation.size, n_classes))
    heads("spectral", 0, prepared.spectral)

    # Gate 3: the same head with the training labels in a random order.
    x_train, x_test = standardize(prepared.spectral["full"][train], prepared.spectral["full"][validation])
    for number in range(settings.null_permutations):
        permuted = np.random.default_rng(_seed("null", number, fold)).permutation(targets[train])
        null_weights = example_weights(permuted, patients[train], n_classes)
        found, _ = logistic_probabilities(x_train, permuted, null_weights, x_test, n_classes, max_iter=LOGISTIC_PASSES)
        add(NULL_ARM, number, "logistic", 0, "full", found)

    rows = torch.from_numpy(fold_interval_rows(study, fold))
    training_spectra = prepared.interval_spectra[rows]
    for seed in settings.seeds:
        began = time.perf_counter()
        model = fit_spectra_bag(
            training_spectra, settings.n_atoms, _seed("spectra", fold, seed), settings.kmeans_passes
        )
        tokens = (spectra_tokens(model, prepared.context_spectra), spectra_tokens(model, prepared.control_spectra))
        seconds["spectra_bag"] += time.perf_counter() - began
        heads(
            "spectra_bag",
            seed,
            _token_features("spectra_bag", tokens, prepared.readouts, settings.n_atoms, record["counts"]),
        )
    del training_spectra

    intervals = study.interval_signals[rows]
    geometry = Geometry(settings.n_atoms, settings.atom_len, settings.pool_len, 1)
    for seed in settings.seeds:
        start, attempts = _valid_start(intervals, geometry, _seed("start", fold, seed), settings.start_attempts)
        record["start_attempts"].append(attempts)
        gauss = torch.randn(
            (settings.n_atoms, settings.atom_len), generator=torch.Generator().manual_seed(_seed("gauss", fold, seed))
        )
        dictionaries = {
            "S1-frozen-data": start_atoms("S1", start, geometry),
            "S1-frozen-gauss": CellAtoms("S1", waveform=unit_norm(gauss)),
            "SF-frozen-data": start_atoms("SF", start, geometry),
        }
        for arm, atoms in dictionaries.items():
            began = time.perf_counter()
            tokens = (
                cell_tokens(study.contexts, atoms, "cosine", settings.pool_len, settings.device),
                cell_tokens(study.controls, atoms, "cosine", settings.pool_len, settings.device),
            )
            seconds["tokens"] += time.perf_counter() - began
            heads(arm, seed, _token_features(arm, tokens, prepared.readouts, settings.n_atoms, record["counts"]))
    record["seconds"] = dict(seconds)
    return blocks, record


def compare_blocks(first: list[Block], second: list[Block]) -> dict[str, Any]:
    """Gate 6: do two computations of a fold give the same predictions?

    Returns
    -------
    dict
        ``blocks``, ``equal_predictions`` (every predicted class is equal) and
        ``max_probability_difference``.
    """
    same = len(first) == len(second) and all(a.key == b.key for a, b in zip(first, second, strict=False))
    largest = 0.0
    if same:
        for a, b in zip(first, second, strict=True):
            same = same and np.array_equal(a.rows, b.rows)
            same = same and np.array_equal(a.probabilities.argmax(axis=1), b.probabilities.argmax(axis=1))
            if not same:
                break
            largest = max(largest, float(np.abs(a.probabilities - b.probabilities).max()))
    return {"blocks": len(first), "equal_predictions": bool(same), "max_probability_difference": largest}


def predictions_frame(study: Study, blocks: list[Block]) -> pl.DataFrame:
    """One row per example and block: the keys, the target, the predicted class and the class probabilities."""
    base = study.events.select("example_id", "target")
    frames = []
    for block in blocks:
        columns = [pl.Series("prediction", block.probabilities.argmax(axis=1).astype(np.int64))]
        columns += [
            pl.Series(f"p{label}", block.probabilities[:, label]) for label in range(block.probabilities.shape[1])
        ]
        frames.append(
            base.select(pl.all().gather(block.rows.tolist()))
            .with_columns(columns)
            .select(
                pl.lit(block.arm).alias("arm"),
                pl.lit(block.seed, dtype=pl.Int64).alias("seed"),
                pl.lit(block.head).alias("head"),
                pl.lit(block.head_seed, dtype=pl.Int64).alias("head_seed"),
                pl.lit(block.readout).alias("readout"),
                pl.lit(block.fold, dtype=pl.Int64).alias("fold"),
                pl.all(),
            )
        )
    return pl.concat(frames)


def _merged_counts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The counts of gate 5 over all folds and seeds, one row per arm and readout."""
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        for row in record["counts"]:
            key = (row["arm"], row["readout"])
            if key not in merged:
                merged[key] = dict(row)
                continue
            kept = merged[key]
            if kept["tokens_per_context"] != row["tokens_per_context"]:
                raise ValueError(f"the tokens of {key} differ between folds or seeds")
            kept["sum_min"] = min(kept["sum_min"], row["sum_min"])
            kept["sum_max"] = max(kept["sum_max"], row["sum_max"])
    return list(merged.values())


def run(
    study: Study, manifest: Manifest, scales: Scales, settings: Settings, output: Path, inputs: dict[str, Any]
) -> dict[str, Any]:
    """Run step 2 and store it in a new directory.

    The directory gets ``predictions.parquet``, ``examples.parquet`` (the patient,
    the class, the row weight and the fold of every example) and ``report.json``
    (settings, inputs, gates, counts and times).

    Parameters
    ----------
    study : Study
        The loaded data of the slice.
    manifest : Manifest
        The manifest of the study.
    scales : Mapping[str, numpy.ndarray]
        The channel scales of the training recordings, for the rule of the ``profile`` input.
    settings : Settings
        The settings of the run.
    output : Path
        New directory of the run.
    inputs : dict
        What identifies the inputs (hashes and paths); it is stored as it is.

    Raises
    ------
    ValueError
        If the directory exists.
    """
    if output.exists():
        raise ValueError(f"{output} exists; a run is never overwritten")
    began = time.perf_counter()
    prepared = prepare(study, settings)
    prepare_seconds = time.perf_counter() - began
    blocks: list[Block] = []
    records = []
    for fold in range(study.n_folds):
        found, record = fold_blocks(study, prepared, settings, fold)
        blocks.extend(found)
        records.append(record)
    repeated, _ = fold_blocks(study, prepare(study, settings), settings, 0)
    determinism = compare_blocks([block for block in blocks if block.fold == 0], repeated)
    clip = clip_shares(study, manifest, scales, study_clip(manifest.config.preprocessing))
    device = resolve_device(settings.device)
    report = {
        "schema_version": SCHEMA_VERSION,
        "step": 2,
        "settings": asdict(settings),
        "inputs": inputs,
        "labels": list(study.labels),
        "examples": study.events.height,
        "patients": study.events["cluster_id"].n_unique(),
        "plan_intervals": study.intervals.height,
        "channels": int(study.contexts.shape[1]),
        "context_samples": int(study.contexts.shape[2]),
        "event_samples": list(study.event_samples),
        "readouts": {
            name: {"windows": list(readout.windows), "ranges": [list(pair) for pair in readout.ranges]}
            for name, readout in prepared.readouts.items()
        },
        "gate_2": {
            "recordings_read": study.recordings_read,
            "training_recordings_only": True,
            "fold_parts_disjoint": True,
        },
        "gate_5": _merged_counts(records),
        "gate_6": determinism,
        "clip_share": clip,
        "logistic": {"largest_passes": max(record["logistic_passes"] for record in records), "limit": LOGISTIC_PASSES},
        "start_attempts": [record["start_attempts"] for record in records],
        "seconds": {"prepare": prepare_seconds, "folds": [record["seconds"] for record in records]},
        "total_seconds": time.perf_counter() - began,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "runtime": {
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "polars": pl.__version__,
            "device": str(device),
        },
    }
    output.mkdir(parents=True)
    predictions_frame(study, blocks).write_parquet(output / "predictions.parquet")
    study.events.select("example_id", "cluster_id", "label", "target", "row_weight", "fold").write_parquet(
        output / "examples.parquet"
    )
    write_json(output / "report.json", report)
    return report

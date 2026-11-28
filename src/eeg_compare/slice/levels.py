"""Step 3 of the slice: the learned arms and the levels (protocol, sections 6 and 14).

A fold of step 3 holds the arms of step 2 and these arms. The name of an arm
with a level is the arm, an ``@`` and the level.

==========================  ==================================================================
Arm                         What changes against the primary setting
==========================  ==================================================================
``S1-L1``, ``S1-L2``        The two alternating single-channel cells, fitted on the plan
``<arm>@raw``               Raw tokens on the same dictionary (learned and frozen arms)
``S1-L?@K<n>``              Another number of atoms; first seed only
``<arm>@P<n>``              A wider pooling window; a new fit for ``S1-L2``, none for the frozen arms
``S1-L2@10min``             The plan with the larger quota; the same start
``spectral@profile``        The ``profile`` input
``SF-frozen-data@profile``  The same windows, taken from the ``profile`` input
``S1-L2@profile-raw``       A fit on the ``profile`` input with raw tokens
``S1-L2@half-in``           Dictionary from one half of the training patients; head on that half
``S1-L2@half-out``          The same dictionary; head on the other half
==========================  ==================================================================

A level uses the first seed of the run. A run of step 3 is long, so it is
stored fold by fold, and a second call with the same settings continues it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl
import torch
from numpy.typing import NDArray

from eeg_compare.cells.api import Budget, Geometry, fit_cell, start_atoms
from eeg_compare.cells.common import CellAtoms, unit_norm
from eeg_compare.data.manifest import Manifest
from eeg_compare.data.provenance import content_hash, write_json
from eeg_compare.device import resolve_device
from eeg_compare.slice.evaluate import patient_strata
from eeg_compare.slice.features import band_features, cell_tokens
from eeg_compare.slice.readouts import Readout, make_readouts
from eeg_compare.slice.runner import (
    SCHEMA_VERSION,
    Block,
    FoldHeads,
    Prepared,
    Settings,
    _seed,
    _token_features,
    _valid_start,
    build_report,
    fold_blocks,
    predictions_frame,
    prepare,
)
from eeg_compare.slice.scaling import Scales, fit_profile, profile_factors
from eeg_compare.slice.signals import study_clip
from eeg_compare.slice.study import Study, clip_shares, fold_interval_rows

__all__ = [
    "KEY_COLUMNS",
    "LEARNED",
    "compare_frames",
    "fold_factors",
    "level_blocks",
    "patient_halves",
    "profile_rows",
    "run_levels",
]

LEARNED = ("S1-L1", "S1-L2")
KEY_COLUMNS = ["arm", "seed", "head", "head_seed", "readout", "fold", "example_id"]


def fold_factors(manifest: Manifest, scales: Scales, fold: int, cap: float = 3.0) -> dict[str, NDArray[np.float64]]:
    """The factor of the ``profile`` input for every training-set recording, with the profile of one fold.

    The profile comes from the recordings of the training patients of the fold.
    A recording of a validation patient uses its own scales and that profile.
    """
    fold_of = {row.cluster_id: row.validation_fold for row in manifest.assignments if row.official_split == "train"}
    recordings = [row for row in manifest.recordings if row.official_split == "train" and not row.missing_channels]
    training = [row.recording_id for row in recordings if fold_of[row.cluster_id] != fold]
    profile = fit_profile(scales, training, cap)
    return profile_factors(scales, profile, [row.recording_id for row in recordings])


def profile_rows(
    rows: torch.Tensor, recordings: Sequence[str], factors: dict[str, NDArray[np.float64]], clip: float
) -> torch.Tensor:
    """The ``profile`` input of rows ``(n, C, T)`` of the primary input; row ``i`` belongs to ``recordings[i]``."""
    if len(recordings) != rows.shape[0]:
        raise ValueError("the profile input needs one recording per row")
    matrix = torch.from_numpy(np.stack([factors[name] for name in recordings]).astype(np.float32))
    return torch.clamp(rows * matrix[:, :, None], -clip, clip)


def patient_halves(patients: Sequence[str], strata: Sequence[int], seed: int) -> tuple[set[str], set[str]]:
    """Two halves of patients, balanced inside every stratum.

    The patients of a stratum are ranked by a hash of the seed and the patient,
    and they go to the two halves in turn.
    """
    first: set[str] = set()
    second: set[str] = set()
    by_stratum: dict[int, list[str]] = {}
    for patient, stratum in sorted(set(zip(patients, strata, strict=True))):
        by_stratum.setdefault(int(stratum), []).append(patient)
    for stratum in sorted(by_stratum):
        ranked = sorted(by_stratum[stratum], key=lambda name: (content_hash([seed, name]), name))
        first.update(ranked[0::2])
        second.update(ranked[1::2])
    return first, second


def level_blocks(
    study: Study,
    prepared: Prepared,
    manifest: Manifest,
    scales: Scales,
    clip: float,
    settings: Settings,
    fold: int,
) -> tuple[list[Block], dict[str, Any]]:
    """The predictions of one fold for the learned arms and the levels, and the record of its fits.

    Returns
    -------
    tuple[list[Block], dict]
        The blocks in a fixed order, and a record with ``fits`` (passes, time and
        health of every alternating fit), ``counts`` of gate 5, ``seconds`` and
        ``logistic_passes``.

    Raises
    ------
    ValueError
        If the study has no wide plan, or a fold has fewer than two training patients.
    """
    helper = FoldHeads(study, settings, fold)
    seconds = helper.seconds
    counts: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    device = resolve_device(settings.device)
    budget = Budget(max_passes=settings.max_passes, tol=settings.tol)
    n_atoms, atom_len, pool_len, wide = settings.n_atoms, settings.atom_len, settings.pool_len, settings.wide_pool_len
    geometry = Geometry(n_atoms, atom_len, pool_len, 1)
    wide_readouts = make_readouts(study.contexts.shape[-1], atom_len, wide, study.event_samples)
    rows = fold_interval_rows(study, fold)
    intervals = study.interval_signals[torch.from_numpy(rows)]

    def fit(arm: str, seed: int, cell: str, segments: torch.Tensor, initial: CellAtoms, shape: Geometry) -> CellAtoms:
        began = time.perf_counter()
        fitted = fit_cell(cell, segments, initial, shape, budget, device=settings.device)
        last = fitted.record["history"][-1]
        assert fitted.atoms.waveform is not None
        atoms = CellAtoms("S1", waveform=fitted.atoms.waveform.detach().cpu())
        elapsed = time.perf_counter() - began
        seconds["fit"] += elapsed
        fits.append(
            {
                "arm": arm,
                "seed": seed,
                "atoms": shape.n_atoms,
                "pool_len": shape.pool_len,
                "segments": int(segments.shape[0]),
                "passes": int(fitted.record["passes"]),
                "converged": bool(fitted.record["converged"]),
                "seconds": elapsed,
                "residual_share": float(last["residual_share"]),
                "assigned_share": float(last["assigned_share"]),
                "empty_atoms": int(last["empty_atoms"]),
            }
        )
        del fitted
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return atoms

    def emit(
        arm: str,
        seed: int,
        atoms: CellAtoms,
        score: str = "cosine",
        pool: int = pool_len,
        reads: dict[str, Readout] = prepared.readouts,
        contexts: torch.Tensor = study.contexts,
        controls: torch.Tensor = study.controls,
        heads: FoldHeads = helper,
    ) -> None:
        began = time.perf_counter()
        tokens = (
            cell_tokens(contexts, atoms, score, pool, settings.device),  # type: ignore[arg-type]
            cell_tokens(controls, atoms, score, pool, settings.device),  # type: ignore[arg-type]
        )
        seconds["tokens"] += time.perf_counter() - began
        heads.heads(arm, seed, _token_features(arm, tokens, reads, atoms.n_atoms, counts))

    # The learned arms, and raw tokens on every dictionary, for every seed.
    first: dict[str, CellAtoms] = {}
    for seed in settings.seeds:
        start, _ = _valid_start(intervals, geometry, _seed("start", fold, seed), settings.start_attempts)
        gauss = torch.randn((n_atoms, atom_len), generator=torch.Generator().manual_seed(_seed("gauss", fold, seed)))
        frozen = {
            "S1-frozen-data": start_atoms("S1", start, geometry),
            "S1-frozen-gauss": CellAtoms("S1", waveform=unit_norm(gauss)),
            "SF-frozen-data": start_atoms("SF", start, geometry),
        }
        learned = {name: fit(name, seed, name, intervals, frozen["S1-frozen-data"], geometry) for name in LEARNED}
        for name, atoms in learned.items():
            emit(name, seed, atoms)
        for name, atoms in {**learned, **frozen}.items():
            emit(f"{name}@raw", seed, atoms, score="raw")
        if seed == settings.seeds[0]:
            first = {**learned, **frozen}
    seed = settings.seeds[0]

    # The number of atoms.
    for count in settings.level_atoms:
        shape = Geometry(count, atom_len, pool_len, 1)
        start, _ = _valid_start(intervals, shape, _seed("start", fold, seed, count), settings.start_attempts)
        initial = start_atoms("S1", start, shape)
        for name in LEARNED:
            emit(f"{name}@K{count}", seed, fit(f"{name}@K{count}", seed, name, intervals, initial, shape))

    # The tiling: a new fit of S1-L2, and the same frozen dictionaries.
    shape = Geometry(n_atoms, atom_len, wide, 1)
    tiled = fit(f"S1-L2@P{wide}", seed, "S1-L2", intervals, first["S1-frozen-data"], shape)
    emit(f"S1-L2@P{wide}", seed, tiled, pool=wide, reads=wide_readouts)
    for name in ("S1-frozen-data", "S1-frozen-gauss", "SF-frozen-data"):
        emit(f"{name}@P{wide}", seed, first[name], pool=wide, reads=wide_readouts)

    # The exposure: the plan with the larger quota, from the same start.
    more = study.interval_signals[torch.from_numpy(fold_interval_rows(study, fold, wide=True))]
    emit("S1-L2@10min", seed, fit("S1-L2@10min", seed, "S1-L2", more, first["S1-frozen-data"], geometry))
    del more

    # The profile input: the same places of the start, taken from the profile input.
    factors = fold_factors(manifest, scales, fold)
    event_recordings = study.events["recording_id"].to_list()
    interval_recordings = study.intervals["recording_id"].gather(rows.tolist()).to_list()
    contexts = profile_rows(study.contexts, event_recordings, factors, clip)
    controls = profile_rows(study.controls, event_recordings, factors, clip)
    scaled = profile_rows(intervals, interval_recordings, factors, clip)
    frame = int(round(study.sfreq))
    spectral = {
        name: band_features(
            controls if readout.displaced else contexts,
            readout.ranges,
            study.sfreq,
            nperseg=frame,
            device=settings.device,
        )
        for name, readout in prepared.readouts.items()
    }
    helper.heads("spectral@profile", 0, spectral)
    start, _ = _valid_start(scaled, geometry, _seed("start", fold, seed), settings.start_attempts)
    emit("SF-frozen-data@profile", seed, start_atoms("SF", start, geometry), contexts=contexts, controls=controls)
    fitted = fit("S1-L2@profile-raw", seed, "S1-L2", scaled, start_atoms("S1", start, geometry), geometry)
    emit("S1-L2@profile-raw", seed, fitted, score="raw", contexts=contexts, controls=controls)
    del contexts, controls, scaled

    # Cross-fitting check: a dictionary from one half of the training patients, and a head on each half.
    patients = helper.patients
    index, stratum = patient_strata(helper.targets[helper.train], patients[helper.train], helper.n_classes)
    halves = patient_halves(patients[helper.train].tolist(), stratum[index].tolist(), _seed("half", fold))
    interval_patients = study.intervals["cluster_id"].gather(rows.tolist()).to_list()
    inside = torch.tensor([name in halves[0] for name in interval_patients])
    part = intervals[inside]
    start, _ = _valid_start(part, geometry, _seed("start", fold, seed, "half"), settings.start_attempts)
    atoms = fit("S1-L2@half", seed, "S1-L2", part, start_atoms("S1", start, geometry), geometry)
    began = time.perf_counter()
    tokens = (
        cell_tokens(study.contexts, atoms, "cosine", pool_len, settings.device),
        cell_tokens(study.controls, atoms, "cosine", pool_len, settings.device),
    )
    seconds["tokens"] += time.perf_counter() - began
    features = _token_features("S1-L2@half", tokens, prepared.readouts, n_atoms, counts)
    blocks = helper.blocks
    passes = helper.logistic_passes
    for name, half in (("S1-L2@half-in", halves[0]), ("S1-L2@half-out", halves[1])):
        chosen = np.asarray([row for row in helper.train.tolist() if patients[row] in half], dtype=np.int64)
        if not chosen.size:
            raise ValueError("the cross-fitting check needs training patients in both halves of a fold")
        other = FoldHeads(study, settings, fold, chosen)
        other.heads(name, seed, features)
        blocks = [*blocks, *other.blocks]
        passes = max(passes, other.logistic_passes)
        for key, value in other.seconds.items():
            seconds[key] += value
    record = {
        "fold": fold,
        "fits": fits,
        "counts": counts,
        "seconds": dict(seconds),
        "logistic_passes": passes,
        "halves": [len(halves[0]), len(halves[1])],
    }
    return blocks, record


def compare_frames(first: pl.DataFrame, second: pl.DataFrame) -> dict[str, Any]:
    """Gate 6 on two stored computations of a fold: the same record as ``compare_blocks``."""
    comparable = first.height == second.height and first.select(KEY_COLUMNS).equals(second.select(KEY_COLUMNS))
    if not comparable:
        return {
            "blocks": 0,
            "comparable": False,
            "equal_predictions": False,
            "predictions": 0,
            "unequal_predictions": 0,
            "max_probability_difference": 0.0,
            "by_arm": [],
        }
    classes = [name for name in first.columns if name.startswith("p") and name[1:].isdigit()]
    joined = first.select("arm", "seed", "head", "head_seed", "readout", "prediction", *classes).hstack(
        second.select(
            pl.col("prediction").alias("prediction_2"), *[pl.col(name).alias(f"{name}_2") for name in classes]
        )
    )
    rows = (
        joined.group_by("arm", maintain_order=True)
        .agg(
            pl.struct("seed", "head", "head_seed", "readout").n_unique().alias("blocks"),
            pl.len().alias("predictions"),
            (pl.col("prediction") != pl.col("prediction_2")).sum().alias("unequal_predictions"),
            pl.max_horizontal([(pl.col(name) - pl.col(f"{name}_2")).abs() for name in classes])
            .max()
            .alias("max_probability_difference"),
        )
        .to_dicts()
    )
    rows = [{**row, "max_probability_difference": float(row["max_probability_difference"])} for row in rows]
    unequal = sum(int(row["unequal_predictions"]) for row in rows)
    return {
        "blocks": sum(int(row["blocks"]) for row in rows),
        "comparable": True,
        "equal_predictions": unequal == 0,
        "predictions": sum(int(row["predictions"]) for row in rows),
        "unequal_predictions": unequal,
        "max_probability_difference": max((row["max_probability_difference"] for row in rows), default=0.0),
        "by_arm": rows,
    }


def run_levels(
    study: Study,
    manifest: Manifest,
    scales: Scales,
    settings: Settings,
    output: Path,
    inputs: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run step 3, fold by fold, into a directory that a later call continues.

    The directory gets ``config.json`` at its start, then one table and one
    record per fold under ``folds/``, then a second computation of fold 0
    (gate 6), and at the end the files of a run: ``predictions.parquet``,
    ``examples.parquet`` and ``report.json``. A part that is stored is never
    computed again.

    Raises
    ------
    ValueError
        If the study has no wide plan, the directory holds a complete run, or
        it holds a run with other settings or inputs.
    """
    if study.wide_plan is None:
        raise ValueError("step 3 needs the wide plan of the exposure level")
    config = orjson.loads(
        orjson.dumps({"schema_version": SCHEMA_VERSION, "step": 3, "settings": asdict(settings), "inputs": inputs})
    )
    parts = output / "folds"
    if output.exists():
        if (output / "report.json").exists():
            raise ValueError(f"{output} holds a complete run; a run is never overwritten")
        stored = output / "config.json"
        if not stored.exists() or orjson.loads(stored.read_bytes()) != config:
            raise ValueError(f"{output} holds a run with other settings or inputs")
    else:
        parts.mkdir(parents=True)
        write_json(output / "config.json", config)
    device = resolve_device(settings.device)
    clip = study_clip(manifest.config.preprocessing)
    prepared = prepare(study, settings)
    names = [(f"fold-{fold}", fold) for fold in range(study.n_folds)] + [("repeat-0", 0)]
    for name, fold in names:
        if (parts / f"{name}.json").exists():
            continue
        began = time.perf_counter()
        fresh = prepare(study, settings) if name.startswith("repeat") else prepared
        blocks, base = fold_blocks(study, fresh, settings, fold)
        more, levels = level_blocks(study, fresh, manifest, scales, clip, settings, fold)
        record = {
            "fold": fold,
            "fits": levels["fits"],
            "counts": [*base["counts"], *levels["counts"]],
            "start_attempts": base["start_attempts"],
            "logistic_passes": max(base["logistic_passes"], levels["logistic_passes"]),
            "halves": levels["halves"],
            "seconds": {"step_2": base["seconds"], "step_3": levels["seconds"]},
            "total_seconds": time.perf_counter() - began,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        }
        table = parts / f"{name}.parquet"
        table.unlink(missing_ok=True)
        predictions_frame(study, [*blocks, *more]).write_parquet(table)
        write_json(parts / f"{name}.json", record)
        if progress is not None:
            progress(f"{name}: {len(blocks) + len(more)} blocks in {record['total_seconds'] / 60:.1f} min")
    records = {name: orjson.loads((parts / f"{name}.json").read_bytes()) for name, _ in names}
    folds = [records[f"fold-{fold}"] for fold in range(study.n_folds)]
    determinism = compare_frames(pl.read_parquet(parts / "fold-0.parquet"), pl.read_parquet(parts / "repeat-0.parquet"))
    report = build_report(
        study, settings, inputs, prepared, folds, determinism, clip_shares(study, manifest, scales, clip), 3
    )
    report["fits"] = [record["fits"] for record in folds]
    report["halves"] = [record["halves"] for record in folds]
    report["seconds"] = {name: record["seconds"] for name, record in records.items()}
    report["total_seconds"] = sum(record["total_seconds"] for record in records.values())
    report["peak_gpu_bytes"] = max(record["peak_gpu_bytes"] for record in records.values())
    pl.concat([pl.read_parquet(parts / f"fold-{fold}.parquet") for fold in range(study.n_folds)]).write_parquet(
        output / "predictions.parquet"
    )
    study.events.select("example_id", "cluster_id", "label", "target", "row_weight", "fold").write_parquet(
        output / "examples.parquet"
    )
    write_json(output / "report.json", report)
    return report

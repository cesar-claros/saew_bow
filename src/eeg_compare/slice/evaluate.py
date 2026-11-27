"""Evaluator of the slice: scores of pooled predictions and a bootstrap over patients (slice protocol, section 9).

This module uses NumPy only, so a stored run can be read without Torch.

- The primary score is the balanced accuracy: the mean recall over the classes.
- An interval comes from a bootstrap over patients inside strata. The stratum
  of a patient is the rarest class that it holds, so every draw keeps the
  patients of a rare class. A class is rarer when fewer patients hold it; equal
  counts are ordered by the number of examples.
- Two arms are compared on the same draws (a paired interval).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "bootstrap_multiplicities",
    "confusion",
    "example_weights",
    "interval",
    "patient_strata",
    "recall_draws",
    "summarize",
]


def example_weights(targets: NDArray[np.int64], patients: NDArray[np.str_], n_classes: int) -> NDArray[np.float64]:
    """One weight per example, with mean 1.

    Inside a class, the examples of a patient share one total that is the same
    for every patient of the class. Every class that has examples has the same
    total.

    Raises
    ------
    ValueError
        If there is no example, the arrays differ in length, or a target is not a class.
    """
    targets = np.asarray(targets)
    patients = np.asarray(patients)
    if targets.ndim != 1 or targets.shape != patients.shape or not targets.size:
        raise ValueError("weights need one target and one patient per example")
    if targets.min() < 0 or targets.max() >= n_classes:
        raise ValueError("a target is not a class")
    weights = np.zeros(targets.shape[0], dtype=np.float64)
    for label in range(n_classes):
        members = np.flatnonzero(targets == label)
        if members.size:
            holders, inverse, counts = np.unique(patients[members], return_inverse=True, return_counts=True)
            weights[members] = 1.0 / (holders.size * counts[inverse])
    return weights * (weights.size / weights.sum())


def confusion(
    targets: NDArray[np.int64],
    predictions: NDArray[np.int64],
    n_classes: int,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """The confusion matrix ``(true class, predicted class)``, with a weight per example when given."""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError("the confusion matrix needs one prediction per target")
    matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(matrix, (targets, predictions), 1.0 if weights is None else np.asarray(weights, dtype=np.float64))
    return matrix


def summarize(matrix: NDArray[np.float64]) -> dict[str, Any]:
    """The scores of a confusion matrix.

    Returns
    -------
    dict
        ``recall`` per class (``nan`` for a class without examples),
        ``balanced_accuracy`` (the mean recall over the classes with examples),
        ``macro_f1`` over the same classes, ``kappa`` (Cohen) and ``accuracy``.
    """
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    hits = np.diag(matrix)
    total = float(matrix.sum())
    present = support > 0
    recall = np.where(present, hits / np.where(present, support, 1.0), np.nan)
    f1 = np.where(
        support + predicted > 0, 2.0 * hits / np.where(support + predicted > 0, support + predicted, 1.0), 0.0
    )
    agreement = float(hits.sum()) / total
    chance = float((support * predicted).sum()) / total**2
    return {
        "recall": recall.tolist(),
        "balanced_accuracy": float(np.nanmean(recall)),
        "macro_f1": float(f1[present].mean()),
        "kappa": (agreement - chance) / (1.0 - chance) if chance < 1.0 else 0.0,
        "accuracy": agreement,
    }


def patient_strata(
    targets: NDArray[np.int64], patients: NDArray[np.str_], n_classes: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The patient of every example and the stratum of every patient.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``index`` ``(n,)``: the patient of each example, as a position in the
        sorted patients. ``stratum`` ``(n_patients,)``: the rarest class that the
        patient holds.
    """
    targets = np.asarray(targets)
    _, index = np.unique(np.asarray(patients), return_inverse=True)
    held = np.zeros((int(index.max()) + 1, n_classes), dtype=bool)
    held[index, targets] = True
    holders = held.sum(axis=0)
    examples = np.bincount(targets, minlength=n_classes)
    order = np.asarray(sorted(range(n_classes), key=lambda label: (holders[label], examples[label], label)))
    rank = np.empty(n_classes, dtype=np.int64)
    rank[order] = np.arange(n_classes)
    return index.astype(np.int64), order[np.where(held, rank[None, :], n_classes).min(axis=1)]


def bootstrap_multiplicities(stratum: NDArray[np.int64], n_draws: int, seed: int) -> NDArray[np.int32]:
    """How often each patient is in each draw; a draw resamples the patients of every stratum with replacement.

    Returns
    -------
    numpy.ndarray
        ``(n_draws, n_patients)``; every draw holds as many patients of a stratum as the data.
    """
    rng = np.random.default_rng(seed)
    stratum = np.asarray(stratum)
    out = np.zeros((n_draws, stratum.size), dtype=np.int32)
    rows = np.arange(n_draws)[:, None]
    for value in np.unique(stratum):
        members = np.flatnonzero(stratum == value)
        picks = members[rng.integers(0, members.size, size=(n_draws, members.size))]
        np.add.at(out, (np.broadcast_to(rows, picks.shape), picks), 1)
    return out


def recall_draws(
    targets: NDArray[np.int64],
    predictions: NDArray[np.int64],
    patient_index: NDArray[np.int64],
    multiplicities: NDArray[np.int32],
    n_classes: int,
) -> NDArray[np.float64]:
    """The recall of every class in every draw.

    Returns
    -------
    numpy.ndarray
        ``(n_draws, n_classes)``; ``nan`` where a draw holds no example of a class.
        The balanced accuracy of a draw is the mean of its row without ``nan``.
    """
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    total = np.zeros((multiplicities.shape[1], n_classes), dtype=np.float64)
    correct = np.zeros_like(total)
    np.add.at(total, (patient_index, targets), 1.0)
    hit = predictions == targets
    np.add.at(correct, (patient_index[hit], targets[hit]), 1.0)
    weights = multiplicities.astype(np.float64)
    count = weights @ total
    return np.where(count > 0, (weights @ correct) / np.where(count > 0, count, 1.0), np.nan)


def interval(values: NDArray[np.float64]) -> dict[str, float]:
    """The 2.5% and 97.5% points of draws, and half the distance between them; a ``nan`` draw is left out."""
    low, high = np.nanpercentile(np.asarray(values, dtype=np.float64), [2.5, 97.5])
    return {"low": float(low), "high": float(high), "half_width": float(high - low) / 2.0}

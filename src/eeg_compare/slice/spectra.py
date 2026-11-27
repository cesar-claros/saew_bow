"""Bag of spectra: k-means on unit-norm amplitude spectra, then one token per window (slice protocol, section 6).

The spectra come from ``eeg_compare.slice.features.window_spectra``. The model
is fitted on the windows of the exposure plan of a fold. A context then gets one
token per channel and pooling window: the nearest center. A window without
signal gets the null token, as a window without a positive response does in a
dictionary arm.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

__all__ = ["fit_spectra_bag", "spectra_tokens"]


def fit_spectra_bag(spectra: torch.Tensor, n_clusters: int, seed: int, max_iter: int = 100) -> KMeans:
    """Fit the centers on spectra ``(..., F)``; a zero spectrum does not enter.

    The fit runs with one thread. With several threads, k-means adds the partial
    sums of its threads in an order that is not fixed, so two fits of the same
    data can differ in the last digits and then in the tokens of a few windows.

    Parameters
    ----------
    spectra : torch.Tensor
        Unit-norm spectra of training windows; the last axis is the frequency.
    n_clusters : int
        Centers ``K``.
    seed : int
        Seed of the start of k-means (one start, ``k-means++``).
    max_iter : int
        Largest number of passes.

    Raises
    ------
    ValueError
        If fewer windows with signal exist than centers.
    """
    values = spectra.reshape(-1, spectra.shape[-1]).numpy()
    values = values[values.any(axis=1)]
    if values.shape[0] < n_clusters:
        raise ValueError(f"{values.shape[0]} windows with signal cannot give {n_clusters} centers")
    with threadpool_limits(limits=1):
        return KMeans(n_clusters=n_clusters, n_init=1, random_state=seed, max_iter=max_iter).fit(values)


def spectra_tokens(model: KMeans, spectra: torch.Tensor) -> torch.Tensor:
    """The token of every window: the nearest center, and ``-1`` for a zero spectrum.

    Parameters
    ----------
    model : sklearn.cluster.KMeans
        A fitted model.
    spectra : torch.Tensor
        ``(n, M, C, F)`` unit-norm spectra.

    Returns
    -------
    torch.Tensor
        ``(n, M, C)`` center indices.
    """
    values = spectra.reshape(-1, spectra.shape[-1]).numpy()
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    signal = values.any(axis=1)
    if signal.any():
        labels[signal] = model.predict(values[signal])
    return torch.from_numpy(labels).view(spectra.shape[:-1])

"""The two heads of the slice, with fixed settings and no search (slice protocol, section 8).

Both heads read the same features, centered and scaled with the training part
of a fold, and use one weight per example: inside a class every patient has the
same total weight, and every class has the same total weight.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from eeg_compare.slice.evaluate import example_weights

__all__ = [
    "example_weights",
    "forest_probabilities",
    "logistic_probabilities",
    "prior_probabilities",
    "standardize",
]


def standardize(
    train: NDArray[np.float64], test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Center and scale both parts with the mean and the standard deviation of the training part.

    A feature that is constant in the training part is centered and not scaled.
    """
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _spread(probabilities: NDArray[np.float64], classes: NDArray[np.int64], n_classes: int) -> NDArray[np.float64]:
    """Put the columns of a model at their classes; a class that the training part lacks gets zero."""
    out = np.zeros((probabilities.shape[0], n_classes), dtype=np.float64)
    out[:, np.asarray(classes, dtype=np.int64)] = probabilities
    return out


def logistic_probabilities(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    w_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    n_classes: int,
    c: float = 1.0,
    max_iter: int = 2000,
) -> tuple[NDArray[np.float64], int]:
    """Multinomial logistic regression with an L2 penalty.

    Returns
    -------
    tuple[numpy.ndarray, int]
        The class probabilities ``(n_test, n_classes)`` and the passes of the solver.
        The passes equal ``max_iter`` when the solver did not converge.
    """
    model = LogisticRegression(C=c, max_iter=max_iter, tol=1e-6)
    model.fit(x_train, y_train, sample_weight=w_train)
    return _spread(model.predict_proba(x_test), model.classes_, n_classes), int(np.max(model.n_iter_))


def forest_probabilities(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    w_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    n_classes: int,
    seed: int,
    trees: int = 500,
    min_leaf: int = 2,
    n_jobs: int = -1,
) -> NDArray[np.float64]:
    """A random forest; the class probabilities ``(n_test, n_classes)``."""
    model = RandomForestClassifier(n_estimators=trees, min_samples_leaf=min_leaf, random_state=seed, n_jobs=n_jobs)
    model.fit(x_train, y_train, sample_weight=w_train)
    return _spread(model.predict_proba(x_test), model.classes_, n_classes)


def prior_probabilities(y_train: NDArray[np.int64], n_test: int, n_classes: int) -> NDArray[np.float64]:
    """The class shares of the training part, for every test example."""
    shares = np.bincount(np.asarray(y_train), minlength=n_classes).astype(np.float64)
    return np.tile(shares / shares.sum(), (n_test, 1))

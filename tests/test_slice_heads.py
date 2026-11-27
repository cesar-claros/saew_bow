"""Heads and evaluator of the slice: weights, the two heads, scores and the bootstrap over patients; run on the HPC."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import torch

from eeg_compare.slice.evaluate import (
    bootstrap_multiplicities,
    confusion,
    example_weights,
    interval,
    patient_strata,
    recall_draws,
    summarize,
)
from eeg_compare.slice.heads import forest_probabilities, logistic_probabilities, prior_probabilities, standardize
from eeg_compare.slice.spectra import fit_spectra_bag


def _blobs(seed: int, per_class: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Three classes that are far apart in four features."""
    rng = np.random.default_rng(seed)
    centers = np.array([[6.0, 0.0, 0.0, 0.0], [0.0, 6.0, 0.0, 0.0], [0.0, 0.0, 6.0, 0.0]])
    targets = np.repeat(np.arange(3), per_class)
    return centers[targets] + rng.standard_normal((targets.size, 4)), targets.astype(np.int64)


def test_weights_make_patients_equal_inside_a_class_and_classes_equal() -> None:
    targets = np.array([0, 0, 0, 1, 1, 2])
    patients = np.array(["a", "a", "b", "c", "c", "c"])
    weights = example_weights(targets, patients, 3)
    np.testing.assert_allclose(weights, [0.5, 0.5, 1.0, 1.0, 1.0, 2.0])
    assert weights.mean() == pytest.approx(1.0)
    assert [weights[targets == label].sum() for label in range(3)] == pytest.approx([2.0, 2.0, 2.0])
    # a class without examples takes no share
    np.testing.assert_allclose(example_weights(targets, patients, 5), weights)
    for bad_targets, bad_patients in (
        (np.array([0, 3]), np.array(["a", "b"])),
        (targets, patients[:5]),
        (targets[:0], patients[:0]),
    ):
        with pytest.raises(ValueError):
            example_weights(bad_targets, bad_patients, 3)


def test_features_are_scaled_with_the_training_part_only() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(5.0, 3.0, (50, 3))
    train[:, 2] = 7.0
    test = rng.normal(-2.0, 1.0, (20, 3))
    scaled_train, scaled_test = standardize(train, test)
    np.testing.assert_allclose(scaled_train[:, :2].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled_train[:, :2].std(axis=0), 1.0)
    np.testing.assert_allclose(scaled_train[:, 2], 0.0)
    np.testing.assert_allclose(scaled_test, (test - train.mean(axis=0)) / np.array([*train[:, :2].std(axis=0), 1.0]))
    assert np.array_equal(standardize(train, test + 100.0)[0], scaled_train)


def test_both_heads_separate_easy_classes_and_keep_a_column_for_an_absent_class() -> None:
    (x_train, y_train), (x_test, y_test) = _blobs(0), _blobs(1)
    weights = np.ones(y_train.size)
    found, passes = logistic_probabilities(x_train, y_train, weights, x_test, 5)
    assert found.shape == (y_test.size, 5) and 0 < passes < 2000
    np.testing.assert_allclose(found.sum(axis=1), 1.0)
    assert not found[:, 3:].any() and np.array_equal(found.argmax(axis=1), y_test)
    forest = forest_probabilities(x_train, y_train, weights, x_test, 5, seed=0, trees=20, n_jobs=1)
    assert forest.shape == (y_test.size, 5) and not forest[:, 3:].any()
    assert np.array_equal(forest.argmax(axis=1), y_test)
    assert np.array_equal(
        forest_probabilities(x_train, y_train, weights, x_test, 5, seed=0, trees=20, n_jobs=1), forest
    )
    # the weight of an example enters the fit: a class with no weight is not predicted
    silent = np.where(y_train == 1, 1e-9, 1.0)
    quiet, _ = logistic_probabilities(x_train, y_train, silent, x_test, 3)
    assert quiet[y_test == 1][:, 1].mean() < found[y_test == 1][:, 1].mean()
    prior = prior_probabilities(np.array([0, 0, 0, 2]), 4, 3)
    np.testing.assert_allclose(prior, np.tile([0.75, 0.0, 0.25], (4, 1)))
    assert (prior.argmax(axis=1) == 0).all()


def test_the_logistic_fit_and_k_means_run_with_one_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    @contextmanager
    def limits(**arguments: Any) -> Iterator[None]:
        calls.append(arguments)
        yield

    monkeypatch.setattr("eeg_compare.slice.heads.threadpool_limits", limits)
    monkeypatch.setattr("eeg_compare.slice.spectra.threadpool_limits", limits)
    features, targets = _blobs(0)
    logistic_probabilities(features, targets, np.ones(targets.size), features, 3)
    fit_spectra_bag(torch.from_numpy(np.abs(features).astype(np.float32)), 3, seed=0)
    assert calls == [{"limits": 1}, {"limits": 1}]


def test_two_fits_of_k_means_on_the_same_data_are_equal() -> None:
    spectra = torch.rand((50000, 30), generator=torch.Generator().manual_seed(4))
    first = fit_spectra_bag(spectra, 8, seed=1, max_iter=20)
    second = fit_spectra_bag(spectra, 8, seed=1, max_iter=20)
    assert np.array_equal(first.cluster_centers_, second.cluster_centers_)
    assert np.array_equal(first.labels_, second.labels_)
    assert not np.array_equal(fit_spectra_bag(spectra, 8, seed=2, max_iter=20).cluster_centers_, first.cluster_centers_)


def test_the_scores_of_a_confusion_matrix() -> None:
    targets = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    predictions = np.array([0, 0, 0, 1, 1, 1, 2, 2, 0, 0])
    matrix = confusion(targets, predictions, 3)
    np.testing.assert_array_equal(matrix, [[3, 1, 0], [0, 2, 0], [2, 0, 2]])
    scores = summarize(matrix)
    assert scores["recall"] == pytest.approx([0.75, 1.0, 0.5])
    assert scores["balanced_accuracy"] == pytest.approx(0.75)
    assert scores["accuracy"] == pytest.approx(0.7)
    assert scores["macro_f1"] == pytest.approx((6 / 9 + 4 / 5 + 4 / 6) / 3)
    assert scores["kappa"] == pytest.approx((0.7 - 0.34) / 0.66)
    # a class without examples does not enter the balanced accuracy
    wider = summarize(confusion(targets, predictions, 4))
    assert np.isnan(wider["recall"][3]) and wider["balanced_accuracy"] == pytest.approx(0.75)
    assert wider["macro_f1"] == pytest.approx(scores["macro_f1"])
    # a weight per example
    weights = np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    weighted = summarize(confusion(targets, predictions, 3, weights))
    assert weighted["recall"] == pytest.approx([0.5, 1.0, 0.5])
    assert weighted["balanced_accuracy"] == pytest.approx(2.0 / 3.0)
    # a prediction of one class for all gives chance
    constant = summarize(confusion(targets, np.zeros(10, dtype=np.int64), 3))
    assert constant["balanced_accuracy"] == pytest.approx(1.0 / 3.0) and constant["kappa"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="one prediction per target"):
        confusion(targets, predictions[:5], 3)


def test_the_stratum_of_a_patient_is_the_rarest_class_that_it_holds() -> None:
    #          a  a  b  b  c  c  d  e  e
    targets = np.array([0, 1, 0, 2, 0, 2, 0, 0, 3])
    patients = np.array(["a", "a", "b", "b", "c", "c", "d", "e", "e"])
    index, stratum = patient_strata(targets, patients, 4)
    np.testing.assert_array_equal(index, [0, 0, 1, 1, 2, 2, 3, 4, 4])
    # class 0 has five patients, class 2 has two, classes 1 and 3 have one each
    assert stratum[1] == 2 and stratum[2] == 2 and stratum[3] == 0
    assert {int(stratum[0]), int(stratum[4])} == {1, 3}
    # equal numbers of patients: the class with fewer examples is rarer
    targets = np.array([0, 1, 1, 0, 1, 1])
    patients = np.array(["a", "a", "a", "b", "b", "b"])
    np.testing.assert_array_equal(patient_strata(targets, patients, 2)[1], [0, 0])


def test_a_draw_resamples_the_patients_of_every_stratum() -> None:
    stratum = np.array([1, 2, 2, 0, 0, 0])
    draws = bootstrap_multiplicities(stratum, 300, seed=0)
    assert draws.shape == (300, 6) and draws.min() >= 0
    assert (draws[:, 0] == 1).all() and (draws[:, 1:3].sum(axis=1) == 2).all() and (draws[:, 3:].sum(axis=1) == 3).all()
    assert len({tuple(row) for row in draws.tolist()}) > 10
    assert np.array_equal(bootstrap_multiplicities(stratum, 300, seed=0), draws)
    assert not np.array_equal(bootstrap_multiplicities(stratum, 300, seed=1), draws)


def test_the_recall_of_a_draw_equals_a_direct_count() -> None:
    rng = np.random.default_rng(0)
    patients = rng.integers(0, 12, 300).astype(str)
    targets = rng.integers(0, 3, 300)
    predictions = np.where(rng.random(300) < 0.6, targets, rng.integers(0, 3, 300))
    index, stratum = patient_strata(targets, patients, 3)
    draws = bootstrap_multiplicities(stratum, 20, seed=0)
    found = recall_draws(targets, predictions, index, draws, 3)
    assert found.shape == (20, 3)
    for number in range(20):
        rows = np.repeat(np.arange(300), draws[number][index])
        direct = summarize(confusion(targets[rows], predictions[rows], 3))
        assert found[number] == pytest.approx(direct["recall"])
        assert np.nanmean(found[number]) == pytest.approx(direct["balanced_accuracy"])
    # a draw of every patient once gives the scores of the data
    once = recall_draws(targets, predictions, index, np.ones((1, stratum.size), dtype=np.int32), 3)
    assert once[0] == pytest.approx(summarize(confusion(targets, predictions, 3))["recall"])
    # a draw without an example of a class has no recall for it
    twice = np.array([[2, 0]], dtype=np.int32)
    lone = recall_draws(np.array([0, 0, 1]), np.array([0, 1, 1]), np.array([0, 0, 1]), twice, 2)
    assert lone[0, 0] == pytest.approx(0.5) and np.isnan(lone[0, 1])


def test_an_interval_is_the_central_95_percent_of_the_draws() -> None:
    found = interval(np.arange(101.0))
    assert found == {"low": pytest.approx(2.5), "high": pytest.approx(97.5), "half_width": pytest.approx(47.5)}
    assert interval(np.array([np.nan, *np.arange(101.0)]))["half_width"] == pytest.approx(47.5)

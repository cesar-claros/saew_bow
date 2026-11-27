"""The profile input of the slice and the reader of plan units; run on the HPC."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch
from numpy.typing import NDArray

from eeg_compare.data.config import PreprocessingConfig
from eeg_compare.data.dataset import Partition, TUEVDataset
from eeg_compare.data.manifest import load_manifest
from eeg_compare.data.recordings import Recording
from eeg_compare.slice.exposure import SOURCES, ExposureConfig, build_plans, fold_units
from eeg_compare.slice.scaling import ChannelProfile, fit_profile, profile_factors, read_scales
from eeg_compare.slice.signals import load_units, study_clip

_FACTORS = np.log(np.array([1.0, 1.5, 0.7, 1.2]))
PROFILE = _FACTORS - _FACTORS.mean()
UNIT_SCHEMA = {"recording_id": pl.String, "start_sample": pl.Int64, "n_samples": pl.Int64}


def _scales(count: int = 6) -> dict[str, NDArray[np.float64]]:
    """Recordings with different levels that all follow one channel profile."""
    level = np.random.default_rng(0).normal(-11.0, 0.5, count)
    return {f"r{index}": np.exp(level[index] + PROFILE) for index in range(count)}


class _Reader:
    """Two recordings of 2 channels and 100 samples, read-only as the reader of the study gives them."""

    def __init__(self) -> None:
        ramp = np.arange(200, dtype=np.float32).reshape(2, 100)
        self.data = {"a": ramp, "b": -2.0 * ramp}
        for value in self.data.values():
            value.setflags(write=False)
        self.calls: list[str] = []

    def recording(self, recording_id: str) -> NDArray[np.float32]:
        self.calls.append(recording_id)
        return self.data[recording_id]


def _table(rows: list[tuple[str, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=UNIT_SCHEMA, orient="row")


def _ramp(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    """A signal of whole numbers that differs by recording, channel and sample, so a wrong offset or row shows."""
    index = int(Path(recording.edf_path).stem.split("-")[1])
    samples = np.arange(recording.n_samples, dtype=np.float32)[None, :]
    channels = np.arange(len(config.channels), dtype=np.float32)[:, None]
    return samples + 5000.0 * channels + 110000.0 * index


def test_the_profile_is_the_mean_channel_part_of_the_training_recordings() -> None:
    scales = _scales()
    fitted = fit_profile(scales, ["r0", "r1", "r2"])
    assert np.allclose(fitted.profile, PROFILE) and abs(fitted.profile.sum()) < 1e-12 and fitted.cap == 3.0
    # a recording outside the training part does not change the profile
    changed = {**scales, "r5": scales["r5"] * np.array([1.0, 9.0, 1.0, 1.0])}
    assert np.array_equal(fit_profile(changed, ["r0", "r1", "r2"]).profile, fitted.profile)
    mixed = fit_profile(changed, ["r4", "r5"])
    assert np.allclose(mixed.profile, PROFILE + 0.5 * (np.log([1.0, 9.0, 1.0, 1.0]) - np.log(9.0) / 4))
    with pytest.raises(ValueError, match="training recordings"):
        fit_profile(scales, [])
    with pytest.raises(ValueError, match="cap"):
        fit_profile(scales, ["r0"], cap=1.0)
    with pytest.raises(ValueError, match="no scale"):
        fit_profile(scales, ["r0", "absent"])


def test_the_factor_keeps_a_recording_that_follows_the_profile_and_caps_a_loud_channel() -> None:
    scales = _scales()
    fitted = fit_profile(scales, scales)
    assert np.allclose(fitted.factor(scales["r3"]), 1.0)
    assert np.allclose(fitted.factor(scales["r3"] * 7.0), 1.0)
    louder = scales["r3"] * np.array([1.0, 1.0, 2.0, 1.0])
    assert np.allclose(fitted.factor(louder), [2**-0.25, 2**-0.25, 2**0.75, 2**-0.25])
    # the profile input is the signal over one recording scale times the profile
    raw = np.random.default_rng(1).standard_normal((4, 50)) * louder[:, None]
    product = np.exp(np.log(louder).mean() + fitted.profile)
    assert np.allclose((raw / louder[:, None]) * fitted.factor(louder)[:, None], raw / product[:, None])
    # a channel counts as at most `cap` times as loud as its profile says
    loud = scales["r3"] * np.array([1.0, 1.0, 20.0, 1.0])
    assert np.allclose(fitted.factor(loud), [20**-0.25, 20**-0.25, 3.0, 20**-0.25])
    assert np.isclose(ChannelProfile(fitted.profile, cap=10.0).factor(loud)[2], 20**0.75)
    for bad in (np.ones(3), np.array([1.0, 0.0, 1.0, 1.0])):
        with pytest.raises(ValueError, match="positive"):
            fitted.factor(bad)
    factors = profile_factors(scales, fitted, ["r3", "r1", "r3"])
    assert list(factors) == ["r1", "r3"] and np.allclose(factors["r1"], 1.0)


def test_scales_come_from_a_report_of_the_same_channels_and_band(tmp_path: Path) -> None:
    def report(name: str, **changes: Any) -> Path:
        content = {
            "band_hz": [1.0, 30.0],
            "measure_1_profile": {"profile_factor_by_channel": {"a": 1.0, "b": 1.2}},
            "scales": [
                {"recording_id": "r0", "cluster": 0, "scale": [1e-5, 2e-5]},
                {"recording_id": "r1", "cluster": 1, "scale": [3e-5, 1e-5]},
            ],
            **changes,
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        return path

    scales = read_scales(report("good"), ["a", "b"], [1.0, 30.0])
    assert list(scales) == ["r0", "r1"] and np.array_equal(scales["r1"], [3e-5, 1e-5])
    with pytest.raises(ValueError, match="other channels"):
        read_scales(report("order"), ["b", "a"], [1.0, 30.0])
    with pytest.raises(ValueError, match="another band"):
        read_scales(report("band"), ["a", "b"], [None, 30.0])
    for name, scale in (("zero", [1e-5, 0.0]), ("short", [1e-5])):
        with pytest.raises(ValueError, match="not one positive value"):
            read_scales(report(name, scales=[{"recording_id": "r0", "scale": scale}]), ["a", "b"], [1.0, 30.0])


def test_the_slice_refuses_a_manifest_without_the_recording_scale() -> None:
    base = PreprocessingConfig(
        input_mode="bipolar",
        highpass_hz=1.0,
        lowpass_hz=30.0,
        notch_hz=(),
        normalization="mad",
        normalization_scope="recording",
        clip=50.0,
    )
    assert study_clip(base) == 50.0
    changes: tuple[dict[str, Any], ...] = (
        {"normalization_scope": "example"},
        {"normalization": "zscore"},
        {"clip": None},
    )
    for change in changes:
        with pytest.raises(ValueError, match="robust scale per recording"):
            study_clip(replace(base, **change))


def test_units_come_in_the_order_of_the_table_with_one_read_per_recording() -> None:
    reader = _Reader()
    units = _table([("b", 10, 20), ("a", 0, 20), ("b", 80, 20), ("a", 50, 20)])
    signals = load_units(reader, units)
    assert signals.shape == (4, 2, 20) and signals.dtype == torch.float32 and reader.calls == ["b", "a"]
    for index, (name, start, length) in enumerate(units.iter_rows()):
        assert np.array_equal(signals[index].numpy(), reader.data[name][:, start : start + length])
    cases = (
        (units.clear(), "nonempty"),
        (_table([("a", 0, 20), ("a", 0, 10)]), "one length"),
        (_table([("a", 90, 20)]), "leaves"),
        (_table([("a", -1, 20)]), "leaves"),
    )
    for bad, message in cases:
        with pytest.raises(ValueError, match=message):
            load_units(reader, bad)


def test_the_profile_input_scales_every_channel_and_clips_again() -> None:
    reader = _Reader()
    units = _table([("b", 10, 20), ("a", 50, 20)])
    factors = {"a": np.array([2.0, 0.5]), "b": np.array([1.0, 3.0])}
    plain = load_units(reader, units).numpy()
    scaled = load_units(reader, units, factors).numpy()
    assert np.allclose(scaled[0], plain[0] * factors["b"][:, None])
    assert np.allclose(scaled[1], plain[1] * factors["a"][:, None])
    clipped = load_units(reader, units, factors, clip=50.0).numpy()
    assert np.array_equal(clipped, np.clip(scaled, -50.0, 50.0))
    assert np.abs(scaled).max() > 50.0 and np.abs(clipped).max() == 50.0
    # the signal of the reader stays as it was
    assert np.array_equal(reader.data["a"], np.arange(200, dtype=np.float32).reshape(2, 100))
    with pytest.raises(ValueError, match="no factor"):
        load_units(reader, units, {"a": factors["a"]})
    with pytest.raises(ValueError, match="needs the factors"):
        load_units(reader, units, clip=50.0)
    with pytest.raises(ValueError, match="positive"):
        load_units(reader, units, factors, clip=0.0)


def test_plan_units_are_the_samples_that_the_dataset_reads(
    mini_corpus: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, directory = mini_corpus
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _ramp)
    plan = build_plans(load_manifest(directory), ExposureConfig())
    for fold in range(2):
        dataset = TUEVDataset(root, directory, Partition("train", fold))
        # a context of the plan is the signal of its example
        contexts = fold_units(plan, fold, SOURCES[1])
        signals = load_units(dataset.reader, contexts)
        examples = {dataset.examples["example_id"][index]: dataset[index]["signal"] for index in range(len(dataset))}
        assert set(contexts["unit_id"]) == set(examples) and signals.shape == (contexts.height, 22, 1250)
        for index, unit_id in enumerate(contexts["unit_id"]):
            assert torch.equal(signals[index], examples[unit_id])
        # the intervals of a recording are the recording, cut on the grid
        intervals = fold_units(plan, fold, SOURCES[0])
        for recording_id in intervals["recording_id"].unique():
            own = intervals.filter(pl.col("recording_id") == recording_id).sort("start_sample")
            joined = np.concatenate(list(load_units(dataset.reader, own).numpy()), axis=-1)
            assert own.height == 4 and np.array_equal(joined, dataset.reader.recording(recording_id))

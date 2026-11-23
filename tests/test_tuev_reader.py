"""EDF mapping, native-rate processing and normalization fixtures for the HPC."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from eeg_compare.data.config import PreprocessingConfig
from eeg_compare.data.recordings import (
    MissingChannelsError,
    Recording,
    RecordingIdentity,
    channel_plan,
    inspect_recording,
    normalize_signal,
    read_recording,
    resampled_length,
)


def test_channel_mapping_preserves_names_aliases_order_and_polarity() -> None:
    config = PreprocessingConfig(channels=("FP1-F7", "F7-T3"))
    plan = channel_plan(["EEG T7-REF", "EEG FP1-REF", "EEG F7-REF", "ECG"], config)
    assert plan.source_channels == ("EEG T7-REF", "EEG FP1-REF", "EEG F7-REF")
    np.testing.assert_array_equal(np.asarray(plan.matrix) @ np.array([1.0, 5.0, 2.0]), [3.0, 1.0])
    bipolar = channel_plan(["F7-FP1", "EEG F7-T3"], replace(config, input_mode="bipolar"))
    np.testing.assert_array_equal(np.asarray(bipolar.matrix) @ np.array([-3.0, 1.0]), [3.0, 1.0])


@pytest.mark.parametrize(
    "names",
    [
        ["FP1-REF", "FP1-REF", "F7-REF"],
        ["FP1-REF", "F7-LE"],
        ["FP1-REF", "F7"],
    ],
)
def test_ambiguous_or_mixed_reference_inputs_fail(names: list[str]) -> None:
    with pytest.raises(ValueError):
        channel_plan(names, PreprocessingConfig(channels=("FP1-F7",)))
    with pytest.raises(MissingChannelsError):
        channel_plan(["FP1-REF"], PreprocessingConfig(channels=("FP1-F7",)))


class _Raw:
    def __init__(self, *, allow_data: bool = False) -> None:
        self.ch_names = ["EEG FP1-REF", "EEG F7-REF"]
        self.info = {"sfreq": 250.0}
        self.n_times = 1003
        self.allow_data = allow_data

    def get_data(self, *, picks: list[str]) -> NDArray[np.float64]:
        assert self.allow_data, "header inspection must not load signal samples"
        assert picks == self.ch_names
        time = np.arange(self.n_times) / self.info["sfreq"]
        return np.stack([np.sin(2 * np.pi * 10 * time), np.zeros_like(time)])


def test_header_inspection_excludes_samples_and_preserves_source_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edf = tmp_path / "sample.edf"
    edf.write_bytes(b"synthetic EDF header")
    edf.with_suffix(".rec").write_text("0,1,2,1")
    calls = []

    @contextmanager
    def opened(path: Path, *, channels: Sequence[str] | None = None, preload: bool = False) -> Iterator[_Raw]:
        calls.append((channels, preload))
        yield _Raw()

    monkeypatch.setattr("eeg_compare.data.recordings._open_edf", opened)
    config = PreprocessingConfig(channels=("FP1-F7",), highpass_hz=None, notch_hz=())
    result = inspect_recording(tmp_path, RecordingIdentity("sample.edf", "p", "s", "fixture"), "train", config)
    assert result.n_samples == 1003 and result.sfreq == 250
    assert len(result.edf_sha256) == len(result.rec_sha256) == 64
    assert calls == [(None, False), (("EEG FP1-REF", "EEG F7-REF"), False)]


def test_reader_filters_full_record_and_preserves_resampling_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording: Recording
) -> None:
    @contextmanager
    def opened(path: Path, *, channels: Sequence[str] | None = None, preload: bool = False) -> Iterator[_Raw]:
        assert preload and channels == ("EEG FP1-REF", "EEG F7-REF")
        yield _Raw(allow_data=True)

    lengths = []

    def filter_data(
        data: NDArray[np.float64], sfreq: float, low: float, high: float | None, **kwargs: object
    ) -> NDArray[np.float64]:
        lengths.append(data.shape[-1])
        assert sfreq == 250 and low == 0.1 and high is None
        return data

    monkeypatch.setattr("eeg_compare.data.recordings._open_edf", opened)
    monkeypatch.setattr("eeg_compare.data.recordings.mne.filter.filter_data", filter_data)
    source = replace(
        recording, n_samples=1003, source_channels=("EEG FP1-REF", "EEG F7-REF"), channel_matrix=((1.0, -1.0),)
    )
    config = PreprocessingConfig(channels=("FP1-F7",), target_sfreq=256, notch_hz=())
    result = read_recording(tmp_path, source, config)
    assert lengths == [1003] and result.shape == (1, 1028) and result.dtype == np.float32
    expected = np.sin(2 * np.pi * 10 * np.arange(1028) / 256)
    np.testing.assert_allclose(result[0, 32:-32], expected[32:-32], atol=0.005, rtol=0)
    assert resampled_length(1003, 250, 256) == 1028


def test_normalization_has_explicit_units_and_constant_channel_behavior() -> None:
    signal = np.array([[1, 2, 3, 4, 5], [2, 2, 2, 2, 2]], dtype=np.float64) * 1e-6
    none = normalize_signal(signal.copy(), PreprocessingConfig())
    np.testing.assert_array_equal(none, signal)
    standardized = normalize_signal(signal.copy(), PreprocessingConfig(normalization="zscore"))
    np.testing.assert_allclose(standardized[0].mean(), 0, atol=1e-14)
    np.testing.assert_allclose(standardized[0].std(), 1, atol=1e-14)
    np.testing.assert_array_equal(standardized[1], 0)
    scaled = normalize_signal(signal.copy(), PreprocessingConfig(normalization="mad", clip=1.0))
    assert scaled[0, 0] == -1 and scaled[0, -1] == 1
    np.testing.assert_array_equal(scaled[1], 0)


def test_unsupported_filter_rate_fails_before_io(tmp_path: Path, recording: Recording) -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        read_recording(tmp_path, replace(recording, sfreq=100), PreprocessingConfig())

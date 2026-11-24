"""Signal audit script: its NumPy helpers and its wiring to the dataset; run on the HPC."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from eeg_compare.data.config import ExampleConfig, PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.manifest import CorpusLayout, build_manifest
from eeg_compare.data.recordings import Recording

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_signals.py"
EDGES = (1.0, 30.0, 40.0)


@pytest.fixture
def script() -> SimpleNamespace:
    return SimpleNamespace(**runpy.run_path(str(SCRIPT)))


def _band_noise(shape: tuple[int, int], low: float, high: float, seed: int) -> NDArray[np.float64]:
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(shape), axis=-1)
    frequency = np.fft.rfftfreq(shape[-1], 1 / 250.0)
    spectrum[..., (frequency < low) | (frequency > high)] = 0
    signal = np.fft.irfft(spectrum, n=shape[-1], axis=-1)
    return signal / signal.std()


def test_channel_auroc_and_second_energies(script: SimpleNamespace) -> None:
    marked = np.array([True, True, False, False])
    assert script.channel_auroc(np.array([3.0, 2.0, 1.0, 0.5]), marked) == 1.0
    assert script.channel_auroc(np.array([0.1, 0.2, 1.0, 2.0]), marked) == 0.0
    assert script.channel_auroc(np.ones(4), marked) == 0.5
    assert np.isnan(script.channel_auroc(np.ones(3), np.zeros(3, dtype=bool)))
    signal = np.zeros((2, 1250))
    signal[1, 500:750] = 2.0
    energies = script.second_energies(signal, 250)
    assert energies.shape == (2, 5) and energies[1, 2] == 4.0 and energies.sum() == 4.0
    assert script.second_energies(np.ones((1, 1300)), 250).shape == (1, 5)


def test_band_shares_separate_filtered_and_unfiltered_signals(script: SimpleNamespace) -> None:
    sine = np.sin(2 * np.pi * 10 * np.arange(1250) / 250)[None, :]
    assert script.band_shares(sine, 250.0, EDGES)[1] > 0.999
    fast = np.sin(2 * np.pi * 60 * np.arange(1250) / 250)[None, :]
    assert script.band_shares(fast, 250.0, EDGES)[3] > 0.999
    white = np.random.default_rng(0).standard_normal((22, 1250))
    assert script.band_shares(white, 250.0, EDGES)[3] > 0.5
    assert script.band_shares(_band_noise((22, 1250), 1.0, 30.0, seed=1), 250.0, EDGES)[3] < 1e-3
    assert all(np.isnan(value) for value in script.band_shares(np.zeros((2, 1250)), 250.0, EDGES))
    assert script.band_shares(sine, 250.0, ()) == [1.0]


def test_a_planted_event_stands_out_only_in_its_second_and_on_its_channels(script: SimpleNamespace) -> None:
    signal = _band_noise((22, 1250), 1.0, 30.0, seed=2)
    signal[[0, 4], 500:750] += 4 * np.sin(2 * np.pi * 6 * np.arange(250) / 250)
    signal[7] = 0.0
    signal[1, 10] = 25.0
    valid = np.ones(22, dtype=bool)
    valid[7] = False
    measures = script.example_measures(signal, valid, (0, 4), 2, 250.0, 20.0, EDGES)
    assert measures["marked_auroc"][2] == 1.0 and len(measures["marked_auroc"]) == 5
    assert all(value < 1.0 for index, value in enumerate(measures["marked_auroc"]) if index != 2)
    assert measures["second_ratio"] > 0.1 and measures["finite"]
    assert measures["clipped_share"] == pytest.approx(1 / (22 * 1250))
    assert len(measures["labeled_energy"]) == 22 and measures["labeled_energy"][7] == 0.0
    flat_marked = script.example_measures(signal, valid, (7,), 2, 250.0, None, EDGES)
    assert np.isnan(flat_marked["marked_auroc"][2]) and np.isnan(flat_marked["clipped_share"])


def test_selection_and_summary(script: SimpleNamespace) -> None:
    rows = [{"example_id": name, "label": "a" if index % 2 else "b"} for index, name in enumerate("fedcba")]
    assert [row["example_id"] for row in script.select(rows, 1)] == ["a", "b"]
    assert script.select(rows, 0) == rows
    result = script.summary([1.0, 2.0, 3.0, float("nan")])
    assert (result["n"], result["median"], result["mean"]) == (3, 2.0, 2.0)
    assert script.summary([])["n"] == 0 and np.isnan(script.summary([float("nan")])["median"])


def _signal(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    time = np.arange(recording.n_samples) / recording.sfreq
    return np.tile(np.sin(2 * np.pi * 10 * time), (len(config.channels), 1)).astype(np.float32)


def test_the_audit_reads_training_events_and_their_controls(
    script: SimpleNamespace, corpus_files: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, identities = corpus_files
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(input_mode="bipolar", highpass_hz=None, notch_hz=(), lowpass_hz=30.0),
        examples=ExampleConfig(displaced_controls=True, displaced_gap_seconds=0.0),
        splits=SplitConfig(n_folds=3),
    )
    build_manifest(CorpusLayout(root, "synthetic"), identities, config, tmp_path / "manifest")
    monkeypatch.setattr("eeg_compare.data.dataset.read_recording", _signal)
    output = tmp_path / "audit.json"
    arguments = ["--root", str(root), "--manifest", str(tmp_path / "manifest"), "--output", str(output)]
    status = script.main([*arguments, "--limit-per-class", "2"])
    report = json.loads(output.read_text())
    assert report["selected_examples"] == 12 and report["available_examples"] == 18
    assert report["band_edges_hz"] == [30.0, 40.0] and report["labeled_second"] == 2
    for label in ("spsw", "gped", "pled", "eyem", "artf", "bckg"):
        data = report["classes"][label]
        assert data["n"] == 2 and data["all_finite"] and data["control_marked_auroc"]["n"] == 2
        assert data["band_share_medians"][0] > 0.999 and data["band_share_medians"][2] < 1e-6
        assert data["marked_auroc_by_second"] == [0.5] * 5 and data["control_marked_auroc"]["mean"] == 0.5
        assert data["second_ratio"]["median"] == pytest.approx(0.0, abs=1e-6)
    checks = {row["check"].split(":")[0]: row["ok"] for row in report["checks"]}
    assert checks["median share of power above 40 Hz"] and checks["all samples are finite"]
    assert not checks["eyem"] and not checks["artf"] and status == 1 and not report["ok"]
    assert script.main(arguments) == 2
    assert (
        script.main(
            [
                "--root",
                str(root),
                "--manifest",
                str(tmp_path / "manifest"),
                "--output",
                str(tmp_path / "manifest" / "in.json"),
            ]
        )
        == 2
    )

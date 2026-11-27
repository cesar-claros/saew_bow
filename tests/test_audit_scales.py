"""Scale audit script: its NumPy helpers, its fixed rules and its wiring to a manifest; run on the HPC."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from eeg_compare.data.config import PipelineConfig, PreprocessingConfig, SplitConfig
from eeg_compare.data.manifest import CorpusLayout, build_manifest
from eeg_compare.data.recordings import Recording, normalize_signal

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_scales.py"
PROFILE = np.array([1.0, 1.5, 0.7, 1.2])


@pytest.fixture
def script() -> SimpleNamespace:
    return SimpleNamespace(**runpy.run_path(str(SCRIPT)))


def test_the_scale_is_that_of_the_pipeline(script: SimpleNamespace) -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((5, 1001)) * np.array([[1.0], [2.0], [5.0], [0.1], [3.0]]) + np.arange(5)[:, None]
    data[3] = 0.25
    scale = script.channel_scales(data)
    assert scale.shape == (5,) and scale[3] == 1e-12 and 0.8 < scale[0] < 1.2 and 4.0 < scale[2] < 6.0
    config = PreprocessingConfig(
        input_mode="bipolar", highpass_hz=None, notch_hz=(), normalization="mad", normalization_scope="recording"
    )
    center = np.partition(data, 500, axis=-1)[:, 500:501]
    assert np.allclose(normalize_signal(data.copy(), config), (data - center) / scale[:, None])
    with pytest.raises(ValueError, match="nonempty"):
        script.channel_scales(np.zeros((3, 0)))


def test_far_samples_keep_the_gap_from_every_mark(script: SimpleNamespace) -> None:
    far = script.far_mask(1000, 100.0, [(3.0, 4.0), (8.5, 9.0)], 1.0)
    assert far.sum() == 200 + 250 and far[199] and not far[200] and not far[499] and far[500] and not far[999]
    assert not script.far_mask(1000, 100.0, [(0.2, 0.5)], 1.0)[:150].any()
    assert script.far_mask(50, 100.0, [], 1.0).all()


def test_the_log_scale_splits_into_recording_channel_and_rest(script: SimpleNamespace) -> None:
    rng = np.random.default_rng(1)
    level = rng.normal(-11.0, 0.5, 30)
    profile = np.log(PROFILE) - np.log(PROFILE).mean()
    extra = rng.normal(0.0, 0.2, (30, 4))
    extra = extra - extra.mean(axis=0) - extra.mean(axis=1, keepdims=True) + extra.mean()
    found = script.two_way(level[:, None] + profile[None, :] + extra)
    assert np.allclose(found[0], level) and np.allclose(found[1], profile) and np.allclose(found[2], extra)


def _collected(scale: NDArray[np.float64], far: NDArray[np.float64]) -> dict[str, Any]:
    """Rows of 4 channels; channel 1 holds marks of ``pled`` and channel 2 marks of ``bckg`` in every recording."""
    rows = [
        {
            "recording_id": f"recording-{index}",
            "cluster": index // 2,
            "seconds": 600.0,
            "far_seconds": 500.0,
            "scale": scale[index].tolist(),
            "far_scale": far[index].tolist(),
            "marked": {"pled": [1], "bckg": [2]},
        }
        for index in range(scale.shape[0])
    ]
    return {
        "rows": rows,
        "channels": ["a", "b", "c", "d"],
        "normalization": ["mad", "recording"],
        "band_hz": [1.0, 30.0],
        "gap_seconds": 10.0,
        "min_far_seconds": 30.0,
        "available": scale.shape[0],
    }


def test_the_fixed_rules_give_the_three_outcomes(script: SimpleNamespace) -> None:
    rng = np.random.default_rng(2)
    exact = np.exp(rng.normal(-11.0, 0.5, 30))[:, None] * PROFILE[None, :]
    report = script.build_report(_collected(exact, exact))
    first = report["measure_1_profile"]
    assert report["outcome"] == "per_channel" and first["passes"] and first["rest_sd"] < 1e-9
    factors = np.array(list(first["profile_factor_by_channel"].values()))
    assert np.allclose(factors / factors[0], PROFILE / PROFILE[0])
    assert report["measure_3_loud_channels"]["share_of_channels_above"] == 0.0

    noisy = exact * np.exp(rng.normal(0.0, 0.3, (30, 4)))
    report = script.build_report(_collected(noisy, noisy))
    classes = report["measure_2_events"]["classes"]
    assert not report["measure_1_profile"]["passes"] and report["measure_1_profile"]["rest_sd"] > 0.15
    assert report["outcome"] == "per_channel_with_profile_level" and report["measure_2_events"]["passes"] is True
    assert classes["pled"]["marked_channels"] == 30 and classes["pled"]["clusters"] == 15 and classes["pled"]["passes"]
    assert classes["pled"]["ratio_all_to_far"]["median"] == 1.0 and not classes["spsw"]["evaluated"]
    assert classes["spsw"]["passes"] is None and classes["spsw"]["marked_channels"] == 0

    far = noisy.copy()
    far[:, 1] /= 2.0
    report = script.build_report(_collected(noisy, far))
    classes = report["measure_2_events"]["classes"]
    assert report["outcome"] == "profile_primary" and report["measure_2_events"]["passes"] is False
    assert classes["pled"]["ratio_all_to_far"]["median"] == pytest.approx(2.0) and classes["pled"]["passes"] is False
    assert classes["bckg"]["passes"] is True and classes["bckg"]["ratio_all_to_far"]["median"] == pytest.approx(1.0)
    assert classes["pled"]["control_ratio_on_unmarked_channels"]["n"] == 60
    assert classes["pled"]["control_ratio_on_unmarked_channels"]["median"] == pytest.approx(1.0)

    few = script.build_report(_collected(noisy, far), min_units=100)
    assert few["outcome"] == "undecided_no_class_with_enough_marked_channels"
    assert few["measure_2_events"]["passes"] is None
    with pytest.raises(ValueError, match="at least two"):
        script.build_report(_collected(exact[:1], exact[:1]))


def test_a_loud_channel_and_a_scale_without_far_samples(script: SimpleNamespace) -> None:
    exact = np.exp(np.random.default_rng(3).normal(-11.0, 0.5, 30))[:, None] * PROFILE[None, :]
    loud = exact.copy()
    loud[0, 0] *= 10.0
    collected = _collected(loud, loud)
    collected["rows"][5]["far_scale"] = None
    report = script.build_report(collected)
    third = report["measure_3_loud_channels"]
    assert third["share_of_channels_above"] == pytest.approx(1 / 120) and third["share_of_channels_below"] == 0.0
    assert (
        third["share_of_recordings_with_one"] == pytest.approx(1 / 30) and third["channels_above_by_channel"]["a"] == 1
    )
    assert report["measure_2_events"]["classes"]["pled"]["marked_channels"] == 29, "a recording without far samples"
    text = script.render(report)
    assert text.startswith("Scale audit of 30 of 30 training recordings") and "Outcome:" in text
    json.dumps(script._json_safe(report))


def _signal(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    """A 10 Hz wave with a channel profile; channel 0 is 5 times larger inside the six marked seconds."""
    assert config.normalization == "none" and config.clip is None, "the audit must read the signal without the scale"
    wave = np.sin(2 * np.pi * 10 * np.arange(recording.n_samples) / recording.sfreq)
    data = np.stack([(1 + 0.1 * channel) * wave for channel in range(len(config.channels))])
    for second in (3, 5, 7, 9, 11, 13):
        data[0, round(second * recording.sfreq) : round((second + 1) * recording.sfreq)] *= 5.0
    return (1e-5 * data).astype(np.float32)


def test_the_audit_reads_the_training_recordings_without_the_scale(
    script: SimpleNamespace, corpus_files: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, identities = corpus_files
    config = PipelineConfig(
        preprocessing=PreprocessingConfig(
            input_mode="bipolar",
            highpass_hz=None,
            notch_hz=(),
            lowpass_hz=30.0,
            normalization="mad",
            normalization_scope="recording",
            clip=50.0,
        ),
        splits=SplitConfig(n_folds=3),
    )
    build_manifest(CorpusLayout(root, "synthetic"), identities, config, tmp_path / "manifest")
    monkeypatch.setattr("eeg_compare.data.recordings.read_recording", _signal)
    output = tmp_path / "scales.json"
    common = ["--root", str(root), "--manifest", str(tmp_path / "manifest")]
    options = ["--gap-seconds", "1", "--min-far-seconds", "2", "--min-units", "3"]
    assert script.main([*common, "--output", str(output), *options]) == 0
    text = output.read_text()
    report = json.loads(text)
    assert report["recordings_read"] == 3 and report["recordings_available"] == 3, "the evaluation recording is closed"
    assert "patient" not in text and "session" not in text, "the report must hold no patient or session identifier"
    assert report["measure_1_profile"]["passes"] and report["outcome"] == "per_channel"
    assert report["measure_1_profile"]["rest_sd"] < 1e-6
    for label in ("spsw", "gped", "pled", "eyem", "artf", "bckg"):
        entry = report["measure_2_events"]["classes"][label]
        assert entry["marked_channels"] == 3 and entry["recordings"] == 3 and entry["evaluated"] and entry["passes"]
        assert 1.1 < entry["ratio_all_to_far"]["median"] < 1.4, "the events on channel 0 raise its scale"
        assert entry["control_ratio_on_unmarked_channels"]["median"] == pytest.approx(1.0, abs=0.02)
    assert len(report["scales"]) == 3 and all(len(row["scale"]) == 22 for row in report["scales"])
    factors = list(report["measure_1_profile"]["profile_factor_by_channel"].values())
    assert factors[21] > factors[1] > 0, "the channel profile of the signal must appear"
    assert script.main([*common, "--output", str(output), *options]) == 2, "an existing report is not replaced"
    assert script.main([*common, "--output", str(tmp_path / "manifest" / "in.json"), *options]) == 2
    limited = tmp_path / "limited.json"
    assert script.main([*common, "--output", str(limited), "--limit", "2", *options]) == 0
    assert json.loads(limited.read_text())["recordings_read"] == 2
    unevaluated = tmp_path / "default.json"
    assert script.main([*common, "--output", str(unevaluated), "--gap-seconds", "1", "--min-far-seconds", "2"]) == 0
    assert json.loads(unevaluated.read_text())["measure_2_events"]["passes"] is None

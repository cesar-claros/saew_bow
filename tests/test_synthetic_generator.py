"""Generator of the synthetic benchmark; run on the HPC."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eeg_compare.synthetic.benchmark import smoke_settings
from eeg_compare.synthetic.generator import GeneratorConfig, _background, generate, make_waveforms

TINY = smoke_settings().generator


def test_waveforms_are_unit_zero_mean_band_limited_and_end_at_zero() -> None:
    config = GeneratorConfig()
    planted, decoys = make_waveforms(config)
    assert planted.shape == (4, 125) and decoys.shape == (4, 125)
    both = np.concatenate([planted, decoys])
    assert np.allclose(np.linalg.norm(both, axis=1), 1.0)
    assert np.abs(both.sum(axis=1)).max() < 1e-9
    assert np.abs(both[:, [0, -1]]).max() < 1e-12
    power = np.abs(np.fft.rfft(both, n=4096, axis=1)) ** 2
    frequency = np.fft.rfftfreq(4096, 1 / 250.0)
    assert (power[:, frequency > 40.0].sum(axis=1) / power.sum(axis=1)).max() < 0.05
    assert (np.abs(both @ both.T) - np.eye(8)).max() < 0.95, "two waveforms are nearly equal"
    assert np.array_equal(planted, make_waveforms(config)[0])
    assert make_waveforms(replace(config, n_templates=2))[0].shape == (2, 125)
    small, small_decoys = make_waveforms(TINY)
    assert small.shape == (2, 32) and small_decoys.shape == (4, 32) and np.isfinite(small).all()


def test_background_has_unit_variance_the_band_and_the_channel_correlation() -> None:
    config = GeneratorConfig()
    rows = _background(50, config, np.random.default_rng(7))
    assert rows.shape == (50, 22, 1250)
    assert abs(rows.std() - 1.0) < 0.05
    power = np.abs(np.fft.rfft(rows, axis=-1)) ** 2
    frequency = np.fft.rfftfreq(1250, 1 / 250.0)
    outside = (frequency < 1.0) | (frequency > 30.0)
    assert power[..., outside].sum() / power.sum() < 1e-12
    low = power[..., (frequency >= 1.0) & (frequency < 3.0)].mean()
    high = power[..., (frequency >= 10.0) & (frequency <= 30.0)].mean()
    assert low > 3 * high, "the power does not fall with the frequency"
    correlation = np.corrcoef(rows.transpose(1, 0, 2).reshape(22, -1))
    assert abs(correlation[~np.eye(22, dtype=bool)].mean() - 0.3) < 0.05
    assert np.allclose(_background(3, config, np.random.default_rng(7)), rows[:3])


def test_a_run_holds_its_events_where_the_layout_says() -> None:
    run = generate(TINY, seed=3)
    assert run.train.shape == (12, 4, 320) and run.test.shape == (8, 4, 320) and run.train.dtype == torch.float32
    events, truth = run.train_events, run.truth
    assert len(events) == 24 and len(run.test_events) == 16
    for segment in range(12):
        chosen = events.segment == segment
        starts = events.start[chosen]
        assert sorted(events.template[chosen].tolist()) == [0, 1]
        assert bool((starts[1:] - starts[:-1] >= 32).all())
        assert int(starts.min()) >= 0 and int(starts.max()) <= 320 - 32
    assert torch.allclose(truth.templates.flatten(1).norm(dim=1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(truth.amplitudes, TINY.level * truth.sigma) and bool((truth.sigma > 0).all())
    assert bool((truth.patterns >= 0).all()) and int((truth.patterns > 0).sum()) == 4
    assert torch.allclose(truth.templates, truth.patterns[:, :, None] * truth.waveforms[:, None, :])
    quiet = generate(replace(TINY, events=False), seed=3)
    assert len(quiet.train_events) == 0 and len(quiet.test_events) == 0
    planted = run.train.double() - quiet.train.double()
    covered = torch.zeros(12, 320, dtype=torch.bool)
    for index in range(len(events)):
        segment, start = int(events.segment[index]), int(events.start[index])
        amplitude = float(events.amplitude[index])
        expected = amplitude * truth.templates[events.template[index]]
        assert torch.allclose(planted[segment, :, start : start + 32], expected, atol=1e-4 * (1.0 + amplitude))
        covered[segment, start : start + 32] = True
    outside = planted.abs().amax(dim=1)[~covered]
    assert float(outside.max()) < 1e-4, "the segments hold a signal outside the listed events"


def test_conditions_with_one_seed_share_patterns_backgrounds_and_layout() -> None:
    base = generate(TINY, seed=5)
    few = generate(replace(TINY, n_train=5), seed=5)
    assert torch.allclose(few.train, base.train[:5], atol=1e-5) and torch.allclose(few.test, base.test, atol=1e-5)
    assert torch.equal(few.train_events.start, base.train_events.start[base.train_events.segment < 5])
    signed = generate(replace(TINY, random_sign=True), seed=5)
    assert torch.equal(signed.train_events.start, base.train_events.start)
    assert torch.equal(signed.train_events.template, base.train_events.template)
    assert set(signed.train_events.sign.tolist()) == {-1, 1} and set(base.train_events.sign.tolist()) == {1}
    assert torch.equal(signed.truth.patterns, base.truth.patterns)
    other = generate(TINY, seed=6)
    assert not torch.equal(other.truth.patterns, base.truth.patterns)
    assert not torch.allclose(other.train, base.train, atol=1e-3)
    with pytest.raises(ValueError, match="calibration seed"):
        generate(TINY, seed=TINY.calibration_seed)


def test_a_delay_shifts_the_waveform_on_the_channels_of_its_pattern() -> None:
    config = replace(TINY, pattern_channels=3, max_delay=4)
    truth = generate(config, seed=2).truth
    plain = generate(replace(config, max_delay=0), seed=2).truth
    assert torch.equal(truth.patterns, plain.patterns) and torch.equal(truth.waveforms, plain.waveforms)
    assert not bool(plain.delays.any())
    for index in range(2):
        channels = torch.nonzero(truth.patterns[index]).flatten().tolist()
        assert truth.delays[index, channels].tolist() == [0, 2, 4] and int(truth.delays[index].sum()) == 6
        for channel in channels:
            delay = int(truth.delays[index, channel])
            shifted = torch.zeros(32, dtype=torch.float64)
            shifted[delay:] = truth.waveforms[index, : 32 - delay]
            row = truth.templates[index, channel]
            assert float(row @ shifted / (row.norm() * shifted.norm())) > 1 - 1e-9
    assert torch.allclose(truth.templates.flatten(1).norm(dim=1), torch.ones(2, dtype=torch.float64))


def test_the_signal_level_is_in_units_of_the_matched_filter_on_the_background() -> None:
    config = GeneratorConfig(n_train=2, n_test=40, calibration_segments=100, events=False)
    quiet = generate(config, seed=1)
    truth = quiet.truth
    responses = F.conv1d(quiet.test.double(), truth.templates)
    measured = responses.pow(2).mean(dim=(0, 2)).sqrt()
    assert torch.allclose(measured, truth.sigma, rtol=0.2), "the calibration does not transfer to new background"
    assert bool((truth.sigma > 1.0).all()), "a smooth waveform must respond to a 1/f background above the white level"
    loud = generate(replace(config, events=True), seed=1)
    assert torch.equal(loud.truth.sigma, truth.sigma)
    planted = loud.test.double() - quiet.test.double()
    events = loud.test_events
    for index in range(8):
        segment, start, template = (int(events.segment[index]), int(events.start[index]), int(events.template[index]))
        response = float((planted[segment, :, start : start + 125] * truth.templates[template]).sum())
        assert abs(response - 8.0 * float(truth.sigma[template])) < 1e-3 * response


def test_config_guards() -> None:
    with pytest.raises(ValueError, match="n_templates"):
        GeneratorConfig(n_templates=5)
    with pytest.raises(ValueError, match="band"):
        GeneratorConfig(sfreq=50.0)
    with pytest.raises(ValueError, match="do not fit"):
        GeneratorConfig(events_per_segment=12)
    with pytest.raises(ValueError, match="correlation"):
        GeneratorConfig(correlation=1.0)

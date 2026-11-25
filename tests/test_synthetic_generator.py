"""Generator of the synthetic benchmark; run on the HPC."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eeg_compare.synthetic.benchmark import smoke_settings
from eeg_compare.synthetic.generator import (
    GeneratorConfig,
    _background,
    _burst_ratios,
    _pick_ratios,
    generate,
    make_waveforms,
)

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


def _power_outside_the_band(rows: torch.Tensor, sfreq: float) -> float:
    power = np.abs(np.fft.rfft(rows.double().numpy(), axis=-1)) ** 2
    frequency = np.fft.rfftfreq(rows.shape[-1], 1 / sfreq)
    return float(power[..., (frequency < 1.0) | (frequency > 30.0)].sum() / power.sum())


def test_artifacts_change_amplitudes_and_add_transients_inside_the_band() -> None:
    config = replace(TINY, amplitude_log_sd=0.5, artifacts_per_segment=2)
    run, base = generate(config, seed=5), generate(TINY, seed=5)
    events = run.train_events
    assert torch.equal(events.start, base.train_events.start) and torch.equal(
        events.template, base.train_events.template
    )
    factor = events.amplitude / run.truth.amplitudes[events.template]
    assert bool((factor > 0).all()) and 0.1 < float(factor.log().std()) < 1.2
    assert torch.equal(base.train_events.amplitude, base.truth.amplitudes[base.train_events.template])
    quiet = generate(replace(config, events=False), seed=5).train - generate(replace(TINY, events=False), seed=5).train
    assert bool((quiet.abs().amax(dim=(1, 2)) > 0.5).all()), "every segment holds its artifacts"
    assert _power_outside_the_band(quiet, TINY.sfreq) < 1e-6
    touched = (quiet.abs().amax(dim=2) > 1e-3).sum(dim=1)
    assert int(touched.min()) >= 3, "an artifact lies on three channels"


def test_the_burst_layout_puts_each_event_on_the_grid_with_a_burst_behind_it() -> None:
    config = replace(TINY, burst=True)
    run = generate(config, seed=5)
    events, truth = run.train_events, run.truth
    center = events.start + 15
    assert (center // 32).view(12, 2).tolist() == [[1, 3]] * 12
    assert int((center % 32).min()) >= 0 and int((center % 32).max()) <= 4
    assert sorted(events.template[:2].tolist()) == [0, 1] and events.group is None
    assert truth.burst_scale is not None and truth.burst_scale.shape == (2,) and bool((truth.burst_scale > 0).all())
    assert truth.burst_check is not None and truth.burst_check.shape == (2, 2)
    assert float(truth.burst_check.min()) >= 0.0 and float(truth.burst_check.max()) <= 1.0
    assert truth.burst_ratio is not None and truth.burst_ratio.tolist() == [1.5, 1.5], (
        "with no floor the first ratio is taken"
    )
    weak = generate(replace(config, burst_ratio=1e-6), seed=5)
    assert torch.equal(weak.train_events.start, events.start)
    assert weak.truth.burst_ratio is not None and weak.truth.burst_ratio.tolist() == [1e-6, 1e-6]
    bursts = run.train.double() - weak.train.double()
    assert float(bursts.abs().max()) > 0.5, "the segments hold no burst"
    assert _power_outside_the_band(run.train - weak.train, TINY.sfreq) < 1e-6
    for segment in range(12):
        used = torch.zeros(4, dtype=torch.bool)
        for template in events.template[events.segment == segment].tolist():
            used |= truth.patterns[template] > 0
        if bool((~used).any()):
            assert float(bursts[segment, ~used].abs().max()) < 1e-4, "a burst lies on a channel outside its pattern"
        first = int(events.start[events.segment == segment].min())
        assert float(bursts[segment, :, :first].abs().max()) < 0.2 * float(bursts[segment].abs().max())
    with pytest.raises(ValueError, match="validity check"):
        generate(replace(config, burst_min_share=1.01), seed=5)


def test_the_generator_takes_the_first_ratio_that_passes_for_each_template() -> None:
    assert _burst_ratios(GeneratorConfig()) == [1.5, 2.0, 2.5, 3.0]
    assert _burst_ratios(GeneratorConfig(burst_ratio=2.0, burst_ratio_max=2.2)) == [2.0]
    shares = [
        np.array([[0.5, 1.0], [0.9, 0.9]]),
        np.array([[0.85, 0.95], [0.95, 0.7]]),
        np.array([[0.9, 0.9], [0.9, 0.9]]),
    ]
    assert _pick_ratios(shares, 0.8).tolist() == [1, 0]
    assert _pick_ratios(shares[:1], 0.8).tolist() == [-1, 0], "no ratio passes for the first template"
    assert _pick_ratios(shares, 0.0).tolist() == [0, 0]
    with pytest.raises(ValueError, match="burst ratios"):
        GeneratorConfig(burst_ratio=4.0)


def test_the_pair_layout_puts_two_templates_in_one_pooling_window() -> None:
    run = generate(replace(TINY, pairs=True), seed=5)
    events = run.train_events
    assert events.group is not None and events.group.tolist() == [index // 2 for index in range(24)]
    start = events.start.view(12, 2)
    apart = start[:, 1] - start[:, 0]
    assert int(apart.min()) >= 6 and int(apart.max()) <= 13
    window = (start + 15) // 32
    assert bool((window[:, 0] == window[:, 1]).all()) and set(window[:, 0].tolist()) <= {1, 3, 5, 7}
    assert all(sorted(pair) == [0, 1] for pair in events.template.view(12, 2).tolist())
    assert len(set(window[:, 0].tolist())) > 1, "all pairs lie in one window"
    assert generate(TINY, seed=5).train_events.group is None


def test_config_guards() -> None:
    with pytest.raises(ValueError, match="exclude"):
        replace(TINY, burst=True, pairs=True)
    with pytest.raises(ValueError, match="pooling windows"):
        replace(TINY, burst=True, n_times=256)
    with pytest.raises(ValueError, match="one event of each template"):
        replace(TINY, pairs=True, events_per_segment=1)
    with pytest.raises(ValueError, match="does not fit behind"):
        replace(TINY, burst=True, pool_len=16, n_times=320)
    with pytest.raises(ValueError, match="n_templates"):
        GeneratorConfig(n_templates=5)
    with pytest.raises(ValueError, match="band"):
        GeneratorConfig(sfreq=50.0)
    with pytest.raises(ValueError, match="do not fit"):
        GeneratorConfig(events_per_segment=12)
    with pytest.raises(ValueError, match="correlation"):
        GeneratorConfig(correlation=1.0)

"""Synthetic segments with planted waveforms (stage 2 protocol, section 3).

A segment is a 1/f background inside the band, with a common source that
correlates the channels, plus events. An event adds
``amplitude * template``, where a template is a spatial pattern times a
waveform, with an optional delay per channel. The amplitude of a template comes
from its signal level: an event of level ``s`` gives the planted template, used
as a matched filter, a response of ``s`` standard deviations of that filter on
the background. The standard deviation comes from background draws with a
calibration seed that no run uses.

A seed fixes the patterns, the backgrounds and the event layout. The random
streams are separate, so two conditions with one seed share their patterns and
backgrounds, and a run with fewer training segments holds the first segments of
the larger run.

This module holds the conditions that change parameters only: the base
condition, no events, a delay across channels, a random sign, and the number of
training segments. The conditions with artifacts, bursts and event pairs are a
later step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["Events", "GeneratorConfig", "SyntheticRun", "Truth", "generate", "make_waveforms"]

_SUPPORT_S = 0.352  # largest length of a waveform: 88 samples at 250 Hz
_FRAME_TAPER = 0.15  # share of the atom frame that tapers to zero at each end
_STREAMS = {"patterns": 1, "train_background": 2, "train_layout": 3, "test_background": 4, "test_layout": 5}


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration of the generator.

    Parameters
    ----------
    sfreq : float
        Sampling rate in Hz.
    n_times, n_channels : int
        Samples and channels of a segment.
    atom_len : int
        Length ``L`` of a template frame.
    band : tuple[float, float]
        Pass band of the background and of the waveforms in Hz.
    correlation : float
        Correlation of two channels of the background.
    n_templates : int
        Planted templates ``G`` (at most 4).
    pattern_channels : int
        Channels with a weight in every spatial pattern.
    events_per_segment : int
        Events of a segment. With ``events_per_segment == n_templates`` a segment
        holds one event of each template, in random order.
    min_gap : int
        Smallest distance of two event starts in samples.
    level : float
        Signal level of an event.
    n_train, n_test : int
        Training and test segments.
    max_delay : int
        Largest delay of the waveform across the channels of its pattern, in samples.
    random_sign : bool
        Give every event a random sign.
    events : bool
        ``False`` plants nothing (the background condition).
    calibration_segments : int
        Background segments of the calibration.
    calibration_seed : int
        Seed of the calibration; no run uses it.
    """

    sfreq: float = 250.0
    n_times: int = 1250
    n_channels: int = 22
    atom_len: int = 125
    band: tuple[float, float] = (1.0, 30.0)
    correlation: float = 0.3
    n_templates: int = 4
    pattern_channels: int = 6
    events_per_segment: int = 4
    min_gap: int = 125
    level: float = 8.0
    n_train: int = 200
    n_test: int = 100
    max_delay: int = 0
    random_sign: bool = False
    events: bool = True
    calibration_segments: int = 200
    calibration_seed: int = 20261124

    def __post_init__(self) -> None:
        if not 1 <= self.n_templates <= 4:
            raise ValueError(f"n_templates must be between 1 and 4, got {self.n_templates}")
        if not 2 <= self.pattern_channels <= self.n_channels:
            raise ValueError("pattern_channels must be between 2 and n_channels")
        if not 0.0 <= self.correlation < 1.0:
            raise ValueError(f"correlation must be in [0, 1), got {self.correlation}")
        low, high = self.band
        if not 0.0 < low < high <= self.sfreq / 2:
            raise ValueError(f"the band {self.band} must lie inside (0, {self.sfreq / 2}]")
        if self.atom_len > self.n_times or min(self.n_train, self.n_test, self.calibration_segments) < 1:
            raise ValueError("a segment must hold an atom, and every set needs a segment")
        free = self.n_times - self.atom_len - (self.events_per_segment - 1) * self.min_gap
        if self.events_per_segment < 1 or free < 0:
            raise ValueError("the events of a segment do not fit with this gap")
        if self.max_delay < 0 or self.level <= 0:
            raise ValueError("max_delay must not be negative and the level must be positive")


@dataclass(frozen=True)
class Truth:
    """What the generator planted.

    Parameters
    ----------
    waveforms : torch.Tensor
        ``(G, L)`` unit waveforms without a delay.
    patterns : torch.Tensor
        ``(G, C)`` unit spatial patterns with positive weights.
    templates : torch.Tensor
        ``(G, C, L)`` planted templates with unit Frobenius norm, with the delays.
    decoys : torch.Tensor
        ``(G', L)`` unit waveforms of the same families that are never planted.
    delays : torch.Tensor
        ``(G, C)`` delay of the waveform on every channel in samples.
    sigma : torch.Tensor
        ``(G,)`` standard deviation of the matched filter of every template on the background.
    amplitudes : torch.Tensor
        ``(G,)`` amplitude of an event of the configured level: ``level * sigma``.
    """

    waveforms: torch.Tensor
    patterns: torch.Tensor
    templates: torch.Tensor
    decoys: torch.Tensor
    delays: torch.Tensor
    sigma: torch.Tensor
    amplitudes: torch.Tensor


@dataclass(frozen=True)
class Events:
    """The events of a set of segments, one entry per event.

    Parameters
    ----------
    segment, template, start : torch.Tensor
        ``(n,)`` segment, planted template, and first sample of the template frame.
    amplitude : torch.Tensor
        ``(n,)`` amplitude of the event (positive).
    sign : torch.Tensor
        ``(n,)`` sign of the event, ``1`` or ``-1``.
    """

    segment: torch.Tensor
    template: torch.Tensor
    start: torch.Tensor
    amplitude: torch.Tensor
    sign: torch.Tensor

    def __len__(self) -> int:
        return int(self.segment.numel())


@dataclass(frozen=True)
class SyntheticRun:
    """The data of one run: training and test segments ``(B, C, T)`` in float32, their events and the truth."""

    config: GeneratorConfig
    seed: int
    train: torch.Tensor
    test: torch.Tensor
    train_events: Events
    test_events: Events
    truth: Truth


def _band_pass(signal: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Remove all content outside the band along the last axis, in the frequency domain."""
    spectrum = np.fft.rfft(signal, axis=-1)
    frequency = np.fft.rfftfreq(signal.shape[-1], 1.0 / sfreq)
    spectrum[..., (frequency < band[0]) | (frequency > band[1])] = 0.0
    return np.fft.irfft(spectrum, n=signal.shape[-1], axis=-1)


def _raw_shapes(time: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The four planted shapes and the four decoys on a time axis in seconds that starts at zero."""

    def spike_wave(spike_at: float, spike_sd: float, wave_at: float, wave_sd: float, ratio: float) -> np.ndarray:
        spike = -np.exp(-((time - spike_at) ** 2) / (2 * spike_sd**2))
        return spike + ratio * np.exp(-((time - wave_at) ** 2) / (2 * wave_sd**2))

    def damped(frequency: float, decay: float) -> np.ndarray:
        return np.sin(2 * np.pi * frequency * time) * np.exp(-time / decay)

    def biphasic(duration: float, at: float) -> np.ndarray:
        inside = (time >= at) & (time < at + duration)
        return np.where(inside, np.sin(2 * np.pi * (time - at) / duration), 0.0)

    def burst(frequency: float, cycles: float) -> np.ndarray:
        length = cycles / frequency
        window = np.where(time < length, np.sin(np.pi * time / length) ** 2, 0.0)
        return np.sin(2 * np.pi * frequency * time) * window

    planted = [spike_wave(0.08, 0.012, 0.21, 0.05, 0.6), damped(10.0, 0.10), biphasic(0.2, 0.07), burst(7.0, 2.0)]
    decoys = [spike_wave(0.25, 0.02, 0.11, 0.04, 0.8), damped(14.0, 0.07), biphasic(0.12, 0.11), burst(4.5, 1.5)]
    return planted, decoys


def make_waveforms(config: GeneratorConfig) -> tuple[np.ndarray, np.ndarray]:
    """The planted waveforms ``(G, L)`` and the decoys ``(4, L)``: band-limited, zero mean, unit norm.

    A shape lives on at most 0.352 s in the middle of the atom frame. It is
    band-passed in a long zero buffer, cut to the frame and tapered to zero at
    the frame ends, so that an atom can hold it at a shift without a cut.
    """
    length = config.atom_len
    support = min(max(int(round(_SUPPORT_S * config.sfreq)), 4), length)
    time = np.arange(support) / config.sfreq
    planted, decoys = _raw_shapes(time)
    edge = np.sin(np.linspace(0.0, np.pi, support)) ** 0.25
    offset = (length - support) // 2
    buffer = 8 * length
    ramp = max(int(round(_FRAME_TAPER * length)), 1)
    taper = np.ones(length)
    taper[:ramp] = np.sin(0.5 * np.pi * np.arange(ramp) / ramp) ** 2
    taper[length - ramp :] = taper[:ramp][::-1]
    finished = []
    for shape in planted + decoys:
        long = np.zeros(buffer)
        first = (buffer - length) // 2 + offset
        long[first : first + support] = shape * edge
        frame = _band_pass(long, config.sfreq, config.band)[(buffer - length) // 2 : (buffer - length) // 2 + length]
        frame = frame * taper
        frame = frame - frame.sum() / taper.sum() * taper
        finished.append(frame / np.linalg.norm(frame))
    stacked = np.stack(finished)
    return stacked[: config.n_templates], stacked[len(planted) :]


def _background(n_segments: int, config: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Background ``(n, C, T)`` with unit variance, a 1/f power spectrum in the band and correlated channels.

    The first segments of a larger draw equal a smaller draw with the same generator state.
    """
    frequency = np.fft.rfftfreq(config.n_times, 1.0 / config.sfreq)
    inside = (frequency >= config.band[0]) & (frequency <= config.band[1])
    amplitude = np.where(inside, 1.0 / np.sqrt(np.maximum(frequency, config.band[0])), 0.0)
    scale = 2.0 / config.n_times * np.sqrt((amplitude**2).sum())
    draws = rng.standard_normal((n_segments, config.n_channels + 1, frequency.size, 2))
    signal = np.fft.irfft(amplitude * (draws[..., 0] + 1j * draws[..., 1]), n=config.n_times, axis=-1) / scale
    own, common = signal[:, :-1], signal[:, -1:]
    return np.sqrt(1.0 - config.correlation) * own + np.sqrt(config.correlation) * common


def _patterns(config: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Spatial patterns ``(G, C)``: positive weights on ``pattern_channels`` random channels, unit norm."""
    patterns = np.zeros((config.n_templates, config.n_channels))
    for index in range(config.n_templates):
        channels = rng.choice(config.n_channels, size=config.pattern_channels, replace=False)
        patterns[index, channels] = rng.uniform(0.5, 1.0, size=config.pattern_channels)
    return patterns / np.linalg.norm(patterns, axis=1, keepdims=True)


def _templates(waveforms: np.ndarray, patterns: np.ndarray, max_delay: int) -> tuple[np.ndarray, np.ndarray]:
    """Planted templates ``(G, C, L)`` and delays ``(G, C)``: the delay grows linearly over the pattern channels."""
    n_templates, length = waveforms.shape
    templates = np.zeros((n_templates, patterns.shape[1], length))
    delays = np.zeros(patterns.shape, dtype=np.int64)
    for index in range(n_templates):
        channels = np.flatnonzero(patterns[index])
        steps = np.rint(max_delay * np.arange(channels.size) / max(channels.size - 1, 1)).astype(np.int64)
        for channel, delay in zip(channels, steps):
            shifted = np.zeros(length)
            shifted[delay:] = waveforms[index, : length - delay]
            templates[index, channel] = patterns[index, channel] * shifted
            delays[index, channel] = delay
    norms = np.sqrt((templates**2).sum(axis=(1, 2), keepdims=True))
    return templates / norms, delays


def _matched_filter_sd(templates: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """Standard deviation ``(G,)`` of every template as a matched filter on calibration background."""
    rng = np.random.default_rng(config.calibration_seed)
    noise = torch.from_numpy(_background(config.calibration_segments, config, rng))
    responses = F.conv1d(noise, torch.from_numpy(templates))
    return responses.pow(2).mean(dim=(0, 2)).sqrt().numpy()


def _layout(n_segments: int, config: GeneratorConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Segment, template, start and sign of every event; the draws of a segment do not depend on later segments."""
    count = config.events_per_segment
    free = config.n_times - config.atom_len - (count - 1) * config.min_gap
    segments, templates, starts, signs = [], [], [], []
    for segment in range(n_segments):
        offsets = np.sort(rng.integers(0, free + 1, size=count)) + config.min_gap * np.arange(count)
        if count == config.n_templates:
            chosen = rng.permutation(config.n_templates)
        else:
            chosen = rng.integers(0, config.n_templates, size=count)
        sign = rng.integers(0, 2, size=count) * 2 - 1
        segments.append(np.full(count, segment))
        starts.append(offsets)
        templates.append(chosen)
        signs.append(sign if config.random_sign else np.ones(count, dtype=np.int64))
    return {
        "segment": np.concatenate(segments).astype(np.int64),
        "template": np.concatenate(templates).astype(np.int64),
        "start": np.concatenate(starts).astype(np.int64),
        "sign": np.concatenate(signs).astype(np.int64),
    }


def _segments(
    n_segments: int, config: GeneratorConfig, templates: np.ndarray, amplitudes: np.ndarray, seed: int, part: str
) -> tuple[torch.Tensor, Events]:
    """One set of segments with its events."""
    background = np.random.default_rng([seed, _STREAMS[f"{part}_background"]])
    rows = _background(n_segments, config, background)
    layout = _layout(n_segments, config, np.random.default_rng([seed, _STREAMS[f"{part}_layout"]]))
    keep = slice(None) if config.events else slice(0, 0)
    segment, template, start, sign = (layout[name][keep] for name in ("segment", "template", "start", "sign"))
    amplitude = amplitudes[template]
    for index in range(segment.size):
        frame = slice(int(start[index]), int(start[index]) + config.atom_len)
        rows[segment[index], :, frame] += sign[index] * amplitude[index] * templates[template[index]]
    events = Events(
        torch.from_numpy(segment),
        torch.from_numpy(template),
        torch.from_numpy(start),
        torch.from_numpy(amplitude.astype(np.float64)),
        torch.from_numpy(sign),
    )
    return torch.from_numpy(rows.astype(np.float32)), events


def generate(config: GeneratorConfig, seed: int) -> SyntheticRun:
    """Generate the training set, the test set and the truth of one run.

    Parameters
    ----------
    config : GeneratorConfig
        The condition.
    seed : int
        Seed of the patterns, the backgrounds and the layouts. It must differ
        from the calibration seed.

    Returns
    -------
    SyntheticRun
        Segments in float32, events, and the truth in float64.

    Raises
    ------
    ValueError
        If the seed is the calibration seed.
    """
    if seed == config.calibration_seed:
        raise ValueError("the calibration seed is not a seed of a run")
    waveforms, decoys = make_waveforms(config)
    patterns = _patterns(config, np.random.default_rng([seed, _STREAMS["patterns"]]))
    templates, delays = _templates(waveforms, patterns, config.max_delay)
    sigma = _matched_filter_sd(templates, config)
    amplitudes = config.level * sigma
    train, train_events = _segments(config.n_train, config, templates, amplitudes, seed, "train")
    test, test_events = _segments(config.n_test, config, templates, amplitudes, seed, "test")
    truth = Truth(
        waveforms=torch.from_numpy(waveforms),
        patterns=torch.from_numpy(patterns),
        templates=torch.from_numpy(templates),
        decoys=torch.from_numpy(decoys),
        delays=torch.from_numpy(delays),
        sigma=torch.from_numpy(sigma),
        amplitudes=torch.from_numpy(amplitudes),
    )
    return SyntheticRun(config, seed, train, test, train_events, test_events, truth)

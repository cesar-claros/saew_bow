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

Three conditions change the segments further. With artifacts, the amplitude of
an event is log-normal and a segment holds transients of high amplitude. With
bursts, every event has a noise burst behind it in its pooling window (protocol,
section 14). With pairs, two events lie in one pooling window. The extra draws
of these conditions come from a stream of their own, so the patterns, the
backgrounds and the templates of the events stay those of the base condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["Events", "GeneratorConfig", "SyntheticRun", "Truth", "generate", "make_waveforms"]

_SUPPORT_S = 0.352  # largest length of a waveform: 88 samples at 250 Hz
_FRAME_TAPER = 0.15  # share of the atom frame that tapers to zero at each end
_STREAMS = {
    "patterns": 1,
    "train_background": 2,
    "train_layout": 3,
    "test_background": 4,
    "test_layout": 5,
    "train_extra": 6,
    "test_extra": 7,
}
_GRID_WINDOWS = (1, 3, 5, 7)  # pooling windows of the events of the burst and the pair layouts
_TOLERANCE_S = 0.048  # distance that counts as "at the event" in the validity check of the bursts


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
    pool_len : int
        Pooling window ``P`` of the cells. Only the burst and the pair layouts use it.
    amplitude_log_sd : float
        Standard deviation of the logarithm of the amplitude factor of an event; ``0`` gives the factor 1.
    artifacts_per_segment : int
        Transients of high amplitude per segment.
    artifact_channels : int
        Channels of one artifact.
    artifact_peak, artifact_log_sd : float
        Median peak of an artifact in background standard deviations, and the
        standard deviation of its logarithm.
    burst : bool
        Put the events on the grid of section 14 of the protocol, each with a noise burst behind it.
    burst_ratio, burst_ratio_max : float
        Smallest and largest ratio of the burst. The ratio of a template is the
        median, over calibration draws, of the largest raw response of the
        planted waveform to the burst among the placements of the pooling window
        of the event, divided by its response to the event, on one channel. The
        generator tries the ratios from the smallest in steps of 0.5 and takes,
        for each template, the first that passes the validity check.
    burst_min_share : float
        Smallest share of the calibration draws in which the raw score must
        prefer the burst and the cosine score the event. Below it ``generate`` refuses the level.
    pairs : bool
        Put the events in pairs: two events of two templates in one pooling window.
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
    pool_len: int = 125
    amplitude_log_sd: float = 0.0
    artifacts_per_segment: int = 0
    artifact_channels: int = 3
    artifact_peak: float = 10.0
    artifact_log_sd: float = 0.5
    burst: bool = False
    burst_ratio: float = 1.5
    burst_ratio_max: float = 3.0
    burst_min_share: float = 0.8
    pairs: bool = False

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
        if self.amplitude_log_sd < 0 or self.artifacts_per_segment < 0 or self.artifact_log_sd < 0:
            raise ValueError("the spreads and the number of artifacts must not be negative")
        if not 1 <= self.artifact_channels <= self.n_channels or self.artifact_peak <= 0:
            raise ValueError("an artifact needs between 1 and n_channels channels and a positive peak")
        if self.burst and self.pairs:
            raise ValueError("the burst layout and the pair layout exclude each other")
        if self.burst or self.pairs:
            grid = self.n_times % self.pool_len == 0 and self.n_times // self.pool_len > _GRID_WINDOWS[-1] + 1
            if not grid or self.events_per_segment != self.n_templates:
                raise ValueError(
                    "the burst and the pair layouts need at least 9 complete pooling windows per segment "
                    "and one event of each template"
                )
        if not 0 < self.burst_ratio <= self.burst_ratio_max:
            raise ValueError("the burst ratios must be positive, with the smallest not above the largest")
        if self.burst and self.pool_len - 1 - _burst_distance(self) < 0:
            raise ValueError("the burst does not fit behind its event in one pooling window")
        if self.pairs and (self.events_per_segment % 2 or self.pool_len <= int(round(0.2 * self.sfreq))):
            raise ValueError("the pair layout needs an even number of events and a pooling window above 0.2 s")


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
    burst_ratio : torch.Tensor | None
        ``(G,)`` ratio of the burst that every template took (burst layout only).
    burst_scale : torch.Tensor | None
        ``(G,)`` factor ``r`` of the burst of every template (burst layout only).
    burst_check : torch.Tensor | None
        ``(G, 2)`` validity check of the burst layout on the calibration draws:
        the share in which the raw score is not at the event, and the share in
        which the cosine score is at the event.
    """

    waveforms: torch.Tensor
    patterns: torch.Tensor
    templates: torch.Tensor
    decoys: torch.Tensor
    delays: torch.Tensor
    sigma: torch.Tensor
    amplitudes: torch.Tensor
    burst_ratio: torch.Tensor | None = None
    burst_scale: torch.Tensor | None = None
    burst_check: torch.Tensor | None = None


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
    group : torch.Tensor | None
        ``(n,)`` pair of the event in the pair layout; ``None`` in other layouts.
    """

    segment: torch.Tensor
    template: torch.Tensor
    start: torch.Tensor
    amplitude: torch.Tensor
    sign: torch.Tensor
    group: torch.Tensor | None = None

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
    support = _support(config)
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


def _support(config: GeneratorConfig) -> int:
    """Samples of the middle of the atom frame that hold a waveform."""
    return min(max(int(round(_SUPPORT_S * config.sfreq)), 4), config.atom_len)


def _burst_distance(config: GeneratorConfig) -> int:
    """Samples from the center of an event frame to the center of its burst: half a waveform plus half a burst."""
    return _support(config) // 2 + config.atom_len // 2


def _burst_offset(config: GeneratorConfig) -> int:
    """Samples from the first sample of an event frame to the first sample of its burst."""
    return (config.atom_len - 1) // 2 + _burst_distance(config) - config.atom_len // 2


def _hann(length: int) -> np.ndarray:
    return np.sin(np.pi * np.arange(length) / length) ** 2


def _grid_starts(
    layout: dict[str, np.ndarray], config: GeneratorConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray | None]:
    """Event starts on the pooling grid for the burst and the pair layouts, and the pair of every event.

    Burst layout: the events of a segment lie in the pooling windows 1, 3, 5 and
    7, in the order of the layout, with the frame center in the first positions
    of the window, so that the burst behind the event has its center in the same
    window. Pair layout: two events share one of those windows, with starts 0.1
    to 0.2 s apart and both frame centers in the window.
    """
    count, pool = config.events_per_segment, config.pool_len
    left = (config.atom_len - 1) // 2
    n_segments = layout["segment"].size // count
    starts = np.zeros(n_segments * count, dtype=np.int64)
    if config.burst:
        latest = pool - 1 - _burst_distance(config)
        windows = np.array(_GRID_WINDOWS[:count])
        for segment in range(n_segments):
            position = rng.integers(0, latest + 1, size=count)
            starts[segment * count : (segment + 1) * count] = windows * pool + position - left
        return starts, None
    pairs = count // 2
    low, high = int(round(0.1 * config.sfreq)), int(round(0.2 * config.sfreq))
    groups = np.zeros(n_segments * count, dtype=np.int64)
    for segment in range(n_segments):
        windows = np.sort(rng.choice(np.array(_GRID_WINDOWS), size=pairs, replace=False))
        for pair in range(pairs):
            apart = int(rng.integers(low, high + 1))
            first = int(windows[pair]) * pool + int(rng.integers(0, pool - apart)) - left
            index = segment * count + 2 * pair
            starts[index], starts[index + 1] = first, first + apart
            groups[index] = groups[index + 1] = segment * pairs + pair
    return starts, groups


def _artifacts(n_segments: int, config: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """Transients of high amplitude ``(n, C, T)``, inside the band.

    Each artifact is one shape on ``artifact_channels`` random channels, with a
    random sign and a log-normal peak. Three families with random parameters: a
    step with an exponential decay (time constant 0.05 to 0.3 s), a half-sine of
    0.2 to 0.5 s, and a noise burst of 0.3 s. The peak is that of the shape
    before the band-pass.
    """
    out = np.zeros((n_segments, config.n_channels, config.n_times))
    for segment in range(n_segments):
        for _ in range(config.artifacts_per_segment):
            family = int(rng.integers(0, 3))
            if family == 0:
                decay = rng.uniform(0.05, 0.3) * config.sfreq
                length = int(round(5 * decay))
                shape = np.exp(-np.arange(length) / decay)
            elif family == 1:
                length = int(round(rng.uniform(0.2, 0.5) * config.sfreq))
                shape = np.sin(np.pi * np.arange(length) / length)
            else:
                length = int(round(0.3 * config.sfreq))
                shape = rng.standard_normal(length) * _hann(length)
                shape = shape / np.abs(shape).max()
            shape = shape[: config.n_times]
            onset = int(rng.integers(0, config.n_times - shape.size + 1))
            channels = rng.choice(config.n_channels, size=config.artifact_channels, replace=False)
            peak = config.artifact_peak * np.exp(config.artifact_log_sd * rng.standard_normal())
            sign = int(rng.integers(0, 2)) * 2 - 1
            out[segment, channels, onset : onset + shape.size] += sign * peak * shape
    return _band_pass(out, config.sfreq, config.band)


def _bursts(
    n_segments: int,
    config: GeneratorConfig,
    events: dict[str, np.ndarray],
    patterns: np.ndarray,
    scale: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """The noise burst behind every event ``(n, C, T)``, inside the band.

    On channel ``c`` of the pattern of the event the burst is white noise with a
    Hann window of one atom length and the standard deviation
    ``scale * amplitude * pattern[c]``. The noise is independent between channels.
    """
    out = np.zeros((n_segments, config.n_channels, config.n_times))
    length, offset = config.atom_len, _burst_offset(config)
    window = _hann(length)
    for index in range(events["segment"].size):
        template = events["template"][index]
        channels = np.flatnonzero(patterns[template])
        noise = rng.standard_normal((channels.size, length)) * window
        level = scale[template] * events["amplitude"][index] * patterns[template, channels]
        first = int(events["start"][index]) + offset
        out[events["segment"][index], channels, first : first + length] += level[:, None] * noise
    return _band_pass(out, config.sfreq, config.band)


def _burst_unit(waveforms: np.ndarray, config: GeneratorConfig) -> np.ndarray:
    """Factor ``(G,)`` of a burst with the ratio 1, measured in the pooling window of its event.

    A calibration draw is one burst behind an event start in pooling window 1,
    without background and without the event: white noise with a Hann window of
    one atom length, band-passed in a buffer of one segment. Its largest response
    is taken over the placements of the planted waveform whose center lies in
    that window. Only these placements compete with the event for the token of
    the window; the other half of the burst belongs to the next window. The
    factor is one over the median of that largest response.
    """
    rng = np.random.default_rng([config.calibration_seed, 1])
    length, pool = config.atom_len, config.pool_len
    left, offset = (length - 1) // 2, _burst_offset(config)
    latest = pool - 1 - _burst_distance(config)
    count = config.calibration_segments
    starts = pool + rng.integers(0, latest + 1, size=count) - left
    bursts = np.zeros((count, config.n_times))
    window = _hann(length)
    for index in range(count):
        first = int(starts[index]) + offset
        bursts[index, first : first + length] = rng.standard_normal(length) * window
    bursts = _band_pass(bursts, config.sfreq, config.band)
    responses = F.conv1d(torch.from_numpy(bursts).unsqueeze(1), torch.from_numpy(waveforms).unsqueeze(1))
    largest = responses[:, :, pool - left : 2 * pool - left].amax(dim=2).numpy()
    return 1.0 / np.median(largest, axis=0)


def _burst_ratios(config: GeneratorConfig) -> list[float]:
    """The ratios that the generator tries: from the smallest to the largest in steps of 0.5."""
    ratios = [float(config.burst_ratio)]
    while ratios[-1] + 0.5 <= config.burst_ratio_max + 1e-9:
        ratios.append(ratios[-1] + 0.5)
    return ratios


def _pick_ratios(shares: list[np.ndarray], min_share: float) -> np.ndarray:
    """For every template the index of the first ratio whose two shares reach ``min_share``; ``-1`` when none does.

    ``shares`` holds one ``(G, 2)`` result of ``_burst_check`` per ratio, in the order of the ratios.
    """
    count = shares[0].shape[0]
    picked = np.full(count, -1, dtype=np.int64)
    for template in range(count):
        for index, result in enumerate(shares):
            if float(result[template].min()) >= min_share:
                picked[template] = index
                break
    return picked


def _burst_check(
    waveforms: np.ndarray, amplitudes: np.ndarray, scale: np.ndarray, config: GeneratorConfig
) -> np.ndarray:
    """Validity check ``(G, 2)`` of the burst layout on one channel of the calibration background.

    For every template, each calibration segment gets one event in pooling
    window 1 and its burst, with the amplitude of a channel of a pattern with
    equal weights. Column 0 is the share of the segments in which the largest
    raw response of the planted waveform in that window is not at the event.
    Column 1 is the share in which its largest cosine in that window is at the
    event. "At the event" means within 0.048 s of the event start.
    """
    rng = np.random.default_rng([config.calibration_seed, 2])
    noise = _background(config.calibration_segments, config, np.random.default_rng(config.calibration_seed))[:, 0]
    length, pool = config.atom_len, config.pool_len
    left, offset = (length - 1) // 2, _burst_offset(config)
    latest = pool - 1 - _burst_distance(config)
    tolerance = int(round(_TOLERANCE_S * config.sfreq))
    window = _hann(length)
    count = noise.shape[0]
    shares = np.zeros((waveforms.shape[0], 2))
    for template in range(waveforms.shape[0]):
        amplitude = amplitudes[template] / np.sqrt(config.pattern_channels)
        starts = pool + rng.integers(0, latest + 1, size=count) - left
        rows = noise.copy()
        bursts = np.zeros_like(rows)
        for index in range(count):
            begin = int(starts[index])
            rows[index, begin : begin + length] += amplitude * waveforms[template]
            burst = scale[template] * amplitude * rng.standard_normal(length) * window
            bursts[index, begin + offset : begin + offset + length] = burst
        rows += _band_pass(bursts, config.sfreq, config.band)
        signal = torch.from_numpy(rows).unsqueeze(1)
        raw = F.conv1d(signal, torch.from_numpy(waveforms[template])[None, None, :])[:, 0]
        energy = F.avg_pool1d(signal.pow(2), length, stride=1)[:, 0] * length
        cosine = raw / energy.clamp_min(1e-12).sqrt()
        low, high = pool - left, 2 * pool - left
        truth = torch.from_numpy(starts)
        by_raw = raw[:, low:high].argmax(dim=1) + low
        by_cosine = cosine[:, low:high].argmax(dim=1) + low
        shares[template, 0] = float(((by_raw - truth).abs() > tolerance).double().mean())
        shares[template, 1] = float(((by_cosine - truth).abs() <= tolerance).double().mean())
    return shares


def _segments(
    n_segments: int, config: GeneratorConfig, planted: dict[str, np.ndarray], seed: int, part: str
) -> tuple[torch.Tensor, Events]:
    """One set of segments with its events.

    ``planted`` holds the templates, the amplitudes, the patterns and, for the
    burst layout, the burst factors. The background and the layout have their own
    random streams. The grid starts, the amplitude factors, the artifacts and the
    bursts draw from a third stream, in this order.
    """
    background = np.random.default_rng([seed, _STREAMS[f"{part}_background"]])
    rows = _background(n_segments, config, background)
    layout = _layout(n_segments, config, np.random.default_rng([seed, _STREAMS[f"{part}_layout"]]))
    extra = np.random.default_rng([seed, _STREAMS[f"{part}_extra"]])
    group: np.ndarray | None = None
    if config.burst or config.pairs:
        layout["start"], group = _grid_starts(layout, config, extra)
    keep = slice(None) if config.events else slice(0, 0)
    segment, template, start, sign = (layout[name][keep] for name in ("segment", "template", "start", "sign"))
    amplitude = planted["amplitudes"][template].astype(np.float64)
    if config.amplitude_log_sd > 0:
        amplitude = amplitude * np.exp(config.amplitude_log_sd * extra.standard_normal(amplitude.size))
    for index in range(segment.size):
        frame = slice(int(start[index]), int(start[index]) + config.atom_len)
        rows[segment[index], :, frame] += sign[index] * amplitude[index] * planted["templates"][template[index]]
    if config.artifacts_per_segment:
        rows += _artifacts(n_segments, config, extra)
    if config.burst and config.events:
        placed = {"segment": segment, "template": template, "start": start, "amplitude": amplitude}
        rows += _bursts(n_segments, config, placed, planted["patterns"], planted["burst_scale"], extra)
    events = Events(
        torch.from_numpy(segment),
        torch.from_numpy(template),
        torch.from_numpy(start),
        torch.from_numpy(amplitude),
        torch.from_numpy(sign),
        None if group is None else torch.from_numpy(group[keep]),
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
        If the seed is the calibration seed, or the burst layout fails its
        validity check for a template at every ratio (protocol, section 14).
    """
    if seed == config.calibration_seed:
        raise ValueError("the calibration seed is not a seed of a run")
    waveforms, decoys = make_waveforms(config)
    patterns = _patterns(config, np.random.default_rng([seed, _STREAMS["patterns"]]))
    templates, delays = _templates(waveforms, patterns, config.max_delay)
    sigma = _matched_filter_sd(templates, config)
    amplitudes = config.level * sigma
    planted = {"templates": templates, "amplitudes": amplitudes, "patterns": patterns}
    scale = check = ratio = None
    if config.burst:
        unit = _burst_unit(waveforms, config)
        ratios = _burst_ratios(config)
        shares = [_burst_check(waveforms, amplitudes, value * unit, config) for value in ratios]
        picked = _pick_ratios(shares, config.burst_min_share)
        if int(picked.min()) < 0:
            lines = [
                f"ratio {value}: "
                + ", ".join(f"template {g} raw {a:.2f} cosine {b:.2f}" for g, (a, b) in enumerate(result))
                for value, result in zip(ratios, shares)
            ]
            raise ValueError(
                f"the burst layout fails its validity check (smallest share {config.burst_min_share}) for a template "
                "at every ratio. Share of draws with the raw score away from the event and with the cosine score at "
                "the event: " + "; ".join(lines)
            )
        ratio = np.array([ratios[index] for index in picked])
        scale = ratio * unit
        check = np.stack([shares[index][template] for template, index in enumerate(picked)])
        planted["burst_scale"] = scale
    train, train_events = _segments(config.n_train, config, planted, seed, "train")
    test, test_events = _segments(config.n_test, config, planted, seed, "test")
    truth = Truth(
        waveforms=torch.from_numpy(waveforms),
        patterns=torch.from_numpy(patterns),
        templates=torch.from_numpy(templates),
        decoys=torch.from_numpy(decoys),
        delays=torch.from_numpy(delays),
        sigma=torch.from_numpy(sigma),
        amplitudes=torch.from_numpy(amplitudes),
        burst_ratio=None if ratio is None else torch.from_numpy(ratio),
        burst_scale=None if scale is None else torch.from_numpy(scale),
        burst_check=None if check is None else torch.from_numpy(check),
    )
    return SyntheticRun(config, seed, train, test, train_events, test_events, truth)

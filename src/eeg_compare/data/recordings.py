"""Explicit EDF channel mapping and complete-record preprocessing for TUEV."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import mne
import numpy as np
from numpy.typing import NDArray

from eeg_compare.data.config import PreprocessingConfig
from eeg_compare.data.provenance import content_hash, file_hash, resolve_path
from eeg_compare.preprocessing.resampling import resample_exact

_ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


class MissingChannelsError(ValueError):
    """A requested canonical derivation cannot be formed."""

    def __init__(self, channels: Sequence[str]) -> None:
        self.channels = tuple(channels)
        super().__init__(f"missing canonical channels: {', '.join(channels)}")


@dataclass(frozen=True)
class ChannelPlan:
    """Ordered EDF names and a linear map to the configured bipolar channels."""

    source_channels: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class RecordingIdentity:
    """Audited patient/session identity for one corpus-relative EDF path.

    ``identity_source`` describes the evidence for patient identity. This mapping
    must use a common patient namespace across the official splits.
    """

    edf_path: str
    patient_id: str
    session_id: str
    identity_source: str


@dataclass(frozen=True)
class Recording:
    """Portable recording inventory entry, including content and channel provenance."""

    recording_id: str
    edf_path: str
    rec_path: str
    patient_id: str
    session_id: str
    identity_source: str
    official_split: Literal["train", "eval"]
    sfreq: float
    n_samples: int
    source_channels: tuple[str, ...]
    channel_matrix: tuple[tuple[float, ...], ...]
    missing_channels: tuple[str, ...]
    edf_sha256: str
    rec_sha256: str

    @property
    def duration_seconds(self) -> float:
        """Duration on the selected-channel native clock, using the exclusive sample endpoint."""
        return self.n_samples / self.sfreq


def _canonical_name(name: str) -> tuple[str, str]:
    name = name.strip().upper()
    if name.startswith("EEG "):
        name = name[4:].strip()
    reference = ""
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            name, reference = name[: -len(suffix)], suffix[1:]
            break
    parts = [part.strip() for part in name.split("-")]
    return "-".join(_ALIASES.get(part, part) for part in parts), reference


def channel_plan(names: Sequence[str], config: PreprocessingConfig) -> ChannelPlan:
    """Map EDF labels by name, preserving output order and polarity.

    Referential inputs must use one common reference suffix. Bipolar inputs may
    be reversed, in which case a recorded coefficient of -1 restores polarity.
    Duplicate canonical inputs and mixed references raise ``ValueError``;
    absent derivations raise ``MissingChannelsError``.
    """
    candidates: dict[str, list[tuple[str, str]]] = {}
    for name in names:
        canonical, reference = _canonical_name(name)
        candidates.setdefault(canonical, []).append((name, reference))
    terms: list[list[tuple[str, float]]] = []
    references = set()
    missing = []
    for channel in config.channels:
        left, right = channel.split("-")
        if config.input_mode == "referential":
            operands = [(left, 1.0), (right, -1.0)]
        else:
            reverse = f"{right}-{left}"
            if channel in candidates and reverse in candidates:
                raise ValueError(f"both polarities are present for {channel}")
            operands = [(channel, 1.0)] if channel in candidates else [(reverse, -1.0)]
        if any(operand not in candidates for operand, _ in operands):
            missing.append(channel)
            continue
        row = []
        for operand, weight in operands:
            matches = candidates[operand]
            if len(matches) != 1:
                raise ValueError(f"ambiguous source channels for {operand}")
            source, reference = matches[0]
            references.add(reference)
            row.append((source, weight))
        terms.append(row)
    if missing:
        raise MissingChannelsError(missing)
    if config.input_mode == "referential" and len(references) != 1:
        raise ValueError("referential electrodes do not have a single common reference suffix")
    used = {source for row in terms for source, _ in row}
    source_names = tuple(name for name in names if name in used)
    index = {name: i for i, name in enumerate(source_names)}
    matrix = np.zeros((len(config.channels), len(source_names)), dtype=np.float64)
    for output, row in enumerate(terms):
        for source, weight in row:
            matrix[output, index[source]] = weight
    return ChannelPlan(source_names, tuple(tuple(float(value) for value in row) for row in matrix))


@contextmanager
def _open_edf(path: Path, *, channels: Sequence[str] | None = None, preload: bool = False) -> Iterator[mne.io.BaseRaw]:
    raw = mne.io.read_raw_edf(
        path, include=None if channels is None else list(channels), preload=preload, infer_types=False, verbose="ERROR"
    )
    try:
        yield raw
    finally:
        raw.close()


def _check_native_rate(sfreq: float, config: PreprocessingConfig) -> None:
    if not math.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("invalid native sampling frequency")
    for frequency in (config.highpass_hz, config.lowpass_hz, *config.notch_hz):
        if frequency is not None and frequency >= sfreq / 2:
            raise ValueError("a configured filter frequency reaches the native Nyquist frequency")


def inspect_recording(
    root: Path,
    identity: RecordingIdentity,
    official_split: Literal["train", "eval"],
    config: PreprocessingConfig,
) -> Recording:
    """Inspect EDF headers and hash both inputs, without loading waveform samples.

    The companion annotation must have the same stem and a .rec/.REC suffix.
    Missing/ambiguous files, invalid metadata or channel failures raise explicitly.
    Only the configured missing-channel policy can exclude a recording.
    """
    path = resolve_path(root, identity.edf_path)
    companions = [
        candidate
        for candidate in path.parent.iterdir()
        if candidate.is_file() and candidate.stem == path.stem and candidate.suffix.lower() == ".rec"
    ]
    if len(companions) != 1:
        raise ValueError("each EDF must have exactly one .rec or .REC companion")
    rec_path = companions[0].relative_to(root.resolve()).as_posix()
    resolve_path(root, rec_path)
    missing: tuple[str, ...] = ()
    with _open_edf(path) as raw:
        sfreq, n_samples = float(raw.info["sfreq"]), int(raw.n_times)
        try:
            plan = channel_plan(raw.ch_names, config)
        except MissingChannelsError as error:
            if config.missing_channels == "error":
                raise
            missing = error.channels
            plan = ChannelPlan((), ())
    if not missing:
        with _open_edf(path, channels=plan.source_channels) as raw:
            sfreq, n_samples = float(raw.info["sfreq"]), int(raw.n_times)
        _check_native_rate(sfreq, config)
    if not math.isfinite(sfreq) or sfreq <= 0 or n_samples <= 0:
        raise ValueError("EDF metadata has an invalid clock or length")
    edf_sha256, rec_sha256 = file_hash(path), file_hash(companions[0])
    return Recording(
        content_hash([identity.edf_path, edf_sha256, rec_sha256]),
        identity.edf_path,
        rec_path,
        identity.patient_id,
        identity.session_id,
        identity.identity_source,
        official_split,
        sfreq,
        n_samples,
        plan.source_channels,
        plan.matrix,
        missing,
        edf_sha256,
        rec_sha256,
    )


def resampled_length(n_samples: int, sfreq: float, target_sfreq: float) -> int:
    """Output length of the retained exact-ratio polyphase resampler."""
    ratio = Fraction(target_sfreq / sfreq).limit_denominator(10_000)
    if not math.isclose(float(ratio), target_sfreq / sfreq, rel_tol=1e-12, abs_tol=0):
        raise ValueError("sampling ratio cannot be represented accurately by the configured resampler")
    return (n_samples * ratio.numerator + ratio.denominator - 1) // ratio.denominator


def normalize_signal(data: NDArray[np.float64], config: PreprocessingConfig) -> NDArray[np.float64]:
    """Normalize real samples per channel and clip in the resulting units.

    Z-scores use population standard deviation; MAD scaling uses the lower median,
    1.4826 calibration and a 1% standard-deviation floor. Flat channels map to zero.
    The absolute scale floor is 1e-12 volts. No statistics are fitted across people.
    """
    if data.ndim != 2 or data.shape[1] == 0 or not np.isfinite(data).all():
        raise ValueError("normalization requires finite (channels, nonempty samples) data")
    if config.normalization == "zscore":
        data = (data - data.mean(axis=-1, keepdims=True)) / data.std(axis=-1, keepdims=True).clip(1e-12)
    elif config.normalization == "mad":
        middle = (data.shape[-1] - 1) // 2
        center = np.partition(data, middle, axis=-1)[:, middle : middle + 1]
        data = data - center
        mad = np.partition(np.abs(data), middle, axis=-1)[:, middle : middle + 1] * 1.4826
        scale = np.maximum(mad, 0.01 * data.std(axis=-1, keepdims=True)).clip(1e-12)
        data = data / scale
    if config.clip is not None:
        data = np.clip(data, -config.clip, config.clip)
    return data


def read_recording(root: Path, recording: Recording, config: PreprocessingConfig) -> NDArray[np.float32]:
    """Load selected electrodes, form the montage, filter, then resample the whole recording.

    The returned array is (configured channels, target samples), anchored at the
    recording's time zero. Nonfinite signals and changed geometry fail explicitly.
    The caller verifies source hashes. No bad-channel interpolation or CAR is applied.
    """
    if recording.missing_channels:
        raise ValueError("cannot read an excluded recording")
    path = resolve_path(root, recording.edf_path)
    _check_native_rate(recording.sfreq, config)
    with _open_edf(path, channels=recording.source_channels, preload=True) as raw:
        if raw.n_times != recording.n_samples or not math.isclose(raw.info["sfreq"], recording.sfreq, rel_tol=1e-12):
            raise ValueError("EDF geometry has changed since manifest construction")
        data = np.asarray(raw.get_data(picks=list(recording.source_channels)), dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError("EDF contains nonfinite signal samples")
    data = np.asarray(recording.channel_matrix, dtype=np.float64) @ data
    if config.highpass_hz is not None or config.lowpass_hz is not None:
        data = mne.filter.filter_data(
            data,
            recording.sfreq,
            config.highpass_hz,
            config.lowpass_hz,
            method="fir",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
            pad="reflect_limited",
            n_jobs=1,
            verbose="ERROR",
        )
    if config.notch_hz:
        data = mne.filter.notch_filter(
            data,
            recording.sfreq,
            freqs=list(config.notch_hz),
            method="fir",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
            pad="reflect_limited",
            n_jobs=1,
            verbose="ERROR",
        )
    expected = resampled_length(recording.n_samples, recording.sfreq, config.target_sfreq)
    if recording.sfreq != config.target_sfreq:
        data = resample_exact(data, recording.sfreq, config.target_sfreq)
    if data.shape != (len(config.channels), expected) or not np.isfinite(data).all():
        raise ValueError("preprocessed recording has invalid samples or geometry")
    if config.normalization_scope == "recording":
        data = normalize_signal(data, config)
    return np.asarray(data, dtype=np.float32)

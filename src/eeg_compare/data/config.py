"""Validated, serializable configuration of the TUEV signal and example contract."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from eeg_compare.data.annotations import MONTAGE


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Signal contract, in volts until optional normalization.

    Filters run on complete recordings at their selected-channel native rate.
    Normalization is per channel, either on the recording or real example samples.
    ``clip`` is in the resulting units. Missing channels either fail the build or
    exclude the whole recording; channels are never silently zero-filled.
    """

    channels: tuple[str, ...] = MONTAGE
    input_mode: Literal["referential", "bipolar"] = "referential"
    target_sfreq: float = 250.0
    highpass_hz: float | None = 0.1
    lowpass_hz: float | None = None
    notch_hz: tuple[float, ...] = (60.0,)
    missing_channels: Literal["error", "drop_recording"] = "error"
    normalization: Literal["none", "zscore", "mad"] = "none"
    normalization_scope: Literal["recording", "example"] = "example"
    clip: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        object.__setattr__(self, "notch_hz", tuple(self.notch_hz))
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must be nonempty and unique")
        if any(channel not in MONTAGE for channel in self.channels):
            raise ValueError("channels must use canonical TUEV derivation names")
        if self.input_mode not in ("referential", "bipolar"):
            raise ValueError("input_mode must be referential or bipolar")
        if self.missing_channels not in ("error", "drop_recording"):
            raise ValueError("missing_channels must be error or drop_recording")
        if self.normalization not in ("none", "zscore", "mad"):
            raise ValueError("normalization must be none, zscore or mad")
        if self.normalization_scope not in ("recording", "example"):
            raise ValueError("normalization_scope must be recording or example")
        _positive(self.target_sfreq, "target_sfreq")
        for name in ("highpass_hz", "lowpass_hz"):
            value = getattr(self, name)
            if value is not None:
                _positive(value, name, allow_zero=True)
                if value == 0:
                    object.__setattr__(self, name, None)
        for name, value in (("highpass_hz", self.highpass_hz), ("lowpass_hz", self.lowpass_hz), ("clip", self.clip)):
            if value is not None:
                _positive(value, name)
        for value in self.notch_hz:
            _positive(value, "notch_hz")
        if len(set(self.notch_hz)) != len(self.notch_hz):
            raise ValueError("notch_hz must be unique")
        if self.highpass_hz is not None and self.lowpass_hz is not None and self.highpass_hz >= self.lowpass_hz:
            raise ValueError("highpass_hz must be below lowpass_hz")
        for value in (self.highpass_hz, self.lowpass_hz):
            if value is not None and value >= self.target_sfreq / 2:
                raise ValueError("passband edges must be below the target Nyquist frequency")


@dataclass(frozen=True)
class ExampleConfig:
    """One onset-anchored context, with explicit duplicate, conflict and edge rules.

    ``interval`` merges channels with identical onset, offset and class;
    ``annotation`` keeps separate channels. Exact duplicate rows always merge,
    retaining all source line numbers. Conflicts concern overlapping annotated
    intervals of different classes on any channel, not the wider context.
    """

    before_seconds: float = 2.0
    after_seconds: float = 3.0
    unit: Literal["interval", "annotation"] = "interval"
    edge_policy: Literal["drop", "zero_pad"] = "drop"
    conflict_policy: Literal["drop", "keep"] = "drop"

    def __post_init__(self) -> None:
        _positive(self.before_seconds, "before_seconds", allow_zero=True)
        _positive(self.after_seconds, "after_seconds")
        if self.unit not in ("interval", "annotation"):
            raise ValueError("unit must be interval or annotation")
        if self.edge_policy not in ("drop", "zero_pad") or self.conflict_policy not in ("drop", "keep"):
            raise ValueError("invalid edge_policy or conflict_policy")

    def sample_count(self, sfreq: float) -> int:
        """Return the exact configured length; reject fractional-sample durations."""
        count = (self.before_seconds + self.after_seconds) * sfreq
        if not math.isclose(count, round(count), rel_tol=0, abs_tol=1e-8) or count < 1:
            raise ValueError("example duration must contain an integer number of target samples")
        return round(count)


@dataclass(frozen=True)
class SplitConfig:
    """Seeded patient folds within official training data; evaluation is untouched."""

    n_folds: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.n_folds, bool) or not isinstance(self.n_folds, int) or self.n_folds < 2:
            raise ValueError("n_folds must be an integer >= 2")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class PipelineConfig:
    """Complete signal/example/split configuration persisted with the manifest."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    examples: ExampleConfig = field(default_factory=ExampleConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        self.examples.sample_count(self.preprocessing.target_sfreq)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible configuration for provenance and hashing."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PipelineConfig:
        """Deserialize validated sections; unknown keys raise rather than being ignored."""
        unknown = set(values) - {"preprocessing", "examples", "splits"}
        if unknown:
            raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
        return cls(
            PreprocessingConfig(**values.get("preprocessing", {})),
            ExampleConfig(**values.get("examples", {})),
            SplitConfig(**values.get("splits", {})),
        )

    @classmethod
    def from_toml(cls, path: Path) -> PipelineConfig:
        """Read a TOML configuration; omitted fields use the documented defaults."""
        with path.open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

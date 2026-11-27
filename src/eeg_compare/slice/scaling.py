"""The ``profile`` input level: one recording scale times a fixed channel profile (slice protocol, section 4).

The primary input of the study divides every channel of a recording by its own
robust scale. The second level divides it by the product of a recording part and
a channel part that is the same for all recordings:

- the recording part is the mean of the log scales of the channels;
- the channel profile is the mean, over the training recordings of a fold, of
  what remains;
- a channel whose own scale is above ``cap`` times the product counts as
  ``cap`` times the product, so that no channel is more than ``cap`` times as
  loud as its profile says.

The second level is computed from the primary input. A channel of the primary
input is multiplied by ``min(ratio, cap)``, where ``ratio`` is its own scale
over the product, and the result is clipped as the primary input is. A sample
that the clip of the primary input changed keeps that change. The scales come
from the report of ``scripts/audit_scales.py``, which holds the scale of every
channel of every training recording.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["ChannelProfile", "fit_profile", "profile_factors", "read_scales"]

Scales = Mapping[str, NDArray[np.float64]]


@dataclass(frozen=True)
class ChannelProfile:
    """The fixed channel profile of a fold.

    Parameters
    ----------
    profile : numpy.ndarray
        ``(C,)`` channel part of the log scale; its mean is zero.
    cap : float
        Largest factor of a channel against its profile.
    """

    profile: NDArray[np.float64]
    cap: float = 3.0

    def factor(self, scale: NDArray[np.float64]) -> NDArray[np.float64]:
        """The factor ``(C,)`` that turns the primary input of a recording into the profile input.

        Parameters
        ----------
        scale : numpy.ndarray
            ``(C,)`` own scale of every channel of the recording, positive.
        """
        values = np.asarray(scale, dtype=np.float64)
        if values.shape != self.profile.shape or not (values > 0).all():
            raise ValueError("the scale must hold one positive value per channel of the profile")
        logs = np.log(values)
        return np.minimum(np.exp(logs - logs.mean() - self.profile), self.cap)


def fit_profile(scales: Scales, training: Iterable[str], cap: float = 3.0) -> ChannelProfile:
    """The channel profile of the training recordings of a fold.

    Parameters
    ----------
    scales : Mapping[str, numpy.ndarray]
        Own channel scales ``(C,)`` by recording.
    training : Iterable[str]
        The recordings of the training part of the fold. No other recording enters.
    cap : float
        Largest factor of a channel against its profile.

    Raises
    ------
    ValueError
        If there is no training recording, a recording has no scale, or the cap is not above 1.
    """
    names = sorted(set(training))
    if not names or cap <= 1:
        raise ValueError("a profile needs training recordings and a cap above 1")
    missing = [name for name in names if name not in scales]
    if missing:
        raise ValueError(f"{len(missing)} training recordings have no scale in the report")
    logs = np.log(np.stack([np.asarray(scales[name], dtype=np.float64) for name in names]))
    return ChannelProfile((logs - logs.mean(axis=1, keepdims=True)).mean(axis=0), float(cap))


def profile_factors(
    scales: Scales, profile: ChannelProfile, recordings: Iterable[str]
) -> dict[str, NDArray[np.float64]]:
    """The factor of every listed recording; a recording uses its own scales and the profile of the fold."""
    return {name: profile.factor(scales[name]) for name in sorted(set(recordings))}


def read_scales(
    report: Path, channels: Sequence[str], band_hz: Sequence[float | None]
) -> dict[str, NDArray[np.float64]]:
    """The channel scales of the training recordings from a report of the scale audit.

    Parameters
    ----------
    report : Path
        JSON report of ``scripts/audit_scales.py``.
    channels : Sequence[str]
        Channels of the manifest, in order; the report must have the same.
    band_hz : Sequence[float | None]
        High-pass and low-pass edge of the manifest; the report must have the same.

    Raises
    ------
    ValueError
        If the report has other channels, another band, or a scale that is not positive.
    """
    content = json.loads(report.read_text(encoding="utf-8"))
    if list(content["measure_1_profile"]["profile_factor_by_channel"]) != list(channels):
        raise ValueError("the scale report has other channels than the manifest")
    if list(content["band_hz"]) != list(band_hz):
        raise ValueError("the scale report has another band than the manifest")
    scales = {row["recording_id"]: np.asarray(row["scale"], dtype=np.float64) for row in content["scales"]}
    if any(value.shape != (len(channels),) or not (value > 0).all() for value in scales.values()):
        raise ValueError("the scale report holds a scale that is not one positive value per channel")
    return scales

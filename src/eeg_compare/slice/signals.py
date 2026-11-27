"""Signals of plan units and of examples for the slice.

``load_units`` reads the samples of a table of units (intervals of a plan, or
contexts) from the recordings of a reader, one recording at a time. The primary
input is the signal of the reader as it is. The ``profile`` input multiplies
every channel by the factor of its recording and clips again
(``eeg_compare.slice.scaling``).

The signal of a recording is the primary input only when the manifest scales
every channel by recording. ``study_clip`` refuses any other manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray

from eeg_compare.data.config import PreprocessingConfig

__all__ = ["RecordingSource", "load_units", "study_clip"]


class RecordingSource(Protocol):
    """A reader that gives the preprocessed signal ``(channels, samples)`` of a recording."""

    def recording(self, recording_id: str) -> NDArray[np.float32]: ...


def study_clip(preprocessing: PreprocessingConfig) -> float:
    """The clip of a manifest that the slice can read, in robust units.

    Raises
    ------
    ValueError
        If the manifest does not divide every channel of a recording by its
        robust scale, or has no clip. The stored signal of such a manifest is
        not the primary input of the study.
    """
    scaled = (preprocessing.normalization, preprocessing.normalization_scope) == ("mad", "recording")
    if not scaled or preprocessing.clip is None:
        raise ValueError("the slice needs a manifest with the robust scale per recording and a clip")
    return float(preprocessing.clip)


def load_units(
    reader: RecordingSource,
    units: pl.DataFrame,
    factors: Mapping[str, NDArray[np.float64]] | None = None,
    clip: float | None = None,
) -> torch.Tensor:
    """The signals ``(n, C, S)`` of units, in the order of the table, in single precision.

    Parameters
    ----------
    reader : RecordingSource
        Gives the signal of a recording; ``TUEVReader`` is one.
    units : polars.DataFrame
        Columns ``recording_id``, ``start_sample`` and ``n_samples``; all units have one length.
    factors : Mapping[str, numpy.ndarray] | None
        Factor ``(C,)`` of every recording for the ``profile`` input; ``None`` gives the primary input.
    clip : float | None
        Clip of the ``profile`` input, in its units; the clip of the manifest.

    Raises
    ------
    ValueError
        If the table is empty, the units differ in length, a unit leaves its
        recording, a recording of the table has no factor, or a clip comes
        without factors or is not positive.
    """
    if clip is not None and (factors is None or not clip > 0):
        raise ValueError("a clip must be positive and needs the factors of the profile input")
    rows = units.select("recording_id", "start_sample", "n_samples").to_dicts()
    lengths = {row["n_samples"] for row in rows}
    if len(lengths) != 1:
        raise ValueError("the units must be a nonempty table with one length")
    length = lengths.pop()
    by_recording: dict[str, list[tuple[int, int]]] = {}
    for index, row in enumerate(rows):
        by_recording.setdefault(row["recording_id"], []).append((index, row["start_sample"]))
    out: NDArray[np.float32] | None = None
    for recording_id, items in by_recording.items():
        data = reader.recording(recording_id)
        if out is None:
            out = np.empty((len(rows), data.shape[0], length), dtype=np.float32)
        factor = None
        if factors is not None:
            if recording_id not in factors:
                raise ValueError("a recording of the units has no factor of the profile input")
            factor = np.asarray(factors[recording_id], dtype=np.float32)[:, None]
        for index, start in items:
            if start < 0 or start + length > data.shape[-1]:
                raise ValueError("a unit leaves its recording")
            values = data[:, start : start + length]
            if factor is not None:
                values = values * factor
                if clip is not None:
                    values = np.clip(values, -clip, clip)
            out[index] = values
    assert out is not None
    return torch.from_numpy(out)

"""Readouts of a 5 s context: the pooling windows and the samples that a head may see (slice protocol, section 7).

A token arm gives tokens per pooling window. A readout names the windows whose
tokens enter the features, and the samples that the placements of those windows
cover. The spectral control reads the same samples.

=========  ==========================================================
Readout    Windows
=========  ==========================================================
full       all windows of the event context
event      the windows whose centers lie in the labeled second
context    the windows whose placements hold no sample of that second
displaced  all windows of the displaced control of the event
=========  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass

from eeg_compare.cells.common import same_padding

__all__ = ["EVENT_SAMPLES", "READOUT_NAMES", "Readout", "make_readouts", "window_support"]

EVENT_SAMPLES = (500, 750)
READOUT_NAMES = ("full", "event", "context", "displaced")


@dataclass(frozen=True)
class Readout:
    """One readout.

    Parameters
    ----------
    name : str
        ``full``, ``event``, ``context`` or ``displaced``.
    windows : tuple[int, ...]
        Pooling windows whose tokens enter the features.
    ranges : tuple[tuple[int, int], ...]
        Half-open sample ranges that the placements of those windows cover.
    displaced : bool
        Whether the features come from the displaced control and not from the event context.
    """

    name: str
    windows: tuple[int, ...]
    ranges: tuple[tuple[int, int], ...]
    displaced: bool = False


def window_support(window: int, n_times: int, atom_len: int, pool_len: int) -> tuple[int, int]:
    """The half-open sample range that the valid placements of a pooling window cover."""
    left, right = same_padding(atom_len)
    return max(0, window * pool_len - left), min(n_times, (window + 1) * pool_len + right)


def _merged(ranges: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for low, high in sorted(ranges):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return tuple(merged)


def make_readouts(
    n_times: int = 1250, atom_len: int = 125, pool_len: int = 125, event: tuple[int, int] = EVENT_SAMPLES
) -> dict[str, Readout]:
    """The four readouts of a context of ``n_times`` samples with the labeled second at ``event``.

    Raises
    ------
    ValueError
        If the context holds no whole number of pooling windows, or a readout has no window.
    """
    if n_times < 1 or n_times % pool_len:
        raise ValueError(f"a context of {n_times} samples holds no whole number of windows of {pool_len}")
    windows = range(n_times // pool_len)
    supports = {window: window_support(window, n_times, atom_len, pool_len) for window in windows}
    inside = tuple(m for m in windows if m * pool_len < event[1] and (m + 1) * pool_len > event[0])
    outside = tuple(m for m in windows if supports[m][1] <= event[0] or supports[m][0] >= event[1])
    if not inside or not outside:
        raise ValueError("the geometry leaves the event readout or the context readout without a window")
    everything = tuple(windows)
    whole = ((0, n_times),)
    return {
        "full": Readout("full", everything, whole),
        "event": Readout("event", inside, _merged([supports[m] for m in inside])),
        "context": Readout("context", outside, _merged([supports[m] for m in outside])),
        "displaced": Readout("displaced", everything, whole, displaced=True),
    }

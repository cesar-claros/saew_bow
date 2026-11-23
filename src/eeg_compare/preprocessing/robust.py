"""Robust row preprocessing retained from the original waveform pipeline."""

from __future__ import annotations

import torch

_EPS = 1e-8
_MAD_TO_STD = 1.4826
_CLIP = 20.0
ROBUST_CLIP = _CLIP
CHECKPOINT_NAME = "sae_state.pt"


def valid_rows(rows: torch.Tensor) -> torch.Tensor:
    """Usable rows ``(m,)`` of raw rows ``(m, T)``: finite everywhere and not constant.

    A constant row (the all-zero substitute the lazy loader emits for a failed
    load, or a flat channel) and a row with non-finite samples carry no signal;
    they are zeroed by ``_rows`` and excluded from the exposure that
    ``calibrate_amp_min`` and the diagnostics divide by. A partially padded
    row (short recording) counts as usable over its whole length.
    """
    finite = torch.isfinite(rows).all(dim=-1)
    spread = torch.nan_to_num(rows.amax(dim=-1) - rows.amin(dim=-1), nan=0.0) > 0
    return finite & spread


def _row_median(x: torch.Tensor) -> torch.Tensor:
    """Last-axis lower median, retaining the axis and propagating NaNs.

    CUDA median with a dimension returns indices even when only values are
    used, and strict determinism rejects that kernel. Stable sorting keeps
    the same median value (including for even lengths) without that kernel.
    """
    if x.is_cuda and torch.are_deterministic_algorithms_enabled():
        middle = (x.shape[-1] - 1) // 2
        median = x.sort(dim=-1, stable=True).values[..., middle : middle + 1]
        return median.masked_fill(x.isnan().any(dim=-1, keepdim=True), float("nan"))
    return x.median(dim=-1, keepdim=True).values


def robust_scale(x: torch.Tensor) -> torch.Tensor:
    """Center every row on its median, scale by 1.4826 * MAD, clip at +-20.

    A floor of 1 percent of the plain std keeps a mostly-flat row from
    exploding; an all-constant row maps to zeros.
    """
    median = _row_median(x)
    centered = x - median
    mad = _row_median(centered.abs()) * _MAD_TO_STD
    scale = torch.maximum(mad, 0.01 * centered.std(dim=-1, keepdim=True))
    return (centered / scale.clamp_min(_EPS)).clamp(-_CLIP, _CLIP)

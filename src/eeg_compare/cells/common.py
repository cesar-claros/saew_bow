"""Geometry and records that all comparison cells share.

A pooling window holds the placements whose atom center lies in it, as in the
retained models: with the padding ``(p_L, p_R)`` of ``same_padding``, a placement
whose first sample is ``s`` has the center ``s + p_L``. Only placements that lie
fully inside the segment are valid (the edge mask of the adapters).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

__all__ = ["CellAtoms", "Structure", "Tokens", "extended_windows", "same_padding", "unit_norm"]

Structure = Literal["S1", "SF", "SR"]
_EPS = 1e-12


def same_padding(atom_len: int) -> tuple[int, int]:
    """Left and right padding ``(p_L, p_R)`` of the retained models, with ``p_L + p_R = L - 1``."""
    left = (atom_len - 1) // 2
    return left, atom_len - 1 - left


def unit_norm(atoms: torch.Tensor) -> torch.Tensor:
    """Scale every atom ``(K, ...)`` to unit norm over its other axes; a zero atom stays zero."""
    norms = torch.linalg.vector_norm(atoms.flatten(1), dim=1).clamp_min(_EPS)
    return atoms / norms.view(-1, *([1] * (atoms.ndim - 1)))


def extended_windows(rows: torch.Tensor, atom_len: int, pool_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut segments into the supports of their pooling windows.

    Parameters
    ----------
    rows : torch.Tensor
        Segments ``(B, C, T)``; ``T`` is a positive multiple of ``pool_len``.
    atom_len : int
        Atom length ``L``.
    pool_len : int
        Pooling window ``P``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The windows ``(B, M, C, P + L - 1)``, zero outside the segment, and the
        boolean mask ``(M, P)`` of the valid shifts. Shift ``j`` of window ``m`` is
        the placement with the center ``m * P + j``.

    Raises
    ------
    ValueError
        If the segments are not 3-D, hold no complete pooling window, hold a
        partial one, or are shorter than one atom.
    """
    if rows.ndim != 3:
        raise ValueError(f"expected segments of shape (B, C, T), got {tuple(rows.shape)}")
    n_times = rows.shape[-1]
    if n_times == 0 or n_times % pool_len:
        raise ValueError(f"segment length {n_times} must be a positive multiple of pool_len={pool_len}")
    if n_times < atom_len:
        raise ValueError(f"a segment of {n_times} samples holds no complete placement of an atom of {atom_len}")
    left, right = same_padding(atom_len)
    windows = F.pad(rows, (left, right)).unfold(-1, pool_len + atom_len - 1, pool_len)
    center = torch.arange(n_times, device=rows.device)
    valid = ((center >= left) & (center < n_times - right)).view(-1, pool_len)
    return windows.transpose(1, 2), valid


@dataclass(frozen=True)
class CellAtoms:
    """The dictionary of a cell in the form of its structure.

    Parameters
    ----------
    structure : str
        ``S1``, ``SF`` or ``SR``.
    waveform : torch.Tensor | None
        ``(K, L)`` unit waveforms of ``S1`` and ``SR``.
    template : torch.Tensor | None
        ``(K, C, L)`` templates of ``SF`` with unit Frobenius norm.
    pattern : torch.Tensor | None
        ``(K, C, D)`` unit spatial patterns of ``SR``.
    """

    structure: Structure
    waveform: torch.Tensor | None = None
    template: torch.Tensor | None = None
    pattern: torch.Tensor | None = None

    def __post_init__(self) -> None:
        needed = {"S1": ("waveform",), "SF": ("template",), "SR": ("waveform", "pattern")}
        if self.structure not in needed:
            raise ValueError(f"unknown structure {self.structure!r}")
        dims = {"waveform": 2, "template": 3, "pattern": 3}
        for name in ("waveform", "template", "pattern"):
            value = getattr(self, name)
            if (value is not None) != (name in needed[self.structure]):
                raise ValueError(f"structure {self.structure} needs exactly the fields {needed[self.structure]}")
            if value is not None and value.ndim != dims[name]:
                raise ValueError(f"{name} must have {dims[name]} axes, got {tuple(value.shape)}")

    @property
    def n_atoms(self) -> int:
        """Number of atoms ``K``."""
        first = self.template if self.waveform is None else self.waveform
        assert first is not None
        return int(first.shape[0])


@dataclass(frozen=True)
class Tokens:
    """One token per segment and pooling window (R = 1).

    Parameters
    ----------
    atom : torch.Tensor
        ``(B, M)`` atom index; ``-1`` is the null token (no positive response).
    start : torch.Tensor
        ``(B, M)`` first sample of the placed atom in the segment; ``-1`` for the null token.
    value : torch.Tensor
        ``(B, M)`` score of the assignment rule of the cell at the placement
        (cosine or raw inner product); zero for the null token.
    response : torch.Tensor
        ``(B, M)`` raw inner product at the placement; zero for the null token.
    channel : torch.Tensor | None
        ``(B, M)`` channel of the token for ``S1`` (``-1`` for the null token); ``None`` otherwise.
    projection : torch.Tensor | None
        ``(B, M)`` spatial pattern that won for ``SR`` (``-1`` for the null token); ``None`` otherwise.
    """

    atom: torch.Tensor
    start: torch.Tensor
    value: torch.Tensor
    response: torch.Tensor
    channel: torch.Tensor | None = None
    projection: torch.Tensor | None = None

    @property
    def active(self) -> torch.Tensor:
        """Boolean ``(B, M)``: the windows that hold a token."""
        return self.atom >= 0

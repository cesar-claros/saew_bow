"""Alternating learners of the comparison (rules L1 to L4 of the plan).

One pass assigns every training unit to an atom and a shift, then replaces every
atom by a weighted sum of its aligned sub-windows, scaled to unit norm. Let ``y``
be the aligned sub-window of a unit and ``a`` its atom.

====  ==========================  =================================
Rule  Assignment                  Weight of ``y / norm(y)``
====  ==========================  =================================
L1    largest cosine              1
L2    largest cosine              ``norm(y)``
L3    largest cosine              ``norm(y)^2 * cosine(y, a)``
L4    largest raw inner product   ``norm(y)^2 * cosine(y, a)``
====  ==========================  =================================

The weight of L3 and L4 is one power step toward the least-squares atom: the new
atom is proportional to the sum of ``inner(y, a) * y``. A unit with no positive
response gets no atom. An atom without a unit keeps its value.

A training unit is the support of one pooling window: of one channel for the
shared single-channel structure (``S1``), of all channels for the full structure
(``SF``). The functions ``assign`` and ``update`` work on any units, so the parity
check against the author code of reference [1] can call them on its own windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from eeg_compare.cells.common import Tokens, extended_windows, same_padding, unit_norm

__all__ = [
    "RULES",
    "AlternatingFit",
    "AlternatingSpec",
    "Assignment",
    "Rule",
    "assign",
    "encode_alternating",
    "fit_alternating",
    "training_units",
    "update",
]

Score = Literal["cosine", "raw"]
Weight = Literal["unit", "norm", "least_squares"]
_EPS = 1e-12


@dataclass(frozen=True)
class Rule:
    """The two choices of an alternating rule: the position score and the window weight."""

    score: Score
    weight: Weight


RULES: dict[str, Rule] = {
    "L1": Rule("cosine", "unit"),
    "L2": Rule("cosine", "norm"),
    "L3": Rule("cosine", "least_squares"),
    "L4": Rule("raw", "least_squares"),
}


@dataclass(frozen=True)
class AlternatingSpec:
    """Configuration of one alternating fit.

    Parameters
    ----------
    rule : str
        ``L1``, ``L2``, ``L3`` or ``L4``.
    atom_len : int
        Atom length ``L``.
    pool_len : int
        Pooling window ``P``.
    single_channel : bool
        ``True`` for ``S1`` (a unit is one channel of one window), ``False`` for ``SF``.
    max_passes : int
        Largest number of passes.
    tol : float
        The passes stop when the mean over atoms of the squared change of an atom is below this value.
    chunk : int
        Units per block of the assignment and of the update.
    """

    rule: str
    atom_len: int
    pool_len: int
    single_channel: bool
    max_passes: int = 100
    tol: float = 1e-4
    chunk: int = 2048

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule must be one of {tuple(RULES)}, got {self.rule!r}")
        for name in ("atom_len", "pool_len", "max_passes", "chunk"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if not self.tol >= 0:
            raise ValueError(f"tol must not be negative, got {self.tol!r}")


@dataclass(frozen=True)
class Assignment:
    """The assignment of ``n`` units.

    Parameters
    ----------
    label : torch.Tensor
        ``(n,)`` atom of each unit; ``-1`` when no valid shift has a positive response.
    shift : torch.Tensor
        ``(n,)`` first sample of the aligned sub-window inside the unit; zero for an unassigned unit.
    response : torch.Tensor
        ``(n,)`` raw inner product at the selected shift; zero for an unassigned unit.
    cosine : torch.Tensor
        ``(n,)`` cosine at the selected shift; zero for an unassigned unit.
    """

    label: torch.Tensor
    shift: torch.Tensor
    response: torch.Tensor
    cosine: torch.Tensor


@dataclass(frozen=True)
class AlternatingFit:
    """Result of ``fit_alternating``.

    Parameters
    ----------
    atoms : torch.Tensor
        ``(K, c, L)`` unit atoms; ``c`` is 1 for ``S1`` and ``C`` for ``SF``.
    passes : int
        Passes that ran.
    converged : bool
        Whether the change fell below the tolerance.
    history : tuple[dict[str, float], ...]
        One record per pass, measured with the atoms before the update of that
        pass: ``change`` (after the update), ``assigned_share``, ``mean_cosine``,
        ``residual_share`` (one minus the summed squared responses over the
        summed unit energies: the common loss of the plan) and ``empty_atoms``.
    """

    atoms: torch.Tensor
    passes: int
    converged: bool
    history: tuple[dict[str, float], ...]


def training_units(
    rows: torch.Tensor, atom_len: int, pool_len: int, single_channel: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """The training units of segments ``(B, C, T)`` and the valid shifts of each.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Units ``(n, c, P + L - 1)`` and the boolean mask ``(n, P)``. For
        ``single_channel`` the unit of segment ``b``, window ``m`` and channel
        ``c`` has the index ``(b * M + m) * C + c`` and one channel. Otherwise the
        unit of segment ``b`` and window ``m`` has the index ``b * M + m`` and
        ``C`` channels.
    """
    windows, valid = extended_windows(rows, atom_len, pool_len)
    n_segments, n_windows, n_channels, length = windows.shape
    if single_channel:
        units = windows.reshape(n_segments * n_windows * n_channels, 1, length)
        mask = valid[None, :, None, :].expand(n_segments, n_windows, n_channels, pool_len)
    else:
        units = windows.reshape(n_segments * n_windows, n_channels, length)
        mask = valid[None].expand(n_segments, n_windows, pool_len)
    return units, mask.reshape(-1, pool_len)


def assign(
    units: torch.Tensor, atoms: torch.Tensor, score: Score, valid: torch.Tensor | None = None, chunk: int = 2048
) -> Assignment:
    """Assign every unit to the atom and the shift with the largest score.

    Parameters
    ----------
    units : torch.Tensor
        ``(n, c, N)`` units.
    atoms : torch.Tensor
        ``(K, c, L)`` atoms. The cosine divides by their norms; the raw score uses them as they are.
    score : str
        ``cosine`` or ``raw``.
    valid : torch.Tensor | None
        Boolean ``(n, N - L + 1)``: the shifts that a unit may use; ``None`` allows all.
    chunk : int
        Units per block.

    Returns
    -------
    Assignment
        Equal scores prefer the lower atom, then the earlier shift.

    Raises
    ------
    ValueError
        If the shapes do not agree, a unit is shorter than an atom, or the score is unknown.
    """
    if score not in ("cosine", "raw"):
        raise ValueError(f"score must be 'cosine' or 'raw', got {score!r}")
    if units.ndim != 3 or atoms.ndim != 3 or units.shape[1] != atoms.shape[1]:
        raise ValueError(f"units {tuple(units.shape)} and atoms {tuple(atoms.shape)} do not agree")
    atom_len = atoms.shape[-1]
    n_shifts = units.shape[-1] - atom_len + 1
    if n_shifts < 1:
        raise ValueError(f"units of {units.shape[-1]} samples are shorter than atoms of {atom_len}")
    if valid is not None and tuple(valid.shape) != (units.shape[0], n_shifts):
        raise ValueError(f"valid must have the shape {(units.shape[0], n_shifts)}, got {tuple(valid.shape)}")
    atom_norm = torch.linalg.vector_norm(atoms, dim=(1, 2)).clamp_min(_EPS)
    labels, shifts, responses, cosines = [], [], [], []
    for first in range(0, units.shape[0], chunk):
        block = units[first : first + chunk]
        raw = F.conv1d(block, atoms)
        energy = F.avg_pool1d(block.pow(2).sum(dim=1, keepdim=True), atom_len, stride=1) * atom_len
        cosine = raw / (energy.clamp_min(0.0).sqrt() * atom_norm[None, :, None]).clamp_min(_EPS)
        scores = cosine if score == "cosine" else raw
        if valid is not None:
            scores = scores.masked_fill(~valid[first : first + chunk, None, :], float("-inf"))
        best, flat = scores.flatten(1).max(dim=1)
        found = best > 0
        zero = torch.zeros_like(best)
        labels.append(torch.where(found, flat // n_shifts, torch.full_like(flat, -1)))
        shifts.append(torch.where(found, flat % n_shifts, torch.zeros_like(flat)))
        responses.append(torch.where(found, raw.flatten(1).gather(1, flat[:, None]).squeeze(1), zero))
        cosines.append(torch.where(found, cosine.flatten(1).gather(1, flat[:, None]).squeeze(1), zero))
    return Assignment(torch.cat(labels), torch.cat(shifts), torch.cat(responses), torch.cat(cosines))


def update(
    units: torch.Tensor,
    label: torch.Tensor,
    shift: torch.Tensor,
    atoms: torch.Tensor,
    weight: Weight,
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace every atom by the weighted sum of its aligned sub-windows, scaled to unit norm.

    Parameters
    ----------
    units : torch.Tensor
        ``(n, c, N)`` units.
    label : torch.Tensor
        ``(n,)`` atom of each unit; a negative label leaves the unit out.
    shift : torch.Tensor
        ``(n,)`` first sample of the aligned sub-window.
    atoms : torch.Tensor
        ``(K, c, L)`` current atoms, with unit norm for the weight ``least_squares``.
    weight : str
        ``unit`` (sum of normalized sub-windows), ``norm`` (sum of raw
        sub-windows) or ``least_squares`` (sum of ``inner(y, a) * y``).
    chunk : int
        Units per block.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The new atoms ``(K, c, L)`` in the dtype of ``atoms`` and the units per
        atom ``(K,)``. An atom without a unit, or with a zero sum, keeps its value.

    Raises
    ------
    ValueError
        If the weight is unknown.
    """
    if weight not in ("unit", "norm", "least_squares"):
        raise ValueError(f"weight must be 'unit', 'norm' or 'least_squares', got {weight!r}")
    n_atoms, n_channels, atom_len = atoms.shape
    current = atoms.double()
    index = torch.nonzero(label >= 0).flatten()
    sums = torch.zeros_like(current)
    offsets = torch.arange(atom_len, device=units.device)
    for first in range(0, index.numel(), chunk):
        chosen = index[first : first + chunk]
        positions = (shift[chosen, None] + offsets)[:, None, :].expand(-1, n_channels, -1)
        aligned = units[chosen].gather(-1, positions).double()
        if weight == "unit":
            aligned = aligned / torch.linalg.vector_norm(aligned, dim=(1, 2), keepdim=True).clamp_min(_EPS)
        elif weight == "least_squares":
            aligned = (aligned * current[label[chosen]]).sum(dim=(1, 2), keepdim=True) * aligned
        sums.index_add_(0, label[chosen], aligned)
    counts = torch.bincount(label[index], minlength=n_atoms)
    norms = torch.linalg.vector_norm(sums, dim=(1, 2), keepdim=True)
    keep = (counts > 0)[:, None, None] & (norms > _EPS)
    return torch.where(keep, sums / norms.clamp_min(_EPS), current).to(atoms.dtype), counts


def fit_alternating(rows: torch.Tensor, initial_atoms: torch.Tensor, spec: AlternatingSpec) -> AlternatingFit:
    """Fit atoms to segments by the alternating rule of ``spec``.

    Parameters
    ----------
    rows : torch.Tensor
        Training segments ``(B, C, T)``, already scaled.
    initial_atoms : torch.Tensor
        ``(K, 1, L)`` for ``single_channel``, else ``(K, C, L)``. They are scaled to unit norm.
    spec : AlternatingSpec
        Rule, geometry and stopping rule.

    Returns
    -------
    AlternatingFit
        The atoms after the last pass and the record of every pass.

    Raises
    ------
    ValueError
        If the initial atoms do not have the shape that the structure and the geometry need.
    """
    rule = RULES[spec.rule]
    expected = (1 if spec.single_channel else rows.shape[1], spec.atom_len)
    if initial_atoms.ndim != 3 or tuple(initial_atoms.shape[1:]) != expected:
        raise ValueError(f"initial atoms must have the shape (K, {expected[0]}, {expected[1]})")
    units, valid = training_units(rows, spec.atom_len, spec.pool_len, spec.single_channel)
    atoms = unit_norm(initial_atoms.to(rows))
    total_energy = float(units.pow(2).sum())
    history: list[dict[str, float]] = []
    converged = False
    for number in range(1, spec.max_passes + 1):
        assignment = assign(units, atoms, rule.score, valid, spec.chunk)
        updated, counts = update(units, assignment.label, assignment.shift, atoms, rule.weight, spec.chunk)
        change = float((updated - atoms).pow(2).sum(dim=(1, 2)).mean())
        assigned = assignment.label >= 0
        n_assigned = int(assigned.sum())
        history.append(
            {
                "pass": float(number),
                "change": change,
                "assigned_share": n_assigned / max(units.shape[0], 1),
                "mean_cosine": float(assignment.cosine[assigned].mean()) if n_assigned else 0.0,
                "residual_share": 1.0 - float(assignment.response.pow(2).sum()) / max(total_energy, _EPS),
                "empty_atoms": float((counts == 0).sum()),
            }
        )
        atoms = updated
        if change < spec.tol:
            converged = True
            break
    return AlternatingFit(atoms, len(history), converged, tuple(history))


def encode_alternating(
    rows: torch.Tensor, atoms: torch.Tensor, score: Score, pool_len: int, single_channel: bool, chunk: int = 2048
) -> Tokens:
    """One token per segment and pooling window, by the assignment rule of the cell.

    For ``single_channel`` every channel has its own best placement in a window,
    and the token is that of the channel with the largest score. Equal scores
    prefer the lower channel.

    Parameters
    ----------
    rows : torch.Tensor
        Segments ``(B, C, T)``.
    atoms : torch.Tensor
        ``(K, 1, L)`` for ``single_channel``, else ``(K, C, L)``.
    score : str
        ``cosine`` or ``raw``.
    pool_len : int
        Pooling window ``P``.
    single_channel : bool
        ``True`` for ``S1``.
    chunk : int
        Units per block.

    Returns
    -------
    Tokens
        ``(B, M)`` tokens; ``channel`` is set for ``single_channel`` only.
    """
    atom_len = atoms.shape[-1]
    units, valid = training_units(rows, atom_len, pool_len, single_channel)
    assignment = assign(units, atoms.to(rows), score, valid, chunk)
    n_segments, n_channels, n_times = rows.shape
    n_windows = n_times // pool_len
    value = assignment.cosine if score == "cosine" else assignment.response
    label, shift, response = assignment.label, assignment.shift, assignment.response
    channel: torch.Tensor | None = None
    shape: tuple[int, ...] = (n_segments, n_windows, n_channels) if single_channel else (n_segments, n_windows)
    label, shift, value, response = (t.view(shape) for t in (label, shift, value, response))
    if single_channel:
        ranked = torch.where(label >= 0, value, torch.full_like(value, float("-inf")))
        best = ranked.argmax(dim=-1, keepdim=True)
        label, shift, value, response = (t.gather(-1, best).squeeze(-1) for t in (label, shift, value, response))
        channel = torch.where(label >= 0, best.squeeze(-1), torch.full_like(label, -1))
    left = same_padding(atom_len)[0]
    first = torch.arange(n_windows, device=rows.device) * pool_len - left + shift
    start = torch.where(label >= 0, first, torch.full_like(first, -1))
    return Tokens(atom=label, start=start, value=value, response=response, channel=channel)

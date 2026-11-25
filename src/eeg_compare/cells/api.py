"""The nine cells of the controlled comparison behind one interface.

A cell is an atom structure with a learning rule (plan, section "Controlled
component comparison"). Every cell takes the same start, fits on the same
segments and gives one token per segment and pooling window.

=========  =========  ====  =========================================
Cell       Structure  Rule  Learner
=========  =========  ====  =========================================
S1-L1..L4  S1         L1-4  alternating, single-channel units
S1-L5      S1         L5    full-template adapter, single-channel rows
SF-L1      SF         L1    alternating, multichannel units
SF-L5      SF         L5    full-template adapter, extended loss
SR-L5      SR         L5    separable adapter
SR-L5n     SR         L5n   separable adapter, window-normalized loss
=========  =========  ====  =========================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from eeg_compare.cells.alternating import RULES, AlternatingSpec, Score, encode_alternating, fit_alternating
from eeg_compare.cells.common import CellAtoms, Structure, Tokens, unit_norm
from eeg_compare.cells.gradient import (
    GradientSpec,
    Model,
    build_model,
    encode_gradient,
    fit_gradient,
    model_atoms,
    separable_start,
    set_model_atoms,
)
from eeg_compare.device import resolve_device

__all__ = [
    "CELLS",
    "CELL_NAMES",
    "Budget",
    "Cell",
    "FittedCell",
    "Geometry",
    "Start",
    "draw_start",
    "fit_cell",
    "frozen_cell",
    "start_atoms",
]

CELL_NAMES = ("S1-L1", "S1-L2", "S1-L3", "S1-L4", "S1-L5", "SF-L1", "SF-L5", "SR-L5", "SR-L5n")
_GRADIENT_RULES = ("L5", "L5n")
_ENERGY_FLOOR = 1e-12


@dataclass(frozen=True)
class Cell:
    """One cell: its name, its atom structure and its learning rule."""

    name: str
    structure: Structure
    rule: str

    @property
    def gradient(self) -> bool:
        """Whether the cell learns by Adam (rules ``L5`` and ``L5n``)."""
        return self.rule in _GRADIENT_RULES

    @property
    def score(self) -> Score:
        """The position score of the cell: ``cosine`` for ``L1`` to ``L3``, else ``raw``."""
        return "raw" if self.gradient else RULES[self.rule].score


def _cell(name: str) -> Cell:
    structure, rule = name.split("-")
    if structure not in ("S1", "SF", "SR") or (rule not in RULES and rule not in _GRADIENT_RULES):
        raise ValueError(f"unknown cell {name!r}")
    return Cell(name, structure, rule)  # type: ignore[arg-type]


CELLS: dict[str, Cell] = {name: _cell(name) for name in CELL_NAMES}


@dataclass(frozen=True)
class Geometry:
    """Atoms ``K``, atom length ``L``, pooling window ``P`` and spatial patterns ``D`` of the rank-1 cells."""

    n_atoms: int = 8
    atom_len: int = 125
    pool_len: int = 125
    n_projections: int = 2

    def __post_init__(self) -> None:
        for name in ("n_atoms", "atom_len", "pool_len", "n_projections"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class Budget:
    """The budget of a fit.

    Parameters
    ----------
    max_passes : int
        Largest number of passes of an alternating rule.
    tol : float
        Stopping tolerance of an alternating rule.
    steps : int
        Optimizer steps of a gradient rule.
    lr : float
        Learning rate of a gradient rule.
    segments_per_step : int
        Segments per optimizer step.
    """

    max_passes: int = 100
    tol: float = 1e-4
    steps: int = 3000
    lr: float = 1e-3
    segments_per_step: int = 2


@dataclass(frozen=True)
class Start:
    """The common start of a run: ``K * D`` patches of the training segments.

    Parameters
    ----------
    patches : torch.Tensor
        ``(K * D, C, L)`` patches. ``SF`` and ``S1`` use the first ``K``.
    channels : torch.Tensor
        ``(K,)`` channel of patch ``k`` that ``S1`` starts from.
    segments : torch.Tensor
        ``(K * D,)`` segment of every patch.
    first : torch.Tensor
        ``(K * D,)`` first sample of every patch.
    """

    patches: torch.Tensor
    channels: torch.Tensor
    segments: torch.Tensor
    first: torch.Tensor


def draw_start(rows: torch.Tensor, geometry: Geometry, generator: torch.Generator) -> Start:
    """Draw the common start from training segments ``(B, C, T)``.

    The locations are uniform over the segments and over the first samples that
    keep a patch inside its segment. The channel of ``S1`` is uniform, as the
    random start of the author code of reference [1], which takes random windows.

    Raises
    ------
    ValueError
        If the segments are shorter than an atom, or a drawn patch or its ``S1`` channel is flat.
    """
    if rows.ndim != 3 or rows.shape[-1] < geometry.atom_len:
        raise ValueError(f"expected segments (B, C, T) with T >= {geometry.atom_len}, got {tuple(rows.shape)}")
    count = geometry.n_atoms * geometry.n_projections
    segments = torch.randint(0, rows.shape[0], (count,), generator=generator)
    first = torch.randint(0, rows.shape[-1] - geometry.atom_len + 1, (count,), generator=generator)
    channels = torch.randint(0, rows.shape[1], (geometry.n_atoms,), generator=generator)
    patches = torch.stack(
        [rows[int(s), :, int(f) : int(f) + geometry.atom_len] for s, f in zip(segments.tolist(), first.tolist())]
    )
    single = patches[torch.arange(geometry.n_atoms, device=patches.device), channels.to(patches.device)]
    if bool((patches.pow(2).sum(dim=(1, 2)) <= _ENERGY_FLOOR).any()) or bool(
        (single.pow(2).sum(dim=1) <= _ENERGY_FLOOR).any()
    ):
        raise ValueError("a patch of the start is flat; the segments hold too little signal for a patch start")
    return Start(patches, channels, segments, first)


def start_atoms(structure: Structure, start: Start, geometry: Geometry) -> CellAtoms:
    """The atoms that a structure takes from the common start.

    ``SF`` takes the first ``K`` patches. ``S1`` takes one channel of each of
    them. ``SR`` takes the factors of the retained patch initialization: the
    waveform of atom ``k`` is the top right singular vector of patch ``k``, and
    pattern ``d`` comes from patch ``d * K + k``.
    """
    n_atoms = geometry.n_atoms
    if start.patches.shape[0] != n_atoms * geometry.n_projections or start.patches.shape[-1] != geometry.atom_len:
        raise ValueError("the start does not have the geometry of the cell")
    if structure == "SR":
        return separable_start(start.patches, n_atoms, geometry.n_projections)
    first = start.patches[:n_atoms]
    if structure == "SF":
        return CellAtoms("SF", template=unit_norm(first))
    if structure == "S1":
        single = first[torch.arange(n_atoms, device=first.device), start.channels.to(first.device)]
        return CellAtoms("S1", waveform=unit_norm(single))
    raise ValueError(f"unknown structure {structure!r}")


@dataclass
class FittedCell:
    """A cell with a dictionary: learned, frozen at a start, or given (the oracle).

    Parameters
    ----------
    cell : Cell
        The cell.
    geometry : Geometry
        Its geometry.
    atoms : CellAtoms
        Its dictionary, under the constraint of the structure.
    record : dict[str, Any]
        The record of the fit: ``passes``, ``converged`` and ``history`` for an
        alternating rule; ``steps``, ``epochs`` and ``step_losses`` for a gradient
        rule; empty for a dictionary without a fit.
    model : ComparisonWaveformSAE | ComparisonSeparableSAE | None
        The adapter of a gradient cell.
    """

    cell: Cell
    geometry: Geometry
    atoms: CellAtoms
    record: dict[str, Any] = field(default_factory=dict)
    model: Model | None = None

    def tokens(self, rows: torch.Tensor) -> Tokens:
        """One token per segment ``(B, C, T)`` and pooling window, by the assignment rule of the cell."""
        if self.model is not None:
            assert self.model.atoms is not None
            return encode_gradient(self.model, rows.to(self.model.atoms.dtype), self.cell.structure)
        return encode_alternating(
            rows, _unit_atoms(self.atoms), self.cell.score, self.geometry.pool_len, self.cell.structure == "S1"
        )


def _unit_atoms(atoms: CellAtoms) -> torch.Tensor:
    """The atoms of ``S1`` or ``SF`` as the ``(K, c, L)`` tensor of the alternating learner."""
    if atoms.structure == "S1":
        assert atoms.waveform is not None
        return atoms.waveform.unsqueeze(1)
    if atoms.structure == "SF":
        assert atoms.template is not None
        return atoms.template
    raise ValueError("the rank-1 structure has no alternating learner")


def _resolve(cell: Cell | str) -> Cell:
    if isinstance(cell, Cell):
        return cell
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}; the cells are {CELL_NAMES}")
    return CELLS[cell]


def _check(cell: Cell, atoms: CellAtoms, geometry: Geometry) -> None:
    """Refuse atoms of another structure or geometry than the cell."""
    if atoms.structure != cell.structure:
        raise ValueError(f"cell {cell.name} needs atoms of the structure {cell.structure}, got {atoms.structure}")
    reference = atoms.template if atoms.waveform is None else atoms.waveform
    assert reference is not None
    if atoms.n_atoms != geometry.n_atoms or reference.shape[-1] != geometry.atom_len:
        raise ValueError(f"atoms {tuple(reference.shape)} do not match K={geometry.n_atoms}, L={geometry.atom_len}")
    if atoms.pattern is not None and atoms.pattern.shape[-1] != geometry.n_projections:
        raise ValueError(
            f"atoms have {atoms.pattern.shape[-1]} spatial patterns, the geometry {geometry.n_projections}"
        )
    if cell.structure == "SR" and not cell.gradient:
        raise ValueError("the rank-1 structure has gradient cells only")


def _gradient_spec(cell: Cell, geometry: Geometry, budget: Budget) -> GradientSpec:
    return GradientSpec(
        structure=cell.structure,
        n_atoms=geometry.n_atoms,
        atom_len=geometry.atom_len,
        pool_len=geometry.pool_len,
        n_projections=geometry.n_projections if cell.structure == "SR" else 1,
        normalized=cell.rule == "L5n",
        steps=budget.steps,
        lr=budget.lr,
        segments_per_step=budget.segments_per_step,
    )


def fit_cell(
    cell: Cell | str,
    rows: torch.Tensor,
    initial: CellAtoms,
    geometry: Geometry,
    budget: Budget | None = None,
    seed: int = 0,
    device: str = "auto",
) -> FittedCell:
    """Fit a cell to training segments ``(B, C, T)`` from the given atoms.

    Parameters
    ----------
    cell : Cell | str
        The cell or its name.
    rows : torch.Tensor
        Training segments, already scaled.
    initial : CellAtoms
        The start, in the structure of the cell.
    geometry : Geometry
        ``K``, ``L``, ``P`` and ``D``.
    budget : Budget | None
        Passes or steps; ``None`` uses the defaults.
    seed : int
        Seed of the batch order of a gradient rule. An alternating rule has no random part.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.

    Returns
    -------
    FittedCell
        The cell with its learned dictionary and the record of the fit.

    Raises
    ------
    ValueError
        If the cell is unknown or the start does not match the cell and the geometry.
    """
    cell = _resolve(cell)
    budget = budget or Budget()
    _check(cell, initial, geometry)
    if cell.gradient:
        fit = fit_gradient(rows, initial, _gradient_spec(cell, geometry, budget), seed, device)
        record: dict[str, Any] = {"steps": fit.steps, "epochs": fit.epochs, "step_losses": list(fit.step_losses)}
        return FittedCell(cell, geometry, fit.atoms, record, fit.model)
    single = cell.structure == "S1"
    spec = AlternatingSpec(cell.rule, geometry.atom_len, geometry.pool_len, single, budget.max_passes, budget.tol)
    result = fit_alternating(rows.to(resolve_device(device)), _unit_atoms(initial), spec)
    atoms = CellAtoms("S1", waveform=result.atoms.squeeze(1)) if single else CellAtoms("SF", template=result.atoms)
    record = {"passes": result.passes, "converged": result.converged, "history": list(result.history)}
    return FittedCell(cell, geometry, atoms, record)


def frozen_cell(cell: Cell | str, atoms: CellAtoms, geometry: Geometry, device: str = "auto") -> FittedCell:
    """A cell that encodes with the given dictionary, without a fit.

    It serves the frozen start of a run and the oracle (the planted dictionary).
    The atoms are put under the constraint of the structure.

    Raises
    ------
    ValueError
        If the cell is unknown or the atoms do not match the cell and the geometry.
    """
    cell = _resolve(cell)
    _check(cell, atoms, geometry)
    if cell.gradient:
        model = build_model(_gradient_spec(cell, geometry, Budget()), device=device)
        set_model_atoms(model, atoms)
        return FittedCell(cell, geometry, model_atoms(model, cell.structure), {}, model)
    target = resolve_device(device)
    if cell.structure == "S1":
        assert atoms.waveform is not None
        return FittedCell(cell, geometry, CellAtoms("S1", waveform=unit_norm(atoms.waveform.to(target))))
    assert atoms.template is not None
    return FittedCell(cell, geometry, CellAtoms("SF", template=unit_norm(atoms.template.to(target))))

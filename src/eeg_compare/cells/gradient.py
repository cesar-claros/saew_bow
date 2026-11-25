"""Gradient cells of the comparison (rule L5): the comparison adapters with a given start.

The three structures use the adapters of the retained models:

- ``S1``: the full-template adapter on single-channel rows. Every channel of
  every segment is one row with one channel, so all channels share the atoms.
- ``SF``: the full-template adapter on the segments.
- ``SR``: the separable adapter.

All of them score the extended loss support, mask the segment edges, take the
signal as it comes and do not re-center atoms. The budget counts optimizer
steps. One step holds ``segments_per_step`` segments; for ``S1`` it holds the
same number of samples as single-channel rows, drawn from all segments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from eeg_compare.adapters.models import (
    ComparisonSeparableSAE,
    ComparisonSeparableSpec,
    ComparisonWaveformSAE,
    ComparisonWaveformSpec,
)
from eeg_compare.cells.common import CellAtoms, Structure, Tokens, same_padding
from eeg_compare.models.waveform_sae import WaveformTrainSpec

__all__ = [
    "GradientFit",
    "GradientSpec",
    "build_model",
    "encode_gradient",
    "fit_gradient",
    "model_atoms",
    "separable_start",
    "set_model_atoms",
]

Model = ComparisonWaveformSAE | ComparisonSeparableSAE
_PROVENANCE = {"train_subjects": ["synthetic"]}


@dataclass(frozen=True)
class GradientSpec:
    """Configuration of one gradient fit.

    Parameters
    ----------
    structure : str
        ``S1``, ``SF`` or ``SR``.
    n_atoms, atom_len, pool_len : int
        Geometry ``K``, ``L`` and ``P``.
    n_projections : int
        Spatial patterns ``D`` per atom; used by ``SR`` only.
    normalized : bool
        Use the window-normalized loss.
    steps : int
        Smallest number of optimizer steps. The fit runs complete passes, so it
        can take a few steps more.
    lr : float
        Learning rate of Adam.
    segments_per_step : int
        Segments per optimizer step.
    steps_per_batch : int
        Steps that one batch of the loader holds; it sets the loader batch only.
    """

    structure: Structure
    n_atoms: int
    atom_len: int
    pool_len: int
    n_projections: int = 1
    normalized: bool = False
    steps: int = 3000
    lr: float = 1e-3
    segments_per_step: int = 2
    steps_per_batch: int = 16

    def __post_init__(self) -> None:
        if self.structure not in ("S1", "SF", "SR"):
            raise ValueError(f"unknown structure {self.structure!r}")
        for name in ("n_atoms", "atom_len", "pool_len", "n_projections", "steps", "segments_per_step"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if isinstance(self.steps_per_batch, bool) or not isinstance(self.steps_per_batch, int):
            raise ValueError(f"steps_per_batch must be an integer, got {self.steps_per_batch!r}")
        if self.steps_per_batch < 1:
            raise ValueError(f"steps_per_batch must be positive, got {self.steps_per_batch!r}")
        if not self.lr >= 0:
            raise ValueError(f"lr must not be negative, got {self.lr!r}")


@dataclass
class GradientFit:
    """Result of ``fit_gradient``.

    Parameters
    ----------
    model : ComparisonWaveformSAE | ComparisonSeparableSAE
        The fitted adapter.
    atoms : CellAtoms
        Its dictionary in the form of the structure.
    steps : int
        Optimizer steps that ran.
    epochs : int
        Passes over the training rows.
    step_losses : tuple[float, ...]
        Training loss of every step, in order.
    """

    model: Model
    atoms: CellAtoms
    steps: int
    epochs: int
    step_losses: tuple[float, ...]


def _model_rows(rows: torch.Tensor, structure: Structure) -> torch.Tensor:
    """The rows that the adapter of a structure reads: ``(B * C, 1, T)`` for ``S1``, else ``(B, C, T)``."""
    if rows.ndim != 3:
        raise ValueError(f"expected segments of shape (B, C, T), got {tuple(rows.shape)}")
    return rows.flatten(0, 1).unsqueeze(1) if structure == "S1" else rows


def build_model(spec: GradientSpec, epochs: int = 1, row_batch: int = 2, seed: int = 0, device: str = "auto") -> Model:
    """An unfitted adapter with the options of the comparison cells.

    Parameters
    ----------
    spec : GradientSpec
        Structure, geometry and loss.
    epochs : int
        Passes of a later fit.
    row_batch : int
        Rows of the adapter per optimizer step.
    seed : int
        Seed of the batch order.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.
    """
    train_spec = WaveformTrainSpec(epochs=epochs, lr=spec.lr, segment_batch=row_batch, recenter_atoms=False)
    normalization = "window" if spec.normalized else "none"
    if spec.structure == "SR":
        separable = ComparisonSeparableSpec(
            n_atoms=spec.n_atoms,
            atom_len=spec.atom_len,
            pool_len=spec.pool_len,
            top_r=1,
            n_projections=spec.n_projections,
            loss_support="extended",
            loss_normalization=normalization,
        )
        return ComparisonSeparableSAE(spec=separable, train_spec=train_spec, random_state=seed, device=device)
    full = ComparisonWaveformSpec(
        n_atoms=spec.n_atoms,
        atom_len=spec.atom_len,
        pool_len=spec.pool_len,
        top_r=1,
        loss_support="extended",
        loss_normalization=normalization,
    )
    return ComparisonWaveformSAE(spec=full, train_spec=train_spec, random_state=seed, device=device)


def _packed(model: Model, atoms: CellAtoms) -> torch.Tensor:
    """The atoms of a cell in the parameter form of its adapter."""
    if isinstance(model, ComparisonSeparableSAE):
        if atoms.waveform is None or atoms.pattern is None:
            raise ValueError("the separable adapter needs waveforms and spatial patterns")
        return model._pack(atoms.waveform, atoms.pattern)
    if atoms.structure == "S1":
        assert atoms.waveform is not None
        return atoms.waveform.unsqueeze(1)
    if atoms.template is None:
        raise ValueError("the full-template adapter needs templates")
    return atoms.template


def set_model_atoms(model: Model, atoms: CellAtoms) -> None:
    """Install the atoms of a cell as the dictionary of its adapter, without a fit."""
    model.set_atoms(_packed(model, atoms))


def model_atoms(model: Model, structure: Structure) -> CellAtoms:
    """The dictionary of an adapter in the form of the structure, detached."""
    if isinstance(model, ComparisonSeparableSAE):
        waveform, pattern = model.factors()
        return CellAtoms("SR", waveform=waveform, pattern=pattern)
    dictionary = model._dictionary().detach()
    if structure == "S1":
        return CellAtoms("S1", waveform=dictionary.squeeze(1))
    return CellAtoms("SF", template=dictionary)


def separable_start(patches: torch.Tensor, n_atoms: int, n_projections: int) -> CellAtoms:
    """The separable start that the retained patch initialization builds from patches ``(K * D, C, L)``."""
    if patches.ndim != 3 or patches.shape[0] != n_atoms * n_projections:
        raise ValueError(f"expected {n_atoms * n_projections} patches of shape (C, L), got {tuple(patches.shape)}")
    spec = GradientSpec("SR", n_atoms, int(patches.shape[-1]), int(patches.shape[-1]), n_projections=n_projections)
    model = build_model(spec, device="cpu")
    assert isinstance(model, ComparisonSeparableSAE)
    waveform, pattern = model.factors_from_patches(patches)
    return CellAtoms("SR", waveform=waveform.to(patches.device), pattern=pattern.to(patches.device))


def fit_gradient(
    rows: torch.Tensor, initial: CellAtoms, spec: GradientSpec, seed: int = 0, device: str = "auto"
) -> GradientFit:
    """Fit a gradient cell to segments, from the given atoms.

    Parameters
    ----------
    rows : torch.Tensor
        Training segments ``(B, C, T)``, already scaled.
    initial : CellAtoms
        The start, in the form of ``spec.structure``.
    spec : GradientSpec
        Structure, geometry, loss and budget.
    seed : int
        Seed of the batch order.
    device : str
        ``cpu``, ``cuda``, ``cuda:N`` or ``auto``.

    Returns
    -------
    GradientFit
        The fitted adapter, its atoms, and the loss of every step.

    Raises
    ------
    ValueError
        If the start has another structure than the cell.
    """
    if initial.structure != spec.structure:
        raise ValueError(f"the start has the structure {initial.structure}, the cell {spec.structure}")
    items = _model_rows(rows, spec.structure).detach().cpu()
    rows_per_step = spec.segments_per_step * (rows.shape[1] if spec.structure == "S1" else 1)
    batch = rows_per_step * spec.steps_per_batch
    n_items = items.shape[0]
    if n_items == 0:
        raise ValueError("a fit needs at least one training segment")
    steps_per_epoch = (n_items // batch) * spec.steps_per_batch + math.ceil((n_items % batch) / rows_per_step)
    epochs = math.ceil(spec.steps / steps_per_epoch)
    model = build_model(spec, epochs=epochs, row_batch=rows_per_step, seed=seed, device=device)
    model.initial_atoms = _packed(model, initial).to(items.dtype)
    loader = DataLoader(TensorDataset(items, torch.zeros(n_items, dtype=torch.long)), batch_size=batch)
    model.fit_unsupervised(loader, provenance=dict(_PROVENANCE))
    steps = int(sum(record["n_steps"] for record in model.history))
    return GradientFit(model, model_atoms(model, spec.structure), steps, epochs, tuple(model.step_losses))


@torch.no_grad()
def encode_gradient(model: Model, rows: torch.Tensor, structure: Structure, chunk: int = 256) -> Tokens:
    """One token per segment and pooling window from the placements of an adapter.

    For ``S1`` every channel has its own placement in a window, and the token is
    that of the channel with the largest amplitude. Equal amplitudes prefer the
    lower channel.

    Parameters
    ----------
    model : ComparisonWaveformSAE | ComparisonSeparableSAE
        An adapter with a dictionary and one atom per window.
    rows : torch.Tensor
        Segments ``(B, C, T)``.
    structure : str
        ``S1``, ``SF`` or ``SR``.
    chunk : int
        Adapter rows per block.

    Returns
    -------
    Tokens
        ``(B, M)`` tokens on the device of the adapter. The value and the
        response are the amplitude of the placement.

    Raises
    ------
    ValueError
        If the adapter selects more than one atom per window.
    """
    if model.spec.top_r != 1:
        raise ValueError("tokens need one atom per pooling window (top_r=1)")
    items = _model_rows(rows, structure)
    amplitudes, positions, winners = [], [], []
    for first in range(0, items.shape[0], chunk):
        placed = model.placements(items[first : first + chunk])
        amplitude, atom = placed.amplitude.max(dim=1)
        amplitudes.append(amplitude)
        positions.append(torch.stack([atom, placed.position.gather(1, atom.unsqueeze(1)).squeeze(1)], dim=-1))
        if placed.projection is not None:
            winners.append(placed.projection.gather(1, atom.unsqueeze(1)).squeeze(1))
    value = torch.cat(amplitudes)
    atom, center = torch.cat(positions).unbind(-1)
    projection = torch.cat(winners) if winners else None
    channel: torch.Tensor | None = None
    if structure == "S1":
        n_segments, n_channels = rows.shape[0], rows.shape[1]
        value, atom, center = (t.reshape(n_segments, n_channels, -1) for t in (value, atom, center))
        value, best = value.max(dim=1, keepdim=True)
        atom, center, value = atom.gather(1, best).squeeze(1), center.gather(1, best).squeeze(1), value.squeeze(1)
        channel = best.squeeze(1)
    active = value > 0
    null = torch.full_like(atom, -1)
    start = center - same_padding(model.spec.atom_len)[0]
    return Tokens(
        atom=torch.where(active, atom, null),
        start=torch.where(active, start, null),
        value=value,
        response=value,
        channel=None if channel is None else torch.where(active, channel, null),
        projection=None if projection is None else torch.where(active, projection, null),
    )

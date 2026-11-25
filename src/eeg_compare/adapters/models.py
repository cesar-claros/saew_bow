"""Comparison adapters of the retained waveform models.

The reference classes in ``eeg_compare.models`` stay unchanged. The classes here
apply the changes that the comparison plan requires. The options are fields of
the spec, so a checkpoint records them and refuses a different setting.

- ``input_scaling="external"``: the loader already scaled the signal per
  recording, so the model does not scale each segment again. The validity check
  and the zeroing of unusable rows are kept. ``"internal"`` keeps the retained
  per-segment median/MAD scaling and clip.
- ``edge_mask=True``: the responses at the ``L - 1`` positions whose template
  overlaps the zero padding are set to zero before selection. Only placements
  inside the real samples can then hold a positive code (section 3 of the
  mathematics note). A placed template never reaches the padding, so the padded
  targets of the separable loss add nothing to it.
- ``placements``: the amplitude, position and, for the separable model, the
  spatial winner of every selected placement, with hard counts, amplitude sums
  and counts by spatial winner as readouts.

- ``loss_support="extended"`` (full-template adapter): score every decoded
  pooling window over its ``P + L - 1`` samples, as the separable reference
  does, so that the two structures have one loss. ``"central"`` keeps the
  retained loss on the ``P`` samples of each window.
- ``loss_normalization="window"``: divide the squared error of every extended
  window by the energy of its target (the window-normalized loss of the plan).
  ``"none"`` keeps the retained loss.
- ``initial_atoms``: a fit starts from these atoms instead of drawing patches,
  so that the cells of the comparison share one start. ``set_atoms`` installs a
  dictionary without a fit. ``step_losses`` holds the loss of every training
  step of the last fit.

With the edge mask, the data-patch initialization also draws its patches from
inside the real samples, so no atom starts from a patch that holds padding.

These are deviations from the reference, with their own checkpoint tags. They
score complete segments only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eeg_compare.models.separable_sae import SeparableSAE, SeparableSpec
from eeg_compare.models.waveform_sae import WaveformSAE, WaveformSpec, WaveformTrainSpec
from eeg_compare.preprocessing.robust import valid_rows

__all__ = [
    "ComparisonSeparableSAE",
    "ComparisonSeparableSpec",
    "ComparisonWaveformSAE",
    "ComparisonWaveformSpec",
    "Placements",
    "SignalPairs",
]

_SCALINGS = ("external", "internal")
_SUPPORTS = ("central", "extended")
_NORMALIZATIONS = ("none", "window")
_WINDOW_ENERGY_FLOOR = 1e-12
WAVEFORM_TAG = "eeg_compare.comparison.waveform_sae"
SEPARABLE_TAG = "eeg_compare.comparison.separable_sae"
CHECKPOINT_FORMAT = 1
_INIT_POOL_FACTOR = 8  # candidate patches per atom, the value of the retained initialization


@dataclass(frozen=True)
class ComparisonWaveformSpec(WaveformSpec):
    """``WaveformSpec`` plus the input scaling, the edge mask and the loss options of the comparison."""

    input_scaling: str = "external"
    edge_mask: bool = True
    loss_support: str = "central"
    loss_normalization: str = "none"


@dataclass(frozen=True)
class ComparisonSeparableSpec(SeparableSpec):
    """``SeparableSpec`` plus the input scaling, the edge mask and the loss normalization of the comparison."""

    input_scaling: str = "external"
    edge_mask: bool = True
    loss_normalization: str = "none"


@dataclass(frozen=True)
class Placements:
    """The selected placements of a batch: one slot per atom and pooling window.

    Parameters
    ----------
    amplitude : torch.Tensor
        ``(B, K, M)`` code value of atom ``k`` in pooling window ``m``; zero where
        the atom was not selected or its best response was not positive.
    position : torch.Tensor
        ``(B, K, M)`` zero-based sample of the atom's best response in the segment.
        It has a meaning only where ``amplitude`` is positive.
    projection : torch.Tensor | None
        ``(B, K, M)`` zero-based spatial pattern that won at ``position``;
        ``None`` for full multichannel atoms.
    """

    amplitude: torch.Tensor
    position: torch.Tensor
    projection: torch.Tensor | None

    @property
    def active(self) -> torch.Tensor:
        """Boolean ``(B, K, M)``: the slots that hold a placement."""
        return self.amplitude > 0

    @staticmethod
    def _take(values: torch.Tensor, windows: Sequence[int] | None) -> torch.Tensor:
        return values if windows is None else values[..., list(windows)]

    def hard_counts(self, windows: Sequence[int] | None = None) -> torch.Tensor:
        """Placements per atom ``(B, K)``, over all pooling windows or the listed ones."""
        return self._take(self.active, windows).sum(-1)

    def amplitude_sums(self, windows: Sequence[int] | None = None) -> torch.Tensor:
        """Summed amplitudes per atom ``(B, K)``; over all windows it is ``M`` times the retained ``forward``."""
        return self._take(self.amplitude, windows).sum(-1)

    def counts_by_projection(self, n_projections: int, windows: Sequence[int] | None = None) -> torch.Tensor:
        """Placements per atom and spatial winner ``(B, K, D)``.

        Raises
        ------
        ValueError
            If the placements carry no spatial winner.
        """
        if self.projection is None:
            raise ValueError("these placements have no spatial winner")
        active = self._take(self.active, windows)
        winners = F.one_hot(self._take(self.projection, windows), n_projections)
        return (winners * active.unsqueeze(-1)).sum(-2)


class _DictDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


class SignalPairs(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """``(signal, target)`` items of a dictionary dataset, the item format of the retained trainer.

    The retained trainer rebuilds its loader with uniform shuffling. This view
    does not add patient-balanced sampling.
    """

    def __init__(self, dataset: _DictDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[index]
        return item["signal"], item["target"]


def _check_comparison(spec: WaveformSpec, train_spec: WaveformTrainSpec, expected: type) -> None:
    """Refuse a spec of another class, unknown options and the crop modes that the edge mask cannot serve."""
    if not isinstance(spec, expected):
        raise TypeError(f"expected a {expected.__name__}, got {type(spec).__name__}")
    scaling, mask = getattr(spec, "input_scaling", None), getattr(spec, "edge_mask", None)
    if scaling not in _SCALINGS:
        raise ValueError(f"spec.input_scaling must be one of {_SCALINGS}, got {scaling!r}")
    if not isinstance(mask, bool):
        raise ValueError(f"spec.edge_mask must be a boolean, got {mask!r}")
    if train_spec.crop_sampling != "full":
        raise ValueError("the comparison adapters score complete segments (crop_sampling='full')")
    support, normalization = getattr(spec, "loss_support", None), getattr(spec, "loss_normalization", None)
    if support not in _SUPPORTS:
        raise ValueError(f"spec.loss_support must be one of {_SUPPORTS}, got {support!r}")
    if normalization not in _NORMALIZATIONS:
        raise ValueError(f"spec.loss_normalization must be one of {_NORMALIZATIONS}, got {normalization!r}")
    if normalization == "window" and support != "extended":
        raise ValueError("the window-normalized loss needs loss_support='extended'")


def _scored_step(
    model: WaveformSAE,
    retained: Any,
    rows: torch.Tensor,
    valid: torch.Tensor,
    atoms: torch.Tensor,
    collect: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The segment step of an adapter: the retained loss or the window-normalized loss, with a loss trace.

    ``retained`` is the ``_segment_step`` of the parent class. With
    ``loss_normalization="window"`` the loss is half the mean, over the extended
    windows with energy, of the squared error of a window divided by the energy
    of its target. For R = 1 this is half of one minus the squared cosine of the
    window and its placed template. The statistics are those of the retained
    step. A step that carries a gradient appends its loss to ``step_losses``.
    """
    if model.spec.loss_normalization == "none":  # type: ignore[attr-defined]
        loss, stats, per_atom = retained(rows, valid, atoms, collect)
    else:
        code, selected, reconstruction, target, estimate = model._encode_reconstruct(rows, valid, atoms)
        if target.ndim != 4:
            raise ValueError("the window-normalized loss needs targets of shape (B, C, M, P + L - 1)")
        energy = target.pow(2).sum(dim=(1, 3))
        error = (target - estimate).pow(2).sum(dim=(1, 3))
        usable = energy > _WINDOW_ENERGY_FLOOR
        ratio = torch.where(usable, error / energy.clamp_min(_WINDOW_ENERGY_FLOOR), torch.zeros_like(error))
        loss = 0.5 * ratio.sum() / usable.sum().clamp_min(1)
        with torch.no_grad():
            stats, per_atom = model._statistics(rows, reconstruction, code, selected)
            if collect:
                model._collect_reseed_patches(rows - reconstruction)
    if loss.requires_grad:
        model.step_losses.append(float(loss.detach()))  # type: ignore[attr-defined]
    return loss, stats, per_atom


def _external_rows(model: WaveformSAE, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The retained ``_rows`` without the per-segment scaling: rows ``(B, C, T')`` and the usable-channel mask.

    Constant and non-finite rows come back as zeros with a false mask, the optional
    first difference is applied, and ``T'`` is ``T`` trimmed to a multiple of
    ``pool_len``, exactly as in the retained method.
    """
    raw = X.to(model.device)
    if raw.ndim != 3:
        raise ValueError(f"expected windows of shape (B, C, T), got {tuple(raw.shape)}")
    n_windows, n_channels, _ = raw.shape
    if model.n_channels is not None and n_channels != model.n_channels:
        raise ValueError(f"windows have {n_channels} channels but the dictionary has {model.n_channels}")
    flat = raw.flatten(0, 1)
    valid = valid_rows(flat)
    if model.spec.pre_emphasis == "diff":
        flat = flat.diff(dim=-1, prepend=flat[:, :1])
    rows = torch.where(valid.unsqueeze(-1), flat, torch.zeros_like(flat))
    n_times = rows.shape[-1] // model.spec.pool_len * model.spec.pool_len
    if n_times == 0:
        raise ValueError(f"windows of {rows.shape[-1]} samples are shorter than pool_len={model.spec.pool_len}")
    return rows[:, :n_times].view(n_windows, n_channels, n_times), valid.view(n_windows, n_channels)


def _mask_edges(responses: torch.Tensor, padding: tuple[int, int]) -> torch.Tensor:
    """Zero the responses ``(..., T)`` at the first ``p_L`` and the last ``p_R`` positions.

    Raises
    ------
    ValueError
        If no position is left, which means the segment is shorter than one atom.
    """
    left, right = padding
    length = responses.shape[-1]
    if length <= left + right:
        raise ValueError(f"a segment of {length} samples holds no complete placement of the atom")
    keep = torch.ones(length, dtype=responses.dtype, device=responses.device)
    keep[:left] = 0
    keep[length - right :] = 0
    return responses * keep


def _inside_patches(
    model: WaveformSAE, dataloader: DataLoader, generator: torch.Generator, count: int | None
) -> torch.Tensor:
    """The retained patch sampler restricted to patches ``(count, C, L)`` that hold real samples only.

    The retained sampler draws encoder receptive fields from the zero-padded
    segment, so a patch near a segment end holds padding. Here every patch lies
    inside the segment. The pool size, the energy gate and the final draw are
    those of the retained sampler.
    """
    spec = model.spec
    count = spec.n_atoms if count is None else count
    per_batch = max(math.ceil(_INIT_POOL_FACTOR * spec.n_atoms / max(len(dataloader), 1)), 1)
    pool = []
    for x, _ in tqdm(dataloader, desc="Waveform SAE patch initialization (inside the segment)"):
        rows, _ = model._rows(x)
        if rows.shape[-1] < spec.atom_len:
            raise ValueError(f"segments of {rows.shape[-1]} samples are shorter than atom_len={spec.atom_len}")
        views = rows.unfold(-1, spec.atom_len, 1)
        window = torch.randint(0, views.shape[0], (per_batch,), generator=generator).to(rows.device)
        start = torch.randint(0, views.shape[2], (per_batch,), generator=generator).to(rows.device)
        patches = views[window, :, start]
        pool.append(patches[model._gate(patches.pow(2).sum(dim=(1, 2)))])
    candidates = torch.cat(pool)
    if candidates.shape[0] < count:
        raise ValueError(
            f"patch initialization needs at least {count} non-flat patches, got {candidates.shape[0]}; "
            "supply more windows"
        )
    chosen = torch.randperm(candidates.shape[0], generator=generator)[:count].to(candidates.device)
    return candidates[chosen]


def _pooled(model: WaveformSAE, responses: torch.Tensor, projection: torch.Tensor | None) -> Placements:
    """Placements from responses ``(B, K, T)`` and winners ``(B, K, T)``, by the rule of the retained ``_selection``."""
    pool = model.spec.pool_len
    if responses.shape[-1] == 0 or responses.shape[-1] % pool:
        raise ValueError("response length must be a positive multiple of pool_len")
    values, local = F.relu(responses).unflatten(-1, (-1, pool)).max(dim=-1)
    selected = model._select_atoms(values)
    position = local + torch.arange(values.shape[-1], device=values.device) * pool
    winner = None if projection is None else projection.gather(-1, position)
    return Placements(values * selected, position, winner)


class ComparisonWaveformSAE(WaveformSAE):
    """``WaveformSAE`` with external input scaling, the edge mask and a placement readout.

    Parameters are those of ``WaveformSAE``; ``spec`` is a ``ComparisonWaveformSpec``.

    Raises
    ------
    TypeError
        If ``spec`` is not a ``ComparisonWaveformSpec``.
    ValueError
        As ``WaveformSAE``, and for an unknown ``input_scaling``, a non-boolean
        ``edge_mask`` or a crop mode other than ``full``.
    """

    spec: ComparisonWaveformSpec
    model_tag = WAVEFORM_TAG
    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(
        self,
        spec: ComparisonWaveformSpec | None = None,
        train_spec: WaveformTrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        pretrained: str | Path | None = None,
        chunk_windows: int = 8,
    ) -> None:
        super().__init__(spec or ComparisonWaveformSpec(), train_spec, random_state, device, pretrained, chunk_windows)
        self.initial_atoms: torch.Tensor | None = None
        self.step_losses: list[float] = []

    def _check_specs(self) -> None:
        super()._check_specs()
        _check_comparison(self.spec, self.train_spec, ComparisonWaveformSpec)

    def set_atoms(self, atoms: torch.Tensor) -> None:
        """Install atoms ``(K, C, L)`` as the dictionary, under the atom constraint; the module stays unfitted.

        Raises
        ------
        ValueError
            If the atoms do not have ``n_atoms`` atoms of ``atom_len`` samples.
        """
        if atoms.ndim != 3 or atoms.shape[0] != self.spec.n_atoms or atoms.shape[-1] != self.spec.atom_len:
            raise ValueError(
                f"expected atoms of shape ({self.spec.n_atoms}, C, {self.spec.atom_len}), got {tuple(atoms.shape)}"
            )
        self.atoms = nn.Parameter(self._unit(self._constrain(atoms.detach().clone().to(self.device))))

    def _init_atoms(self, dataloader: DataLoader, generator: torch.Generator) -> None:
        """Start from ``initial_atoms`` when they are set, else as the retained model; clear the loss trace."""
        self.step_losses = []
        if self.initial_atoms is None:
            super()._init_atoms(dataloader, generator)
        else:
            self.set_atoms(self.initial_atoms)

    def _encode_reconstruct(
        self, rows: torch.Tensor, valid: torch.Tensor, atoms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The retained method, or the extended loss support of the separable reference.

        Under ``extended`` the target is the padded segment cut into overlapping
        windows of ``P + L - 1`` samples and the estimate is the synthesis of every
        pooling window without the crop, both ``(B, C, M, P + L - 1)``. The segment
        reconstruction ``(B, C, T)`` is the central part of every window, as in the
        retained decoder.
        """
        if self.spec.loss_support == "central":
            return super()._encode_reconstruct(rows, valid, atoms)
        padded = F.pad(rows, self.padding)
        code, selected = self._selection(F.conv1d(padded, atoms))
        synthesis = F.conv_transpose1d(self._pool_batch(code), atoms)
        extended = synthesis.unflatten(0, (code.shape[0], -1)).transpose(1, 2)
        extended = extended * valid.to(rows.dtype)[..., None, None]
        left, pool = self.padding[0], self.spec.pool_len
        reconstruction = extended[..., left : left + pool].flatten(-2)
        target = padded.unfold(-1, pool + self.spec.atom_len - 1, pool)
        return code, selected, reconstruction, target, extended

    def _segment_step(
        self, rows: torch.Tensor, valid: torch.Tensor, atoms: torch.Tensor, collect: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _scored_step(self, super()._segment_step, rows, valid, atoms, collect)

    def _rows(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _external_rows(self, X) if self.spec.input_scaling == "external" else super()._rows(X)

    def _selection(self, responses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._selection(_mask_edges(responses, self.padding) if self.spec.edge_mask else responses)

    def _sample_patches(
        self, dataloader: DataLoader, generator: torch.Generator, count: int | None = None
    ) -> torch.Tensor:
        if not self.spec.edge_mask:
            return super()._sample_patches(dataloader, generator, count)
        return _inside_patches(self, dataloader, generator, count)

    @torch.no_grad()
    def placements(self, X: torch.Tensor) -> Placements:
        """The selected placements of complete segments ``(B, C, T)``; no spatial winner."""
        rows, _ = self._full_rows(X)
        responses = self._responses(rows, self._dictionary())
        if self.spec.edge_mask:
            responses = _mask_edges(responses, self.padding)
        return _pooled(self, responses, None)


class ComparisonSeparableSAE(SeparableSAE):
    """``SeparableSAE`` with external input scaling, the edge mask and a placement readout.

    Parameters are those of ``SeparableSAE``; ``spec`` is a ``ComparisonSeparableSpec``.

    Raises
    ------
    TypeError
        If ``spec`` is not a ``ComparisonSeparableSpec``.
    ValueError
        As ``SeparableSAE``, and for an unknown ``input_scaling`` or a non-boolean
        ``edge_mask``.
    """

    spec: ComparisonSeparableSpec
    model_tag = SEPARABLE_TAG
    checkpoint_format = CHECKPOINT_FORMAT

    def __init__(
        self,
        spec: ComparisonSeparableSpec | None = None,
        train_spec: WaveformTrainSpec | None = None,
        random_state: int = 42,
        device: str | None = "auto",
        pretrained: str | Path | None = None,
        chunk_windows: int = 8,
    ) -> None:
        super().__init__(spec or ComparisonSeparableSpec(), train_spec, random_state, device, pretrained, chunk_windows)
        self.initial_atoms: torch.Tensor | None = None
        self.step_losses: list[float] = []

    def _check_specs(self) -> None:
        super()._check_specs()
        _check_comparison(self.spec, self.train_spec, ComparisonSeparableSpec)

    def set_atoms(self, atoms: torch.Tensor) -> None:
        """Install packed factors ``(K, L + C * D)`` as the dictionary, under equation (4); the module stays unfitted.

        Raises
        ------
        ValueError
            If the packed atoms do not match ``n_atoms``, ``atom_len`` and ``n_projections``.
        """
        spec = self.spec
        width = atoms.shape[1] - spec.atom_len if atoms.ndim == 2 else 0
        if (
            atoms.ndim != 2
            or atoms.shape[0] != spec.n_atoms
            or width < spec.n_projections
            or width % spec.n_projections
        ):
            raise ValueError(
                f"expected packed atoms of shape ({spec.n_atoms}, {spec.atom_len} + C * {spec.n_projections}), "
                f"got {tuple(atoms.shape)}"
            )
        self.atoms = nn.Parameter(self._constrain(atoms.detach().clone().to(self.device)))

    def set_factors(self, waveform: torch.Tensor, pattern: torch.Tensor) -> None:
        """Install waveforms ``(K, L)`` and spatial patterns ``(K, C, D)`` as the dictionary."""
        self.set_atoms(self._pack(waveform, pattern))

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The constrained waveforms ``(K, L)`` and spatial patterns ``(K, C, D)``, detached."""
        return self._split(self._dictionary().detach())

    def factors_from_patches(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The factors that the retained patch initialization builds from patches ``(K * D, C, L)``.

        Patch ``k`` gives the waveform of atom ``k`` and patch ``d * K + k`` its
        spatial pattern ``d``, as in the retained ``_init_atoms``.
        """
        grouped = patches.unflatten(0, (self.spec.n_projections, self.spec.n_atoms))
        return self._split(self._atoms_from_patches(grouped))

    def _init_atoms(self, dataloader: DataLoader, generator: torch.Generator) -> None:
        """Start from ``initial_atoms`` when they are set, else as the retained model; clear the loss trace."""
        self.step_losses = []
        if self.initial_atoms is None:
            super()._init_atoms(dataloader, generator)
        else:
            self.set_atoms(self.initial_atoms)

    def _segment_step(
        self, rows: torch.Tensor, valid: torch.Tensor, atoms: torch.Tensor, collect: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _scored_step(self, super()._segment_step, rows, valid, atoms, collect)

    def _rows(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _external_rows(self, X) if self.spec.input_scaling == "external" else super()._rows(X)

    def _selection(self, responses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super()._selection(_mask_edges(responses, self.padding) if self.spec.edge_mask else responses)

    def _sample_patches(
        self, dataloader: DataLoader, generator: torch.Generator, count: int | None = None
    ) -> torch.Tensor:
        if not self.spec.edge_mask:
            return super()._sample_patches(dataloader, generator, count)
        return _inside_patches(self, dataloader, generator, count)

    @torch.no_grad()
    def placements(self, X: torch.Tensor) -> Placements:
        """The selected placements of complete segments ``(B, C, T)``, with the spatial winner of each."""
        rows, _ = self._full_rows(X)
        responses, projection = self._correlate(F.pad(rows, self.padding), self._dictionary()).max(dim=2)
        if self.spec.edge_mask:
            responses = _mask_edges(responses, self.padding)
        return _pooled(self, responses, projection)

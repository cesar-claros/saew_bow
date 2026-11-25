"""Gradient cells of the comparison and the loss options of the adapters; run on the HPC.

The loss identity is gate 3 of the stage 2 protocol: with one atom per pooling
window, the loss of every gradient cell is the energy of the extended windows
minus the squared best responses.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from eeg_compare.adapters.models import (
    ComparisonSeparableSAE,
    ComparisonSeparableSpec,
    ComparisonWaveformSAE,
    ComparisonWaveformSpec,
)
from eeg_compare.cells.common import CellAtoms, extended_windows, unit_norm
from eeg_compare.cells.gradient import (
    GradientSpec,
    build_model,
    encode_gradient,
    fit_gradient,
    model_atoms,
    separable_start,
    set_model_atoms,
)
from eeg_compare.models.waveform_sae import WaveformTrainSpec

PROVENANCE = {"train_subjects": ["synthetic"]}
Adapter = ComparisonWaveformSAE | ComparisonSeparableSAE
N_CHANNELS, ATOM_LEN, POOL_LEN, N_TIMES = 3, 8, 8, 32


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _full(**options: object) -> ComparisonWaveformSAE:
    spec = ComparisonWaveformSpec(n_atoms=3, atom_len=ATOM_LEN, pool_len=POOL_LEN, **options)  # type: ignore[arg-type]
    return ComparisonWaveformSAE(spec=spec, device="cpu")


def _separable(**options: object) -> ComparisonSeparableSAE:
    spec = ComparisonSeparableSpec(  # type: ignore[arg-type]
        n_atoms=3, atom_len=ATOM_LEN, pool_len=POOL_LEN, n_projections=2, **options
    )
    return ComparisonSeparableSAE(spec=spec, device="cpu")


def _energy_and_response(model: Adapter, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Energy of every extended window ``(B, M)`` and the best response of its one selected atom ``(B, M)``."""
    rows, _ = model._full_rows(x)
    windows, _ = extended_windows(rows, ATOM_LEN, POOL_LEN)
    return windows.pow(2).sum(dim=(2, 3)), model.placements(x).amplitude.sum(dim=1)


def _loss(model: Adapter, x: torch.Tensor) -> torch.Tensor:
    rows, valid = model._full_rows(x)
    return model._segment_step(rows, valid, model._dictionary())[0]


def test_loss_identity_of_the_full_template_cells() -> None:
    samples = POOL_LEN + ATOM_LEN - 1
    for channels, seed in ((N_CHANNELS, 1), (1, 2)):  # SF, then S1 on single-channel rows
        model = _full(loss_support="extended")
        model.set_atoms(_randn(3, channels, ATOM_LEN, seed=seed))
        x = _randn(4, channels, N_TIMES, seed=seed + 10)
        energy, response = _energy_and_response(model, x)
        assert bool((response > 0).any()) and energy.shape == response.shape == (4, N_TIMES // POOL_LEN)
        expected = 0.5 * (energy - response.pow(2)).sum() / (4 * channels * energy.shape[1] * samples)
        assert torch.allclose(_loss(model, x), expected, rtol=1e-10, atol=1e-12)


def test_loss_identity_of_the_separable_cells() -> None:
    samples = POOL_LEN + ATOM_LEN - 1
    packed = _randn(3, ATOM_LEN + N_CHANNELS * 2, seed=3)
    x = _randn(4, N_CHANNELS, N_TIMES, seed=13)
    plain = _separable()
    plain.set_atoms(packed)
    energy, response = _energy_and_response(plain, x)
    expected = 0.5 * (energy - response.pow(2)).sum() / (4 * N_CHANNELS * energy.shape[1] * samples)
    assert torch.allclose(_loss(plain, x), expected, rtol=1e-10, atol=1e-12)
    normalized = _separable(loss_normalization="window")
    normalized.set_atoms(packed)
    expected = 0.5 * (1.0 - response.pow(2) / energy).mean()
    assert torch.allclose(_loss(normalized, x), expected, rtol=1e-10, atol=1e-12)
    assert torch.equal(normalized.placements(x).amplitude, plain.placements(x).amplitude)


def test_window_normalized_loss_of_the_full_template_adapter() -> None:
    model = _full(loss_support="extended", loss_normalization="window")
    model.set_atoms(_randn(3, N_CHANNELS, ATOM_LEN, seed=4))
    x = _randn(4, N_CHANNELS, N_TIMES, seed=14)
    energy, response = _energy_and_response(model, x)
    assert torch.allclose(_loss(model, x), 0.5 * (1.0 - response.pow(2) / energy).mean(), rtol=1e-10, atol=1e-12)
    x[1] = 0.0  # a segment without energy has no usable window and adds nothing
    energy, response = _energy_and_response(model, x)
    kept = torch.tensor([0, 2, 3])
    expected = 0.5 * (1.0 - response[kept].pow(2) / energy[kept]).mean()
    assert torch.allclose(_loss(model, x), expected, rtol=1e-10, atol=1e-12)


def test_both_loss_supports_share_the_code_and_the_reconstruction() -> None:
    atoms = _randn(3, N_CHANNELS, ATOM_LEN, seed=5)
    central, extended = _full(), _full(loss_support="extended")
    central.set_atoms(atoms)
    extended.set_atoms(atoms)
    rows, valid = central._full_rows(_randn(4, N_CHANNELS, N_TIMES, seed=15))
    ours = extended._encode_reconstruct(rows, valid, extended._dictionary())
    theirs = central._encode_reconstruct(rows, valid, central._dictionary())
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
    assert torch.allclose(ours[2], theirs[2], atol=1e-12)
    samples = POOL_LEN + ATOM_LEN - 1
    assert ours[3].shape == ours[4].shape == (4, N_CHANNELS, N_TIMES // POOL_LEN, samples)
    assert theirs[3].shape == (4, N_CHANNELS, N_TIMES)


def test_a_fit_starts_from_the_given_atoms_and_traces_every_step() -> None:
    x = _randn(12, N_CHANNELS, N_TIMES, seed=6)
    loader = DataLoader(TensorDataset(x, torch.zeros(12, dtype=torch.long)), batch_size=4)
    frozen = WaveformTrainSpec(epochs=2, lr=0.0, segment_batch=4, recenter_atoms=False)
    initial = _randn(3, N_CHANNELS, ATOM_LEN, seed=7)
    full = ComparisonWaveformSAE(
        spec=ComparisonWaveformSpec(n_atoms=3, atom_len=ATOM_LEN, pool_len=POOL_LEN, loss_support="extended"),
        train_spec=frozen,
        device="cpu",
    )
    full.initial_atoms = initial
    full.fit_unsupervised(loader, provenance=PROVENANCE)
    assert torch.allclose(full._dictionary().detach(), unit_norm(initial), atol=1e-12)
    assert len(full.step_losses) == 6 == sum(int(record["n_steps"]) for record in full.history)
    packed = _randn(3, ATOM_LEN + N_CHANNELS * 2, seed=8)
    separable = ComparisonSeparableSAE(
        spec=ComparisonSeparableSpec(n_atoms=3, atom_len=ATOM_LEN, pool_len=POOL_LEN, n_projections=2),
        train_spec=frozen,
        device="cpu",
    )
    separable.initial_atoms = packed
    separable.fit_unsupervised(loader, provenance=PROVENANCE)
    reference = _separable()
    reference.set_atoms(packed)
    for ours, theirs in zip(separable.factors(), reference.factors()):
        assert torch.allclose(ours, theirs, atol=1e-12)
    assert len(separable.step_losses) == 6
    learning = replace(frozen, lr=1e-2)
    moved = ComparisonWaveformSAE(spec=full.spec, train_spec=learning, device="cpu")
    moved.initial_atoms = initial
    moved.fit_unsupervised(loader, provenance=PROVENANCE)
    assert not torch.allclose(moved._dictionary().detach(), unit_norm(initial), atol=1e-6)


def test_factors_can_be_set_read_and_built_from_patches() -> None:
    model = _separable()
    waveform = unit_norm(_randn(3, ATOM_LEN, seed=9))
    pattern = _randn(3, N_CHANNELS, 2, seed=10)
    pattern = pattern / pattern.norm(dim=1, keepdim=True)
    model.set_factors(waveform, pattern)
    for ours, theirs in zip(model.factors(), (waveform, pattern)):
        assert torch.allclose(ours, theirs, atol=1e-12)
    patches = _randn(6, N_CHANNELS, ATOM_LEN, seed=11)
    start = separable_start(patches, n_atoms=3, n_projections=2)
    assert start.waveform is not None and start.pattern is not None
    assert start.waveform.shape == (3, ATOM_LEN) and start.pattern.shape == (3, N_CHANNELS, 2)
    for atom in range(3):
        top = torch.linalg.svd(patches[atom], full_matrices=False).Vh[0]
        assert abs(float(start.waveform[atom] @ top)) > 1 - 1e-10
        for projection in range(2):
            source = patches[projection * 3 + atom]
            direction = source @ start.waveform[atom]
            assert torch.allclose(start.pattern[atom, :, projection], direction / direction.norm(), atol=1e-10)
            assert float(start.pattern[atom, :, projection] @ source @ start.waveform[atom]) > 0
    with pytest.raises(ValueError, match="packed atoms"):
        model.set_atoms(_randn(3, ATOM_LEN + 5))
    with pytest.raises(ValueError, match="atoms of shape"):
        _full().set_atoms(_randn(2, N_CHANNELS, ATOM_LEN))


def test_spec_guards_of_the_loss_options() -> None:
    with pytest.raises(ValueError, match="loss_support"):
        _full(loss_support="window")
    with pytest.raises(ValueError, match="loss_normalization"):
        _separable(loss_normalization="energy")
    with pytest.raises(ValueError, match="needs loss_support"):
        _full(loss_normalization="window")
    with pytest.raises(ValueError, match="needs loss_support"):
        _separable(loss_normalization="window", loss_support="central")
    with pytest.raises(ValueError, match="structure"):
        GradientSpec("S2", 3, ATOM_LEN, POOL_LEN)  # type: ignore[arg-type]


def test_fit_gradient_counts_steps_and_reads_the_atoms_of_each_structure() -> None:
    rows = _randn(6, N_CHANNELS, N_TIMES, seed=12)
    patches = torch.stack([rows[index % 6, :, 4 : 4 + ATOM_LEN] for index in range(6)])
    starts = {
        "S1": CellAtoms("S1", waveform=unit_norm(patches[:3, 0])),
        "SF": CellAtoms("SF", template=unit_norm(patches[:3])),
        "SR": separable_start(patches, n_atoms=3, n_projections=2),
    }
    shapes = {"S1": {"waveform": (3, ATOM_LEN)}, "SF": {"template": (3, N_CHANNELS, ATOM_LEN)}}
    shapes["SR"] = {"waveform": (3, ATOM_LEN), "pattern": (3, N_CHANNELS, 2)}
    for structure, start in starts.items():
        spec = GradientSpec(
            structure,  # type: ignore[arg-type]
            n_atoms=3,
            atom_len=ATOM_LEN,
            pool_len=POOL_LEN,
            n_projections=2 if structure == "SR" else 1,
            steps=4,
            lr=1e-2,
        )
        fit = fit_gradient(rows, start, spec, seed=0, device="cpu")
        assert fit.steps == 6 and fit.epochs == 2 and len(fit.step_losses) == 6
        for name, shape in shapes[structure].items():
            assert getattr(fit.atoms, name).shape == shape
        tokens = encode_gradient(fit.model, rows, structure)  # type: ignore[arg-type]
        assert tokens.atom.shape == tokens.start.shape == (6, N_TIMES // POOL_LEN)
        assert (tokens.channel is not None) == (structure == "S1")
        assert (tokens.projection is not None) == (structure == "SR")
        active = tokens.active
        assert bool(active.any()) and bool((tokens.value[active] > 0).all())
        assert int(tokens.start[active].min()) >= 0 and int(tokens.start[active].max()) <= N_TIMES - ATOM_LEN
        assert bool((tokens.atom[~active] == -1).all()) and bool((tokens.start[~active] == -1).all())
    with pytest.raises(ValueError, match="structure"):
        fit_gradient(rows, starts["SF"], replace(spec, structure="S1"), device="cpu")


def test_a_model_without_a_fit_encodes_with_the_atoms_of_its_cell() -> None:
    spec = GradientSpec("S1", n_atoms=2, atom_len=ATOM_LEN, pool_len=16)
    model = build_model(spec, device="cpu")
    waveform = unit_norm(_randn(2, ATOM_LEN, seed=16))
    set_model_atoms(model, CellAtoms("S1", waveform=waveform))
    read = model_atoms(model, "S1")
    assert read.waveform is not None and torch.allclose(read.waveform, waveform, atol=1e-12)
    rows = torch.zeros(1, N_CHANNELS, 64, dtype=torch.float64)
    rows[0, 2, 36:44] = 2.5 * waveform[1]
    tokens = encode_gradient(model, rows, "S1")
    assert tokens.active.sum() == 1 and tokens.channel is not None
    assert (int(tokens.atom[0, 2]), int(tokens.start[0, 2]), int(tokens.channel[0, 2])) == (1, 36, 2)
    assert abs(float(tokens.value[0, 2]) - 2.5) < 1e-10

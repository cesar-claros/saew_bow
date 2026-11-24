"""Checks of the comparison adapters against the retained reference models; run on the HPC.

The adapters must equal the reference when both options are off, keep every
placement inside the real samples when the edge mask is on, and report the
placements that the retained selection makes.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eeg_compare.adapters.models import (
    SEPARABLE_TAG,
    ComparisonSeparableSAE,
    ComparisonSeparableSpec,
    ComparisonWaveformSAE,
    ComparisonWaveformSpec,
    SignalPairs,
)
from eeg_compare.models.separable_sae import SeparableSAE, SeparableSpec
from eeg_compare.models.waveform_sae import CHECKPOINT_NAME, WaveformSAE, WaveformSpec, WaveformTrainSpec

PROVENANCE = {"train_subjects": ["patient:a", "patient:b"]}
N_CHANNELS = 3
GEOMETRY = {"n_atoms": 3, "atom_len": 8, "pool_len": 8, "n_projections": 2}


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _separable(**spec_kwargs: object) -> ComparisonSeparableSAE:
    """A CPU adapter (K=3, L=8, P=8, D=2, R=1) with Gaussian factors over three channels."""
    spec = ComparisonSeparableSpec(**{**GEOMETRY, **spec_kwargs})  # type: ignore[arg-type]
    model = ComparisonSeparableSAE(spec=spec, device="cpu")
    width = spec.atom_len + N_CHANNELS * spec.n_projections
    model.atoms = nn.Parameter(model._constrain(_randn(spec.n_atoms, width, seed=7)))
    return model


def _reference(model: ComparisonSeparableSAE) -> SeparableSAE:
    """The retained model with the adapter's geometry and atoms."""
    reference = SeparableSAE(spec=SeparableSpec(**GEOMETRY), device="cpu")  # type: ignore[arg-type]
    reference.atoms = nn.Parameter(model.atoms.detach().clone())
    return reference


def test_adapter_with_both_options_off_equals_the_reference() -> None:
    model = _separable(input_scaling="internal", edge_mask=False)
    reference = _reference(model)
    x = _randn(2, N_CHANNELS, 32, seed=1)
    x[0, 1] = 0.25
    for ours, theirs in zip(model._rows(x), reference._rows(x)):
        assert torch.equal(ours, theirs)
    assert torch.equal(model.encode(x)[0], reference.encode(x)[0])
    assert torch.equal(model(x), reference(x))
    rows, valid = model._full_rows(x)
    ours_step = model._segment_step(rows, valid, model._dictionary())
    theirs_step = reference._segment_step(rows, valid, reference._dictionary())
    assert torch.equal(ours_step[0], theirs_step[0]) and torch.equal(ours_step[1], theirs_step[1])


def test_external_scaling_passes_usable_rows_unchanged() -> None:
    model = _separable()
    x = 3.0 * _randn(2, N_CHANNELS, 32, seed=2)
    x[0, 1] = 0.25
    x[1, 2, 5] = float("nan")
    rows, valid = model._rows(x)
    expected = torch.ones(2, N_CHANNELS, dtype=torch.bool)
    expected[0, 1] = expected[1, 2] = False
    assert torch.equal(valid, expected)
    assert torch.equal(rows[valid], x[valid]), "a usable row was rescaled"
    assert not rows[~valid].any() and torch.isfinite(rows).all()
    scaled, _ = _reference(model)._rows(x)
    assert not torch.allclose(scaled[valid], x[valid]), "the reference scaling had no effect on this input"


def test_edge_mask_keeps_every_placement_inside_the_segment() -> None:
    model = _separable()
    left, right = model.padding
    assert (left, right) == (3, 4)
    for seed in range(5):
        x = _randn(4, N_CHANNELS, 32, seed=10 + seed)
        code, _ = model.encode(x)
        assert not code[..., :left].any() and not code[..., 32 - right :].any()
        placements = model.placements(x)
        positions = placements.position[placements.active]
        assert positions.numel() and int(positions.min()) >= left and int(positions.max()) <= 31 - right
    with pytest.raises(ValueError, match="no complete placement"):
        model._selection(torch.zeros(1, 3, 7, dtype=torch.float64))


def _impulse_model(edge_mask: bool) -> ComparisonSeparableSAE:
    """Atom 0 is an impulse at the last template sample on channel 0 or 1; atoms 1 and 2 read channels 1 and 2 only."""
    model = _separable(edge_mask=edge_mask)
    time = torch.zeros(3, 8, dtype=torch.float64)
    time[0, 7] = time[1, 0] = time[2, 3] = 1.0
    channel = torch.zeros(3, N_CHANNELS, 2, dtype=torch.float64)
    channel[0, 0, 0] = channel[0, 1, 1] = 1.0
    channel[1, 1, 0] = channel[1, 2, 1] = 1.0
    channel[2, 2, 0] = channel[2, 1, 1] = 1.0
    model.atoms = nn.Parameter(model._constrain(model._pack(time, channel)))
    return model


def test_a_response_on_the_padded_edge_is_selected_only_without_the_mask() -> None:
    x = torch.zeros(1, N_CHANNELS, 32, dtype=torch.float64)
    x[0, 0, 4] = 2.0
    unmasked, masked = _impulse_model(edge_mask=False), _impulse_model(edge_mask=True)
    code = unmasked.encode(x)[0]
    assert abs(float(code[0, 0, 0]) - 2.0) < 1e-12 and int((code > 0).sum()) == 1, "atom 0 must win at position 0"
    assert not masked.encode(x)[0].any(), "the mask left a placement whose template reaches the padding"
    assert not masked.placements(x).active.any()


def test_placements_report_the_retained_selection() -> None:
    model = _separable()
    x = _randn(3, N_CHANNELS, 32, seed=3)
    placements = model.placements(x)
    code, _ = model.encode(x)
    n_windows = 32 // model.spec.pool_len
    assert placements.amplitude.shape == placements.position.shape == (3, 3, n_windows)
    assert torch.equal(code.gather(-1, placements.position), placements.amplitude)
    assert torch.equal((code > 0).sum(-1), placements.hard_counts())
    assert torch.allclose(code.sum(-1), placements.amplitude_sums())
    assert torch.allclose(placements.amplitude_sums() / n_windows, model(x))
    assert int(placements.hard_counts().sum(1).max()) <= n_windows * model.spec.top_r
    assert torch.equal(placements.hard_counts([1]), placements.active[..., 1].long())
    by_winner = placements.counts_by_projection(model.spec.n_projections)
    assert by_winner.shape == (3, 3, 2) and torch.equal(by_winner.sum(-1), placements.hard_counts())
    assert torch.equal(placements.counts_by_projection(2, [0, 2]).sum(-1), placements.hard_counts([0, 2]))


def test_a_planted_template_is_found_with_its_atom_position_and_winner() -> None:
    model = _separable()
    time, channel = model._split(model._dictionary())
    left = model.padding[0]
    for atom, winner, start in ((0, 1, 10), (2, 0, 17), (1, 1, 0), (1, 0, 24)):
        x = torch.zeros(1, N_CHANNELS, 32, dtype=torch.float64)
        x[0, :, start : start + 8] = 2.0 * channel[atom, :, winner, None] * time[atom][None, :]
        placements = model.placements(x)
        center = start + left
        window = center // model.spec.pool_len
        assert bool(placements.active[0, atom, window]), f"atom {atom} holds no placement in window {window}"
        assert int(placements.position[0, atom, window]) == center
        assert placements.projection is not None and int(placements.projection[0, atom, window]) == winner
        assert abs(float(placements.amplitude[0, atom, window]) - 2.0) < 1e-10
        assert int(placements.counts_by_projection(2, [window])[0, atom, winner]) == 1


def test_full_template_adapter_has_placements_without_a_winner() -> None:
    spec = ComparisonWaveformSpec(n_atoms=3, atom_len=8, pool_len=8)
    model = ComparisonWaveformSAE(spec=spec, device="cpu")
    model.atoms = nn.Parameter(model._constrain(_randn(3, N_CHANNELS, 8, seed=5)))
    x = _randn(2, N_CHANNELS, 32, seed=6)
    placements = model.placements(x)
    code, _ = model.encode(x)
    assert placements.projection is None
    assert torch.equal(code.gather(-1, placements.position), placements.amplitude)
    assert not code[..., :3].any() and not code[..., 28:].any()
    with pytest.raises(ValueError, match="no spatial winner"):
        placements.counts_by_projection(2)
    reference = WaveformSAE(spec=WaveformSpec(n_atoms=3, atom_len=8, pool_len=8), device="cpu")
    reference.atoms = nn.Parameter(model.atoms.detach().clone())
    off = ComparisonWaveformSAE(spec=replace(spec, input_scaling="internal", edge_mask=False), device="cpu")
    off.atoms = nn.Parameter(model.atoms.detach().clone())
    assert torch.equal(off.encode(x)[0], reference.encode(x)[0])


def test_initial_patches_hold_no_padding() -> None:
    x = 5.0 + torch.rand(16, N_CHANNELS, 32, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
    loader = DataLoader(TensorDataset(x, torch.zeros(16, dtype=torch.long)), batch_size=4)
    model = _separable()
    patches = model._sample_patches(loader, torch.Generator().manual_seed(0), count=6)
    assert patches.shape == (6, N_CHANNELS, 8) and bool((patches > 4.0).all())
    padded = _separable(edge_mask=False)
    drawn = padded._sample_patches(loader, torch.Generator().manual_seed(0), count=6)
    assert drawn.shape == (6, N_CHANNELS, 8)


def test_fit_runs_on_signal_pairs_and_the_checkpoint_keeps_its_options() -> None:
    generator = torch.Generator().manual_seed(11)
    items = [
        {"signal": torch.randn(N_CHANNELS, 64, generator=generator), "target": torch.tensor(index % 2)}
        for index in range(12)
    ]
    pairs = SignalPairs(items)
    assert len(pairs) == 12 and pairs[3][0] is items[3]["signal"] and int(pairs[3][1]) == 1
    spec = ComparisonSeparableSpec(n_atoms=2, atom_len=16, pool_len=32, n_projections=2)
    train_spec = WaveformTrainSpec(epochs=2, lr=1e-2, segment_batch=4, recenter_atoms=False)
    model = ComparisonSeparableSAE(spec=spec, train_spec=train_spec, random_state=0, device="cpu")
    model.fit_unsupervised(DataLoader(pairs, batch_size=4), provenance=PROVENANCE)
    assert model.fitted and len(model.history) == 2
    x = torch.stack([item["signal"] for item in items[:2]])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        model.save_artifacts(out)
        path = out / CHECKPOINT_NAME
        checkpoint = torch.load(path, weights_only=True)
        assert checkpoint["model"] == SEPARABLE_TAG and checkpoint["format"] == 1
        assert checkpoint["spec"]["input_scaling"] == "external" and checkpoint["spec"]["edge_mask"] is True
        reloaded = ComparisonSeparableSAE(spec=spec, device="cpu", pretrained=path)
        assert torch.equal(reloaded.encode(x)[0], model.encode(x)[0])
        with pytest.raises(ValueError, match="spec differs"):
            ComparisonSeparableSAE(spec=replace(spec, edge_mask=False), device="cpu", pretrained=path)
        with pytest.raises(ValueError, match="not a"):
            SeparableSAE(
                spec=SeparableSpec(n_atoms=2, atom_len=16, pool_len=32, n_projections=2), device="cpu", pretrained=path
            )


def test_spec_guards() -> None:
    with pytest.raises(TypeError, match="ComparisonSeparableSpec"):
        ComparisonSeparableSAE(spec=SeparableSpec(), device="cpu")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ComparisonWaveformSpec"):
        ComparisonWaveformSAE(spec=WaveformSpec(), device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="input_scaling"):
        ComparisonSeparableSAE(spec=ComparisonSeparableSpec(input_scaling="record"), device="cpu")
    with pytest.raises(ValueError, match="edge_mask"):
        ComparisonWaveformSAE(spec=ComparisonWaveformSpec(edge_mask=1), device="cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="complete segments"):
        ComparisonWaveformSAE(train_spec=WaveformTrainSpec(crop_sampling="tiled"), device="cpu")
    assert ComparisonSeparableSAE(device="cpu").n_channels is None

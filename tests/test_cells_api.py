"""The nine cells behind one interface: start, fit, frozen dictionary and tokens; run on the HPC."""

from __future__ import annotations

import pytest
import torch

from eeg_compare.cells.api import (
    CELL_NAMES,
    CELLS,
    Budget,
    Geometry,
    draw_start,
    fit_cell,
    frozen_cell,
    start_atoms,
)
from eeg_compare.cells.common import CellAtoms, unit_norm

GEOMETRY = Geometry(n_atoms=3, atom_len=8, pool_len=16, n_projections=2)
N_CHANNELS, N_TIMES = 3, 64


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _planted_dictionary() -> dict[str, CellAtoms]:
    """One planted dictionary in the three structures: rank-1 templates ``u w^T`` with positive patterns."""
    waveform = unit_norm(_randn(3, 8, seed=1))
    pattern = _randn(3, N_CHANNELS, seed=2).abs() + 0.1
    pattern = pattern / pattern.norm(dim=1, keepdim=True)
    other = _randn(3, N_CHANNELS, seed=3)
    other = other / other.norm(dim=1, keepdim=True)
    return {
        "S1": CellAtoms("S1", waveform=waveform),
        "SF": CellAtoms("SF", template=pattern[:, :, None] * waveform[:, None, :]),
        "SR": CellAtoms("SR", waveform=waveform, pattern=torch.stack([pattern, other], dim=-1)),
    }


def test_the_nine_cells_of_the_plan() -> None:
    assert CELL_NAMES == ("S1-L1", "S1-L2", "S1-L3", "S1-L4", "S1-L5", "SF-L1", "SF-L5", "SR-L5", "SR-L5n")
    assert set(CELLS) == set(CELL_NAMES)
    assert [CELLS[name].gradient for name in CELL_NAMES] == [False, False, False, False, True, False, True, True, True]
    scores = {name: CELLS[name].score for name in CELL_NAMES}
    assert [name for name, score in scores.items() if score == "cosine"] == ["S1-L1", "S1-L2", "S1-L3", "SF-L1"]
    assert CELLS["SR-L5n"].structure == "SR" and CELLS["SR-L5n"].rule == "L5n"
    with pytest.raises(ValueError, match="unknown cell"):
        frozen_cell("SR-L1", _planted_dictionary()["SR"], GEOMETRY, device="cpu")
    with pytest.raises(ValueError, match="positive integer"):
        Geometry(n_atoms=0)


def test_the_common_start_gives_each_structure_its_atoms() -> None:
    rows = _randn(5, N_CHANNELS, N_TIMES, seed=4)
    start = draw_start(rows, GEOMETRY, torch.Generator().manual_seed(0))
    again = draw_start(rows, GEOMETRY, torch.Generator().manual_seed(0))
    assert start.patches.shape == (6, N_CHANNELS, 8) and start.channels.shape == (3,)
    assert torch.equal(start.patches, again.patches) and torch.equal(start.channels, again.channels)
    for index in range(6):
        segment, first = int(start.segments[index]), int(start.first[index])
        assert 0 <= first <= N_TIMES - 8
        assert torch.equal(start.patches[index], rows[segment, :, first : first + 8])
    full = start_atoms("SF", start, GEOMETRY)
    single = start_atoms("S1", start, GEOMETRY)
    separable = start_atoms("SR", start, GEOMETRY)
    assert full.template is not None and single.waveform is not None
    assert separable.waveform is not None and separable.pattern is not None
    assert torch.allclose(full.template, unit_norm(start.patches[:3]))
    for atom in range(3):
        channel = int(start.channels[atom])
        assert torch.allclose(single.waveform[atom], unit_norm(start.patches[:3, channel])[atom])
        for projection in range(2):
            source = start.patches[projection * 3 + atom]
            assert float(separable.pattern[atom, :, projection] @ source @ separable.waveform[atom]) > 0
    assert torch.allclose(separable.waveform.norm(dim=1), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(separable.pattern.norm(dim=1), torch.ones(3, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="flat"):
        draw_start(torch.zeros(2, N_CHANNELS, N_TIMES, dtype=torch.float64), GEOMETRY, torch.Generator())
    with pytest.raises(ValueError, match="geometry"):
        start_atoms("SF", start, Geometry(n_atoms=4, atom_len=8, pool_len=16))


@pytest.mark.parametrize("name", CELL_NAMES)
def test_the_planted_dictionary_finds_its_event_in_every_cell(name: str) -> None:
    cell = CELLS[name]
    planted = _planted_dictionary()
    oracle = frozen_cell(name, planted[cell.structure], GEOMETRY, device="cpu")
    assert oracle.record == {} and (oracle.model is not None) == cell.gradient
    waveform, template = planted["S1"].waveform, planted["SF"].template
    assert waveform is not None and template is not None
    for atom, start in ((0, 0), (1, 20), (2, 36), (1, 56)):
        window = (start + 3) // 16
        rows = torch.zeros(2, N_CHANNELS, N_TIMES, dtype=torch.float64)
        if cell.structure == "S1":
            rows[1, 2, start : start + 8] = 2.0 * waveform[atom]
        else:
            rows[1, :, start : start + 8] = 2.0 * template[atom]
        tokens = oracle.tokens(rows)
        assert tokens.atom.shape == (2, 4) and int(tokens.active.sum()) == 1
        assert int(tokens.atom[1, window]) == atom and int(tokens.start[1, window]) == start
        assert abs(float(tokens.response[1, window]) - 2.0) < 1e-9
        if cell.structure == "S1":
            assert tokens.channel is not None and int(tokens.channel[1, window]) == 2
        if cell.structure == "SR":
            assert tokens.projection is not None and int(tokens.projection[1, window]) == 0


@pytest.mark.parametrize("name", CELL_NAMES)
def test_every_cell_fits_from_the_common_start_and_gives_tokens(name: str) -> None:
    cell = CELLS[name]
    rows = 0.2 * _randn(6, N_CHANNELS, N_TIMES, seed=5)
    planted = _planted_dictionary()["SF"].template
    assert planted is not None
    for start in (20, 36):
        rows[:, :, start : start + 8] += 3.0 * planted[0]
    start_patches = draw_start(rows, GEOMETRY, torch.Generator().manual_seed(1))
    initial = start_atoms(cell.structure, start_patches, GEOMETRY)
    budget = Budget(max_passes=3, steps=4, lr=1e-2)
    fitted = fit_cell(name, rows, initial, GEOMETRY, budget, seed=0, device="cpu")
    if cell.gradient:
        assert fitted.model is not None and fitted.record["steps"] == 6 == len(fitted.record["step_losses"])
    else:
        assert fitted.model is None and 1 <= fitted.record["passes"] <= 3
        assert len(fitted.record["history"]) == fitted.record["passes"]
    shapes = {"S1": [(3, 8)], "SF": [(3, N_CHANNELS, 8)], "SR": [(3, 8), (3, N_CHANNELS, 2)]}[cell.structure]
    fields = [
        value for value in (fitted.atoms.waveform, fitted.atoms.template, fitted.atoms.pattern) if value is not None
    ]
    assert [tuple(value.shape) for value in fields] == shapes
    for value in fields:
        norms = value.flatten(1).norm(dim=1) if value.ndim == 2 or cell.structure == "SF" else value.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-8)
    tokens = fitted.tokens(rows)
    assert tokens.atom.shape == (6, 4) and bool(tokens.active.any())
    assert int(tokens.atom.max()) < 3 and int(tokens.atom.min()) >= -1
    frozen = frozen_cell(name, initial, GEOMETRY, device="cpu")
    assert frozen.tokens(rows).atom.shape == (6, 4)


def test_a_cell_refuses_atoms_of_another_structure_or_geometry() -> None:
    planted = _planted_dictionary()
    rows = _randn(2, N_CHANNELS, N_TIMES, seed=6)
    with pytest.raises(ValueError, match="structure"):
        fit_cell("SF-L1", rows, planted["S1"], GEOMETRY, device="cpu")
    with pytest.raises(ValueError, match="do not match"):
        frozen_cell("S1-L3", planted["S1"], Geometry(n_atoms=4, atom_len=8, pool_len=16), device="cpu")
    with pytest.raises(ValueError, match="spatial patterns"):
        frozen_cell("SR-L5", planted["SR"], Geometry(n_atoms=3, atom_len=8, pool_len=16, n_projections=4), device="cpu")
    with pytest.raises(ValueError, match="exactly the fields"):
        CellAtoms("S1", template=planted["SF"].template)

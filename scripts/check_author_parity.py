"""Parity of the alternating cells with the author code of reference [1]; execute on the HPC.

This is gate 2 of the stage 2 protocol (docs/Synthetic Benchmark Protocol.md).
It needs a checkout of the author repository of the mouse pipeline. The script
loads two functions from the checkout at run time: ``si_vq`` (assignment) and
``_centroids_update_step`` (update). The author repository has no license file,
so no code of it is in this repository, and this script copies none.

Checks, on single-channel windows of equal length and with equal atoms:

1. Assignment. ``assign`` with the cosine score gives the labels and shifts of
   ``si_vq`` with the cosine metric for at least 99.9% of the windows.
2. Update with raw sub-windows. One ``update`` with the weight ``norm`` (rule
   L2) equals ``_centroids_update_step`` with ``do_sphere=False``.
3. Update with normalized sub-windows. One ``update`` with the weight ``unit``
   (rule L1) equals ``_centroids_update_step`` with ``do_sphere=True``.

Both updates get the labels and shifts of the author assignment. They are
compared after a scale to unit norm, within 1e-5, on the atoms that have a
window. The checks run in two rounds: with atoms from random windows, and with
the centroids that the author update gives from them (not of unit norm).

The report also records the commit of the checkout, the modules that had to be
replaced by an empty module to import the author files, and the ``do_sphere``
settings that the checkout shows. Which update made the published results is a
question for the authors.

Run from the repository root with ``PYTHONPATH=src``::

    python scripts/check_author_parity.py --author-repo /path/to/checkout --output artifacts/author-parity.json
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

AGREEMENT_FLOOR = 0.999
UPDATE_TOLERANCE = 1e-5
_OPTIONAL_PACKAGE = ("BOWaves", "BOWaves.utilities", "BOWaves.utilities.sikmeans_utils")
_AUTHOR_MODULES = ("si_vq", "si2_kmeans")

Array = NDArray[np.float64]


def _git(repo: Path, *arguments: str) -> str | None:
    """Output of a git command in the checkout, or ``None`` when git or the repository is not available."""
    try:
        done = subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _stub_optional_package() -> list[str]:
    """Register empty modules for the package that the author clustering file imports but these checks do not use."""
    try:
        importlib.import_module(_OPTIONAL_PACKAGE[-1])
        return []
    except ImportError:
        pass
    created = []
    parent: types.ModuleType | None = None
    for name in _OPTIONAL_PACKAGE:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = module
            created.append(name)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[-1], module)
        parent = module
    return created


def load_author(repo: Path) -> dict[str, Any]:
    """Import the author functions from a checkout and describe it.

    Parameters
    ----------
    repo : Path
        The checkout that holds ``si_vq.py`` and ``si2_kmeans.py``.

    Returns
    -------
    dict[str, Any]
        ``si_vq`` and ``update_step`` (the functions) and ``info`` (commit, state
        of the work tree, replaced modules, and the ``do_sphere`` settings).

    Raises
    ------
    ValueError
        If a file is missing, an import fails for any reason, a module comes from
        another folder, or one of the two functions is missing.
    """
    repo = repo.resolve()
    for name in _AUTHOR_MODULES:
        if not (repo / f"{name}.py").is_file():
            raise ValueError(f"{repo} holds no {name}.py")
    sys.path.insert(0, str(repo))
    importlib.invalidate_caches()
    stubbed = _stub_optional_package()
    modules = {}
    for name in _AUTHOR_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as error:  # the file is foreign code: any failure of its import is one error of this check
            raise ValueError(
                f"the author module {name} did not import ({type(error).__name__}: {error}); run this check in "
                "an environment that has the dependencies of the author repository and this package"
            ) from error
        origin = Path(getattr(module, "__file__", "") or "").resolve()
        if origin.parent != repo:
            raise ValueError(f"the module {name} came from {origin}, not from the checkout {repo}")
        modules[name] = module
    kmeans = modules["si2_kmeans"]
    assignment = getattr(modules["si_vq"], "si_vq", None)
    update_step = getattr(kmeans, "_centroids_update_step", None)
    if not callable(assignment) or not callable(update_step):
        raise ValueError("the checkout has no function si_vq.si_vq or si2_kmeans._centroids_update_step")
    defaults = {}
    for function in ("si_kmeans", "si_kmeans_single", "_centroids_update_step"):
        target = getattr(kmeans, function, None)
        parameter = inspect.signature(target).parameters.get("do_sphere") if callable(target) else None
        defaults[function] = None if parameter is None else parameter.default
    pipeline = repo / "mice_utils.py"
    pipeline_lines = (
        [line.strip() for line in pipeline.read_text(encoding="utf-8").splitlines() if "do_sphere" in line]
        if pipeline.is_file()
        else []
    )
    status = _git(repo, "status", "--porcelain")
    info = {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "work_tree_clean": None if status is None else status == "",
        "replaced_modules": stubbed,
        "do_sphere_defaults": defaults,
        "do_sphere_lines_of_the_pipeline_file": pipeline_lines,
    }
    return {"si_vq": assignment, "update_step": update_step, "info": info}


def synthetic_windows(n_windows: int, length: int, atom_len: int, seed: int) -> Array:
    """Single-channel windows ``(n, N)``: smooth noise, with a damped oscillation in every second window."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n_windows, length + 4))
    smooth = np.stack([np.convolve(row, np.full(5, 0.2), mode="valid") for row in white])
    windows = smooth / smooth.std()
    time = np.arange(atom_len) / atom_len
    waveform = np.sin(2 * np.pi * 3 * time) * np.exp(-3 * time)
    waveform /= np.linalg.norm(waveform)
    for index in range(0, n_windows, 2):
        start = int(rng.integers(0, length - atom_len + 1))
        windows[index, start : start + atom_len] += 4.0 * waveform * np.sqrt(atom_len)
    return windows


def compare(
    si_vq: Callable[..., Any], update_step: Callable[..., Any], windows: Array, centroids: Array
) -> tuple[dict[str, Any], Array]:
    """One round of the three checks.

    Parameters
    ----------
    si_vq, update_step : Callable
        The author functions.
    windows : numpy.ndarray
        ``(n, N)`` single-channel windows.
    centroids : numpy.ndarray
        ``(K, L)`` atoms of both codes; any norm.

    Returns
    -------
    tuple[dict[str, Any], numpy.ndarray]
        The measures of the round, and the centroids of the author update with
        ``do_sphere=True`` (the atoms of the next round).
    """
    import torch

    from eeg_compare.cells.alternating import assign, update
    from eeg_compare.cells.common import unit_norm

    n_atoms, atom_len = centroids.shape
    labels, shifts, distances = si_vq(windows, centroids, "cosine")
    labels, shifts = np.asarray(labels, dtype=np.int64), np.asarray(shifts, dtype=np.int64)
    units = torch.from_numpy(windows).unsqueeze(1)
    atoms = torch.from_numpy(centroids).unsqueeze(1)
    ours = assign(units, atoms, "cosine")
    same = (ours.label.numpy() == labels) & (ours.shift.numpy() == shifts)
    gap = np.abs((1.0 - ours.cosine.numpy()) - np.asarray(distances, dtype=np.float64))
    result: dict[str, Any] = {
        "n_windows": int(windows.shape[0]),
        "assignment_agreement": float(same.mean()),
        "windows_without_an_atom": int((ours.label < 0).sum()),
        "largest_distance_gap_where_equal": float(gap[same].max()) if bool(same.any()) else None,
    }
    signs = np.ones_like(shifts)
    spherical: Array = centroids
    for do_sphere, weight in ((False, "norm"), (True, "unit")):
        theirs = np.asarray(update_step(windows, atom_len, n_atoms, labels, shifts, signs, do_sphere), dtype=np.float64)
        mine, counts = update(units, torch.from_numpy(labels), torch.from_numpy(shifts), unit_norm(atoms), weight)
        filled = counts.numpy() > 0
        norms = np.linalg.norm(theirs[filled], axis=1, keepdims=True)
        difference = np.abs(mine.numpy()[filled, 0] - theirs[filled] / np.maximum(norms, 1e-300))
        result[f"update_gap_{weight}"] = float(difference.max()) if bool(filled.any()) else None
        result[f"empty_atoms_{weight}"] = int((~filled).sum())
        if do_sphere:
            spherical = theirs
    return result, spherical


def build_report(
    author: dict[str, Any], n_windows: int, length: int, atom_len: int, n_atoms: int, seed: int
) -> dict[str, Any]:
    """Run both rounds and apply the fixed thresholds of the gate.

    The second round uses the centroids of the author update. A centroid without
    a window is zero there; it keeps its earlier value for the second round, and
    the report counts these atoms.
    """
    windows = synthetic_windows(n_windows, length, atom_len, seed)
    chosen = np.random.default_rng(seed + 1).permutation(n_windows)[:n_atoms]
    centroids = windows[chosen, :atom_len].copy()
    rounds = []
    for name in ("atoms_from_random_windows", "atoms_after_one_author_update"):
        measures, updated = compare(author["si_vq"], author["update_step"], windows, centroids)
        gaps = [measures["update_gap_norm"], measures["update_gap_unit"]]
        measures["round"] = name
        measures["ok"] = bool(
            measures["assignment_agreement"] >= AGREEMENT_FLOOR
            and all(value is not None and value <= UPDATE_TOLERANCE for value in gaps)
        )
        unusable = ~np.isfinite(updated).all(axis=1) | (np.linalg.norm(np.nan_to_num(updated), axis=1) == 0)
        measures["atoms_kept_for_the_next_round"] = int(unusable.sum())
        centroids = np.where(unusable[:, None], centroids, updated)
        rounds.append(measures)
    return {
        "gate": "parity with the author code of reference [1] (stage 2 protocol, gate 2)",
        "thresholds": {"assignment_agreement": AGREEMENT_FLOOR, "update_gap": UPDATE_TOLERANCE},
        "settings": {
            "n_windows": n_windows,
            "window_length": length,
            "atom_len": atom_len,
            "n_atoms": n_atoms,
            "seed": seed,
        },
        "author": author["info"],
        "rounds": rounds,
        "ok": bool(len(rounds) == 2 and all(entry["ok"] for entry in rounds)),
    }


def render(report: dict[str, Any]) -> str:
    """The text view of a report."""
    author = report["author"]
    lines = [
        "Parity with the author code of reference [1]",
        f"  checkout commit: {author['commit']} (clean work tree: {author['work_tree_clean']})",
        f"  replaced modules: {author['replaced_modules'] or 'none'}",
        f"  do_sphere defaults: {author['do_sphere_defaults']}",
    ]
    lines += [f"  pipeline file: {line}" for line in author["do_sphere_lines_of_the_pipeline_file"]]
    for entry in report["rounds"]:
        lines.append(
            f"  {entry['round']}: assignment agreement {entry['assignment_agreement']:.5f}, "
            f"update gap raw {entry['update_gap_norm']}, update gap normalized {entry['update_gap_unit']}, "
            f"windows without an atom {entry['windows_without_an_atom']}, ok {entry['ok']}"
        )
    lines.append(f"Gate 2: {'PASS' if report['ok'] else 'FAIL'}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the JSON report and print its text view; return 0 when the gate passes, 1 otherwise, 2 on an error."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--author-repo", type=Path, required=True, help="Checkout of the author repository")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report")
    parser.add_argument("--n-windows", type=int, default=4000, help="Single-channel windows of the check")
    parser.add_argument("--window-length", type=int, default=249, help="Samples per window")
    parser.add_argument("--atom-len", type=int, default=125, help="Samples per atom")
    parser.add_argument("--n-atoms", type=int, default=8, help="Atoms")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the windows and of the atoms")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("the output must be a new file")
        if args.window_length < args.atom_len or args.n_windows < args.n_atoms:
            raise ValueError("the windows must be as long as an atom, and there must be a window for every atom")
        author = load_author(args.author_repo)
        report = build_report(author, args.n_windows, args.window_length, args.atom_len, args.n_atoms, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, default=str) + "\n")
    except (OSError, ValueError) as error:
        sys.stderr.write(f"Author parity check failed: {error}\n")
        return 2
    sys.stdout.write(render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The author parity script against a stand-in checkout; run on the HPC.

The stand-in files are written by this test. They have the function names and
signatures that the script loads from the author repository, with independent
direct implementations. They hold no code of the author repository. The check
against the real checkout is a separate run of the script.
"""

from __future__ import annotations

import json
import runpy
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_author_parity.py"
LOADED = ("si_vq", "si2_kmeans", "BOWaves", "BOWaves.utilities", "BOWaves.utilities.sikmeans_utils")

STAND_IN_ASSIGNMENT = """
import numpy as np


def si_vq(X, Y, metric):
    assert metric == "cosine"
    count, length = X.shape
    size = Y.shape[1]
    unit = Y / np.sqrt((Y**2).sum(axis=1, keepdims=True))
    labels = np.zeros(count, dtype=int)
    shifts = np.zeros(count, dtype=int)
    distances = np.full(count, np.inf)
    for shift in range(length - size + 1):
        part = X[:, shift : shift + size]
        cosine = (part / np.sqrt((part**2).sum(axis=1, keepdims=True))) @ unit.T
        best = cosine.argmax(axis=1)
        distance = 1.0 - cosine[np.arange(count), best]
        better = distance < distances
        labels[better] = best[better]
        shifts[better] = shift
        distances[better] = distance[better]
    return labels, shifts, distances
"""

STAND_IN_CLUSTERING = """
import numpy as np
from BOWaves.utilities import sikmeans_utils


def si_kmeans(X, n_clusters, centroid_length, metric="cosine", do_sphere=True):
    raise NotImplementedError


def si_kmeans_single(X, n_clusters, centroid_length, metric="euclidean", do_sphere=False):
    raise NotImplementedError


def _centroids_update_step(X, centroid_length, n_clusters, labels, shifts, signs, do_sphere=False):
    centroids = np.zeros((n_clusters, centroid_length))
    counts = np.zeros(n_clusters)
    for index in range(X.shape[0]):
        part = signs[index] * X[index, shifts[index] : shifts[index] + centroid_length]
        centroids[labels[index]] += SCALE * (part / np.linalg.norm(part) if do_sphere else part)
        counts[labels[index]] += 1
    filled = counts > 0
    centroids[filled] /= counts[filled, None]
    return centroids
"""


@pytest.fixture
def script() -> Iterator[SimpleNamespace]:
    path, modules = list(sys.path), {name: sys.modules.get(name) for name in LOADED}
    yield SimpleNamespace(**runpy.run_path(str(SCRIPT)))
    sys.path[:] = path
    for name, module in modules.items():
        sys.modules.pop(name, None)
        if module is not None:
            sys.modules[name] = module


def _checkout(folder: Path, scale: str = "1.0") -> Path:
    folder.mkdir()
    (folder / "si_vq.py").write_text(STAND_IN_ASSIGNMENT, encoding="utf-8")
    (folder / "si2_kmeans.py").write_text(STAND_IN_CLUSTERING.replace("SCALE", scale), encoding="utf-8")
    (folder / "mice_utils.py").write_text("x = dict(do_sphere=True)\n", encoding="utf-8")
    return folder


def test_synthetic_windows_hold_an_event_in_every_second_window(script: SimpleNamespace) -> None:
    windows = script.synthetic_windows(40, 60, 16, seed=3)
    assert windows.shape == (40, 60) and np.isfinite(windows).all()
    energy = (windows**2).sum(axis=1)
    assert energy[0::2].min() > energy[1::2].max()
    assert np.array_equal(windows, script.synthetic_windows(40, 60, 16, seed=3))


def test_the_gate_passes_for_a_stand_in_with_the_same_rules(script: SimpleNamespace, tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    arguments = ["--author-repo", str(_checkout(tmp_path / "checkout")), "--output", str(output)]
    arguments += ["--n-windows", "300", "--window-length", "47", "--atom-len", "16", "--n-atoms", "4"]
    assert script.main(arguments) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True and len(report["rounds"]) == 2
    for entry in report["rounds"]:
        assert entry["assignment_agreement"] >= 0.999 and entry["windows_without_an_atom"] == 0
        assert entry["update_gap_norm"] < 1e-10 and entry["update_gap_unit"] < 1e-10
    author = report["author"]
    assert author["do_sphere_defaults"] == {
        "si_kmeans": True,
        "si_kmeans_single": False,
        "_centroids_update_step": False,
    }
    assert author["do_sphere_lines_of_the_pipeline_file"] == ["x = dict(do_sphere=True)"]
    assert set(author["replaced_modules"]) <= set(LOADED[2:])
    assert script.main(arguments) == 2, "an existing report must not be replaced"


def test_the_gate_fails_for_another_update_rule(script: SimpleNamespace, tmp_path: Path) -> None:
    # The stand-in adds a constant to every sub-window, so its centroids differ from both rules.
    checkout = _checkout(tmp_path / "checkout", scale="0.2 * np.ones(centroid_length) + 0.5")
    output = tmp_path / "report.json"
    arguments = ["--author-repo", str(checkout), "--output", str(output)]
    arguments += ["--n-windows", "300", "--window-length", "47", "--atom-len", "16", "--n-atoms", "4"]
    assert script.main(arguments) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is False and report["rounds"][0]["update_gap_norm"] > 1e-5


def test_an_author_file_that_does_not_import_is_an_error(script: SimpleNamespace, tmp_path: Path) -> None:
    checkout = _checkout(tmp_path / "checkout")
    (checkout / "si2_kmeans.py").write_text("def broken(:\n", encoding="utf-8")
    output = tmp_path / "report.json"
    assert script.main(["--author-repo", str(checkout), "--output", str(output)]) == 2
    assert not output.exists()
    (checkout / "si2_kmeans.py").write_text("value = 1\n", encoding="utf-8")
    assert script.main(["--author-repo", str(checkout), "--output", str(output)]) == 2
    assert not output.exists()


def test_a_folder_without_the_author_files_is_an_error(script: SimpleNamespace, tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert script.main(["--author-repo", str(tmp_path / "empty"), "--output", str(tmp_path / "report.json")]) == 2
    assert not (tmp_path / "report.json").exists()

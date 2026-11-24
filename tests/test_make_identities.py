"""Identity CSV script checks on a synthetic folder tree; execute on the HPC."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from eeg_compare.data.manifest import read_identities
from eeg_compare.data.recordings import cluster_of

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_tuev_identities.py"


@pytest.fixture
def script() -> SimpleNamespace:
    return SimpleNamespace(**runpy.run_path(str(SCRIPT)))


def _tree(root: Path) -> None:
    for name in ("train/p1/p1_a.edf", "train/p1/p1_b.EDF", "train/p2/p2_a.edf", "eval/032/bckg_032_a_.edf"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"header")
        path.with_suffix(".rec").write_text("0,1,2,1")
    (root / "train/p2/p2_a.rec").unlink()
    (root / "train/p2/notes.txt").write_text("not an EDF")


def test_rows_follow_the_folder_of_each_file(script: SimpleNamespace, tmp_path: Path) -> None:
    _tree(tmp_path)
    rows, summary = script.collect(tmp_path)
    assert [(row["edf_path"], row["patient_id"], row["session_id"]) for row in rows] == [
        ("train/p1/p1_a.edf", "p1", ""),
        ("train/p1/p1_b.EDF", "p1", ""),
        ("train/p2/p2_a.edf", "p2", ""),
        ("eval/032/bckg_032_a_.edf", "", "032"),
    ]
    assert all(row["identity_source"] for row in rows)
    assert summary["train"] == {
        "files": 3,
        "groups": 2,
        "files_without_one_rec": 1,
        "files_whose_name_lacks_the_folder_name": 0,
    }
    assert summary["eval"]["files"] == summary["eval"]["groups"] == 1


def test_the_written_csv_feeds_the_manifest_reader(
    script: SimpleNamespace, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, output = tmp_path / "edf", tmp_path / "data" / "identities.csv"
    _tree(root)
    assert script.main(["--root", str(root), "--output", str(output)]) == 0
    assert '"matches_reference_v2.0.1": false' in capsys.readouterr().out
    identities = read_identities(output)
    assert set(identities) == {
        "train/p1/p1_a.edf",
        "train/p1/p1_b.EDF",
        "train/p2/p2_a.edf",
        "eval/032/bckg_032_a_.edf",
    }
    assert cluster_of(identities["train/p2/p2_a.edf"], "train", "session") == ("patient:p2", "patient")
    assert cluster_of(identities["eval/032/bckg_032_a_.edf"], "eval", "session") == ("session:032", "session")
    before = output.read_bytes()
    assert script.main(["--root", str(root), "--output", str(output)]) == 1
    assert output.read_bytes() == before


def test_ambiguous_or_missing_folders_fail(script: SimpleNamespace, tmp_path: Path) -> None:
    _tree(tmp_path)
    loose = tmp_path / "eval" / "loose.edf"
    loose.write_bytes(b"header")
    with pytest.raises(ValueError, match="no group folder"):
        script.collect(tmp_path)
    loose.unlink()
    twin = tmp_path / "train" / "site_b" / "p1" / "p1_c.edf"
    twin.parent.mkdir(parents=True)
    twin.write_bytes(b"header")
    with pytest.raises(ValueError, match="ambiguous"):
        script.collect(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        script.collect(tmp_path / "absent")
    assert script.main(["--root", str(tmp_path / "absent"), "--output", str(tmp_path / "never.csv")]) == 1
    assert not (tmp_path / "never.csv").exists()

"""Environment preflight regressions; execute only on the HPC."""

from __future__ import annotations

import runpy
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_environment.py"


@pytest.fixture
def checker(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Load the standalone checker and isolate it from installed distributions."""
    namespace = runpy.run_path(str(SCRIPT))
    monkeypatch.setattr(namespace["platform"], "python_version", lambda: "3.12.14")
    return SimpleNamespace(**namespace)


def test_cuda_local_version_satisfies_torch_pin(checker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The container's cu124 wheel satisfies the package's torch==2.6.0 pin."""
    monkeypatch.setattr(checker.metadata, "version", lambda name: "2.6.0+cu124" if name == "torch" else "999.0")
    report = checker.collect_report(SCRIPT.parents[1] / "pyproject.toml")
    torch = next(item for item in report["distributions"] if item["requirement"] == "torch==2.6.0")
    assert torch["ok"]
    assert not report["ok"]  # Other versions deliberately exceed the upper bounds.


def test_missing_distribution_fails(checker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """An incomplete environment produces a report rather than aborting imports."""

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(checker.metadata, "version", missing)
    report = checker.collect_report(SCRIPT.parents[1] / "pyproject.toml", include_dev=True)
    assert not report["ok"]
    assert "Missing distribution: orjson" in report["errors"]
    assert "Missing distribution: hatchling" in report["errors"]
    assert "Missing distribution: pytest" in report["errors"]


def test_wrong_python_and_broken_import_fail(checker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Metadata alone cannot certify a usable Python or native-library import."""
    monkeypatch.setattr(checker.platform, "python_version", lambda: "3.13.0")
    monkeypatch.setattr(checker.metadata, "version", lambda name: "2.6.0")

    def broken(name: str) -> None:
        raise OSError("missing shared library")

    monkeypatch.setattr(checker.importlib, "import_module", broken)
    report = checker.collect_report(SCRIPT.parents[1] / "pyproject.toml", check_imports=True)
    assert not report["ok"]
    assert any("Python 3.13.0" in error for error in report["errors"])
    assert any("Import torch failed: OSError" in error for error in report["errors"])
    assert report["gpu_checked"] is False


def test_existing_report_is_preserved(checker: SimpleNamespace, tmp_path: Path) -> None:
    """A rerun must never overwrite evidence from an earlier check."""
    output = tmp_path / "environment.json"
    output.write_text("original report", encoding="utf-8")
    assert checker.main(["--output", str(output)]) == 1
    assert output.read_text(encoding="utf-8") == "original report"


def test_complete_environment_writes_success_report(
    checker: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A valid environment yields a successful CLI exit and durable report."""
    versions = {
        "loguru": "0.7.3",
        "mne": "1.11.0",
        "numpy": "2.2.6",
        "orjson": "3.10.0",
        "polars": "1.34.0",
        "scikit-learn": "1.5.0",
        "scipy": "1.14.0",
        "threadpoolctl": "3.5.0",
        "torch": "2.6.0+cu124",
        "tqdm": "4.67.1",
        "mypy": "1.11.0",
        "packaging": "24.0",
        "pytest": "8.4.0",
        "ruff": "0.9.0",
        "hatchling": "1.26.0",
    }
    monkeypatch.setattr(checker.metadata, "version", versions.__getitem__)
    output = tmp_path / "run" / "environment.json"
    assert checker.main(["--include-dev", "--output", str(output)]) == 0
    report = checker.json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"]
    assert report["errors"] == []
    assert report["imports_checked"] is False

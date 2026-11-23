"""Content hashes and portable paths used by data artifacts."""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import orjson


def content_hash(value: object) -> str:
    """SHA-256 of canonical JSON, independent of dictionary insertion order."""
    return hashlib.sha256(orjson.dumps(value, option=orjson.OPT_SORT_KEYS)).hexdigest()


def file_hash(path: Path) -> str:
    """Stream a file through SHA-256 without loading the complete file into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative_path(value: str) -> str:
    """Validate a nonempty portable corpus-relative path; reject absolute paths and traversal."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) == ".":
        raise ValueError("expected a nonempty corpus-relative POSIX path without '..'")
    return path.as_posix()


def resolve_path(root: Path, value: str) -> Path:
    """Resolve a relative input under its root, including a symlink containment check."""
    base = root.resolve()
    path = (base / relative_path(value)).resolve()
    if not path.is_relative_to(base):
        raise ValueError("an input path escapes the declared root")
    return path


def runtime_fingerprint() -> dict[str, Any]:
    """Record data-pipeline source hashes and numerical-library versions for cache isolation."""
    package = Path(__file__).resolve().parents[1]
    sources = sorted([*Path(__file__).parent.glob("*.py"), *(package / "preprocessing").glob("*.py")])
    return {
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ("mne", "numpy", "scipy", "polars", "orjson")},
        "source_hashes": {path.relative_to(package).as_posix(): file_hash(path) for path in sources},
    }


def write_json(path: Path, value: object) -> None:
    """Create a new JSON artifact; existing artifacts are never overwritten."""
    with path.open("xb") as handle:
        handle.write(orjson.dumps(value, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")

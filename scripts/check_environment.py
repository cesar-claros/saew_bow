"""Check the installed HPC environment without reading EEG data or initializing CUDA.

Requires the development dependency ``packaging``. Standard-library JSON is used
so a missing or broken runtime dependency, including orjson, can be reported.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import platform
import sys
import tomllib
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

logger = logging.getLogger(__name__)
RUNTIME_MODULES = (
    "loguru",
    "mne",
    "numpy",
    "orjson",
    "polars",
    "scipy",
    "sklearn",
    "threadpoolctl",
    "torch",
    "tqdm",
    "eeg_compare",
)


def collect_report(pyproject: Path, *, include_dev: bool = False, check_imports: bool = False) -> dict[str, object]:
    """Compare installed distributions with a project's declared requirements.

    Args:
        pyproject: Project TOML file to read.
        include_dev: Also check the dev group and build backend requirements.
        check_imports: Import the runtime modules to detect shared-library errors.

    Returns:
        A JSON-compatible report; ``ok`` is false if a check fails. Imports do not
        test GPU kernels or numerical behavior.

    Raises:
        OSError: The project file cannot be read.
        ValueError: TOML or requirement syntax is invalid.
        KeyError: A required project metadata field is absent.
    """
    with pyproject.open("rb") as handle:
        config = tomllib.load(handle)
    project = config["project"]
    python_version = platform.python_version()
    python_requirement = project["requires-python"]
    errors: list[str] = []
    if python_version not in SpecifierSet(python_requirement):
        errors.append(f"Python {python_version} does not satisfy {python_requirement}")

    requirements = list(project["dependencies"])
    if include_dev:
        requirements.extend(config["dependency-groups"]["dev"])
        requirements.extend(config["build-system"]["requires"])
    distributions: list[dict[str, object]] = []
    for requirement_text in requirements:
        requirement = Requirement(requirement_text)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        installed: str | None = None
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f"Missing distribution: {requirement.name}")
        satisfies = installed is not None and installed in requirement.specifier
        if installed is not None and not satisfies:
            errors.append(f"{requirement.name} {installed} does not satisfy {requirement.specifier}")
        distributions.append({"requirement": requirement_text, "installed": installed, "ok": satisfies})

    imports: list[dict[str, str]] = []
    if check_imports:
        for name in RUNTIME_MODULES:
            try:
                module = importlib.import_module(name)
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                message = f"{type(error).__name__}: {error}"
                errors.append(f"Import {name} failed: {message}")
                imports.append({"module": name, "error": message})
            else:
                imports.append({"module": name, "path": str(module.__file__)})

    return {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "python": python_version,
        "python_requirement": python_requirement,
        "executable": sys.executable,
        "platform": platform.platform(),
        "distributions": distributions,
        "imports_checked": check_imports,
        "imports": imports,
        "gpu_checked": False,
        "errors": errors,
        "ok": not errors,
    }


def main(argv: list[str] | None = None) -> int:
    """Write an environment report and return 0 on success, otherwise 1.

    Args:
        argv: CLI arguments; defaults to the process arguments.

    Returns:
        Exit status. Existing output files are refused. Argument parsing errors
        follow argparse's standard SystemExit behavior.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-dev", action="store_true", help="Check dev and build requirements too")
    parser.add_argument("--imports", action="store_true", help="Check runtime imports without testing CUDA")
    parser.add_argument("--output", type=Path, help="New JSON report path (also written to stdout)")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        report = collect_report(
            Path(__file__).resolve().parents[1] / "pyproject.toml",
            include_dev=args.include_dev,
            check_imports=args.imports,
        )
        payload = json.dumps(report, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
    except (OSError, ValueError, KeyError) as error:
        logger.error("Environment check failed: %s", error)
        return 1
    sys.stdout.write(payload)
    if not report["ok"]:
        logger.error("Environment requirements or imports failed; inspect the report's errors")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

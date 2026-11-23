# HPC Container Setup

Stage 1 can use the existing EEG container with a writable virtual environment.
A full CUDA image rebuild is not required if the checks below pass. Run these
commands on the HPC, inside the container, from this repository's root.

The audit on 2026-11-23 inspected the original workspace's `container/Dockerfile`
and `container/pyproject.toml`. It did not inspect a built image or run any code
inside it. Actual installed versions and native-library compatibility remain
HPC checks.

## Dependency Audit

| Requirement | Original recipe | Action |
|---|---|---|
| Python 3.12 | Both build stages use Python 3.12 | Reuse it to create the environment |
| Torch 2.6.0 | Pinned, with the cu124 package source | Preserve the existing Torch and compiled CUDA extensions |
| MNE, Polars, Loguru, tqdm | Declared directly | Check installed versions against this package's bounds |
| NumPy, SciPy | Transitive dependencies | Check versions; now explicit in the shared recipe |
| orjson | Not explicitly declared | Install if missing; now explicit in the shared recipe |
| pytest, mypy, Ruff | Dev group declared but not selected during installation | Install the comparison's dev group |
| uv | Builder executable omitted from the final image | Bootstrap in the writable environment |
| Hatchling, packaging | Build/check tooling | Install for editable packaging and the environment report |

The shared recipe has been updated to declare NumPy, SciPy and orjson, install
the dev group with Hatchling and packaging, and copy uv, pytest, mypy and Ruff
executables into the runtime stage. These changes apply to future builds;
they do not modify an existing SIF. The shared container files remain in the
original workspace, outside this repository.

## Use the Existing Image

Clone this repository into a writable location visible inside the container:

```bash
git clone https://github.com/cesar-claros/saew_bow.git
cd saew_bow
```

Use the container's Python interpreter to create a new environment. Do not reuse
an environment created on the laptop or outside the container. The following
block installs only the small additions, then installs the comparison package
without resolving or replacing its Torch dependency:

```bash
set -e
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install uv

UV_SYSTEM_PYTHON=false .venv/bin/uv pip install --python .venv/bin/python \
  --group dev "orjson>=3.10,<4" "hatchling>=1.26"
UV_SYSTEM_PYTHON=false .venv/bin/uv pip install --python .venv/bin/python \
  --no-deps --no-build-isolation -e .
```

The one pip command bootstraps uv because the old runtime recipe omits its
executable. The image sets `UV_SYSTEM_PYTHON=true`; each uv installation above
overrides it and explicitly targets the writable environment. Group selection
is explicit too, following the [uv package installation documentation](https://docs.astral.sh/uv/pip/packages/)
and [environment selection documentation](https://docs.astral.sh/uv/pip/environments/).

Run this setup where package downloads are allowed by the cluster. An offline
node needs staged wheels or a rebuilt image that already contains the additions.
Avoid `uv sync` in this inherited environment: the intended setup reuses the
container's scientific stack. Use the environment's Python directly for tests
and data commands.

## Check the Installed Environment

The report checks the Python version and declared runtime, dev and build
requirements, including upper bounds. `--imports` also checks imports for native
library errors. `torch==2.6.0` accepts the container's `2.6.0+cu124` version.
This check does not allocate a GPU, read EEG data, or certify numerical results.

```bash
set -e
mkdir -p artifacts/hpc-stage1-001
.venv/bin/python scripts/check_environment.py --include-dev --imports \
  --output artifacts/hpc-stage1-001/environment.json
.venv/bin/python -m pip freeze --all > artifacts/hpc-stage1-001/packages.txt
git rev-parse HEAD > artifacts/hpc-stage1-001/code-commit.txt

.venv/bin/python -m ruff check src tests scripts
.venv/bin/python -m mypy src/eeg_compare scripts/check_environment.py
.venv/bin/python -m pytest -q
```

Choose a new run directory for each attempt; the checker refuses to overwrite
an existing report. If the checker fails, resolve the listed missing or
out-of-range packages before running pytest or building manifests. Import
failures can require an image fix even when package versions satisfy the bounds.
Do not change Torch independently of the image's compiled CUDA extensions.

Record the SIF checksum or immutable image digest with the run. The package
snapshot, report and Git revision complement that image identity. A `uv.lock`
alone does not capture packages inherited from the image. For a future fully
isolated environment, resolve and commit its lockfile on the target platform,
with the intended PyTorch wheel source, before benchmarking.

Stage-1 reader and synthetic tests can run on a CPU allocation. GPU driver and
kernel validation belongs to later model experimentation; it is not certified
by the import check. The full three-method benchmark is not yet implemented.

## Build and Inspect TUEV Manifests

After the environment check and tests pass, follow the input identity contract
and acceptance checks in [TUEV Data Pipeline](TUEV%20Data%20Pipeline.md). Use the
same Python interpreter throughout:

```bash
set -e
.venv/bin/python -m eeg_compare.data.cli build \
  --root /path/to/tuev/edf --version YOUR_RELEASE \
  --identities data/tuev_identities.csv --config configs/tuev.toml \
  --output artifacts/tuev-manifest-001

.venv/bin/python -m eeg_compare.data.cli check \
  --root /path/to/tuev/edf --manifest artifacts/tuev-manifest-001 \
  --split train --fold 0 --limit 8 --cache-dir cache/tuev \
  --output artifacts/tuev-signal-check-001.json
```

The identity CSV must supply verified patient identities, consistently across
official train and evaluation splits. A passed environment report is only the
first gate; real-data timing, channels, exclusions and split checks remain
required on the HPC.

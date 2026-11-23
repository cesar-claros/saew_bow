# EEG Waveform Comparison

This is a standalone, selective copy of the reusable core from the earlier EEG
project, maintained at [cesar-claros/saew_bow](https://github.com/cesar-claros/saew_bow).
TUEV event classification is the first benchmark. The comparison plan is
in [Comparison Plan](docs/Comparison%20Plan.md); the extraction inventory is in
[Code Reuse](docs/Code%20Reuse.md).

The separable reference model, its full-template parent, spectral features,
normalization, resampling, FIR context calculation, and annotation parser are
present. The TUEV reader, example manifests, patient folds, configurable
preprocessing, lazy dataset and optional recording cache are implemented; their
runtime and corpus checks remain for the HPC. See the [TUEV data pipeline](docs/TUEV%20Data%20Pipeline.md)
for the input identity map, commands and exact policies.

The two published BOW implementations, common method adapters, classifiers, and
experiment runner remain planned. This folder does not yet run the three-way
benchmark.

## Layout

```text
docs/                       Comparison protocol and extraction record
src/eeg_compare/models/     Separable reference, parent model, spectral control
src/eeg_compare/preprocessing/
                            Row scaling, exact resampling, filter context
src/eeg_compare/data/        TUEV reader, manifest, splits, dataset and CLI
configs/tuev.toml            Explicit comparison preprocessing/example defaults
tests/                      Reference, extraction and TUEV regression checks
scripts/check_environment.py HPC dependency and import report
provenance.json             Source revision, source hashes, extraction changes
pyproject.toml              Standalone package and minimal dependencies
```

The package name is `eeg_compare`, so it cannot silently resolve imports from the
old project's generic `src` package. Model checkpoint tags also use this namespace.
Old checkpoint tags are rejected. No old artifacts or patient manifests are
included. This repository has its own Git history; the former repository's
history was not copied. Package files live directly at the repository root.

## Development on the HPC

For the existing EEG container, follow [HPC Container Setup](docs/HPC%20Container.md).
It creates a writable environment that reuses the image's Torch and scientific
stack, installs the small missing dependencies, and records an environment
report before testing. A full image rebuild is unnecessary if those checks pass.

For a fresh, isolated environment, use Python 3.12 and Torch 2.6.0. Choose the
CPU or CUDA wheel source for the cluster when resolving dependencies. The full
former container dependency list is intentionally omitted. These commands are
for that isolated workflow, not the inherited container environment:

```bash
uv sync --group dev
uv run ruff check src tests scripts
uv run mypy src/eeg_compare scripts/check_environment.py
uv run pytest
```

Resolve and commit `uv.lock` for the isolated workflow before benchmark runs.
For the inherited environment, record the image identity and full installed
package snapshot as described in the container guide. Record Python, Torch,
CUDA, MNE, NumPy, SciPy, and GPU versions with each run.
Dependency resolution and runtime tests require the HPC environment; the initial
copy records local static checks in [Code Reuse](docs/Code%20Reuse.md).
No runtime test or corpus validation was run during the local stage-1 implementation.

The retained model API accepts `DataLoader` batches of `(X, y)`, where `X` has
shape `(batch, channels, samples)`. Dictionary fitting requires training-patient
provenance. Patient-aware sample selection belongs in the future common sampler;
the copied trainer currently accepts `sampling_unit="segment"` only.
It rebuilds the supplied loader with uniform shuffling, so passing a custom
patient sampler alone will not implement the planned balancing. The benchmark
training adapter must preserve its declared sampling policy explicitly.

Reference defaults retain the original model behavior. In particular, the model
performs internal per-channel median/MAD scaling and clipping, and its full input
length must be divisible by the pooling length. Benchmark adapters must address
these explicitly instead of silently cropping five-second TUEV examples or
normalizing them twice. The separable model's existing `forward` output is a
morphology count readout; the planned adapter must preserve spatial winners too.

## Repository and Local Outputs

Clone `https://github.com/cesar-claros/saew_bow.git` to work on this package.
The local `code_comparison` directory is itself the repository root and has no
dependency on its parent directory. Keep data and outputs under the ignored
directories or at external locations supplied by configuration.

# EEG Waveform Comparison

This is a standalone, selective copy of the reusable core from the earlier EEG project, maintained at [cesar-claros/saew_bow](https://github.com/cesar-claros/saew_bow). TUEV event classification is the first benchmark. The comparison plan is in [Comparison Plan](docs/Comparison%20Plan.md); the extraction inventory is in [Code Reuse](docs/Code%20Reuse.md).

The separable reference model, its full-template parent, spectral features, normalization, resampling, FIR context calculation, and annotation parser are present. The TUEV reader, example manifests, patient folds, configurable preprocessing, lazy dataset and optional recording cache are implemented. The changes that the plan review required in stage 1 are implemented too: session clusters for the evaluation set, folds stratified by class presence, row weights, a conflict report, displaced control contexts, a merge rule for EDF files with identical bytes, the common signal contract, and the comparison adapters of the two retained models. Runtime and corpus checks of all of this remain for the HPC. See the [TUEV data pipeline](docs/TUEV%20Data%20Pipeline.md) for the input identity map, commands and exact policies.

The two published BOW implementations, the common tiling and features, classifiers, and experiment runner remain planned. This folder does not yet run the three-way benchmark.

## Layout

```text
docs/                       Comparison protocol and extraction record
src/eeg_compare/models/     Separable reference, parent model, spectral control (unchanged copies)
src/eeg_compare/adapters/   Comparison adapters of the retained models
src/eeg_compare/preprocessing/
                            Row scaling, exact resampling, filter context
src/eeg_compare/data/        TUEV reader, manifest, splits, dataset and CLI
configs/tuev.toml            Common comparison protocol, primary band (1 to 30 Hz)
configs/tuev_wide.toml       The declared second band (1 to 70 Hz with the notch)
tests/                      Reference, extraction, TUEV and adapter checks
scripts/check_environment.py HPC dependency and import report
scripts/make_tuev_identities.py
                            Identity CSV from the folder layout of the corpus
provenance.json             Source revision, source hashes, extraction changes
pyproject.toml              Standalone package and minimal dependencies
```

The package name is `eeg_compare`, so it cannot silently resolve imports from the old project's generic `src` package. Model checkpoint tags also use this namespace. Old checkpoint tags are rejected. No old artifacts or patient manifests are included. This repository has its own Git history; the former repository's history was not copied. Package files live directly at the repository root.

## Development on the HPC

For the existing EEG container, follow [HPC Container Setup](docs/HPC%20Container.md). It creates a writable environment that reuses the image's Torch and scientific stack, installs the small missing dependencies, and records an environment report before testing. A full image rebuild is unnecessary if those checks pass.

For a fresh, isolated environment, use Python 3.12 and Torch 2.6.0. Choose the CPU or CUDA wheel source for the cluster when resolving dependencies. The full former container dependency list is intentionally omitted. These commands are for that isolated workflow, not the inherited container environment:

```bash
uv sync --group dev
uv run ruff check src tests scripts
uv run mypy src/eeg_compare scripts/check_environment.py
uv run pytest
```

Resolve and commit `uv.lock` for the isolated workflow before benchmark runs. For the inherited environment, record the image identity and full installed package snapshot as described in the container guide. Record Python, Torch, CUDA, MNE, NumPy, SciPy, and GPU versions with each run. Dependency resolution and runtime tests require the HPC environment; the initial copy records local static checks in [Code Reuse](docs/Code%20Reuse.md). No runtime test or corpus validation was run during the local stage-1 implementation or its review changes.

## Models and adapters

The retained model API accepts `DataLoader` batches of `(X, y)`, where `X` has shape `(batch, channels, samples)`. Dictionary fitting requires training-patient provenance. Patient-aware sample selection belongs in the future common sampler; the copied trainer currently accepts `sampling_unit="segment"` only. It rebuilds the supplied loader with uniform shuffling, so passing a custom patient sampler alone will not implement the planned balancing. The benchmark training adapter must preserve its declared sampling policy explicitly.

Reference defaults retain the original model behavior. The reference performs internal per-channel median/MAD scaling and clipping on each segment, pads each segment with zeros, and needs an input length that the pooling length divides. Its `forward` output is the sum of the selected amplitudes of each atom divided by the number of pooling windows. It is not an occurrence count, and it has no spatial index.

`eeg_compare.adapters.models` holds the comparison versions, `ComparisonSeparableSAE` and `ComparisonWaveformSAE`, with their own checkpoint tags. They leave the reference files unchanged and differ in three points, which the spec and the checkpoint record:

- `input_scaling="external"`: the loader scales the signal per recording (`configs/tuev.toml`), and the model does not scale each segment again.
- `edge_mask=True`: a placement whose template reaches the zero padding cannot be selected, and the data-patch initialization draws its patches from inside the segment. All arms then read the same 1,250 real samples.
- `placements(X)`: the amplitude, position and spatial winner of every selected placement, with hard counts, amplitude sums and counts by spatial winner, over all pooling windows or a listed subset (for the event and context readouts).

With both options off, an adapter equals its reference. `SignalPairs` gives the `(signal, target)` items that the retained trainer reads from the dictionary items of `TUEVDataset`. A five-second example at 250 Hz has 1,250 samples, so the pooling length is 125, 250 or 625 samples.

## Repository and Local Outputs

Clone `https://github.com/cesar-claros/saew_bow.git` to work on this package. The local `code_comparison` directory is itself the repository root and has no dependency on its parent directory. Keep data and outputs under the ignored directories or at external locations supplied by configuration.

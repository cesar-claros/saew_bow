# EEG Waveform Comparison

This project compares three waveform-learning methodologies on TUEV. Read
`docs/Comparison Plan.md` and `docs/Code Reuse.md` before changing an experiment.

- Keep published reference implementations and proposed improvements separate.
- Use the `eeg_compare` package. Never import the old `src` package or add a sibling
  repository to `sys.path`.
- Data loading, training, evaluation, GPU work, and runtime tests run on the HPC.
  Local work is limited to editing, static analysis, syntax checks, and linting.
- Use patient-disjoint splits. Fit dictionaries, feature selection, normalization
  statistics, and classifiers using the designated training partition only.
- Treat the official evaluation set as held out until the study configuration is
  frozen. Never infer background labels for unannotated intervals.
- Keep data, manifests containing patient identifiers, caches, checkpoints, and
  results outside versioned source. Record their hashes in run artifacts.
- Use explicit run directories under `artifacts/`; never discover an old run by
  taking the newest directory or loading a checkpoint implicitly.
- Preserve numerical behavior when maintaining a reference. Record changes in
  provenance and test them against fixed synthetic fixtures on the HPC.
- Use `uv`, Ruff, mypy, and pytest. Keep dependencies in `pyproject.toml`, use
  Polars for new dataframe code, and add type hints and public API docstrings.
- Do not invent a license for copied code. Preserve upstream notices when
  external implementations are added.

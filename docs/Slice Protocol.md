# Slice Protocol (stage 3)

Confirmed by the user on 2026-11-26, with the tiling level of section 5. It makes stage 3 of the [Comparison Plan](Comparison%20Plan.md) exact. The slice is the first work on real data. It uses the training patients of TUEV only. The evaluation set stays closed.

## 1. What the stage decides

The slice is a measurement, not the comparison. It runs a few cheap arms through the complete path (signal, tokens, features, heads, evaluator) and measures what the later stages need:

1. **The resolution.** How wide is the interval of a paired difference between two arms on the 274 training patients? This gives the margin of a useful gain.
2. **The bar.** What do a class prior, spectral features and a bag of spectra reach?
3. **The record effect.** For each class, does the score come from the marked second, from its surroundings, or from the recording?
4. **The defaults of stage 4.** The number of atoms, the token score, the exposure and the input level.

It does not decide which method is better. Its learned arms are the two cheap single-channel cells only.

## 2. Data

- **Manifest:** `tuev-manifest-clip50-001` (primary band of 1 to 30 Hz, clip of 50, conflict rule `background_yields`).
- **Examples:** the 7,963 kept events of the 274 training patients, as merged intervals with their 5 s contexts, and their 7,963 displaced controls.
- **Folds:** the 5 folds of the manifest. Each fold fits everything on its training patients and predicts its validation patients. The predictions of the 5 folds are pooled before any score, so every training patient is predicted once.
- **Evaluation set:** not read. The code refuses it, and gate 2 checks this.

## 3. Exposure plans (decided on 2026-11-26)

The balance between patients is fixed in a stored list, and not in the training loop. A plan is a table of training units for one fold and one source. It is drawn once, without replacement, with a quota per patient, by a hash of the unit and the seed of the plan. Every arm reads the same list. The counts that the plan asks for (unique signal seconds, patients, passes) come from it.

| Source | Unit | Quota (primary) | Second level |
|---|---|---|---|
| Whole recordings | A 5 s interval on a fixed grid of a recording, with no use of the marks | 5 minutes per patient (60 intervals) | 10 minutes |
| Annotated contexts | A kept event with its 5 s context | 20 events per patient and class | 50 events |

On the full training set the first quota keeps 22.7 h of 94.1 h, with 269 of 274 patients at the quota. The second keeps 4,424 of 7,963 events and raises the effective number of patients from 64 to 217.

The slice fits its dictionaries on the whole-recording source. The annotated-context source needs cross-fitting and belongs to stage 4; the slice builds and checks its plan only.

## 4. Input levels (outcome of the scale audit)

- **Primary:** one robust scale per channel and recording, as in the manifest.
- **Second level, `profile`:** one recording scale times a fixed channel profile. The recording part is the mean log scale over the channels. The profile is the mean over the training recordings of a fold of what remains. A channel whose own scale is more than 3 times this product is divided by a third of its own scale, so that no channel is more than 3 times as loud as its profile says. The clip of 50 applies after the scale.

The second level runs for the arms that depend on the scale (section 6). It reads the recording without the scale and applies its own. Amendment of the build (2026-11-26, section 16.1, item 1): the level is computed from the primary input, with no second read of the recording.

## 5. Geometry and tokens

| Quantity | Value |
|---|---|
| Atom length L | 125 samples (0.5 s) |
| Pooling window P | 125 positions, by the center of the placement, as in stage 2; 250 as a level |
| Samples that the placements of one window cover | P + L - 1 = 249 |
| Windows of a 5 s context | 10; windows 4 and 5 are the labeled second |
| Atoms K | 32 (primary); 16 and 64 as levels |
| Atoms per window R | 1 |
| Token score | Cosine (primary); raw as a level on the same dictionaries |

A single-channel arm gives one token per channel and window (220 per context). A full-template arm gives one token per window (10 per context).

**P counts positions, not samples** (note of 2026-11-26, after a question of the user). A pooling window of 125 positions evaluates every atom at 125 placements and keeps the best one. These placements cover 249 samples. In the terms of [1] this is a window of 249 samples with an atom of 125 samples; the published tiling of [1] is a window of twice the atom, which is the same ratio. So an atom as long as P is already evaluated at many positions.

The ratio of P to L still matters, for another reason. The supports of two windows overlap by L - 1 samples. With P equal to L, a window next to an event sees about half of that event and gives a token from the part. With P of 250, an event in the middle of a window is not seen by the next windows, and a context gives half as many tokens (5 windows; the labeled second is window 2; windows 0 and 4 are the `context` readout). The slice measures P of 250 as a level.

## 6. Arms

| Arm | Definition | Seeds |
|---|---|---|
| `prior` | Predicts the class shares of the training part | |
| `spectral` | The retained spectral features: log energy and relative share in 1 to 4, 4 to 8, 8 to 13 and 13 to 30 Hz, per channel (176 features) | |
| `spectra_bag` | Bag of spectra [1]: k-means on the normalized amplitude spectra of the single-channel windows, K = 32, then counts | 3 |
| `S1-L1`, `S1-L2` | The two alternating single-channel cells, fitted on the whole-recording plan. The author checkout of [1] uses the update of `L1`; the text of the paper describes `L2`. Both run. | 3 |
| `S1-frozen-data`, `S1-frozen-gauss` | Single-channel dictionaries without learning: random windows of the plan, and Gaussian draws | 3 |
| `SF-frozen-data` | Full-template dictionary without learning, from random windows of the plan | 3 |

Levels, each with one seed unless stated:

- **K:** 16 and 64 for `S1-L1` and `S1-L2`.
- **Tiling:** P of 250 for `S1-L2` (one new fit per fold) and for the three frozen arms (no new fit).
- **Token score:** raw tokens for the learned and the frozen dictionaries at K = 32 (3 seeds; no new fit).
- **Input level:** `profile` for `spectral`, for `SF-frozen-data` and for the raw tokens of `S1-L2` (no new fit for the frozen arms; one new fit of `S1-L2`).
- **Exposure:** 10 minutes per patient for `S1-L2`.

A dictionary learned on events does not exist in the slice. The proposal of stage 2, a background dictionary as the null of a morphology claim, starts in stage 4, where event dictionaries exist.

## 7. Features and readouts

- **Features of a token arm:** the counts per atom over the tokens of the readout, divided by the number of tokens. The null token has its own count.
- **Scaling:** every feature is centered and scaled with the training part of the fold.
- **Readouts,** as in the plan, with no new fit of a dictionary:

| Readout | Windows | Tokens of a single-channel arm |
|---|---|---|
| `full` | All 10 | 220 |
| `event` | Windows 4 and 5 | 44 |
| `context` | Windows 0, 1, 2, 7, 8 and 9: their supports hold no sample of the labeled second | 132 |
| `displaced` | All 10 windows of the displaced control of the event | 220 |

For `spectral`, a readout uses the samples that the token windows of that readout cover: samples 0 to 1,249 for `full`, 438 to 811 for `event`, 0 to 436 and 813 to 1,249 for `context`, and the whole displaced context for `displaced`. A head is trained and tested inside one readout.

## 8. Heads

Two heads on the same features, with fixed settings and no search:

- **Logistic regression:** multinomial, L2 penalty with C = 1.
- **Random forest:** 500 trees, at least 2 examples per leaf, 3 seeds.

Both use one weight per example: inside a class every patient has the same total weight, and every class has the same total weight. A ranking of two arms counts only if it holds on both heads.

## 9. Evaluator

- **Primary score:** balanced accuracy over the six classes on the merged intervals, on the pooled out-of-fold predictions, with the logistic head and the `full` readout.
- **Also reported:** recall of each class, macro F1, Cohen kappa, the row-weighted and the patient-weighted form of each score, and the confusion matrix.
- **Intervals:** a paired bootstrap over patients, 2,000 draws, inside strata. The stratum of a patient is the rarest class that it holds. The half-width of an interval is half the distance between its 2.5% and 97.5% points.
- **Seeds:** the score of an arm is the mean over its seeds. The spread between seeds is reported apart from the bootstrap interval.

## 10. What the slice measures

| | Measure | Use |
|---|---|---|
| M1 | Half-width `h` of the paired interval of `S1-L2` against `S1-frozen-data` | The margin of later stages is not below `h` |
| M2 | The scores of `prior`, `spectral` and `spectra_bag` | The bar |
| M3 | The recall of each class in the four readouts, for every arm | The record effect |
| M4 | Learned against frozen: `S1-L1` and `S1-L2` against the two frozen single-channel arms | Does learning add to a random dictionary? |
| M5 | Cosine against raw tokens on one dictionary | The token score of stage 4 |
| M6 | K of 16, 32 and 64 | The number of atoms of stage 4 |
| M7 | Primary input against `profile` | The input of stage 4 |
| M8 | 5 against 10 minutes per patient | The exposure of stage 4 |
| M9 | Head features in sample against cross-fitted, once: a dictionary from one half of the training patients; one head trained on that half and one on the other half; both tested on the validation patients | The need for cross-fitting with the whole-recording source |
| M10 | Time and memory of every fit and every pass over the examples | The cost of stage 4 |
| M11 | P of 125 against 250 | The tiling of stage 4 |

## 11. Predictions

Fixed here, before any code of the slice exists. `h` is the half-width of M1.

| | Prediction | My probability |
|---|---|---|
| S1 | `h` lies between 0.02 and 0.05 | 0.60 |
| S2 | `S1-L2` is above `S1-frozen-data` by more than `h` | 0.35 |
| S3 | `spectra_bag` is within `h` of `S1-L2`, or above it | 0.60 |
| S4 | `spectral` has the highest score of the slice | 0.55 |
| S5 | For `gped` and for `pled`, the recall of the `displaced` readout is within 0.10 of the `full` readout | 0.65 |
| S6 | Cosine tokens are above raw tokens by more than `h` for `S1-L2` | 0.40 |
| S7 | The input level changes `spectral` by less than `h` | 0.60 |

S2, S3 and S4 follow the earlier TUEV stage of the source project, in which crop dictionaries did not beat random waveforms or theta power.

## 12. Gates

1. **Tests.** The suite passes on the HPC with the new modules.
2. **No leak.** No patient is in both parts of a fold. Every plan holds training patients of its fold only. No evaluation recording is read.
3. **Null.** With the labels of the training part permuted inside each fold, `spectral` reads within 0.03 of chance (0.167).
4. **Plans.** No patient exceeds a quota, no unit repeats, and a second build gives the same plan.
5. **Counts.** The counts of every context add up to its number of tokens, in every readout.
6. **Determinism.** A repeated fold gives equal predictions.

## 13. What the slice sets

- **Margin.** The margin of a useful gain in the later stages is the larger of 0.02 and `h`.
- **Record effect.** If the `displaced` or the `context` readout of a class is within the margin of its `full` readout for the best arm, the later stages make no waveform claim for that class.
- **Defaults.** For K, the token score, the tiling, the input level and the exposure, a level replaces the primary value only if it is above the primary by more than `h` with `S1-L2` on both heads. If not, the primary or the cheaper value stays. A score that follows such a choice is labeled as optimistic.
- **Resolution.** If `h` is above 0.05, the benchmark cannot rank methods that differ as the published ones do (0.40 to 0.47). Stage 4 then runs as a described comparison with a reduced grid, and the plan says so before it starts.

## 14. Order and cost

1. The exposure plans, the `profile` input, the interval reader, and their tests (gates 1, 2 and 4).
2. The token features, the bag of spectra, the heads and the evaluator, on the frozen arms and the controls (gates 3, 5 and 6; M2 and M3 for these arms).
3. The learned arms and the levels (M1 and M4 to M10), then the read of the predictions.

My estimate, without a measured time: about 65 alternating fits, some minutes each, so 3 to 6 hours of GPU time for the whole slice.

Memory. A fit of one fold reads 13,140 intervals, which are 2.89 million single-channel windows of 249 samples. In single precision the windows take 2.9 GB, the intervals 1.4 GB and the mask of valid positions 0.4 GB. The peak of a fit is about 6.3 GB, and about 12.5 GB at 10 minutes per patient. The GPU of the project has 32 GB, so the learner as built fits, and no second form is necessary. From the times of stage 2, one pass takes about 4 s at 32 atoms.

New declared dependency: scikit-learn, for the two heads and for k-means.

## 15. Points to confirm

1. The geometry: atoms of 0.5 s, pooling windows of 0.5 s (125 positions that cover 249 samples), with 1 s as a level, and K of 32 with 16 and 64 as levels.
2. The arms: both `S1-L1` and `S1-L2`, the three frozen arms, and no gradient cell.
3. The heads with fixed settings and no search.
4. The rules of section 13.

## 16. Build record

### 16.1 Step 1: plans, second input level, reader of plan units (built 2026-11-26)

**State.** Built, and run on the HPC on 2026-11-27 at commit `ed6fc96`. Gate 1 passes (236 tests). Gate 4 passes: the build wrote `artifacts/tuev-exposure-001`, and the check found the stored plan equal to a new build. The part of gate 2 that concerns the plans passes, because the build checks every rule before it writes. The 12 lines of the script equal the expected counts below, line by line. The other part of gate 2 (no evaluation recording is read) belongs to the runs of step 2.

**Code.** The slice has its own package, `src/eeg_compare/slice/`. It is not part of `eeg_compare.data`, because the reader refuses a manifest when a module of that package changes, and a plan does not change how a manifest is made.

| File | Content |
|---|---|
| `slice/exposure.py` | The units of both sources, the draw, the plans of all folds as one table, the check of a plan, and storage |
| `slice/scaling.py` | The channel profile of a fold and the factor of a recording for the `profile` input; it reads the scales from the report of the scale audit |
| `slice/signals.py` | `load_units` gives the signals of plan units with one read per recording; `study_clip` refuses a manifest that the slice cannot read |
| `scripts/make_exposure.py` | Builds the plans of a manifest in a new directory, or verifies stored plans |
| `tests/test_slice_exposure.py`, `tests/test_slice_signals.py` | 16 tests |

**How a plan is made.** The units of a patient are ranked once, over all its training recordings, by a hash of the seed and the unit. The first units up to the quota are kept. The plan of a fold is the cut of that one draw to the training patients of the fold. So a patient has the same units in every fold where it trains, the draw of a patient does not depend on the order of the table or on other patients, and a larger quota holds the units of a smaller one. A stored plan is a directory with `plan.parquet` and `metadata.json` (configuration, hashes of the manifest and of the plan, counts).

**Gates 2 and 4 in code.** `check_plans` raises for a patient outside the training part of its fold, an evaluation recording, a repeated unit, a patient above a quota, an interval that is off the grid or outside its recording, and a context that is not a kept training event. `load_plans` refuses a plan whose file changed, a plan of another manifest, and a plan that differs from a new build with its configuration.

**Details that the confirmed text did not fix.**

1. **The `profile` input comes from the primary input.** This changes one sentence of section 4. The build multiplies every channel of the primary input by `min(r, 3)` and clips again at 50. `r` is the own scale of the channel, divided by the product of the recording scale and the profile. Reason: a second read needs a second manifest without the scale and a second signal cache of 94 h. The two forms are equal for every sample that the first clip did not change. They differ only for a sample at the first clip in a channel with a factor below 1: the build gives 50 times the factor, and the exact form gives a value between that and 50. Rule, fixed here before any signal is read: step 2 reports the share of such samples in the plan intervals, and if the share is above 0.1%, M7 is reported as approximate.
2. **A plan holds real samples only.** `context_units` refuses a manifest in which a kept event has zero padding. The manifest of the study drops the events at the edge of a recording, so it has none (0 of 7,963).
3. **The scale of the manifest.** `study_clip` refuses a manifest that does not divide every channel by its robust scale per recording, or that has no clip. The stored signal of such a manifest is not the primary input.
4. **The source of the scales.** The factors come from `artifacts/tuev-scale-audit-001.json`, which holds the scale of every channel of the 339 training recordings. `read_scales` refuses a report with other channels or another band.

**Expected counts.** I counted the manifest tables with a separate short script that shares no code with `exposure.py` (tables only, no signal). The script of the plans must print these numbers at the primary quotas. The line `all` counts a unit once over all folds.

| Fold | Source | Units | Patients | Effective patients | Largest patient | Hours |
|---|---|---|---|---|---|---|
| 0 | annotated contexts | 3,530 | 218 | 170.5 | 54 | 4.90 |
| 0 | whole recordings | 13,017 | 218 | 217.7 | 60 | 18.08 |
| 1 | annotated contexts | 3,555 | 219 | 173.0 | 54 | 4.94 |
| 1 | whole recordings | 13,111 | 219 | 218.9 | 60 | 18.21 |
| 2 | annotated contexts | 3,517 | 219 | 172.0 | 54 | 4.88 |
| 2 | whole recordings | 13,092 | 219 | 218.8 | 60 | 18.18 |
| 3 | annotated contexts | 3,544 | 220 | 176.8 | 40 | 4.92 |
| 3 | whole recordings | 13,132 | 220 | 219.7 | 60 | 18.24 |
| 4 | annotated contexts | 3,550 | 220 | 174.4 | 54 | 4.93 |
| 4 | whole recordings | 13,136 | 220 | 219.7 | 60 | 18.24 |
| all | whole recordings | 16,372 | 274 | 273.7 | 60 | 22.74 |
| all | annotated contexts | 4,424 | 274 | 216.6 | 54 | 6.14 |

A fit of one fold reads at most 13,136 intervals, so the memory estimate of section 14 (13,140 intervals) holds. At the second levels the line `all` must give 31,457 intervals (43.69 h, 268.5 effective patients) and 5,884 events (177.7 effective patients).

**Commands on the HPC,** from the repository root inside the container:

```bash
python -m pytest -q
```

```bash
PYTHONPATH="$PWD/src" python scripts/make_exposure.py --manifest artifacts/tuev-manifest-clip50-001 --output artifacts/tuev-exposure-001
```

```bash
PYTHONPATH="$PWD/src" python scripts/make_exposure.py --manifest artifacts/tuev-manifest-clip50-001 --check artifacts/tuev-exposure-001
```

The first command is gate 1. The second and third are gates 2 and 4 on the manifest of the study: the build checks every rule before it writes, and the check compares the stored plan with a new build. The plans of the second levels are not necessary before step 3.

### 16.2 Step 2: features, heads, evaluator, frozen arms and controls (built 2026-11-27)

**State.** Built and run. The suite passes on the HPC (263 tests at commit `36f1fd3`). The full run passes gates 2, 3, 5 and 6; its results are in section 17.

**Code.**

| File | Content |
|---|---|
| `slice/readouts.py` | The four readouts from the geometry: their pooling windows and the samples that those windows cover |
| `slice/features.py` | Token counts, the spectral control for a readout, and the spectra of the single-channel windows |
| `slice/spectra.py` | The k-means model of the bag of spectra and its tokens |
| `slice/heads.py` | The two heads and the `prior` arm |
| `slice/evaluate.py` | Example weights, scores of a confusion matrix, strata and the bootstrap over patients (NumPy only) |
| `slice/study.py` | The events with their controls and folds, one pass over the recordings, a reader that refuses the evaluation set, and the count for the rule of section 16.1 |
| `slice/runner.py` | The fold loop for the arms of this step, gates 3, 5 and 6, and the stored run |
| `slice/report.py` | The read of a stored run: gates, scores, intervals (NumPy and Polars only, so a copied run can be read without Torch) |
| `scripts/run_slice.py` | `run` writes a run in a new directory and prints its report; `read` prints the report of a stored run |
| `tests/test_slice_features.py`, `tests/test_slice_heads.py`, `tests/test_slice_runner.py` | 27 tests |

A run is a directory with `predictions.parquet` (one row per example, arm, seed, head and readout, with the class probabilities), `examples.parquet` and `report.json`. New declared dependencies: `scikit-learn>=1.4,<2` and `threadpoolctl>=3.1,<4` (scikit-learn installs it; the slice uses it to fix the number of threads); the environment check imports both.

**Details that the confirmed text did not fix.** These are my choices. A change is cheap before the run and not after it.

1. **Frames of the spectral control.** A frame is 1 s, so a bin is 1 Hz. On a whole context this is the retained calculation, and a test compares the two. The `event` readout has 374 samples and each part of the `context` readout has 437. There the frames are spaced evenly with an overlap of at least 50%: 2 frames for `event`, and 3 and 3 for `context`, which enter one mean.
2. **Bag of spectra.** A window is the support of one pooling window of one channel (249 samples), as for the token arms, so the readouts cover the same samples. Only the samples inside the context enter; their mean is removed and a Hann taper is applied. The spectrum is the amplitude in the bins of 1 to 30 Hz (30 numbers), scaled to unit norm. k-means (scikit-learn, one `k-means++` start, at most 100 passes) is fitted on all windows of the plan of the fold. A window without signal gets the null token.
3. **Frozen dictionaries.** `S1-frozen-data` and `SF-frozen-data` come from the common start of stage 2, drawn from the plan intervals of the fold: the same K windows, with one channel or with all channels. The learned arms of step 3 start from this draw, so M4 compares a learned dictionary with its own start. A draw that holds a flat patch is drawn again from the same stream, and the report holds the number of draws.
4. **`prior`.** It predicts the most frequent class of the training part, by events. Its balanced accuracy is 1/6 by construction.
5. **Gate 3 reads the mean of 10 permutations.** The patients of a class cluster, so one permutation can leave chance by more than 0.03 with no leak. The report gives the mean and the range of the 10 values.
6. **Gate 6 repeats fold 0 from the signals:** features, k-means, dictionaries, tokens and heads. It passes when every predicted class is equal. The largest probability difference is reported.
7. **Gate 5.** The report compares the tokens per context with the numbers of section 7 (220, 44, 132 and 220 for a single-channel arm; 10, 2, 6 and 10 for a full-template arm) and checks that the shares of every context add up to 1.
8. **Strata of the bootstrap.** A class is rarer when fewer patients hold it; equal numbers are ordered by the number of events. In the corpus the order is `pled` (23 patients), `spsw` (24), `gped` (25), `eyem` (33), `artf` (117) and `bckg` (180).
9. **Seeds.** The forest has 3 seeds for every seed of an arm, so an arm with seeds has 9 forests per fold and readout.
10. **Logistic regression.** scikit-learn with its default solver, C = 1, at most 2,000 passes and a tolerance of 1e-6. The report holds the largest number of passes, so a fit that did not converge is visible. The features are centered and scaled with the unweighted mean and standard deviation of the training part.
11. **Preview of the resolution.** The report prints the paired difference of five pairs of arms on the `full` readout. This is not M1, which needs `S1-L2`.

**Smoke run of 2026-11-27** (commit `4b19634`, root `artifacts/slice-step2-smoke`: 8 atoms, 1 seed, 20 trees, 2 permutations; 30 minutes). The suite passed at this commit on the HPC (261 tests, gate 1). The run read all 339 training recordings, 7,963 events of 274 patients and 16,372 plan intervals.

| Gate | Result |
|---|---|
| 2, no leak | Passes: 339 training recordings read, no evaluation recording, no patient in both parts of a fold |
| 3, null | Passes at 2 permutations: mean 0.177. The two values are 0.150 and 0.205; the second alone is 0.038 from chance, outside the margin. This is the spread that item 5 expects, and the full run reads 10 |
| 5, counts | Passes: 220, 44, 132 and 220 tokens for the single-channel arms, 10, 2, 6 and 10 for the full-template arm; every context adds up to 1 |
| 6, determinism | **Fails.** The second computation of fold 0 gave another predicted class for at least one example; the largest probability difference before that point was 0.0032 |

- **Cause of the failure of gate 6: probably k-means with several threads.** scikit-learn adds the partial sums of its threads in an order that is not fixed, so two fits of the same data differ in the last digits, and a few windows change their token. A change of one token in 220 moves a logistic probability by about the size that the run shows. This is an inference: the stored report stopped at the first difference and did not name the arm, which is a defect of my report. Changes of the build: k-means runs with one thread, and gate 6 now stores, for every arm, the number of predictions that differ and the largest probability difference. A test fits k-means twice and compares the centers.
- **Speed.** A logistic fit on the 176 spectral features took about 45 s for 1,025 passes, and the logistic fits were most of the 30 minutes. I attribute this to the threads of a whole node on a small problem; I did not measure it. The logistic fit now runs with one thread. No fit reached the limit of 2,000 passes.
- **Rule of the `profile` input** (section 16.1, item 1). The share of plan samples at the clip is 0.05% to 0.07% by fold. The share at the clip in a channel with a factor below 1 is 0.022% to 0.033%, below the limit of 0.1%. So M7 needs no label. This number does not depend on the small settings.
- **Numbers that are already final.** `spectral` with the logistic head has no setting that the smoke run changes, so its numbers are those of the full run: balanced accuracy 0.441 (0.386 to 0.509) on the `full` readout, 0.466 on `event`, 0.380 on `context` and 0.342 on `displaced`. `prior` reads 0.167.
- **Not results.** The scores of the token arms and of the bag of spectra come from 8 atoms and one seed. They show that the path works (with the logistic head all are between 0.33 and 0.38, above the prior) and nothing more.

**Second smoke run of 2026-11-27** (commit `36f1fd3`, root `artifacts/slice-step2-smoke-2`, the same small settings; 2.6 minutes). The suite passed at this commit (263 tests).

- **All four gates pass.** Gate 6: the two computations of fold 0 are equal bit by bit in all 7 arms (56,545 predictions, largest probability difference 0).
- **Speed.** The run took 157 s against 1,804 s. The logistic fits took 9 s per fold against 190 s, so the threads were the cause of the slow fits.
- **Cause of the first failure of gate 6.** I compared the stored predictions of the two smoke runs, which differ only in the number of threads. The three dictionary arms and the forest on the spectral features are equal bit by bit. The logistic head on the spectral features differs by at most 0.005 in a probability and in no predicted class. Only the bag of spectra changes its predicted classes: 7 of 31,852 with the logistic head and 2,247 of 31,852 with the forest. So the tokens of k-means were the source, and the signals, the features on the GPU, the dictionaries and the forest repeat exactly.
- **A caution for the bag of spectra.** Last-digit differences in k-means moved 7% of the forest predictions at the small settings (8 centers, 20 passes, 20 trees), while the balanced accuracy moved from 0.368 to 0.369. Single predictions of this arm are fragile; its score is not. The three seeds of the full run measure this spread.
- **Numbers that did not move:** `spectral` with the logistic head reads 0.441 (0.386 to 0.509) again, and the null and the rule of the `profile` input are as in the first run.

**Cost.** My estimate from the times of the second smoke run: about 1 hour (40 to 90 minutes), most of it on the CPU (936 forests of 500 trees and 18 k-means fits on about 2.9 million spectra). The first read filters 339 recordings. With `--cache-dir` the filtered recordings are stored (about 7.4 GB for the whole training set) and later runs read them from disk.

**Commands on the HPC,** from the repository root inside the container. `$DATA_DIR/v2.0.1/edf` is the corpus root of the earlier commands.

```bash
PYTHONPATH="$PWD/src" python scripts/check_environment.py --imports
```

```bash
python -m pytest -q
```

```bash
PYTHONPATH="$PWD/src" python scripts/run_slice.py run --smoke --root "$DATA_DIR/v2.0.1/edf" --manifest artifacts/tuev-manifest-clip50-001 --plans artifacts/tuev-exposure-001 --scales artifacts/tuev-scale-audit-001.json --cache-dir cache/tuev --output artifacts/slice-step2-smoke
```

```bash
PYTHONPATH="$PWD/src" python scripts/run_slice.py run --root "$DATA_DIR/v2.0.1/edf" --manifest artifacts/tuev-manifest-clip50-001 --plans artifacts/tuev-exposure-001 --scales artifacts/tuev-scale-audit-001.json --cache-dir cache/tuev --output artifacts/slice-step2-001
```

The first command confirms that scikit-learn is present. Result of 2026-11-27: scikit-learn 1.9.1 is in the image and every declared requirement is met. My first form of the command had no `PYTHONPATH`, so it reported that `eeg_compare` was missing; that was a defect of the command and not of the environment. The second is gate 1 (263 tests with the changes after the smoke run; 261 passed at commit `4b19634`). The third checks the wiring on the real data with small settings; its scores are not results. The fourth is the run of step 2 and prints gates 2, 3, 5 and 6, M2 and M3.

### 16.3 Step 3: learned arms, levels and the rules of the read (built 2026-11-27)

**State.** Built and run. The full run passes gates 2, 3, 5 and 6; its results are in section 18. I have no test output for the commits of step 3 (the suite then has 270 tests); the smoke run and the full run used that code.

**Code.**

| File | Content |
|---|---|
| `slice/levels.py` | The learned arms and the levels of one fold, the `profile` input of rows, the halves of the cross-fitting check, and a run that is stored fold by fold |
| `slice/decide.py` | `measure` takes M1 and M4 to M11 from a stored run; `evaluate` applies the rules below to those numbers and reads no data |
| `slice/study.py`, `slice/runner.py`, `slice/report.py` | A second plan with the larger quota; the heads of a fold as a shared class; a shared loader of a stored run; the token numbers of the wide tiling in gate 5 |
| `scripts/run_slice.py` | `levels` runs or continues step 3; `read` also prints the measures and the rules |
| `tests/test_slice_levels.py` | 7 tests |

**A run of step 3 holds the arms of step 2 too,** so every paired difference lies inside one run. An arm with a level has the name of the arm, an `@` and the level:

| Arm | What changes against the primary setting | Seeds |
|---|---|---|
| `S1-L1`, `S1-L2` | The two alternating cells, fitted on the plan intervals of the fold | 3 |
| `<arm>@raw` | Raw tokens on the same dictionary, for the two learned and the three frozen arms | 3 |
| `S1-L1@K16`, `S1-L1@K64`, `S1-L2@K16`, `S1-L2@K64` | The number of atoms | 1 |
| `S1-L2@P250`, and the three frozen arms `@P250` | The pooling window; a new fit for `S1-L2`, none for the frozen arms | 1 |
| `S1-L2@10min` | The plan of 10 minutes per patient | 1 |
| `spectral@profile`, `SF-frozen-data@profile`, `S1-L2@profile-raw` | The `profile` input | 1 |
| `S1-L2@half-in`, `S1-L2@half-out` | The cross-fitting check (M9) | 1 |

**The rules of the read, fixed here before any learned arm has run.** A difference is the first arm minus the second on the `full` readout, as the mean over the seeds of each arm. A level has one seed (the first), and it is compared with the same seed of the primary arm. "Both heads" means the logistic head and the forest.

| | Rule |
|---|---|
| `h` (M1) | The half-width of the paired interval of `S1-L2` minus `S1-frozen-data`, logistic head, 3 seeds each |
| Margin | The larger of 0.02 and `h` |
| Resolution | `h` above 0.05 gives a described comparison in stage 4 |
| S1 | `h` is from 0.02 to 0.05 |
| S2 | `S1-L2` minus `S1-frozen-data` is above `h` |
| S3 | `spectra_bag` minus `S1-L2` is not below minus `h` |
| S4 | `spectral` has the highest primary score among the seven arms at their primary setting (`spectral`, `spectra_bag`, `S1-L1`, `S1-L2` and the three frozen arms) |
| S5 | For the arm with the highest primary score, the `full` recall minus the `displaced` recall is at most 0.10 for `gped` and for `pled` |
| S6 | `S1-L2` minus `S1-L2@raw` is above `h` |
| S7 | The absolute difference of `spectral@profile` and `spectral` is below `h` |
| Defaults | A level replaces the primary value only if `S1-L2` at the level is above `S1-L2` at the primary value by more than `h` on both heads. The pairs: `@K16` and `@K64`, `@raw`, `@P250` and `@10min` against `S1-L2`; `@profile-raw` against `S1-L2@raw` |
| Record effect | For the arm with the highest primary score: a class gets no waveform claim if its `full` recall minus its `displaced` recall, or minus its `context` recall, is at most the margin |

**Details that the confirmed text did not fix.**

1. **Budget of a fit.** At most 100 passes and a tolerance of 1e-4, the values of stage 2. The report holds, for every fit, the passes, whether it converged, the residual share, the share of assigned units and the atoms without a unit.
2. **The start.** A learned arm starts from the common start of its fold and seed, which is the dictionary of `S1-frozen-data`. So M4 compares a dictionary with its own start. A level of the number of atoms has its own draw.
3. **The tiling.** `S1-L2@P250` is a new fit from the same start. The frozen arms keep their dictionaries. A context then has 5 windows: the event is window 2 and the context readout is windows 0 and 4.
4. **The `profile` input for a single-channel dictionary uses raw tokens.** A factor per channel does not change a cosine token of one channel, so only raw tokens can show the input level. `S1-L2@profile-raw` is a new fit on the `profile` input and is compared with `S1-L2@raw`.
5. **`SF-frozen-data@profile`** takes the windows of the start at the same places, from the `profile` input. This is no fit, as the protocol states, and the arm then sees one input level in its dictionary and in its contexts.
6. **The exposure.** The plan of 10 minutes holds every unit of the plan of 5 minutes. The fit starts from the same start, so only the exposure changes.
7. **M9 runs in every fold,** not once, so that both heads have pooled predictions. The training patients of a fold are cut into two halves, balanced inside every stratum. The dictionary comes from the plan intervals of the first half. One head trains on the events of the first half and one on the events of the second half. Both predict the validation patients.
8. **Gate 6** computes fold 0 twice, with all its fits.
9. **A run is stored fold by fold.** The same command continues a run that stopped. A stored fold is never computed again, and a directory with other settings or inputs is refused.

**Cost.** A fold has 14 alternating fits (6 at the primary setting, 4 for the number of atoms, and one each for the tiling, the exposure, the `profile` input and the cross-fitting check), 576 forests and 192 logistic fits. With the repeat of fold 0 the run has 84 fits. I have no time of a pass on the corpus. If a pass takes 4 s at 32 atoms, as the times of stage 2 suggest, a fold takes between 1 hour (fits that stop after 30 passes) and 2.2 hours (all fits at 100 passes), and the run 6 to 13 hours. The smoke run measures the time of a pass. The run needs about 20 GB of memory and about 12 GB on the GPU (the fit at 10 minutes).

**Commands on the HPC,** from the repository root inside the container.

```bash
python -m pytest -q
```

```bash
PYTHONPATH="$PWD/src" python scripts/make_exposure.py --manifest artifacts/tuev-manifest-clip50-001 --minutes 10 --output artifacts/tuev-exposure-10min-001
```

```bash
PYTHONPATH="$PWD/src" python scripts/run_slice.py levels --smoke --root "$DATA_DIR/v2.0.1/edf" --manifest artifacts/tuev-manifest-clip50-001 --plans artifacts/tuev-exposure-001 --wide-plans artifacts/tuev-exposure-10min-001 --scales artifacts/tuev-scale-audit-001.json --cache-dir cache/tuev --output artifacts/slice-step3-smoke
```

```bash
PYTHONPATH="$PWD/src" python scripts/run_slice.py levels --root "$DATA_DIR/v2.0.1/edf" --manifest artifacts/tuev-manifest-clip50-001 --plans artifacts/tuev-exposure-001 --wide-plans artifacts/tuev-exposure-10min-001 --scales artifacts/tuev-scale-audit-001.json --cache-dir cache/tuev --output artifacts/slice-step3-001
```

The suite must show 270 passed. The plan of 10 minutes must print, in its two `all` lines, 31,457 intervals (43.69 h, 268.5 effective patients) and 4,424 events. The smoke run uses 8 atoms, one seed and 5 passes; its scores are not results. The last command is the run of step 3; if it stops, the same command continues it.

**Smoke run of 2026-11-27** (commit `40031f4`, root `artifacts/slice-step3-smoke`: 8 atoms, one seed, 5 passes, 20 trees; 12.9 minutes). I have no test output for this commit. The plan of 10 minutes was built (31,457 intervals, as expected).

- **All four gates pass.** Gate 6: the two computations of fold 0 are equal bit by bit in all 28 arms (277,465 predictions). So the alternating fits on the GPU repeat exactly.
- **The arms of step 2 did not move.** Their 342,409 predictions in this run equal those of the second smoke run of step 2 bit by bit, the bag of spectra included. So the changes of the build in the runner changed no result, and a run repeats in another process.
- **The read works on the corpus:** the 92 pairs of gate 5, the fits, M1, M4 to M11 and the rules.
- **Time of a fit.** One pass over the 13,017 intervals of a fold takes 0.9 s at 4 atoms, 1.1 s at 8 and 1.2 s at 16; 1.9 s for the plan of 10 minutes; 0.8 s with windows of 250 positions. My estimate before the run was 4 s at 32 atoms; from these times I expect 1.5 to 2 s. No fit can converge in 5 passes, and none did. No atom was without a unit.
- **Cost of the full run, corrected:** 3.5 to 6.5 hours. A fold has at most 43 minutes of fits (all at 100 passes), 19 minutes of forests and about 4 minutes for the rest. Peak GPU memory was 13.7 GB.
- **A number that is already final.** `spectral@profile` has no setting that the smoke run changes: it reads 0.424 against 0.441 for `spectral`, a difference of -0.017 (-0.048 to +0.026) with the logistic head and +0.009 with the forest. S7 needs `h` from the full run.
- **Not results.** All scores of the learned arms and of the levels come from 8 atoms, one seed and 5 passes.

## 17. Results of step 2 (run of 2026-11-27)

Root `artifacts/slice-step2-001`, commit `36f1fd3`, 45.3 minutes. The run read 339 training recordings: 7,963 events of 274 patients, their displaced controls and 16,372 plan intervals. All scores are balanced accuracies of pooled out-of-fold predictions on the training patients. The evaluation set was not read. Intervals come from 2,000 bootstrap draws over patients.

### 17.1 Gates

| Gate | Result |
|---|---|
| 1, tests | Passes: 263 tests |
| 2, no leak | Passes: 339 training recordings read, no evaluation recording, no patient in both parts of a fold |
| 3, null | Passes by its rule: mean 0.187 of 10 permutations, against chance 0.167 and a margin of 0.03. See section 17.5, item 1 |
| 5, counts | Passes: 16 pairs of arm and readout equal the token numbers of section 7 |
| 6, determinism | Passes: fold 0 repeats bit by bit in all 7 arms (287,985 predictions, largest probability difference 0) |

The rule of the `profile` input (section 16.1) is met: 0.033% at most, against 0.1%. No logistic fit reached its limit (1,034 passes of 2,000). No start needed a second draw.

### 17.2 The bar and the arms without a learned dictionary (M2)

`full` readout. The score of an arm is the mean over its 3 seeds. "By patients" gives every patient the same weight inside a class.

| Arm | Logistic (primary) | Forest | Seeds, logistic | By patients, logistic | By patients, forest |
|---|---|---|---|---|---|
| `prior` | 0.167 | | | | |
| `spectral` | 0.441 (0.386 to 0.509) | 0.476 (0.410 to 0.561) | | 0.468 | 0.512 |
| `spectra_bag` | 0.396 (0.339 to 0.471) | 0.383 (0.322 to 0.469) | 0.379 to 0.412 | 0.432 | 0.447 |
| `S1-frozen-data` | 0.434 (0.388 to 0.477) | 0.406 (0.361 to 0.453) | 0.427 to 0.443 | 0.425 | 0.428 |
| `S1-frozen-gauss` | 0.424 (0.378 to 0.477) | 0.381 (0.334 to 0.442) | 0.422 to 0.428 | 0.428 | 0.413 |
| `SF-frozen-data` | 0.453 (0.419 to 0.483) | 0.429 (0.400 to 0.456) | 0.423 to 0.492 | 0.417 | 0.407 |

Paired differences (first minus second):

| Pair | Logistic | Forest |
|---|---|---|
| `spectral` and `spectra_bag` | +0.045 (-0.006 to +0.091) | +0.093 (+0.057 to +0.118) |
| `spectral` and `S1-frozen-data` | +0.006 (-0.043 to +0.066) | +0.070 (+0.025 to +0.124) |
| `spectral` and `S1-frozen-gauss` | +0.016 (-0.024 to +0.064) | +0.095 (+0.057 to +0.132) |
| `spectral` and `SF-frozen-data` | -0.013 (-0.073 to +0.065) | +0.047 (-0.014 to +0.130) |
| `S1-frozen-data` and `S1-frozen-gauss` | +0.010 (-0.023 to +0.038) | +0.025 (-0.003 to +0.045) |
| `S1-frozen-data` and `SF-frozen-data` | -0.019 (-0.063 to +0.028) | -0.023 (-0.061 to +0.021) |
| `spectra_bag` and `S1-frozen-data` | -0.038 (-0.075 to +0.013) | -0.023 (-0.062 to +0.025) |

- **Every arm is far above the prior** (0.167) and the null (0.187).
- **With the logistic head, a dictionary that was never trained reads as much as the spectral control.** The three random dictionaries read 0.42 to 0.45 and `spectral` reads 0.44. No pair among these four has an interval that excludes zero.
- **With the forest, the spectral control leads** the single-channel dictionaries and the bag of spectra by 0.07 to 0.10.
- **Rankings that hold on both heads:** none with an interval that excludes zero on both. `spectral` is above `spectra_bag` in sign on both.
- **A Gaussian dictionary reads as much as random windows of the data.** So the counts of a random dictionary do not need the shapes of the data.

### 17.3 The record effect (M3)

Balanced accuracy by readout, logistic head. The share is the gain of the `displaced` readout over the prior, divided by the gain of the `full` readout.

| Arm | `full` | `event` | `context` | `displaced` | `full` minus `displaced` | Share |
|---|---|---|---|---|---|---|
| `spectral` | 0.441 | 0.466 | 0.380 | 0.342 | +0.099 (+0.052 to +0.146) | 0.64 |
| `spectra_bag` | 0.396 | 0.375 | 0.364 | 0.316 | +0.080 (+0.045 to +0.113) | 0.65 |
| `S1-frozen-data` | 0.434 | 0.412 | 0.385 | 0.343 | +0.091 (+0.051 to +0.124) | 0.66 |
| `S1-frozen-gauss` | 0.424 | 0.382 | 0.384 | 0.330 | +0.094 (+0.068 to +0.120) | 0.63 |
| `SF-frozen-data` | 0.453 | 0.394 | 0.395 | 0.334 | +0.119 (+0.085 to +0.148) | 0.59 |

Recall by class, logistic head: the range over the five arms of the difference between two readouts.

| Class | `full` minus `displaced` | `full` minus `context` | `full` minus `event` |
|---|---|---|---|
| `spsw` | -0.01 to +0.08 | -0.04 to +0.03 | +0.00 to +0.14 |
| `gped` | +0.03 to +0.08 | -0.01 to +0.04 | +0.02 to +0.25 |
| `pled` | -0.08 to +0.02 | +0.00 to +0.03 | -0.03 to +0.13 |
| `eyem` | +0.13 to +0.38 | +0.03 to +0.22 | -0.16 to -0.04 |
| `artf` | +0.08 to +0.11 | +0.02 to +0.06 | -0.06 to +0.04 |
| `bckg` | +0.12 to +0.22 | +0.04 to +0.10 | -0.04 to +0.09 |

- **About two thirds of the score comes from the recording.** A context of the same recording with no mark reads 0.32 to 0.34. That is 59% to 66% of the gain over the prior.
- **`gped` and `pled`: the control reads them as well as the event context.** The difference is inside 0.10 in all five arms, so S5 holds for every arm of this step. For the dictionary arms the labeled second alone reads them 0.09 to 0.25 lower than the whole context.
- **`spsw`: no arm shows a gain of the event context over the control** with an interval that excludes zero.
- **`eyem` is tied to the event.** The labeled second alone reads it best, and the control loses 0.13 to 0.38.
- **By the rule of section 13,** these arms allow no waveform claim for `gped`, `pled` and `spsw`. The rule is read with the best arm after step 3.
- **What this does and does not show.** Periodic discharges can continue outside the marks, so a control far from every mark can still hold the waveform. The control does not show that these classes have no waveform. It shows that this design cannot separate a waveform at the marked second from other properties of the recording.

### 17.4 Preview of the resolution

M1 needs `S1-L2`. The paired half-widths of this step are 0.030 and 0.024 (logistic and forest) for the two single-channel random dictionaries, which is the pair nearest to that of M1, and 0.027 to 0.072 for the other pairs. The interval of one arm has a half-width of 0.03 to 0.075.

### 17.5 Observations outside the plan of the step (post hoc)

1. **The null is biased upward.** The 10 values are 0.150, 0.205, 0.172, 0.176, 0.183, 0.221, 0.156, 0.204, 0.193 and 0.212: mean 0.187, standard deviation 0.024, which is 2.7 standard errors above 0.167. My explanation, not tested: the weights give every patient of a class the same total, and a permuted rare label lies mostly on patients with many events. So the permuted label still carries the size of the patient, and true `spsw` events (recall 0.29 under the null) come from such patients. With patients weighted equally the null reads 0.175. The gate keeps its rule. The floor for a statement "above chance" with this head is about 0.19, not 0.167.
2. **The forest does not predict `spsw`.** Its recall of `spsw` is 0.00 to 0.04 on the `full` readout in every arm; the logistic head reads 0.10 to 0.37. The two heads give the same class for 57% to 63% of the examples.
3. **One patient moves the primary score.** One patient holds 702 of the 1,986 `gped` events. `spectral` reads `gped` at 0.43 by events and at 0.60 by patients. With patients weighted equally, `spectral` leads on both heads (0.468 and 0.512; the next arm reads 0.432 and 0.447), and `SF-frozen-data` falls from first to last with the logistic head. The primary score stays as the protocol fixed it; both forms are reported.
4. **Three seeds are too few for a full-template dictionary.** `SF-frozen-data` reads 0.446, 0.423 and 0.492 by seed (logistic). That range of 0.069 is twice its bootstrap half-width. The seeds of the forest add little: the standard deviation between forests on one dictionary is 0.002.
5. **The largest number of the run** is `spectral` with the forest on the `event` readout, 0.532. It is not the primary score.

### 17.6 State of the predictions

| | State after step 2 |
|---|---|
| S5 | Holds for every arm of this step (largest difference 0.08). The final read uses the best arm after step 3 |
| S4 | Open. On the primary score an untrained dictionary is nominally above `spectral` (-0.013, interval -0.073 to +0.065) |
| S1, S2, S3, S6, S7 | Need step 3 |

### 17.7 Cost of the step (M10)

45.3 minutes for 5 folds and the repeat of fold 0. Per fold: 309 s for 156 forests of 500 trees, 88 s for 3 k-means fits with their tokens, 21 s for the 10 null fits, 19 s for 52 logistic fits, 14 s for the tokens of 9 dictionaries on 15,926 contexts. The features that do not depend on a fold take 8 s. Peak GPU memory 0.44 GB.

## 18. Results of step 3 (run of 2026-11-27 and 2026-11-28)

Root `artifacts/slice-step3-001`, commit `ca8b79c`, 3.56 hours. The run holds the arms of step 2 and the 21 arms of step 3: 48 feature sets per fold. All scores are balanced accuracies of pooled out-of-fold predictions on the 274 training patients, on the `full` readout unless stated. The evaluation set was not read. Intervals come from 2,000 bootstrap draws over patients. The rules are those of section 16.3, fixed before the run.

### 18.1 Gates and health of the run

| Gate | Result |
|---|---|
| 2, no leak | Passes: 339 training recordings read, no evaluation recording |
| 3, null | Passes by its rule: 0.187 (the bias of section 17.5 stays) |
| 5, counts | Passes: 92 pairs of arm and readout equal the token numbers of the protocol |
| 6, determinism | Passes: fold 0 repeats bit by bit in all 28 arms (1,024,385 predictions) |

- **The arms of step 2 repeat across runs.** Their 1,743,897 predictions in this run equal those of the run of step 2 bit by bit.
- **All 70 alternating fits converged,** in 24 to 91 passes. No atom was without a unit, and every unit was assigned.
- Gate 1 is not confirmed for the commits of step 3: I have no test output.

### 18.2 The resolution and the predictions

**M1.** `S1-L2` minus `S1-frozen-data` is +0.024 (-0.007 to +0.058), so `h` is 0.033. The margin of a useful gain is 0.033. By the rule of section 13 a ranking is possible (`h` is not above 0.05).

| | Prediction | My probability | Result |
|---|---|---|---|
| S1 | `h` is from 0.02 to 0.05 | 0.60 | Holds: 0.033 |
| S2 | `S1-L2` is above `S1-frozen-data` by more than `h` | 0.35 | Does not hold: +0.024 |
| S3 | `spectra_bag` is within `h` of `S1-L2`, or above it | 0.60 | Does not hold: -0.062 |
| S4 | `spectral` has the highest primary score | 0.55 | Does not hold: `S1-L1` reads 0.460 and `spectral` 0.441. With the forest `spectral` is the highest |
| S5 | `gped` and `pled`: `displaced` within 0.10 of `full` for the best arm | 0.65 | Holds for `S1-L1`: +0.083 and -0.110 |
| S6 | Cosine tokens are above raw tokens by more than `h` for `S1-L2` | 0.40 | Does not hold: +0.002 |
| S7 | The input level changes `spectral` by less than `h` | 0.60 | Holds: -0.017 |

Five of the seven results are on the side that I gave more than half. S3 and S4 are not.

### 18.3 The seven arms at their primary setting

| Arm | Logistic (primary) | Forest | By patients, logistic | By patients, forest |
|---|---|---|---|---|
| `S1-L1` | 0.460 (0.413 to 0.515) | 0.416 (0.367 to 0.476) | 0.446 | 0.447 |
| `S1-L2` | 0.458 (0.411 to 0.513) | 0.425 (0.374 to 0.486) | 0.465 | 0.452 |
| `SF-frozen-data` | 0.453 (0.419 to 0.483) | 0.429 (0.400 to 0.456) | 0.417 | 0.407 |
| `spectral` | 0.441 (0.386 to 0.509) | 0.476 (0.410 to 0.561) | 0.468 | 0.512 |
| `S1-frozen-data` | 0.434 (0.388 to 0.477) | 0.406 (0.361 to 0.453) | 0.425 | 0.428 |
| `S1-frozen-gauss` | 0.424 (0.378 to 0.477) | 0.381 (0.334 to 0.442) | 0.428 | 0.413 |
| `spectra_bag` | 0.396 (0.339 to 0.471) | 0.383 (0.322 to 0.469) | 0.432 | 0.447 |

Paired differences (first minus second):

| Pair | Logistic | Forest |
|---|---|---|
| `spectral` and `S1-L2` | -0.018 (-0.062 to +0.032) | +0.051 (+0.011 to +0.094) |
| `S1-L1` and `S1-L2` | +0.002 (-0.011 to +0.014) | -0.009 (-0.015 to -0.002) |
| `S1-L2` and `spectra_bag` | +0.062 (+0.017 to +0.101) | +0.042 (+0.004 to +0.072) |
| `S1-L2` and `SF-frozen-data` | +0.005 (-0.043 to +0.063) | -0.004 (-0.046 to +0.052) |

- **The five best arms lie inside 0.026 with the logistic head,** which is less than the margin. The order of `spectral` and the learned arms changes with the head, so by the rule of section 8 there is no ranking between them.
- **One ranking holds on both heads and is above the margin on both:** `S1-L2` is above the bag of spectra.
- **The two update rules give the same score.** The open question of which rule made the results of [1] does not matter at this resolution.
- **With patients weighted equally,** `spectral` and `S1-L2` are equal with the logistic head (0.468 and 0.465), and `spectral` leads with the forest (0.512 against 0.452).

### 18.4 Does learning add to a random dictionary? (M4)

| Pair | Logistic | Forest |
|---|---|---|
| `S1-L1` minus `S1-frozen-data` | +0.026 (-0.004 to +0.061) | +0.010 (-0.008 to +0.035) |
| `S1-L2` minus `S1-frozen-data` | +0.024 (-0.007 to +0.058) | +0.019 (+0.000 to +0.044) |
| `S1-L1` minus `S1-frozen-gauss` | +0.036 (+0.008 to +0.065) | +0.035 (+0.018 to +0.049) |
| `S1-L2` minus `S1-frozen-gauss` | +0.034 (+0.004 to +0.066) | +0.044 (+0.024 to +0.060) |

- **Against its own start, learning adds about 0.02.** The sign is the same for both rules, both heads and all three seeds (12 of 12), but the gain is below the margin, and its interval holds zero with the logistic head. By the pre-stated rule this is not a useful gain.
- **Against a Gaussian dictionary, learning adds 0.034 to 0.044** on both heads, with intervals that exclude zero. This is at the margin.
- **Post hoc: the two heads place the gain differently.** With the logistic head the gain of `S1-L2` is as large in the control context with no event (+0.028, +0.011 to +0.047) as in the event context (+0.024). With the forest the gain is in the labeled second (+0.040, +0.025 to +0.058) and not in the control (+0.006). So the slice cannot say that the gain of learning comes from the event.

### 18.5 The levels (M5, M6, M7, M8, M11) and the defaults of stage 4

A level against the primary value, for `S1-L2`:

| Level | Logistic | Forest | Replaces the primary value |
|---|---|---|---|
| 16 atoms | +0.011 (-0.028 to +0.044) | -0.018 (-0.052 to +0.007) | No |
| 64 atoms | +0.001 (-0.029 to +0.029) | +0.002 (-0.015 to +0.017) | No |
| Raw tokens | -0.002 (-0.023 to +0.018) | +0.013 (-0.004 to +0.028) | No |
| Windows of 250 positions | -0.030 (-0.052 to -0.008) | -0.014 (-0.037 to +0.004) | No |
| `profile` input (raw tokens) | +0.030 (+0.002 to +0.050) | -0.002 (-0.011 to +0.007) | No |
| 10 minutes per patient | -0.019 (-0.033 to -0.003) | -0.003 (-0.012 to +0.007) | No |

- **No level replaces a primary value.** Stage 4 keeps 32 atoms, windows of 125 positions, the scale per channel and 5 minutes per patient.
- **Cosine and raw tokens are equal** on all five dictionaries (differences from -0.017 to +0.011). The large effect of the token score in the synthetic benchmark does not appear on the corpus.
- **The wider window is worse** for all four arms (-0.014 to -0.048).
- **The input level changes nothing that the rule counts:** `spectral` -0.017 and -0.002, `SF-frozen-data` +0.020 and +0.014.
- **More exposure does not help.**

### 18.6 Cross-fitting (M9)

The head on the patients of the dictionary minus the head on the other half: +0.048 (-0.006 to +0.091) with the logistic head and +0.033 (-0.030 to +0.078) with the forest. If features in sample misled a head, the first head would be worse on new patients. It is not. So the slice shows no need for cross-fitting with the whole-recording source. A limit of the design: the two heads train on different patients, and the two halves differ by chance, so this check cannot measure a small bias.

### 18.7 The record effect for the best arm

`S1-L1`, logistic head: the `full` recall minus the recall of another readout.

| Class | Minus `displaced` | Minus `context` | Waveform claim |
|---|---|---|---|
| `spsw` | +0.025 (-0.091 to +0.178) | +0.022 (-0.036 to +0.075) | None |
| `gped` | +0.083 (+0.005 to +0.127) | +0.006 (-0.011 to +0.030) | None |
| `pled` | -0.110 (-0.184 to -0.028) | +0.005 (-0.011 to +0.019) | None |
| `eyem` | +0.304 (+0.160 to +0.419) | +0.191 (+0.125 to +0.244) | Possible |
| `artf` | +0.073 (+0.019 to +0.135) | +0.045 (+0.023 to +0.065) | Possible |
| `bckg` | +0.173 (+0.130 to +0.215) | +0.068 (+0.055 to +0.082) | Possible |

- **The three epileptiform classes get no waveform claim** in the later stages, by the rule of section 13.
- **The control reads `pled` better than the event context** (0.57 against 0.47).
- **The learned arms take more of their score from the recording than the frozen arms:** their control reads 0.37, which is 70% of their gain over the prior.
- The caution of section 17.3 stays: this does not show that these classes have no waveform.
- **Correction of 2026-11-28 (section 19.3): the `context` readout shows no record effect for a class that comes in runs.** For `gped` and `pled`, 99% of the events have other marked seconds of their own class inside the windows of the `context` readout (for `spsw` 42%). So `full` minus `context` is near zero for such a class whatever the arm reads. The rule of section 13 gives `gped` no waveform claim through the `context` readout only; by the `displaced` readout `gped` is outside the margin (+0.083). The rule stands as it was fixed. Its result for `gped` rests on a readout that I designed badly.

### 18.8 Cost (M10)

213.5 minutes for 5 folds and the repeat of fold 0, about 36 minutes per fold. A fold: 12 to 14 minutes for 14 fits, 18.5 minutes for 576 forests, 1.5 minutes for k-means, 1 to 2 minutes for the logistic fits and 1 minute for the tokens. A fit at 32 atoms takes 49 s (26 to 49 passes of 1.5 s); at 64 atoms 76 s; on the plan of 10 minutes 97 s. Peak GPU memory 13.7 GB.

### 18.9 What the slice sets, and my recommendation for stage 4

By the rules of section 13:

- **Margin:** 0.033.
- **Record effect:** no waveform claim for `gped`, `pled` and `spsw`.
- **Defaults:** all primary values stay.
- **Resolution:** a ranking is possible in the sense of the rule.

The rule of the resolution passes, but the effects that stage 4 would rank are smaller than the margin. Learning adds 0.02 to its own start. The two update rules differ by 0.002, the two token scores by 0.002, and the numbers of atoms by 0.02 or less. The order of the best arms changes with the head and with the weight of a patient, and one patient holds a third of the `gped` events. A full grid of nine cells on this target would most likely give "undecided" for almost every contrast.

My recommendation, which waits for the decision of the user:

1. **Do not build the full grid of stage 4 on this target.**
2. **First run a cheap test that removes the recording:** for every event, tell its context from its displaced control of the same recording, by class, with folds by patient. Chance is 0.5, and the patient cannot help. If no arm reads `gped`, `pled` or `spsw` events against their own controls, the marks of this corpus cannot support a comparison of waveform methods for these classes. This test is a proposal and is not in the protocol.
3. **If a comparison of the three methods is still wanted,** run it as a described comparison with a small grid: each method at the defaults with its start as the null, more seeds for full-template dictionaries (their seeds spread by 0.07), both heads, and one read of the evaluation set with at most three contrasts.

## 19. The recording test (decided by the user on 2026-11-28)

The user chose this test before any build of stage 4. Its design and rules are fixed here, before it runs.

### 19.1 Question

About two thirds of the six-class score come from the recording. Can an arm tell a marked event of a class from a stretch of the same recording that is far from every mark? If no arm can, the marks of this corpus cannot support a comparison of waveform methods for that class.

### 19.2 Design

- **Pairs.** Every kept event has one displaced control: a context of the same recording, 10 s or more from every mark. The median distance between an event and its control is 4 to 7 minutes. The 7,963 pairs of the slice are used, with its 5 folds by patient.
- **Task.** For one class at a time, a head learns to tell the event contexts of that class from their controls, on the training patients. It then gives a score to both contexts of every event of the validation patients.
- **Measure.** The share of pairs in which the event has the higher score; a tie counts as a half. It is taken per patient and then averaged over the patients that hold the class, so every patient counts equally. Chance is 0.5. A property of a patient or a recording cannot move it, because both contexts of a pair come from one recording.
- **Intervals.** A bootstrap over the patients that hold the class, 2,000 draws. A difference of two arms or two readouts is paired by patient.
- **Arms.** The seven arms of the slice at their primary setting, with the same dictionaries: `spectral`, `spectra_bag`, `S1-L1`, `S1-L2`, `S1-frozen-data`, `S1-frozen-gauss`, `SF-frozen-data`; 3 seeds for an arm with a random part.
- **Readouts.** `full` (all windows of a context), `event` (the windows of the labeled second, and the same windows of the control) and `context` (the outer windows of both).
- **Heads.** The two heads of the slice, with one forest seed. Every patient has the same total weight inside each of the two labels.

### 19.3 A fact of the corpus, found before the design was fixed

I counted the manifest tables before I fixed the rules. Events of one class come in runs. The table counts, for an event, the other kept events of its class whose labeled second lies inside its context, and inside the windows of the `context` readout.

| Class | Events with such a second in their context | Mean number | Events with such a second in the `context` windows | Mean number |
|---|---|---|---|---|
| `gped` | 99.3% | 10.7 | 98.6% | 9.8 |
| `pled` | 99.4% | 10.5 | 98.7% | 9.5 |
| `bckg` | 99.7% | 4.1 | 99.7% | 4.0 |
| `artf` | 96.7% | 4.9 | 95.1% | 4.5 |
| `eyem` | 92.5% | 4.3 | 64.2% | 3.1 |
| `spsw` | 89.5% | 2.2 | 41.9% | 1.3 |

Two consequences:

1. **The `context` readout of the slice was a bad control for the record effect.** Its windows hold no sample of the labeled second, but for `gped` and `pled` they hold other marked seconds of the class in 99% of the events, and for `spsw` in 42%. I did not check this when I wrote the rule of section 13. The correction is in section 18.7.
2. **The rule of this test uses the `full` readout.** The question is whether a marked stretch differs from an unmarked one, and a stretch of a run class holds many marks.

### 19.4 Rules

| | Rule |
|---|---|
| Readable class | For `spectral` or for `S1-L2` (the two arms that decide), the measure on the `full` readout is 0.60 or more and its interval lies above 0.5, with both heads |
| Outcome | `event_signal` if `spsw`, `gped` or `pled` is readable; else `no_event_signal` |
| Negative control | The events of `bckg` are marked background. If `bckg` is readable, marked and unmarked stretches differ with no event, and the outcome carries the label `region_confound` |
| Reported, with no rule | `S1-L2` minus `spectral`, `S1-L2` minus `S1-frozen-data`, and `event` minus `context`, by class and head |

What each outcome sets:

- **`no_event_signal`:** no comparison of waveform methods runs on the epileptiform classes of this corpus. The choice is then to stop or to take another corpus.
- **`event_signal`:** the recording test becomes the target of a reduced stage 4 for the readable classes, and the methods are compared on it.
- **`region_confound`:** a readable class is read with caution, because part of its value can come from the stretch and not from the event.

### 19.5 Predictions

Fixed before the run.

| | Prediction | My probability |
|---|---|---|
| T1 | `eyem` is readable by `spectral` | 0.90 |
| T2 | `gped` is readable | 0.50 |
| T3 | `pled` is readable | 0.45 |
| T4 | `spsw` is readable | 0.45 |
| T5 | The outcome is `event_signal` | 0.70 |
| T6 | `bckg` is readable (`region_confound`) | 0.40 |
| T7 | For every readable epileptiform class, `S1-L2` minus `spectral` is below 0.03 with at least one head | 0.65 |

### 19.6 Gates

1. **Tests.** The suite passes on the HPC (275 tests).
2. **Null.** `spectral` with the two labels of every training pair in a random order reads within 0.05 of 0.5 for every class (mean of 10 orders).
3. **Determinism.** Fold 0 is computed twice and gives equal scores.

The folds are those of the slice, so no patient is in both parts of a fold, and the reader refuses the evaluation set.

### 19.7 Limits

- **Unmarked events.** A control is far from every mark, but the marks cover 1.8% of the time, and a stretch without a mark can hold the same events. A low value then shows that marked and unmarked stretches do not differ. It does not show that the class has no waveform.
- **The stretch, not the second.** A high value shows that a marked stretch differs from an unmarked one. The `event` readout and the negative control limit this, and do not remove it.
- **Few patients.** `pled` has 23 patients, `spsw` 24 and `gped` 25, so the intervals are wide.
- **Fixed threshold.** The limit of 0.60 is my choice, made before the run.

### 19.8 Build and commands

Built on 2026-11-28; the static checks pass; 5 tests have not run.

| File | Content |
|---|---|
| `slice/detect.py` | The run of the test, its reader (NumPy and Polars only) and the rules as a function of the measures |
| `scripts/run_slice.py` | `detect` runs the test into a new directory; `read` prints its report |
| `tests/test_slice_detect.py` | 5 tests |

My estimate of the cost is 1.2 to 1.5 hours (36 fits, 18 k-means fits and about 4,500 small heads, with the repeat of fold 0); the smoke run takes some minutes.

```bash
python -m pytest -q
```

```bash
PYTHONPATH="$PWD/src" python scripts/run_slice.py detect --smoke --root "$DATA_DIR/v2.0.1/edf" --manifest artifacts/tuev-manifest-clip50-001 --plans artifacts/tuev-exposure-001 --scales artifacts/tuev-scale-audit-001.json --cache-dir cache/tuev --output artifacts/slice-detect-smoke
```

```bash
nohup env PYTHONPATH="$PWD/src" python -u scripts/run_slice.py detect --root "$DATA_DIR/v2.0.1/edf" --manifest artifacts/tuev-manifest-clip50-001 --plans artifacts/tuev-exposure-001 --scales artifacts/tuev-scale-audit-001.json --cache-dir cache/tuev --output artifacts/slice-detect-001 >> artifacts/slice-detect-001.log 2>&1 &
```

## References

1. Cano Achuri et al. (2026), reference [1] of the comparison plan.

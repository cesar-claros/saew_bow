# Synthetic Benchmark Protocol (stage 2)

Confirmed by the user on 2026-11-24, with one decision: every test and every run is on the HPC (section 9). Corrections of fact that the build found after the confirmation are in section 11. It makes stage 2 of the [Comparison Plan](Comparison%20Plan.md) exact: the generator, the nine cells, the measures, and the predictions with their decision rules. The synthetic benchmark is the only place of the study where waveform identification has a known answer. It uses no corpus.

## 1. What the stage decides

The stage answers two questions.

1. **Are the cells correct?** Six gates (section 7) check the implementation and the signal level before any prediction is read. A cell that fails a gate is repaired first.
2. **Do the mathematical distinctions of the plan appear on ideal data?** Eight predictions (section 6) each compare two cells that differ in one choice. Each prediction ends as `holds`, `opposite` or `undecided` by a fixed rule.

The stage decides nothing about TUEV. Its result sets priorities for stage 3 and stage 4: a factor whose prediction holds is carried to the corpus first, and a factor that shows no effect on ideal data needs a reason to stay in the corpus design.

Limit of the stage: I chose the generator. A generator can show any effect that its author builds in. So each stress condition names the mechanism that it tests, the oracle and the fixed-point gate separate the rule from the search, and the result is read as "the mechanism exists", not as "the mechanism matters on EEG".

## 2. Geometry

All cells share one geometry.

| Quantity | Value |
|---|---|
| Sampling rate | 250 Hz |
| Segment | 1,250 samples (5 s), 22 channels |
| Atom length L | 125 samples (0.5 s) |
| Pooling window P | 125 positions |
| Support of a window | P + L - 1 = 249 samples |
| Valid placements | 1,126 (edge mask on). A pooling window holds the placements whose center is in it, as in the retained models. So 8 windows are full, and the first and the last window hold 63 placements each, as on the corpus. |
| Atoms per window R | 1 |
| Planted templates G | 4 |
| Atoms K | 8 |
| Spatial patterns D (rank-1 cells) | 2 |

## 3. Generator

- **Background.** Each channel is noise with a 1/f spectrum between 1 and 30 Hz and no power outside. A common source gives the channels a correlation of 0.3. Each channel has unit variance.
- **Templates.** Four waveforms of at most 88 samples (0.35 s), filtered to 1 to 30 Hz, with zero mean and unit norm: a spike with a slow wave, a damped 10 Hz oscillation, a biphasic sharp wave of 200 ms, and a 7 Hz burst of two cycles. A template is shorter than an atom, so an atom can hold it at a shift without a cut. The waveforms are the same in all runs. Each template has a spatial pattern with unit norm and positive weights on 6 of the 22 channels, drawn with the seed. Four more waveforms of the same families with other parameters are the decoys. They are never planted.
- **Signal level.** An event of level `s` gives its planted C x L template, used as a matched filter, a response of `s` standard deviations of that filter on the background. The scale of each template comes from background draws with a calibration seed that no run uses. The base level is 8. For a pattern with equal weights this is about 5 standard deviations for the matched filter of one channel. A level in white-noise units would be wrong here: a smooth waveform has a large response to a 1/f background.
- **Events.** A segment holds 4 events at random starts that are at least 125 samples apart, each fully inside the segment. An event adds `amplitude * pattern * waveform`.
- **Sets.** 200 training segments and 100 test segments per run, from independent draws. Five seeds per condition. A seed fixes the patterns, the data and the start of the fits.

Conditions, each with the mechanism that it tests:

| Condition | Change against `base` | Mechanism | Levels |
|---|---|---|---|
| `base` | None | Reference | 1 |
| `amplitude_artifacts` | The event level is log-normal (median 8, log standard deviation 0.5). Each segment also holds two artifacts on 3 random channels each. An artifact comes from three families with random parameters: a step with an exponential decay (time constant 0.05 to 0.3 s), a half-sine of 0.2 to 0.5 s, and a noise burst of 0.3 s. Its sign is random, and its peak is log-normal with a median of 10 background standard deviations. | An update that weights a window by its energy gives atoms to rare events of high energy. | 1 |
| `burst_near_event` | Each event has a noise burst on its channels in the same pooling window, at least 62 samples away. The standard deviation of the burst is set once, with the calibration seed, so that the median largest raw response of the planted waveform to the burst is 1.5 times its response to the event. The value is recorded. | The raw score places an atom on energy. The cosine score places it on shape. | 1 |
| `delay` | The waveform reaches the channels of its pattern with a delay that grows linearly to a maximum. | A rank-1 atom has one waveform for all channels. | Maximum delay 4, 8, 16 samples |
| `few_examples` | The number of training segments changes. | A rank-1 atom has fewer parameters. | 10, 25, 50 segments |
| `overlap` | Events come in pairs of two templates in one pooling window, with starts 25 to 50 samples apart. | R=1 codes one event per window. | 1 |
| `polarity` | Each event has a random sign. K equals G. | A positive response does not see an inverted event. | 1 |
| `background` | No events. | Null of the recovery measure. | 1 |
| `more_atoms` | K is 16. | Dependence of the result on K (descriptive). | 1 |

These are 13 levels. Artifacts with one random family, and not noise alone, are necessary for the first mechanism: noise windows of high energy do not add up in an atom, and a stereotyped artifact takes only one spare atom.

A lesson from the earlier harness for the mouse pipeline enters here: in a 1/f background a short smooth transient had a null floor of 0.925 for its recovery measure. So every template has oscillatory content, and every recovery value is read against a floor from `background` (section 5).

## 4. The nine cells

A cell is an atom structure with a learning rule.

**Structures.** `S1`: one set of 1 x L atoms; the training unit is one channel of one pooling window. `SF`: full C x L atoms with unit Frobenius norm; the training unit is one pooling window of all channels. `SR`: rank-1 atoms with D spatial patterns.

**Start.** A run draws K * D locations (segment and first sample) with its seed. `SF` starts from the C x L patch at each of the first K locations. `S1` starts from one random channel of that patch, as the random start of the author code of [1]. `SR` starts as the retained separable model does: the waveform of atom k is the top singular vector of patch k, and each pattern comes from one of the K * D patches. Every fit has one start. Every cell is also evaluated frozen at its start.

**Alternating rules** (`L1` to `L4`). One pass assigns every training window, then updates every atom. Let `y` be the sub-window of an assigned window at its assigned position and `a` its atom.

| Rule | Assignment of a window | Weight of `y / norm(y)` in the update |
|---|---|---|
| `L1` | Largest cosine over atoms and valid positions | 1 |
| `L2` | Largest cosine | `norm(y)` |
| `L3` | Largest cosine | `norm(y)^2 * cosine(y, a)` |
| `L4` | Largest raw inner product | `norm(y)^2 * cosine(y, a)` |

- A window with no positive response gets no atom.
- The new atom is the weighted sum, scaled to unit norm. An atom without a window keeps its value.
- The weight of `L3` and `L4` is one power step toward the least-squares atom: the new atom is proportional to the sum of `inner(y, a) * y`.
- The passes stop when the mean over atoms of the squared change of an atom is below 1e-4, or after 100 passes. The number of fits that reach 100 passes is reported.
- The update function of the author code of [1] has both weights: raw sub-windows (`do_sphere=False`) are `L2`, and normalized sub-windows (`do_sphere=True`) are `L1`. The parity gate checks both. Section 11 records which of them the pipeline of the checkout calls.
- The author code differs from these cells in four points that belong to the published-recipe arm: the iterate with the lowest mean cosine distance is kept, a shift adjustment follows the first assignment, the stop is on the change of centroids that are not scaled to unit norm (below 1e-4 times the variance of the windows, per sample), and an empty cluster gets a zero centroid. The rule of [2] also makes a z-score of each channel in each window. None of these is part of `L1` or `L2`, so that neighbors on the path differ in one choice.

**Gradient rule** (`L5`). Adam on the mean squared error of the extended windows, with the comparison adapters of the retained models and their defaults: learning rate 1e-3, 2 segments per step (44 single-channel rows for `S1`), step-matched factors, edge mask on, no internal scaling, and no re-centering of atoms (the separable reference has none, so the full-template cells have none). For R=1 this loss equals the window energy minus the squared best response, which is the loss that `L4` minimizes by alternation.

- The budget counts steps, not passes. It starts at 3,000 steps (30 passes at 200 segments). With fewer segments the passes increase so that the steps stay equal.
- The budget is set by convergence of the training loss and not by recovery: on `base` with seed 0, the steps double (to at most 24,000) until the loss changes by less than 1% over the last tenth of the steps. The largest value over the four gradient cells is then the budget of all gradient cells in all conditions.

| Cell | Implementation |
|---|---|
| `S1-L1` to `S1-L4` | New alternating learner on single-channel windows |
| `S1-L5` | Full-template adapter on single-channel rows, with the extended loss (new option) |
| `SF-L1` | New alternating learner on multichannel windows |
| `SF-L5` | Full-template adapter with the extended loss (new option) |
| `SR-L5` | Separable adapter |
| `SR-L5n` | Separable adapter with the window-normalized loss (new option): the loss of each window is divided by the energy of its target |

**Encoder for the measures.** Each cell encodes with its own assignment rule, one token per pooling window. `S1` uses its best-channel token, so that all cells have the same number of tokens.

**Oracle.** The oracle of a cell is the planted dictionary in the structure of the cell, encoded with the assignment rule of the cell. It has no learning. It gives the ceiling of the event detection in every condition, and it separates the effect of a score at inference from its effect on learning.

## 5. Measures

- **Waveform recovery** of a planted template: the largest cross-correlation, over atoms and shifts of at most 62 samples, between the planted waveform and the waveform of the atom, both with unit norm. For `S1` the waveform of an atom is the atom. For `SF` and `SR` it is the atom projected on the planted spatial pattern, scaled to unit norm. The value is signed. The waveform recovery of a run is the mean over the four templates.
- **Template recovery** (`SF` and `SR` only): the same with the full C x L template.
- **Floor** of a template for a cell: its mean recovery by the learned dictionary of that cell in `background`, where no template is planted. This floor is conservative for the levels with fewer atoms or fewer segments.
- **Recovered**: a template with a recovery of at least 0.80 that is at least 0.20 above its floor.
- **Decoys**: the recovery of the four decoys is reported in every condition. It shows whether atoms that hold a planted waveform also match another waveform of its family.
- **Event detection** on the test segments. An atom maps to the template with its largest waveform recovery, if that value is at least 0.50. The time of a token is its placement plus the shift of that recovery. An event is detected when a token of an atom that maps to its template lies within 12 samples of its start. The measures are recall, precision and F1.

Every measure is reported for the learned dictionary, the frozen start and the oracle, as mean and standard deviation over the five seeds.

## 6. Predictions

The predictions are fixed here, before any code of the benchmark exists. Rule for a prediction with a margin `m`: `holds` when the mean of the paired differences over the five seeds is at least `m` and at least four seeds have the predicted sign; `opposite` when the same is true for the other sign; else `undecided`. An equivalence prediction `holds` when the absolute mean difference is below its margin. All eight results are reported.

| | Factor | Condition | Prediction | Margin | My probability |
|---|---|---|---|---|---|
| P1 | Window weight, as the plan states it | `amplitude_artifacts` | Waveform recovery of `S1-L1` is above `S1-L3` | 0.05 | 0.55 |
| P1s | Window weight, sensitivity | `base` and `amplitude_artifacts` | The loss of waveform recovery from `base` to `amplitude_artifacts` is larger for `S1-L3` than for `S1-L1` | 0.05 | 0.70 |
| P2 | Position score | `burst_near_event` | Recall of `S1-L3` is above `S1-L4` | 0.10 | 0.70 |
| P3 | Atom structure, delay | `delay` | At 16 samples, template recovery of `SF-L5` is above `SR-L5`. In `base` the two are equal. | 0.05 | 0.85 |
| P4 | Atom structure, sample size | `few_examples` | At 10 segments, template recovery of `SR-L5` is above `SF-L5`. In `base` the two are equal. | 0.05 | 0.60 |
| P5 | Optimizer | `base` | Waveform recovery of `S1-L4` and `S1-L5` are equal | 0.05 | 0.60 |
| P6 | Window weight, gradient | `base` and `amplitude_artifacts` | The loss of template recovery from `base` to `amplitude_artifacts` is smaller for `SR-L5n` than for `SR-L5` | 0.05 | 0.60 |
| P7 | Null | `background` | No learned cell reaches a recovery of 0.80 for a template | None | 0.85 |

Two notes on these values.

- P1 has a low probability for a reason that I state before the run. With equal weights, an atom follows the share of windows. In `S1` about 9 of 10 windows hold only background, so `L1` can lose templates to the background while `L3` loses them to artifacts. The plan states the absolute form, so it stays as P1. The sensitivity P1s is the test of the mechanism.
- The plan stated the null as an equal downstream score of learned and frozen dictionaries. A condition without events has no downstream task, so P7 tests the recovery measure in its place.

Descriptive, with no rule: `L2` against `L1` and `L3` in `amplitude_artifacts`; the absolute form of P6; the recall of the oracle under both scores in `burst_near_event`; the recall of the event pairs in `overlap`; the recall of the inverted events in `polarity`, also for a rank-1 cell with D=1; all cells in `more_atoms`; the position error of the detected events; the passes, the steps and the time of every fit.

## 7. Gates

The gates come before the predictions. A failed gate stops the read of every prediction that uses the cell.

1. **Tests.** The test suite passes on the HPC with the new modules.
2. **Parity with the author code of [1].** On single-channel windows of equal length and with equal atoms, the assignment of `L2` gives the labels and shifts of the author function `si_vq` with the cosine metric for at least 99.9% of the windows. One update of `L2` equals `_centroids_update_step` with `do_sphere=False`, and one update of `L1` equals it with `do_sphere=True`, both after a scale to unit norm and within 1e-5. The check loads the author functions from a checkout at run time and records its commit. The author repository has no license file, so no code of it enters this repository.
3. **Loss identity.** For R=1 the loss of every gradient cell equals the window energy minus the squared best response, within rounding.
4. **Signal level.** In `base`, the oracle of every cell has a recall of at least 0.90. If one oracle is below, the level rises from 8 to 12, then to 16, for all cells.
5. **Fixed point.** In `base`, each learned cell starts once at the planted dictionary, with random patches for the other atoms, and runs its full budget. It must keep at least 3 of the 4 templates recovered in at least 4 of the 5 seeds. A failure is examined before any other read. If the implementation is correct, the failure is a result: the planted waveforms are not a fixed point of that rule. It is recorded, and the cell continues.
6. **Determinism.** Two runs with one seed give equal results.

One rule for the range: if the mean number of recovered templates in `base` from the common start, over all learned cells, is below 2 of 4, the level rises one step for all cells, once, before any other condition is read. Recovery from the common start is a result and not a gate.

## 8. Order and cost

1. The cells, their tests, the parity check and the loss identity (gates 1 to 3).
2. The generator, the measures and the oracle, then `base` and `background` (gates 4 to 6, the budget of the gradient cells, P5 and P7).
3. The other conditions (P1 to P4 and P6, and the descriptive reads).

The full run is 13 levels, 9 cells and 5 seeds, with the fixed-point fits: about 640 small fits. I have no measured time. The first step gives it, and I report it before step 3 starts.

## 9. Where the code runs

The benchmark runs on the HPC, as every experiment of this project. The draft proposed unit tests of the new modules on the local machine, in a temporary environment with a CPU build of Torch. The user decided against it on 2026-11-24: every test and every run is on the HPC. The local machine does static checks only (format, lint, types, syntax).

## 10. What can change the design

- A factor whose prediction holds becomes a main arm on the corpus. If P1s or P6 holds, the window-normalized loss is one.
- If P3 and P4 both fail, the structure contrast has no support on ideal data, and the plan must say why the corpus would show one.
- If P5 fails, the alternating and the gradient rule do not reach the same atoms at a converged budget, and the optimizer is a real factor of the comparison.
- The signal level, the rates and the first budget of this protocol are choices without data. Section 7 fixes the only changes that the first results can cause: the level (8, 12, 16) and the budget of the gradient cells. Any other change is an amendment with its reason, written before the affected condition runs.

## 11. Corrections after the confirmation

These are corrections of fact from the build. No prediction, margin, gate or rule changed. Each is written before any run.

1. **Pooling windows (2026-11-24).** The draft said that 9 pooling windows are full and a tenth holds one placement. The retained models index a placement by the center of the atom, so the first and the last window hold 63 placements each and 8 windows are full. The alternating cells use the same windows. Section 2 is corrected.
2. **Code path of the author checkout (2026-11-24).** The draft said that the pipeline script of the author code uses raw sub-windows. This was wrong. The pipeline function `create_codebooks` (HEAD b7716e1 of the local checkout) always calls the clustering with `do_sphere=True`, one start, and a tolerance of 1e-4 times the variance of the windows. So the code path of the checkout is the update of `L1`, not of `L2`. The `--sphere` flag of the scripts is another step: it scales each whole window to unit norm before the clustering. The plan names `L2` as the rule of [1] from the text of the paper. Which update made the published results is a question for the authors. The nine cells do not change, because the path holds both rules. Only the label "published point" of `L1` and `L2` depends on the answer.

3. **Details that the draft left open (2026-11-24, before any run of the benchmark).** They are fixed here as the code has them.
   - A segment holds one event of each template, in random order. The draft said only that a segment holds 4 events.
   - The oracle of the rank-1 cells has one spatial pattern per atom (the planted one). The start of the fixed-point gate for the rank-1 cells keeps the second pattern of the common start.
   - Gate 6 reads "equal" as: every measure of a repeated run differs by at most 1e-6. A GPU can add numbers in another order.
   - The budget rule compares the mean loss of the last tenth of the steps with the mean of the tenth before it. One fit with the largest budget gives every checkpoint, because a shorter fit with the same seed takes the same first steps.
   - A cell has no threshold. Every pooling window with a positive response gives a token, so the precision is bounded by the share of windows that hold an event. Recall is the measure of the predictions; precision and F1 are reported.
   - Conditions with one seed share their patterns, their backgrounds and their event layout. A run with fewer training segments holds the first segments of the larger run.

## 12. Build record

- **Step 1, built 2026-11-24, not yet run.** `src/eeg_compare/cells/` holds the nine cells: `alternating.py` (rules `L1` to `L4` for `S1` and `SF`), `gradient.py` (rule `L5` on the adapters, with the extended loss and the window-normalized loss as new options of the adapters), `api.py` (common start, fit, frozen dictionary, tokens) and `common.py`. The loss identity of gate 3 is a test (`tests/test_cells_gradient.py`). Gate 2 is `scripts/check_author_parity.py`, which needs a checkout of the author repository. The suite has 50 new tests in four files. Local checks: format, lint, types and syntax only.
- **First run on the HPC, 2026-11-24 (commit d923fd4).** 182 of 183 tests passed in 10 s. The one failure was a defect of a test, not of a cell: the stand-in file that `tests/test_author_parity.py` writes for its failing-gate case had a doubled operator and did not compile. The tests of the loss identity passed, so gate 3 passes. The defect is repaired, and the script now reports any failure of an author import as an error of the check (exit code 2), with one more test for it. The suite has 184 tests. Gate 1 waits for the rerun, and gate 2 for the run of the parity script on the author checkout.
- **Gates 1 to 3, 2026-11-24 (commit 5b07e19).** Gate 1 passes: 184 tests passed on the HPC. Gate 2 passes: on the author checkout at commit b7716e1, the assignment agrees for all 4,000 windows in both rounds, and the two updates differ from the author update by at most 2e-16. Gate 3 passes with the suite. The report of gate 2 said that the work tree of the checkout was not clean. The script read that state after its own import, which writes cache files into the checkout, so the line proved nothing. The script now reads the state before the import, and it records whether the three files that it reads equal the commit, with their hashes. A rerun settles the record.
- **Step 2, built 2026-11-24, not yet run.** `src/eeg_compare/synthetic/` holds `generator.py` (background, templates, signal level, the levels base, background, delay, polarity, few examples), `measures.py` (recovery, atom map, event detection) and `benchmark.py` (one run per level and seed, the budget rule, the reader of gates 4 to 6, of the range rule, and of P5 and P7). `scripts/run_synthetic.py` has the commands `calibrate`, `run` and `read`. The suite has 22 more tests (206 in all). A small smoke setting (`--smoke`) checks the wiring; its results decide nothing.
- **Tests of step 2, 2026-11-24 (commit 4750ad1).** 205 of 206 tests passed on the HPC. The tests of the small smoke setting passed: the runs of all nine cells, the budget rule, the reader, the repeated run and the commands of the script. The one failure was a defect of a measure: the detection computed its ratios in single precision, so a recall of 2/3 was wrong by 2e-8. The ratios now come from integer counts. No run of the benchmark used the earlier code.
- **Step 3.** Not built: the levels with artifacts, bursts and event pairs, and the reader of P1 to P4 and P6.

Open before step 3: the level `burst_near_event` cannot test its mechanism as section 3 states it. A short burst inside the band has few degrees of freedom, so its cosine with a waveform is high, and a burst close to an event lowers the cosine of the event. I will propose a changed layout as an amendment before I build that level.

Commands on the HPC for step 2 (gates 4 to 6, the budget, P5 and P7):

```bash
python -m pytest -q
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py calibrate --root artifacts/synthetic-001
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-001 --levels base --fixed-point
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-001 --levels background
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-001-repeat --levels base --seeds 0 --fixed-point --budget-from artifacts/synthetic-001
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py read --root artifacts/synthetic-001 --determinism-root artifacts/synthetic-001-repeat
```

Commands on the HPC for gates 1 and 2:

```bash
python -m pytest -q
PYTHONPATH="$PWD/src" python scripts/check_author_parity.py --author-repo /path/to/the/author/checkout --output artifacts/author-parity-001.json
```

## References

1. Cano Achuri et al. (2026), reference [1] of the comparison plan. Author code: a local checkout of the mouse pipeline; its clustering is in `si_vq.py` and `si2_kmeans.py`.
2. Papastathopoulos-Katsaros et al. (2026), reference [2] of the comparison plan.

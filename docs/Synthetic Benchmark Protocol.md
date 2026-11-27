# Synthetic Benchmark Protocol (stage 2)

**Stage 2 is complete: the user accepted it on 2026-11-26 with the results of section 15.** The protocol was confirmed by the user on 2026-11-24, with one decision: every test and every run is on the HPC (section 9). Corrections of fact that the build found after the confirmation are in section 11. It makes stage 2 of the [Comparison Plan](Comparison%20Plan.md) exact: the generator, the nine cells, the measures, and the predictions with their decision rules. The synthetic benchmark is the only place of the study where waveform identification has a known answer. It uses no corpus.

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
- The budget is set by convergence of the training loss and not by recovery: on `base` with seed 0, the steps double (to at most 24,000) until the loss changes by less than 1% over the last tenth of the steps. The largest value over the four gradient cells is then the budget of all gradient cells in all conditions. **This rule was replaced on 2026-11-25 (section 11, item 4): the budget is 12,000 steps.**

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

4. **Budget rule of the gradient cells (2026-11-25, after the first read of step 2; decision of the user).** The rule of section 4 gave 3,000 steps. It measured a relative change of the whole loss, and 93% to 97% of that loss is background energy that no atom explains. At 3,000 steps the gradient cells had 72% to 100% of the loss decrease that they have at 24,000 steps (section 13.3), while every alternating cell had stopped by its own rule. The user fixed the budget at 12,000 steps. The rule in the code is now: the start level of a cell is its mean loss over the first 300 steps; its level at a checkpoint is its mean loss over the 300 steps before it; the budget is the first checkpoint at which every gradient cell has at least 95% of the decrease that it has at the largest checkpoint. On the stored traces this gives 12,000 steps, where the four cells have 98.9% to 100% of that decrease. Two limits. This is a change after results, and more steps can only help the gradient cells, which include the separable encoder. And the threshold is at a margin: with 99% the rule gives 24,000 steps, because `SF-L5` has 98.9% at 12,000. In my first report I wrote that 99% also gives 12,000; that used the first 100 steps as the start level and was not the rule as coded. The base and background levels run again in a new root. The results at 3,000 steps stay in section 13 as a record.

5. **Details of step 3 that the protocol left open (2026-11-25, before any run of these levels).**
   - `amplitude_artifacts`: the peak of an artifact has a log standard deviation of 0.5. One artifact is one shape on its 3 channels. The artifacts pass the band of 1 to 30 Hz before they are added, so the peak is that of the shape before the band-pass.
   - `burst_near_event`: the burst is white noise inside the band. The validity check uses one channel of the calibration background, with the amplitude of a channel of a pattern with equal weights. "At the event" means within 12 samples.
   - `overlap`: a segment holds two pairs, in two of the pooling windows 1, 3, 5 and 7. The two starts of a pair are 25 to 50 samples apart, and both frame centers lie in the window. The record holds the shares of the pairs with both events, one event and no event detected.
   - `polarity_d1`: the level `polarity` for a rank-1 cell with one spatial pattern per atom. It is a described read and runs with the `SR` cells only.
   - The extra draws of these levels come from a random stream of their own, so the patterns, the backgrounds and the templates of the events stay those of the base level with the same seed.
   - Beside every prediction on recovery, the reader gives the same contrast on recall, with no rule (section 13.5, item 2).

## 12. Build record

- **Step 1, built 2026-11-24, not yet run.** `src/eeg_compare/cells/` holds the nine cells: `alternating.py` (rules `L1` to `L4` for `S1` and `SF`), `gradient.py` (rule `L5` on the adapters, with the extended loss and the window-normalized loss as new options of the adapters), `api.py` (common start, fit, frozen dictionary, tokens) and `common.py`. The loss identity of gate 3 is a test (`tests/test_cells_gradient.py`). Gate 2 is `scripts/check_author_parity.py`, which needs a checkout of the author repository. The suite has 50 new tests in four files. Local checks: format, lint, types and syntax only.
- **First run on the HPC, 2026-11-24 (commit d923fd4).** 182 of 183 tests passed in 10 s. The one failure was a defect of a test, not of a cell: the stand-in file that `tests/test_author_parity.py` writes for its failing-gate case had a doubled operator and did not compile. The tests of the loss identity passed, so gate 3 passes. The defect is repaired, and the script now reports any failure of an author import as an error of the check (exit code 2), with one more test for it. The suite has 184 tests. Gate 1 waits for the rerun, and gate 2 for the run of the parity script on the author checkout.
- **Gates 1 to 3, 2026-11-24 (commit 5b07e19).** Gate 1 passes: 184 tests passed on the HPC. Gate 2 passes: on the author checkout at commit b7716e1, the assignment agrees for all 4,000 windows in both rounds, and the two updates differ from the author update by at most 2e-16. Gate 3 passes with the suite. The report of gate 2 said that the work tree of the checkout was not clean. The script read that state after its own import, which writes cache files into the checkout, so the line proved nothing. The script now reads the state before the import, and it records whether the three files that it reads equal the commit, with their hashes. A rerun settles the record.
- **Step 2, built 2026-11-24, not yet run.** `src/eeg_compare/synthetic/` holds `generator.py` (background, templates, signal level, the levels base, background, delay, polarity, few examples), `measures.py` (recovery, atom map, event detection) and `benchmark.py` (one run per level and seed, the budget rule, the reader of gates 4 to 6, of the range rule, and of P5 and P7). `scripts/run_synthetic.py` has the commands `calibrate`, `run` and `read`. The suite has 22 more tests (206 in all). A small smoke setting (`--smoke`) checks the wiring; its results decide nothing.
- **Tests of step 2, 2026-11-24 (commit 4750ad1).** 205 of 206 tests passed on the HPC. The tests of the small smoke setting passed: the runs of all nine cells, the budget rule, the reader, the repeated run and the commands of the script. The one failure was a defect of a measure: the detection computed its ratios in single precision, so a recall of 2/3 was wrong by 2e-8. The ratios now come from integer counts. No run of the benchmark used the earlier code.
- **Step 3, built 2026-11-25.** The generator has the levels `amplitude_artifacts`, `burst_near_event` (section 14) and `overlap`. The reader applies the rules of P1, P1s, P2, P3, P4 and P6. The burst level refused to run at its first check on the HPC; its calibration changed that day (section 14.1). The suite has 214 tests.

Commands on the HPC for the rerun of step 2 at 12,000 steps (2026-11-25; the first command reads the stored loss traces and makes no new fit):

```bash
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py calibrate --root artifacts/synthetic-002 --traces-from artifacts/synthetic-001
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-002 --levels base --fixed-point
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-002 --levels background
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-002-repeat --levels base --seeds 0 --fixed-point --budget-from artifacts/synthetic-002
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py read --root artifacts/synthetic-002 --determinism-root artifacts/synthetic-002-repeat
```

Commands on the HPC for step 3, after the gates of the rerun pass. The burst level runs first: its validity check stops it in the first minute if the layout is not valid.

```bash
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-002 --levels burst_near_event
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-002 --levels amplitude_artifacts overlap delay_4 delay_8 delay_16 few_10 few_25 few_50 polarity more_atoms
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py run --root artifacts/synthetic-002 --levels polarity_d1 --cells SR-L5
PYTHONPATH="$PWD/src" python scripts/run_synthetic.py read --root artifacts/synthetic-002 --determinism-root artifacts/synthetic-002-repeat
```

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

## 13. Results of step 2 (base and background, signal level 8)

Root `artifacts/synthetic-001` on the HPC, code at commit a19f1c4, 206 tests passed. Read on 2026-11-25. Every number is a mean over the five seeds unless stated. The four templates are, in this order: spike with slow wave, damped 10 Hz oscillation, biphasic sharp wave, 7 Hz burst.

### 13.1 Gates and predictions

| Rule | Result |
|---|---|
| Gate 4, signal level | Pass. Oracle recall: 0.97 for the single-channel cells, 1.00 for the multichannel cells. |
| Gate 5, fixed point | Pass. Every cell keeps at least 3 of 4 templates in all 5 seeds. |
| Range rule | Pass. 2.98 recovered templates of 4, over all learned cells. |
| Gate 6, determinism | Pass. The repeated run of seed 0 has no difference in 36 records, on the GPU. |
| P5, optimizer | Holds. `S1-L4` minus `S1-L5` is -0.019 in waveform recovery. Four seeds lie within 0.02; seed 1 is -0.127. |
| P7, null | Does not hold. The largest recovery in `background` is 0.916. My probability was 0.85. |

Before the run I gave about even odds that gate 4 fails for the single-channel cosine cells. It passed.

### 13.2 P7: two templates look like the background

Waveform recovery by template: `base`, learned | `background`, learned (the floor).

| Cell | Base | Floor |
|---|---|---|
| `S1-L1` | 0.92, 0.48, 0.97, 0.87 | 0.75, 0.30, 0.77, 0.49 |
| `S1-L2` | 0.96, 0.55, 0.98, 0.95 | 0.75, 0.30, 0.77, 0.47 |
| `S1-L3` | 0.98, 0.55, 0.99, 0.93 | 0.76, 0.28, 0.73, 0.44 |
| `S1-L4` | 0.97, 0.54, 0.99, 0.86 | 0.73, 0.25, 0.70, 0.42 |
| `S1-L5` | 0.98, 0.56, 0.99, 0.91 | 0.73, 0.25, 0.67, 0.40 |
| `SF-L1` | 1.00, 0.80, 1.00, 0.96 | 0.74, 0.32, 0.72, 0.58 |
| `SF-L5` | 0.99, 0.89, 1.00, 1.00 | 0.73, 0.28, 0.72, 0.44 |
| `SR-L5` | 0.99, 0.83, 0.99, 0.97 | 0.76, 0.30, 0.72, 0.46 |
| `SR-L5n` | 0.99, 0.83, 0.99, 0.96 | 0.80, 0.30, 0.75, 0.50 |

- A dictionary that never saw an event matches the spike with slow wave and the biphasic wave at 0.67 to 0.80. In 15 of the 180 background values the match is 0.80 or more, for seven of the nine cells and only for these two templates. Random background patches (the frozen start) match them at 0.55 to 0.69, so learning on background adds about 0.1.
- The cause is the slow content of the two templates in a 1/f background. This is the lesson of the earlier harness again. My rule for the templates ("oscillatory content") did not prevent it.
- The decoys read 0.69 to 0.79 in `base` and 0.54 to 0.59 in `background`. So a dictionary that holds the planted waveforms also matches other waveforms of their families.
- Consequences. The waveform recovery has little range for two of the four templates (floor about 0.75, ceiling 1). The differences between cells come from the damped oscillation and the burst. The count of recovered templates is not usable as a result: for a floor of 0.80 the rule asks for a recovery above 1. It served gate 5 and the range rule, which passed in spite of it. Recovery is read by template and against the floor, and recall is read with it.
- For the corpus: an atom that looks like a spike with a slow wave is no evidence of such events. The waveform review of stage 7 needs a dictionary learned on background as its null.

### 13.3 The budget rule is too weak

The rule gave 3,000 steps: every gradient cell changed its loss by less than 1% over the last two tenths. The loss traces of the calibration (seed 0) show why this says little. Most of the loss is background energy that no atom can explain, so the whole decrease from the start to 24,000 steps is only 2.6% to 7.4% of the loss.

| Cell | Loss at the start | 3,000 | 6,000 | 12,000 | 24,000 |
|---|---|---|---|---|---|
| `S1-L5` | 0.4065 | 0.3809 (85%) | 0.3763 (100%) | 0.3764 (100%) | 0.3764 |
| `SF-L5` | 0.5029 | 0.4834 (91%) | 0.4822 (97%) | 0.4817 (99%) | 0.4815 |
| `SR-L5` | 0.4933 | 0.4811 (87%) | 0.4803 (93%) | 0.4792 (100%) | 0.4793 |
| `SR-L5n` | 0.4435 | 0.4321 (100%) | 0.4321 (100%) | 0.4321 (100%) | 0.4321 |

The value is the mean loss of the last 300 steps before the checkpoint; in parentheses, the share of the decrease that the cell has at 24,000 steps, with the first 100 steps as the start level. The rule as coded on 2026-11-25 takes the first 300 steps as the start level (section 11, item 4). Its shares are 72%, 85%, 79% and 100% at 3,000 steps, and 100%, 98.9%, 100% and 100% at 12,000 steps, for the four cells in the order of the table. At 3,000 steps three of the four cells have 85% to 91% of their decrease. The alternating cells stop by their own rule after 21 to 50 passes, all below the limit of 100. So the present comparison sets converged alternating cells against gradient cells that are not converged. The rule was my design, and the defect is mine: a relative change of the whole loss does not measure convergence when most of the loss cannot decrease.

### 13.4 What the base level shows

Recall in `base`: frozen start, learned, fixed point, oracle; then the learned recall by template.

| Cell | Frozen | Learned | Fixed point | Oracle | Learned, by template |
|---|---|---|---|---|---|
| `S1-L1` | 0.25 | 0.55 | 0.67 | 0.97 | 0.84, 0.00, 0.93, 0.44 |
| `S1-L2` | 0.25 | 0.61 | 0.69 | 0.97 | 0.92, 0.00, 0.94, 0.56 |
| `S1-L3` | 0.25 | 0.61 | 0.78 | 0.97 | 0.96, 0.00, 0.96, 0.54 |
| `S1-L4` | 0.45 | 0.62 | 0.84 | 0.97 | 0.99, 0.00, 0.98, 0.51 |
| `S1-L5` | 0.45 | 0.66 | 0.84 | 0.97 | 0.99, 0.00, 0.99, 0.66 |
| `SF-L1` | 0.50 | 0.78 | 0.91 | 1.00 | 0.99, 0.41, 0.98, 0.75 |
| `SF-L5` | 0.51 | 0.86 | 0.91 | 1.00 | 0.99, 0.58, 0.99, 0.87 |
| `SR-L5` | 0.48 | 0.76 | 0.87 | 1.00 | 0.97, 0.41, 0.97, 0.69 |
| `SR-L5n` | 0.48 | 0.75 | 0.88 | 1.00 | 0.99, 0.40, 0.97, 0.65 |

These are reads of one level, with no rule. The standard deviation of the learned recall over the seeds is 0.03 to 0.16, so a difference below about 0.1 between two cells is not resolved here.

- **All cells find the two slow templates** (recall 0.84 to 0.99). The cells differ on the damped 10 Hz oscillation and on the burst.
- **The single-channel cells do not find the damped oscillation** from the common start (recovery 0.48 to 0.56, recall 0.00). The multichannel cells find it in part (recovery 0.80 to 0.89, recall 0.40 to 0.58). All templates have one signal level. The damped oscillation lies where the background is weak, so its raw amplitude is the smallest. An equal signal level does not give an equal chance to be learned.
- **Finding is not the only loss.** From the planted dictionary, the cells keep the templates (recovery 0.91 to 1.00), with one exception: `S1-L1` and `S1-L2` let the damped oscillation drift (0.70 and 0.75), and `S1-L3` to `S1-L5` keep it (0.91 to 0.97). Yet the fixed-point recall (0.67 to 0.91) is below the oracle recall (0.97 to 1.00). The oracle has 4 atoms and the cells have 8. With one token per window, an atom that fits the background takes windows from a weak event.
- **The score acts at inference too.** With the same frozen single-channel atoms, raw tokens find 0.45 of the events and cosine tokens 0.25.
- **Rules inside `S1`**: `L2` to `L5` are close (recovery 0.84 to 0.86, recall 0.61 to 0.66); `L1` is lowest (0.81, 0.55).
- **`SF-L5` and `SR-L5`** have equal template recovery within the margin (0.935 and 0.922). The window-normalized loss changes nothing here (`SR-L5n` 0.918), as expected without artifacts.
- **Cost.** An alternating fit takes 0.2 to 0.7 s. A gradient fit of 3,000 steps takes 17 to 30 s.

### 13.6 Rerun at 12,000 steps (root `artifacts/synthetic-002`)

Code at commit acef908. Read on 2026-11-25 from the printed report of the reader; the files of this root were not on the local machine. The four gates pass again, P5 holds and P7 does not hold. The alternating cells have the same values as before, as they must: they have no steps.

Gradient cells in `base`, learned dictionary, 3,000 steps to 12,000 steps:

| Cell | Waveform recovery | Recall | Recall of the fixed-point fit |
|---|---|---|---|
| `S1-L5` | 0.861 to 0.858 | 0.661 to 0.653 | 0.837 to 0.802 |
| `SF-L5` | 0.967 to 0.975 | 0.857 to 0.864 | 0.912 to 0.903 |
| `SR-L5` | 0.944 to 0.945 | 0.760 to 0.762 | 0.875 to 0.871 |
| `SR-L5n` | 0.942 to 0.946 | 0.753 to 0.754 | 0.877 to 0.875 |

- Four times the steps change the learned recovery and recall by at most 0.01. This is far below the spread between seeds. So the reads of section 13.4 stand, and the weak budget rule did not distort the base level.
- The fixed-point recall of `S1-L5` falls by 0.035: with more steps the cell moves farther from the planted dictionary.
- P5: `S1-L4` minus `S1-L5` is -0.016 in waveform recovery (0.842 and 0.858).
- The count of recovered templates of `SR-L5n` rises from 2.6 to 3.6 with no change of its recovery (0.942 to 0.946). Only its background floor moved. This shows again that the count is not usable.
- A limit of my recommendation: the change to 12,000 steps cost four times the gradient time and changed no read of the base level. The alternative that I named (a check of the base level alone) would have shown this at a lower cost. For the stress levels the question is open, so step 3 runs at 12,000 steps as decided.

### 13.5 Decisions after step 2

State on 2026-11-25: the user decided item 1 (12,000 steps; section 11, item 4). Item 2 is adopted as proposed, with no objection of the user. Item 3 is the proposal of section 14 and waits for confirmation.

1. **Budget of the gradient cells.** Proposal: replace the rule by "the first checkpoint at which every gradient cell has at least 99% of the loss decrease that it has at the largest checkpoint". It uses loss traces only. It gives 12,000 steps, and a threshold of 95% gives the same. The base and background levels then run again. The gradient fits take four times longer.
2. **Reading of the recovery.** Proposal: keep the templates, read recovery by template against the floor, and report recall beside every prediction on recovery. New templates would be a change after the results.
3. **The level `burst_near_event`** needs a changed layout before step 3 (section 12).

## 14. Layout of `burst_near_event` (confirmed by the user on 2026-11-25; built, not yet run)

Section 3 states the level as a noise burst in the pooling window of each event, at least 62 samples away, with a raw response of 1.5 times that of the event. In this form the level cannot test its mechanism, for two reasons.

- **Few degrees of freedom.** A burst inside the band of 1 to 30 Hz that lasts 0.1 to 0.3 s holds about 6 to 15 free values. The largest cosine of such a burst with a waveform, over the placements, is then about 0.7. So the cosine score takes a short burst for a waveform too.
- **Overlap.** A burst 62 samples from the event center lies inside the atom frame of the event. Its energy enters the norm of the frame and lowers the cosine of the event.

Layout, which replaces the row of this level in section 3:

1. **Events.** A segment holds one event of each template, in pooling windows 1, 3, 5 and 7, in random order. The center of the event frame lies in the first 19 positions of its window. The next window holds no event.
2. **Burst.** Noise inside the band, 0.5 s long (125 samples) with a Hann window, on the channels of the pattern of the event, independent between channels. Its center lies 106 samples after the center of the event frame, so in the same pooling window. It does not touch the waveform (the middle 88 samples of the frame). It touches the last 18 samples of the event frame only with its tapered start.
3. **Scale.** On channel `c` the burst has the standard deviation `r * amplitude * pattern[c]`. The factor `r` is set once per template with the calibration seed, so that the median largest raw response of the planted waveform to the burst is 1.5 times its response to the event on that channel.
4. **Validity check, before any cell runs.** On the calibration draws, per template: the raw response to the burst is above the response to the event in at least 80% of the draws, and the cosine at the event is above the largest cosine at the burst in at least 80% of the draws. The generator records both shares. If a share is below 80%, the level does not run, and the layout changes again.

### 14.1 First check on the HPC and the changed calibration (2026-11-25)

The level did not run. Its validity check gave, for the four templates in order:

| Share of the calibration draws | Spike with slow wave | Damped 10 Hz | Biphasic wave | 7 Hz burst |
|---|---|---|---|---|
| Raw score away from the event (at least 0.80 needed) | 0.54 | 0.39 | 0.69 | 0.68 |
| Cosine score at the event (at least 0.80 needed) | 1.00 | 0.99 | 0.96 | 0.97 |

- The cosine side passes for every template. The raw side fails for every template. Before the run I named the cosine side of the damped oscillation as the risk. That was wrong.
- Cause, in my calibration of item 3. The center of the burst lies at the end of the pooling window of the event. About half of the placements that match the burst have their center in the next window, and they do not compete with the event. I had taken the largest response over all placements. So the burst was weaker inside the window than the ratio of 1.5 says.
- Change 1. The factor of the burst is calibrated on the placements whose center lies in the pooling window of the event.
- Change 2. The ratio is not fixed at 1.5. The generator tries 1.5, 2.0, 2.5 and 3.0 and takes, for each template, the first ratio for which both shares are at least 0.80. If no ratio passes for a template, the level does not run. The record of a run holds the ratio and the two shares of every template.
- This sets the strength of the manipulation with the check of the manipulation. It uses the calibration draws only. No result of a cell enters.

P2 does not change: in this level, the recall of `S1-L3` is above that of `S1-L4` by at least 0.10. Two reads are added, with no rule: the recall of the oracle under both scores, which shows the effect of the score at inference alone; and the multichannel cells, where a burst that is independent between channels adds up less than an event.

## 15. Results of the stage (all levels, signal level 8, 12,000 steps)

Root `artifacts/synthetic-002` on the HPC. The burst level ran with the code of commit e44f236. The other levels ran with acef908 or e44f236; the two commits differ in the burst level only. Read on 2026-11-26 from the files of the root. 14 levels, 5 seeds, 9 cells: 1,815 records, and 7.6 hours of fits. Every number is a mean over the five seeds unless stated. The four gates pass (section 13.6).

### 15.1 The eight predictions

| | Prediction | Result | Mean difference (margin) | Seeds with the predicted sign | My probability |
|---|---|---|---|---|---|
| P1 | With artifacts, `S1-L1` recovers more than `S1-L3` | **Opposite** | -0.100 (0.05) | 1 of 5 | 0.55 |
| P1s | Artifacts cost `S1-L3` more recovery than `S1-L1` | Undecided | -0.047 (0.05) | 2 of 5 | 0.70 |
| P2 | With bursts, recall of `S1-L3` is above `S1-L4` | Undecided | +0.070 (0.10) | 5 of 5 | 0.70 |
| P3 | At a delay of 16 samples, `SF-L5` is above `SR-L5`; equal in `base` | **Holds** | +0.192 (0.05); base +0.012 | 5 of 5 | 0.85 |
| P4 | At 10 segments, `SR-L5` is above `SF-L5`; equal in `base` | **Holds** | +0.112 (0.05); base -0.012 | 5 of 5 | 0.60 |
| P5 | `S1-L4` and `S1-L5` are equal in `base` | **Holds** | -0.016 (0.05) | | 0.60 |
| P6 | Artifacts cost `SR-L5n` less recovery than `SR-L5` | Undecided | +0.003 (0.05) | 2 of 5 | 0.65 |
| P7 | No recovery of 0.80 on background | **Does not hold** | largest value 0.852 | | 0.85 |

Three predictions hold, one is opposite, three are undecided, and the null fails. The three that hold are those on the atom structure and on the optimizer. The four on the window weight and on the position score do not hold by their rule. I was too sure of P7, and of the three that are undecided.

### 15.2 Atom structure (P3, P4)

Template recovery of the learned dictionaries:

| Level | `SF-L1` | `SF-L5` | `SR-L5` |
|---|---|---|---|
| Delay 0 (`base`) | 0.88 | 0.93 | 0.92 |
| Delay 4 | 0.86 | 0.91 | 0.87 |
| Delay 8 | 0.82 | 0.83 | 0.73 |
| Delay 16 | 0.70 | 0.68 | 0.49 |
| 200 segments (`base`) | 0.88 | 0.93 | 0.92 |
| 50 segments | 0.79 | 0.79 | 0.90 |
| 25 segments | 0.72 | 0.85 | 0.90 |
| 10 segments | 0.64 | 0.72 | 0.83 |

- A delay costs the rank-1 atoms more, in proportion to the delay. With the planted factors and no learning, the rank-1 oracle finds 0.76 of the events at a delay of 16 samples; the full oracle finds all.
- A delay costs the full atoms too. At 16 samples `SF-L5` keeps the two slow templates (0.98 and 0.97) and loses the damped oscillation (0.31) and the burst (0.47). So the full structure can hold a delay, but it does not learn every delayed template here.
- With few examples the rank-1 atoms are stable (0.83 to 0.92). The full atoms vary more (0.72 to 0.93), and the alternating full cell `SF-L1` falls in a line from 0.88 to 0.64.
- The recall beside P4 is +0.035, which is small. P4 holds on template recovery, its own measure.
- A limit of the delay levels: the event detection maps an atom to a template by its waveform recovery. For a delayed oscillation the projection on the pattern cancels in part, so some good atoms have no template and their tokens do not count. The recall of the delay levels is too low for this reason. P3 uses template recovery and is not affected.

### 15.3 Optimizer (P5)

The alternating cell `S1-L4` and the gradient cell `S1-L5` have equal waveform recovery in `base` (0.842 and 0.858). In six other levels the mean difference is between -0.03 and +0.04. An alternating fit takes 0.5 s; a gradient fit of 12,000 steps takes 73 s for `S1` and `SF` and 120 s for `SR`.

### 15.4 Window weight (P1, P1s, P6)

- With artifacts, the equal weights of `S1-L1` recover less than the other rules of the path: 0.81 against 0.91 (`L2`), 0.91 (`L3`), 0.91 (`L4`) and 0.89 (`L5`). The plan expected the opposite.
- The level changes two things at once. The amplitude of an event is log-normal, and a segment holds two artifacts. The rules that weight a window by its energy gain from the strong events: their recovery rises from `base` to this level (`L3`: 0.86 to 0.91), most for the damped oscillation (0.55 to 0.69). The rare artifacts of random shape do not take their atoms. So the level cannot say what the artifacts alone do. This is a defect of my design of the level.
- The window-normalized loss changes nothing in this level (template recovery 0.93 for `SR-L5` and for `SR-L5n`).
- In the burst level the picture turns (section 15.5).

### 15.5 Position score (P2) and the burst level

Each template took the ratio 2.0, with one exception (the 7 Hz burst in seed 2 took 1.5). The check passed with 0.80 to 0.89 on the raw side and 0.94 to 0.97 on the cosine side.

Recall in `burst_near_event`:

| | `S1-L1` | `S1-L2` | `S1-L3` | `S1-L4` | `S1-L5` | `SF-L1` | `SF-L5` | `SR-L5` | `SR-L5n` |
|---|---|---|---|---|---|---|---|---|---|
| Oracle | 0.70 | 0.70 | 0.70 | 0.07 | 0.07 | 1.00 | 0.58 | 0.58 | 0.58 |
| Learned | 0.23 | 0.19 | 0.16 | 0.09 | 0.12 | 0.71 | 0.06 | 0.05 | 0.14 |

- **At inference the score decides.** With the planted dictionary, single-channel cosine tokens find 0.70 of the events and raw tokens 0.07. The difference is 0.59 to 0.66 in every seed. For full templates it is 1.00 against 0.58.
- **P2 is undecided by its rule.** The learned `S1-L3` is above `S1-L4` in all five seeds, but by 0.07, and the margin is 0.10. Both cells are near the floor, because neither learns the templates in this level (waveform recovery 0.71 and 0.65).
- **Read after the results, with no rule:** learning with the raw score and the least-squares weight fails when a burst of high energy follows every event. `SF-L5` and `SR-L5` reach a template recovery of 0.21 and 0.29. The full cell with the cosine score and equal weights, `SF-L1`, reaches 0.79 and a recall of 0.71; it is above `SF-L5` by 0.45 to 0.75 in every seed. The two cells differ in three choices, so this does not name the cause. On the single-channel path, where one choice changes per step, the recall falls with every step from `L1` to `L4` (0.23, 0.19, 0.16, 0.09), but all of these values are low.
- **The window-normalized loss helps here**, also read after the results: `SR-L5n` has a template recovery of 0.64 against 0.29 for `SR-L5`, with 0.20 to 0.60 more in every seed. Its recall stays low (0.14), because its tokens still use the raw score.
- A limit of the validity check: it used one atom on one channel. In the run, four atoms compete on 22 channels, and the cosine oracle finds only 0.28 of the damped oscillations. The check was necessary and not enough.
- With artifacts and no burst, the score acts in the same direction and less: the single-channel oracle finds 0.84 with the cosine score and 0.76 with the raw score.

### 15.6 Described levels

- **More atoms.** With 16 atoms every cell recovers every template (waveform recovery 0.97 to 1.00), and the recall rises: `S1-L5` from 0.65 to 0.84, `SF-L5` from 0.86 to 0.90, `SR-L5` from 0.76 to 0.85. So the miss of the damped oscillation by the single-channel cells in `base` was a limit of 8 atoms, and not of the structure. This corrects the read of section 13.4. The recall of the damped oscillation stays below the oracle (0.33 to 0.70 against 0.89 to 1.00), so the loss of windows to other atoms at one token per window remains.
- **Overlap.** One event of a pair is found in 0.74 to 0.84 of the pairs and both in 0.09 to 0.15. The oracle finds exactly one in 0.93 to 0.96. This is the expected limit of one token per window.
- **Polarity** (4 atoms for 4 templates). No single-channel or full cell finds an inverted event (0.00 to 0.02). The rank-1 cells with two patterns find 0.20 to 0.21 of them, and with one pattern 0.00. So a second pattern takes over a part of the inverted events, as the plan expected.

### 15.7 Limits

- Five seeds. The standard deviation of the recall between seeds is 0.03 to 0.12, so three predictions are undecided, and a difference below about 0.1 is not resolved.
- Eight predictions and many described reads. Every read of sections 15.4 to 15.6 that is marked "after the results" has no rule and counts as a lead.
- I made the generator. The artifact level is mild and mixes two changes; the burst level is extreme. The stage shows that a mechanism exists on ideal data. It does not show how large it is on EEG.
- The recovery measure has a high floor for two templates (section 13.2), and the recall depends on the map from atoms to templates.
- The budget and the layout of the burst level changed after first results (section 11, item 4; section 14.1).

### 15.8 What the stage sets for the corpus

By the rules of section 10:

- P3 and P4 hold, so the contrast of the atom structures has support on ideal data. Full atoms gain with delays between channels; rank-1 atoms gain with few examples. On TUEV the spike class has few examples, so this contrast goes to the corpus first.
- P5 holds, so the optimizer is not a factor. Where an alternating rule exists, it gives the same atoms at about one hundredth of the time.
- Neither P1s nor P6 holds, so the window-normalized loss does not become a main arm by the rule. The burst level gives a lead for it, read after the results.

Beyond the rules, as proposals:

- Put the position score at inference into the first slice on the corpus. It needs no new fit: one dictionary, two encoders. It showed the largest effect of the stage.
- Choose the number of atoms with a margin and vary it on the slice. With 8 atoms the cells lost a template that they all found with 16.
- Compare every learned waveform with a dictionary learned on background, before any claim on morphology.

## References

1. Cano Achuri et al. (2026), reference [1] of the comparison plan. Author code: a local checkout of the mouse pipeline; its clustering is in `si_vq.py` and `si2_kmeans.py`.
2. Papastathopoulos-Katsaros et al. (2026), reference [2] of the comparison plan.

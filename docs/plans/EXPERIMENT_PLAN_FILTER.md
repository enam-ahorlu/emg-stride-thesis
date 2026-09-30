# EXPERIMENT_PLAN_FILTER.md: what do the two acausal preprocessing stages actually cost?

**Why.** The thesis claims deployability as its fourth contribution and fixes a causal constraint in
§3.13.1, but two stages of the preprocessing chain look backwards in time and neither has ever been
measured. `preprocess_emg.py` line 194 applies the 20 to 450 Hz 4th-order Butterworth with
`scipy.signal.filtfilt`, which is zero-phase forward-and-backward and therefore acausal by
construction. Line 208 builds the 50 ms linear envelope with
`rolling(win_samples, center=True, min_periods=1).mean()`, which is centred and therefore reads
25 ms into the future of every sample. §3.2.1 and §3.2.2 now disclose both, and §5.11 and §5.10 carry
the limitation, but a disclosure is not a measurement. This experiment turns both into numbers.

The measurement matters in a specific direction. If the fully causal chain costs almost nothing, the
disclosure becomes a strength: the thesis can state that its headline pipeline transfers to a causal
implementation intact. If it costs several points, the 81.7% causal figure in §4.13.1 is itself
optimistic, because it was computed on acausally filtered signals, and the deployability contribution
needs rewriting rather than a footnote. Either way the thesis is better off knowing.

**A 2 by 2, not a single causal arm.** The two stages were disclosed separately and there is no
reason to assume they cost the same, so they are crossed rather than flipped together. This costs one
extra preprocessing pass and buys the ability to say which stage carries the penalty, which is the
part an engineer reading the thesis would act on.

| Arm | Bandpass | Envelope | Status |
| --- | --- | --- | --- |
| A | `filtfilt` (zero-phase) | centred 50 ms | published reference |
| B | `lfilter` (single pass) | centred 50 ms | filter made causal |
| C | `filtfilt` (zero-phase) | trailing 50 ms | envelope made causal |
| D | `lfilter` (single pass) | trailing 50 ms | fully causal |

**Verified on disk.** The raw dataset is present at `06_Code/SIAT_LLMD20230404/Sub01..Sub40`, so
re-preprocessing from source is possible rather than hypothetical. The published 250 ms Freq-72
features are `features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz`
with its `_meta.csv` and `_cfg.json`; the cfg records `use_freq: true`, `use_wavelet: false`,
`sampling_rate: 2000.0`, so the published Freq-72 set was built with `--freq --no-wavelet --prefix
freq`. The reference results are `results_loso_freq_persubj/` at 0.7767 for the SVM and 0.7732 for
the Random Forest, confirmed against `results_featureset_loso/featureset_loso_summary.csv`.

Run from `06_Code/` in the project `.venv`. CPU only for Phases 0 to 3.

**Scoring is doubled.** Gate A of `EXPERIMENT_RUN_ORDER.md` resolved to Outcome A: causal five-window
probability-averaged smoothing lifts the offline configuration by about 9 pp and cuts the critical
error rate by more than half. Every arm here is therefore scored twice, unsmoothed and at k = 5
probability-averaged, using the same trial-boundary-respecting vote `run_causal_smoothing.py` already
implements. Reuse that function; do not write a second one.

Do NOT fabricate numbers. Do NOT overwrite `features_out/`, `results_loso_freq_persubj/` or any other
published results directory. Do NOT edit any thesis chapter.

---

## Phase 0. Add the causal switches, then gate on reproducing arm A exactly [RUN, CPU]

1. Add two flags to `preprocess_emg.py`, defaulting to the current behaviour so nothing already on
   disk changes meaning: `--causal-bandpass` swaps `filtfilt(b, a, x)` for `lfilter(b, a, x)` in
   `apply_bandpass`, and `--causal-envelope` swaps `center=True` for `center=False` in
   `rectify_and_envelope`. Keep `min_periods=1` in both cases. Record both flags in the config dump
   so the provenance survives.
2. Extend `_auto_tag` so the arm is visible in every output filename, for example
   `w250_ov50_conf60_AorR_cbTrue_ceFalse`. Do not reuse the published stems.
3. Regenerate arm A with both flags off, at 250 ms, subjects 1 to 40, all four movements, then
   extract Freq-72 features from it with `--freq --no-wavelet`.
4. **GATE:** the regenerated arm A feature matrix must match the published
   `freq_windows_..._w250_ov50_conf60_AorR_features_ext.npz` to within floating-point tolerance
   (max absolute difference below 1e-8 on a per-column basis), with an identical row count of 26,347
   and an identical label vector. If it does not match, stop and report the first differing column
   and the size of the discrepancy. A pipeline that cannot rebuild its own published features cannot
   be trusted to measure a perturbation of them, and every number downstream would be a mix of the
   effect and the drift.

## Phase 1. Build arms B, C and D [RUN, CPU]

Preprocess and extract features for the remaining three arms with the same settings and the same
250 ms window. Then:

1. **GATE:** all four arms must produce exactly 26,347 windows with identical `y_int` vectors and
   identical `subject`, `movement` and `t_start` columns. Labels come from the Status column and the
   windowing is unchanged, so any divergence here means a flag is touching more than it should. If
   the counts differ, stop and report which arm and by how much.
2. Report, for one representative subject and channel, the mean absolute difference between arm A and
   each of B, C and D on the envelope signal, plus the group delay the single-pass filter introduces
   at 100 Hz. This is three lines of output and it makes the rest of the experiment interpretable:
   if B is numerically almost identical to A the classifier result is unsurprising rather than
   suspicious.

## Phase 2. Classical arm, four arms, no new hyperparameter search [RUN, CPU]

`train_classical_loso.py` with Freq-72, `--norm-mode persubj`, and critically
`--reuse-params-dir results_loso_freq_persubj` so that the published hyperparameters are held fixed
and the only thing varying across arms is the preprocessing. This is the correct choice for a
perturbation study and not merely a cost saving: re-tuning per arm would confound the filter effect
with search noise. State in the report that hyperparameters were held at the published values.

It also avoids the failure mode recorded in `sept-2026-wave1-experiments-outcomes.md`, where a nested
search on this 15.7 GB machine restarted 499 times in five hours. Even so, run one model at a time,
never `SVM,RF` together, each wrapped in `run_with_memory_guard.py --max-mem-percent 92
--min-free-gb 1.2`, with `--rf-n-jobs 1`, and launch through the harness's own background mechanism
rather than `nohup`. Pass the inner python as an absolute Windows path.

Use `--save-preds --flush-preds --save-proba` for every arm, with `--proba-out` pointed inside that
arm's own output directory, because Phase 3 needs per-window probabilities.

**GATE:** arm A must reproduce 0.7767 for the SVM and 0.7732 for the Random Forest to within 0.003.
If it misses, stop. Everything in this experiment is a delta against arm A.

Outputs: `results_filter_causal/{A,B,C,D}/` with the usual summary and subjectwise CSVs.

## Phase 3. Score every arm twice, unsmoothed and smoothed [RUN, CPU]

For each arm and each model, compute the 40 per-subject macro-F1 values and the pooled DNS to WAK
critical-error rate, first on the raw per-window predictions and then after the causal five-window
probability-averaged vote from `run_causal_smoothing.py`, sorting by `movement` then `t_start` so the
vote never crosses a movement-trial boundary. Use the **pooled** critical-error definition, not the
per-subject mean: the pooled form is what Table 4.6's 6.6% headline is computed with, and the
per-subject mean gives roughly 5.8% for the same data, which would silently fail to line up.

## Phase 4. Statistics [RUN, CPU]

Per-subject paired contrasts across the 40 subjects, within each model and each scoring mode: B vs A,
C vs A, D vs A, and D vs the better of B and C. Paired Wilcoxon signed-rank, Cohen's d, BCa 95%
intervals, Holm correction within this family. Same again for the critical-error rate.

Judge every difference against the 0.5 pp run-to-run nondeterminism band of §3.16, as the Deep CORAL
D1 run did. A significant contrast whose interval does not clear that band is a near-tie and must be
reported as one.

These add members to the whole-thesis Benjamini-Hochberg family. Count them and say how many. Do
**not** run `recompute_unified_fdr_v5.py` here; the family is rebuilt once, in the write-up wave.

## Phase 5. Figure [RUN, CPU]

`report_figs/new_experiments/causal_filter.png`. Two panels sharing an x axis of the four arms:
macro-F1 on the left, DNS to WAK critical-error rate on the right, one line per model, solid for
unsmoothed and dashed for k = 5 smoothed. Error bars are one standard error across the 40 subjects.
Match the existing `report_figs/new_experiments` style.

## Phase 6. Deep arm, GATED [do not run unless the gate opens]

**Gate C.** Run the ResNet-SE+CD arm on the fully causal preprocessing (arm D) only if the classical
result shows a drop of 1 pp or more for either model. If the classical drop is under 1 pp, the deep
arm is not worth a GPU run: the preprocessing perturbation is small and the conclusion is already
established. If the gate opens, run arm D only, not all four, with the published training
configuration and seed, and report it beside arm A's published 0.840.

Stop and report before starting Phase 6 either way. Do not decide the gate silently.

---

## One thing this experiment does not measure, and must say so

Every arm here is scored under per-subject z-score normalization computed on the held-out subject's
own data, which is the published transductive protocol. The experiment therefore isolates the
preprocessing factor alone. It does **not** produce an end-to-end deployable figure, because the
calibration-buffer discount of §4.13 and the preprocessing discount measured here have never been
composed. They are not safely additive and nobody should add them. If both discounts turn out to be
material, composing them is a further run and the report should say so rather than letting a reader
subtract twice.

## What each outcome means

**Outcome 1, the fully causal arm D costs under 1 pp for both models.** The best result available.
§3.2.1, §3.2.2, §5.10 and §5.11 keep their disclosure and gain a measured sentence saying the acausal
stages were cosmetic, and §6.2 can state that the pipeline transfers to a causal implementation
intact. Gate C stays shut.

**Outcome 2, arm D costs between 1 and 3 pp.** Report the number in §4.13 as a second deployability
discount, name it in §5.11 and in the limitation at §5.10, and make sure the abstract does not imply
a causal figure it has not measured. Gate C opens.

**Outcome 3, arm D costs more than 3 pp.** The deployability contribution needs real rewriting. The
81.7% in §4.13.1 was computed on acausally filtered signals and is itself optimistic, so the honest
headline for a deployable pipeline is lower than anything currently in the thesis. Do not soften
this. Gate C opens and the composed run becomes necessary rather than optional.

**Outcome 4, the 2 by 2 separates cleanly.** Whatever the total, if one stage carries nearly all of
it, say which and by how much. The centred envelope is the likelier culprit, since 25 ms of lookahead
on a 50 ms smoother is a larger fraction of the operation than the phase correction is of a
wide-band filter, but that is a prediction and not a finding. If it goes the other way, report it as
the surprise it would be.

**Outcome 5, a gate fails.** Phase 0 failing means the pipeline cannot rebuild its own published
features, which is a more serious finding than anything else in this plan and must be reported as
such rather than worked around. Phase 1 or Phase 2 failing means the flags are touching more than
they should. Stop and report in all three cases.

## After the runs: hand back for the write-up cascade

Leave the CSVs and the PNG. Report:

1. Whether Phase 0's exact-reproduction gate passed, and the size of any discrepancy.
2. Window counts and label identity across all four arms.
3. The envelope-difference and group-delay lines from Phase 1.
4. The full four-arm table, both models, both scoring modes, macro-F1 and pooled critical-error rate,
   with arm A's published figures beside it.
5. The Phase 4 contrasts with intervals, and which of them clear the 0.5 pp nondeterminism band.
6. Which of Outcomes 1 to 5 occurred, in one sentence, with the numbers that decide it.
7. Whether Gate C opened, and nothing further until that is answered.
8. How many new BH family members this adds.

---

## AMENDMENT, 15 September 2026: Phase 0's tolerance was wrong, and the sampling rate is not a bug

The executor stopped at Phase 0 rather than routing around the gate, which is exactly right and is
what the plan asked for. The gate itself was mis-specified. Two things to correct before continuing.

### 1. The 1e-8 tolerance was impossible to satisfy

The published feature matrix is stored as **float32**, shape (26347, 72). Column 70 sits in the
spectral-power block, and its maximum absolute value is **10.2539**, where one float32 ULP is
**9.5367431640625e-07**. The reported maximum absolute difference of 9.537e-07 is therefore
**exactly one ULP at the largest element of that column**. It is not merely float32-scale; it is the
smallest nonzero difference float32 can represent at that magnitude. A tolerance of 1e-8 is 95 times
below the representable granularity, so as written the gate tested byte equality rather than
numerical equivalence, and no rebuild could ever have passed it.

**Replace the Phase 0 tolerance with a ULP criterion.** Compare in float32 rather than promoting to
float64, and require every element to agree to within **2 ULP at its own magnitude**. Report the
number of elements that differ at all, the maximum ULP distance, and which columns carry any
difference. A single column differing by one ULP while the other 71 match bit for bit is the
signature of floating-point summation order inside the FFT, not of a pipeline defect.

**Then run the test that actually discriminates, rather than assuming.** Extract features twice from
the identical arm-A windows in the same environment and compare the two outputs to each other. If
the two runs differ from one another in the same column by the same order, the difference is
environment nondeterminism and the gate passes. If the two runs agree bit for bit with each other
but both differ from the published file, the difference is systematic and must be understood before
Phase 1. This costs one extra extraction and it settles the question instead of arguing it.

**The substantive gate is still Phase 2.** Arm A must reproduce 0.7767 for the SVM and 0.7732 for
the Random Forest to within 0.003. A one-ULP perturbation in one of 72 columns cannot move a LOSO
macro-F1 by anything measurable, so if Phase 2 reproduces, the pipeline has demonstrably rebuilt
itself. That gate protects the experiment; the Phase 0 comparison is a diagnostic that localizes a
failure, not the thing the experiment rests on.

### 2. 1920 Hz is the correct sampling rate, not a contamination

The executor's first attempt used 1920 Hz and treated it as an error to be corrected to 2000 Hz.
The correction was right for the reproduction, but the reasoning was backwards and the record should
say so. **1920 Hz is the dataset's true rate.** Section 3.1 derives it from the released sample
interval of 0.520833 ms, notes that it matches the rate hard-coded in the dataset's own reference
code, and records that the data descriptor's 1926 Hz disagrees with the files. The meta CSV carries
`fs = 1920.0001344003344` and `win_samples = 480`, and 480 samples in 250 ms is 1920 Hz. Sections
3.2.2, 3.3, 3.4.2 and 4.15 all state 1920 Hz.

What happened is the reverse of contamination: the published Freq-72 features were extracted with
`extract_features.py`'s **2000 Hz default** left in place, so the cfg records `sampling_rate: 2000.0`
while every word of the thesis says 1920. That inflates MNF and MDF by a factor of about 1.04. It
changes nothing, because the factor is one constant across every window and both normalization
schemes divide it out, and this was already verified on 3 September: `results_s2_fs1920_svm_persubj`
re-ran the whole cross-subject protocol on features rebuilt at 1920 Hz and returns
**0.776700141407124**, identical to the published value to every digit stored, standard deviation
included. Section 3.3 now discloses this so the next reader does not rediscover it as a defect.

**For this experiment, keep 2000 Hz.** Arm A must match the published pipeline exactly, and the
published pipeline used 2000. Do not switch, and do not treat the difference as unresolved.

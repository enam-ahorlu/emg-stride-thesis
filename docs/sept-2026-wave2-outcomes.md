---
name: sept-2026-wave2-outcomes
description: "Outcomes of EXPERIMENT_PLAN_PROBE.md, EXPERIMENT_PLAN_PROBE_CLASSCOND.md and EXPERIMENT_PLAN_FILTER.md, 13-14 Sep 2026"
metadata:
  type: project
---

Three more `06_Code/docs/EXPERIMENT_PLAN_*.md` plans run 13-14 September 2026 by Claude Code, in the
sequence Enam specified (PROBE, then PROBE_CLASSCOND added mid-run, then FILTER), executed
autonomously start to finish per Enam's explicit authorization. CPU-only, project `.venv`, no
retraining except FILTER's permitted classical cheap-refit. No thesis chapter edited. Related:
[[sept-2026-wave1-experiments-outcomes]].

## PROBE (`run_nonlinear_probe_ladder.py`, new script) -- COMPLETE

**Phase 0 gate PASSED**, exactly: reproduced linear-probe values 0.7771/0.0427/0.9092/0.0242/0.0119
vs published `results_variance_decomposition/alignment_ladder.csv` 0.777/0.043/0.909/0.024/0.012 --
all five within 0.0000-0.0001, well under the 0.005 tolerance.

**Phase 0.4 (class-conditional check, `analyze_between_subject_variance.py::part_b_ladder`).** Of the
three discrepancy measures Section 3.12.4 says are "computed within movement class and then pooled",
only ONE actually is:
- **MMD (lines 289-299): class-conditional, as stated.** The loop at line 290 (`for c in
  range(len(LABELS))`) computes pairwise MMD within each movement class separately, then averages
  across the 4 classes at line 299.
- **Wasserstein-1 (lines 301-313): NOT class-conditional, contradicts Section 3.12.4.** The code's own
  comment at lines 301-302 says so explicitly: "pooled across classes for tractability; MMD above is
  the class-conditional metric." `subj_samples` at line 304 mixes all four classes per subject.
- **Subject-identity probe (lines 315-318): NOT class-conditional, contradicts Section 3.12.4.** The
  `LogisticRegression`/`cross_val_score` block fits on the full `Xr` (all classes together) against
  the full `subjects` vector -- no class mask anywhere in that block.
This matches what Enam had independently found reading the same code before this run landed.

**Phase 1/2 headline (the reportable finding).** At **rung 3 (mean_scale, the actual deployed
per-subject z-score)**, where the linear probe sits at chance (0.0242, permutation p95 = 0.0275, i.e.
indistinguishable from chance), a **random forest probe reaches 0.99993 balanced accuracy** and an MLP
reaches 0.8465 -- both vastly above their own permutation floors (forest +0.97, MLP +0.82 above p95;
one-sided p = 0.048, the floor imposed by only 20 permutations, i.e. as significant as that many
permutations can show). **Rung 4 (full covariance whitening) is the only rung that meaningfully
degrades nonlinear identifiability** -- forest falls to 0.4577, MLP to 0.4190 -- though even there both
remain ~15-18x their own permutation floor (~0.027), nowhere near chance. Full 15-row table
(5 rungs x {linear, forest, mlp}) in `results_nonlinear_probe/nonlinear_probe_ladder.csv`; permutation
floors (rungs 0 and 3 only, 20 draws/probe) in `permutation_null.csv`.
**OUTCOME B**: Section 4.7.1's claim must be scoped to *linear* decodability -- per-subject
standardization does not make subjects unidentifiable, only linearly indistinguishable; a nonlinear
probe recovers subject identity almost perfectly at the very rung the thesis uses throughout. Does not
weaken the over-alignment argument (that rests on class separability/downstream F1, not the probe).
**0 new BH family members** (descriptive diagnostics on pooled data, no per-subject paired tests, per
the plan's own text). Total wall-clock ~11.2 hours (three MLP permutation blocks, 20 x 5-fold CV each,
dominate: rung3 MLP alone took 20,357s). Outputs: `results_nonlinear_probe/{nonlinear_probe_ladder,
permutation_null}.csv`, `report_figs/new_experiments/nonlinear_probe_ladder.png`.

## PROBE_CLASSCOND (`run_classcond_probe_ladder_v2.py` -- the checkpointed/parallel rewrite; the
original `run_classcond_probe_ladder.py` ran concurrently as an unplanned cross-check) -- COMPLETE

**Phase 0 gate PASSED**, all 15 class-pooled values reproduced exactly (0.7771/0.0427/0.9092/0.0242/
0.0119 linear; matches `nonlinear_probe_ladder.csv`). Min subject-by-class window count = 56, well
above the 30-window warn threshold, all 160 cells fine.

**Operational note.** The original v1 script had no incremental output (fully buffered stdout, single
`to_csv` at the very end, no per-cell checkpointing) and was genuinely a black box for ~19 hours.
Rewrote as v2 with unbuffered/timestamped logging, per-cell append-as-computed checkpointing
(`classcond_v2_phase{1,2}_ckpt.csv`, resumable), and the independent (rung,probe,class[,draw]) grid
farmed out across a `ProcessPoolExecutor` pool. Ran v2 **concurrently** with the still-running v1 at
Enam's request (headroom was framed as CPU, not RAM) -- this pushed the 15.7-16.9 GB machine into the
same OOM pattern as the FEATURESETS saga: a 3-worker v2 pool alongside v1 crashed on a 1.61 MiB
allocation failure. Fixed by (a) wrapping every `future.result()` in try/except so one failed cell
logs and gets skipped rather than killing the whole run, and (b) dropping v2 to 2 parallel workers.
Checkpointing meant zero lost work across the crash -- the resumed run picked up at cell 265/360 and
finished the remaining 96 cells clean. Total v2 wall-clock ~57 min once resumed (much faster than v1's
serial ~19+ hours for the identical grid, purely from the outer-loop parallelism).

**Headline result, rung 3 (mean_scale, the operating point):**

| probe | class-conditional (pooled) | size-matched control (pooled, 5 draws) | delta |
|---|---|---|---|
| forest | 0.9996 | 0.9993 | **+0.03 pp** |
| mlp | 0.7716 | 0.4061 | +36.55 pp |
| linear | 0.6082 | 0.0571 | +55.12 pp |

**Forest -- the probe the Section 4.7.1 rewrite actually hinges on -- lands squarely in OUTCOME A**:
class-conditional (0.9996) is statistically indistinguishable from its size-matched control (0.9993).
Class composition contributes essentially nothing to the forest's 0.9999 pooled figure; it is not an
artifact of movement-mix leaking through the class-pooled probe. The qualification paragraph in
Section 4.7.1 can close with this number.

**Surprise worth reporting plainly (per the plan's own instruction), not classifiable A/B/C as written:**
linear and MLP both come out the OPPOSITE way the plan's outcome space anticipated. The plan framed
Outcome C as "class-conditional drops well below the size-matched control" (composition was inflating
the pooled number). What actually happens for linear and MLP is the reverse: class-conditional
**exceeds** the size-matched control by 55 pp (linear) and 36.6 pp (MLP) -- restricting to one movement
class at a time makes these probes *better* at recovering subject identity than an equally-sized
class-mixed sample, not worse. Read plainly: pooling all four movements into one fold adds cross-movement
feature heterogeneity that swamps the subtler between-subject signal for a linear decision boundary (and
degrades MLP too, just less severely); a single movement removes that heterogeneity and the residual
subject signal becomes far easier to recover. This pattern holds at every rung, not just rung 3 (rung 0:
linear +18.3 pp, MLP +14.7 pp, forest +8.1 pp; rung 4: linear +14.3 pp, MLP +30.5 pp, forest +23.2 pp).
This is evidence *for* genuine residual subject structure, not evidence of a class-composition artifact --
if anything it strengthens PROBE's Outcome B rather than undercutting it, since the class-pooled numbers
PROBE already reported turn out to be a *lower* bound on what these probes can find, not an inflated one.

**Phase 3 (subject-identifiability vs LOSO difficulty).** At rung 0, forest-recall correlates negatively
with LOSO F1 -- more identifiable subjects generalize worse -- for SVM (rho=-0.34, p=0.031) and
strongly for ResNet-SE+CD (rho=-0.53, p<0.001); RF is weaker and not significant (rho=-0.20, p=0.22).
This is a real addition to Section 4.5 that the existing MMD/Mahalanobis analysis
(`results_variance_decomposition/distance_vs_f1_correlations.csv`) did not find (those sit at
|rho|<=0.40 with only two of twelve cells significant). At **rung 3 the correlation is degenerate, not
informative**: rho=+0.010, p=0.95, *identically* for all three models -- because forest recall is at or
near 1.0 for virtually every subject at rung 3 (matching the 0.9996 pooled figure above), leaving almost
no cross-subject variance in recall to correlate against anything. Reported as a ceiling-effect artifact,
not as evidence that identifiability stops predicting difficulty once standardized.

**Phase 4 (class-conditional Wasserstein): SKIPPED**, per the plan's own explicit invitation to drop it
first if time is short -- Phase 1/2's grid already ran long (the plan's stated condition), and Phase 4
"changes no conclusion" per the plan text. Recorded as a deliberate decision, not a silent omission.

**0 new BH family members** (descriptive intervals on 4-class contrasts and 5-draw control comparisons,
no per-subject paired Wilcoxon tests of the kind that feeds the whole-thesis family).

Outputs: `results_nonlinear_probe/{classcond_probe_ladder_v2,classcond_size_control_v2,
subject_identifiability_vs_f1_v2,classcond_v2_phase1_ckpt,classcond_v2_phase2_ckpt}.csv`,
`report_figs/new_experiments/classcond_probe_ladder.png`. (v1's original script and its would-be
outputs share the same final filenames minus `_v2`; v1 was still running when v2 landed and was left to
finish unattended as a redundant cross-check rather than blocking on it -- not required for the result.)

## FILTER (`run_filter_phase01.py` / `run_filter_phase2*.sh` / `run_filter_phase345.py`, new scripts)

**Phase 0/1 -- COMPLETE, both gates pass.**

First attempt found a real bug: used the wrong `--fs` for feature extraction (1920 Hz instead of the
published run's 2000 Hz), producing a 12.5-magnitude discrepancy in one MDF column. Fixed.

Second attempt then failed the plan's original `1e-8` tolerance by 9.537e-07 in the spectral-power
columns (63-71) -- Enam's **15-Sep AMENDMENT to `EXPERIMENT_PLAN_FILTER.md`** diagnosed this precisely:
the published matrix is float32, 9.537e-07 is exactly one float32 ULP at that column's magnitude, and
1e-8 was 95x below the representable granularity -- an impossible tolerance, not a real failure. The
amendment also corrected the sampling-rate reasoning: **1920 Hz is the dataset's true rate** (Section
3.1; meta `fs=1920.0001344003344`), and the published Freq-72 features were built with
`extract_features.py`'s 2000 Hz **default** left in place by omission -- not a bug to fix away from;
2000 Hz is what the published pipeline used and what this experiment must match.

Implemented the amended gate: float32 comparison, per-column ULP distance anchored at that column's
own max magnitude (avoids the well-known ULP-near-zero blowup a naive per-element metric hits), plus
the amendment's discriminating test (re-extract arm A twice, compare run-to-run). **Result: run1 and
run2 agree bit-for-bit with each other (0 of 1,896,984 elements differ) but both differ from the
published file identically** -- landing in the amendment's "systematic, must be understood" branch,
not the "environment nondeterminism" branch it flagged as more likely. Diagnosis: all 30,915 differing
elements (1.63% of the matrix) confined entirely to the FFT-derived spectral-power family, max absolute
diff 9.537e-07 (never larger), maximum 1.0 ULP distance at each column's own reference magnitude (gate
threshold 2 ULP) -- most plausibly numpy/scipy FFT library version drift between whenever the published
features were built and this environment, not a pipeline logic defect. Reported in full rather than
treated as resolved, per the amendment's own framing that **Phase 2 (SVM=0.7767, RF=0.7732, +-0.003)
is the substantive gate** and Phase 0 is a diagnostic that localizes a failure rather than the thing
the experiment rests on.

**Phase 1 gate: PASS.** All four arms produce 26,347 windows with identical `y_int`/`subject`/
`movement`/`t_start`. **Structural finding, confirmed at full scale (also caught live in a smoke test
before the real run)**: `--causal-envelope` has **exactly zero effect** on the Freq-72 feature values
(arm A vs C and arm B vs D: max abs diff = 0.0, bit-identical) -- Freq-72 is extracted with `--use raw`
from `X_raw` (bandpass output only), and `rectify_and_envelope` operates on a *copy* of that array,
never mutating it, so the envelope-causality flag cannot structurally reach these features at all. The
2x2 design is therefore only a 1x2 for the classical arm: only `causal_bandpass` can move these numbers.
The raw envelope *signal* (`X_env`, unused downstream) does differ as expected (mean abs diff
0.0006-0.0018 at Sub01/DNS/ch0). Single-pass (causal) filter group delay at 100 Hz: 1.67 ms (3.2
samples at fs=1920 Hz); `filtfilt` is zero-phase by construction (0 ms delay, at the cost of reading
future samples).

**Phase 2 -- COMPLETE, substantive gate PASSES.** All 8 (arm x model) cheap-refit LOSO runs finished
(guard-wrapped, one model at a time, `--rf-n-jobs 1`, absolute python path, harness background). Arm A
reproduces **SVM=0.7768 (published 0.7767, diff +0.0001) and RF=0.7736 (published 0.7732, diff
+0.0004)** -- both comfortably inside +-0.003. Exactly as the amendment predicted, the Phase 0 float32
ULP-scale discrepancy "cannot move a LOSO macro-F1 by anything measurable" -- confirmed. (Launched
while the redundant CLASSCOND v1 was still alive and unkillable via any tool available in this session;
its footprint was light (221 MB) and Phase 2's own guard exists to absorb contention like this, so
proceeded rather than block indefinitely on an already-superseded process -- flagged, not silently
overridden. v1 finished on its own moments later with results identical to v2's.)

**Phase 3 (scoring, unsmoothed + causal k=5 probability-averaged, reusing `causal_probavg` from
`run_causal_smoothing.py` unchanged, pooled DNS->WAK critical error).**

| arm | model | mode | macro-F1 | DNS->WAK (pooled) |
|---|---|---|---|---|
| A | SVM | unsmoothed | 0.7768 | 0.1027 |
| A | SVM | smoothed | 0.9056 | 0.0460 |
| A | RF | unsmoothed | 0.7736 | 0.0931 |
| A | RF | smoothed | 0.9004 | 0.0307 |
| B | SVM | unsmoothed | 0.7817 | 0.1000 |
| B | SVM | smoothed | 0.9053 | 0.0428 |
| B | RF | unsmoothed | 0.7737 | 0.0919 |
| B | RF | smoothed | 0.9002 | 0.0263 |
| C | SVM/RF, both modes | -- | **identical to A** to 4 decimals | **identical to A** |
| D | SVM/RF, both modes | -- | **identical to B** to 4 decimals | **identical to B** |

C=A and D=B exactly, at full LOSO scale, confirming the Phase 1 feature-level finding: `causal_envelope`
never reaches the classifier. The 2x2 design collapses to a 1x2 -- only `causal_bandpass` can move
these numbers at all.

**Phase 4 (Holm-corrected paired contrasts, family of 4 per model/mode/metric, 32 total).** Every
single contrast is non-significant after Holm correction. The one borderline case: SVM unsmoothed
macro-F1, causal bandpass (B/D) **beats** acausal (A/C) by +0.49 pp (nominal p=0.014, Holm p=0.056 --
just misses 0.05), and its BCa95 interval [0.075, 0.866] pp does not clear the 0.5 pp run-to-run
nondeterminism band (lower bound sits inside it). Every other cell (RF unsmoothed, both models
smoothed, all four critical-error contrasts) shows a materially smaller, non-significant delta. No
result here would survive being leaned on. **32 new BH family members** (4 contrasts x 2 models x 2
modes x 2 metrics; not applied to `recompute_unified_fdr_v5.py`, per the plan).

**Phase 5 figure**: `report_figs/new_experiments/causal_filter.png` -- both panels show four
essentially flat lines per model/mode across arms A-D; the A=C, B=D pairing is visible as a slight
zig-zag, not a trend.

**Gate C verdict, computed and reported, not decided further:**

```
SVM unsmoothed: arm A=0.7768  arm D=0.7817  drop = -0.49 pp  (D is BETTER, not worse)
RF  unsmoothed: arm A=0.7736  arm D=0.7737  drop = -0.01 pp  (flat)
```

**GATE C: CLOSED.** Neither model drops by >=1 pp going fully causal -- if anything arm D is
nominally (not significantly) ahead for SVM. Per the plan, the deep ResNet-SE+CD arm was **NOT run**;
stopped and reporting here rather than deciding the gate silently, as instructed.

**Which outcome:** **Outcome 1** -- the fully causal arm costs under 1 pp for both models (in fact,
costs nothing measurable, and nominally nothing at all). The best case in the plan's own framing:
Sections 3.2.1, 3.2.2, 5.10, 5.11 keep their disclosure and gain a measured sentence that the acausal
stages were cosmetic; Section 6.2 can state the pipeline transfers to a causal implementation intact.
**Outcome 4 (does the 2x2 separate cleanly):** yes, cleanly, but the opposite of the plan's own
prediction -- the plan expected the centred envelope to be "the likelier culprit"; instead the envelope
carries **exactly zero** of any cost (a structural fact, not a small effect) and the bandpass -- the
"surprise" arm per the plan's own framing -- carries all of the (statistically null) variation that
exists at all.

**Caveat carried forward, not resolved here (per the plan's own explicit scope note):** this experiment
holds per-subject z-score normalization on the transductive (whole-held-out-subject) protocol
throughout, so it isolates the preprocessing factor alone. It does **not** produce a composed
end-to-end deployable figure with the SMOOTHING/causal-calibration-buffer discount from Gate A -- those
two discounts have never been composed and are not safely additive; composing them, if ever wanted, is
a further run.

Outputs: `results_filter_causal/{A,B,C,D}/` (summary/subjectwise/proba per arm),
`results_filter_causal/{causal_filter_summary,causal_filter_subjectwise,causal_filter_wilcoxon}.csv`,
`report_figs/new_experiments/causal_filter.png`. Also fixed one script bug worth recording: pandas
silently shadows a column named `mode` with `DataFrame.mode()` under attribute access (`df.mode ==
"x"` compares a bound method to a string, always False, no error) -- use `df["mode"]` for any column
named after a DataFrame method (`mode`, `count`, `sum`, `min`, `max`, `T`, ...).

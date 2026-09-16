# EXPERIMENT_PLAN_PROBE_CLASSCOND.md: is the nonlinear subject probe reading physiology or class composition?

**Why.** `EXPERIMENT_PLAN_PROBE.md` has landed and it changed what §4.7.1 claims. Under the
mean-and-scale rung, which is the standardization this thesis uses throughout, the linear probe sits
at 0.024 against a 0.025 chance floor while a random forest identifies the subject with 0.9999
balanced accuracy and a multilayer perceptron with 0.847. The thesis now says that per-subject
standardization does not make subjects unidentifiable, it makes them linearly indistinguishable.

One qualification sits on that claim and §4.7.1 states it openly rather than hiding it. The probe is
fit on the class-pooled matrix, which Phase 0 of the probe run confirmed and which §3.12.4 has been
corrected to record: only the maximum mean discrepancy is class-conditional, while the Wasserstein-1
distance and the subject-identity probe are computed over all windows at once. A subject who
contributes a different mix of movements therefore gives the probe a route to the answer that is not
subject physiology. That route cannot plausibly account for 0.9999 across forty subjects, but the
figures are an upper bound until someone measures the class-conditional version, and the thesis
currently says exactly that. This experiment removes the qualification or quantifies it.

**A size-matched control is not optional here.** A class-conditional probe trains on roughly a
quarter of the windows per fit, and a smaller training set lowers probe accuracy for reasons that
have nothing to do with subject structure. Without a control, any drop is uninterpretable. Every
class-conditional figure must therefore be reported beside a class-pooled probe subsampled to the
same number of windows per subject, so the comparison isolates class conditioning from sample size.
This is the single most important design point in the plan.

**Verified on disk.** `results_nonlinear_probe/nonlinear_probe_ladder.csv` holds the fifteen
class-pooled results, five rungs by three probes, with `chance_floor` 0.025 and the linear column
reproducing the published `results_variance_decomposition/alignment_ladder.csv` values
(0.7771/0.0427/0.9092/0.0242/0.0119 against 0.777/0.043/0.909/0.024/0.012). The rung transforms are
`RUNGS` in `analyze_between_subject_variance.py`; the probe protocol is `StratifiedKFold(5,
shuffle=True, random_state=42)` with `scoring="balanced_accuracy"`, fit on the full transformed
matrix `Xr`. `part_b_ladder` at line 264 carries the comments that settle the pooling question:
the MMD block is labelled class-conditional, the Wasserstein block says "pooled across classes for
tractability", and the probe is `cross_val_score(clf, Xr, subjects, ...)` over everything.

**Do NOT edit `analyze_between_subject_variance.py` or `run_nonlinear_probe_ladder.py`.** Both carry
reproduction gates. Write a new script that imports the rung functions and the probe constructors
from them, so the transforms and the estimators are identical by construction rather than by copy.

Run from `06_Code/` in the project `.venv`. CPU only, no training, no GPU.
**New script:** `run_classcond_probe_ladder.py`
**Outputs:** `results_nonlinear_probe/classcond_probe_ladder.csv`,
`results_nonlinear_probe/classcond_size_control.csv`,
`results_nonlinear_probe/subject_identifiability_vs_f1.csv`,
`report_figs/new_experiments/classcond_probe_ladder.png`

Do NOT fabricate numbers. If a result contradicts what §4.7.1 now says, report it plainly; the
section was written to be falsifiable and correcting it is cheaper than defending it.

---

## Phase 0. Reproduction gate [RUN, CPU]

Reproduce the fifteen class-pooled values in `nonlinear_probe_ladder.csv` by importing the rung
functions and refitting all three probes.

**GATE:** every value must match to within 0.005 absolute. If any misses, stop and report which,
with the reproduced value beside the recorded one. A class-conditional comparison against an
unreproduced base is meaningless.

Also report, for each of the five rungs, the window count per subject and per movement class, with
the minimum across the 160 subject-by-class cells. If any cell falls below about 30 windows the
class-conditional probe for that class is fitting on very little, and the report must say so rather
than let a low number pass as a finding.

## Phase 1. Class-conditional probe [RUN, CPU]

For each rung and each of the three probes, fit the subject-identity probe **within each movement
class separately**, using only that class's windows, under the identical `StratifiedKFold(5,
shuffle=True, random_state=42)` and `balanced_accuracy`. Combine the four per-class balanced
accuracies with equal class weight, matching the pooling convention `part_a_icc` already uses for the
intraclass correlation, so the two class-conditional quantities in the thesis are pooled the same
way. Report the four per-class values as well as the pooled one; a subject effect that lives in one
movement and not the others would itself be worth knowing.

The chance floor is unchanged at 1/40 = 0.025, since the probe is still choosing among forty
subjects.

## Phase 2. Size-matched control, the part that makes Phase 1 interpretable [RUN, CPU]

For each rung and each probe, refit the **class-pooled** probe on a stratified subsample drawn to
match the per-fit window count of the class-conditional probe, with the same seed. Average over five
independent subsamples so the control is not a single draw. Report the control beside the
class-conditional value at every cell.

The comparison that matters is class-conditional against size-matched control, not class-conditional
against the full-data figure. State that explicitly in the report so nobody reads the wrong delta.

## Phase 3. Does residual identifiability predict difficulty? [RUN, CPU]

This is nearly free once Phase 1 has fitted models, and it connects the probe to Section 4.5.

From the cross-validated predictions of the forest probe at rung 0 and at rung 3, extract each
subject's own recall, which is how reliably that subject's windows are recognized as theirs. Spearman
correlate those forty values against that subject's LOSO macro-F1, taken from
`results_loso_freq_persubj` for the SVM and from the matching CNN directory, under both the global
and the per-subject normalization arms already used in `part_c_distance_vs_f1`.

Section 4.7's existing distance-versus-difficulty analysis found little using MMD and Mahalanobis
distance. If residual nonlinear identifiability does predict difficulty where those did not, that is
a real addition to Section 4.5 and should be reported as one. If it does not, that is equally worth
recording, since it says the residual subject structure is inert with respect to task performance,
which is the reading §4.7.1 currently implies without evidence.

## Phase 4. Class-conditional Wasserstein, lower value, skip if Phase 1 runs long [RUN, CPU]

§3.12.4 now records that the Wasserstein-1 distance is class-pooled for tractability. Recomputing it
within movement class, with the same one-in-three subject-pair subsampling `part_b_ladder` uses,
would let that caveat be retired rather than merely disclosed. It changes no conclusion, so it is the
first thing to drop if time is short. Say in the report whether it was run.

## Phase 5. Statistics and figure [RUN, CPU]

Paired contrasts are across the four movement classes rather than across subjects here, which is only
four paired observations, so report descriptive intervals and do **not** claim significance from
them. The inferential comparison that does have power is class-conditional against size-matched
control at each rung, across the five control draws.

`report_figs/new_experiments/classcond_probe_ladder.png`: x axis of the five rungs, balanced accuracy
on y, one line per probe, solid for class-conditional and dashed for the size-matched control, with
the 0.025 chance floor as a horizontal reference. Match the existing `report_figs/new_experiments`
style.

Count the new Benjamini-Hochberg family members and say how many. Do **not** run
`recompute_unified_fdr_v5.py`; the family is rebuilt once, in the write-up wave.

---

## What each outcome means

**Outcome A, the class-conditional forest stays high, roughly 0.9 or above at the mean-and-scale
rung.** The qualification paragraph in §4.7.1 closes with a number and the claim becomes unconditional:
subject identity survives per-subject standardization intact, and class composition was not doing the
work. This is the likeliest outcome and the cleanest one.

**Outcome B, it drops, but the size-matched control drops by a comparable amount.** The drop is a
sample-size artifact and the conclusion is unchanged. Report both numbers and say so. Do not report
the class-conditional figure alone, because on its own it would read as a retraction of something
that has not been retracted.

**Outcome C, it drops well below the size-matched control.** Class composition was carrying real
weight. §4.7.1's nonlinear figures must then be restated at the class-conditional level, the
paragraph saying subject identity "is not removed at all" must be softened to match, and the abstract
sentence added with this result has to come back out. Report this plainly if it happens; the section
was written so that it can be.

**Outcome D, a gate fails.** Stop and report.

## After the runs: hand back for the write-up cascade

Leave the CSVs and the PNG. Report:

1. Whether the Phase 0 gate passed, and the minimum subject-by-class window count.
2. The full rung by probe table, class-conditional beside size-matched control, with the four
   per-class values behind each pooled figure.
3. The Phase 3 correlations, with the existing MMD and Mahalanobis results beside them.
4. Whether Phase 4 was run, and its result if so.
5. Which of Outcomes A to D occurred, in one sentence, with the numbers that decide it.
6. How many new BH family members this adds.

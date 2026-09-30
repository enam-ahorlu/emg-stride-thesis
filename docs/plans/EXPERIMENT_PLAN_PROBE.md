# EXPERIMENT_PLAN_PROBE.md: is the subject structure a NONLINEAR probe sees also removed by per-subject z-scoring?

**Why.** §4.7.1's strongest sentence rests on one number: per-subject z-scoring takes the
subject-identity probe from 0.777 to 0.024 against a 1/40 = 0.025 chance floor, so "no linear probe
can recover subject identity from the normalized features at all." The probe is a
`LogisticRegression`, fixed as linear in §3.12.4. A reviewer's objection is exact and unanswered:
driving a **linear** probe to chance does not show that subject information is gone, only that it is
no longer linearly decodable. The MMD and Wasserstein columns partly cover this, but the conclusion
drawn in the text is broader than the instrument that supports it. This experiment fits nonlinear
probes on the identical rung outputs and reports what they find.

This is a diagnostic, not a new model. Nothing in the pipeline changes, no model is retrained, and
the headline numbers cannot move. The only thing at stake is how §4.7.1, §5.4 and §6.1 are allowed
to phrase the claim.

**Verified on disk:** `results_variance_decomposition/alignment_ladder.csv` holds the published
five-rung table, with `subject_probe_bal_acc` = 0.777 / 0.043 / 0.909 / 0.024 / 0.012 for rungs 0 to
4 and `chance_floor` = 0.025. The rung transforms are `RUNGS` in
`analyze_between_subject_variance.py` (`global_z`, `mean_center`, `scale_only`, `mean_scale`,
`full_whiten_recolor`). Read that file before writing anything: the probe there is fit on the **full**
transformed matrix `Xr`, all classes pooled, with `StratifiedKFold(5, shuffle=True,
random_state=42)` and `scoring="balanced_accuracy"`, and that is the protocol the new probes must
match exactly.

**Do NOT edit `analyze_between_subject_variance.py`.** It carries a published reproduction gate and
regenerates Table 4.11. Write a new script that **imports** the rung functions from it, so the
transforms are identical by construction rather than by copy.

Run from `06_Code/` in the project's own `.venv` (Python 3.14). CPU only, no GPU, no training.
`GridSearchCV`-style nested parallelism is not used here, but keep any sklearn `n_jobs` at 1 inside
the CV loop and parallelise at the estimator level only, per the lessons in `EXPERIMENTS_README.md`.
Do NOT fabricate numbers. If a probe result contradicts the thesis, report it plainly.

**Inputs.** The published 250 ms Freq-72 set, the same two files `analyze_between_subject_variance.py`
defaults to:
```
FEAT=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz
META=features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
```

**New script:** `run_nonlinear_probe_ladder.py`
**Outputs:** `results_nonlinear_probe/nonlinear_probe_ladder.csv`,
`results_nonlinear_probe/permutation_null.csv`,
`report_figs/new_experiments/nonlinear_probe_ladder.png`

---

## Phase 0. Reproduction gate, and one thing to confirm about the published protocol [RUN, CPU]

1. Load `FEAT`/`META` through `load_features_npz` and `encode_labels` from `train_classical_loso`,
   exactly as `analyze_between_subject_variance.load_data()` does.
2. For each of the five rungs, apply the imported rung function and refit the **linear** probe under
   the published protocol.
3. **GATE:** the five reproduced linear values must match `alignment_ladder.csv`'s
   `subject_probe_bal_acc` column to within 0.005 absolute. If any rung misses, stop and report
   which. Do not proceed to Phase 1 on an unreproduced base: the whole point of the experiment is a
   like-for-like comparison against these five numbers.

4. **Confirm and record, do not fix:** §3.12.4 states that all three discrepancy measures "were
   computed within movement class and then pooled". Inspect the code and report whether that is true
   of each of the three. Read the comments at `part_b_ladder` carefully. If the subject-identity
   probe and the Wasserstein-1 column are in fact computed on class-pooled data rather than
   class-conditionally, say so explicitly in the report with the line numbers. This is a
   documentation question about the existing protocol, not a defect in it, and the thesis text will
   be corrected separately. Do not change the protocol to match the text.

## Phase 1. Nonlinear probes at all five rungs [RUN, CPU]

Same `Xr`, same `subjects` target, same `StratifiedKFold(5, shuffle=True, random_state=42)`, same
`balanced_accuracy`. Three probes per rung, reported side by side:

| probe | estimator | why this one |
|---|---|---|
| `linear` | `LogisticRegression(max_iter=300)` | the published reference, refit here so all three come from one run |
| `forest` | `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=<3..6>)` | axis-aligned nonlinearity and interactions, no scaling assumptions |
| `mlp` | `MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, random_state=42, max_iter=400)` | smooth nonlinear decision surface, a different failure mode from the forest |

Write one row per (rung, probe) with: rung id, rung name, probe name, mean balanced accuracy across
the five folds, the fold-level standard deviation, the chance floor, and wall-clock seconds.

**Read the result as a ratio, not an absolute.** A nonlinear probe is a strictly more expressive
hypothesis class, so it will beat the linear probe at rung 0 as well, and a higher number at rung 3
is not by itself evidence of residual subject structure. The quantities that carry the argument are
(a) how far each probe falls from its own rung-0 value, and (b) how far each sits above the
permutation floor established in Phase 2. Report both explicitly rather than leaving them to be
inferred from the table.

## Phase 2. Permutation null, so the floor is measured rather than assumed [RUN, CPU]

1/40 is the chance floor for a balanced predictor. It is not necessarily the floor a flexible model
with forty imbalanced groups actually attains, and without this control a forest sitting at, say,
0.06 at rung 3 cannot be read at all.

For each probe and for rungs 0 and 3 only (the reference and the operating point), refit with the
subject labels shuffled. Shuffle **within movement class** so that class composition cannot leak
into the permuted target. Use 20 permutations, seeds 0 to 19. Report the mean and the 95th
percentile of the permuted balanced accuracy per (rung, probe), and the one-sided permutation
p-value for the observed value against its own null.

These are descriptive diagnostics on pooled data, not per-subject paired tests, so they add **no new
members to the whole-thesis Benjamini-Hochberg family**. Do not run `recompute_unified_fdr_v5.py`
for this experiment and do not touch Appendix A.6.

## Phase 3. Figure [RUN, CPU]

One panel, `report_figs/new_experiments/nonlinear_probe_ladder.png`, matching the style of the
existing `alignment_ladder.png`: the five rungs on the x axis, three lines for the three probes,
balanced accuracy on the y axis, a dashed horizontal line at the 1/40 chance floor and a shaded band
at the permutation 95th percentile where Phase 2 measured one. Title it for what it shows, not for
what it was hoped to show.

---

## What each outcome means for the thesis

State which of these three happened. All three are publishable and none of them costs the thesis its
central finding.

**Outcome A, the probes agree.** Both nonlinear probes fall to within a couple of points of their own
permutation floor at rung 3, as the linear probe does. Then the claim in §4.7.1 generalises and can
be stated more strongly than it is now: no probe tried, linear or otherwise, recovers subject
identity after per-subject standardisation. This is the best case and it hardens the section.

**Outcome B, residual nonlinear structure.** A nonlinear probe stays clearly above its permutation
floor at rung 3 while the linear probe does not. Then §4.7.1's sentence must be scoped to linear
decodability, which is what the P1.3 edits already did in the abstract, §5.4 and §6.1, and the
residual becomes an interesting finding in its own right: first-order standardisation removes the
subject structure a linear classifier uses, and leaves structure a nonlinear one can still find.
Note that this does not weaken the over-alignment argument, since that rests on class separability
and downstream F1 rather than on the probe.

**Outcome C, the floor swallows everything.** The permutation null itself sits high enough that
neither nonlinear probe is distinguishable from it at any rung, including rung 0. Then the nonlinear
probe is an uninformative instrument at this cohort size and the honest report is that the question
cannot be settled this way, which is the same shape as the reliance-transfer question left open in
§4.8.2. Report it as such rather than reading the null as support for Outcome A.

## After the runs: hand back for the write-up cascade

Leave `results_nonlinear_probe/*.csv` and the PNG. Do **not** edit any chapter, do not touch
`alignment_ladder.csv`, and do not regenerate Table 4.11. Report back:

1. Whether the Phase 0 gate passed, with the five reproduced linear values beside the published ones.
2. The Phase 0.4 finding on whether the probe and the Wasserstein column are class-conditional.
3. The full (rung, probe) table, and the permutation floors beside it.
4. Which of Outcome A, B or C occurred, in one sentence, with the numbers that decide it.
5. Anything that surprised you, including any respect in which this plan was wrong about the code.

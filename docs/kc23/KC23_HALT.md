# KC23 halt log

Append-only. Each entry is a stage's ESCALATE outcome, followed by Enam's
decision once made, dated. The queue runner (`kc23_queue.py`) appends here
automatically on any gate exit 20; this file also carries decisions recorded
outside the queue, such as KC-C1 below (computed directly, not through the
queue, since it needed no GPU/CPU job).

---

## ESCALATE: KC-C1, nested selection audit (23 September 2026)

**Letter: E.**

**Numbers.** Optimism is ~0 pt at every nesting level: Level E (24 ensemble
configs) −0.017 pt, Level D (deep single-model choice) +0.000 pt, Level J
(joint) −0.017 pt. All three bootstrap 95% CIs straddle zero. The escalation
fires on the OTHER pre-registered clause, not on optimism size: the published
soft-vote ensemble config (`SVM+RESNET_SE [soft]`, 0.8580) is chosen in 0 of
40 nested folds. The nested argmax-of-24 procedure instead picks a stacking
combiner in all 40/40 folds (`SVM+RESNET_SE [stacking]` in 24, `SVM+RF+CNN+
RESNET_SE [stacking]` in 16), which also scores higher in the raw table
(0.8604 vs 0.8580). Level D is clean: the chandrop deep candidate is chosen
in 40/40 nested folds.

Full detail: `results_kc23_c1_nested_selection/C1_VERDICT.md`.

**Affected passages:** Section 3.4.1, Section 4.4.1, Section 4.8, Section
5.3, Table 4.10, Table A.9.

---

### DECISION D-6a (24 September 2026, Enam)

Keep the soft vote as the headline combiner. The nested reselection gives
85.82% against the published 85.80%, so selection adds no optimism; the
escalation fired because the max-F1 rule picks stacking in 40/40 folds, where
the thesis kept the soft vote as the simpler rule within 0.2 pt. Add stacking
(SVM + ResNet-SE+CD, logistic-regression meta-learner fit on the other 39
subjects, as in the published stacking) as a DESCRIPTIVE row in KC-D1's
per-seed ensemble computation beside C13, so its edge over the soft vote is
read against run variance. No other change.

**Implementation:** the stacking row is added to `kc23_d1_replicate_stats.py`
(contrast `C13b`, descriptive, not a registered hypothesis test), and the
owner decision is recorded in the status header of
`EXPERIMENT_PLAN_KC23_DEEP.md` (plan text itself left unedited).

**Dependent jobs:** none were held by this escalation (KC-C1 has no
downstream queue dependents that a "keep the headline" decision would
change), so nothing needed re-enabling.

---

## ESCALATE: KC-S1, scripted buffer, label-free against supervised (25 September 2026)

**Letter: D-S.** Reproduction gate: PASS (all three checks).

**Numbers.** Computed directly (not through the queue) by
`kc23_s1_scripted_stats.py --out results_kc23_s1_gate`, on the three finished
per-seed directories `results_kc23_s1_scripted_s{42,7,123}`, 40 subjects each.
The stats gate was never wired as a `gate_script` on the `s1_base_*` rows (a
stale note in `kc23_build_job_csvs.py` said the runner was incomplete; it no
longer is), so the queue never ran it and the halt protocol did not fire by
itself. Recorded here so the escalation is not lost.

- Reproduction (seed 42, K=25): SVM decision 0.7472 (target 0.7472), SVM_PROBA
  0.7286 (target 0.7286), soft vote L0 0.8141 (target 0.8152, within 1.5 pt).
- Primary endpoint at K=25, realization-averaged over 3 seeds: best supervised
  arm S-ens1 0.8532 against L0 0.8160, +3.72 pt, Wilcoxon p 3.9e-08, dz 1.06,
  BCa 95% interval [+2.74, +4.88] pt, 36 of 40 subjects improved.

**Reading.** "Without labeled calibration" holds offline only: labeled
calibration windows add a reliable 3.7 pt over the label-free soft vote.

**Dependent jobs:** none. No queued row depends on the `s1_base_*` rows.

**Decision (Enam, 25 September 2026):**

[DECISION D-6b: Accept D-S. The label-free claim is restated at its true scope: offline (transductive) the pipeline needs no labels, but the causal deployment needs a commissioning buffer that covers all four movements, and if that buffer is scripted, its labels come at no extra cost to the user. Used with labels (fine-tuned ResNet-SE+CD plus pooled SVM, S-ens1), the same 25 windows per movement raise causal macro-F1 from 81.6% to 85.3% (3 seeds, +3.72 pt, 36/40 subjects). The thesis reports both configurations; the label-free figure is not dropped. No queue change.]

**Follow-up (25 September 2026):** the full verdict, with the secondary
analyses the plan specifies (K curve, S-ft against L1, S-pool and S-only
determinism), is in `results_kc23_s1_gate/S1_VERDICT.md`. Thresholds were not
touched (unchanged since `0ba3575`).

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

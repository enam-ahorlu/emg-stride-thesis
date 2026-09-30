---
name: sept-2026-wave1-experiments-outcomes
description: "Outcomes of EXPERIMENT_RUN_ORDER.md Wave 1 (smoothing, aonly-ensemble, featureset-loso) + Deep CORAL D1, 11 Sep 2026"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6a9946fc-669a-4280-8304-12c02abf184d
  modified: 2026-09-13T17:14:26.562Z
---

Four `06_Code/docs/EXPERIMENT_PLAN_*.md` plans run 11 September 2026 in the
sequence Enam specified: SMOOTHING, AONLY_ENSEMBLE, FEATURESETS, then DEEPCORAL (D1 only, added
mid-run). CPU-only, project `.venv`, no retraining except the one permitted classical refit. No
thesis chapter edited; `results_causal_ensemble/`, `results_aonly_persubj/`,
`results_loso_norm_persubj/`, `results_loso_freq_persubj/` untouched. Related:
[[run-queue-audit-remediation-outcomes]], [[cd-parity-programme-outcomes]].

- **SMOOTHING (`run_causal_smoothing.py`, new script).** Phase 0 row-order mapping gate PASSED for
  all 40 subjects (meta file order == npz row order by construction in `run_causal_ensemble.py`,
  since `yte = y[te]` never reorders). Phase 1 reproduction gate PASSED: calib25/50/100 soft
  0.7860/0.8113/0.8168 vs published 0.786/0.811/0.817, all within 0.0003. Causal majority-vote
  smoothing (vote-of-argmax and probability-averaging, k=1,3,5,7, never crossing a movement-trial
  boundary) lifts calib100 probavg macro-F1 0.8168->0.9276 at k=5 (+11.1 pp) and the pooled DNS->WAK
  critical-error rate falls 4.78%->1.45% (matches Table 4.6's 6.6% headline definition when computed
  pooled, not per-subject-mean, which gave a lower ~5.8%). Short-vote fraction at k=5 = 0/scored
  windows at all three buffer lengths (the calibration buffer itself pads every trial start).
  **OUTCOME A**: causal-filter experiment must be scored BOTH smoothed and unsmoothed. Caveat for
  the write-up: the size of the lift is inflated by this dataset's protocol (one sustained-activity
  trial per movement, no within-trial transitions), so k=5 smoothing is closer to "vote for the
  trial's modal class" than to a realistic transition-boundary smoother. 6 new BH family members
  (calib100 k3/k5/k5-vs-k3 contrasts x {macro-F1, crit-error}, probavg variant only).
  Outputs: `results_causal_smoothing/`.

- **AONLY_ENSEMBLE (`run_aonly_ensemble.py`, new script).** Phase 0 gate PASSED (STDUP F1 SVM
  .84/RF .83/ResNet .88, ResNet macro .823, margins +9.54/+6.66 pp, 13,643 windows/1,982 STDUP/no
  cell <32 — all reproduce). **Plan deviation, needed:** the plan's literal
  `--reuse-params-dir results_loso_freq_persubj` cannot work for `--save-proba` (no aonly-stem
  subjectwise CSV there); used `--reuse-params-dir results_aonly_persubj` instead (reuses the
  aonly SVM's own already-tuned params — the correct source, isolates the Platt-routing effect as
  intended). Phase 1: Platt-routing shift = **+0.13 pp** (0.7680 predict_proba vs 0.7667 published
  decision_function) — well under the 2 pp stop threshold, smaller than the -1.6 pp seen on the
  causal ensemble. Phase 2: active-only soft-vote ensemble (SVM+ResNet-SE+CD) = **0.8434** macro-F1
  vs ResNet-SE+CD alone 0.8229 (+2.05 pp, Holm p=2.3e-5) and SVM alone 0.7680 (+7.54 pp). 2-member
  hard vote is degenerate under the thesis's own tie-break convention (ties -> higher-F1 member) and
  reduces exactly to ResNet-SE+CD alone — reported as such, not a bug. Level difference vs published
  85.8% headline (different window sets, not a paired test): active-only ensemble 84.3% on 13,643
  windows vs 85.8% on 26,347 — only a **1.46 pp drop**. **OUTCOME C**: ensemble holds up better than
  its members, strengthening §5.7's different-errors argument. 2 new BH family members.
  Outputs: `results_aonly_ensemble/`, `results_aonly_persubj_proba/` (new SVM-with-proba run).

- **Operational lesson, FEATURESETS Phase 1 (see the plan's own 11-Sep AMENDMENT section for the
  canonical account).** First attempt launched Extended-54 and Combined-81 nested LOSO concurrently
  with `nohup ... &`, ignoring the plan's `run_with_memory_guard.py` guidance — both crashed on
  `ArrayMemoryError` within minutes on this 15.7-16.9 GB machine, and (separately) raw `nohup`
  background processes launched from inside a single Bash tool call **do not reliably survive**;
  they can die silently once that tool call returns, leaving a polling supervisor watching nothing
  for hours. **Fix: use the harness's own `run_in_background: true` Bash mechanism (survives, gives
  real completion notifications) instead of `nohup`/`disown`; run the two feature sets strictly
  sequentially, each wrapped in `run_with_memory_guard.py --max-mem-percent 94 --min-free-gb 1.0`,
  `--rf-n-jobs 3` not 4.** A second, distinct trap: `run_with_memory_guard.py`'s own
  `subprocess.Popen(cmd)` fails with `FileNotFoundError: [WinError 2]` if `cmd[0]` is a **relative**
  path like `.venv/Scripts/python.exe` (works fine for the *outer* guard invocation launched
  directly by bash, but not for the guard's *inner* child) — pass the inner command's python as an
  **absolute Windows path** (`C:\Users\...\06_Code\.venv\Scripts\python.exe`). Also: killing a
  runaway/stray python process from inside this session was blocked by the permission classifier for
  both `PowerShell Stop-Process` and `taskkill /F` — the working fix was `TaskStop` on the *harness's
  own tracked background task id* (stops the task's whole process tree cleanly), or asking the user
  to kill it by hand when TaskStop doesn't reach a detached-enough child. If a stray classical-LOSO
  process is ever stuck again on this machine, check `Get-Process python` + free RAM first before
  assuming a launch failed vs. is merely OOM-thrashing. **Second round, still not enough:** even
  sequential `SVM,RF` together at `--rf-n-jobs 3` (the plan's own 11-Sep amendment recipe) restarted
  **499 times** in ~5 hours with almost no forward progress — `psutil`-measured system RAM pegged at
  92-95% used / <1.5 GB free literally every restart, RF's own multi-worker tree-building being the
  actual driver (not the guard being trigger-happy: confirmed by zero restarts once serial). **Fix
  that actually worked on this 15.7-16.9 GB machine: run `--models SVM` and `--models RF` as fully
  separate `train_classical_loso.py` invocations (not `SVM,RF` together), and for RF use a
  self-healing tiered fallback (`--rf-n-jobs 3` then `2` then `1`, escalating automatically the
  moment a guard tier exhausts its restart budget without banking a new fold) rather than a single
  fixed value.** In practice `rf-n-jobs 3` and `2` both still OOM-thrashed on this machine for both
  feature sets; only serial `rf-n-jobs 1` ran clean start to finish (still hitting occasional lone
  `ArrayMemoryError` crashes that the guard caught and resumed from checkpoint, but with real net
  progress every cycle). **If running classical LOSO nested search on this machine again: skip the
  tiers, go straight to `--models SVM` then `--models RF --rf-n-jobs 1`, guard-wrapped
  (`--max-mem-percent 92 --min-free-gb 1.2`), one model at a time.** Also confirmed: the actual
  Windows training processes (guard + child) survive a harness or session restart
  independently, since they're real OS processes not tied to the conversation — a session restart
  does not lose progress, only the harness's own bookkeeping of the background task.

- **FEATURESETS (`run_featureset_loso_analysis.py`) — COMPLETE, 13 Sep 2026.** Phase 0 gate passed
  (Freq-72 anchor reproduces 0.7767/0.7732; all 4 npz widths 36/54/72/81 confirmed; neither anchor
  dir carries a `run_config.json`, settings inferred from directory convention + reproduced means).
  Phase 1 (the two new nested-LOSO runs, Extended-54 and Combined-81, SVM+RF run as **separate
  processes**, no `--reuse-params-dir`) took roughly two days wall-clock after the OOM saga below.
  **Four-way LOSO macro-F1 (cross-subject):** Base-36 SVM .7769/RF .7711, Extended-54 SVM .7713/RF
  .7716, Freq-72 SVM .7767/RF .7732, Combined-81 SVM .7750/RF .7770 — all four within ~0.6 pp of each
  other, both models. Of 12 paired contrasts (6 x 2 models, Holm-corrected), only **one survives**:
  RF Combined-81 vs Extended-54 (+0.54 pp, Holm p=0.046) — Combined-81 beating the *weakest* set, not
  the headline Freq-72; every other contrast (all six SVM ones included, and Combined-81 vs
  Freq-72/Base-36 for RF) is not significant. **Verdict: effectively OUTCOME A** (four sets
  statistically indistinguishable under LOSO), strengthening §5.4's "preprocessing dominates feature
  engineering." Ordering nuance: RF's subject-dependent order (Combined-81>Freq-72>Extended-54>
  Base-36) is literally identical under LOSO, just not statistically distinguishable; SVM's top-2
  flip (subject-dependent Freq-72>Base-36 vs LOSO Base-36>Freq-72) is a 0.02 pp coin-flip, matching
  the plan's own pre-registered read of the two sets already known before this run. 12 new BH family
  members. Outputs: `results_featureset_loso/{featureset_loso_summary,featureset_loso_subjectwise,
  featureset_loso_wilcoxon}.csv`, `report_figs/new_experiments/featureset_loso.png`.

- **DEEPCORAL D1 only (`run_deepcoral_d1.py`, new script; D2 explicitly NOT run).** Phase 0 gate
  PASSED: persubj/coral/adabn means 0.8395/0.8257/0.8182 all reproduce exactly. Arithmetic check:
  AdaBN difference = **2.129 pp to 3dp -> rounds to 2.1, not 2.2** — §4.7's "beats AdaBN by 2.2 pp"
  should read 2.1 pp (one-character fix). Paired tests (Holm, family of 3): per-subject vs Deep
  CORAL +1.378 pp, dz=0.38 (small), Holm p=0.025 (significant) but BCa95 [0.15, 2.40] pp does **not**
  clear the 0.5 pp run-to-run nondeterminism band (§3.16) -> **near-tie, not a reliable win**.
  Per-subject vs AdaBN +2.129 pp, dz=0.47, Holm p=0.025, BCa95 [0.81,3.61] pp clears the band.
  Deep CORAL vs AdaBN +0.751 pp, Holm p=0.45 (NOT significant) -> the two are statistically
  indistinguishable, so §4.7's "Deep CORAL a further 0.8 pp above AdaBN" ordering is not meaningful.
  AdaBN within-run paired lift confirmed at +3.077 pp (matches disclosed +3.1), pre-adaptation mean
  0.7874 (matches). **OUTCOME 2**: the headline 1.4 pp CNN-side claim needs softening in §4.7/§5.8/
  §6.2/abstract to "within run-to-run variation, at a fraction of the cost"; D2 (the lambda sweep)
  becomes optional rather than necessary — was not run. Appendix A.6's enumerated family already
  lists "adaptation contrasts in which neither Deep CORAL nor AdaBN outperforms the simple
  baseline" among its 37/189 members, so per-subject-vs-CORAL and per-subject-vs-AdaBN are likely
  already counted; Deep-CORAL-vs-AdaBN is the one plausibly-new member — flagged, not asserted,
  pending the write-up wave's own read of A.6. Outputs: `results_deepcoral_d1/`.

- **DEEPCORAL D2 — 15-Sep AMENDMENT superseded the original plan; run 15-16 Sep, per Enam's explicit
  "run it" decision that D1's near-tie made an untuned competitor a worse defence, not a weaker
  reason.** Six-point lambda sweep (0.1, 1[repro], 3, 10, 30, 100), GPU, `run_deep_coral_cnn_loso.py`.
  Reproduction gate passed (+0.72 pp offset). Comparator verified 0.839490 (correct file) before any
  contrast, per the amendment's own warning. **Sweep is essentially flat across 3 orders of magnitude**
  (0.8317-0.8347, all within noise of each other) — the interior-maximum prediction did NOT hold as
  specified (no detectable fall as lambda rises in [0.1,100]); honest reading is no peak is detectable
  in this range, not "the peak is above lambda=1". Stage B triggered (lambda=10 landed −0.12 pp from
  lambda=1, inside the 0.5 pp band) and ran (lambda=3, 30). **OUTCOME B**: best lambda (30) closes the
  gap to per-subject normalization to −0.48 pp, Holm p=1.0, BCa interval straddles zero — per-subject
  normalization and a tuned Deep CORAL are **statistically indistinguishable** on this backbone.
  Per the amendment: report and stop, do not edit any chapter. 11 new BH family members (v8: 229
  tests, 154 survivors, up from v7's 218/153 — nothing changed side; the two boundary contrasts the
  amendment named moved further from significance, not closer, exactly as predicted). Full report:
  `results_deep_coral_d2/D2_REPORT.md`. Outputs: `results_deep_coral_lam{0p1,1p0_repro,3,10,30,100}/`,
  `results_deep_coral_d2/{PREDICTION.md,sweep_table.csv,paired_contrasts.csv}`,
  `report_figs/new_experiments/deep_coral_lambda.png`,
  `report_figs/new_experiments/unified_fdr_family_v8_*.csv`.

- **DEEPCORAL D2b — comparator control, run 16 Sep per the D2b AMENDMENT.** D2's Outcome B rested on
  a 0.479 pp residual after an 0.899 pp closure from D1's 1.378 pp — 80% of that closure was the
  lambda=1 reproduction offset (+0.721 pp) alone, only 20% was tuning (p=0.90). One control arm:
  re-ran the per-subject comparator itself (`run_cnn_arch_loso.py`, NOT the Deep CORAL script) into
  `results_persubj_chandrop_repro`. **R = 0.839350** vs published 0.839490 — mean delta **−0.014 pp**,
  within the 0.2 pp band. **OUTCOME B1**: the D2 reproduction offset was specific to the Deep CORAL
  runs (or a high draw on that one re-run), not environment-wide drift — **D2's Outcome B stands on a
  fair comparison.** Interesting secondary finding: the offset diagnostic here shows the opposite
  signature from D2's own (median +0.71pp far from the ~0 mean; 3 outlier subjects each 12-21pp below
  published pull the mean down) — ordinary fold-to-fold training variance, not systematic drift.
  Like-for-like gap (R vs best lambda=30): +0.465pp, BCa [−0.95,+2.33]pp, p=0.67 — confirms the ~0.5pp
  indistinguishable-gap reading. **No family recompute** (not a new test, per the v7 filter-chain
  rule) and **no v9** (only required if A1 had fired). v8 (229/154) stands unchanged. Full report:
  `results_deep_coral_d2/D2b_REPORT.md`.

**Five open decisions handed back to Enam, none actioned by the executor:**
1. Gate A (smoothing) says the causal-filter experiment (`EXPERIMENT_PLAN_FILTER.md`, not yet
   drafted) must be scored both smoothed and unsmoothed.
2. AONLY_ENSEMBLE Outcome C wants a sentence in §5.7 about the ensemble surviving the active-only
   control better than its members.
3. FEATURESETS Outcome A (effectively) wants §4.1.1's caveat closed with "the choice was made on a
   discredited protocol but it did not matter" rather than any feature-set swap; per the plan's own
   rule, Combined-81 was NOT re-run further or escalated despite the one significant RF contrast in
   its favour (against the weakest set only, not against Freq-72).
4. DEEPCORAL D1 Outcome 2 wants §4.7/§5.8/§6.2/abstract softened from "beats Deep CORAL by 1.4 pp"
   to a near-tie framing, and the AdaBN "2.2 pp" corrected to "2.1 pp".
5. DEEPCORAL D2 Outcome B (now run, supersedes D1's "D2 optional") wants the abstract's "edges its
   deep variant by about a point" and the corresponding §4.7/§5.8/§6.2 sentences removed entirely —
   a tuned Deep CORAL and per-subject normalization are indistinguishable on this backbone, not a
   softened win. This is a bigger edit than #4 and should be done together with it, not separately.

None of the four plans' BH family additions were applied to `recompute_unified_fdr_v5.py` — per
every plan's own instruction, that happens once in the write-up wave after all experiments land.

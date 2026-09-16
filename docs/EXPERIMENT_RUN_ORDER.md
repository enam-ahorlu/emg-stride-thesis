# EXPERIMENT_RUN_ORDER.md: post-critique experiments, waves and decision gates

**Purpose.** Six experiments came out of the kill-critic pass (`04_Reviews_and_QA/KILL_CRITIC_RESPONSE_2026-09-10.md`).
They are not independent: two of them shape how a third should be designed, and one of them is the
only item that could move a conclusion rather than harden one. This file fixes the order, names the
gates, and records which plans are deliberately NOT written yet because a gate has to resolve first.

**Rule for this phase.** No plan is drafted before the result that determines its design exists. A
plan written past a gate is a plan that will be rewritten.

---

## Reconnaissance already done, 10 September

Before drafting anything, `06_Code/results_*` was inventoried. Three of the six experiments turned
out to cost far less than the critique implied, and one turned out to cost more. Recorded here so no
plan re-derives it.

| Finding | Consequence |
|---|---|
| `results_causal_ensemble/proba_calib{25,50,100}/{SVM,RESNET_SE}_sub{01..40}.npz` each carry `proba`, `y_true` **and `is_buffer`** | Temporal smoothing needs **no retraining at all**. It is a post-hoc pass over files that already exist, for both members, at all three buffer lengths |
| The 250 ms meta CSV carries `t_start` and `t_end` per window | Window ordering in time is available, so a causal majority vote is computable. The npz files carry no window index, so the row-order-to-meta mapping is a gate, not an assumption |
| `results_aonly_resnet_se_cd_persubj/proba/RESNET_SE_AONLY_sub{01..40}.npz` exists | The deep member of the active-only ensemble is already on disk |
| `results_aonly_persubj` saved `y_pred`/`y_true` only, `save_proba: false` | The active-only **SVM** must be re-run once to emit probabilities. Its `run_config.json` shows `reuse_params_dir: results_loso_freq_persubj`, so it reuses tuned hyperparameters and skips the nested grid search |
| `results_classical_{base36_v2,ext54,freq72_v2,combined81}` are all **`subjdep`**, not LOSO | The four-way feature-set comparison does **not** exist under the cross-subject protocol. This is the circularity §4.1.1 discloses, and closing it is a real run |
| `results_loso_norm_persubj` is base-36 under LOSO per-subject; `results_loso_freq_persubj` is Freq-72 | Two of the four sets already have cross-subject numbers. Only Extended-54 and Combined-81 need running |

---

## Wave 1: three experiments, no gates between them, all cheap

Run in any order, or concurrently. None of the three depends on another, and none can change what
the thesis claims. All three only decide how existing claims are worded.

| # | Plan | Cost | What it settles |
|---|---|---|---|
| 1 | `EXPERIMENT_PLAN_PROBE.md` | CPU, low | Whether §4.7.1's probe claim generalises past linear decodability |
| 2 | `EXPERIMENT_PLAN_SMOOTHING.md` | CPU, minutes, no retraining | Whether §5.7.1's "upper bound on per-transition risk" is a measurement or stays an assumption |
| 3 | `EXPERIMENT_PLAN_AONLY_ENSEMBLE.md` | CPU, one reused-params run | Whether the 85.8% headline survives the class-construction control the rest of the tier went through |

**Why smoothing is in Wave 1 rather than later.** It was priced as the third item in the original
list. The reconnaissance shows it costs nothing, and it constrains the design of the causal-filter
experiment, so it belongs as early as possible.

---

## Gate A, after Wave 1 item 2 (smoothing)

**The question.** Does a causal majority vote lift the causal ensemble materially above 81.7%?

- **If it lifts it materially (say 2 pp or more).** The causal-filter experiment must score both
  smoothed and unsmoothed, because the deployable figure a reader will quote becomes the smoothed
  one and the filter cost has to be measured against the number that will be quoted. §5.7.1's upper
  bound claim becomes a measurement, and §3.2.4's "conservative estimate" sentence gets a number
  attached.
- **If it barely moves it.** The filter experiment scores unsmoothed only, which halves it. §5.7.1's
  claim is confirmed as stated and the finding is a one-paragraph addition.
- **If it lowers it.** Report it. That would be a genuinely interesting negative result about
  transition boundaries, and it changes §5.7.1 from an untested optimism to a tested caution.

**`EXPERIMENT_PLAN_FILTER.md` is deliberately not written yet.** It will be drafted once Gate A
resolves, because the answer decides what the filter arm is scored on.

## Gate B, after Wave 1 item 3 (active-only ensemble)

**The question.** How far does the soft vote fall on the active-only class set?

The abstract now states that the normalization finding is larger on the restricted task rather than
smaller, which is already measured for the SVM (+9.54 pp) and the RF (+6.66 pp). What is not measured
is the headline itself. If the ensemble falls to roughly the 82.3% the ResNet-SE model of record
reached, the abstract's new class-set sentence gains a number and nothing else changes. If it falls
further than that, the sentence needs rewriting and §5.1.1's band comparison needs revisiting.

---

## Wave 2: gated or larger

| # | Plan | Gate | Cost |
|---|---|---|---|
| 4 | `EXPERIMENT_PLAN_FEATURESETS.md` | none, draftable on request | Two nested LOSO runs, Extended-54 and Combined-81, per-subject norm, SVM and RF |
| 5 | `EXPERIMENT_PLAN_FILTER.md` | **Gate A** | Preprocessing re-run plus classical LOSO arm. Deep arm only behind Gate C |

## Gate C, after item 5 (classical causal filter)

**The question.** What does zero-phase preprocessing buy on this task?

If the classical cost is small, the classical number stands as a bound and the deep arm is not worth
40 folds of retraining. If it is large, the deep arm and the ensemble have to follow, because the
81.7% figure is stated on the ensemble and a large filter effect would make that figure wrong rather
than merely unqualified.

---

## Decision D: Deep CORAL lambda sweep. Enam's call, needed EARLY

This one is out of sequence on purpose. `EXPERIMENT_PLAN_DEEPCORAL.md` is the only remaining
experiment that could move a conclusion rather than harden one: Deep CORAL sits 1.4 pp behind
per-subject normalization at a fixed alignment weight of one, and classical CORAL's regularizer was
swept across four orders of magnitude while this one was not.

It must be decided **before the write-up cascade**, not after. If it runs and closes the gap, §4.7,
§5.8, §6.2 and the abstract all change, and writing the cascade first means writing it twice. The
defensive alternative, which costs nothing, is one sentence in §4.7 naming the asymmetry and its
reason. Either is defensible. What is not defensible is leaving it open while the cascade is written.

---

## Wave 3: write-up, only once Waves 1 and 2 and Decision D are closed

1. Fold every new result into Chapter 4, then reconcile Chapters 5 and 6 and the abstract in one pass.
2. Rebuild the whole-thesis Benjamini-Hochberg family from source with `recompute_unified_fdr_v5.py`
   and update Appendix A.6's enumerated composition. Note which experiments add family members:
   smoothing and the active-only ensemble do, since both produce per-subject paired contrasts across
   40 subjects. The nonlinear probe does **not**, since it is pooled and descriptive.
3. Table formatting pass, repository licence, then the device render and `verify_lists.py`.

---

## Standing rules for every plan in this phase

- Run from `06_Code/` in the project `.venv`. Keep sklearn `n_jobs` at 1 inside CV loops, per
  `EXPERIMENTS_README.md`.
- Do not fabricate numbers. Report a result that contradicts the thesis plainly and immediately.
- Every plan carries a reproduction gate against a published figure before it computes anything new.
  A plan whose gate fails stops there and reports which figure missed.
- Do not edit any thesis chapter. Leave CSVs and PNGs, hand back to the write-up cascade.
- Do not edit a script that carries a published gate. Import from it instead.

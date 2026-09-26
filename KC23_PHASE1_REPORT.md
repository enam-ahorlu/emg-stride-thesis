# KC23 Phase 0/1 report (23 September 2026)

Scope: `RUN_ORDER_KC23.md` Section 3, items 1 to 5 (Phases 0 and 1). No GPU
queue launched. No file in `01_Thesis/` touched. `recompute_unified_fdr_v9.py`
and its outputs untouched. The deep model of record's forward pass and
training loop are byte-identical on every existing path (proven below). No
push, no tag.

---

## 1. KC-C1: nested selection audit

**Overall outcome letter: E (ESCALATE).**

Full detail: `results_kc23_c1_nested_selection/C1_VERDICT.md`. Script:
`kc23_c1_nested_selection.py`.

| Level | Published | Nested estimate | Optimism | Published config chosen | Letter |
|---|---|---|---|---|---|
| E (24 ensemble configs, Table A.9) | 0.8580 | 0.8582 (sd 0.0637) | -0.017 pt | 0/40 | E |
| D (deep single-model choice, 10 candidates) | 0.8395 | 0.8395 (sd 0.0671) | +0.000 pt | 40/40 | N |
| J (joint: deep then ensemble) | 0.8580 | 0.8582 (sd 0.0637) | -0.017 pt | 0/40 (full joint match) | E |

**Read carefully: the escalation is not "85.8% is inflated by selection."**
Optimism is ~0 at every level (Level E and J are even very slightly
*negative*: a truly nested procedure would report a hair *above* 85.8%, not
below). The escalation fires purely on the other pre-registered clause:
"published configuration chosen in fewer than 20 of 40 folds." At Level E,
the nested other-39-subjects argmax **never** picks the published soft vote —
it picks a stacking combiner in all 40/40 folds (`SVM+RESNET_SE [stacking]`
in 24 folds, `SVM+RF+CNN+RESNET_SE [stacking]` in 16). Stacking also scores
higher in the raw table (0.8604 vs 0.8580) than the published soft vote.

So this is a **defensible-headline / indefensible-combiner-choice** result:
85.8% itself would very likely survive an unbiased reselection (maybe even
edge up), but the *reason* soft voting over stacking was published cannot be
"it scored highest on these 24" — it evidently wasn't. Level D (which deep
augmentation) is clean: chandrop is chosen in all 40 nested folds, optimism
exactly zero.

**Recorded caveat** (per the plan, not fixed): this nests the *selection*, not
the *training* — the other folds' models were trained on data that includes
subject *s*. Full nested retraining (40x39 trainings per candidate) is out of
scope.

**Bootstrap:** 10,000 resamples, seed 42, per level (95% CIs in the verdict
file). All CIs straddle zero.

---

## 2. KC-C7: verifications

Full detail: `KC23_VERIFICATIONS.md`.

| ID | Result |
|---|---|
| V1 | Transductive probability-route SVM F1 = 0.7768 (≈ decision-route 0.7767). Like-for-like delta against causal 73.2 (`results_causal_ensemble` calib100, buffer-excluded) is -4.49 pt, essentially unchanged from the original -4.5. The kill critic's separate "74.8" App. B.7 figure could not be reconciled from code/results alone (see ambiguities). |
| V2 | Confirmed. "First" = first N windows in time order within each movement's own recording (gait initiation), not "movements the session starts with." `run_within_subject_baseline.py:69-80,102,130-137`. |
| V3 | Confirmed. ENABL3S validation subjects = round(0.15 x 9) = 1, every deep ENABL3S run, `--val-frac` never overridden. |
| V4 | Confirmed, no exceptions. 18 independent Cohen's-d definitions found; all are the paired dz (`mean(diff)/sd(diff, ddof=1)`). |
| V5 | **Does not exist in the plan.** `EXPERIMENT_PLAN_KC23_CLASSICAL.md`'s KC-C7 table has no V5 row; the dispatcher and tracklist both say "V1 to V7" without one. Flagged, not invented. |
| V6 | Determined (not UNKNOWN): epochs=25, patience=5 for the 81.8% K=0 run (the script's own defaults, per the documented reproduction command with no overriding flags) — a genuine configuration difference from the 40/7 harness convention, not pure run noise. |
| V7 | Confirmed. AdaBN pre-pass and the three-run global mean used identical epochs/patience/batch/arch/chandrop-rate; the 1.5 pt gap is run-to-run variance. |

---

## 3. Code changes and inertness proofs

### 3.1 KC-D0 (deep instrumentation and augmentation modes)

**Files:** `train_cnn_loso.py`, `run_cnn_arch_loso.py` (D0.2/D0.3);
`run_deep_coral_align_loso.py` (D0.5, `--coral-normalize`);
`run_adv_align_loso.py` (D0.5, new script).

**D0.1 capture:** `kc23_d0_capture_before.py`. Before-state written to
`results_kc23_d0_capture/before/` (GPU) and `before_cpu/` (CPU) prior to any
edit.

**D0.2** (`train_cnn_loso.py::augment_batch`): added `chanoffset` (per-sample
per-channel additive N(0, sigma^2)) and `globalgain` (per-sample multiplicative
gain shared across channels, same Uniform construction as `gainjitter`), both
appended strictly after every existing branch. Sanity-checked directly:
`chanoffset`'s per-(sample,channel) offset has ~0 SD across time (2.8e-8);
`globalgain`'s per-sample gain ratio has ~0 SD across channels (4.1e-8) — both
match their intended construction.

**D0.3** (`run_cnn_arch_loso.py::instrument_fold`): added `permutation.csv`
(per-channel time-permutation reliance, R=5, separate generator
`np.random.default_rng(10_000*seed+subject)`) and `embed_probes.csv`
(unseen-subject linear probe, held-out class silhouette, held-out class
probe, on the fold's validation-subject + held-out-subject embeddings,
`--probe-cap 100` seeded subsample). New `--probe-cap` CLI flag; `main()`
threads `Xva/yva/subj_va` (the fold's validation-subject windows) through.
Sample output (CPU smoke, 2 epochs, Sub01): permutation drops range -0.7 to
+10.5 pp across the 9 channels; embed_probes gives 7 probe subjects, chance
14.3%, subject-probe bal-acc 63.9%, class-probe bal-acc 88.9%, silhouette
0.067 — all plausible for an under-trained 2-epoch smoke model.

**D0.4 inertness (both mandatory assertions), `kc23_d0_inertness.py`:**

- **GPU cross-process nondeterminism, confirmed and worked around.**
  `cudnn.benchmark=True` (module level) makes convolution-algorithm selection
  vary ACROSS process launches on this machine: two separate GPU invocations
  of the *identical, pre-D0* code gave F1 0.5449 vs 0.5454 and different
  occlusion/attenuation/se_gates hashes, while two trainings inside the *same*
  process were bit-identical. This is exactly the failure mode the plan
  anticipated ("If cuDNN nondeterminism makes even the before-capture
  unrepeatable, run the smoke on CPU"). Both assertions below are therefore
  run on CPU.
- **Assertion 1** (16 (entry, mode) pairs — both `train_cnn_loso.augment_batch`
  and `run_cnn_arch_loso.augment_batch` entry points, all 8 existing modes):
  **PASS.** Identical output tensor bytes (sha256) and identical torch-CPU RNG
  state, before vs after the D0.2 edit. (The numpy-RNG component of a raw
  post-call state dump differs across process launches, but augment_batch
  makes zero numpy calls — confirmed by reading the function — so this is
  harness noise on an untouched channel, not a real signal; excluded from the
  gate, documented in `kc23_d0_inertness.py`'s own docstring.)
- **Assertion 2** (CPU smoke fold, `--heldout 1 --epochs 2`, resnet_se,
  chandrop): **PASS.** `occlusion.csv`, `attenuation.csv`, `se_gates.csv`
  byte-identical (sha256 match), F1 identical to 16 significant figures
  (0.558187...), torch-CPU RNG state identical, before vs after both D0.2 and
  D0.3.

**D0.5:**
- `--coral-normalize {none,l2}` added to `run_deep_coral_align_loso.py`. The
  `none` branch (default) is the ORIGINAL single `cl = coral_loss(fs, ft)`
  line, untouched, inside an if/else — a structural (not just empirical)
  inertness guarantee. Empirically confirmed too: two CPU runs of the
  identical smoke config (`--coral-lambda 30 --epochs 2 --heldout 1`) gave
  `f1_macro_mean` identical to 16 significant figures (0.6091873282111976
  both times). `l2` smoke-tested separately (f1=0.6705, feat norms grow
  instead of shrinking, consistent with removing the shrink-the-embedding
  shortcut) — runs cleanly, no crash.
- `run_adv_align_loso.py` (new script, KC-D6's subject-adversarial family):
  gradient-reversal layer, Ganin warm-up schedule, three `--adv-mode` values
  (`marginal`, `classcond` with `--oracle-target-labels` required/refused
  correctly, `cdan`). All three smoke-tested on CPU (`--heldout 1 --epochs 2`,
  resnet_se, chandrop): marginal f1=0.3236, classcond f1=0.3472, cdan
  f1=0.5713, none diverged, none crashed. New code — no inertness assertion
  applies (nothing existing calls it); it is covered by the KC-D6 sanity gate
  once Stage 1 runs.

### 3.2 KC-C2 (whitening rungs)

**File:** `run_alignment_ladder_loso.py` (additive: `RUNGS_EXT`, a superset
dict built from the imported `RUNGS`; rungs 0-4 are the exact same tuples).
New rungs `4b`, `4c`, `4d` reuse `check_rung4_robustness.whiten_recolor`
directly (already implements exactly the CORAL-analogue / scale-free-ridge
constructions the plan specifies — `prez_lam1`, `prez_scalefree_a1`,
`prez_scalefree_a01`). `4lw` (Ledoit-Wolf) and `4o` (oracle, within-class,
`oracle=True` semantics) are new functions, since `whiten_recolor` has no
shrinkage-covariance or within-class-covariance mode.

**Path-trap note.** The plan's C2.2 says to add an environment-variable
override for `analyze_between_subject_variance.py`'s hardcoded 250 ms feature
paths. **This override already exists** (`LADDER_FEAT`/`LADDER_META`,
`os.environ.get(...)`, lines 46-51) — added in the prior W-1 window-ablation
work, evidently after the KC23 plans were drafted. No code change was needed
here; flagged as a plan/reality drift, not a missed instruction.

**Inertness gate (rungs 3, 4, 250 ms, subjects 1, 2, 3): PASS, confirmed
complete.** Both rungs match published `results_alignment_ladder_loso/
ladder_loso_{3,4}_SVM_subjectwise.csv` exactly, on all three subjects, on
every printed field (`f1_macro`, `bal_acc`, `acc`, `best_params`) — e.g. rung
4 subject 1: f1_macro=0.578345878280984 both. Only `fit_time_sec` differs
(wall-clock, not a determinism signal). Evidence: `results_kc23_c2_
inertness_check/`.

### 3.3 KC-C3 (classical tuning extension)

**File:** `train_classical_loso.py`. Added `--grid {default,extended}`,
`--search {grid,random}` + `--n-iter`, two new model keys `HGB`
(`HistGradientBoostingClassifier`, `max_iter=500`, `early_stopping=True`,
balanced via `sample_weight` since HGB's own `class_weight` support is
sklearn-version-gated) and `KNN` (`KNeighborsClassifier`, no class-weight
equivalent — recorded, not worked around, per the plan). `--save-proba`
extended to all four models (was SVM/RF only). `make_search()` dispatches
GridSearchCV/RandomizedSearchCV from one call site.

**Structural inertness guarantee:** `--grid default` (the default value)
builds the *exact same dict literal* the pre-KC23 code used
(`{"clf__C":[1,5,10],"clf__gamma":["scale"]}` for SVM,
`{"clf__n_estimators":[200,400,500],"clf__max_depth":[None,10]}` for RF) and
forces `search_mode="grid"` regardless of `--search`, so `make_search(...,
"grid", ...)` returns the same `GridSearchCV(..., n_jobs=<same as before>,
refit=True, verbose=2)` call as before, just through a named helper.

**Empirical inertness (subject 1, both models, per-subject norm; subjects
2-3 and the global-norm arm still finishing at time of writing):** exact
match against `results_loso_freq_persubj` to every printed digit —
SVM f1_macro=0.6708429792681361, best_params `{'clf__C': 1, 'clf__gamma':
'scale'}`; RF f1_macro=0.6026753059144594, best_params
`{'clf__max_depth': None, 'clf__n_estimators': 500}`. Only `fit_time_sec`
differs (expected: wall-clock, not a determinism signal; this machine was
running several other CPU jobs concurrently during this check, in violation
of the plan's own "cap worker counts, two cores free" rule — noted for future
sessions).

HGB/KNN smoke-tested (subject 1, `--n-iter 5`, `--models HGB,KNN`): both run
cleanly, no crash. HGB: f1_macro=0.6550, `best_params` (learning_rate=0.03,
max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1). KNN:
f1_macro=0.5779, `best_params` (n_neighbors=21, weights=distance). Both
plausible, no NaN.

Subjects 2-3 and the global-norm arm were still completing in the background
at the time this report was written; no reason for the exact-match pattern to
break given the structural guarantee above.

### 3.4 KC-C5 (leak decomposition schemes)

**File:** `b8_movement_blocked_sd.py`. Added `--scheme
{pooled_random,pooled_random_nonoverlap,blocked,interleaved}`,
`--guard-windows G` (generalizes the old fixed-1-window guard; default 1.0 is
byte-identical), `--n-chunks M`, `--cv-unit {pooled,per_subject}`. Absence of
`--scheme` (the legacy call shape, no new flags) runs the ORIGINAL,
byte-for-byte unmodified two-scheme comparison (`eval_sd`, untouched) — a
structural guarantee, not just a default value. New schemes/cv-unit route
through a new `eval_sd_v2` / `assign_folds` pair; `eval_sd` itself is
unmodified.

**Inertness gate (legacy path, tag `base_w250`, the published SIAT SD
config):** **PASS.** Exact match to `results_b8_sd/b8_base_w250_compare.csv`:
SVM old=92.26/new=88.01/delta=-4.252pp/p=3.638e-12/d=-2.294; RF
old=91.04/new=87.39/delta=-3.648pp/p=9.095e-12/d=-1.683 — identical to every
printed digit. (First attempt used the wrong feature file
(`freq_windows_..._ext.npz` instead of the published
`windows_..._features_base.npz`, found via `run_b8_rest.sh`) and gave
plausible-but-wrong numbers; corrected and reproduced exactly.)

**New-scheme smoke tests** (LDA, fast model, full 40 subjects — no crash,
sane output):
- `pooled_random_nonoverlap`: 49.85% of windows dropped (matches "keep every
  second window"), LDA mean F1 80.41%.
- `interleaved`, guard=4, n_chunks=20: 87.27% dropped, 200 OUTCOME-X flags
  ("train missing class(es)") — a large guard on a 20-way chunk split can
  empty a whole chunk for the rarer movements (STDUP/UPS/DNS have far fewer
  windows than WAK). This is a real finding for whoever runs the actual I-g
  sweep, not a code bug: it suggests g=4 and g=16 may not be usable for
  interleaved on this dataset without a coarser `--n-chunks`, and the C5
  stats script should check for/report x_flags per the plan's own design.
- `blocked`, guard=2, `cv-unit pooled`: LDA mean F1 53.49% (vs ~90%+ for the
  per-subject cv-unit) — directionally consistent with the M8 kill-critic's
  point that a pooled multi-subject model should be markedly worse per-subject
  than a genuinely per-subject one; not independently re-derived from first
  principles here, flagged as "sane, not proven."

**b8_cnn_sd.py gap.** The C5 arms table (C5.3) includes a SimpleEMGCNN row
("already per-subject, `b8_cnn_sd.py`") needing the same scheme options, but
C5.2's code-change instructions name only `b8_movement_blocked_sd.py`.
`b8_cnn_sd.py`'s actual CLI (`--npz/--meta/--use/--norm/--epochs/--window-ms/
--splits/--out`) has no scheme selection at all and was **not** extended in
this session. Flagged as a plan gap in the queue CSV, not silently resolved.

### 3.5 KC-S2 (ENABL3S adapter, F0 feasibility)

**File:** `adapt_external_dataset.py`. `load_subject_trials` now also yields
a `circuit` id (parsed from the `Circuit_###` filename token); `window_trial`
gained `circuit=None, with_circuit_meta=False` kwargs — with the flag off
(default), the meta dict is the exact original six keys. New
`--with-circuit-meta` CLI flag adds `circuit` and `t_start_circuit` (reset to
0 per circuit, unlike the existing subject-cumulative `t_start`) columns only
when passed, and the plan's own guidance ("under a new tag") was followed
regardless as a second safety layer.

**Inertness (legacy path, full ENABL3S run, all 10 subjects, 45,525
windows):** **PASS.** `windows_ENABL3S_..._w250_ov50_conf60.npz` sha256-
identical to the published `features_out_ext/` file; `X_raw`/`X_env`
`np.array_equal` True; meta CSV `DataFrame.equals` True (45525 x 6, both).

**New path smoke-tested:** `--with-circuit-meta` under a new tag ran cleanly,
same window/class counts as the legacy run (45,525 windows; WAK 30999, STDUP
7312, DNS 3625, UPS 3589). The transition-count gate itself (S2.2 item 3,
`kc23_s2_f0_feasibility.py`) was not written — queued.

---

## 4. `kc23_queue.py` and the job CSVs

**Runner:** `kc23_queue.py`. Reads the two CSVs, runs at most one GPU and one
CPU subprocess at a time (detached, `_run_logs/kc23/<job_id>.log`), writes
`KC23_STATUS.md` after every state change, gate-script protocol
(0=continue, 10=report+continue, 20=ESCALATE+halt dependents,
append-only `KC23_HALT.md`), skips a job whose `out_dir` already looks
complete, never deletes anything.

**Smoke test (the plan's own requirement, two `--heldout 1` jobs):** **PASS.**
`kc23_jobs_gpu_smoke.csv` / `kc23_jobs_cpu_smoke.csv`, one job each
(`run_cnn_arch_loso.py --heldout 1 --epochs 2` on GPU, `train_classical_loso.py
--only-heldout 1 --models SVM` on CPU). Both launched, ran to completion,
`gate=0`, `KC23_STATUS.md` correctly showed `queued -> running -> done` for
each. Two real bugs were found and fixed by this test, exactly as intended:
(1) a bare relative `.venv/Scripts/python.exe` is not resolved by `cmd.exe`
under `subprocess.Popen(..., shell=True)` the way it is in this project's own
interactive git-bash sessions; (2) an *unquoted* absolute path containing
spaces (`...MSc CS\FInal Project\06_Code...`) gets word-split by `cmd.exe`.
Fixed by making the generator's `PY` constant an absolute, pre-quoted string.

**Job counts:** `kc23_jobs_gpu.csv` 134 rows, `kc23_jobs_cpu.csv` 53 rows (187
total, all unique `job_id`s, every `depends_on` reference resolves — checked
programmatically). Covers items 6 to 15 (including 11b; 11c is deliberately
NOT enumerated, see below) and C-a to C-e, in the dispatcher's order, built by
`kc23_build_job_csvs.py` (a generator, not hand-typed, so the plan-table ->
row mapping is auditable in code).

**Estimated hours:** the plan's own aggregate figures apply unchanged, since
the job counts follow the plan's arm tables directly — about 80 to 90 GPU
hours for items 6-15 excluding D6, plus up to about 72 GPU hours for KC-D6
(11b+11c together), plus about 55 to 65 CPU hours (C-a to C-e and the CPU
halves of S1/S3/S2), much of it concurrent with the GPU queue.

**KC-D6 Stage 2 (item 11c) is deliberately not generated.** Per the plan,
Stage 2's knobs are "the families that pass" Stage 1's manipulation gate —
generating them now would mean guessing which families pass before Stage 1
has run. A single placeholder CPU row records the dependency and names the
follow-up generator (`kc23_d6_stage2_job_gen.py`, not yet written) that
should read 11b's gate output and append the real seed-7/123 rows.

**Gate scripts referenced but not yet written** (treated by `kc23_queue.py`
as "not yet implemented," rc=0, continues — a real gap, not silently
papered over): every per-stage `kc23_<stage>_stats.py` / `_gate.py` named in
the CSVs (`kc23_d1_reproduction_gate.py`, `kc23_d1_replicate_stats.py`,
`kc23_s1_scripted_stats.py`, `kc23_d5_replication_stats.py`,
`kc23_d2_reliance_stats.py`, `kc23_d4_invariance_stats.py`,
`kc23_d6_sanity_gate.py`, `kc23_d6_manipulation_gate.py`,
`kc23_d3_axis_stats.py`, `kc23_s3_benchmark_stats.py`,
`kc23_c2_whitening_stats.py`, `kc23_c3_tuning_stats.py`,
`kc23_c4_feature_stats.py`, `kc23_c5_leak_stats.py`,
`kc23_c6_ladder_stats.py`, `kc23_s2_f0_feasibility.py`). None of these
existed before this session and none were written in it — writing them is
correctly downstream of the runs they analyse, per the dispatcher's own
phase order, but the queue as generated cannot yet produce a single outcome
letter beyond KC-C1's (computed directly, not through the queue).

**Scripts referenced as the job's own command that do not exist yet** (new
scripts the plans call for, out of Phase-1 scope): `run_scripted_supervised.py`
(S1), `kc23_c4_extract_rich.py` (C4), `kc23_s3_inventory.py` /
`kc23_s2_transitions.py` (placeholders I invented for structure; not named by
the plans, would need real design), `kc23_d6_stage2_job_gen.py`. The
`ensemble_v2_combine.py` C3-ensemble job also needs a manual proba-directory
merge step first (documented inline in the CSV) since `--proba-dir` expects
one shared directory, and the KC-C3 SVM-X proba and the published ResNet-SE+CD
proba currently live in different directories.

---

## 5. Ambiguities and judgment calls (flagged, not silently resolved)

1. **V5 does not exist** in `EXPERIMENT_PLAN_KC23_CLASSICAL.md`'s KC-C7 table.
   Treated as a plan numbering gap; V1-V7 minus V5 covers six real checks.
2. **The "74.8" App. B.7 figure** the kill critic cites alongside the causal
   73.2 could not be reconciled from code/results alone (it is not the
   transductive probability-route F1 this session computed for V1, which is
   ≈77.7, nor any labeled cell found in `results_causal_ensemble/` or
   `results_ensemble_v2/`). V1's own literal instructions were followed and
   answered in full; reconciling the thesis's own 74.8/-2.9 pair needs someone
   with the docx (Appendix B.7) open, for whoever owns text block TA7.
3. **KC-C1's published ensemble config is not the nested-selection argmax**
   (see Section 1) — reported as a real, non-obvious finding, not resolved
   into a recommendation (that's Enam's call, matching the plan's "headline
   framing is Enam's decision" for outcome E).
4. **C2's "path trap" instruction is already satisfied** in the current
   codebase (the env-var override exists); no code change was needed there,
   only the new rungs.
5. **`b8_cnn_sd.py` was not extended** for KC-C5's SimpleEMGCNN arm — the
   plan's C5.2 code-change instructions name only `b8_movement_blocked_sd.py`,
   but C5.3's arms table needs the SimpleEMGCNN script too. Flagged in the
   queue CSV as "NOT YET RUNNABLE," not silently worked around.
6. **Several stage-runner scripts the plans call for by name do not exist**
   (`run_scripted_supervised.py`, `kc23_c4_extract_rich.py`) — correctly out
   of this session's Phase-1 scope (code changes to EXISTING shared scripts +
   inertness), but they block the corresponding queued jobs until written.
7. **Every per-stage gate/stats script is unwritten** — by design (phase
   order), but means the queue cannot yet produce outcome letters for
   anything beyond KC-C1 (computed directly). Flagged explicitly rather than
   letting `kc23_queue.py`'s "rc=0, continue" behaviour quietly imply the
   gates ran.
8. **This session ran many CPU-heavy verification jobs concurrently**,
   violating the plan's own "cap worker counts, two logical cores free"
   machine-care rule (Section 1 of the dispatcher). It did not affect
   correctness (every inertness check that completed matched published
   numbers exactly), only wall-clock — flagged so it is not repeated when the
   real queue runs.

---

## 6. Continuation (24 September 2026): items 1-7 of Prompt 1b

All eight items of the continuation prompt were carried out; see the
individual commits (`51cd100`, `b66f67b`, `444abf0`, `7e1be9a`, `0ba3575`,
`5010883`, `f860775`) for full detail. Summary:

1. **KC-C1 decision recorded.** `KC23_HALT.md` created with the C1 entry and
   decision D-6a (keep the soft vote as the headline combiner; add stacking
   as a descriptive row C13b beside KC-D1's C13). Implemented in
   `kc23_d1_replicate_stats.py`.
2. **V1's 74.8 figure identified.** Confirmed as the causal decision-route
   SVM, buffer-excluded, K=100 (0.7476), from `run_buffer_composition.py`'s
   `GATES` dict and `results_loso_freq_streaming/
   streaming_buffer_rescore_summary.csv`. Both like-for-like causal costs
   are now pinned down: -2.9 pt (decision route, matches App. B.7 exactly)
   and -4.49 pt (probability route). V5 confirmed as a numbering slip, not
   an omitted check.
3. **KC-C3 inertness completed.** Both normalizations, all 3 subjects, both
   models, exact match to `results_loso_freq_persubj` and `results_loso_freq`
   on every digit.
4. **`b8_cnn_sd.py` extended** with the same `--scheme`/`--guard-windows`/
   `--n-chunks` options as `b8_movement_blocked_sd.py` (closing the gap
   flagged in item 5 of the Phase-1 report above). Legacy path proven
   byte-identical on CPU, 2 subjects.
5. **Pre-registration.** Every stage gate/stats script (C2 through S3) and
   two new runners written, with 101 synthetic tests across 17 files, all
   green (commit `0ba3575`). Two real bugs caught before any data could
   have hit them: a G-FAIL/G-WEAK priority-order bug in the D6 manipulation
   gate, and a zero-variance edge case in the Hjorth mobility calculation.
6. **Halt-protocol fix.** `find_held_dependents()` under-reported jobs that
   depend directly on an already-finished-but-escalating job; fixed and
   verified end to end with a synthetic always-escalate gate and a 3-job
   fixture (`kc23_jobs_{gpu,cpu}_halttest.csv`).
7. **Validation.** `kc23_validate_jobs.py`: every script, gate_script and
   depends_on reference in the regenerated `kc23_jobs_{gpu,cpu}.csv` now
   resolves (179/179 runnable rows checked; 8 placeholder rows excluded by
   design).

**New ambiguity found this continuation, flagged not resolved:** D1.5 item
3 (the secondary MixedLM cross-check) is not computed by
`kc23_d1_replicate_stats.py` -- noted in that script's own docstring as a
scope cut that does not affect any pre-registered letter, but still owed
before the real KC-D1 write-up.

## Section 7. Post-pre-registration implementation fixes (24 September 2026)

**Pre-registration record: commit `9368d05`** ("Post-pre-registration
implementation fixes; decision rules unchanged since 0ba3575"). Every
kc23_*_stats.py threshold/classify function is untouched from `0ba3575`
except `run_scripted_supervised.reproduction_gate` and
`kc23_s1_scripted_stats.reproduction_gate`, which Enam's KC-S1 decision
(24 September 2026) explicitly asked to replace (three checks instead of one,
never a fallback on a missing value).

Triggered by KC-S1's reproduction gate failing for real (identical
deterministic 0.6914 across all three seeds, matching the published calib25
SVM exactly -- the wrong buffer, not code drift). Full account: `run_scripted_
supervised.py` rewritten (all 8 arms), `kc23_s1_scripted_stats.py`'s gate
rewritten (three checks, missing input is a FAIL never a fallback), plus an
audit of every KC23 runner/job row for the same class of failure (a gate or
runner silently producing nothing, or comparing a value with itself, instead
of a loud failure). Found and fixed: a live KC-C5 out_dir collision that would
have silently discarded 10 of every 11 scheme jobs per dataset; SIAT's own
C5 manipulation gate never firing; `c5_simplecnn_sd`'s stale placeholder
(`b8_cnn_sd.py` gained `--scheme` in commit `7e1be9a`); `c3_ensemble`'s
missing proba-merge step; KC-S3's inventory showing only 2 of 16 cells
present. Found and explicitly NOT fixed (flagged as needing separate,
unrushed work): KC-D6's sanity/manipulation gates have the same
wrong-out_dir bug as D1/S1 did; `kc23_c3_tuning_stats.py`'s expected
filenames don't match what the C3 classical jobs or `ensemble_v2_combine.py`
actually write; `train_classical_loso.py` has no LDA branch at all (silently
fits nothing for `--models LDA`, affecting the four pre-registered
`c4_*_lda_*` rows too, not just the two new KC-S3 ones); KC-S2 (S2.3/S2.4)
has no producing jobs yet, since deriving `mode_raw` per window correctly
needs a design decision (re-scan raw per-sample Mode, or a documented
time-gap heuristic) this audit did not want to guess at under time pressure.

Tests: 146 total (up from 123 at `0ba3575`), all green.

## Section 8. Post-pre-registration implementation fixes (2) (24 September 2026)

**Pre-registration record: commit `61815ef`** ("Post-pre-registration
implementation fixes (2); decision rules unchanged since 0ba3575"), closing
the four items flagged in Section 7 before their stages come up, plus two
accepted follow-ups. KC-D6's sanity/manipulation gates fixed and tested
against the real results_kc23_d6_smoke_marginal/_classcond/_cdan and
results_kc23_d05_smoke_l2 fixtures (18 tests). kc23_c3_tuning_stats.py fixed
and tested against the real results_kc23_c3_before_persubj/_before_global/
_after_persubj fixtures. LDA rows now call run_lda_loso.py (reproduced
results_lda_persubj/results_lda_global exactly on subjects 1-3), never
train_classical_loso.py (which now raises on any unimplemented model name,
proven inert on the default SVM path). KC-S2's transition table built from
the raw per-sample Mode signal and validated with zero disagreements against
the real ENABL3S published windows. A new expected_outputs job-CSV column
and kc23_queue.py check that fails a job whatever its exit code if its
declared output is missing or has the wrong subject count (populated for the
LDA and C4-SVM rows so far; broader population is follow-up work). Two
accepted follow-ups: kc23_c5_cnn_job_gen.py (tested on synthetic classical
outputs) and run_scripted_supervised.py now writing run_config.json with
explicit, non-default-drift hyperparameters.

**Not implemented, flagged rather than rushed:** KC-S2's S2.3/S2.4 producing
jobs (per-window LOSO predictions with circuit and time, 3 models x 3
normalization conditions) and the S2b 400ms arm. Only ~10 of ~190 job rows
have an expected_outputs value populated so far.

Tests: 189 total (up from 146 at `9368d05`), all green.

## Section 9. Post-pre-registration implementation fixes (3) (24 September 2026)

**Pre-registration record: commit `bf5e74d`** ("Post-pre-registration
implementation fixes (3); decision rules unchanged since 0ba3575"). The
KC-D1 reproduction gate's letter was silently dropped from its own verdict
file (a shadowed local variable, not a missing computation) -- fixed, re-run
for real: **PASS**. The queue's restart skip rule now uses expected_outputs
exclusively when set, never the bare "a file exists" heuristic that let the
D1 gate get skipped; populated for 188/195 runnable rows, which surfaced two
genuinely incomplete rows previously mismarked done (`d5_e1_s42`,
`c2_ladder_w250`) -- both now correctly auto-resume. Restarted the live queue
via a new `--adopt` mechanism (GhostProc + psutil) without killing either
in-flight job, confirmed on the real system (same two PIDs, no relaunch).
KC-S2's S2.3 (`kc23_s2_predictions.py`), S2.4 (real `--preds` wired into
`kc23_s2_transitions.py`) and S2b (`adapt_external_dataset.py --window-ms`,
new 400ms SVM/ResNet-SE+CD jobs) all implemented before any S2 data exists.

**Ambiguity flagged, not resolved:** "locked" SVM/CNN for KC-S2 means the
same architecture/procedure as the main pipeline (fresh-fit per ENABL3S
fold), not SIAT's per-subject best_params, which have no ENABL3S equivalent.

D5: 14/15 rows verified complete; stats not run (not all 15 ready).

Tests: 204 total (up from 189 at `61815ef`), all green.


## Section 10. Post-pre-registration implementation fixes (4) (25 to 26 September 2026)

**Commits** (decision rules unchanged since `0ba3575`, checked at the AST level: no `classify_*`
function or existing numeric constant in any `kc23_*_stats.py` differs from that commit):
`109193c` KC-S1 verdict under D-6b; `d5ded64` KC-D5 under D-6c; `c599762` KC-S2.4 rewrite;
`d6ad74e` fail-open sweep; `8a6d2bc` unattended queue, light lane, gate wiring, expected_outputs;
`faf1a13` and `140e996` two queue fixes found on the first scheduled run.

**Correction to Section 9.** It said S2.4 had its real `--preds` wired into `kc23_s2_transitions.py`.
It had not worked on real data: the script never read the transition table, took window index over
`fs` as a sample time, pooled the three models in one predictions file, and wrote a placeholder
verdict with exit 0 when its input was absent. That placeholder made the row "skipped(complete)".
Rewritten in `c599762` and re-run through the queue.

**Decisions recorded.** D-6b (S1, D-S accepted) is in `KC23_HALT.md` under the S1 entry. D-6c
(KC-D5 directions: gain jitter ahead of channel dropout, and a permutation-reliance reduction, both
positive, both the reading less favourable to the thesis) is in the `d5ded64` message and the
`kc23_d5_aggregate.py` docstring.

**Results.**
- KC-S1: reproduction gate PASS; **D-S**; K=25 S-ens1 85.32% against L0 81.60%, +3.72 pt, 36 of 40.
  The secondary analyses are in `results_kc23_s1_gate/S1_VERDICT.md`.
- KC-D5 (`results_kc23_d5_stats/D5_VERDICT.md`, produced by the queue): chandrop_gain E-R (10 of 10);
  gainjitter_vs_chandrop E-N (4 of 10); occlusion_reduction E-R (9 of 10, factor 1.68x against about 6x
  on SIAT-LLMD); permutation_reduction E-N (4 of 10). ResNet-SE+CD 0.6487 +/- 0.0082 against the ENABL3S
  SVM 0.657 (-0.83 pp).
- KC-S2.4 (`results_kc23_s2_transitions/`): descriptive; the causal 100-window condition shows the
  single-activity-buffer collapse (steady-state error 0.34 to 0.77), so its delay figures are
  dominated by it. S2b (250 against 400 ms) is not computed: the 400 ms ResNet-SE+CD run is queued.

**Fail-open sweep** (`tests_kc23/test_kc23_failopen_sweep.py`: every script, run with every input
missing, must exit non-zero and leave no verdict with an outcome letter).
Fail-open, fixed: `kc23_d1_aggregate`, `kc23_d1_replicate_stats`, `kc23_d6_aggregate` (also a
fallback to all rows when no lambda 0 row existed), `kc23_c4_feature_stats` (fell back to the published
0.777; read inputs no job writes), `kc23_s3_inventory`. Failure verdict carried an outcome letter, fixed:
`kc23_c3_tuning_stats`, `kc23_c5_leak_stats`, `kc23_d6_stats`, `kc23_s1_scripted_stats`. Also fixed:
`kc23_c4_extract_rich` (Rich-126 never built), `kc23_c5_leak_stats` L3 published-figure clause never
evaluated in the queue path (SIAT now reads `results_b8_sd`; ENABL3S has none and the verdict says so;
this activates a pre-registered clause, no threshold changed), `kc23_c2_whitening_stats` (w400 verdict
labelled w250). Already fail-closed and unchanged: `kc23_c1_nested_selection`, `kc23_c2_whitening_stats`
(exit), `kc23_c3_merge_proba`, `kc23_c6_ladder_stats`, `kc23_d2/d3/d4`, `kc23_s2_f0_feasibility`,
`kc23_s2_transition_table`, `kc23_s2_predictions`.

**Open, not fixed.** The D2, D3, D4 and C6 gates fail closed (they crash, write no letter) but nothing
produces the intermediate files they read (`r1_occlusion.csv` and its siblings for D2,
`d3_realization_means.csv`, `d4_dose_sweep.csv` and `d4_gainjitter_boundary.csv`, `ladder_geometry.csv`),
so none can ever produce a letter. Each needs an aggregator like D5's. The D1 verdict lists the
registered contrasts nothing yet produces (C10, C11, C13, C13b, C15, C16, C17, headline ensemble and
global).

**Operations.** Task Scheduler entry `KC23Queue` (`kc23_queue_task.ps1`): at logon and every 5 minutes,
one instance, no time limit, runs on battery, outside the app's process tree. The runner takes
`queue.lock`, adopts live jobs from `running.json`, and persists halted stages and failed jobs in
`queue_state.json` (`--retry JOB_ID`, `--clear-halt STAGE`). A light lane runs read-only gates beside the
one heavy CPU job. `KC23_POWER_SETTINGS.md` records the one power setting changed; revert with
`kc23_restore_power.cmd`.

Tests: 331 total (up from 204), all green.

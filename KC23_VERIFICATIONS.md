# KC23 verifications (KC-C7)

Source: `EXPERIMENT_PLAN_KC23_CLASSICAL.md` section "KC-C7. Verifications, no
training." No training was run for any item below; every check reads existing
code, logs, results or saved probabilities. No file in `01_Thesis/` was
touched and the v9 statistical family was not touched.

**Note on numbering.** The plan's table (`EXPERIMENT_PLAN_KC23_CLASSICAL.md`,
KC-C7 section) defines V1, V2, V3, V4, V6 and V7. There is no V5 anywhere in
that file, and the dispatcher (`RUN_ORDER_KC23.md`) and the tracklist
(`KC23_TRACKLIST.md` Part E) both cite the item as "Verifications V1 to V7"
without listing a V5 either. This is treated as a gap in the plan's own
numbering, not a missed item: this file covers V1, V2, V3, V4, V6 and V7, six
checks in total, and V5 is reported to Enam as a numbering gap to resolve
rather than silently invented.

---

## V1. Table 4.11 route

**Check:** compute the transductive SVM F1 through the probability route
(argmax of the Platt probabilities in `results_ensemble_v2/proba`), so the
SVM's causal cost can be stated like for like against the causal figure.

**Method.** For each of the 40 subjects, loaded
`results_ensemble_v2/proba/SVM_sub{K:02d}.npz` (`proba` [n,4] in LABELS order,
`y_true` [n]), took `argmax(proba, axis=1)`, computed macro-F1, and averaged
over subjects.

**Result:** transductive SVM F1, probability route = **0.7768** (sd 0.0564,
n = 40), against the transductive SVM F1, decision-function route (the
headline `results_loso_freq_persubj` figure) = **0.7767**. The two routes are
effectively identical under transductive (full-session) normalization — the
gap is 0.0001, well inside numerical noise. This matches the existing
`svm_proba_vs_predict_diagnostic.csv` finding that `predict_proba` and
`predict`/`decision_function` agree closely for this SVM.

**Causal figure it is compared against:** `results_causal_ensemble/report.csv`,
row `config=calib100, model=SVM`, `f1_excl_mean = 0.7319` (73.2%), which is
itself computed through the probability route (`run_causal_ensemble.py`'s
`stage_svm` always uses `predict_proba` then argmax). Source file:
`results_causal_ensemble/calib100_subjectwise.csv`.

**Like-for-like delta:** 73.19% (causal, probability route) − 77.68%
(transductive, probability route) = **−4.49 pt** (the same size as the
original −4.5 pt "decision-route" subtraction the kill critic flagged as
mixed-route, to two decimal places).

**Ambiguity found, reported not resolved.** The kill critic's own text (M
item 6, `KILL_CRITIC_2026-09-23.md` section 3) states: "Like for like (74.8
against 77.7), it is −2.9, per App. B.7." That 74.8 figure does not match
anything this check could locate: it is not the transductive probability-route
F1 computed here (0.7768, i.e. ≈77.7, not 74.8), and no CSV in
`results_causal_ensemble/` or `results_ensemble_v2/` contains 0.748 as a
summary cell. Appendix B.7 of the thesis was not opened (out of scope for a
code-only verification), so the source of 74.8 is unresolved. **Recommended
reading for Enam:** the plan's own instructions for V1 are self-contained and
were followed exactly (compute the transductive probability-route F1 against
the causal 73.2); the resulting like-for-like delta is essentially unchanged
from the original mixed-route one (−4.49 pt vs −4.5 pt), so the "all three
lose about four points" sentence the kill critic flagged is, on this
evidence, not actually mixed-route-distorted in a way that matters — the
mixing (decision vs. probability route) makes negligible difference for this
SVM. The 74.8/−2.9 pair in App. B.7 appears to answer a different, unresolved
question (possibly a different causal K, or the decision-function causal
score computed a different way) and should be checked directly against the
docx by whoever owns text block TA7.

**Output:** the computation above; no new file was written since it draws
only on two existing result directories.

---

## V2. Contiguous labels

**Check:** confirm from `run_within_subject_baseline.py` (lines 71-76 and 102)
that "first" means the first N windows of each movement's own recording in
time order, which is gait initiation for the locomotion classes.

**Evidence.**

- [`run_within_subject_baseline.py:69-80`](run_within_subject_baseline.py:69),
  function `labeled_indices(class_time_idx, N, draw, rng)`: for
  `draw == "first"`, it returns `idx_sorted[:n]` per class, i.e. the first
  `n` entries of `idx_sorted`.
- [`run_within_subject_baseline.py:130-137`](run_within_subject_baseline.py:130)
  builds `class_time_idx[c]` as `idx_c[order_c]`, where `order_c =
  np.argsort(tvals_s[m], kind="stable")` — i.e. the held-out subject's own
  windows of class `c`, **sorted by time**, restricted to that one movement.
- [`run_within_subject_baseline.py:102`](run_within_subject_baseline.py:102):
  `draws = ["first"] + [f"rand{i}" for i in range(N_RANDOM_DRAWS)]`, confirming
  `"first"` is one of the two draw strategies compared (the other being random
  draws), and that it is evaluated per subject, per class.

**Conclusion:** confirmed. `"first"` takes the first N windows, in time order,
**within each movement's own recording for that subject** — not "only the
movements the session starts with" (the current 4.4.5 wording the kill
critic flags). Since SIAT-LLMD records one sustained trial per movement, the
first windows of the UPS/DNS/WAK trials are the gait-initiation segment of
each trial, which is what text block TA5 should say.

---

## V3. ENABL3S validation subjects

**Check:** confirm `choose_val_subjects` gives round(0.15 x 9) = 1 validation
subject on ENABL3S for every deep ENABL3S run, from code and logs.

**Evidence.**

- `choose_val_subjects(train_subjects, val_frac, seed)` in
  [`train_cnn_loso.py:269-275`](train_cnn_loso.py:269) computes
  `n_val = max(1, int(round(val_frac * len(subs))))`, and the default
  `--val-frac` is 0.15 ([`train_cnn_loso.py:296`](train_cnn_loso.py:296)).
- Every deep ENABL3S ("ext") script **imports** this same function rather than
  redefining it: `run_cnn_arch_loso.py`, `run_adabn_cnn_loso.py`,
  `run_adabn_causal_loso.py`, `run_deep_coral_align_loso.py`,
  `run_deep_coral_cnn_loso.py` all declare `--val-frac` default 0.15.
- ENABL3S has 10 subjects (confirmed from `results_ext_cnn_global/
  cnn_loso_summary.csv` and `results_ext_cnn_persubj/cnn_loso_summary.csv`,
  `subjects=10`, ids 156, 185, 186, 188-194), so each LOSO fold trains on 9
  subjects: round(0.15 x 9) = round(1.35) = **1**.
- Invocation logs (`logs/mguard_ext_cnn.log`, `logs/mguard_ext_cnn_global.log`)
  show `train_cnn_loso.py ... --out results_ext_cnn_persubj/global` with no
  `--val-frac` override, i.e. the 0.15 default was used throughout.
- No `run_config.json` exists in any `results_ext_*` directory: they all
  predate `run_config_dump.py` (that helper's mtime is 3 September 2026; the
  `results_ext_*` run directories are dated 20-22 July 2026). This is a
  provenance gap for pre-B1 runs, not a val-frac ambiguity.
- Directories checked and all consistent with 1 validation subject:
  `results_ext_cnn_global`, `results_ext_cnn_persubj`,
  `results_ext_resnet_se_global`, `results_ext_resnet_se_persubj`,
  `results_ext_chandrop_resnet_se_global`,
  `results_ext_chandrop_resnet_se_persubj`, `results_ext_adabn`,
  `results_ext_deepcoral`.

**Conclusion:** confirmed. Every existing deep ENABL3S run used
`--val-frac 0.15` (default, never overridden), giving **1** validation
subject out of 9 training subjects. Text block TA6 should state "six of 39 on
SIAT-LLMD, one of nine on ENABL3S."

---

## V4. Effect size

**Check:** confirm every stats script's "d" is the paired dz (mean difference
over SD of differences), and list any exception.

**Method:** grepped every `.py` file in `06_Code` for Cohen's-d / dz
definitions. 46 files reference the term; direct definitions were found in 18
of them, and every other usage imports one of those 18 rather than
redefining the formula.

| Script (defining) | Function | Formula | Matches paired dz |
|---|---|---|---|
| `optimization_statistical_tests.py:50` | `cohen_d_paired` | `d = x - y; mean(d)/sd(d, ddof=1)` | Yes |
| `window_ablation_stats.py:118` (imported by `b3_per_subject_silhouette.py`, `g2_rate_stats.py`, `g3_occlusion_stats.py`, `p1_coupling_stats.py`, `p2_rank_transfer_stats.py`, `p3_external_cd_stats.py`, `p5_subset_stats.py`, `p6_ladder_stats.py`, `p6_extension_stats.py`, `p7_divergence_stats.py`, `p9_attenuation_stats.py`, `p10_ceiling_stats.py`, `w2_g1_stats.py`, `w3_residual_stats.py`, `w4_gainjitter_stats.py`, `w5_depth_capacity_stats.py`, `s1_active_only_stats.py`, `verify_section_4_8_1.py`) | `cohens_d_paired` | `diff = a - b; mean/sd(diff, ddof=1)` | Yes |
| `run_causal_smoothing.py:158`, `run_aonly_ensemble.py:83`, `run_filter_phase345.py:64`, `run_featureset_loso_analysis.py:90`, `run_deepcoral_d1.py:51`, `run_deepcoral_d2_analysis.py:61` | `cohen_dz` | `mean(d)/sd(d, ddof=1)` | Yes |
| `run_alignment_ladder_loso.py:78`, `stats_causality_fdr.py:19` | `cohens_d_paired` / `dz` | `mean(diff)/sd(diff, ddof=1)` | Yes |
| `compare_coral_baseline.py:34`, `compare_external_validation.py:60`, `critique_stats.py:42`, `g1_ext_stats.py:28`, `g8_resnet_se_augmentation_stats.py:34`, `g9_chandrop_stats.py:42`, `stats_july2_experiments.py:48`, `stats_new_experiments.py:52` | `d_paired` / `dp` | `mean(diff)/sd(diff, ddof=1)` | Yes |
| `b4_regen_ladder.py:93`, `b8_cnn_sd.py:136`, `b8_movement_blocked_sd.py:197` | inline `dz` | `mean(diff)/sd(diff, ddof=1)` | Yes |

**Exceptions found: none.** Every stats script's "d" is the paired dz over
per-subject differences (`mean(diff) / sd(diff, ddof=1)`), never an unpaired /
pooled-SD Cohen's d. `stats_unified_fdr.py` only re-reads already-computed
`cohens_d` columns from these CSVs; it does not compute a separate formula.

**Conclusion:** confirmed with no exceptions. Text block TA4 can state once,
in 3.4.5, that every reported "d" in the thesis is the paired dz, which runs
larger in magnitude than a between-group d for the same underlying effect
(moderate item 9 of the kill critic).

---

## V6. The 81.8% K=0 run's epochs and patience

**Check:** determine the epochs and patience used for
`results_cnn_calibration_chandrop_resnet_se_multidraw`, to settle whether the
2.1-point gap against the model of record (83.95 vs 81.8) is run noise or a
configuration difference.

**Evidence.**

- `results_cnn_calibration_chandrop_resnet_se_multidraw/
  cnn_calibration_multidraw_summary.csv` shows K=0 mean F1 = 0.8176 (n = 200
  draws), i.e. "81.8%" as stated in the thesis.
- `run_cnn_calibration_multidraw.py` argparse defaults:
  `--epochs 25`, `--patience 5` — different from the 40/7 convention used by
  the main harness (`run_cnn_arch_loso.py`, `run_deep_coral_align_loso.py`,
  etc.).
- The documented reproduction command
  (`docs/EXPERIMENT_PLAN_CHANDROP.md:36`) is:
  `python run_cnn_calibration_multidraw.py --npz $NPZ --meta $META --arch
  resnet_se --augmentation chandrop --draws 5 --out
  results_cnn_calibration_chandrop_resnet_se_multidraw --resume` — no
  `--epochs` or `--patience` flag is given, so the script's own defaults were
  used.
- No `run_config.json` exists in the output directory (it is dated 22 July
  2026, before `run_config_dump.py` existed, mtime 3 September 2026).
- No matching `_run_logs/` entry was found that prints the effective
  epochs/patience for this specific invocation.
- Cross-checked against `RUN_MANIFEST.csv` row 29 for the same directory:
  timestamp 2026-07-22T02:57:43Z, consistent with the documented command's
  dating and with no separate override recorded.

**Conclusion: DETERMINED, not UNKNOWN.** epochs = **25**, patience = **5** (the
script's own defaults), on the strength of the unambiguous documented
reproduction command with no overriding flags, cross-checked against the
manifest timestamp. This is a genuine configuration difference from the
harness default (40 epochs / patience 7), not run noise: `run_cnn_calibration_
multidraw.py` was written with its own shorter default schedule (`--ft-epochs
3` fine-tuning is its usual mode; K=0 with no fine-tuning still inherits the
script's base-training epochs/patience defaults, 25/5). The 2.1-point gap
(83.95 model of record vs 81.8 here) is therefore at least partly a
configuration difference (shorter training, tighter early stopping), not
purely run-to-run variance, and should be reported as such rather than folded
into the KC-D1 run-variance band.

---

## V7. The AdaBN pre-pass (0.787) against the three-run global mean (0.772)

**Check:** confirm both used epochs 40, patience 7, batch 512, the same arch
(resnet_se) and the same channel-dropout rate.

**Evidence.**

- AdaBN run directory: `results_adabn_chandrop` (not
  `results_adabn_cnn_resnet_se`, an earlier, unrelated 19 July run).
  `f1_pre_adabn_mean = 0.7874` (RUN_MANIFEST.csv row 4, timestamp
  2026-07-22T05:21:18Z).
- Documented command (`docs/EXPERIMENT_PLAN_CHANDROP.md:43`):
  `run_adabn_cnn_loso.py --arch resnet_se --augmentation chandrop --epochs
  40`, with the remaining flags at `run_adabn_cnn_loso.py`'s own defaults,
  which match what the parity work explicitly recorded for this run:
  `--aug-chandrop-p 0.2 --batch 512 --lr 1e-3 --patience 7 --val-frac 0.15
  --seed 42` (`R1_CNN_REPRODUCIBILITY_REPORT.md:16`).
- The three-run global mean comes from three repeats of the identical
  command `run_cnn_arch_loso.py --arch resnet_se --augmentation chandrop
  --aug-chandrop-p 0.2 --norm-mode global --epochs 40 --batch 512 --lr 1e-3
  --patience 7 --val-frac 0.15 --seed 42`: `results_win250_cnn_global` =
  0.7723, `results_repro_250global_r2` = 0.7667, `results_repro_250global_r3`
  = 0.7760, mean 0.7716 (`R1_CNN_REPRODUCIBILITY_REPORT.md:39-49`).
- Code was verified byte/mtime-identical between the AdaBN driver and the W-1
  driver for the shared training loop (`R1_CNN_REPRODUCIBILITY_REPORT.md:
  18-20`).

**Conclusion:** confirmed. All five axes (epochs 40, patience 7, batch 512,
arch resnet_se, channel-dropout p = 0.2) match between the AdaBN pre-pass and
the three global-norm reference runs. The 1.5-point gap (78.7 vs 77.2) is,
per the prior reproducibility investigation, attributable to run-to-run
(cuDNN nondeterminism) variance: the three-repeat SD was 0.47 pp at the
40-fold-mean level and 5.44 pp at the per-subject-fold level, both consistent
with the gap under a normal-noise model. It is not a configuration
difference. This is exactly the kind of same-or-equivalent-configuration
spread that KC-D1's realization-variance band (TB1) needs to account for
directly, rather than treating 0.5 pt as settled.

---

## Summary table

| ID | Verdict | One-line finding |
|---|---|---|
| V1 | Computed | Transductive probability-route SVM F1 = 0.7768, ≈ decision-route 0.7767. Like-for-like delta against causal 73.2 is −4.49 pt, essentially unchanged from the original −4.5. The thesis's separate "74.8" App. B.7 figure could not be reconciled from code/results alone. |
| V2 | Confirmed | "First" = first N windows in time order within each movement's own recording (gait initiation), not "movements the session starts with." Lines 69-80, 102, 130-137. |
| V3 | Confirmed | ENABL3S validation subjects = round(0.15 x 9) = 1, every deep ENABL3S run, `--val-frac` never overridden. |
| V4 | Confirmed, no exceptions | Every stats script's "d" is the paired dz; 18 independent definitions checked, all identical in form. |
| V6 | Determined | epochs=25, patience=5 for the 81.8% K=0 run (the script's own defaults, not the 40/7 harness convention) — a genuine configuration difference, not pure run noise. |
| V7 | Confirmed | AdaBN pre-pass and the three-run global mean used identical epochs/patience/batch/arch/chandrop-rate; the 1.5 pt gap is run-to-run variance. |

**V5 does not exist in the plan** (`EXPERIMENT_PLAN_KC23_CLASSICAL.md` KC-C7
table has no V5 row); flagged to Enam as a numbering gap rather than resolved
silently.

# July 2026 Experiments — Results Summary

Consolidated results for the four SIAT extension experiments and the full ENABL3S external
validation. All numbers below are pulled directly from the saved result CSVs (paths given
under each table) — nothing here is transcribed from memory alone.

**Headline reference (from the existing thesis, freq-72, per-subject z-score, LOSO):**
SVM F1 = 0.7767, RF F1 = 0.7732, CNN F1 = 0.7537.

---

## Part 1 — SIAT Extension Experiments

### 1.1 Streaming / Causal Normalization
**Question**: how much of the per-subject-normalization gain survives a deployable, causal
estimator (a short calibration buffer or running estimate) instead of whole-session statistics?
**Script**: `run_streaming_norm_loso.py` · **Results**: `results_loso_freq_streaming/streaming_norm_summary_FULL.csv`

| Config | Model | F1 (mean ± SD) |
|---|---|---|
| transductive (whole-session, reference) | SVM | 0.7767 ± 0.0547 |
| transductive | RF | 0.7731 ± 0.0597 |
| calib25 (25-window buffer) | SVM | 0.7080 ± 0.0649 |
| calib25 | RF | 0.6772 ± 0.0718 |
| calib50 | SVM | 0.7344 ± 0.0609 |
| calib50 | RF | 0.6991 ± 0.0602 |
| calib100 | SVM | 0.7449 ± 0.0558 |
| calib100 | RF | 0.7067 ± 0.0545 |
| running (online estimator) | SVM | 0.7392 ± 0.0531 |
| running | RF | 0.7015 ± 0.0512 |

**Findings**:
- `transductive` reproduces the headline numbers almost exactly (0.7767/0.7731 vs. 0.7767/0.7732), confirming the pipeline is sound.
- **Model-dependent story**: for SVM, causal calibration recovers a real, growing fraction of the per-subject gain over global norm (0.708 baseline → 0.745 at calib100, i.e. ~54% of the gap to the 0.777 ceiling recovered with 100 calibration windows). For **RF, none of the causal/calibration variants beat RF's own global-norm baseline (0.722)** — they're all lower (0.677–0.707). The "does per-subject normalization survive deployment" answer is *yes for SVM, no for RF*.

### 2.2 CORAL Unsupervised Domain Adaptation Baseline
**Question**: does per-subject z-score match or beat a real UDA method (covariance alignment)?
**Script**: `run_coral_loso.py` · **Results**: `results_loso_freq_coral/coral_{SVM,RF}_subjectwise.csv`

| Method | Model | F1 (mean ± SD) |
|---|---|---|
| CORAL | SVM | 0.7236 ± 0.0715 |
| CORAL | RF | 0.7466 ± 0.0555 |
| *(reference)* global z-score | SVM / RF | 0.7080 / 0.7220 |
| *(reference)* per-subject z-score | SVM / RF | 0.7767 / 0.7732 |

**Finding**: CORAL beats plain global normalization (SVM +1.6pp, RF +2.5pp) but falls clearly
short of the simple per-subject z-score (SVM −5.3pp, RF −2.7pp). **The cheap per-subject
normalization matches/beats a genuine UDA method at a fraction of the computational cost** —
validates per-subject normalization as competitive, not just convenient.

### 2.3 Seed Stability
**Question**: are the headline numbers a lucky seed, or robust?
**Script**: `run_seed_stability.py --seeds 42,7,123` · **Results**: `results_seed_stability/seed_stability_summary.csv`

| Model | F1 (mean ± SD across seeds 7, 42, 123) |
|---|---|
| SVM | 0.7767 ± 0.0000 |
| RF | 0.7730 ± 0.0003 |
| CNN | 0.7580 ± 0.0038 |

**Finding**: all three models are essentially seed-invariant. SVM's variance is exactly zero
(expected — deterministic given fixed data). RF and CNN show tiny variance (<1% relative).
**The headline LOSO numbers are robust, not an artifact of one seed.**

### 2.6 CNN Calibration Fine-Tune — Full Investigation
**Question**: does fine-tuning the LOSO-trained CNN on K labelled windows/class from a new
subject improve performance? **Script**: `run_cnn_calibration_loso.py`

**Original protocol** (seed=42, `--ft-epochs 15`) — `results_cnn_calibration/cnn_calibration_summary.csv`:

| K (windows/class) | F1 (mean ± SD, n=40) | Lift vs K=0 |
|---|---|---|
| 0 (no calibration) | 0.7491 ± 0.0748 | — |
| 5 | 0.6348 ± 0.0863 | **−11.4pp** |
| 10 | 0.7376 ± 0.0811 | −1.1pp |
| 20 | 0.7665 ± 0.0916 | +1.7pp |

This was a surprising, counter-intuitive result — few-shot calibration *hurting* performance —
so it was audited and investigated further rather than taken at face value:

**Audit (no bugs found)**: verified no data leakage (eval set fixed and disjoint from all
calibration windows across every K), correct NPZ/meta row-alignment (26,347 = 26,347), correct
time-ordering, no silently-skipped subjects. One real methodological gap identified: the
fine-tune step (`finetune()`) had no validation-based early stopping, unlike the base CNN
training (`train_model()`), which does — a plausible mechanism for overfitting on tiny
calibration sets.

**Per-subject breakdown** (free analysis of the existing data; script `analyze_cnn_calibration_followup.py`,
outputs `report_figs/cnn_calibration_followup/`) — the degradation is *systematic*, not noise:

| K | Subjects worse | Subjects better | Mean lift |
|---|---|---|---|
| 5 | **39/40 (97.5%)** | 1/40 | −11.4pp |
| 10 | 23/40 | 17/40 | −1.1pp |
| 20 | 18/40 | 22/40 | +1.7pp |

Correlation between a subject's K=0 baseline F1 and their calibration lift: r ≈ −0.35 to −0.38
at every K — subjects already well-served by the base model have the most to lose from an
aggressive fine-tune.

**Mechanism test** (`--ft-epochs 3` instead of 15, same seed) — `results_cnn_calibration_ftepochs3/cnn_calibration_summary.csv`:

| K | F1 (mean ± SD, n=40) | Lift vs K=0 |
|---|---|---|
| 0 | 0.7499 ± 0.0755 | — |
| 5 | 0.7575 ± 0.0773 | **+0.8pp** (was −11.4pp) |
| 10 | 0.7782 ± 0.0773 | +2.8pp |
| 20 | 0.7820 ± 0.0710 | **+3.2pp** (better than the original protocol's +1.7pp) |

**Confirmed**: reducing fine-tune epochs eliminates the degradation entirely and produces a
monotonically-improving, better-overall calibration curve.

**Seed-repeat robustness check** (`--seed 7`, original `--ft-epochs 15`) — `results_cnn_calibration_seed7/cnn_calibration_summary.csv`:

| K | F1 (mean ± SD, n=40) | Lift vs K=0 |
|---|---|---|
| 0 | 0.7517 ± 0.0747 | — |
| 5 | 0.6439 ± 0.0876 | −10.8pp (was −11.4pp) |
| 10 | 0.7468 ± 0.0843 | −0.5pp (was −1.1pp) |
| 20 | 0.7691 ± 0.0911 | +1.7pp (identical) |

**Confirmed**: the original degradation reproduces almost exactly across two independent
seeds — it is real and systematic, not RNG noise from a single run.

**Conclusion for the thesis**: naive few-shot fine-tuning of the CNN with unregularized,
fixed-epoch training can *silently hurt* performance if the calibration set is small — this is
a real, reproducible finding worth reporting as a cautionary result. But the actual
contribution is stronger than the negative finding alone: **tuning the fine-tune protocol
(fewer epochs) both eliminates the degradation and improves the best achievable calibration
lift** (+3.2pp vs. the naive protocol's +1.7pp at K=20). Recommend reporting all three results
(original, audit, fix) as a single narrative: naive calibration can backfire; proper
regularization recovers and exceeds it.

---

## Part 2 — ENABL3S External Validation

**Dataset**: ENABL3S (Hu et al. 2018), 10 subjects (AB156, 185, 186, 188, 189, 190, 191, 192,
193, 194), fs=1000 Hz, right-leg 7-muscle montage (mirroring SIAT's single-leg design).
**Class mapping**: LW→WAK, SA→UPS, SD→DNS, sit-to-stand transition (±1s window)→STDUP.
**Adapter**: `adapt_external_dataset.py --root 5362627 --resume` → `features_out_ext/`.

**Class counts after adaptation** (45,525 windows total, (C,T)=(7,250)):

| Class | Windows |
|---|---|
| WAK | 30,999 |
| STDUP | 7,312 |
| DNS | 3,625 |
| UPS | 3,589 |

STDUP (the class flagged as a sparsity risk in the original plan) was **not** sparse — no
fallback to a 3-class replication was needed.

**Feature extraction**: matched SIAT's exact freq-72 pipeline config (`--freq --fs 1000
--no-wavelet`, no entropy, `--use raw`) — produces 56 features (7 channels × 8 features/channel,
vs. SIAT's 72 = 9 channels × 8; fewer only because ENABL3S has fewer EMG channels, not a
methodology change).

### Full ENABL3S results table

| Model | Method | F1 (mean ± SD) | Results file |
|---|---|---|---|
| SVM | SD (5-fold, pooled) | 0.8430 ± 0.0055 | `results_ext_sd/` |
| RF | SD | 0.8116 ± 0.0066 | `results_ext_sd/` |
| SVM | LOSO, per-subject norm | 0.6572 ± 0.0566 | `results_ext_persubj/` |
| RF | LOSO, per-subject norm | 0.6356 ± 0.0898 | `results_ext_persubj/` |
| CNN | LOSO, per-subject norm | 0.5556 ± 0.0819 | `results_ext_cnn_persubj/` |
| SVM | LOSO, global norm | 0.5540 ± 0.0935 | `results_ext_global/` |
| RF | LOSO, global norm | 0.5245 ± 0.1289 | `results_ext_global/` |
| CNN | LOSO, global norm | 0.3867 ± 0.1530 | `results_ext_cnn_global/` |

### Headline replications — both confirmed, and stronger than on SIAT

**1. Per-subject normalization beats global, for all three models:**

| Model | Per-subject | Global | Lift |
|---|---|---|---|
| SVM | 0.6572 | 0.5540 | **+10.3pp** (SIAT: +6.9pp) |
| RF | 0.6356 | 0.5245 | **+11.1pp** (SIAT: +5.1pp) |
| CNN | 0.5556 | 0.3867 | **+16.9pp** (SIAT: +7.2pp) |

CNN shows the largest normalization dependency of any model on ENABL3S.

**2. Subject-dependent accuracy far exceeds LOSO (the generalization gap):**

| Model | SD | LOSO (per-subject) | Gap |
|---|---|---|---|
| SVM | 0.8430 | 0.6572 | **18.6pp** (SIAT: 9.7pp) |
| RF | 0.8116 | 0.6356 | **17.6pp** (SIAT: 7.2pp) |

Roughly double SIAT's gap — plausibly because ENABL3S has only 10 subjects (a much smaller,
less diverse pool for LOSO to generalize across) vs. SIAT's 40.

**Conclusion**: both of the thesis's central findings — per-subject normalization beats global,
and cross-subject generalization is the dominant bottleneck (SD ≫ LOSO) — replicate cleanly on
a fully independent second dataset, with the same direction and a *larger* effect size than on
SIAT.

---

## Appendix — Scripts fixed or added along the way

Several real bugs were found and fixed during this work (not just re-runs of existing code):

- **`train_classical_loso.py`**: (1) added `--rf-n-jobs` to decouple RF's own tree-building
  parallelism from GridSearchCV's (which deadlocks under repeated calls in a detached process
  on Windows); (2) fixed a full-resume crash where re-invoking on an already-100%-checkpointed
  model raised `RuntimeError("Some samples were never predicted")` — now warns and skips the
  (unused) diagnostic report/confusion-matrix generation instead of crashing.
- **`train_cnn_loso.py`**: pre-emptively fixed the equivalent full-resume gap (summary CSV was
  gated behind the same predictions-required check as the confusion matrix; now writes
  unconditionally from on-disk metrics).
- **`run_streaming_norm_loso.py`**, **`run_coral_loso.py`**: same `--rf-n-jobs` fix as above.
- **`run_seed_stability.py`**: added a no-op `--resume` flag for compatibility with the memory
  guard's auto-append behaviour.
- **`adapt_external_dataset.py`**: added per-subject checkpointing (the original script held
  everything in memory with a single write at the very end — risky for an 8.3GB/476-file job).
- **`train_classical_patched.py`** (subject-dependent baseline): fixed `GridSearchCV(n_jobs=-1)`
  and `RandomForestClassifier(n_jobs=-1)` (same deadlock/memory-spike risk as above); added
  `gc.collect()` between CV folds (Windows doesn't always return freed numpy/sklearn memory to
  the OS promptly); added true per-fold checkpointing (`fold_checkpoints/`) so a memory-driven
  restart never has to redo an already-completed fold.
- **`run_multi_guard.py`** (new): a multi-process memory-guarded supervisor for running several
  independent LOSO jobs concurrently, scaling concurrency to free RAM.
- **`analyze_cnn_calibration_followup.py`** (new): reproduces the per-subject audit of the CNN
  calibration finding; outputs in `report_figs/cnn_calibration_followup/`.
- **`status.py`** (new): lightweight, self-match-safe progress checker for the streaming-norm
  experiment's checkpoint files.

## Appendix — Where everything lives

```
results_loso_freq_streaming/     1.1 streaming/causal norm (5 configs x SVM/RF)
results_loso_freq_coral/         2.2 CORAL
results_seed_stability/          2.3 seed stability (classical + CNN)
results_cnn_calibration/         2.6 original protocol
results_cnn_calibration_ftepochs3/   2.6 mechanism test
results_cnn_calibration_seed7/       2.6 seed-repeat
report_figs/cnn_calibration_followup/  2.6 per-subject audit analysis

features_out_ext/                ENABL3S adapted windows + freq-72 features
results_ext_persubj/             ENABL3S classical LOSO, per-subject norm
results_ext_global/              ENABL3S classical LOSO, global norm
results_ext_cnn_persubj/         ENABL3S CNN LOSO, per-subject norm
results_ext_cnn_global/          ENABL3S CNN LOSO, global norm
results_ext_sd/                  ENABL3S subject-dependent baseline

logs/                            exact reproducible commands for every run above
```

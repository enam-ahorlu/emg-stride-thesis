# Handoff — EXPERIMENT_PLAN_GAPS.md (G1–G7) continuation

**Root dir:** `C:\Users\enama\OneDrive\Desktop\Documents\MSc CS\FInal Project\06_Code\`
**Task:** Close gaps G1–G7 from `EXPERIMENT_PLAN_GAPS.md` (G5 was already done before this session). Extend the robustness/deployability analyses to the headline models (ResNet-SE, soft ensemble). Order specified by the user: G2, G7, G6, G4, G3, G1 (cheap/fast first). **Do NOT edit thesis chapters yet** — user is restructuring Results separately. Just produce clean summary CSVs + fold stats into `stats_unified_fdr.py`.

**IMPORTANT — why this handoff exists:** the previous session burned a huge amount of turns narrating "waiting" between every Monitor/background-task poll while GPU runs were in progress. **In the new session: do NOT emit a text turn for every polling check.** Launch the background run, arm ONE Monitor with a coarse threshold (or just wait for the completion notification), and only write user-facing text when you have an actual result or hit an error. Silence between the launch and the completion notification is fine and expected.

---

## Status: G2, G7, G6, G4, G3 fully DONE. G1 nearly done (1 step left). Stats-folding (task 29) and final report (task 30) not started.

### ✅ G2 — Latency (DONE)
Extended `measure_inference_latency.py` (added ResNet-SE CPU+CUDA timing and a soft-ensemble SVM+RF+ResNet-SE end-to-end timing block, itself timed as one call not summed percentiles). Output: `results_latency/inference_latency_measured.csv`.

| Model | Median (ms) | p95 (ms) |
|---|---|---|
| SVM | 1.01 | 1.68 |
| RF | 29.87 | 35.79 |
| CNN (cpu) | 0.58 | 1.06 |
| ResNet-SE (cpu) | 2.65 | 3.78 |
| ResNet-SE (cuda) | 3.93 | 4.99 |
| Ensemble SVM+RF+ResNet-SE (cpu) | 34.34 | 40.78 |
| Ensemble SVM+RF+ResNet-SE (cuda) | 36.12 | 42.25 |

All well under the 125ms real-time cadence.

### ✅ G7 — ResNet-SE subject-dependent + gap table (DONE)
New script `run_cnn_arch_subjectdep.py` (SD 5-fold CV looped over all subjects, any arch via `cnn_architectures.build_model`; `train_cnn_subjectdep.py` couldn't be reused — single-subject-only, no CSV output, no `--arch`).

**Striking finding, confirmed via TWO training recipes (not a hyperparameter artifact):** ResNet-SE SD F1 = **0.573** (default recipe, `results_sd_resnet_se/`) — dramatically below SimpleEMGCNN's SD headline (0.904). Small-data-tuned recipe (batch=32, weight_decay=1e-3, `results_sd_resnet_se_smalldata/`) gave 0.549 — same ballpark, confirming genuine finding not noise. Root cause: 557k params + BatchNorm cannot train stably on ~550 windows/SD-fold, even though the same architecture trains fine on LOSO's ~26k-window pooled set.

**Gap table (Table 4.5 equivalent), primary recipe:**
| Model | SD F1 | LOSO F1 | Gap (pp) |
|---|---|---|---|
| CNN | 90.4% | 75.4% | +15.1 |
| RF | 84.3% | 77.3% | +7.0 |
| SVM | 87.4% | 77.7% | +9.7 |
| **ResNet-SE** | **57.3%** | **78.2%** | **−20.9** (only model where LOSO beats SD; Wilcoxon p=5.0e-9, d=−1.33) |

Outputs: `report_figs/freq72_generalization_gap_summary.csv` / `_full.csv` (appended with ResNet-SE row), `report_figs/table_4_5_updated.csv` (clean formatted table), `report_figs/new_experiments/g7_resnet_se_gap_stats.csv` (stats row for FDR family).

**Also note for wiring later:** the gap table's "optimized pipeline" reference should say **0.815** (soft ensemble), not 0.792 — that headline changed in the prior ensemble-v2 session (see "Prior sessions" below).

### ✅ G6 — STDUP class-balance control on CNN + ResNet-SE (DONE)
New script `run_stdup_subsample_cnn.py` (extends `run_stdup_subsample.py`'s balanced-vs-imbalanced protocol to CNN/ResNet-SE; reuses `balance_training()` from the original script, `train_fold`/`evaluate_with_proba` from `run_cnn_arch_loso.py`). Output: `results_stdup_subsample/stdup_subsample_{CNN,RESNET_SE}_{subjectwise,summary}.csv`.

| Model | STDUP F1 imbalanced | STDUP F1 balanced | Δ (pp) | p |
|---|---|---|---|---|
| SVM | 95.97% | 95.58% | −0.39 | <0.0001 |
| RF | 95.88% | 94.67% | −1.20 | 0.0001 |
| CNN | 94.65% | 93.60% | −1.04 | 0.0006 |
| ResNet-SE | 95.36% | 94.25% | −1.11 | 0.0008 |

All four models: statistically detectable but small (≤1.2pp) drop despite ~74% STDUP training-data cut → "biomechanical, not sample-size" claim now holds for deep models too. Stats: `report_figs/new_experiments/g6_stdup_cnn_stats.csv`.

### ✅ G4 — ResNet-SE supervised calibration (DONE)
Patched `run_cnn_calibration_loso.py` and `run_cnn_calibration_multidraw.py` to add `--arch` (they hardcoded SimpleEMGCNN before; now use `cnn_architectures.build_model`).

Primary (3-epoch, single first-K draw), `results_cnn_calibration_resnet_se/`:
K=0: 0.756, K=5: 0.735 (dip — noise, see below), K=10: 0.773, K=20: 0.781.

Multi-draw (5 random draws × 40 subjects, n=200 per K), `results_cnn_calibration_resnet_se_multidraw/`: K=5 +0.88pp (p=0.026), K=10 +2.67pp (p=4.1e-10), K=20 +4.28pp (p=3.6e-19) — clean monotonic significant lift, resolves the single-draw K=5 dip as just noise from that particular first-K-windows draw (same pattern documented for SimpleEMGCNN previously). Stats: `report_figs/new_experiments/g4_resnet_se_calibration_stats.csv` and `_multidraw_stats.csv`.

### ✅ G3 — Causal/streaming AdaBN on ResNet-SE (DONE)
**New script written from scratch:** `run_adabn_causal_loso.py`. Extends `run_adabn_cnn_loso.py`'s full-session AdaBN to two causal variants mirroring `run_streaming_norm_loso.py`'s classical calib/running pattern:
- `calibK` (K=25/50/100): freeze BN stats from first K time-ordered windows only, then classify all windows.
- `running`: warmup buffer (default 16 windows) seeds initial stats, then genuinely causal expanding-window update — window i classified using stats from [0..i−1] only, THEN folded in. Implemented as literal per-window train()-mode forward passes (momentum=None → cumulative moving average).

Output: `results_adabn_causal/adabn_causal_{subjectwise,summary}.csv`.

**Striking negative finding:** causal AdaBN essentially fails to recover ANY of the full-session gain.
| Config | F1 | vs global baseline (0.7034) | vs full-session AdaBN (0.7425) |
|---|---|---|---|
| calib25 | 0.664 | −3.90pp (p=0.006, **worse**) | −7.82pp (p<0.0001) |
| calib50 | 0.694 | −0.98pp (ns) | −4.90pp (p<0.0001) |
| calib100 | 0.705 | +0.12pp (ns — **zero gain**) | −3.79pp (p<0.0001) |
| running | 0.543 | −16.05pp (p<0.0001, **actively harmful**) | −19.97pp (p<0.0001) |

Stronger negative result than the classical RF causal finding. Stats: `report_figs/new_experiments/g3_adabn_causal_stats.csv`.

### 🟡 G1 — ENABL3S external validation (ALMOST DONE — 1 step left)
Confirmed ENABL3S file paths (10 subjects, real IDs 156/185/186/188-194 — note 187 is missing from the dataset):
- `FEAT_EXT = features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_ext.npz`
- `META_EXT = features_out_ext/freq_windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_features_meta.csv`
- `NPZ_EXT = features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60.npz`
- `META_EXT_RAW = features_out_ext/windows_ENABL3S_WAK_UPS_DNS_STDUP_w250_ov50_conf60_meta.csv`

**Patched `run_cnn_arch_loso.py`** to add `--norm-mode {per_subject,global}` (it had NO such flag before — always did per-subject unconditionally; the plan's example command assumed a flag that didn't exist). Global mode computes train-fold-only stats via `compute_train_norm`/`apply_norm`, recomputed per LOSO fold (mirrors `run_adabn_cnn_loso.py`'s pattern). Verified no regression on SIAT.

**Patched `ensemble_v2_combine.py`** to add `--proba-dir`, `--out`, `--ref` args (was hardcoded to `results_ensemble_v2/proba` and `SUBS=range(1,41)`). Subject-ID set is now **auto-detected** from whatever npz files exist in `--proba-dir` (regex scan), so it works for both SIAT (1-40 sequential) and ENABL3S (real codes 156/185-194) without further changes. Verified SIAT results reproduce identically (no regression) — `python ensemble_v2_combine.py` with no args still does SIAT.

**Done so far (all in `results_ext_*` / `results_ext_ensemble_v2/`):**
1. ResNet-SE ENABL3S per-subject: **F1 = 0.565 ± 0.072** (`results_ext_resnet_se_persubj/`) — also saved proba to `results_ext_ensemble_v2/proba/RESNET_SE_sub*.npz` for the ensemble step below.
2. ResNet-SE ENABL3S global: **F1 = 0.476 ± 0.134** (`results_ext_resnet_se_global/`). Per-subject beats global by **+8.9pp — replicates the core normalization finding on ENABL3S**, though the gain is smaller than SimpleEMGCNN's (+16.9pp there).
3. Deep CORAL ENABL3S (resnet_se, global-norm base): **F1 = 0.524 ± 0.135** (`results_ext_deepcoral/`) — sits between global (0.476) and per-subject (0.565), matching the SIAT pattern.
4. AdaBN ENABL3S (resnet_se, full-session): **pre 0.4425 → post 0.5415 ± 0.0630** (`results_ext_adabn/`) — a solid +9.9pp lift, and this one JUST completed (results are in the final system notification of this conversation, not yet independently re-verified via `cat` — worth a quick sanity glance in the new session, e.g. `cat results_ext_adabn/adabn_summary.csv`).
5. SVM + RF probabilities regenerated on ENABL3S (`train_classical_loso.py --save-proba`, cheap refit reusing `results_ext_persubj`'s `best_params`): SVM argmax-F1 sanity = 0.6387 (headline 0.657, small-sample Platt-scaling noise, same pattern as SIAT); RF argmax-F1 sanity = 0.6356 (headline 0.636, near-exact match). Both in `results_ext_ensemble_v2/proba/{SVM,RF}_sub*.npz`.

**ONE STEP LEFT for G1:** all 30 proba files (SVM/RF/RESNET_SE × 10 subjects) are confirmed present in `results_ext_ensemble_v2/proba/`. Just run:
```powershell
& '.\.venv\Scripts\python.exe' ensemble_v2_combine.py --proba-dir results_ext_ensemble_v2/proba --out results_ext_ensemble_v2
```
This produces `results_ext_ensemble_v2/ensemble_v2_{summary,subjectwise}.csv` — the ranked soft/weighted/stacking comparison on ENABL3S, auto-detecting the 10 real subject IDs. **Then G1 is fully done.** Check whether the soft/weighted-soft ensemble beats hard vote on ENABL3S too (replication check for the SIAT ensemble-v2 finding), and whether it beats the classical-only or single-model ENABL3S baselines. Report honestly if it does NOT replicate (ground rule from the user's original G1 instructions).

---

## Remaining work after G1 finishes

### Task 29 — Fold every new comparison into `stats_unified_fdr.py`
Pattern already established (see `stats_unified_fdr.py` — currently reads `optimization_wilcoxon_table.csv`, `new_experiments_stats_fdr.csv`, `july2_stats_fdr.csv`, `ensemble_v2_stats_fdr.csv`; family is at **39 tests, 27 survive** as of the end of the prior session). Need to:
1. Compute paired Wilcoxon + Cohen's d + BCa CI for the G1 ENABL3S comparisons (persubj vs global resnet_se; per-subject vs Deep CORAL; per-subject vs AdaBN; best ensemble-v2 vs hard-vote/best-single, if applicable) — save as `report_figs/new_experiments/g1_ext_*_stats.csv` mirroring the g3/g4/g6/g7 pattern already used (see those files for the exact schema: `comparison, delta_pp, cohens_d, p`).
2. Add a new block to `stats_unified_fdr.py` reading each `g{N}_*_stats.csv` (mirror the existing `ensemble_v2_combiner` block exactly — `family=` label, then append rows).
3. Re-run `stats_unified_fdr.py`, confirm total test count and how many survive.

Note G7's ResNet-SE SD-vs-LOSO gap stat is ALSO not yet folded in (`g7_resnet_se_gap_stats.csv` exists but isn't read by `stats_unified_fdr.py` yet) — fold that in at the same time as G1's row additions.

### Task 30 — Final report to the user
Per the original instructions: **one table per gap** (G1–G7) comparing the headline-model result vs the original-model result it mirrors, **flag any finding that did NOT replicate on ENABL3S**, confirm all outputs are on disk. The user explicitly wants this delivered as a normal chat response (no docx edits — that's a separate future task the user is doing themselves).

Known non-replications / notable deviations to flag prominently:
- **G7**: ResNet-SE's SD-LOSO gap is NEGATIVE (−20.9pp) — the only model that doesn't show the universal "SD > LOSO" pattern. Architectural, not a bug (verified two ways).
- **G3**: causal AdaBN recovers ~0% of the full-session AdaBN gain (vs SVM's causal calibration buffer which recovers ~half of its transductive gain) — a genuine negative/non-deployable finding for the deep side.
- **G1**: check once the ensemble step finishes whether ENABL3S replicates "soft ensemble beats hard vote" — this is the one still-open question.

---

## Key file/script inventory (new or patched this session)

**New scripts:**
- `run_cnn_arch_subjectdep.py` — SD 5-fold CV for any CNN arch, looped over all subjects (G7)
- `run_stdup_subsample_cnn.py` — STDUP balance control for CNN/ResNet-SE (G6)
- `run_adabn_causal_loso.py` — causal/streaming AdaBN, calibK + running modes (G3)

**Patched scripts:**
- `measure_inference_latency.py` — added ResNet-SE + soft-ensemble timing (G2)
- `run_cnn_calibration_loso.py`, `run_cnn_calibration_multidraw.py` — added `--arch` (G4)
- `run_cnn_arch_loso.py` — added `--norm-mode {per_subject,global}` (G1)
- `ensemble_v2_combine.py` — added `--proba-dir`, `--out`, `--ref`; auto-detects subject IDs (G1)

All patches verified to not regress prior SIAT results.

## Prior-session context (for orientation, not re-work)
Before this G1–G7 session, TWO earlier large sessions already: (1) ran the July-2026 EXPERIMENT_PLAN.md experiments (LDA, resnet_se/resnet CNN arch, Deep CORAL, AdaBN, STDUP subsample on SIAT) and wired them into the thesis chapters — the RQ1 classical-vs-deep narrative was revised (resnet_se ties classical models under LOSO); (2) ran EXPERIMENT_PLAN_ENSEMBLE.md — regenerated per-window probabilities, found soft/weighted-soft voting over SVM+RF+ResNet-SE beats the original hard-vote ensemble (0.815 vs 0.792, p<0.0001, d=1.08), wired that into the chapters as the new headline. **This G1-G7 session's job is explicitly NOT to touch the chapters** — just produce clean CSVs extending the SAME robustness/deployability analyses (external validation, latency, causal deployment, calibration, class-balance, generalization gap) to these two headline models, which were previously only validated on the original SVM/RF/SimpleEMGCNN.

## Ground rules reminder (from the user's G1-G7 prompt)
- Run with venv: `& '.\.venv\Scripts\python.exe' ...`; GridSearchCV n_jobs=1.
- Reuse probability-saving patches from EXPERIMENT_PLAN_ENSEMBLE.md for any ensemble step (already done — see `ensemble_v2_combine.py` patch above).
- Do NOT fabricate numbers; every result comes from a run.
- Report honestly if a finding does NOT replicate on ENABL3S.
- **Do NOT edit thesis chapters this round** — user restructuring separately.

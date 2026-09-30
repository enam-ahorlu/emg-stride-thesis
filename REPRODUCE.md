# Reproducing the Thesis Results

This file maps every headline number, table, and figure in the thesis to the script and output that produces it, so the results can be regenerated from the raw dataset.

Paths below follow the repository layout on `main` since 30 September 2026 (`src/`, `results/`, `data/`, ...),
and every command is run from the repository root. The archived releases, including the version of record
v1.2.0, use the earlier flat layout; `docs/LAYOUT.md` maps one onto the other.

## Environment
- Python 3.10, packages in `requirements.txt` (pin exact versions with `pip freeze`).
- Fixed random seed **42** throughout (NumPy, scikit-learn, PyTorch - `torch.manual_seed` + `torch.cuda.manual_seed_all`). The CNN validation split is seeded per fold as `seed + held-out-subject id`.
- Repository: https://github.com/enam-ahorlu/emg-stride-thesis (public, default branch `main`).
- Archived releases: **v1.2.0** (https://doi.org/10.5281/zenodo.22801920) is the version of record and is
  cut from the commit carrying the September remediation programme and the 229-test family. As with
  v1.1.0, the DOI does not exist until the GitHub release is published, so the tagged commit itself
  cannot cite it; this line is added on `main` immediately afterwards. v1.1.0 (10 September 2026,
  https://doi.org/10.5281/zenodo.22684107) and v1.0.0 (30 August 2026,
  https://doi.org/10.5281/zenodo.22179743) both precede it.
  The concept identifier https://doi.org/10.5281/zenodo.22179742 always resolves to the most recent version.
  The earlier v1.0.0 snapshot (30 August 2026) remains at https://doi.org/10.5281/zenodo.22179743.
- **Scope note.** The superseded v1.0.0 snapshot predates the September parity programme (P-1 to P-8), the ENABL3S movement-blocked
  subject-dependent control, and `archive/scripts/recompute_unified_fdr_v4.py` (itself now superseded by `archive/scripts/recompute_unified_fdr_v5.py`). Those sit on `main` after the tag, so
  a reader working from the archived snapshot alone will find the earlier FDR family rather than the 189-test
  family of record. Release v1.1.0 corrects this: it is cut from the commit that carries the September work and the 189-test family.

## Dataset
- **SIAT-LLMD** - Wei, W., Tan, F., Zhang, H., Mao, H., Fu, M., Samuel, O. W., & Li, G. (2023). *Surface electromyogram, kinematic, and kinetic dataset of lower limb walking for movement intent recognition.* Scientific Data, 10, 358. https://doi.org/10.1038/s41597-023-02263-3
- Local: `data/raw/SIAT_LLMD20230404/Sub01…Sub40/`. 40 subjects, 9 sEMG channels @ 1920 Hz (the released files
  and reference code imply 1920, the dataset descriptor states 1926, and the published features
  were extracted at `src/extract_features.py`'s 2000 Hz default, which is a null operation on the
  result because the rate enters as one constant that normalization divides out); four classes used: WAK, UPS, DNS, STDUP.

## Pipeline order
```
src/preprocess_emg.py          # bandpass 20–450 Hz (4th-order Butterworth, zero-phase) → rectify → 50 ms
                           #   moving-average envelope → 250 ms windows, 125 ms step (50% overlap),
                           #   60% label-purity rule → windowed .npz
src/extract_features.py        # Base-36 / Extended-54 / Freq-72 / Combined-81 feature .npz
                           #   (Freq-72 = MAV,RMS,WL,ZC,WAMP,MNF,MDF,spectral power × 9 channels)
src/train_classical_loso.py    # SVM / RF nested LOSO; --norm-mode {none,global,per_subject,robust};
                           #   --feat-sel {none,rfe,mi}; inner 5-fold GroupKFold, GridSearchCV scoring=f1_macro
src/train_cnn_loso.py          # SimpleEMGCNN LOSO; --norm-mode; --augmentation {none,gaussian,chandrop,timemask,combined}
run_*_loso.py              # ablation drivers (norm / feat-sel / augmentation) that call the trainers
src/run_ensemble_loso.py       # hard-vote ensembles from saved per-subject predictions
src/optimization_statistical_tests.py   # Wilcoxon + Cohen's d  → optimization_wilcoxon_table.csv
src/compare_all_optimizations.py        # 4-stage journey + CI plot
src/analyze_movement_errors.py          # confusion matrices + per-class metrics
```

## Number / table / figure → source

Exhibit and section numbers below are the final ones, as of the 5 September renumbering: Chapter 4 runs
4.1 to 4.16 in four movements and carries Tables 4.1 to 4.21 and Figures 4.1 to 4.13; the appendix carries
Tables A.1 to A.7 and Figures A.1 to A.8. Fourteen appendix figures that duplicated body exhibits were
dropped in the same pass, and rows below say so where a script's figure output is no longer an exhibit.

| Thesis item | Produced by | Output file |
|---|---|---|
| LOSO F1 SVM 77.7 / RF 77.3 (Table 4.3, §4.2.1) | `src/train_classical_loso.py --norm-mode per_subject --feature_set freq72` | `results/loso_freq_persubj/…_summary.csv` |
| LOSO F1 CNN 75.4 (§4.2.2) | `src/train_cnn_loso.py --norm-mode per_subject` | `results/cnn_loso_norm_persubj/cnn_loso_summary.csv` |
| Ensemble 79.2, hard vote (Table 4.14, §4.10; the four-stage history is §A.2, Table A.1) | `src/run_ensemble_loso.py` | `figures/report_figs/ensemble_summary.csv`, `ensemble_3way_per_subject.csv` |
| SD F1 87.4 / 84.3 / 90.4 (§4.1) | `src/train_classical_patched.py` (SD), `src/train_cnn_subjectdep.py` | `figures/report_figs/summary_mean_sd.csv` |
| Per-class F1 (Table 4.5, §4.3) | `src/analyze_movement_errors.py` | `figures/report_figs/freq72_error_analysis/*_per_class_metrics.csv`, `freq72_all_models_per_class_f1.png` |
| Confusion matrices (Fig 4.5, §4.3; ResNet-SE+CD and the soft-vote ensemble) | `src/analyze_movement_errors.py` + `figures/report_figs/stats_finalization` build | `freq72_error_analysis/{SVM,RF,CNN}_confusion_matrix.csv`, `confusion_matrices_loso_3model.png` |
| Gaps 9.7 / 7.0 / 15.1 (Table 4.7, §4.4) | `src/compute_generalization_gap.py` | `figures/report_figs/freq72_generalization_gap_summary.csv` |
| Norm ablation (Table 4.8, §4.6; the bar figure is no longer an exhibit) | `src/run_norm_ablation_loso.py` + `src/compare_norm_ablation.py` | `figures/report_figs/norm_ablation_bar.png` |
| Feature selection (Table 4.13, §4.9; the bar figure is no longer an exhibit) | `src/run_classical_featsel_loso.py` + `src/compare_featsel_results.py` | `figures/report_figs/featsel_bar.png` |
| CNN augmentation (Table 4.12, §4.8; the bar figure is no longer an exhibit) | `src/run_cnn_augmentation_loso.py` + `src/compare_cnn_augmentation.py` | `figures/report_figs/cnn_aug_bar.png` |
| Optimization journey (Fig 4.11, §4.11; the five-stage table is Table A.7) | `src/compare_all_optimizations.py` | `figures/report_figs/optimization_summary.csv`, `optimization_journey.png` |
| Wilcoxon + Cohen's d (§4.16) | `src/optimization_statistical_tests.py` | `figures/report_figs/optimization_wilcoxon_table.csv` |
| **Multiple-comparison correction (§4.16)** | stats-finalization pass (Holm + BH) | `figures/report_figs/stats_finalization/wilcoxon_multiplecomparison_corrected.csv` |
| **BCa bootstrap CIs (§4.2, §4.10)** | stats-finalization pass | `figures/report_figs/stats_finalization/bootstrap_cis.csv` |
| **Inference latency (§4.15, §5.12)** | per-subject `infer_ms_per_window` / CNN `latency_ms` | `figures/report_figs/stats_finalization/inference_latency.csv` |
| **Protocol diagram (Fig 3.1)** | `figures/report_figs/loso_protocol_diagram.png` | embedded in Methodology |
| **External validation ENABL3S - experiment (§4.12)** | `src/adapt_external_dataset.py` + `src/train_classical_loso.py` / `src/train_cnn_loso.py` on ENABL3S features | `results/ext_persubj/`, `results/ext_global/`, `results/ext_cnn_persubj/`, `results/ext_cnn_global/`, `results/ext_sd/` |
| **Fig 4.12, Table 4.16, ENABL3S per-class/confusion (§4.12)** | `src/compare_external_validation.py` | `figures/report_figs/new_experiments/external_persubj_vs_global.png`, `enabl3s_confusion.png`, `external_validation_table.csv`, `enabl3s_per_class_f1.csv`, `enabl3s_confusion_matrix.csv` |
| **CORAL UDA baseline - experiment (§4.7)** | `src/run_coral_loso.py` | `results/loso_freq_coral/coral_summary.csv` |
| **Table 4.9 (§4.7); the CORAL bar figure is no longer an exhibit** | `src/compare_coral_baseline.py` | `figures/report_figs/new_experiments/coral_comparison.png`, `coral_comparison_table.csv` |
| **Causal/streaming norm - experiment (§4.13)** | `src/run_streaming_norm_loso.py` | `results/loso_freq_streaming/streaming_norm_summary_FULL.csv` |
| **Table 4.17 (§4.13); the causal-retention figure is no longer an exhibit** | `src/compare_causal_normalization.py` | `figures/report_figs/new_experiments/causal_retention.png`, `causal_normalization_table.csv` |
| **CNN calibration - experiment (§4.14)** | `src/run_cnn_calibration_loso.py` (`--ft-epochs 15` and `--ft-epochs 3`) | `results/cnn_calibration/`, `results/cnn_calibration_ftepochs3/`, `_seed7/` |
| **Fig A.3, Table A.3 + §4.14 significance (§4.14, §A.4)** | `src/compare_cnn_calibration_schedules.py` | `figures/report_figs/new_experiments/calibration_f1_vs_k.png`, `cnn_calibration_table.csv`, `cnn_calibration_significance.csv` |
| **Table 4.21 + Table A.4, all BCa CIs & FDR (§4.16, §A.5, and the stats sentences of §§4.12-4.15)** | `src/stats_new_experiments.py` | `figures/report_figs/new_experiments/cross_dataset_synthesis.csv`, `new_experiments_stats_fdr.csv`, `new_experiments_cis.csv` |
| **Whole-thesis multiple-comparison correction, first version (29 tests; superseded, see the v5 row below)** | `src/stats_unified_fdr.py` (reads `optimization_wilcoxon_table.csv` + `new_experiments_stats_fdr.csv`) | `figures/report_figs/new_experiments/unified_fdr_all_experiments.csv` |
| **Seed stability (§4.16, §5.12.1): SVM/RF/CNN over seeds 7/42/123; calibration seed 7** | `src/run_seed_stability.py`; `src/run_cnn_calibration_loso.py --seed 7` | `results/seed_stability/seed_stability_summary.csv`, `results/cnn_calibration_seed7/` |
| **LDA under LOSO - §4.2.1 prose, Table 4.8 row (§4.6)** | `src/run_lda_loso.py --norm-mode {per_subject,global}` | `results/lda_persubj/lda_summary.csv`, `results/lda_global/lda_summary.csv` |
| **STDUP class-balance control - §4.3 prose, §5.3 confirmation** | `src/run_stdup_subsample.py --models SVM,RF --conditions imbalanced,balanced` | `results/stdup_subsample/stdup_subsample_summary.csv` |
| **CNN architecture comparison (resnet_se/resnet/simple) - §4.2.2, §4.2.3, RQ1 (Conclusion), §5.1/§5.2 (Discussion)** | `src/run_cnn_arch_loso.py --arch {simple,resnet,resnet_se}` | `results/cnn_loso_simple_repro/`, `results/cnn_loso_resnet/`, `results/cnn_loso_resnet_se/cnn_arch_summary.csv` |
| **Channel-dropout mechanism, W-2 Stage G1 (SE ablation) - §4.8.1** | `scripts/run_w2_g1.sh` (`src/run_cnn_arch_loso.py --arch resnet --augmentation {none,chandrop}`) | `results/cd_resnet_noaug_repro/`, `results/cd_resnet_nose_chandrop/cnn_arch_summary.csv` |
| **Every figure quoted in §4.8.1 and §5.7 (verification, no GPU)** | `src/verify_section_4_8_1.py` | stdout; expected values are in the script's docstring |
| **Channel-occlusion sensitivity, W-2 Stage G3 - §4.8.2, §5.7 mechanism** | `scripts/run_g3_occlusion.sh` then `src/g3_occlusion_stats.py`; see `docs/plans/EXPERIMENT_PLAN_G3_OCCLUSION.md` | `results/g3_noaug_instr/instr/occlusion.csv`, `results/g3_occlusion_per_subject.csv`, `g3_profiles_{chandrop,noaug}.csv` |
| **Gain-jitter control, W-4 (planned) - §5.7 mechanism** | `scripts/run_w4_gainjitter.sh` then `src/w4_gainjitter_stats.py`; see `docs/plans/EXPERIMENT_PLAN_GAIN_JITTER.md` | `results/w4_gainjitter/`, `results/w4_gainjitter_pairs.csv` |
| **Dropout-rate sweep, W-2 Stage G2 - §4.8.2 dose-response** | `scripts/run_g2_rate_sweep.sh` then `src/g2_rate_stats.py`; pre-registration is `docs/plans/EXPERIMENT_PLAN_CHANNEL_DROPOUT.md` §4.1, §4.3 | `results/cd_rate_p{0.1,0.2,0.3,0.5}/`, `results/g2_rate_pairs.csv`, `results/g2_rate_tests.csv` |
| **Skip-connection ablation, W-3 (planned) - §4.8.1 final paragraph, §5.13** | `scripts/run_w3_nores.sh` then `src/w3_residual_stats.py`; see `docs/plans/EXPERIMENT_PLAN_RESIDUAL_ABLATION.md` | `results/w3_nores_noaug/`, `results/w3_nores_chandrop/`, `results/w3_residual_pairs.csv` |
| **Deep CORAL for the CNN - §4.7, Table 4.10 (CNN-side extension)** | `src/run_deep_coral_cnn_loso.py --arch resnet_se --coral-lambda 1.0` | `results/deep_coral_cnn_resnet_se/deep_coral_summary.csv` |
| **AdaBN for the CNN - §4.7, Table 4.10 (CNN-side extension), §4.13 (deployability link)** | `src/run_adabn_cnn_loso.py --arch resnet_se` | `results/adabn_cnn_resnet_se/adabn_summary.csv` |
| **July-2026 second-pass stats (9-test family: LDA, resnet_se/Deep CORAL/AdaBN vs CNN headline, STDUP control)** | `src/stats_july2_experiments.py` | `figures/report_figs/new_experiments/july2_stats_fdr.csv`, `july2_cis.csv`, `july2_supplementary_comparisons.csv` |
| **Whole-thesis multiple-comparison correction, second version (38 tests: 18 optimization + 11 new-experiment + 9 July-2026 second pass; superseded, see the v5 row below)** | `src/stats_unified_fdr.py` (now also reads `july2_stats_fdr.csv`) | `figures/report_figs/new_experiments/unified_fdr_all_experiments.csv` |
| **Latency fix (Discussion §5.12) and Abstract/Conclusion scope fix - text-only, no new run** | manual docx edit against `results/latency/inference_latency_measured.csv` | n/a (see git history / `_prebackup_experiments_*.docx` for before/after) |
| **September parity programme P-1 to P-8 - §4.8.2 dose and divergence results, §5.13 rank-consistency null** | `p1_..._stats.py` through `src/p8_informativeness_stats.py`; see `PARITY_PLAN.md` | `results/parity/p{1,2,3,5,6,7}_outcome.json`, `results/locus/p8_outcome.json`, `results/parity/PARITY_REPORT.md` |
| **ENABL3S movement-blocked subject-dependent control - §4.1.3** | `src/b8_movement_blocked_sd.py --time-units samples --min-guard-frac 0.02` on the ENABL3S features | `results/b8_ext/b8_ext_freq56_w250_compare.csv`, `_subjectwise.csv`, `B8_EXT_VERDICT.md` |
| **Window-length ablation (W-1) - §3.7.1 protocol, §4.15 results, Table 4.20** | `src/window_ablation_stats.py --out results/window_ablation` over the 150/250/400 ms SVM and ResNet-SE+CD arms (`results/win{150,250,400}_*`) | `results/window_ablation/window_ablation_tests.csv`, `_summary.csv`, `_verdict.md`, `_outcome.json`, `fig_window_ablation.png` |
| **Alignment ladder re-run at 400 ms (W-1 Stage 4) - §4.15** | the Section 3.12.4 ladder harness on the 400 ms feature set; accuracy column measured fresh, discrepancy/probe/silhouette columns carried over from the 250 ms ladder and not recomputed | `results/win400_ladder/alignment_ladder_loso_summary.csv`, `_stats.csv`, `alignment_ladder_full.csv`, `ladder_loso_{0..4}_SVM_subjectwise.csv` |
| **Whole-thesis multiple-comparison correction, superseded v4 (187 tests, 141 survive)** | `archive/scripts/recompute_unified_fdr_v4.py` (rebuilds v3's 172 rows from source, then adds the 15 tests the writing phase brought into the thesis) | `figures/report_figs/new_experiments/unified_fdr_family_v4_A_all_reported.csv`, `unified_fdr_family_v4_B_bearing.csv`, `unified_fdr_v4_run.log` |
| **Whole-thesis multiple-comparison correction, VERSION OF RECORD (§4.16, §A.6: 189 tests, 143 survive; bearing-only scope 186 tests, 142 survive)** | `archive/scripts/recompute_unified_fdr_v5.py` (rebuilds v4's 187 rows from source, then adds the 2 alignment-ladder contrasts at 400 ms that §4.15 now reports) | `figures/report_figs/new_experiments/unified_fdr_family_v5_A_all_reported.csv`, `unified_fdr_family_v5_B_bearing.csv`, `unified_fdr_v5_run.log` |

## External validation & deployment experiments - exact commands
ENABL3S root `data/raw/5362627/` (Hu, Rouse & Hargrove 2018), 7 right-leg EMG ch @ 1000 Hz, mapped to WAK/UPS/DNS/STDUP.
```
# --- External validation (ENABL3S) ---
python src/adapt_external_dataset.py --root data/raw/5362627 --resume
python src/extract_features.py --npz data/features_out_ext/windows_ENABL3S_..._w250_ov50_conf60.npz \
    --meta data/features_out_ext/..._meta.csv --out-dir data/features_out_ext --prefix freq --use raw --freq --fs 1000 --no-wavelet
python src/train_classical_loso.py --features $FEXT --meta $MEXT --out results/ext_persubj --models SVM,RF --norm-mode per_subject --n-jobs 1 --rf-n-jobs 6 --resume
python src/train_classical_loso.py --features $FEXT --meta $MEXT --out results/ext_global  --models SVM,RF --norm-mode global      --n-jobs 1 --rf-n-jobs 6 --resume
python src/train_cnn_loso.py --npz $NEXT --meta $MEXT_RAW --out results/ext_cnn_persubj --norm-mode per_subject --resume
python src/train_cnn_loso.py --npz $NEXT --meta $MEXT_RAW --out results/ext_cnn_global  --norm-mode global      --resume
python src/train_classical_patched.py --features $FEXT --meta $MEXT --subjects all --splits 5 --models SVM,RF --svm-scale --out results/ext_sd --save-preds --resume  # SD baseline

# --- CORAL UDA baseline (SIAT) ---
python src/run_coral_loso.py --features $FEAT --meta $META --models SVM,RF --n-jobs 1 --rf-n-jobs 6 --resume

# --- Causal / streaming normalisation (SIAT) ---
python src/run_streaming_norm_loso.py --features $FEAT --meta $META \
    --configs transductive,calib25,calib50,calib100,running --models SVM,RF --n-jobs 1 --rf-n-jobs 6 --resume
# airtight check - re-score every calib config on post-buffer windows only (first K excluded from the F1):
python src/rescore_streaming_buffer.py        # faithful: re-runs GridSearchCV per subject (slow)
python src/rescore_streaming_buffer_v2.py     # fast: refits with the already-selected best_params (validated identical incl-F1)

# --- CNN calibration / transfer learning (SIAT) ---
python src/run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 15 --resume
python src/run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 3 --out results/cnn_calibration_ftepochs3 --resume
python src/run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 15 --seed 7 --out results/cnn_calibration_seed7 --resume
# airtight check - draw-robustness of the calibration lift (5 random draws of the K windows/subject, regularised schedule):
python src/run_cnn_calibration_multidraw.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 3 --n-draws 5 --resume
```
Long runs on 16 GB Windows: keep `GridSearchCV n_jobs=1`, use `--rf-n-jobs 3–6`, `--resume` per-fold checkpointing, optionally wrap in `src/run_with_memory_guard.py`. See `EXPERIMENTS_README.md` for the memory-safety rationale.

## July-2026 second-pass experiments - exact commands
Answers the actionable items from the July-2026 external-review pass (see `EXPERIMENT_PLAN.md`). Shared
inputs are the same `$FEAT`/`$META`/`$NPZ` as above.
```
# --- LDA carried through LOSO (classical-minimal baseline, completes the classical-vs-deep comparison) ---
python src/run_lda_loso.py --features $FEAT --meta $META --norm-mode per_subject --out results/lda_persubj --resume
python src/run_lda_loso.py --features $FEAT --meta $META --norm-mode global      --out results/lda_global  --resume

# --- STDUP class-balance sub-sampling control (is the STDUP F1 lead biomechanical or a sample-size effect?) ---
python src/run_stdup_subsample.py --features $FEAT --meta $META --models SVM,RF --conditions imbalanced,balanced --out results/stdup_subsample --resume

# --- Fairer deep baseline: compact 1D ResNet + squeeze-excitation attention, isolates the architecture confound ---
python src/run_cnn_arch_loso.py --npz $NPZ --meta $META --arch simple    --epochs 40 --out results/cnn_loso_simple_repro --resume   # sanity: reproduces ~0.754
python src/run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results/cnn_loso_resnet_se --resume     # fairer deep baseline
python src/run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet    --epochs 40 --out results/cnn_loso_resnet    --resume     # SE-attention ablation

# --- Deep CORAL for the CNN (CORAL is no longer classical-only) ---
python src/run_deep_coral_cnn_loso.py --npz $NPZ --meta $META --arch resnet_se --coral-lambda 1.0 --epochs 40 --out results/deep_coral_cnn_resnet_se --resume

# --- AdaBN for the CNN (parameter-free, label-free, transductive deep analogue of per-subject normalization) ---
python src/run_adabn_cnn_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results/adabn_cnn_resnet_se --resume

# --- Stats: paired Wilcoxon + Cohen's d + BCa CIs for the 9-test July-2026 family, then fold into the whole-thesis family ---
python src/stats_july2_experiments.py
python src/stats_unified_fdr.py   # now pools 18 (§4.11) + 11 (new-experiment) + 9 (July-2026) + 1 (ensemble-v2) = 39 paired tests
```
All five drivers are `--resume` checkpointed per subject like every other `run_*_loso.py` script. Full-run
headline numbers (40/40 subjects): LDA per-subject 0.6874, LDA global 0.6278; STDUP-class F1 balanced/imbalanced
SVM 0.9558/0.9597, RF 0.9467/0.9588; CNN arch simple 0.7602, resnet 0.7563, resnet_se 0.7822; Deep CORAL
(resnet_se) 0.7637; AdaBN (resnet_se) pre 0.7034 → post 0.7425.

## Ensemble-v2: combiner comparison (soft/weighted/stacking) + ResNet-SE - exact commands
Answers EXPERIMENT_PLAN_ENSEMBLE.md: was hard voting the best combiner, and does folding in
ResNet-SE help? Only hard predictions were saved originally, so per-window class probabilities
had to be regenerated for all four models first.
```
# --- Phase 1a: classical probabilities (cheap refit, reuses best_params from results/loso_freq_persubj, no GridSearch) ---
python src/train_classical_loso.py --features $FEAT --meta $META --models SVM --norm-mode per_subject --save-proba --proba-out results/ensemble_v2/proba --out results/ensemble_v2/svm_run --resume
python src/train_classical_loso.py --features $FEAT --meta $META --models RF  --norm-mode per_subject --save-proba --proba-out results/ensemble_v2/proba --out results/ensemble_v2/rf_run --rf-n-jobs 6 --resume

# --- Phase 1b: CNN probabilities (full retrain per fold, GPU; per-subject norm matches the 0.754 / 0.782 headline runs) ---
python src/run_cnn_arch_loso.py --npz $NPZ --meta $META --arch simple    --epochs 40 --out results/ensemble_v2/cnn_run --save-proba results/ensemble_v2/proba --model-tag CNN --resume
python src/run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results/ensemble_v2/resnet_se_run --save-proba results/ensemble_v2/proba --model-tag RESNET_SE --resume
# sanity check: argmax(proba) per-subject F1 must reproduce the headline for each model before Phase 2

# --- Phase 2: combiner comparison (hard/soft/weighted-soft/stacking across every model subset) ---
python src/ensemble_v2_combine.py
```
Outputs: `results/ensemble_v2/proba/{MODEL}_sub{K:02d}.npz` (keys `proba` [n,4] in LABELS order, `y_true`
[n]); `results/ensemble_v2/ensemble_v2_summary.csv` (ranked, 24 combiner×subset rows, paired Wilcoxon vs
the original SVM+RF+CNN hard vote); `results/ensemble_v2/ensemble_v2_subjectwise.csv`. Winner: soft /
weighted-soft voting over SVM+RF+ResNet-SE, 0.8151 (95% BCa CI [0.7946, 0.8330]) vs the original hard-vote
0.7917 (95% BCa CI [0.7725, 0.8098]); paired Wilcoxon p < 0.0001, Cohen's d = 1.08. Folded into the
whole-thesis FDR family via `figures/report_figs/new_experiments/ensemble_v2_stats_fdr.csv` and `src/stats_unified_fdr.py`
(39 paired tests at the time; the family of record is now the 229 tests of `src/recompute_unified_fdr_v8.py`).

## Regenerating the new-experiment figures & tables (§§4.12-4.16, §A.4, §A.5)
These five scripts read only the LOSO result CSVs above (no retraining) and write every figure and table CSV used in Sections 4.12–4.16 to `figures/report_figs/new_experiments/`. They are fast (seconds) and deterministic - the BCa confidence intervals are seeded from a hash of the input vector, so reruns give identical numbers. Run from the project root:
```
python src/compare_external_validation.py        # Fig 4.15, Fig 4.16, Table 4.12, ENABL3S per-class + confusion
python src/compare_coral_baseline.py             # Fig 4.17, Table 4.13
python src/compare_causal_normalization.py       # Fig 4.18, Table 4.14
python src/compare_cnn_calibration_schedules.py  # Fig 4.19, Table 4.15
python src/stats_new_experiments.py              # Table 4.16 (synthesis), Table 4.17 (FDR family), all BCa CIs
python src/stats_july2_experiments.py            # July-2026 second-pass family (LDA, CNN arch, Deep CORAL, AdaBN, STDUP)
python src/recompute_unified_fdr_v8.py           # whole-thesis correction, version of record: 229 paired tests, 154 survive
```
Each script has an "Expects / Outputs" header naming its exact input dirs and output files. `figures/report_figs/new_experiments/README.md` lists every output and the thesis item it backs. The reported BCa CIs and the Holm/BH-FDR corrected p-values in the thesis are taken verbatim from `new_experiments_cis.csv` and `new_experiments_stats_fdr.csv`.

## EXPERIMENT_PLAN_CRITIQUE.md (E1-E5): external-critique response experiments - exact commands
Answers the five highest-value items from `CRITIQUE_TRIAGE.md` (five external LLM reviews of the
85.8% thesis). Shared inputs are the same `$FEAT`/`$META`/`$NPZ` as above. Seed 42 throughout.
```
# --- Step 0: regenerate results/_bestparams.json (deleted; rebuilt from results/loso_freq_persubj best_params) ---
python src/regenerate_bestparams.py

# --- E1: between-subject variance decomposition (ICC, alignment ladder, distance-vs-difficulty) ---
python src/analyze_between_subject_variance.py

# --- E2: t-SNE/UMAP feature-space visualisation (reuses E1's rung0/rung3/rung4 + probe numbers) ---
python src/make_feature_space_viz.py

# --- E3: causal (deployable) score for the headline SVM+ResNet-SE+CD ensemble ---
python src/run_causal_ensemble.py --stage svm --resume       # CPU: causal SVM proba + transductive honesty check
python src/run_causal_ensemble.py --stage cnn --resume       # GPU: causal ResNet-SE+CD proba
python src/run_causal_ensemble.py --stage combine            # buffer-incl/excl soft-vote scoring -> report.csv

# --- E4: within-subject baseline at matched label budget (regimes A/B/C, N in {5,10,20,50,100}) ---
python src/run_within_subject_baseline.py --resume --rf-n-jobs 6

# --- E5: RF probability calibration and re-vote ---
python src/run_rf_calibrated_ensemble.py --stage metrics                          # ECE/Brier/reliability diagrams
python src/run_rf_calibrated_ensemble.py --stage calibrate --resume --rf-n-jobs 4 # CalibratedClassifierCV RF, isotonic+sigmoid
python src/run_rf_calibrated_ensemble.py --stage combine                          # re-vote via src/ensemble_v2_combine.py

# --- Stats: paired Wilcoxon + Cohen's d + deterministic BCa for every E1-E5 comparison, folded into the whole-thesis family ---
python src/critique_stats.py
python src/stats_unified_fdr.py   # now also reads figures/report_figs/new_experiments/critique_stats_fdr.csv
```
Outputs: `results/variance_decomposition/{variance_components,alignment_ladder,subject_distance_vs_f1,
distance_vs_f1_correlations,embedding_metrics}.csv`; `figures/report_figs/new_experiments/{icc_histogram,
alignment_ladder,distance_vs_f1,feature_space_by_subject,feature_space_by_class}.png`;
`results/causal_ensemble/{proba_calib25,proba_calib50,proba_calib100,proba_transductive_check}/*.npz`,
`honesty_check_transductive_svm.csv`, `report.csv`; `results/within_subject/{within_subject_subjectwise,
within_subject_summary,crossover}.csv`, `figures/report_figs/new_experiments/within_subject_learning_curve.png`;
`results/rf_calibrated/{calibration_metrics.csv,proba_isotonic/,proba_sigmoid/,isotonic/,sigmoid/}`,
`figures/report_figs/new_experiments/reliability_diagrams.png`; `figures/report_figs/new_experiments/critique_stats_fdr.csv`.

## Notes
- `src/train_classical_loso.py` line 141/344 hardcode `random_state=42` (same as the default seed; `src/train_classical_patched.py` is the authoritative SD trainer).
- DOCX edits use the unpack → edit XML → pack workflow with `--validate false` only where the chapter has pre-existing `mc:Ignorable` w14 namespace quirks.

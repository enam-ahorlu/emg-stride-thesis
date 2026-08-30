# Reproducing the Thesis Results

This file maps every headline number, table, and figure in the thesis to the script and output that produces it, so the results can be regenerated from the raw dataset.

## Environment
- Python 3.10, packages in `requirements.txt` (pin exact versions with `pip freeze`).
- Fixed random seed **42** throughout (NumPy, scikit-learn, PyTorch — `torch.manual_seed` + `torch.cuda.manual_seed_all`). The CNN validation split is seeded per fold as `seed + held-out-subject id`.
- Git commit of record: **124f33b** (`main`).

## Dataset
- **SIAT-LLMD** — Wei, W., Tan, F., Zhang, H., Mao, H., Fu, M., Samuel, O. W., & Li, G. (2023). *Surface electromyogram, kinematic, and kinetic dataset of lower limb walking for movement intent recognition.* Scientific Data, 10, 358. https://doi.org/10.1038/s41597-023-02263-3
- Local: `SIAT_LLMD20230404/Sub01…Sub40/`. 40 subjects, 9 sEMG channels @ 2000 Hz; four classes used: WAK, UPS, DNS, STDUP.

## Pipeline order
```
preprocess_emg.py          # bandpass 20–450 Hz (4th-order Butterworth, zero-phase) → rectify → 50 ms
                           #   moving-average envelope → 250 ms windows, 125 ms step (50% overlap),
                           #   60% label-purity rule → windowed .npz
extract_features.py        # Base-36 / Extended-54 / Freq-72 / Combined-81 feature .npz
                           #   (Freq-72 = MAV,RMS,WL,ZC,WAMP,MNF,MDF,spectral power × 9 channels)
train_classical_loso.py    # SVM / RF nested LOSO; --norm-mode {none,global,per_subject,robust};
                           #   --feat-sel {none,rfe,mi}; inner 5-fold GroupKFold, GridSearchCV scoring=f1_macro
train_cnn_loso.py          # SimpleEMGCNN LOSO; --norm-mode; --augmentation {none,gaussian,chandrop,timemask,combined}
run_*_loso.py              # ablation drivers (norm / feat-sel / augmentation) that call the trainers
run_ensemble_loso.py       # hard-vote ensembles from saved per-subject predictions
optimization_statistical_tests.py   # Wilcoxon + Cohen's d  → optimization_wilcoxon_table.csv
compare_all_optimizations.py        # 4-stage journey + CI plot
analyze_movement_errors.py          # confusion matrices + per-class metrics
```

## Number / table / figure → source

| Thesis item | Produced by | Output file |
|---|---|---|
| LOSO F1 SVM 77.7 / RF 77.3 (Table 4.2, §4.2) | `train_classical_loso.py --norm-mode per_subject --feature_set freq72` | `results_loso_freq_persubj/…_summary.csv` |
| LOSO F1 CNN 75.4 (§4.2.2) | `train_cnn_loso.py --norm-mode per_subject` | `results_cnn_loso_norm_persubj/cnn_loso_summary.csv` |
| Ensemble 79.2 (Table 4.10, §4.9) | `run_ensemble_loso.py` | `report_figs/ensemble_summary.csv`, `ensemble_3way_per_subject.csv` |
| SD F1 87.4 / 84.3 / 90.4 (§4.1) | `train_classical_patched.py` (SD), `train_cnn_subjectdep.py` | `report_figs/summary_mean_sd.csv` |
| Per-class F1 (Table 4.4) + Fig 4.4 | `analyze_movement_errors.py` | `report_figs/freq72_error_analysis/*_per_class_metrics.csv`, `freq72_all_models_per_class_f1.png` |
| Confusion matrices (Fig 4.5, 3-model) | `analyze_movement_errors.py` + `report_figs/stats_finalization` build | `freq72_error_analysis/{SVM,RF,CNN}_confusion_matrix.csv`, `confusion_matrices_loso_3model.png` |
| Gaps 9.7 / 7.0 / 15.1 (Table 4.5) | `compute_generalization_gap.py` | `report_figs/freq72_generalization_gap_summary.csv` |
| Norm ablation (Table 4.7, Fig 4.10) | `run_norm_ablation_loso.py` + `compare_norm_ablation.py` | `report_figs/norm_ablation_bar.png` |
| Feature-selection (Table 4.8, Fig 4.11) | `run_classical_featsel_loso.py` + `compare_featsel_results.py` | `report_figs/featsel_bar.png` |
| CNN augmentation (Table 4.9, Fig 4.12) | `run_cnn_augmentation_loso.py` + `compare_cnn_augmentation.py` | `report_figs/cnn_aug_bar.png` |
| Optimization journey (Table 4.11, Fig 4.13) | `compare_all_optimizations.py` | `report_figs/optimization_summary.csv`, `optimization_journey.png` |
| Wilcoxon + Cohen's d (§4.11) | `optimization_statistical_tests.py` | `report_figs/optimization_wilcoxon_table.csv` |
| **Multiple-comparison correction (§4.11)** | stats-finalization pass (Holm + BH) | `report_figs/stats_finalization/wilcoxon_multiplecomparison_corrected.csv` |
| **BCa bootstrap CIs (§4.2, §4.9)** | stats-finalization pass | `report_figs/stats_finalization/bootstrap_cis.csv` |
| **Inference latency (§5.6)** | per-subject `infer_ms_per_window` / CNN `latency_ms` | `report_figs/stats_finalization/inference_latency.csv` |
| **Protocol diagram (Fig 3.1)** | `report_figs/loso_protocol_diagram.png` | embedded in Methodology |
| **External validation ENABL3S — experiment (§4.12)** | `adapt_external_dataset.py` + `train_classical_loso.py` / `train_cnn_loso.py` on ENABL3S features | `results_ext_persubj/`, `results_ext_global/`, `results_ext_cnn_persubj/`, `results_ext_cnn_global/`, `results_ext_sd/` |
| **Fig 4.15 + 4.16, Table 4.12, ENABL3S per-class/confusion (§4.12)** | `compare_external_validation.py` | `report_figs/new_experiments/external_persubj_vs_global.png`, `enabl3s_confusion.png`, `external_validation_table.csv`, `enabl3s_per_class_f1.csv`, `enabl3s_confusion_matrix.csv` |
| **CORAL UDA baseline — experiment (§4.13)** | `run_coral_loso.py` | `results_loso_freq_coral/coral_summary.csv` |
| **Fig 4.17, Table 4.13 (§4.13)** | `compare_coral_baseline.py` | `report_figs/new_experiments/coral_comparison.png`, `coral_comparison_table.csv` |
| **Causal/streaming norm — experiment (§4.14)** | `run_streaming_norm_loso.py` | `results_loso_freq_streaming/streaming_norm_summary_FULL.csv` |
| **Fig 4.18, Table 4.14 (§4.14)** | `compare_causal_normalization.py` | `report_figs/new_experiments/causal_retention.png`, `causal_normalization_table.csv` |
| **CNN calibration — experiment (§4.15)** | `run_cnn_calibration_loso.py` (`--ft-epochs 15` and `--ft-epochs 3`) | `results_cnn_calibration/`, `results_cnn_calibration_ftepochs3/`, `_seed7/` |
| **Fig 4.19, Table 4.15 + §4.15 significance (§4.15)** | `compare_cnn_calibration_schedules.py` | `report_figs/new_experiments/calibration_f1_vs_k.png`, `cnn_calibration_table.csv`, `cnn_calibration_significance.csv` |
| **Tables 4.16 + 4.17, all BCa CIs & FDR (§4.16, §§4.12–4.15 stats sentences)** | `stats_new_experiments.py` | `report_figs/new_experiments/cross_dataset_synthesis.csv`, `new_experiments_stats_fdr.csv`, `new_experiments_cis.csv` |
| **Whole-thesis multiple-comparison correction (§4.16, all 29 tests)** | `stats_unified_fdr.py` (reads `optimization_wilcoxon_table.csv` + `new_experiments_stats_fdr.csv`) | `report_figs/new_experiments/unified_fdr_all_experiments.csv` |
| **Seed stability (§4.11): SVM/RF/CNN over seeds 7/42/123; calibration seed 7** | `run_seed_stability.py`; `run_cnn_calibration_loso.py --seed 7` | `results_seed_stability/seed_stability_summary.csv`, `results_cnn_calibration_seed7/` |
| **LDA under LOSO — §4.2.1 prose, Table 4.7 row (§4.6)** | `run_lda_loso.py --norm-mode {per_subject,global}` | `results_lda_persubj/lda_summary.csv`, `results_lda_global/lda_summary.csv` |
| **STDUP class-balance control — §4.3 prose, §5.3 confirmation** | `run_stdup_subsample.py --models SVM,RF --conditions imbalanced,balanced` | `results_stdup_subsample/stdup_subsample_summary.csv` |
| **CNN architecture comparison (resnet_se/resnet/simple) — §4.2.2, §4.2.3, RQ1 (Conclusion), §5.1/§5.2 (Discussion)** | `run_cnn_arch_loso.py --arch {simple,resnet,resnet_se}` | `results_cnn_loso_simple_repro/`, `results_cnn_loso_resnet/`, `results_cnn_loso_resnet_se/cnn_arch_summary.csv` |
| **Deep CORAL for the CNN — §4.13 (CNN-side extension)** | `run_deep_coral_cnn_loso.py --arch resnet_se --coral-lambda 1.0` | `results_deep_coral_cnn_resnet_se/deep_coral_summary.csv` |
| **AdaBN for the CNN — §4.13 (CNN-side extension), §4.14 (deployability link)** | `run_adabn_cnn_loso.py --arch resnet_se` | `results_adabn_cnn_resnet_se/adabn_summary.csv` |
| **July-2026 second-pass stats (9-test family: LDA, resnet_se/Deep CORAL/AdaBN vs CNN headline, STDUP control)** | `stats_july2_experiments.py` | `report_figs/new_experiments/july2_stats_fdr.csv`, `july2_cis.csv`, `july2_supplementary_comparisons.csv` |
| **Whole-thesis multiple-comparison correction, updated (§4.16-equivalent, 38 tests: 18 optimization + 11 new-experiment + 9 July-2026 second pass)** | `stats_unified_fdr.py` (now also reads `july2_stats_fdr.csv`) | `report_figs/new_experiments/unified_fdr_all_experiments.csv` |
| **Latency fix (Discussion §5.12 "Eighth" limitation) and Abstract/Conclusion scope fix — text-only, no new run** | manual docx edit against `results_latency/inference_latency_measured.csv` | n/a (see git history / `_prebackup_experiments_*.docx` for before/after) |

## External validation & deployment experiments — exact commands
ENABL3S root `5362627/` (Hu, Rouse & Hargrove 2018), 7 right-leg EMG ch @ 1000 Hz, mapped to WAK/UPS/DNS/STDUP.
```
# --- External validation (ENABL3S) ---
python adapt_external_dataset.py --root 5362627 --resume
python extract_features.py --npz features_out_ext/windows_ENABL3S_..._w250_ov50_conf60.npz \
    --meta features_out_ext/..._meta.csv --out-dir features_out_ext --prefix freq --use raw --freq --fs 1000 --no-wavelet
python train_classical_loso.py --features $FEXT --meta $MEXT --out results_ext_persubj --models SVM,RF --norm-mode per_subject --n-jobs 1 --rf-n-jobs 6 --resume
python train_classical_loso.py --features $FEXT --meta $MEXT --out results_ext_global  --models SVM,RF --norm-mode global      --n-jobs 1 --rf-n-jobs 6 --resume
python train_cnn_loso.py --npz $NEXT --meta $MEXT_RAW --out results_ext_cnn_persubj --norm-mode per_subject --resume
python train_cnn_loso.py --npz $NEXT --meta $MEXT_RAW --out results_ext_cnn_global  --norm-mode global      --resume
python train_classical_patched.py --features $FEXT --meta $MEXT --subjects all --splits 5 --models SVM,RF --svm-scale --out results_ext_sd --save-preds --resume  # SD baseline

# --- CORAL UDA baseline (SIAT) ---
python run_coral_loso.py --features $FEAT --meta $META --models SVM,RF --n-jobs 1 --rf-n-jobs 6 --resume

# --- Causal / streaming normalisation (SIAT) ---
python run_streaming_norm_loso.py --features $FEAT --meta $META \
    --configs transductive,calib25,calib50,calib100,running --models SVM,RF --n-jobs 1 --rf-n-jobs 6 --resume
# airtight check — re-score every calib config on post-buffer windows only (first K excluded from the F1):
python rescore_streaming_buffer.py        # faithful: re-runs GridSearchCV per subject (slow)
python rescore_streaming_buffer_v2.py     # fast: refits with the already-selected best_params (validated identical incl-F1)

# --- CNN calibration / transfer learning (SIAT) ---
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 15 --resume
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 3 --out results_cnn_calibration_ftepochs3 --resume
python run_cnn_calibration_loso.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 15 --seed 7 --out results_cnn_calibration_seed7 --resume
# airtight check — draw-robustness of the calibration lift (5 random draws of the K windows/subject, regularised schedule):
python run_cnn_calibration_multidraw.py --npz $NPZ --meta $META --calib-list 0,5,10,20 --ft-epochs 3 --n-draws 5 --resume
```
Long runs on 16 GB Windows: keep `GridSearchCV n_jobs=1`, use `--rf-n-jobs 3–6`, `--resume` per-fold checkpointing, optionally wrap in `run_with_memory_guard.py`. See `EXPERIMENTS_README.md` for the memory-safety rationale.

## July-2026 second-pass experiments — exact commands
Answers the actionable items from the July-2026 external-review pass (see `EXPERIMENT_PLAN.md`). Shared
inputs are the same `$FEAT`/`$META`/`$NPZ` as above.
```
# --- LDA carried through LOSO (classical-minimal baseline, completes the classical-vs-deep comparison) ---
python run_lda_loso.py --features $FEAT --meta $META --norm-mode per_subject --out results_lda_persubj --resume
python run_lda_loso.py --features $FEAT --meta $META --norm-mode global      --out results_lda_global  --resume

# --- STDUP class-balance sub-sampling control (is the STDUP F1 lead biomechanical or a sample-size effect?) ---
python run_stdup_subsample.py --features $FEAT --meta $META --models SVM,RF --conditions imbalanced,balanced --out results_stdup_subsample --resume

# --- Fairer deep baseline: compact 1D ResNet + squeeze-excitation attention, isolates the architecture confound ---
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch simple    --epochs 40 --out results_cnn_loso_simple_repro --resume   # sanity: reproduces ~0.754
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results_cnn_loso_resnet_se --resume     # fairer deep baseline
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet    --epochs 40 --out results_cnn_loso_resnet    --resume     # SE-attention ablation

# --- Deep CORAL for the CNN (CORAL is no longer classical-only) ---
python run_deep_coral_cnn_loso.py --npz $NPZ --meta $META --arch resnet_se --coral-lambda 1.0 --epochs 40 --out results_deep_coral_cnn_resnet_se --resume

# --- AdaBN for the CNN (parameter-free, label-free, transductive deep analogue of per-subject normalization) ---
python run_adabn_cnn_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results_adabn_cnn_resnet_se --resume

# --- Stats: paired Wilcoxon + Cohen's d + BCa CIs for the 9-test July-2026 family, then fold into the whole-thesis family ---
python stats_july2_experiments.py
python stats_unified_fdr.py   # now pools 18 (§4.11) + 11 (new-experiment) + 9 (July-2026) + 1 (ensemble-v2) = 39 paired tests
```
All five drivers are `--resume` checkpointed per subject like every other `run_*_loso.py` script. Full-run
headline numbers (40/40 subjects): LDA per-subject 0.6874, LDA global 0.6278; STDUP-class F1 balanced/imbalanced
SVM 0.9558/0.9597, RF 0.9467/0.9588; CNN arch simple 0.7602, resnet 0.7563, resnet_se 0.7822; Deep CORAL
(resnet_se) 0.7637; AdaBN (resnet_se) pre 0.7034 → post 0.7425.

## Ensemble-v2: combiner comparison (soft/weighted/stacking) + ResNet-SE — exact commands
Answers EXPERIMENT_PLAN_ENSEMBLE.md: was hard voting the best combiner, and does folding in
ResNet-SE help? Only hard predictions were saved originally, so per-window class probabilities
had to be regenerated for all four models first.
```
# --- Phase 1a: classical probabilities (cheap refit, reuses best_params from results_loso_freq_persubj, no GridSearch) ---
python train_classical_loso.py --features $FEAT --meta $META --models SVM --norm-mode per_subject --save-proba --proba-out results_ensemble_v2/proba --out results_ensemble_v2/svm_run --resume
python train_classical_loso.py --features $FEAT --meta $META --models RF  --norm-mode per_subject --save-proba --proba-out results_ensemble_v2/proba --out results_ensemble_v2/rf_run --rf-n-jobs 6 --resume

# --- Phase 1b: CNN probabilities (full retrain per fold, GPU; per-subject norm matches the 0.754 / 0.782 headline runs) ---
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch simple    --epochs 40 --out results_ensemble_v2/cnn_run --save-proba results_ensemble_v2/proba --model-tag CNN --resume
python run_cnn_arch_loso.py --npz $NPZ --meta $META --arch resnet_se --epochs 40 --out results_ensemble_v2/resnet_se_run --save-proba results_ensemble_v2/proba --model-tag RESNET_SE --resume
# sanity check: argmax(proba) per-subject F1 must reproduce the headline for each model before Phase 2

# --- Phase 2: combiner comparison (hard/soft/weighted-soft/stacking across every model subset) ---
python ensemble_v2_combine.py
```
Outputs: `results_ensemble_v2/proba/{MODEL}_sub{K:02d}.npz` (keys `proba` [n,4] in LABELS order, `y_true`
[n]); `results_ensemble_v2/ensemble_v2_summary.csv` (ranked, 24 combiner×subset rows, paired Wilcoxon vs
the original SVM+RF+CNN hard vote); `results_ensemble_v2/ensemble_v2_subjectwise.csv`. Winner: soft /
weighted-soft voting over SVM+RF+ResNet-SE, 0.8151 (95% BCa CI [0.7946, 0.8330]) vs the original hard-vote
0.7917 (95% BCa CI [0.7725, 0.8098]); paired Wilcoxon p < 0.0001, Cohen's d = 1.08. Folded into the
whole-thesis FDR family via `report_figs/new_experiments/ensemble_v2_stats_fdr.csv` and `stats_unified_fdr.py`
(39 paired tests total, 27 survive).

## Regenerating the new-experiment figures & tables (§§4.12–4.16)
These five scripts read only the LOSO result CSVs above (no retraining) and write every figure and table CSV used in Sections 4.12–4.16 to `report_figs/new_experiments/`. They are fast (seconds) and deterministic — the BCa confidence intervals are seeded from a hash of the input vector, so reruns give identical numbers. Run from the project root:
```
python compare_external_validation.py        # Fig 4.15, Fig 4.16, Table 4.12, ENABL3S per-class + confusion
python compare_coral_baseline.py             # Fig 4.17, Table 4.13
python compare_causal_normalization.py       # Fig 4.18, Table 4.14
python compare_cnn_calibration_schedules.py  # Fig 4.19, Table 4.15
python stats_new_experiments.py              # Table 4.16 (synthesis), Table 4.17 (FDR family), all BCa CIs
python stats_july2_experiments.py            # July-2026 second-pass family (LDA, CNN arch, Deep CORAL, AdaBN, STDUP)
python stats_unified_fdr.py                  # whole-thesis correction across all 38 paired tests (18 + 11 + 9)
```
Each script has an "Expects / Outputs" header naming its exact input dirs and output files. `report_figs/new_experiments/README.md` lists every output and the thesis item it backs. The reported BCa CIs and the Holm/BH-FDR corrected p-values in the thesis are taken verbatim from `new_experiments_cis.csv` and `new_experiments_stats_fdr.csv`.

## EXPERIMENT_PLAN_CRITIQUE.md (E1-E5): external-critique response experiments — exact commands
Answers the five highest-value items from `CRITIQUE_TRIAGE.md` (five external LLM reviews of the
85.8% thesis). Shared inputs are the same `$FEAT`/`$META`/`$NPZ` as above. Seed 42 throughout.
```
# --- Step 0: regenerate _bestparams.json (deleted; rebuilt from results_loso_freq_persubj best_params) ---
python regenerate_bestparams.py

# --- E1: between-subject variance decomposition (ICC, alignment ladder, distance-vs-difficulty) ---
python analyze_between_subject_variance.py

# --- E2: t-SNE/UMAP feature-space visualisation (reuses E1's rung0/rung3/rung4 + probe numbers) ---
python make_feature_space_viz.py

# --- E3: causal (deployable) score for the headline SVM+ResNet-SE+CD ensemble ---
python run_causal_ensemble.py --stage svm --resume       # CPU: causal SVM proba + transductive honesty check
python run_causal_ensemble.py --stage cnn --resume       # GPU: causal ResNet-SE+CD proba
python run_causal_ensemble.py --stage combine            # buffer-incl/excl soft-vote scoring -> report.csv

# --- E4: within-subject baseline at matched label budget (regimes A/B/C, N in {5,10,20,50,100}) ---
python run_within_subject_baseline.py --resume --rf-n-jobs 6

# --- E5: RF probability calibration and re-vote ---
python run_rf_calibrated_ensemble.py --stage metrics                          # ECE/Brier/reliability diagrams
python run_rf_calibrated_ensemble.py --stage calibrate --resume --rf-n-jobs 4 # CalibratedClassifierCV RF, isotonic+sigmoid
python run_rf_calibrated_ensemble.py --stage combine                          # re-vote via ensemble_v2_combine.py

# --- Stats: paired Wilcoxon + Cohen's d + deterministic BCa for every E1-E5 comparison, folded into the whole-thesis family ---
python critique_stats.py
python stats_unified_fdr.py   # now also reads report_figs/new_experiments/critique_stats_fdr.csv
```
Outputs: `results_variance_decomposition/{variance_components,alignment_ladder,subject_distance_vs_f1,
distance_vs_f1_correlations,embedding_metrics}.csv`; `report_figs/new_experiments/{icc_histogram,
alignment_ladder,distance_vs_f1,feature_space_by_subject,feature_space_by_class}.png`;
`results_causal_ensemble/{proba_calib25,proba_calib50,proba_calib100,proba_transductive_check}/*.npz`,
`honesty_check_transductive_svm.csv`, `report.csv`; `results_within_subject/{within_subject_subjectwise,
within_subject_summary,crossover}.csv`, `report_figs/new_experiments/within_subject_learning_curve.png`;
`results_rf_calibrated/{calibration_metrics.csv,proba_isotonic/,proba_sigmoid/,isotonic/,sigmoid/}`,
`report_figs/new_experiments/reliability_diagrams.png`; `report_figs/new_experiments/critique_stats_fdr.csv`.

## Notes
- `train_classical_loso.py` line 141/344 hardcode `random_state=42` (same as the default seed; `train_classical_patched.py` is the authoritative SD trainer).
- DOCX edits use the unpack → edit XML → pack workflow with `--validate false` only where the chapter has pre-existing `mc:Ignorable` w14 namespace quirks.

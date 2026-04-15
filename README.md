# sEMG Lower-Limb Movement Recognition Pipeline

MSc thesis project: *Surface Electromyography-Based Lower-Limb Movement Recognition Using Classical Machine Learning and Deep Learning Approaches*.

**Author:** Enam Ahorlu
**Repository:** [github.com/enam-ahorlu/emg-stride-thesis](https://github.com/enam-ahorlu/emg-stride-thesis)

---

## Overview

This project implements a complete sEMG signal processing and classification pipeline for recognising four lower-limb movements — walking (WAK), ascending stairs (UPS), descending stairs (DNS), and standing up (STDUP) — from 9-channel surface EMG recordings. The pipeline spans raw signal preprocessing through feature extraction, model training, and a systematic multi-stage optimisation study comparing classical machine learning (SVM, Random Forest) against a 1D convolutional neural network (CNN).

Two evaluation paradigms are used throughout:
- **Subject-Dependent (SD):** per-subject 5-fold stratified cross-validation (N=40 subjects)
- **Leave-One-Subject-Out (LOSO):** cross-subject generalisation (train on 39, test on 1)

The central research question is how large the *generalisation gap* (SD minus LOSO performance) is for each model family, and what preprocessing and ensemble strategies can reduce it.

---

## Key Results

### Optimisation Journey (LOSO Macro F1)

| Stage | Configuration | SVM | RF | CNN |
|-------|--------------|-----|-----|-----|
| 0 — Baseline | Freq-72, global norm | 0.708 | 0.722 | 0.682 |
| 1 — Normalisation | Freq-72, per-subject norm | 0.777 | 0.773 | 0.754 |
| 2 — Feature sel / augmentation | RFE-36 (SVM) / Gaussian (CNN) | 0.780 | 0.773 | 0.760 |
| 3 — Ensemble | SVM + RF + CNN majority vote | — | — | **0.792** |

### Generalization Gaps (Optimised Pipeline)

| Model | SD F1 | LOSO F1 | Gap (pp) |
|-------|-------|---------|----------|
| SVM | 0.874 | 0.777 | 9.7 |
| RF | 0.843 | 0.773 | 7.0 |
| CNN | 0.904 | 0.754 | 15.1 |

### Statistical Validation

Per-subject normalisation is the dominant optimisation factor (all p < 0.0001): Cohen's d = 1.48 (SVM), 1.15 (RF), 1.34 (CNN). End-to-end improvement from Stage 0 to Stage 3 ensemble: p < 0.0001, d = 1.58 (SVM), 0.75 (RF), 0.97 (CNN).

---

## Dataset

**SIAT-LLMD** (Liang et al., 2023): 40 healthy subjects, 9 sEMG channels at 2000 Hz, 4 movement classes. Raw data lives in `SIAT_LLMD20230404/`.

Reference: Liang, T., Zhang, Q., Liu, X., Lou, L., & Cao, H. (2023). A wearable lower-limb motion dataset. *Scientific Data*, 10, Article 279.

---

## Project Structure

```
MSc Python Project/
├── README.md                          ← this file
│
├── ── PIPELINE SCRIPTS ──────────────────────────────────────
├── run_full_pipeline.py               Master orchestrator (feature extraction → training → analysis)
├── preprocess_emg.py                  Bandpass (20-500 Hz) + notch filtering, windowing, z-score options
├── extract_features.py                Feature extraction: TD (MAV/WL/ZC/SSC/WAMP), wavelet, frequency (MNF/MDF/spectral)
├── merge_freq_wavelet_features.py     Combines extended-54 + frequency features → combined-81 set
├── validate_npz.py                    Validates preprocessed .npz files, visualises sample distributions
│
├── ── TRAINING SCRIPTS ──────────────────────────────────────
├── train_classical_loso.py            Classical LOSO training (SVM/RF) with per-subject norm, Freq-72
├── train_classical_patched.py         Classical SD training with stratified splitting
├── train_cnn_loso.py                  CNN LOSO training with augmentation and normalisation options
├── train_cnn_subjectdep.py            CNN SD training for gap comparison
│
├── ── OPTIMISATION EXPERIMENTS ──────────────────────────────
├── run_norm_ablation_loso.py          Classical normalisation ablation (none/global/per-subject/robust)
├── run_cnn_norm_ablation_loso.py      CNN normalisation ablation (4 conditions)
├── run_classical_featsel_loso.py      Feature selection with per-subject norm (RFE/MI × 36/27)
├── run_classical_featsel_globnorm_loso.py  Feature selection with global norm (independent baseline)
├── run_cnn_augmentation_loso.py       CNN augmentation with per-subject norm (gaussian/chandrop/timemask/combined)
├── run_cnn_aug_globnorm_loso.py       CNN augmentation with global norm (independent baseline)
├── run_ensemble_loso.py               Ensemble evaluation: hard majority vote (4 combinations)
│
├── ── ANALYSIS & COMPARISON ─────────────────────────────────
├── run_all_analyses.py                Runs all supplementary analyses (Wilcoxon tests)
├── compare_all_optimizations.py       Stages 0-3 summary, optimisation journey plot, gap comparison
├── compare_norm_ablation.py           Classical normalisation ablation summary and visualisation
├── compare_cnn_norm_ablation.py       CNN normalisation ablation summary and visualisation
├── compare_featsel_results.py         Feature selection ablation comparison
├── compare_cnn_augmentation.py        CNN augmentation comparison (4 modes)
├── compare_feature_sets_4way.py       4-way feature set comparison with Wilcoxon tests
├── compare_feature_sets_loso.py       Base-36 vs Freq-72 across SD and LOSO
├── compare_easy_hard_confusion.py     Confusion matrices for easy vs hard subjects
├── compute_generalization_gap.py      SD vs LOSO gap calculation for all models
├── compute_per_subject_metrics.py     Per-subject performance extraction
├── freq72_analysis.py                 Comprehensive Freq-72 analysis: stats, CIs, subject ranking
├── analyze_classical_compute.py       Computational cost metrics
├── analyze_hyperparam_behavior.py     Hyperparameter sweep analysis
├── analyze_movement_errors.py         Per-class error analysis and confusion matrices
├── analyze_subject_difficulty.py      Subject difficulty ranking
├── analyze_subject_metadata.py        Subject demographic analysis
├── dataset_balance_analysis.py        Class balance distribution analysis
├── optimization_statistical_tests.py  Wilcoxon signed-rank tests with Cohen's d effect sizes
│
├── ── VISUALISATION ─────────────────────────────────────────
├── plot_results_report_figs.py        Report-ready CNN vs classical comparison plots
├── plot_emg_gait.py                   EMG channel visualisation with gait-phase colouring
├── plot_window_comparison.py          150 ms vs 250 ms window comparison
├── batch_save_plot.py                 Batch EMG+gait plots for all 40 subjects
├── generate_ci_plots.py              95% CI bar and line plots across 4 optimisation stages
├── generate_gap_plots.py             Baseline vs optimised generalization gap plots
├── generate_subject_line_plots.py    Per-subject trajectory plots + improvement heatmap
├── merge_plot_generalization.py      Multi-model generalization gap merge and plot
├── merge_all_gaps.py                 Merges RF/SVM/CNN gaps into single summary file
├── report_dataset.py                 Dataset audit report generation
│
├── ── DATA & FEATURES ──────────────────────────────────────
├── SIAT_LLMD20230404/                 Raw dataset (40 subjects × 9 channels × 4 movements)
├── features_out/                      Extracted feature matrices (.npz) for all 4 feature sets
├── windows_*_w150_*.npz              Preprocessed signal windows (150 ms, 50% overlap)
├── windows_*_w250_*.npz              Preprocessed signal windows (250 ms, 50% overlap)
│
├── ── RESULTS DIRECTORIES ───────────────────────────────────
│   ┌─ Subject-Dependent ─────────────────────────
├── results_classical/                 Legacy SD results (base-36)
├── results_classical_base36_v2/       SD results (base-36 features)
├── results_classical_ext54/           SD results (extended-54 features)
├── results_classical_freq72_v2/       SD results (freq-72 features)
├── results_classical_combined81/      SD results (combined-81 features)
├── results_cnn/                       SD results (CNN)
│   │
│   ┌─ LOSO Baseline ─────────────────────────────
├── results_loso_light/                LOSO baseline (base-36, global norm)
├── results_loso_freq/                 LOSO primary (freq-72, per-subject norm)
│   │
│   ┌─ LOSO Normalisation Ablation ───────────────
├── results_loso_freq_norm_none/       Freq-72, no normalisation
├── results_loso_freq_persubj/         Freq-72, per-subject z-score
├── results_loso_freq_norm_robust/     Freq-72, robust scaler
├── results_loso_norm_none/            Base-36, no normalisation
├── results_loso_norm_persubj/         Base-36, per-subject z-score
├── results_loso_norm_robust/          Base-36, robust scaler
├── results_cnn_loso/                  CNN LOSO baseline (global norm)
├── results_cnn_loso_norm_none/        CNN, no normalisation
├── results_cnn_loso_norm_persubj/     CNN, per-subject z-score
├── results_cnn_loso_norm_robust/      CNN, robust scaler
│   │
│   ┌─ LOSO Feature Selection ────────────────────
├── results_loso_freq_rfe36/           RFE 36 features, per-subject norm
├── results_loso_freq_rfe27/           RFE 27 features, per-subject norm
├── results_loso_freq_mi36/            MI 36 features, per-subject norm
├── results_loso_freq_mi27/            MI 27 features, per-subject norm
├── results_loso_freq_rfe36_globnorm/  RFE 36, global norm (independent baseline)
├── results_loso_freq_rfe27_globnorm/  RFE 27, global norm
├── results_loso_freq_mi36_globnorm/   MI 36, global norm
├── results_loso_freq_mi27_globnorm/   MI 27, global norm
│   │
│   ┌─ LOSO CNN Augmentation ─────────────────────
├── results_cnn_loso_aug_gaussian/     Gaussian noise, per-subject norm
├── results_cnn_loso_aug_chandrop/     Channel dropout, per-subject norm
├── results_cnn_loso_aug_timemask/     Time masking, per-subject norm
├── results_cnn_loso_aug_combined/     Combined augmentation, per-subject norm
├── results_cnn_loso_aug_gaussian_globnorm/   Gaussian, global norm
├── results_cnn_loso_aug_chandrop_globnorm/   Channel dropout, global norm
├── results_cnn_loso_aug_timemask_globnorm/   Time masking, global norm
│
├── ── REPORT OUTPUTS ───────────────────────────────────────
├── report_figs/                       57 PNG plots + 43 CSV summary tables (see below)
├── reports/                           Dataset audit (CSV + Markdown)
├── plots/                             Batch EMG+gait visualisations by subject
└── analysis_results.json              Wilcoxon test results from run_all_analyses.py
```

---

## Feature Sets

| Name | Dimensions | Features per Channel | Description |
|------|-----------|---------------------|-------------|
| Base-36 | 36 | MAV, WL, ZC, SSC | Time-domain fundamentals |
| Extended-54 | 54 | Base + WAMP + Wavelet energy | Adds amplitude and wavelet features |
| **Freq-72** | **72** | **Base + WAMP + MNF, MDF, Spectral power** | **Primary feature set (TD + frequency domain)** |
| Combined-81 | 81 | Extended + MNF, MDF, Spectral power | Union of all features |

All feature sets are computed per 250 ms window (50% overlap) across 9 EMG channels. The Freq-72 set was selected as primary based on a 4-way SD comparison showing it maximises classical model performance while keeping dimensionality manageable.

---

## Models

### SVM (RBF kernel)
- Hyperparameters: C=10, gamma='scale' (from SD sweep)
- LOSO uses nested 5-fold GroupKFold for per-fold hyperparameter selection
- Best individual LOSO F1: 0.780 (with RFE-36 + per-subject norm)

### Random Forest
- Hyperparameters: n_estimators=500, max_depth=None
- LOSO uses nested CV as with SVM
- Best individual LOSO F1: 0.773 (per-subject norm; feature selection hurts RF)

### 1D CNN (SimpleEMGCNN)
- Architecture: 3 conv blocks (32→64→128 filters, kernel sizes 9/7/5), BatchNorm, ReLU, AdaptiveAvgPool1d, Dropout(0.25), FC→4 classes
- Input: raw windowed signals (not hand-crafted features)
- Best individual LOSO F1: 0.760 (Gaussian augmentation + per-subject norm)
- SD F1: 0.904 (highest of all models)

### Ensemble (Hard Majority Vote)
- SVM + RF + CNN 3-way ensemble: **0.792 LOSO F1** (best overall)
- SVM + RF: 0.777; CNN + SVM: 0.777; CNN + RF: 0.773

---

## Optimisation Pipeline

The project follows a 4-stage optimisation pipeline, each building on the previous:

### Stage 0: Baseline
Global z-score normalisation, Freq-72 features, no feature selection or augmentation.

### Stage 1: Normalisation Ablation
Four conditions tested (none, global, per-subject, robust) across all 3 models. Per-subject z-score normalisation emerged as universally optimal: SVM +6.9 pp, RF +5.1 pp, CNN +7.2 pp improvement over global baseline. All improvements p < 0.0001.

### Stage 2: Feature Selection + CNN Augmentation
- **Feature selection** (RFE and MI at 36 and 27 features): SVM benefits marginally from RFE-36 (+0.3 pp); RF performance degrades with feature reduction (−1.3 pp). MI generally underperforms RFE.
- **CNN augmentation** (Gaussian noise, channel dropout, time masking, combined): Gaussian noise provides marginal +0.6 pp; combined augmentation degrades performance, suggesting the CNN has sufficient implicit regularisation.

### Stage 3: Ensemble Combination
Hard majority vote across SVM, RF, and CNN. The 3-way ensemble achieves 0.792 F1, a +1.2 pp gain over the best individual model (SVM at 0.780), by leveraging complementary error patterns across model families.

---

## Report Figures

The `report_figs/` directory contains 57 publication-ready PNG plots and 43 CSV tables documenting every experiment. Key outputs include:

**Core Performance:**
- `acc_boxplot.png`, `macro_f1_boxplot.png`, `bal_acc_boxplot.png` — distribution plots across 40 subjects
- `freq72_sd_vs_loso_f1.png` — SD vs LOSO comparison (primary feature set)
- `freq72_delta_f1_bar.png` — generalization gap per model
- `cnn_vs_best_classical_f1.png` — per-subject CNN vs RF comparison

**Optimisation:**
- `norm_ablation_bar.png`, `cnn_norm_ablation_bar.png` — normalisation ablation results
- `featsel_bar.png` — feature selection comparison
- `cnn_aug_bar.png` — CNN augmentation comparison
- `optimization_journey.png` — 4-stage pipeline progression (key thesis figure)
- `gap_comparison_baseline_vs_optimized.png` — before/after generalization gaps

**Statistical Validation:**
- `optimization_ci_bar.png`, `optimization_ci_line.png` — 95% CI plots across stages
- `subject_improvement_heatmap.png` — per-subject improvement across optimisation stages
- `subject_lines_all_models.png` — per-subject trajectories across 4 stages

**Feature Comparison:**
- `feature_4way_bar.png`, `feature_4way_boxplot.png` — 4-way feature set comparison
- `feature_set_comparison_bar.png` — base-36 vs extended-54 (SD)

**CSV Summary Tables:**
- `optimization_summary.csv` — master 4-stage optimisation table
- `optimization_wilcoxon_table.csv` — all statistical tests with p-values and Cohen's d
- `ensemble_summary.csv` — ensemble results
- `freq72_generalization_gap_summary.csv` — SD/LOSO/gap per model
- `norm_ablation_results.csv`, `cnn_norm_ablation_results.csv` — ablation results
- `featsel_results.csv` — feature selection results
- `cnn_aug_results.csv` — augmentation results

---

## How to Reproduce

### Prerequisites

```
Python 3.10+
numpy, scipy, pandas, scikit-learn
torch (PyTorch, CPU or GPU)
matplotlib, seaborn
```

### Reproducibility

All experiments use a fixed random seed (`seed=42`) by default. This is set consistently across:
- **NumPy**: `np.random.seed(42)`
- **PyTorch**: `torch.manual_seed(42)` and `torch.cuda.manual_seed_all(42)`
- **scikit-learn**: `random_state=42` for StratifiedKFold, RandomForest, and inner CV splits
- **CNN validation splits**: `seed + heldout_subject_id` for deterministic val/train partitioning within each LOSO fold

The seed can be overridden via `--seed` argument in all training scripts.

### Full Pipeline

```bash
# 1. Preprocess raw signals → windowed .npz files
python preprocess_emg.py

# 2. Extract features (produces features_out/*.npz for all 4 feature sets)
python extract_features.py

# 3. Merge frequency + wavelet features into combined-81
python merge_freq_wavelet_features.py

# 4. Run full pipeline (orchestrates training + analysis)
python run_full_pipeline.py
```

### Individual Experiments

```bash
# Normalisation ablation (classical)
python run_norm_ablation_loso.py

# Normalisation ablation (CNN)
python run_cnn_norm_ablation_loso.py

# Feature selection (per-subject norm)
python run_classical_featsel_loso.py

# CNN augmentation (per-subject norm)
python run_cnn_augmentation_loso.py

# Ensemble evaluation
python run_ensemble_loso.py

# Generate all comparison plots and tables
python compare_all_optimizations.py

# Statistical validation
python optimization_statistical_tests.py
```

### Generate Report Figures

```bash
python plot_results_report_figs.py        # Core performance plots
python generate_ci_plots.py               # 95% CI plots
python generate_gap_plots.py              # Generalization gap plots
python generate_subject_line_plots.py     # Per-subject trajectories
```

---

## Evaluation Protocol

**Subject-Dependent (SD):** Each subject's data (~650-850 windows) is split into 5 stratified folds. A separate model is trained per subject. Reports one metric per subject, averaged across N=40.

**Leave-One-Subject-Out (LOSO):** Each subject is held out in turn; the model trains on all 39 remaining subjects. Per-subject z-score normalisation is applied within the LOSO loop (leak-free by design). Classical models use nested 5-fold GroupKFold for per-fold hyperparameter selection. Reports one metric per held-out subject, averaged across N=40.

**Generalization gap** = F1_SD − F1_LOSO. This quantifies the performance cost of not having labelled data from the target subject.

**Primary metric:** Macro F1-score (class-balanced, robust to the STDUP class imbalance at 55.7% of windows).

---

## Windowing

Two window sizes were evaluated: 150 ms and 250 ms (both at 50% overlap, confidence threshold 60%). The 250 ms window was selected as primary based on consistently higher SD performance across all models (`plot_window_comparison.py` and `window_comparison_bar.png`).

---

## Citation

```
Ahorlu, E. (2026). Surface Electromyography-Based Lower-Limb Movement Recognition
Using Classical Machine Learning and Deep Learning Approaches. MSc Thesis.
```

Dataset:
```
Liang, T., Zhang, Q., Liu, X., Lou, L., & Cao, H. (2023).
A wearable lower-limb motion dataset. Scientific Data, 10, Article 279.
https://doi.org/10.1038/s41597-023-02263-3
```

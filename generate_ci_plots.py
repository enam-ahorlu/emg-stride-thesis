#!/usr/bin/env python3
"""
generate_ci_plots.py
====================
Compute per-subject F1 across 4 optimisation stages and generate
95% CI bar and line plots for all 3 models.

Outputs:
  report_figs/optimization_ci_table.csv
  report_figs/optimization_ci_bar.png
  report_figs/optimization_ci_line.png
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent
RPT_DIR = ROOT / "report_figs"
RPT_DIR.mkdir(exist_ok=True)

# ============================================================
# Helpers
# ============================================================

def _get_subj_col(df):
    return "heldout_subject" if "heldout_subject" in df.columns else "subject"


def load_classical_subjectwise(results_dir, model):
    """Return sorted np.array of 40 per-subject F1 values."""
    rdir = ROOT / results_dir
    files = sorted(rdir.glob(f"*{model}*subjectwise*.csv"))
    if not files:
        # Try checkpoints
        ckpt = rdir / "checkpoints"
        if ckpt.exists():
            files = sorted(ckpt.glob(f"*{model}*ckpt.csv"))
    if not files:
        raise FileNotFoundError(f"No {model} subjectwise CSV in {rdir}")
    df = pd.read_csv(files[0])
    sc = _get_subj_col(df)
    df = df.drop_duplicates(subset=[sc]).sort_values(sc)
    subjects = df[sc].astype(int).tolist()
    f1s = df["f1_macro"].tolist()
    return np.array(f1s), subjects


def load_cnn_subjectwise(results_dir):
    """Return sorted np.array of per-subject F1 values."""
    path = ROOT / results_dir / "per_subject_metrics_cnn_loso.csv"
    df = pd.read_csv(path)
    sc = _get_subj_col(df)
    df = df.drop_duplicates(subset=[sc]).sort_values(sc)
    subjects = df[sc].astype(int).tolist()
    f1s = df["f1_macro"].tolist()
    return np.array(f1s), subjects


def align_to_subjects(arr, subj_list, target_subjects):
    """Return arr reindexed to target_subjects order."""
    mapping = {s: v for s, v in zip(subj_list, arr)}
    return np.array([mapping[s] for s in target_subjects])


def compute_ci(arr):
    n = len(arr)
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    se = s / np.sqrt(n)
    return m, s, m - 1.96 * se, m + 1.96 * se


# ============================================================
# Step 1: Load all stages
# ============================================================
print("=" * 60)
print("LOADING PER-SUBJECT F1 FOR ALL STAGES")
print("=" * 60)

# --- Baseline (global norm, full-72) ---
svm_base_arr, svm_base_subj = load_classical_subjectwise("results_loso_freq", "SVM")
rf_base_arr,  rf_base_subj  = load_classical_subjectwise("results_loso_freq", "RF")
cnn_base_arr, cnn_base_subj = load_cnn_subjectwise("results_cnn_loso")

# Shared subjects
all_subjects = sorted(
    set(svm_base_subj) & set(rf_base_subj) & set(cnn_base_subj)
)
print(f"  Baseline subjects: {len(all_subjects)}")

svm_base = align_to_subjects(svm_base_arr, svm_base_subj, all_subjects)
rf_base  = align_to_subjects(rf_base_arr,  rf_base_subj,  all_subjects)
cnn_base = align_to_subjects(cnn_base_arr, cnn_base_subj, all_subjects)

print(f"  SVM baseline mean: {svm_base.mean():.4f}")
print(f"  RF  baseline mean: {rf_base.mean():.4f}")
print(f"  CNN baseline mean: {cnn_base.mean():.4f}")

# --- Norm optimised (per_subject norm, full-72) ---
svm_norm_arr, svm_norm_subj = load_classical_subjectwise("results_loso_freq_persubj", "SVM")
rf_norm_arr,  rf_norm_subj  = load_classical_subjectwise("results_loso_freq_persubj", "RF")
cnn_norm_arr, cnn_norm_subj = load_cnn_subjectwise("results_cnn_loso_norm_persubj")

svm_norm = align_to_subjects(svm_norm_arr, svm_norm_subj, all_subjects)
rf_norm  = align_to_subjects(rf_norm_arr,  rf_norm_subj,  all_subjects)
cnn_norm = align_to_subjects(cnn_norm_arr, cnn_norm_subj, all_subjects)

print(f"\n  SVM norm mean: {svm_norm.mean():.4f}")
print(f"  RF  norm mean: {rf_norm.mean():.4f}")
print(f"  CNN norm mean: {cnn_norm.mean():.4f}")

# --- Feat Sel / Aug (per_subject norm + best feat sel) ---
# SVM: rfe36 persubj
svm_fs_arr, svm_fs_subj = load_classical_subjectwise("results_loso_freq_rfe36", "SVM")
svm_fs = align_to_subjects(svm_fs_arr, svm_fs_subj, all_subjects)

# RF: full-72 persubj (best for RF, not rfe36)
rf_fs = rf_norm.copy()

# CNN: gaussian aug (per_subject norm)
cnn_aug_arr, cnn_aug_subj = load_cnn_subjectwise("results_cnn_loso_aug_gaussian")
cnn_aug = align_to_subjects(cnn_aug_arr, cnn_aug_subj, all_subjects)

print(f"\n  SVM feat-sel mean: {svm_fs.mean():.4f}")
print(f"  RF  feat-sel (full72 persubj) mean: {rf_fs.mean():.4f}")
print(f"  CNN gaussian aug mean: {cnn_aug.mean():.4f}")

# --- Ensemble (3-way) ---
ens_path = RPT_DIR / "ensemble_3way_per_subject.csv"
ens_df = pd.read_csv(ens_path)
sc = _get_subj_col(ens_df)
ens_df = ens_df.drop_duplicates(subset=[sc]).sort_values(sc)
ens_subj = ens_df[sc].astype(int).tolist()
ens_f1   = ens_df["f1_macro"].tolist()
ens_all = align_to_subjects(np.array(ens_f1), ens_subj, all_subjects)

print(f"\n  Ensemble mean: {ens_all.mean():.4f}")

# ============================================================
# Step 2: Compute 95% CI per stage x model
# ============================================================
print("\n" + "=" * 60)
print("COMPUTING 95% CI")
print("=" * 60)

stages = ["Baseline", "Norm\n(per-subject)", "Feat Sel / Aug\n(+ per-subject)", "Ensemble\n(3-way)"]
stage_keys = ["Baseline", "Norm_persubject", "FeatSel_Aug_persubject", "Ensemble_3way"]

data = {
    "SVM": [svm_base, svm_norm, svm_fs, ens_all],
    "RF":  [rf_base,  rf_norm,  rf_fs,  ens_all],
    "CNN": [cnn_base, cnn_norm, cnn_aug, ens_all],
}

ci_rows = []
for model, arrs in data.items():
    for sk, stage_label, arr in zip(stage_keys, stages, arrs):
        m, s, lo, hi = compute_ci(arr)
        n = len(arr)
        ci_rows.append({
            "stage": sk,
            "stage_label": stage_label.replace("\n", " "),
            "model": model,
            "mean_f1": round(m, 4),
            "std_f1":  round(s, 4),
            "ci_low":  round(lo, 4),
            "ci_high": round(hi, 4),
            "n": n,
        })
        print(f"  {model:4s} | {sk:30s} | mean={m:.4f} ± {s:.4f}  CI=[{lo:.4f}, {hi:.4f}]")

ci_df = pd.DataFrame(ci_rows)
ci_path = RPT_DIR / "optimization_ci_table.csv"
ci_df.to_csv(ci_path, index=False)
print(f"\n  [saved] {ci_path.name}")

# ============================================================
# Step 3: Plot A — grouped bar chart with CI error bars
# ============================================================
print("\n" + "=" * 60)
print("GENERATING PLOT A: optimization_ci_bar.png")
print("=" * 60)

models   = ["SVM", "RF", "CNN"]
colors   = {"SVM": "#2196F3", "RF": "#4CAF50", "CNN": "#FF9800"}
n_stages = len(stages)
n_models = len(models)
bar_w    = 0.22
x        = np.arange(n_stages)

fig, ax = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    offset = (i - 1) * bar_w
    means  = [ci_df[(ci_df.stage == sk) & (ci_df.model == model)]["mean_f1"].values[0]
              for sk in stage_keys]
    lows   = [ci_df[(ci_df.stage == sk) & (ci_df.model == model)]["ci_low"].values[0]
              for sk in stage_keys]
    highs  = [ci_df[(ci_df.stage == sk) & (ci_df.model == model)]["ci_high"].values[0]
              for sk in stage_keys]
    errs   = [np.array([m - lo, hi - m]) for m, lo, hi in zip(means, lows, highs)]
    err_arr = np.array(errs).T  # shape (2, n_stages)

    bars = ax.bar(x + offset, means, bar_w, label=model,
                  color=colors[model], alpha=0.85,
                  yerr=err_arr, capsize=5, error_kw={"elinewidth": 1.5})

    for bar, m_val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{m_val:.3f}", ha="center", va="bottom", fontsize=8)

ax.axhline(0.75, ls="--", color="grey", linewidth=1.2, label="F1=0.75 reference")
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylabel("Macro F1 (mean ± 95% CI)", fontsize=12)
ax.set_title("Optimisation Stages: Mean Macro F1 with 95% CI", fontsize=13, fontweight="bold")
ax.set_ylim(0.60, 0.88)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
out_bar = RPT_DIR / "optimization_ci_bar.png"
plt.savefig(out_bar, dpi=300)
plt.close()
print(f"  [saved] {out_bar.name}")

# ============================================================
# Step 3: Plot B — line plot with shaded CI bands
# ============================================================
print("\n" + "=" * 60)
print("GENERATING PLOT B: optimization_ci_line.png")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 6))

x_line = np.arange(n_stages)

for model in models:
    means  = [ci_df[(ci_df.stage == sk) & (ci_df.model == model)]["mean_f1"].values[0]
              for sk in stage_keys]
    lows   = [ci_df[(ci_df.stage == sk) & (ci_df.model == model)]["ci_low"].values[0]
              for sk in stage_keys]
    highs  = [ci_df[(ci_df.stage == sk) & (ci_df.model == model)]["ci_high"].values[0]
              for sk in stage_keys]

    ax.plot(x_line, means, marker="o", linewidth=2.5, color=colors[model],
            label=model, zorder=3)
    ax.fill_between(x_line, lows, highs, alpha=0.18, color=colors[model])

    for xi, m_val in zip(x_line, means):
        ax.text(xi, m_val + 0.003, f"{m_val:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=colors[model])

ax.axhline(0.75, ls="--", color="grey", linewidth=1.2, label="F1=0.75 reference")
ax.set_xticks(x_line)
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylabel("Macro F1 (mean ± 95% CI)", fontsize=12)
ax.set_title("Optimisation Journey: Macro F1 per Stage (shaded = 95% CI)", fontsize=13, fontweight="bold")
ax.set_ylim(0.60, 0.88)
ax.legend(fontsize=11)
ax.grid(alpha=0.4)
plt.tight_layout()
out_line = RPT_DIR / "optimization_ci_line.png"
plt.savefig(out_line, dpi=300)
plt.close()
print(f"  [saved] {out_line.name}")

print("\n=== generate_ci_plots.py COMPLETE ===")

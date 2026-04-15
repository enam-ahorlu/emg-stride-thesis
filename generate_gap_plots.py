#!/usr/bin/env python3
"""
generate_gap_plots.py
======================
Generate baseline and optimised generalization gap plots.

1. Baseline gap plots (base-36, global norm) — preserving the original analysis
2. Optimised gap plots (freq-72, per_subject norm) — with bal_acc added

Outputs:
  report_figs/baseline_all_models_generalization_gap.csv
  report_figs/baseline_all_models_delta_f1_bar.png
  report_figs/baseline_all_models_delta_balacc_bar.png
  report_figs/baseline_all_models_sd_vs_loso_f1.png
  report_figs/all_models_delta_balacc_bar.png   (freq-72 per_subject version)
  report_figs/freq72_generalization_gap_full.csv (with bal_acc columns)
"""
from __future__ import annotations

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
# PART 1: Baseline gap plots (base-36, global norm)
# ============================================================

print("=" * 60)
print("PART 1: Baseline gap plots (base-36, global norm)")
print("=" * 60)

# Load baseline gap data from original result directories
classical_gap = pd.read_csv(ROOT / "results_loso_light" / "generalization_plots" / "mean_gap_by_model.csv")
cnn_gap = pd.read_csv(ROOT / "results_cnn_loso" / "generalization_plots_cnn" / "mean_gap_by_model.csv")

# Parse the full summary CSVs for absolute SD and LOSO values
def parse_gap_summary(path):
    """Parse the multi-level header gap summary CSV."""
    raw = pd.read_csv(path, header=None)
    # Row 0: metric names, Row 1: stat names, Row 2+: data
    rows_out = []
    for i in range(2, len(raw)):
        model = raw.iloc[i, 0]
        rows_out.append({
            "model": model,
            "f1_sd_mean": float(raw.iloc[i, 1]),
            "f1_sd_sd": float(raw.iloc[i, 2]),
            "f1_loso_mean": float(raw.iloc[i, 4]),
            "f1_loso_sd": float(raw.iloc[i, 5]),
            "delta_f1_mean": float(raw.iloc[i, 7]),
            "delta_f1_sd": float(raw.iloc[i, 8]),
            "bal_acc_sd_mean": float(raw.iloc[i, 10]),
            "bal_acc_sd_sd": float(raw.iloc[i, 11]),
            "bal_acc_loso_mean": float(raw.iloc[i, 13]),
            "bal_acc_loso_sd": float(raw.iloc[i, 14]),
            "delta_bal_acc_mean": float(raw.iloc[i, 16]),
            "delta_bal_acc_sd": float(raw.iloc[i, 17]),
        })
    return pd.DataFrame(rows_out)

classical_summary = parse_gap_summary(ROOT / "results_loso_light" / "generalization_gap_summary.csv")
cnn_summary = parse_gap_summary(ROOT / "results_cnn_loso" / "generalization_gap_cnn_summary.csv")

# Combine
baseline_gap_df = pd.concat([classical_summary, cnn_summary], ignore_index=True)
baseline_gap_df.to_csv(RPT_DIR / "baseline_all_models_generalization_gap.csv", index=False)
print(f"  [saved] baseline_all_models_generalization_gap.csv")
print(baseline_gap_df[["model", "f1_sd_mean", "f1_loso_mean", "delta_f1_mean",
                        "bal_acc_sd_mean", "bal_acc_loso_mean", "delta_bal_acc_mean"]].to_string(index=False))

# --- Delta F1 bar chart (baseline) ---
models = baseline_gap_df["model"].tolist()
delta_f1 = baseline_gap_df["delta_f1_mean"].tolist()
delta_f1_sd = baseline_gap_df["delta_f1_sd"].tolist()

fig, ax = plt.subplots(figsize=(7, 5))
colors_base = ["#2196F3", "#4CAF50", "#FF9800"]
bars = ax.bar(models, delta_f1, yerr=delta_f1_sd, capsize=5, color=colors_base[:len(models)],
              edgecolor="black", alpha=0.85)
for bar, val in zip(bars, delta_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("ΔF1 (SD − LOSO)", fontsize=12)
ax.set_title("Baseline Generalization Gap — ΔF1 (SD − LOSO)\n(Base-36 Features, Global Norm)", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(RPT_DIR / "baseline_all_models_delta_f1_bar.png", dpi=150)
plt.close(fig)
print(f"  [saved] baseline_all_models_delta_f1_bar.png")

# --- Delta Bal Acc bar chart (baseline) ---
delta_ba = baseline_gap_df["delta_bal_acc_mean"].tolist()
delta_ba_sd = baseline_gap_df["delta_bal_acc_sd"].tolist()

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(models, delta_ba, yerr=delta_ba_sd, capsize=5, color=colors_base[:len(models)],
              edgecolor="black", alpha=0.85)
for bar, val in zip(bars, delta_ba):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("ΔBal Acc (SD − LOSO)", fontsize=12)
ax.set_title("Baseline Generalization Gap — ΔBal Acc (SD − LOSO)\n(Base-36 Features, Global Norm)", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(RPT_DIR / "baseline_all_models_delta_balacc_bar.png", dpi=150)
plt.close(fig)
print(f"  [saved] baseline_all_models_delta_balacc_bar.png")

# --- SD vs LOSO F1 (baseline) ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(models))
width = 0.35
sd_vals = baseline_gap_df["f1_sd_mean"].tolist()
loso_vals = baseline_gap_df["f1_loso_mean"].tolist()

bars1 = ax.bar(x - width/2, sd_vals, width, label="SD (5-fold CV)", color="#64B5F6", edgecolor="black")
bars2 = ax.bar(x + width/2, loso_vals, width, label="LOSO", color="#EF5350", edgecolor="black")

for bar, val in zip(bars1, sd_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=10)
for bar, val in zip(bars2, loso_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Macro F1", fontsize=12)
ax.set_title("Baseline: SD vs LOSO F1\n(Base-36 Features, Global Norm)", fontsize=13)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0.5, 1.05)
plt.tight_layout()
fig.savefig(RPT_DIR / "baseline_all_models_sd_vs_loso_f1.png", dpi=150)
plt.close(fig)
print(f"  [saved] baseline_all_models_sd_vs_loso_f1.png")

# ============================================================
# PART 2: Optimised gap with bal_acc (freq-72, per_subject norm)
# ============================================================

print(f"\n{'='*60}")
print("PART 2: Optimised gap with bal_acc (freq-72, per_subject norm)")
print("=" * 60)

# Load existing freq-72 gap CSV (has f1 only)
gap_df = pd.read_csv(RPT_DIR / "freq72_generalization_gap.csv")
print(f"  Loaded freq72_generalization_gap.csv: {len(gap_df)} rows, cols={gap_df.columns.tolist()}")

# Add bal_acc from source files
# Classical SD
sd_classical = pd.read_csv(ROOT / "results_classical_freq72_v2" / "per_subject_metrics_freq72_sd.csv")
# Map model names
model_map = {"SVM_RBF_balanced_scaled": "SVM", "RF_balanced": "RF", "LDA_scaled": "LDA"}
sd_classical["model_short"] = sd_classical["model"].map(model_map)

# Classical LOSO per_subject
loso_svm = pd.read_csv(sorted((ROOT / "results_loso_freq_persubj").glob("*SVM*subjectwise*.csv"))[0])
loso_rf  = pd.read_csv(sorted((ROOT / "results_loso_freq_persubj").glob("*RF*subjectwise*.csv"))[0])

# CNN SD
sd_cnn = pd.read_csv(ROOT / "results_cnn" / "cnn_subjectdep_w250_env_zscore_5fold.csv")

# CNN LOSO per_subject
loso_cnn = pd.read_csv(ROOT / "results_cnn_loso_norm_persubj" / "per_subject_metrics_cnn_loso.csv")

# Build bal_acc lookup dicts
ba_sd = {}
ba_loso = {}

for _, r in sd_classical.iterrows():
    m = r.get("model_short", "")
    s = int(r["subject"])
    if m in ["SVM", "RF"]:
        ba_sd[(m, s)] = r["bal_acc"]

for _, r in sd_cnn.iterrows():
    ba_sd[("CNN", int(r["subject"]))] = r["bal_acc"]

for _, r in loso_svm.drop_duplicates("heldout_subject").iterrows():
    ba_loso[("SVM", int(r["heldout_subject"]))] = r["bal_acc"]

for _, r in loso_rf.drop_duplicates("heldout_subject").iterrows():
    ba_loso[("RF", int(r["heldout_subject"]))] = r["bal_acc"]

for _, r in loso_cnn.drop_duplicates("subject").iterrows():
    ba_loso[("CNN", int(r["subject"]))] = r["bal_acc"]

# Merge into gap_df
gap_df["bal_acc_sd"] = gap_df.apply(
    lambda r: ba_sd.get((r["model"], int(r["subject"])), None), axis=1)
gap_df["bal_acc_loso"] = gap_df.apply(
    lambda r: ba_loso.get((r["model"], int(r["subject"])), None), axis=1)
gap_df["delta_bal_acc"] = gap_df["bal_acc_sd"] - gap_df["bal_acc_loso"]

gap_df.to_csv(RPT_DIR / "freq72_generalization_gap_full.csv", index=False)
print(f"  [saved] freq72_generalization_gap_full.csv  ({len(gap_df)} rows)")

# Summary stats
print(f"\n  Optimised per-subject norm gap (with bal_acc):")
for model in ["SVM", "RF", "CNN"]:
    sub = gap_df[gap_df["model"] == model].dropna(subset=["delta_bal_acc"])
    f1_gap = sub["delta_f1"].mean()
    ba_gap = sub["delta_bal_acc"].mean()
    print(f"    {model}: ΔF1={f1_gap:.4f}, ΔBal_Acc={ba_gap:.4f}")

# --- Delta Bal Acc bar chart (optimised, freq-72 per_subject) ---
models_opt = ["SVM", "RF", "CNN"]
delta_ba_opt = []
delta_ba_sd_opt = []
for model in models_opt:
    sub = gap_df[gap_df["model"] == model].dropna(subset=["delta_bal_acc"])
    delta_ba_opt.append(sub["delta_bal_acc"].mean())
    delta_ba_sd_opt.append(sub["delta_bal_acc"].std(ddof=1))

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(models_opt, delta_ba_opt, yerr=delta_ba_sd_opt, capsize=5,
              color=["#2196F3", "#4CAF50", "#FF9800"], edgecolor="black", alpha=0.85)
for bar, val in zip(bars, delta_ba_opt):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("ΔBal Acc (SD − LOSO)", fontsize=12)
ax.set_title("Optimised Generalization Gap — ΔBal Acc (SD − LOSO)\n(Freq-72 Features, Per-Subject Norm)", fontsize=13)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(RPT_DIR / "all_models_delta_balacc_bar.png", dpi=150)
plt.close(fig)
print(f"  [saved] all_models_delta_balacc_bar.png")

print("\n=== generate_gap_plots.py COMPLETE ===")

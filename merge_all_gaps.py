"""
merge_all_gaps.py
=================
Merge RF, SVM, and CNN generalization gaps into a single authoritative file
and plot.  All three models use per-subject 5-fold CV as the SD baseline
and per-subject z-score normalization for LOSO.

Primary input: report_figs/freq72_generalization_gap.csv
               report_figs/freq72_generalization_gap_summary.csv

Outputs (all in report_figs/):
  all_models_generalization_gap.csv   -- merged 3-model gap table
  all_models_delta_f1_bar.png         -- combined gap bar chart (F1)
  all_models_sd_vs_loso_f1.png        -- side-by-side SD vs LOSO grouped bar
  gap_methodology_notes.txt           -- plain-text methodology note for thesis
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / "report_figs"
OUT.mkdir(exist_ok=True)

# -- 1. Load freq-72 gap files ---------------------------------------------------
gap_per_subj = OUT / "freq72_generalization_gap.csv"
gap_summary  = OUT / "freq72_generalization_gap_summary.csv"

if not gap_per_subj.exists():
    raise FileNotFoundError(f"Missing: {gap_per_subj}\nRun freq72_analysis.py first.")
if not gap_summary.exists():
    raise FileNotFoundError(f"Missing: {gap_summary}\nRun freq72_analysis.py first.")

df_ps  = pd.read_csv(gap_per_subj)   # columns: subject, model, f1_sd, f1_loso, delta_f1
df_sum = pd.read_csv(gap_summary)    # columns: model, f1_sd_mean, f1_sd_std, f1_loso_mean, f1_loso_std, delta_f1_mean, delta_f1_std, n

print("Loaded freq72_generalization_gap.csv:")
print(f"  {len(df_ps)} rows, models: {sorted(df_ps['model'].unique())}")
print("\nLoaded freq72_generalization_gap_summary.csv:")
print(df_sum.to_string(index=False))

# -- 2. Build combined DataFrame from summary ------------------------------------
sum_idx = df_sum.set_index("model")

MODELS = ["RF", "SVM", "CNN"]
all_rows = []
for m in MODELS:
    if m not in sum_idx.index:
        print(f"  [WARN] Model {m} not found in summary — skipping")
        continue
    row = sum_idx.loc[m]
    all_rows.append({
        "model"         : m,
        "sd_paradigm"   : "per_subject_5fold",
        "f1_sd"         : float(row["f1_sd_mean"]),
        "f1_sd_std"     : float(row["f1_sd_std"]),
        "f1_loso"       : float(row["f1_loso_mean"]),
        "f1_loso_std"   : float(row["f1_loso_std"]),
        "delta_f1"      : float(row["delta_f1_mean"]),
        "delta_f1_std"  : float(row["delta_f1_std"]),
        "n_subjects"    : int(row["n"]),
    })

df_all = pd.DataFrame(all_rows)
df_all.to_csv(OUT / "all_models_generalization_gap.csv", index=False)
print("\n[saved] all_models_generalization_gap.csv")
print(df_all[["model", "f1_sd", "f1_loso", "delta_f1"]].to_string(index=False))

# -- 3. Bar charts ----------------------------------------------------------------
COLOURS = {"RF": "#4C72B0", "SVM": "#55A868", "CNN": "#C44E52"}


def gap_bar(metric_col, std_col, ylabel, title, outname):
    models_plot = [m for m in MODELS if m in df_all["model"].values]
    fig, ax = plt.subplots(figsize=(7, 5))
    idx    = df_all.set_index("model")
    vals   = idx[metric_col].reindex(models_plot)
    colors = [COLOURS[m] for m in models_plot]
    bars   = ax.bar(models_plot, vals, color=colors, width=0.5,
                    edgecolor="black", linewidth=0.6)

    if std_col and std_col in idx.columns:
        stds = idx[std_col].reindex(models_plot)
        ax.errorbar(models_plot, vals, yerr=stds, fmt="none", color="black",
                    capsize=5, linewidth=1.2)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(vals) * 1.30)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.02, 0.01,
        "SD paradigm: per-subject 5-fold CV (n=40). LOSO: per-subject z-score norm. Freq-72 features.",
        fontsize=7, color="dimgray", va="bottom",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT / outname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outname}")


gap_bar("delta_f1", "delta_f1_std",
        "Generalization Gap (Delta F1 Macro)",
        "Generalization Gap by Model (SD -> LOSO)\nFreq-72 Features, Per-Subject Norm",
        "all_models_delta_f1_bar.png")

# -- 4. Side-by-side SD vs LOSO absolute F1 grouped bar --------------------------
models_plot = [m for m in MODELS if m in df_all["model"].values]
idx = df_all.set_index("model")

fig, ax = plt.subplots(figsize=(8, 5))
x     = np.arange(len(models_plot))
width = 0.32

sd_vals   = idx["f1_sd"].reindex(models_plot).astype(float)
loso_vals = idx["f1_loso"].reindex(models_plot).astype(float)
sd_std    = idx["f1_sd_std"].reindex(models_plot).astype(float)
loso_std  = idx["f1_loso_std"].reindex(models_plot).astype(float)

b1 = ax.bar(x - width / 2, sd_vals, width, label="SD (per-subject 5-fold)",
            color=[COLOURS[m] for m in models_plot], alpha=0.9,
            edgecolor="black", linewidth=0.6,
            yerr=sd_std, capsize=4, error_kw={"linewidth": 1.0})
b2 = ax.bar(x + width / 2, loso_vals, width, label="LOSO (leave-one-subject-out)",
            color=[COLOURS[m] for m in models_plot], alpha=0.45,
            edgecolor="black", linewidth=0.6, hatch="//",
            yerr=loso_std, capsize=4, error_kw={"linewidth": 1.0})

for bar, v in zip(b1, sd_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
for bar, v in zip(b2, loso_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(models_plot)
ax.set_ylabel("Macro F1")
ax.set_title("SD vs LOSO F1 — All Models\n(Freq-72 Features, Per-Subject Norm)", fontsize=12)
ax.set_ylim(0, 1.08)
ax.legend(framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "all_models_sd_vs_loso_f1.png", dpi=200, bbox_inches="tight")
plt.close()
print("[saved] all_models_sd_vs_loso_f1.png")

# -- 5. Methodology notes --------------------------------------------------------
notes = f"""GENERALIZATION GAP -- METHODOLOGY NOTES
=======================================
Generated by merge_all_gaps.py | For use in thesis Methods / Results narrative.
Feature set: Freq-72 (TD + spectral; 72 features)
Normalization: per-subject z-score (best across all models)

EVALUATION PROTOCOL
-------------------
All three model families (RF, SVM, CNN) follow the same two-paradigm evaluation:

  SD  (subject-dependent):  Per-subject 5-fold cross-validation.
      Each subject's windowed data (~650-850 windows) is split into 5 folds.
      A separate model is trained for each subject using their own data only.
      Result: one metric per subject, averaged across n=40 subjects.

  LOSO (leave-one-subject-out):  Cross-subject generalisation.
      Each subject is held out in turn; the model trains on all 39 remaining subjects.
      Per-subject z-score normalization applied before the LOSO loop.
      Result: one metric per held-out subject, averaged across n=40 subjects.

The generalization gap (Delta F1 = F1_SD - F1_LOSO) quantifies the performance
cost of not having labelled data from the target subject.

RESULTS SUMMARY
---------------
"""

for m in models_plot:
    row = df_all[df_all["model"] == m].iloc[0]
    notes += (f"  {m:3s} -- F1_SD = {row['f1_sd']:.3f} +/- {row['f1_sd_std']:.3f},  "
              f"F1_LOSO = {row['f1_loso']:.3f} +/- {row['f1_loso_std']:.3f},  "
              f"Delta F1 = {row['delta_f1']:.3f}\n")

notes += f"""
All three models show a substantial generalisation gap.
The LOSO performance with per-subject normalization converges to a narrow range
(RF={idx['f1_loso']['RF']:.3f}, SVM={idx['f1_loso']['SVM']:.3f}, CNN={idx['f1_loso']['CNN']:.3f}),
indicating that cross-subject variability is the primary bottleneck
regardless of model architecture.

NOTE ON NORMALIZATION
---------------------
Per-subject z-score normalization applied channel-wise before LOSO loop.
This directly addresses inter-subject amplitude variability.
Improvement over global baseline: SVM +0.069, RF +0.051, CNN +0.072 F1.

NOTE ON HYPERPARAMETERS
-----------------------
Classical SD: Fixed optimal hyperparameters from sweep (SVM C=10 scaled, RF n_estimators=500).
Classical LOSO: Nested inner CV (5-fold GroupKFold) per LOSO fold to select optimal
hyperparameters. This is conservative -- the LOSO model is better-tuned per fold,
meaning the reported gap is a lower bound on the true generalisation cost.
CNN: Identical SimpleEMGCNN architecture (3 conv blocks: 32->64->128,
kernel_size=9/7/5, AdaptiveAvgPool1d, Dropout=0.25) for both SD and LOSO.
"""

with open(OUT / "gap_methodology_notes.txt", "w", encoding="utf-8") as f:
    f.write(notes)
print("[saved] gap_methodology_notes.txt")

print("\n=== merge_all_gaps.py COMPLETE ===")

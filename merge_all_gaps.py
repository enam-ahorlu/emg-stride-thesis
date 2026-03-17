"""
merge_all_gaps.py
=================
Merge RF, SVM, and CNN generalization gaps into a single authoritative file
and plot.  All three models use per-subject 5-fold CV as the SD baseline.

Outputs (all in report_figs/):
  all_models_generalization_gap.csv   -- merged 3-model gap table
  all_models_delta_f1_bar.png         -- combined gap bar chart (F1)
  all_models_delta_balacc_bar.png     -- combined gap bar chart (BalAcc)
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

# -- 1. Load raw gap files --------------------------------------------------------
classical_gap = ROOT / "results_loso_light" / "generalization_plots" / "mean_gap_by_model.csv"
cnn_gap       = ROOT / "results_cnn_loso"   / "generalization_plots_cnn" / "mean_gap_by_model.csv"

df_cls = pd.read_csv(classical_gap)          # columns: model, delta_f1, delta_bal_acc
df_cnn = pd.read_csv(cnn_gap)               # columns: model, sd_paradigm, delta_f1, delta_bal_acc

# -- 2. Load summary-level SD/LOSO absolute values --------------------------------
cls_summary = ROOT / "results_loso_light" / "generalization_gap_summary.csv"
cnn_summary = ROOT / "results_cnn_loso"   / "generalization_gap_cnn_summary.csv"

# Both summary files use pandas multi-level header from .agg():
# row 0 = metric names (repeated), row 1 = stat names (mean/std/count)
# Parse them the same way.

def parse_gap_summary(path):
    """Parse a generalization_gap_summary.csv with multi-level columns.

    The summary CSV is produced by pandas .agg(["mean","std","count"]) which
    writes two header rows (metric names + stat names) followed by data rows.
    Rather than relying on fragile positional indices we reconstruct proper
    column names from the two header rows and look up by name.
    """
    raw = pd.read_csv(path, header=None)
    # Row 0: top-level metric name (e.g. "f1_sd", repeated 3 times for mean/std/count)
    # Row 1: stat name ("mean", "std", "count")
    # Row 2+: data (model name in col 0, values in subsequent cols)
    metric_row = raw.iloc[0].tolist()
    stat_row   = raw.iloc[1].tolist()

    # Build a flat column name list like "f1_sd_mean", "f1_sd_std", etc.
    col_names = []
    for m, s in zip(metric_row, stat_row):
        m_str = str(m).strip()
        s_str = str(s).strip()
        if m_str == "model" or s_str == "NaN":
            col_names.append("model")
        else:
            col_names.append(f"{m_str}_{s_str}")

    data_rows = raw.iloc[2:].reset_index(drop=True)
    data_rows.columns = col_names

    results = {}
    for _, row in data_rows.iterrows():
        model = str(row["model"]).strip()
        d = {}
        for key in ["f1_sd_mean", "f1_sd_std", "f1_loso_mean", "f1_loso_std",
                    "delta_f1_mean", "delta_f1_std"]:
            if key in row.index:
                try:
                    d[key] = float(row[key])
                except (ValueError, TypeError):
                    d[key] = None
            else:
                d[key] = None
        results[model] = d
    return results

cls_parsed = parse_gap_summary(cls_summary)
cnn_parsed = parse_gap_summary(cnn_summary)

# Build classical rows
cls_rows = []
for _, row in df_cls.iterrows():
    model = row["model"]
    p = cls_parsed.get(model, {})
    cls_rows.append({
        "model"         : model,
        "sd_paradigm"   : "per_subject_5fold",
        "f1_sd"         : p.get("f1_sd_mean"),
        "f1_loso"       : p.get("f1_loso_mean"),
        "delta_f1"      : float(row["delta_f1"]),
        "delta_bal_acc" : float(row["delta_bal_acc"]),
        "delta_f1_std"  : p.get("delta_f1_std"),
    })

# CNN row
cnn_p = cnn_parsed.get("CNN", {})
cnn_row = {
    "model"         : "CNN",
    "sd_paradigm"   : "per_subject_5fold",
    "f1_sd"         : cnn_p.get("f1_sd_mean"),
    "f1_loso"       : cnn_p.get("f1_loso_mean"),
    "delta_f1"      : float(df_cnn["delta_f1"].iloc[0]),
    "delta_bal_acc" : float(df_cnn["delta_bal_acc"].iloc[0]),
    "delta_f1_std"  : cnn_p.get("delta_f1_std"),
}

# -- 3. Build combined DataFrame ---------------------------------------------------
all_rows = cls_rows + [cnn_row]
df_all = pd.DataFrame(all_rows)
df_all.to_csv(OUT / "all_models_generalization_gap.csv", index=False)
print("[saved] all_models_generalization_gap.csv")
print(df_all[["model", "f1_sd", "f1_loso", "delta_f1", "delta_bal_acc"]].to_string(index=False))

# -- 4. Bar charts -----------------------------------------------------------------
COLOURS = {"RF": "#4C72B0", "SVM": "#55A868", "CNN": "#C44E52"}
MODELS  = ["RF", "SVM", "CNN"]


def gap_bar(metric_col, ylabel, title, outname):
    fig, ax = plt.subplots(figsize=(7, 5))
    vals   = df_all.set_index("model")[metric_col].reindex(MODELS)
    colors = [COLOURS[m] for m in MODELS]
    bars   = ax.bar(MODELS, vals, color=colors, width=0.5, edgecolor="black", linewidth=0.6)

    if metric_col == "delta_f1":
        stds = df_all.set_index("model")["delta_f1_std"].reindex(MODELS)
        ax.errorbar(MODELS, vals, yerr=stds, fmt="none", color="black",
                    capsize=5, linewidth=1.2)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(
        0.02, 0.01,
        "SD paradigm: per-subject 5-fold CV (n=40) for all models.",
        fontsize=7, color="dimgray", va="bottom",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(OUT / outname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {outname}")


gap_bar("delta_f1", "Generalization Gap (Delta F1 Macro)",
        "Generalization Gap by Model (SD -> LOSO)",
        "all_models_delta_f1_bar.png")

gap_bar("delta_bal_acc", "Generalization Gap (Delta Balanced Accuracy)",
        "Generalization Gap by Model -- Balanced Accuracy (SD -> LOSO)",
        "all_models_delta_balacc_bar.png")

# -- 5. Side-by-side SD vs LOSO absolute F1 grouped bar ---------------------------
fig, ax = plt.subplots(figsize=(8, 5))
x     = np.arange(len(MODELS))
width = 0.32

sd_vals   = df_all.set_index("model")["f1_sd"].reindex(MODELS).astype(float)
loso_vals = df_all.set_index("model")["f1_loso"].reindex(MODELS).astype(float)

b1 = ax.bar(x - width / 2, sd_vals, width, label="SD (per-subject 5-fold)",
            color=[COLOURS[m] for m in MODELS], alpha=0.9,
            edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + width / 2, loso_vals, width, label="LOSO (leave-one-subject-out)",
            color=[COLOURS[m] for m in MODELS], alpha=0.45,
            edgecolor="black", linewidth=0.6, hatch="//")

for bar, v in zip(b1, sd_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
for bar, v in zip(b2, loso_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.set_ylabel("Macro F1")
ax.set_title("SD vs LOSO F1 -- All Models", fontsize=12)
ax.set_ylim(0, 1.0)
ax.legend(framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT / "all_models_sd_vs_loso_f1.png", dpi=200, bbox_inches="tight")
plt.close()
print("[saved] all_models_sd_vs_loso_f1.png")

# -- 6. Methodology notes (consistent per-subject paradigm) ------------------------
notes = f"""GENERALIZATION GAP -- METHODOLOGY NOTES
=======================================
Generated by merge_all_gaps.py | For use in thesis Methods / Results narrative.

EVALUATION PROTOCOL
-------------------
All three model families (RF, SVM, CNN) follow the same two-paradigm evaluation:

  SD  (subject-dependent):  Per-subject 5-fold cross-validation.
      Each subject's windowed data (~650-850 windows) is split into 5 folds.
      A separate model is trained for each subject using their own data only.
      Result: one metric per subject, averaged across n=40 subjects.

  LOSO (leave-one-subject-out):  Cross-subject generalisation.
      Each subject is held out in turn; the model trains on all 39 remaining subjects.
      Result: one metric per held-out subject, averaged across n=40 subjects.

The generalization gap (Delta F1 = F1_SD - F1_LOSO) quantifies the performance
cost of not having labelled data from the target subject.

RESULTS SUMMARY
---------------
  RF  -- F1_SD = {sd_vals['RF']:.3f},  F1_LOSO = {loso_vals['RF']:.3f},  Delta F1 = {sd_vals['RF'] - loso_vals['RF']:.3f}
  SVM -- F1_SD = {sd_vals['SVM']:.3f},  F1_LOSO = {loso_vals['SVM']:.3f},  Delta F1 = {sd_vals['SVM'] - loso_vals['SVM']:.3f}
  CNN -- F1_SD = {sd_vals['CNN']:.3f},  F1_LOSO = {loso_vals['CNN']:.3f},  Delta F1 = {sd_vals['CNN'] - loso_vals['CNN']:.3f}

All three models show a substantial generalisation gap (~24-38% F1 drop).
The LOSO performance across all models converges to a narrow range
(RF={loso_vals['RF']:.3f}, SVM={loso_vals['SVM']:.3f}, CNN={loso_vals['CNN']:.3f}),
indicating that cross-subject generalisation is the primary bottleneck
regardless of model architecture.

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

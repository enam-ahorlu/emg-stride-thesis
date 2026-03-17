"""
feature_set_comparison.py
=========================
Flag 4: Summarise and visualise base vs extended feature set performance
for RF and SVM (w=250ms, canonical runs only).

Outputs (report_figs/):
  feature_set_comparison.csv        -- clean comparison table
  feature_set_comparison_bar.png    -- grouped bar chart (base vs ext, RF/SVM)
  feature_set_comparison_line.png   -- F1 / BalAcc / Acc line comparison
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).parent
MASTER  = ROOT / "results_classical" / "master_classical_results.csv"
RPT_DIR = ROOT / "report_figs"
RPT_DIR.mkdir(exist_ok=True)

# ── 1. Load and filter master results ─────────────────────────────────────
df = pd.read_csv(MASTER)

# Canonical only, w=250ms
canon = df[(df["notes"] == "canonical") & (df["window_ms"] == 250)].copy()

# ── 2. Best result per (model_family, feat_set) ────────────────────────────
# For RF: pick best macro F1 row
# For SVM: pick best macro F1 row (C=10 is best for base; ext only has one row)
def best_row(subdf):
    return subdf.sort_values("f1_macro_mean", ascending=False).iloc[0]

rows = []
for feat_set in ["base", "ext"]:
    sub = canon[canon["feat_set"] == feat_set]
    for model_tag, model_label in [("RF_balanced", "RF"), ("SVM_RBF_balanced_scaled", "SVM")]:
        sub_m = sub[sub["model"] == model_tag]
        if sub_m.empty:
            continue
        row = best_row(sub_m)
        rows.append({
            "model"           : model_label,
            "feat_set"        : feat_set,
            "n_features"      : int(row["n_features"]),
            "f1_macro_mean"   : row["f1_macro_mean"],
            "f1_macro_std"    : row["f1_macro_std"],
            "bal_acc_mean"    : row["bal_acc_mean"],
            "bal_acc_std"     : row["bal_acc_std"],
            "acc_mean"        : row["acc_mean"],
            "acc_std"         : row["acc_std"],
            "source_file"     : row["source_file"],
            # RF hyperparams
            "rf_n_estimators" : row.get("rf_n_estimators", np.nan),
            # SVM hyperparams
            "svm_c"           : row.get("svm_c", np.nan),
        })

comp = pd.DataFrame(rows)

# Add delta columns (ext - base) per model
for model in ["RF", "SVM"]:
    base_row = comp[(comp["model"] == model) & (comp["feat_set"] == "base")]
    ext_row  = comp[(comp["model"] == model) & (comp["feat_set"] == "ext")]
    if base_row.empty or ext_row.empty:
        continue
    for metric in ["f1_macro_mean", "bal_acc_mean", "acc_mean"]:
        delta = ext_row[metric].values[0] - base_row[metric].values[0]
        comp.loc[(comp["model"] == model) & (comp["feat_set"] == "ext"),
                 f"delta_{metric}"] = delta

comp.to_csv(RPT_DIR / "feature_set_comparison.csv", index=False)
print("[saved] feature_set_comparison.csv")
print(comp[["model","feat_set","n_features","f1_macro_mean","f1_macro_std",
            "bal_acc_mean","acc_mean"]].to_string(index=False))

# ── 3. Grouped bar chart: base vs ext for each model ──────────────────────
COLOURS = {"base": "#4C72B0", "ext": "#DD8452"}
MODELS  = ["RF", "SVM"]
feat_sets = ["base", "ext"]
FEAT_LABELS = {"base": "Base (36 feat)", "ext": "Extended (54 feat)"}

x     = np.arange(len(MODELS))
width = 0.32

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)

metrics_info = [
    ("f1_macro_mean", "f1_macro_std", "Macro F1",        "Macro F1 — Base vs Extended"),
    ("bal_acc_mean",  "bal_acc_std",  "Balanced Accuracy","Balanced Accuracy — Base vs Extended"),
    ("acc_mean",      "acc_std",      "Accuracy",         "Accuracy — Base vs Extended"),
]

for ax, (metric, std_col, ylabel, title) in zip(axes, metrics_info):
    for i, feat in enumerate(feat_sets):
        vals = [comp[(comp["model"]==m) & (comp["feat_set"]==feat)][metric].values[0]
                for m in MODELS]
        stds = [comp[(comp["model"]==m) & (comp["feat_set"]==feat)][std_col].values[0]
                for m in MODELS]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=FEAT_LABELS[feat],
                      color=COLOURS[feat], edgecolor="black", linewidth=0.5,
                      yerr=stds, capsize=4, error_kw={"linewidth":1.0})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0.75, 0.97)
    ax.legend(fontsize=8, framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle("Feature Set Comparison: Base vs Extended (w=250ms, 5-fold SD, canonical runs)",
             fontsize=11, y=1.01)
fig.text(0.02, -0.02,
         "Base: 36 TD features (MAV, RMS, WL, ZC x 9 channels).  "
         "Extended: +WAMP, wavelet energy, entropy (54 features total, 9 channels).  "
         "Best hyperparameter per condition shown (RF: n_trees=300-500; SVM: C=10).",
         fontsize=7.5, color="dimgray")
plt.tight_layout()
plt.savefig(RPT_DIR / "feature_set_comparison_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("[saved] feature_set_comparison_bar.png")

# ── 4. Delta table: highlight direction of change ──────────────────────────
print("\nDelta (ext - base):")
delta_rows = []
for model in MODELS:
    base_r = comp[(comp["model"]==model) & (comp["feat_set"]=="base")].iloc[0]
    ext_r  = comp[(comp["model"]==model) & (comp["feat_set"]=="ext")].iloc[0]
    delta_rows.append({
        "model"       : model,
        "delta_f1"    : ext_r["f1_macro_mean"] - base_r["f1_macro_mean"],
        "delta_balacc": ext_r["bal_acc_mean"]   - base_r["bal_acc_mean"],
        "delta_acc"   : ext_r["acc_mean"]        - base_r["acc_mean"],
        "interpretation": (
            "Marginal gain" if (ext_r["f1_macro_mean"] - base_r["f1_macro_mean"]) > 0.001
            else "No improvement" if abs(ext_r["f1_macro_mean"] - base_r["f1_macro_mean"]) < 0.001
            else "Drop"
        ),
    })

delta_df = pd.DataFrame(delta_rows)
print(delta_df.to_string(index=False))
delta_df.to_csv(RPT_DIR / "feature_set_comparison_delta.csv", index=False)
print("[saved] feature_set_comparison_delta.csv")

# ── 5. Clean summary narrative to stdout ──────────────────────────────────
print("\n=== Feature Set Finding ===")
print("RF:  base F1={:.4f}  ext F1={:.4f}  delta={:+.4f}".format(
    comp[(comp["model"]=="RF") & (comp["feat_set"]=="base")]["f1_macro_mean"].values[0],
    comp[(comp["model"]=="RF") & (comp["feat_set"]=="ext")]["f1_macro_mean"].values[0],
    delta_df[delta_df["model"]=="RF"]["delta_f1"].values[0]))
print("SVM: base F1={:.4f}  ext F1={:.4f}  delta={:+.4f}".format(
    comp[(comp["model"]=="SVM") & (comp["feat_set"]=="base")]["f1_macro_mean"].values[0],
    comp[(comp["model"]=="SVM") & (comp["feat_set"]=="ext")]["f1_macro_mean"].values[0],
    delta_df[delta_df["model"]=="SVM"]["delta_f1"].values[0]))
print("-> Extended features show negligible or no consistent improvement over base.")
print("-> Motivates principled feature selection in April (RFE/mutual info).")

print("\n=== feature_set_comparison.py COMPLETE ===")

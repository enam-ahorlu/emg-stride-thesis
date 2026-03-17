#!/usr/bin/env python3
"""
compare_feature_sets_loso.py
============================
Compare base-36 vs freq-72 feature sets across both SD and LOSO paradigms.

Collects results from:
  - SD:   results_classical_optimal/ (base-36) and results_classical_freq72/ (freq-72)
  - LOSO: results_loso_light/ (base-36) and results_loso_freq/ (freq-72)
  - CNN:  results_cnn_loso/ (raw signals, for reference)

Outputs (report_figs/):
  loso_feature_comparison.csv           -- LOSO results by feature set
  sd_feature_comparison_freq72.csv      -- SD results by feature set
  combined_feature_comparison.csv       -- SD + LOSO combined
  loso_feature_comparison_bar.png       -- LOSO bar chart
  sd_vs_loso_feature_comparison_bar.png -- SD vs LOSO comparison
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RPT_DIR = ROOT / "report_figs"
RPT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 1. Collect LOSO results
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  PHASE 6: Feature Set Comparison (base-36 vs freq-72)")
print("=" * 60)

loso_rows = []

# Base-36 LOSO results (from existing results_loso_light/)
base_loso_dir = ROOT / "results_loso_light"
for model in ["SVM", "RF"]:
    summaries = list(base_loso_dir.glob(f"*__{model}_nested_loso_summary.csv"))
    if not summaries:
        # Try checkpoint-derived subjectwise files
        subjectwise = list(base_loso_dir.glob(f"*__{model}_subjectwise_ckpt.csv"))
        if subjectwise:
            df = pd.read_csv(subjectwise[0])
            loso_rows.append({
                "paradigm": "LOSO",
                "feat_set": "base-36",
                "model": model,
                "f1_macro_mean": float(df["f1_macro"].mean()),
                "f1_macro_sd": float(df["f1_macro"].std(ddof=1)),
                "bal_acc_mean": float(df["bal_acc"].mean()),
                "acc_mean": float(df["acc"].mean()),
                "n_subjects": len(df),
            })
    else:
        s = pd.read_csv(summaries[0])
        loso_rows.append({
            "paradigm": "LOSO",
            "feat_set": "base-36",
            "model": model,
            "f1_macro_mean": float(s["f1_macro_mean"].values[0]),
            "f1_macro_sd": float(s["f1_macro_sd"].values[0]),
            "bal_acc_mean": float(s["bal_acc_mean"].values[0]),
            "acc_mean": float(s["acc_mean"].values[0]),
            "n_subjects": 40,
        })

# Freq-72 LOSO results
freq_loso_dir = ROOT / "results_loso_freq"
for model in ["SVM", "RF"]:
    summaries = list(freq_loso_dir.glob(f"*__{model}_nested_loso_summary.csv"))
    if not summaries:
        subjectwise = list(freq_loso_dir.glob(f"checkpoints/*__{model}_subjectwise_ckpt.csv"))
        if subjectwise:
            df = pd.read_csv(subjectwise[0])
            loso_rows.append({
                "paradigm": "LOSO",
                "feat_set": "freq-72",
                "model": model,
                "f1_macro_mean": float(df["f1_macro"].mean()),
                "f1_macro_sd": float(df["f1_macro"].std(ddof=1)),
                "bal_acc_mean": float(df["bal_acc"].mean()),
                "acc_mean": float(df["acc"].mean()),
                "n_subjects": len(df),
            })
        else:
            print(f"  [MISSING] No LOSO results for {model} on freq-72 in {freq_loso_dir}")
    else:
        s = pd.read_csv(summaries[0])
        loso_rows.append({
            "paradigm": "LOSO",
            "feat_set": "freq-72",
            "model": model,
            "f1_macro_mean": float(s["f1_macro_mean"].values[0]),
            "f1_macro_sd": float(s["f1_macro_sd"].values[0]),
            "bal_acc_mean": float(s["bal_acc_mean"].values[0]),
            "acc_mean": float(s["acc_mean"].values[0]),
            "n_subjects": 40,
        })

# CNN LOSO (raw signals — for reference only, no feature comparison)
cnn_loso_dir = ROOT / "results_cnn_loso"
cnn_ckpt = cnn_loso_dir / "per_subject_metrics_cnn_loso.csv"
if cnn_ckpt.exists():
    cdf = pd.read_csv(cnn_ckpt)
    loso_rows.append({
        "paradigm": "LOSO",
        "feat_set": "raw-signal",
        "model": "CNN",
        "f1_macro_mean": float(cdf["f1_macro"].mean()),
        "f1_macro_sd": float(cdf["f1_macro"].std(ddof=1)),
        "bal_acc_mean": float(cdf["bal_acc"].mean()) if "bal_acc" in cdf.columns else 0.0,
        "acc_mean": float(cdf["acc"].mean()) if "acc" in cdf.columns else 0.0,
        "n_subjects": len(cdf),
    })

# ═══════════════════════════════════════════════════════════════
# 2. Collect SD results
# ═══════════════════════════════════════════════════════════════
sd_rows = []

# Base-36 SD (from results_classical_optimal/)
sd_base_dir = ROOT / "results_classical_optimal"
sd_base_csv = sd_base_dir / "per_subject_metrics_250_base.csv"
if sd_base_csv.exists():
    sdf = pd.read_csv(sd_base_csv)
    for model in ["SVM", "RF"]:
        # Model names in SD are like SVM_RBF_balanced_scaled, RF_balanced
        if model == "SVM":
            mask = sdf["model"].str.contains("SVM", case=False, na=False)
        else:
            mask = sdf["model"].str.contains("RF", case=False, na=False)
        sub = sdf[mask]
        if not sub.empty:
            sd_rows.append({
                "paradigm": "SD",
                "feat_set": "base-36",
                "model": model,
                "f1_macro_mean": float(sub["f1_macro"].mean()),
                "f1_macro_sd": float(sub["f1_macro"].std(ddof=1)),
                "bal_acc_mean": float(sub["bal_acc"].mean()),
                "acc_mean": float(sub["acc"].mean()),
                "n_subjects": len(sub),
            })

# Freq-72 SD (from results_classical_freq72/)
sd_freq_dir = ROOT / "results_classical_freq72"
sd_freq_csvs = list(sd_freq_dir.glob("*_subjdep_cv.csv"))
if sd_freq_csvs:
    fdf = pd.read_csv(sd_freq_csvs[0])
    for _, row in fdf.iterrows():
        model_name = row["model"]
        if "SVM" in model_name:
            model = "SVM"
        elif "RF" in model_name:
            model = "RF"
        else:
            continue
        sd_rows.append({
            "paradigm": "SD",
            "feat_set": "freq-72",
            "model": model,
            "f1_macro_mean": float(row["f1_macro_mean"]),
            "f1_macro_sd": float(row["f1_macro_std"]),
            "bal_acc_mean": float(row["bal_acc_mean"]),
            "acc_mean": float(row["acc_mean"]),
            "n_subjects": 40,
        })

# CNN SD
cnn_sd_csv = ROOT / "results_cnn" / "cnn_subjectdep_w250_env_zscore.csv"
if cnn_sd_csv.exists():
    csd = pd.read_csv(cnn_sd_csv)
    sd_rows.append({
        "paradigm": "SD",
        "feat_set": "raw-signal",
        "model": "CNN",
        "f1_macro_mean": float(csd["f1_macro"].mean()),
        "f1_macro_sd": float(csd["f1_macro"].std(ddof=1)),
        "bal_acc_mean": float(csd["bal_acc"].mean()) if "bal_acc" in csd.columns else 0.0,
        "acc_mean": float(csd["acc"].mean()) if "acc" in csd.columns else 0.0,
        "n_subjects": len(csd),
    })

# ═══════════════════════════════════════════════════════════════
# 3. Build comparison tables
# ═══════════════════════════════════════════════════════════════
all_rows = loso_rows + sd_rows
if not all_rows:
    print("[ERROR] No results found to compare. Run LOSO/SD phases first.")
    sys.exit(1)

comp = pd.DataFrame(all_rows)

# Save LOSO-only comparison
loso_comp = comp[comp["paradigm"] == "LOSO"]
if not loso_comp.empty:
    loso_comp.to_csv(RPT_DIR / "loso_feature_comparison.csv", index=False)
    print(f"[saved] {RPT_DIR / 'loso_feature_comparison.csv'}")
    print("\n  === LOSO Feature Set Comparison ===")
    print(loso_comp[["feat_set", "model", "f1_macro_mean", "f1_macro_sd",
                      "bal_acc_mean", "acc_mean"]].to_string(index=False))

# Save SD-only comparison
sd_comp = comp[comp["paradigm"] == "SD"]
if not sd_comp.empty:
    sd_comp.to_csv(RPT_DIR / "sd_feature_comparison_freq72.csv", index=False)
    print(f"\n[saved] {RPT_DIR / 'sd_feature_comparison_freq72.csv'}")
    print("\n  === SD Feature Set Comparison ===")
    print(sd_comp[["feat_set", "model", "f1_macro_mean", "f1_macro_sd",
                    "bal_acc_mean", "acc_mean"]].to_string(index=False))

# Save combined
comp.to_csv(RPT_DIR / "combined_feature_comparison.csv", index=False)
print(f"\n[saved] {RPT_DIR / 'combined_feature_comparison.csv'}")

# ═══════════════════════════════════════════════════════════════
# 4. Compute deltas (freq-72 minus base-36) per model per paradigm
# ═══════════════════════════════════════════════════════════════
delta_rows = []
for paradigm in ["SD", "LOSO"]:
    for model in ["SVM", "RF"]:
        base_r = comp[(comp["paradigm"] == paradigm) &
                       (comp["model"] == model) &
                       (comp["feat_set"] == "base-36")]
        freq_r = comp[(comp["paradigm"] == paradigm) &
                       (comp["model"] == model) &
                       (comp["feat_set"] == "freq-72")]
        if not base_r.empty and not freq_r.empty:
            delta_f1 = freq_r["f1_macro_mean"].values[0] - base_r["f1_macro_mean"].values[0]
            delta_rows.append({
                "paradigm": paradigm,
                "model": model,
                "base_f1": base_r["f1_macro_mean"].values[0],
                "freq_f1": freq_r["f1_macro_mean"].values[0],
                "delta_f1": delta_f1,
                "delta_balacc": freq_r["bal_acc_mean"].values[0] - base_r["bal_acc_mean"].values[0],
                "delta_acc": freq_r["acc_mean"].values[0] - base_r["acc_mean"].values[0],
                "interpretation": (
                    "Improvement" if delta_f1 > 0.005
                    else "No change" if abs(delta_f1) <= 0.005
                    else "Degradation"
                ),
            })

if delta_rows:
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(RPT_DIR / "feature_comparison_delta.csv", index=False)
    print(f"\n[saved] {RPT_DIR / 'feature_comparison_delta.csv'}")
    print("\n  === Deltas (freq-72 minus base-36) ===")
    print(delta_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 5. Generate plots
# ═══════════════════════════════════════════════════════════════
COLOURS = {"base-36": "#4C72B0", "freq-72": "#DD8452", "raw-signal": "#C44E52"}

# --- 5a. LOSO bar chart ---
loso_classical = loso_comp[loso_comp["feat_set"] != "raw-signal"]
feat_sets_loso = sorted(loso_classical["feat_set"].unique())
models_loso = [m for m in ["SVM", "RF"] if m in loso_classical["model"].values]

if len(feat_sets_loso) >= 2 and len(models_loso) >= 1:
    # Include CNN as reference
    all_models = models_loso + (["CNN"] if "CNN" in loso_comp["model"].values else [])
    all_feat_sets = feat_sets_loso + (["raw-signal"] if "CNN" in loso_comp["model"].values else [])

    x = np.arange(len(all_models))
    width = 0.8 / len(all_feat_sets)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, fs in enumerate(all_feat_sets):
        vals, stds = [], []
        for m in all_models:
            r = loso_comp[(loso_comp["model"] == m) & (loso_comp["feat_set"] == fs)]
            if not r.empty:
                vals.append(r["f1_macro_mean"].values[0])
                stds.append(r["f1_macro_sd"].values[0])
            else:
                vals.append(0)
                stds.append(0)
        offset = (i - len(all_feat_sets) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=fs,
                      color=COLOURS.get(fs, "#999999"), edgecolor="black",
                      linewidth=0.5, yerr=stds, capsize=4)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(all_models, fontsize=11)
    ax.set_ylabel("Macro F1 (LOSO)")
    ax.set_title("Feature Set Comparison: LOSO Cross-Subject (w=250ms, StandardScaler ON)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.5, 0.85)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "loso_feature_comparison_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {RPT_DIR / 'loso_feature_comparison_bar.png'}")

# --- 5b. SD vs LOSO comparison bar chart ---
if not sd_comp.empty and not loso_comp.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, paradigm, title in zip(axes, ["SD", "LOSO"],
                                    ["Subject-Dependent (5-fold CV)", "LOSO (Cross-Subject)"]):
        sub = comp[comp["paradigm"] == paradigm]
        feat_sets = [fs for fs in ["base-36", "freq-72"] if fs in sub["feat_set"].values]
        models_plot = [m for m in ["SVM", "RF"] if m in sub["model"].values]

        if not feat_sets or not models_plot:
            continue

        x = np.arange(len(models_plot))
        width = 0.35

        for i, fs in enumerate(feat_sets):
            vals, stds = [], []
            for m in models_plot:
                r = sub[(sub["model"] == m) & (sub["feat_set"] == fs)]
                if not r.empty:
                    vals.append(r["f1_macro_mean"].values[0])
                    stds.append(r["f1_macro_sd"].values[0])
                else:
                    vals.append(0)
                    stds.append(0)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=fs,
                          color=COLOURS.get(fs, "#999"), edgecolor="black",
                          linewidth=0.5, yerr=stds, capsize=4)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(models_plot, fontsize=11)
        ax.set_ylabel("Macro F1")
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(0.5, 1.0)
    fig.suptitle("Base-36 vs Freq-72 Feature Comparison (w=250ms)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "sd_vs_loso_feature_comparison_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {RPT_DIR / 'sd_vs_loso_feature_comparison_bar.png'}")

# ═══════════════════════════════════════════════════════════════
# 6. Determine winner per paradigm
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  WINNERS")
print("=" * 60)

for paradigm in ["SD", "LOSO"]:
    print(f"\n  --- {paradigm} ---")
    for model in ["SVM", "RF"]:
        sub = comp[(comp["paradigm"] == paradigm) & (comp["model"] == model)]
        if len(sub) >= 2:
            best = sub.sort_values("f1_macro_mean", ascending=False).iloc[0]
            second = sub.sort_values("f1_macro_mean", ascending=False).iloc[1]
            delta = best["f1_macro_mean"] - second["f1_macro_mean"]
            print(f"  {model}: Best = {best['feat_set']} (F1={best['f1_macro_mean']:.4f}), "
                  f"vs {second['feat_set']} (F1={second['f1_macro_mean']:.4f}), "
                  f"delta={delta:+.4f}")
        elif len(sub) == 1:
            print(f"  {model}: Only {sub.iloc[0]['feat_set']} available (F1={sub.iloc[0]['f1_macro_mean']:.4f})")

print("\n=== compare_feature_sets_loso.py COMPLETE ===")

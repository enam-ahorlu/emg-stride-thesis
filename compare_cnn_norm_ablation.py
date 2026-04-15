#!/usr/bin/env python3
"""
compare_cnn_norm_ablation.py
=============================
Compile and visualize CNN normalization ablation results from LOSO runs.

Expects results in:
  results_cnn_loso/           (norm_mode=global, existing baseline)
  results_cnn_loso_norm_none/     (norm_mode=none)
  results_cnn_loso_norm_persubj/  (norm_mode=per_subject)
  results_cnn_loso_norm_robust/   (norm_mode=robust)

Outputs (report_figs/):
  cnn_norm_ablation_results.csv     — full comparison table
  cnn_norm_ablation_bar.png         — bar chart (4 conditions)
  cnn_norm_ablation_delta.csv       — deltas vs global baseline
  cnn_norm_ablation_per_subject.png — per-subject F1 boxplot
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

# Condition name -> results directory
CNN_NORM_DIRS = {
    "none":        ROOT / "results_cnn_loso_norm_none",
    "global":      ROOT / "results_cnn_loso",
    "per_subject": ROOT / "results_cnn_loso_norm_persubj",
    "robust":      ROOT / "results_cnn_loso_norm_robust",
}

NORM_LABELS = {
    "none":        "No Scaling",
    "global":      "Per-Fold\nZ-Score (Global)",
    "per_subject": "Per-Subject\nZ-Score",
    "robust":      "RobustScaler\n(Median/IQR)",
}

COLOURS = {
    "none":        "#D9D9D9",
    "global":      "#4C72B0",
    "per_subject": "#55A868",
    "robust":      "#DD8452",
}

NORM_ORDER = ["none", "global", "per_subject", "robust"]

# ═══════════════════════════════════════════════════════════════
# 1. Collect results
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  CNN Normalization Ablation Comparison")
print("=" * 60)

rows = []
per_subj_data = []

for norm_mode, rdir in CNN_NORM_DIRS.items():
    if not rdir.exists():
        print(f"  [SKIP] {norm_mode}: directory {rdir} not found")
        continue

    summary_csv = rdir / "cnn_loso_summary.csv"
    persubj_csv = rdir / "per_subject_metrics_cnn_loso.csv"

    if not summary_csv.exists():
        # Fall back to per-subject CSV to compute summary
        if persubj_csv.exists():
            ps = pd.read_csv(persubj_csv)
            if len(ps) < 40:
                print(f"  [INCOMPLETE] {norm_mode}: only {len(ps)}/40 subjects in per_subject_metrics")
                continue
            rows.append({
                "norm_mode": norm_mode,
                "f1_mean": float(ps["f1_macro"].mean()),
                "f1_sd":   float(ps["f1_macro"].std(ddof=1)),
                "bal_acc_mean": float(ps["bal_acc"].mean()),
                "bal_acc_sd":   float(ps["bal_acc"].std(ddof=1)),
                "n_subjects": len(ps),
            })
            for _, r in ps.iterrows():
                per_subj_data.append({
                    "norm_mode": norm_mode,
                    "subject": int(r["subject"]),
                    "f1_macro": float(r["f1_macro"]),
                })
        else:
            print(f"  [MISSING] {norm_mode}: no summary or per-subject CSV found")
        continue

    s = pd.read_csv(summary_csv)
    # summary may have multiple rows if groupby includes norm_mode column;
    # take the first row (all folds should be one group per directory)
    s0 = s.iloc[0]
    rows.append({
        "norm_mode": norm_mode,
        "f1_mean":     float(s0["mean_f1"]),
        "f1_sd":       float(s0["std_f1"]) if "std_f1" in s.columns else 0.0,
        "bal_acc_mean": float(s0["mean_balacc"]) if "mean_balacc" in s.columns else 0.0,
        "bal_acc_sd":   float(s0["std_balacc"]) if "std_balacc" in s.columns else 0.0,
        "n_subjects":  int(s0["subjects"]) if "subjects" in s.columns else 40,
    })

    if persubj_csv.exists():
        ps = pd.read_csv(persubj_csv)
        for _, r in ps.iterrows():
            per_subj_data.append({
                "norm_mode": norm_mode,
                "subject": int(r["subject"]),
                "f1_macro": float(r["f1_macro"]),
            })

if not rows:
    print("[ERROR] No results found. Run the CNN normalization ablation first.")
    sys.exit(1)

comp = pd.DataFrame(rows)
comp.to_csv(RPT_DIR / "cnn_norm_ablation_results.csv", index=False)
print(f"\n[saved] {RPT_DIR / 'cnn_norm_ablation_results.csv'}")
print("\n  === CNN Results ===")
print(comp[["norm_mode", "f1_mean", "f1_sd", "bal_acc_mean", "n_subjects"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 2. Compute deltas vs global baseline
# ═══════════════════════════════════════════════════════════════
global_r = comp[comp["norm_mode"] == "global"]
delta_rows = []
if not global_r.empty:
    base_f1 = global_r["f1_mean"].values[0]
    for nm in ["none", "per_subject", "robust"]:
        r = comp[comp["norm_mode"] == nm]
        if not r.empty:
            delta = r["f1_mean"].values[0] - base_f1
            delta_rows.append({
                "norm_mode":      nm,
                "f1_mean":        r["f1_mean"].values[0],
                "global_f1":      base_f1,
                "delta_f1":       delta,
                "interpretation": (
                    "Improvement" if delta > 0.005
                    else "No change" if abs(delta) <= 0.005
                    else "Degradation"
                ),
            })

if delta_rows:
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(RPT_DIR / "cnn_norm_ablation_delta.csv", index=False)
    print(f"\n[saved] {RPT_DIR / 'cnn_norm_ablation_delta.csv'}")
    print("\n  === Deltas vs Global ===")
    print(delta_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 3. Bar chart (4 conditions, single model)
# ═══════════════════════════════════════════════════════════════
norm_modes_present = [nm for nm in NORM_ORDER if nm in comp["norm_mode"].values]

if len(norm_modes_present) >= 2:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(norm_modes_present))
    vals = []
    stds = []
    colors = []
    for nm in norm_modes_present:
        r = comp[comp["norm_mode"] == nm]
        vals.append(r["f1_mean"].values[0])
        stds.append(r["f1_sd"].values[0])
        colors.append(COLOURS.get(nm, "#999"))

    bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="black",
                  linewidth=0.6, yerr=stds, capsize=4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.010,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([NORM_LABELS.get(nm, nm).replace("\n", " ")
                        for nm in norm_modes_present], fontsize=10)
    ax.set_ylabel("Macro F1 (LOSO)", fontsize=11)
    ax.set_title("CNN Normalization Ablation: LOSO Cross-Subject Performance\n"
                 "(Raw Windows X_env, w=250ms, SimpleEMGCNN)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ymin = max(0.3, min(vals) - 0.10)
    ymax = min(1.0, max(vals) + 0.08)
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "cnn_norm_ablation_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {RPT_DIR / 'cnn_norm_ablation_bar.png'}")

# ═══════════════════════════════════════════════════════════════
# 4. Per-subject F1 boxplot
# ═══════════════════════════════════════════════════════════════
if per_subj_data:
    ps_df = pd.DataFrame(per_subj_data)

    fig, ax = plt.subplots(figsize=(9, 5))
    box_data = []
    box_labels = []
    box_colors = []
    for nm in norm_modes_present:
        d = ps_df[ps_df["norm_mode"] == nm]["f1_macro"].values
        if len(d) > 0:
            box_data.append(d)
            box_labels.append(NORM_LABELS.get(nm, nm).replace("\n", " "))
            box_colors.append(COLOURS.get(nm, "#999"))

    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

    ax.set_title("CNN Normalization Ablation: Per-Subject F1 Distribution (LOSO)", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "cnn_norm_ablation_per_subject.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {RPT_DIR / 'cnn_norm_ablation_per_subject.png'}")

# ═══════════════════════════════════════════════════════════════
# 5. Winner summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  WINNERS (CNN LOSO)")
print("=" * 60)
ranked = comp.sort_values("f1_mean", ascending=False)
best = ranked.iloc[0]
print(f"  Best = {best['norm_mode']} (F1={best['f1_mean']:.4f} +/- {best['f1_sd']:.4f})")
for _, r in ranked.iloc[1:].iterrows():
    delta = r["f1_mean"] - best["f1_mean"]
    print(f"         {r['norm_mode']:15s} F1={r['f1_mean']:.4f} (delta={delta:+.4f})")

print("\n=== compare_cnn_norm_ablation.py COMPLETE ===")

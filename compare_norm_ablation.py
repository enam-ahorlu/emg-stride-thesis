#!/usr/bin/env python3
"""
compare_norm_ablation.py
========================
Compile and visualize normalization ablation results from LOSO runs.

Expects results in:
  results_loso_freq_norm_none/    (norm_mode=none)
  results_loso_freq/              (norm_mode=global, existing baseline)
  results_loso_freq_persubj/     (norm_mode=per_subject)
  results_loso_freq_norm_robust/  (norm_mode=robust)

Outputs (report_figs/):
  norm_ablation_results.csv         — full comparison table
  norm_ablation_bar.png             — grouped bar chart
  norm_ablation_delta.csv           — deltas vs global baseline
  norm_ablation_per_subject.png     — per-subject F1 by norm mode
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
NORM_DIRS = {
    "none":        ROOT / "results_loso_freq_norm_none",
    "global":      ROOT / "results_loso_freq",
    "per_subject": ROOT / "results_loso_freq_persubj",
    "robust":      ROOT / "results_loso_freq_norm_robust",
}

NORM_LABELS = {
    "none":        "No Scaling",
    "global":      "StandardScaler\n(Global)",
    "per_subject": "Per-Subject\nZ-Score",
    "robust":      "RobustScaler\n(Median/IQR)",
}

MODELS = ["SVM", "RF"]

# ═══════════════════════════════════════════════════════════════
# 1. Collect results
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  Normalization Ablation Comparison")
print("=" * 60)

rows = []
per_subj_data = []  # for per-subject plots

for norm_mode, rdir in NORM_DIRS.items():
    if not rdir.exists():
        print(f"  [SKIP] {norm_mode}: directory {rdir} not found")
        continue

    for model in MODELS:
        # Try summary CSV first
        summaries = list(rdir.glob(f"*__{model}_nested_loso_summary.csv"))
        subjectwise_files = list(rdir.glob(f"*__{model}_nested_loso_subjectwise.csv"))

        if not summaries:
            # Fall back to checkpoint
            ckpts = list(rdir.glob(f"checkpoints/*__{model}_subjectwise_ckpt.csv"))
            if ckpts:
                df = pd.read_csv(ckpts[0])
                if len(df) < 40:
                    print(f"  [INCOMPLETE] {norm_mode}/{model}: only {len(df)}/40 subjects")
                    continue
                rows.append({
                    "norm_mode": norm_mode,
                    "model": model,
                    "f1_macro_mean": float(df["f1_macro"].mean()),
                    "f1_macro_sd": float(df["f1_macro"].std(ddof=1)),
                    "bal_acc_mean": float(df["bal_acc"].mean()),
                    "bal_acc_sd": float(df["bal_acc"].std(ddof=1)),
                    "acc_mean": float(df["acc"].mean()),
                    "n_subjects": len(df),
                })
                for _, r in df.iterrows():
                    per_subj_data.append({
                        "norm_mode": norm_mode,
                        "model": model,
                        "subject": int(r["heldout_subject"]),
                        "f1_macro": float(r["f1_macro"]),
                    })
            else:
                print(f"  [MISSING] {norm_mode}/{model}: no results found")
        else:
            s = pd.read_csv(summaries[0])
            row = {
                "norm_mode": norm_mode,
                "model": model,
                "f1_macro_mean": float(s["f1_macro_mean"].values[0]),
                "f1_macro_sd": float(s["f1_macro_sd"].values[0]),
                "bal_acc_mean": float(s["bal_acc_mean"].values[0]),
                "bal_acc_sd": float(s["bal_acc_sd"].values[0]) if "bal_acc_sd" in s.columns else 0.0,
                "acc_mean": float(s["acc_mean"].values[0]),
                "n_subjects": 40,
            }
            rows.append(row)

            # Load per-subject data
            if subjectwise_files:
                sw = pd.read_csv(subjectwise_files[0])
                for _, r in sw.iterrows():
                    per_subj_data.append({
                        "norm_mode": norm_mode,
                        "model": model,
                        "subject": int(r["heldout_subject"]),
                        "f1_macro": float(r["f1_macro"]),
                    })

if not rows:
    print("[ERROR] No results found. Run the ablation first.")
    sys.exit(1)

comp = pd.DataFrame(rows)
comp.to_csv(RPT_DIR / "norm_ablation_results.csv", index=False)
print(f"\n[saved] {RPT_DIR / 'norm_ablation_results.csv'}")
print("\n  === Results ===")
print(comp[["norm_mode", "model", "f1_macro_mean", "f1_macro_sd",
            "bal_acc_mean", "acc_mean"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 2. Compute deltas vs global baseline
# ═══════════════════════════════════════════════════════════════
delta_rows = []
for model in MODELS:
    global_r = comp[(comp["model"] == model) & (comp["norm_mode"] == "global")]
    if global_r.empty:
        continue
    base_f1 = global_r["f1_macro_mean"].values[0]

    for norm_mode in ["none", "per_subject", "robust"]:
        r = comp[(comp["model"] == model) & (comp["norm_mode"] == norm_mode)]
        if not r.empty:
            delta = r["f1_macro_mean"].values[0] - base_f1
            delta_rows.append({
                "model": model,
                "norm_mode": norm_mode,
                "f1_macro": r["f1_macro_mean"].values[0],
                "global_f1": base_f1,
                "delta_f1": delta,
                "interpretation": (
                    "Improvement" if delta > 0.005
                    else "No change" if abs(delta) <= 0.005
                    else "Degradation"
                ),
            })

if delta_rows:
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(RPT_DIR / "norm_ablation_delta.csv", index=False)
    print(f"\n[saved] {RPT_DIR / 'norm_ablation_delta.csv'}")
    print("\n  === Deltas vs Global ===")
    print(delta_df.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 3. Grouped bar chart
# ═══════════════════════════════════════════════════════════════
norm_modes = [nm for nm in ["none", "global", "per_subject", "robust"]
              if nm in comp["norm_mode"].values]
models_plot = [m for m in MODELS if m in comp["model"].values]

COLOURS = {
    "none": "#D9D9D9",
    "global": "#4C72B0",
    "per_subject": "#55A868",
    "robust": "#DD8452",
}

if len(norm_modes) >= 2 and len(models_plot) >= 1:
    x = np.arange(len(models_plot))
    width = 0.8 / len(norm_modes)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, nm in enumerate(norm_modes):
        vals, stds = [], []
        for m in models_plot:
            r = comp[(comp["model"] == m) & (comp["norm_mode"] == nm)]
            if not r.empty:
                vals.append(r["f1_macro_mean"].values[0])
                stds.append(r["f1_macro_sd"].values[0])
            else:
                vals.append(0)
                stds.append(0)
        offset = (i - len(norm_modes) / 2 + 0.5) * width
        label = NORM_LABELS.get(nm, nm).replace("\n", " ")
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=COLOURS.get(nm, "#999"), edgecolor="black",
                      linewidth=0.5, yerr=stds, capsize=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models_plot, fontsize=12)
    ax.set_ylabel("Macro F1 (LOSO)", fontsize=11)
    ax.set_title("Normalization Ablation: LOSO Cross-Subject Performance\n"
                 "(Freq-72 Features, w=250ms)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.5, 0.85)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "norm_ablation_bar.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {RPT_DIR / 'norm_ablation_bar.png'}")

# ═══════════════════════════════════════════════════════════════
# 4. Per-subject F1 comparison (boxplot)
# ═══════════════════════════════════════════════════════════════
if per_subj_data:
    ps_df = pd.DataFrame(per_subj_data)

    fig, axes = plt.subplots(1, len(models_plot), figsize=(7 * len(models_plot), 6),
                              sharey=True)
    if len(models_plot) == 1:
        axes = [axes]

    for ax, model in zip(axes, models_plot):
        model_data = ps_df[ps_df["model"] == model]
        box_data = []
        box_labels = []
        for nm in norm_modes:
            d = model_data[model_data["norm_mode"] == nm]["f1_macro"].values
            if len(d) > 0:
                box_data.append(d)
                box_labels.append(NORM_LABELS.get(nm, nm).replace("\n", " "))

        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for i, nm in enumerate(norm_modes):
                if i < len(bp["boxes"]):
                    bp["boxes"][i].set_facecolor(COLOURS.get(nm, "#999"))
                    bp["boxes"][i].set_alpha(0.7)

        ax.set_title(f"{model} — Per-Subject F1 by Norm Mode", fontsize=11)
        ax.set_ylabel("Macro F1" if ax == axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Normalization Ablation: Per-Subject F1 Distribution (LOSO)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(RPT_DIR / "norm_ablation_per_subject.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {RPT_DIR / 'norm_ablation_per_subject.png'}")

# ═══════════════════════════════════════════════════════════════
# 5. Determine best normalization per model
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  WINNERS")
print("=" * 60)
for model in models_plot:
    sub = comp[comp["model"] == model].sort_values("f1_macro_mean", ascending=False)
    if not sub.empty:
        best = sub.iloc[0]
        print(f"  {model}: Best = {best['norm_mode']} (F1={best['f1_macro_mean']:.4f} +/- {best['f1_macro_sd']:.4f})")
        for _, r in sub.iloc[1:].iterrows():
            delta = r["f1_macro_mean"] - best["f1_macro_mean"]
            print(f"         {r['norm_mode']:15s} F1={r['f1_macro_mean']:.4f} (delta={delta:+.4f})")

print("\n=== compare_norm_ablation.py COMPLETE ===")

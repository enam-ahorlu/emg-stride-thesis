#!/usr/bin/env python3
"""
compare_feature_sets_4way.py
=============================
4-way feature set comparison for Subject-Dependent (SD) paradigm:
  Base-36, Extended-54, Freq-72, Combined-81

For each feature set, computes per-subject macro F1 from saved predictions
(y_true, y_pred .npy files + meta CSV for subject labels).

Runs Wilcoxon signed-rank tests between all pairs (non-parametric, suitable
for F1 scores which may not be normally distributed). Outputs:
  report_figs/feature_4way_comparison.csv       — mean±SD per feature set × model
  report_figs/feature_4way_persubject.csv        — per-subject F1 (80 rows × 4 sets)
  report_figs/feature_4way_wilcoxon.csv          — pairwise p-values
  report_figs/feature_4way_bar.png               — grouped bar chart
  report_figs/feature_4way_boxplot.png           — per-subject boxplot

Usage:
  python compare_feature_sets_4way.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
RPT  = ROOT / "report_figs"
RPT.mkdir(exist_ok=True)

# ─── Feature set definitions ────────────────────────────────────────────────
# Each entry: (label, n_features, predictions_dir, meta_csv)
# predictions_dir must contain:
#   <stem>_<MODEL>_subjdep_y_true.npy
#   <stem>_<MODEL>_subjdep_y_pred.npy
#   (no fold_id needed — we use meta subject column)

META_BASE = ROOT / "features_out" / "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
META_COMB = ROOT / "features_out" / "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"

FEATURE_SETS = [
    {
        "label":   "Base-36",
        "n_feat":  36,
        "pred_dir": ROOT / "results_classical_optimal" / "predictions",
        "meta":    META_BASE,
        "models": {
            "SVM": "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base_SVM_RBF_balanced_scaled_subjdep",
            "RF":  "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_base_RF_balanced_subjdep",
        },
    },
    {
        "label":   "Extended-54",
        "n_feat":  54,
        "pred_dir": ROOT / "results_classical_ext54" / "predictions",
        "meta":    META_BASE,
        "models": {
            "SVM": "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext_SVM_RBF_balanced_scaled_subjdep",
            "RF":  "windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext_RF_balanced_subjdep",
        },
    },
    {
        "label":   "Freq-72",
        "n_feat":  72,
        "pred_dir": ROOT / "results_classical_freq72" / "predictions",
        "meta":    META_BASE,
        "models": {
            "SVM": "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext_SVM_RBF_balanced_scaled_subjdep",
            "RF":  "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext_RF_balanced_subjdep",
        },
    },
    {
        "label":   "Combined-81",
        "n_feat":  81,
        "pred_dir": ROOT / "results_classical_combined81" / "predictions",
        "meta":    META_COMB,
        "models": {
            "SVM": "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_full_SVM_RBF_balanced_scaled_subjdep",
            "RF":  "combined_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_full_RF_balanced_subjdep",
        },
    },
]

MODELS = ["SVM", "RF"]
COLOURS = {
    "Base-36":     "#4C72B0",
    "Extended-54": "#DD8452",
    "Freq-72":     "#55A868",
    "Combined-81": "#C44E52",
}

# ─── Per-subject F1 from predictions ────────────────────────────────────────
def load_per_subject_f1(pred_dir: Path, stem: str, meta: pd.DataFrame):
    """
    Load y_true and y_pred from .npy files, join with subject labels from meta,
    compute per-subject macro F1.
    Returns dict {subject_id: f1_macro}.
    """
    y_true = np.load(pred_dir / f"{stem}_y_true.npy")
    y_pred = np.load(pred_dir / f"{stem}_y_pred.npy")

    if len(y_true) != len(meta):
        raise ValueError(
            f"Prediction length {len(y_true)} != meta length {len(meta)} "
            f"for stem {stem}"
        )

    subjects = meta["subject"].astype(int).values
    per_sub = {}
    for sid in sorted(set(subjects.tolist())):
        mask = (subjects == sid)
        f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
        per_sub[sid] = float(f1)
    return per_sub


# ─── Collect all per-subject F1 data ─────────────────────────────────────────
print("=" * 65)
print("  4-Way Feature Set Comparison (SD, Subject-Dependent 5-Fold CV)")
print("=" * 65)

# { (feat_label, model): {subject: f1} }
all_f1: dict[tuple[str, str], dict[int, float]] = {}
available = []

for fs in FEATURE_SETS:
    meta_df = pd.read_csv(fs["meta"])
    for model in MODELS:
        stem = fs["models"][model]
        y_true_path = fs["pred_dir"] / f"{stem}_y_true.npy"
        if not y_true_path.exists():
            print(f"  [SKIP] {fs['label']} / {model}: predictions not found at {y_true_path}")
            continue
        try:
            per_sub = load_per_subject_f1(fs["pred_dir"], stem, meta_df)
            all_f1[(fs["label"], model)] = per_sub
            available.append((fs["label"], fs["n_feat"], model))
            print(f"  [OK]   {fs['label']} / {model}: {len(per_sub)} subjects loaded")
        except Exception as e:
            print(f"  [ERR]  {fs['label']} / {model}: {e}")

if not all_f1:
    print("[ERROR] No data loaded. Run SD training first.")
    sys.exit(1)

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n  === Summary (mean ± SD macro F1 across 40 subjects) ===")
summary_rows = []
for (feat_label, n_feat, model) in available:
    vals = list(all_f1[(feat_label, model)].values())
    summary_rows.append({
        "feature_set": feat_label,
        "n_features":  n_feat,
        "model":       model,
        "f1_mean":     float(np.mean(vals)),
        "f1_sd":       float(np.std(vals, ddof=1)),
        "f1_median":   float(np.median(vals)),
        "n_subjects":  len(vals),
    })

summary_df = pd.DataFrame(summary_rows).sort_values(["model", "f1_mean"], ascending=[True, False])
summary_df.to_csv(RPT / "feature_4way_comparison.csv", index=False)
print(summary_df[["feature_set", "n_features", "model", "f1_mean", "f1_sd"]].to_string(index=False))
print(f"\n[saved] {RPT / 'feature_4way_comparison.csv'}")

# ─── Per-subject long-form table ──────────────────────────────────────────────
per_subj_rows = []
for (feat_label, n_feat, model) in available:
    for sid, f1 in all_f1[(feat_label, model)].items():
        per_subj_rows.append({
            "feature_set": feat_label,
            "n_features":  n_feat,
            "model":       model,
            "subject":     sid,
            "f1_macro":    f1,
        })
per_subj_df = pd.DataFrame(per_subj_rows)
per_subj_df.to_csv(RPT / "feature_4way_persubject.csv", index=False)
print(f"[saved] {RPT / 'feature_4way_persubject.csv'}")

# ─── Wilcoxon signed-rank tests ───────────────────────────────────────────────
# For each model: pairwise tests between all feature set pairs
print("\n  === Wilcoxon Signed-Rank Tests (pairwise per model) ===")
print(f"  (H0: no difference in per-subject F1 distributions; p<0.05 = significant)")

wilcox_rows = []
feat_labels = list(dict.fromkeys([fs["label"] for fs in FEATURE_SETS]))  # preserve order

for model in MODELS:
    available_feats = [fl for fl in feat_labels if (fl, model) in all_f1]
    if len(available_feats) < 2:
        continue
    print(f"\n  Model: {model}")
    for i, fa in enumerate(available_feats):
        for fb in available_feats[i+1:]:
            subs_a = all_f1[(fa, model)]
            subs_b = all_f1[(fb, model)]
            # Use shared subjects only
            shared = sorted(set(subs_a.keys()) & set(subs_b.keys()))
            if len(shared) < 5:
                continue
            vals_a = np.array([subs_a[s] for s in shared])
            vals_b = np.array([subs_b[s] for s in shared])
            diff = vals_b - vals_a
            if np.all(diff == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(vals_a, vals_b, alternative="two-sided")

            mean_a = float(np.mean(vals_a))
            mean_b = float(np.mean(vals_b))
            winner = fb if mean_b > mean_a else fa
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"    {fa:15s} vs {fb:15s}  |  d={mean_b-mean_a:+.4f}  p={p:.4f} {sig}  winner={winner}")
            wilcox_rows.append({
                "model": model,
                "feat_A": fa,
                "feat_B": fb,
                "mean_f1_A": mean_a,
                "mean_f1_B": mean_b,
                "delta_BminusA": mean_b - mean_a,
                "winner": winner,
                "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
                "p_value": float(p),
                "significant_p05": bool(p < 0.05),
                "significant_p01": bool(p < 0.01),
                "significance": sig,
                "n_subjects": len(shared),
            })

wilcox_df = pd.DataFrame(wilcox_rows)
wilcox_df.to_csv(RPT / "feature_4way_wilcoxon.csv", index=False)
print(f"\n[saved] {RPT / 'feature_4way_wilcoxon.csv'}")

# ─── Bar chart ────────────────────────────────────────────────────────────────
feat_labels_avail = list(dict.fromkeys([r["feature_set"] for r in summary_rows]))
models_avail = list(dict.fromkeys([r["model"] for r in summary_rows]))

fig, ax = plt.subplots(figsize=(11, 5.5))
n_feats = len(feat_labels_avail)
n_models = len(models_avail)
width = 0.35
x = np.arange(n_feats)

for mi, model in enumerate(models_avail):
    means, sds, colors = [], [], []
    for fl in feat_labels_avail:
        row = summary_df[(summary_df["feature_set"] == fl) & (summary_df["model"] == model)]
        if row.empty:
            means.append(0); sds.append(0); colors.append("#ccc")
        else:
            means.append(row["f1_mean"].values[0])
            sds.append(row["f1_sd"].values[0])
            colors.append(COLOURS.get(fl, "#999"))

    offset = (mi - (n_models - 1) / 2) * (width + 0.02)
    bars = ax.bar(x + offset, means, width=width,
                  label=model, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.6,
                  yerr=sds, capsize=4)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    # Add hatch pattern to distinguish models
    if mi == 1:
        for bar in bars:
            bar.set_hatch("//")

ax.set_xticks(x)
ax.set_xticklabels(feat_labels_avail, fontsize=11)
ax.set_ylabel("Macro F1 (mean ± SD across 40 subjects)", fontsize=11)
ax.set_title("4-Way Feature Set Comparison — SD 5-Fold CV\n"
             "(Solid=SVM, Hatched=RF; error bars = ±1 SD)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ymin = max(0.5, min(means) - 0.08)
ymax = min(1.0, max(means) + 0.08)
ax.set_ylim(ymin, ymax)
ax.legend(loc="lower right", fontsize=10)

# Add feature count labels on x-axis
n_feats_map = {fs["label"]: fs["n_feat"] for fs in FEATURE_SETS}
ax.set_xticklabels([f"{fl}\n({n_feats_map.get(fl,'?')} feat)" for fl in feat_labels_avail], fontsize=10)

plt.tight_layout()
plt.savefig(RPT / "feature_4way_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"[saved] {RPT / 'feature_4way_bar.png'}")

# ─── Boxplot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, n_models, figsize=(12, 5), sharey=True)
if n_models == 1:
    axes = [axes]

for ax, model in zip(axes, models_avail):
    box_data, box_labels, box_colors = [], [], []
    for fl in feat_labels_avail:
        if (fl, model) not in all_f1:
            continue
        box_data.append(list(all_f1[(fl, model)].values()))
        box_labels.append(f"{fl}\n({n_feats_map.get(fl,'?')}f)")
        box_colors.append(COLOURS.get(fl, "#999"))

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_title(f"Model: {model}", fontsize=11)
    ax.set_ylabel("Per-Subject Macro F1" if ax == axes[0] else "", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)

fig.suptitle("4-Way Feature Set Comparison — Per-Subject F1 Distribution (SD)",
             fontsize=12)
plt.tight_layout()
plt.savefig(RPT / "feature_4way_boxplot.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"[saved] {RPT / 'feature_4way_boxplot.png'}")

# ─── Winner declaration ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  WINNER SUMMARY")
print("=" * 65)
for model in models_avail:
    model_rows = summary_df[summary_df["model"] == model].sort_values("f1_mean", ascending=False)
    if model_rows.empty:
        continue
    best = model_rows.iloc[0]
    print(f"\n  {model}: Best = {best['feature_set']} (F1={best['f1_mean']:.4f} ± {best['f1_sd']:.4f})")
    for _, r in model_rows.iloc[1:].iterrows():
        delta = r["f1_mean"] - best["f1_mean"]
        print(f"           {r['feature_set']:15s} F1={r['f1_mean']:.4f} (delta={delta:+.4f})")

# Recommend primary feature set (best overall across both models)
best_by_model = {}
for model in models_avail:
    model_rows = summary_df[summary_df["model"] == model]
    if not model_rows.empty:
        best_by_model[model] = model_rows.sort_values("f1_mean", ascending=False).iloc[0]["feature_set"]

winners = list(best_by_model.values())
if len(set(winners)) == 1:
    primary = winners[0]
else:
    # Pick by average rank
    rank_sum = {fl: 0 for fl in feat_labels_avail}
    for model in models_avail:
        model_rows = summary_df[summary_df["model"] == model].sort_values("f1_mean", ascending=False)
        for rank, (_, row) in enumerate(model_rows.iterrows()):
            rank_sum[row["feature_set"]] += rank
    primary = min(rank_sum, key=rank_sum.get)

print(f"\n  *** RECOMMENDED PRIMARY FEATURE SET: {primary} ***")
print(f"      (best or tied-best across both SVM and RF)")

# Check statistical significance of best vs runner-up
if not wilcox_df.empty:
    print("\n  Statistical significance of top feature set comparisons:")
    top_rows = wilcox_df[
        (wilcox_df["feat_A"] == primary) | (wilcox_df["feat_B"] == primary)
    ].sort_values("p_value")
    for _, r in top_rows.iterrows():
        other = r["feat_B"] if r["feat_A"] == primary else r["feat_A"]
        direction = "better" if r["winner"] == primary else "worse"
        print(f"    {primary} vs {other} [{r['model']}]: "
              f"d={abs(r['delta_BminusA']):.4f}  p={r['p_value']:.4f} {r['significance']}  "
              f"({primary} is {direction})")

print("\n=== compare_feature_sets_4way.py COMPLETE ===")

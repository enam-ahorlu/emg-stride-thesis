"""
freq72_analysis.py
==================
Comprehensive analysis for the Freq-72 feature set with per-subject normalization.

Merges the functionality of the former freq72_downstream_analysis.py and
freq72_statistical_tests.py into a single script.

Produces:
  1. Generalization gap (SD vs LOSO) for all 3 models with freq-72
  2. 3-model comparison bar charts (SD vs LOSO)
  3. Error analysis (per-movement F1, confusion matrices) for SVM+RF+CNN
  4. Per-subject difficulty analysis
  5. Wilcoxon signed-rank tests (pairwise) + 95% confidence intervals
  6. Generalization gap statistics with one-sided tests
  7. Comprehensive JSON output

All outputs go to report_figs/freq72_*
"""

import sys, io, json, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.stats import wilcoxon as scipy_wilcoxon

ROOT = Path(__file__).parent
OUT  = ROOT / "report_figs"
OUT.mkdir(exist_ok=True)

LABELS  = ["DNS", "STDUP", "UPS", "WAK"]
COLOURS = {"RF": "#4C72B0", "SVM": "#55A868", "CNN": "#C44E52"}
MODELS  = ["RF", "SVM", "CNN"]


# ============================================================================
# SHARED UTILITIES
# ============================================================================

def wilcoxon_signed_rank(x, y):
    """Two-sided Wilcoxon signed-rank test via scipy."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    diff = a - b
    n_eff = int(np.sum(diff != 0))
    mean_diff = round(float(diff.mean()), 4)
    if n_eff == 0:
        return dict(W_plus=0, W_minus=0, z=float("nan"), p=1.0, n_eff=0, mean_diff=mean_diff)
    stat, p = scipy_wilcoxon(diff, alternative="two-sided")
    W_plus = float(stat)
    W_minus = float(n_eff * (n_eff + 1) / 2.0 - W_plus)
    return dict(W_plus=W_plus, W_minus=W_minus, z=float("nan"),
                p=round(float(p), 6), n_eff=n_eff, mean_diff=mean_diff)


def ci_95(arr):
    arr = np.asarray(arr, dtype=float)
    n   = len(arr)
    m   = arr.mean()
    se  = arr.std(ddof=1) / math.sqrt(n)
    return dict(mean=round(m, 4), std=round(arr.std(ddof=1), 4),
                se=round(se, 4),
                ci_low=round(m - 1.96 * se, 4),
                ci_high=round(m + 1.96 * se, 4),
                min=round(arr.min(), 4), max=round(arr.max(), 4), n=n)


def pearson_r(x, y):
    mx, my = x.mean(), y.mean()
    num = ((x - mx) * (y - my)).sum()
    den = math.sqrt(((x - mx)**2).sum() * ((y - my)**2).sum())
    return num / den if den > 0 else 0.0


def compute_per_class_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = support - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1        = np.divide(2 * precision * recall, precision + recall,
                          out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
    return pd.DataFrame({
        "label": labels, "support": support.astype(int),
        "precision": precision, "recall": recall, "f1": f1,
    }), cm


def top_confusions(cm, labels, top_k=12):
    rows = []
    total = cm.sum()
    for i, tlab in enumerate(labels):
        row_sum = cm[i].sum()
        for j, plab in enumerate(labels):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            rows.append({
                "true": tlab, "pred": plab, "count": c,
                "rate_within_true": (c / row_sum) if row_sum else 0.0,
                "rate_overall": c / total if total else 0.0,
            })
    return pd.DataFrame(rows).sort_values("count", ascending=False).head(top_k)


def load_fold_predictions(pred_dir, model_prefix, n_subjects=40):
    """Load per-fold prediction files and concatenate."""
    y_true_all, y_pred_all = [], []
    for s in range(1, n_subjects + 1):
        yt_path = pred_dir / f"freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext_{model_prefix}_sub{s:02d}_y_true.npy"
        yp_path = pred_dir / f"freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext_{model_prefix}_sub{s:02d}_y_pred.npy"
        if yt_path.exists() and yp_path.exists():
            y_true_all.append(np.load(yt_path))
            y_pred_all.append(np.load(yp_path))
        else:
            print(f"  [warn] Missing predictions for {model_prefix} subject {s}")
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


# ============================================================================
# STEP 1: LOAD PER-SUBJECT METRICS
# ============================================================================

print("=" * 60)
print("STEP 1: Loading per-subject metrics")
print("=" * 60)

# --- SD per-subject (freq-72): from v2 results with nested GridSearchCV ---
sd_freq72_raw = pd.read_csv(ROOT / "results_classical_freq72_v2" / "per_subject_metrics_freq72_sd.csv")
_model_map = {"SVM_RBF_balanced_scaled": "SVM", "RF_balanced": "RF", "LDA_scaled": "LDA"}
sd_freq72_raw["model"] = sd_freq72_raw["model"].map(_model_map).fillna(sd_freq72_raw["model"])
sd_freq72 = sd_freq72_raw[sd_freq72_raw["model"].isin(["SVM", "RF"])].copy()
sd_freq72 = sd_freq72.rename(columns={"f1_macro": "f1_sd"})
sd_freq72["subject"] = sd_freq72["subject"].astype(int)
print(f"  SD freq-72: {len(sd_freq72)} rows")

# --- SD per-subject (CNN): from results_cnn/ (5-fold CV with early stopping) ---
sd_cnn = pd.read_csv(ROOT / "results_cnn" / "cnn_subjectdep_w250_env_zscore_5fold.csv")
sd_cnn = sd_cnn[["subject", "f1"]].copy()
sd_cnn = sd_cnn.rename(columns={"f1": "f1_sd"})
sd_cnn["model"] = "CNN"
sd_cnn["subject"] = sd_cnn["subject"].astype(int)
print(f"  SD CNN: {len(sd_cnn)} rows")

sd_combined = pd.concat([
    sd_freq72[["subject", "model", "f1_sd"]],
    sd_cnn[["subject", "model", "f1_sd"]],
], ignore_index=True)

# --- LOSO per-subject (freq-72 per_subject norm): SVM + RF ---
loso_svm = pd.read_csv(ROOT / "results_loso_freq_persubj" /
    "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__SVM_nested_loso_subjectwise.csv")
loso_svm["model"] = "SVM"
loso_svm = loso_svm.rename(columns={"heldout_subject": "subject"})

loso_rf = pd.read_csv(ROOT / "results_loso_freq_persubj" /
    "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__RF_nested_loso_subjectwise.csv")
loso_rf["model"] = "RF"
loso_rf = loso_rf.rename(columns={"heldout_subject": "subject"})

# --- LOSO per-subject (CNN per_subject norm) ---
loso_cnn = pd.read_csv(ROOT / "results_cnn_loso_norm_persubj" / "per_subject_metrics_cnn_loso.csv")
loso_cnn = loso_cnn.rename(columns={"f1_macro": "f1_loso"})
loso_cnn["model"] = "CNN"

loso_combined = pd.concat([
    loso_svm[["subject", "model", "f1_macro", "bal_acc", "acc"]].rename(columns={"f1_macro": "f1_loso"}),
    loso_rf[["subject", "model", "f1_macro", "bal_acc", "acc"]].rename(columns={"f1_macro": "f1_loso"}),
    loso_cnn[["subject", "model", "f1_loso", "bal_acc"]],
], ignore_index=True)
loso_combined["subject"] = loso_combined["subject"].astype(int)
print(f"  LOSO per_subject: {len(loso_combined)} rows")

# ============================================================================
# STEP 2: GENERALIZATION GAP
# ============================================================================

print("\n" + "=" * 60)
print("STEP 2: Computing generalization gap")
print("=" * 60)

gap = sd_combined.merge(loso_combined[["subject", "model", "f1_loso"]], on=["subject", "model"], how="inner")
gap["delta_f1"] = gap["f1_sd"] - gap["f1_loso"]
gap = gap.sort_values(["model", "subject"]).reset_index(drop=True)

gap.to_csv(OUT / "freq72_generalization_gap.csv", index=False)
# Also save the full version
gap.to_csv(OUT / "freq72_generalization_gap_full.csv", index=False)
print(f"  Saved: freq72_generalization_gap.csv ({len(gap)} rows)")

gap_summary = gap.groupby("model").agg(
    f1_sd_mean=("f1_sd", "mean"), f1_sd_std=("f1_sd", "std"),
    f1_loso_mean=("f1_loso", "mean"), f1_loso_std=("f1_loso", "std"),
    delta_f1_mean=("delta_f1", "mean"), delta_f1_std=("delta_f1", "std"),
    n=("delta_f1", "count"),
).reset_index()
gap_summary.to_csv(OUT / "freq72_generalization_gap_summary.csv", index=False)
print("\nGeneralization Gap Summary (Freq-72, per_subject norm for LOSO):")
print(gap_summary[["model", "f1_sd_mean", "f1_loso_mean", "delta_f1_mean", "delta_f1_std"]].to_string(index=False))

# --- Gap bar chart ---
fig, ax = plt.subplots(figsize=(7, 5))
gs = gap_summary.set_index("model").reindex(MODELS)
bars = ax.bar(MODELS, gs["delta_f1_mean"], color=[COLOURS[m] for m in MODELS],
              width=0.5, edgecolor="black", linewidth=0.6)
ax.errorbar(MODELS, gs["delta_f1_mean"], yerr=gs["delta_f1_std"],
            fmt="none", color="black", capsize=5, linewidth=1.2)
for bar, v in zip(bars, gs["delta_f1_mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, max(gs["delta_f1_mean"]) * 1.35)
ax.set_xlabel("Model")
ax.set_ylabel("Generalization Gap (Delta F1)")
ax.set_title("Generalization Gap: SD vs LOSO (Freq-72, per-subject norm)")
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.02, 0.01, "SD: per-subject 5-fold CV | LOSO: per-subject z-score normalization",
         fontsize=7, color="dimgray", va="bottom")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT / "freq72_delta_f1_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: freq72_delta_f1_bar.png")

# --- SD vs LOSO side-by-side grouped bar ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(MODELS))
width = 0.32
sd_vals = gs["f1_sd_mean"]
loso_vals = gs["f1_loso_mean"]
b1 = ax.bar(x - width / 2, sd_vals, width, label="SD (per-subject 5-fold)",
            color=[COLOURS[m] for m in MODELS], alpha=0.9, edgecolor="black", linewidth=0.6)
b2 = ax.bar(x + width / 2, loso_vals, width, label="LOSO (per-subject z-score)",
            color=[COLOURS[m] for m in MODELS], alpha=0.45, edgecolor="black", linewidth=0.6, hatch="//")
for bar, v in zip(b1, sd_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
for bar, v in zip(b2, loso_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.set_ylabel("Macro F1")
ax.set_title("SD vs LOSO F1 - All Models (Freq-72)")
ax.set_ylim(0, 1.0)
ax.legend(framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "freq72_sd_vs_loso_f1.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: freq72_sd_vs_loso_f1.png")

# ============================================================================
# STEP 3: ERROR ANALYSIS (per-movement F1)
# ============================================================================

print("\n" + "=" * 60)
print("STEP 3: Error analysis (per-movement F1)")
print("=" * 60)

ERR_OUT = OUT / "freq72_error_analysis"
ERR_OUT.mkdir(exist_ok=True)

print("  Label mapping: DNS=0, STDUP=1, UPS=2, WAK=3")

pred_dir = ROOT / "results_loso_freq_persubj" / "predictions_folds"
all_per_class = []

for model_prefix, model_name in [("SVM", "SVM"), ("RF", "RF")]:
    print(f"\n  Processing {model_name}...")
    y_true, y_pred = load_fold_predictions(pred_dir, model_prefix)
    print(f"    Loaded {len(y_true)} predictions")

    metrics, cm = compute_per_class_metrics(y_true, y_pred, LABELS)
    metrics["model"] = model_name
    all_per_class.append(metrics)

    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.to_csv(ERR_OUT / f"{model_name}_confusion_matrix.csv")
    metrics.to_csv(ERR_OUT / f"{model_name}_per_class_metrics.csv", index=False)

    conf = top_confusions(cm, LABELS)
    conf.to_csv(ERR_OUT / f"{model_name}_top_confusions.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(metrics["label"], metrics["f1"], color=COLOURS[model_name], alpha=0.8, edgecolor="black")
    for i, (lab, f1v) in enumerate(zip(metrics["label"], metrics["f1"])):
        ax.text(i, f1v + 0.01, f"{f1v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{model_name} LOSO Per-Class F1 (Freq-72, per-subject norm)")
    ax.set_ylim(0, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ERR_OUT / f"{model_name}_f1_by_class.png", dpi=200)
    plt.close()

    print(f"    Per-class F1: {dict(zip(metrics['label'], metrics['f1'].round(3)))}")

# CNN per-class from confusion matrix
cnn_cm_path = ROOT / "results_cnn_loso_norm_persubj" / "confusion_matrices"
if cnn_cm_path.exists():
    cnn_cm_files = list(cnn_cm_path.glob("*confusion*.csv"))
    if cnn_cm_files:
        cnn_cm_df = pd.read_csv(cnn_cm_files[0], index_col=0)
        cnn_cm = cnn_cm_df.loc[LABELS, LABELS].values.astype(int)
        support = cnn_cm.sum(axis=1)
        tp = np.diag(cnn_cm)
        fp = cnn_cm.sum(axis=0) - tp
        fn = support - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)
        cnn_metrics = pd.DataFrame({
            "label": LABELS, "support": support.astype(int),
            "precision": precision, "recall": recall, "f1": f1, "model": "CNN",
        })
        all_per_class.append(cnn_metrics)
        cnn_metrics.to_csv(ERR_OUT / "CNN_per_class_metrics.csv", index=False)
        print(f"\n  CNN per-class F1: {dict(zip(cnn_metrics['label'], cnn_metrics['f1'].round(3)))}")

# Combined per-class metrics + chart
if all_per_class:
    combined_pc = pd.concat(all_per_class, ignore_index=True)
    combined_pc.to_csv(OUT / "freq72_all_models_per_class_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    models_in = combined_pc["model"].unique().tolist()
    x = np.arange(len(LABELS))
    width = 0.25
    offsets = np.linspace(-width * (len(models_in) - 1) / 2,
                           width * (len(models_in) - 1) / 2, len(models_in))

    for i, model in enumerate(MODELS):
        if model not in models_in:
            continue
        sub = combined_pc[combined_pc["model"] == model]
        f1_vals = [sub[sub["label"] == lab]["f1"].values[0]
                   if len(sub[sub["label"] == lab]) > 0 else 0 for lab in LABELS]
        ax.bar(x + offsets[i], f1_vals, width, label=model,
               color=COLOURS.get(model, "gray"), alpha=0.8, edgecolor="black", linewidth=0.5)
        for j, v in enumerate(f1_vals):
            ax.text(x[j] + offsets[i], v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Macro F1")
    ax.set_title("Per-Class F1 by Model (Freq-72, LOSO, per-subject norm)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "freq72_all_models_per_class_f1.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\n  Saved: freq72_all_models_per_class_f1.png")

# ============================================================================
# STEP 4: PER-SUBJECT DIFFICULTY ANALYSIS
# ============================================================================

print("\n" + "=" * 60)
print("STEP 4: Per-subject difficulty analysis")
print("=" * 60)

DIFF_OUT = OUT / "freq72_subject_difficulty"
DIFF_OUT.mkdir(exist_ok=True)

for model in MODELS:
    d = gap[gap["model"] == model].copy().sort_values("subject")
    if len(d) == 0:
        print(f"  Skipping {model} (no data)")
        continue

    q = 0.25
    low_thr = d["f1_loso"].quantile(q)
    high_thr = d["f1_loso"].quantile(1 - q)

    hard = d[d["f1_loso"] <= low_thr].sort_values("f1_loso")
    easy = d[d["f1_loso"] >= high_thr].sort_values("f1_loso", ascending=False)

    hard.to_csv(DIFF_OUT / f"{model}_hard_subjects.csv", index=False)
    easy.to_csv(DIFF_OUT / f"{model}_easy_subjects.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    colors = ["#C44E52" if f <= low_thr else "#55A868" if f >= high_thr else "#4C72B0"
              for f in d["f1_loso"]]
    ax.bar(d["subject"].astype(str), d["f1_loso"], color=colors, edgecolor="black", linewidth=0.3)
    ax.axhline(low_thr, linestyle="--", color="red", alpha=0.5, label=f"Hard threshold ({low_thr:.3f})")
    ax.axhline(high_thr, linestyle="--", color="green", alpha=0.5, label=f"Easy threshold ({high_thr:.3f})")
    ax.axhline(d["f1_loso"].mean(), linestyle="-", color="black", alpha=0.3, label=f"Mean ({d['f1_loso'].mean():.3f})")
    ax.set_xlabel("Subject")
    ax.set_ylabel("F1 Macro (LOSO)")
    ax.set_title(f"{model}: LOSO F1 by Subject (Freq-72, per-subject norm)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(DIFF_OUT / f"{model}_f1_loso_by_subject.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(d["subject"].astype(str), d["delta_f1"], color=COLOURS.get(model, "gray"),
           edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.axhline(d["delta_f1"].mean(), linestyle="--", color="black", alpha=0.5,
               label=f"Mean gap ({d['delta_f1'].mean():.3f})")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Delta F1 (SD - LOSO)")
    ax.set_title(f"{model}: Generalization Gap by Subject (Freq-72)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(DIFF_OUT / f"{model}_gap_by_subject.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  {model}: hard={len(hard)} subjects (F1<={low_thr:.3f}), easy={len(easy)} (F1>={high_thr:.3f})")

# Cross-model hard subjects overlap
print("\n  Cross-model hard subject overlap:")
hard_sets = {}
for model in MODELS:
    d = gap[gap["model"] == model]
    if len(d) == 0:
        continue
    thr = d["f1_loso"].quantile(0.25)
    hard_sets[model] = set(d[d["f1_loso"] <= thr]["subject"].tolist())

if len(hard_sets) >= 2:
    all_hard = set.intersection(*hard_sets.values())
    print(f"    Subjects hard across ALL models: {sorted(all_hard)}")
    for m1 in hard_sets:
        for m2 in hard_sets:
            if m1 >= m2:
                continue
            overlap = hard_sets[m1] & hard_sets[m2]
            print(f"    {m1} & {m2} overlap: {sorted(overlap)} ({len(overlap)} subjects)")

cross_summary = []
for model in MODELS:
    d = gap[gap["model"] == model]
    if len(d) == 0:
        continue
    cross_summary.append({
        "model": model,
        "f1_sd_mean": d["f1_sd"].mean(),
        "f1_loso_mean": d["f1_loso"].mean(),
        "delta_f1_mean": d["delta_f1"].mean(),
        "delta_f1_std": d["delta_f1"].std(),
        "worst_subject": int(d.loc[d["f1_loso"].idxmin(), "subject"]),
        "worst_f1": d["f1_loso"].min(),
        "best_subject": int(d.loc[d["f1_loso"].idxmax(), "subject"]),
        "best_f1": d["f1_loso"].max(),
    })
pd.DataFrame(cross_summary).to_csv(DIFF_OUT / "cross_model_summary.csv", index=False)

# ============================================================================
# STEP 5: WILCOXON SIGNED-RANK TESTS
# ============================================================================

print("\n" + "=" * 60)
print("STEP 5: Wilcoxon signed-rank tests (two-sided)")
print("=" * 60)

# Build per-subject F1 arrays for statistical testing
LOSO_DIR = ROOT / "results_loso_freq_persubj"
svm_sw = pd.read_csv(sorted(LOSO_DIR.glob("*SVM*subjectwise.csv"))[0]).drop_duplicates(subset=["heldout_subject"]).sort_values("heldout_subject")
rf_sw  = pd.read_csv(sorted(LOSO_DIR.glob("*RF*subjectwise.csv"))[0]).drop_duplicates(subset=["heldout_subject"]).sort_values("heldout_subject")
cnn_ps = pd.read_csv(ROOT / "results_cnn_loso_norm_persubj" / "per_subject_metrics_cnn_loso.csv").drop_duplicates(subset=["subject"]).sort_values("subject")

subjects = sorted(
    set(svm_sw["heldout_subject"].astype(int).tolist()) &
    set(rf_sw["heldout_subject"].astype(int).tolist()) &
    set(cnn_ps["subject"].astype(int).tolist())
)

svm_f1 = {int(r["heldout_subject"]): r["f1_macro"] for _, r in svm_sw.iterrows()}
rf_f1  = {int(r["heldout_subject"]): r["f1_macro"] for _, r in rf_sw.iterrows()}
cnn_f1 = {int(r["subject"]): r["f1_macro"] for _, r in cnn_ps.iterrows()}

svm_arr = np.array([svm_f1[s] for s in subjects])
rf_arr  = np.array([rf_f1[s]  for s in subjects])
cnn_arr = np.array([cnn_f1[s] for s in subjects])

print(f"  SVM LOSO F1: {svm_arr.mean():.4f} +/- {svm_arr.std(ddof=1):.4f}")
print(f"  RF  LOSO F1: {rf_arr.mean():.4f} +/- {rf_arr.std(ddof=1):.4f}")
print(f"  CNN LOSO F1: {cnn_arr.mean():.4f} +/- {cnn_arr.std(ddof=1):.4f}")

comparisons = {
    "RF_vs_SVM": (rf_arr, svm_arr),
    "RF_vs_CNN": (rf_arr, cnn_arr),
    "SVM_vs_CNN": (svm_arr, cnn_arr),
}

wilcoxon_results = {}
wilcoxon_rows = []
for name, (a, b) in comparisons.items():
    res = wilcoxon_signed_rank(a, b)
    sig = "p<0.05" if res["p"] < 0.05 else "ns"
    print(f"  {name}: W+={res['W_plus']:.1f}, W-={res['W_minus']:.1f}, "
          f"p={res['p']:.6f}, mean_diff={res['mean_diff']:.4f}, sig={sig}")
    wilcoxon_results[name] = res
    wilcoxon_rows.append({
        "comparison": name,
        "W_plus": res["W_plus"], "W_minus": res["W_minus"],
        "z": res["z"], "p_value": res["p"], "n_eff": res["n_eff"],
        "mean_diff_F1": res["mean_diff"],
        "significant_p005": res["p"] < 0.05,
        "interpretation": sig,
    })

pd.DataFrame(wilcoxon_rows).to_csv(OUT / "freq72_wilcoxon_table.csv", index=False)
print(f"\n  [saved] freq72_wilcoxon_table.csv")

# ============================================================================
# STEP 6: 95% CONFIDENCE INTERVALS + SUBJECT DIFFICULTY
# ============================================================================

print("\n" + "=" * 60)
print("STEP 6: Confidence intervals + subject difficulty")
print("=" * 60)

ci_results = {}
ci_rows = []
for name, arr in [("SVM", svm_arr), ("RF", rf_arr), ("CNN", cnn_arr)]:
    ci = ci_95(arr)
    ci_results[name] = ci
    ci_rows.append({"model": name, **ci})
    print(f"  {name}: {ci['mean']*100:.2f}% [{ci['ci_low']*100:.2f}%, {ci['ci_high']*100:.2f}%]")

pd.DataFrame(ci_rows).to_csv(OUT / "freq72_ci_table.csv", index=False)

# Cross-model subject difficulty
subj_avg = {s: (rf_f1[s] + svm_f1[s] + cnn_f1[s]) / 3.0 for s in subjects}
sorted_subjs = sorted(subj_avg.items(), key=lambda x: x[1])

hard5 = [{"subject": s, "avg_f1": round(v, 4),
          "RF_f1": round(rf_f1[s], 4), "SVM_f1": round(svm_f1[s], 4), "CNN_f1": round(cnn_f1[s], 4)}
         for s, v in sorted_subjs[:5]]
easy5 = [{"subject": s, "avg_f1": round(v, 4),
          "RF_f1": round(rf_f1[s], 4), "SVM_f1": round(svm_f1[s], 4), "CNN_f1": round(cnn_f1[s], 4)}
         for s, v in sorted_subjs[-5:]]

r_rf_svm = pearson_r(rf_arr, svm_arr)
r_rf_cnn = pearson_r(rf_arr, cnn_arr)
r_svm_cnn = pearson_r(svm_arr, cnn_arr)
f1_range  = (sorted_subjs[-1][1] - sorted_subjs[0][1]) * 100

print(f"  Pearson r: RF-SVM={r_rf_svm:.3f}, RF-CNN={r_rf_cnn:.3f}, SVM-CNN={r_svm_cnn:.3f}")

avg_rows = [{"subject": s, "avg_f1": round(v, 4),
             "RF_f1": round(rf_f1[s], 4), "SVM_f1": round(svm_f1[s], 4), "CNN_f1": round(cnn_f1[s], 4)}
            for s, v in sorted(subj_avg.items())]
pd.DataFrame(avg_rows).to_csv(OUT / "freq72_subject_difficulty_avg.csv", index=False)

# ============================================================================
# STEP 7: GENERALIZATION GAP STATISTICS
# ============================================================================

print("\n" + "=" * 60)
print("STEP 7: Generalization gap statistics")
print("=" * 60)

gap_results = {}
for model, loso_arr_m in [("RF", rf_arr), ("SVM", svm_arr), ("CNN", cnn_arr)]:
    sub_df = gap[gap["model"] == model].sort_values("subject")
    sd_arr_m = sub_df["f1_sd"].values
    loso_sub = sub_df["f1_loso"].values

    res = wilcoxon_signed_rank(sd_arr_m, loso_sub)
    p_one = res["p"] / 2.0
    gap_vals = sd_arr_m - loso_sub

    print(f"  {model}: gap={gap_vals.mean()*100:.1f}pp +/- {gap_vals.std(ddof=1)*100:.1f}pp, "
          f"p(one-sided)={p_one:.2e}")
    gap_results[model] = {
        "sd_mean": round(float(sd_arr_m.mean()), 4),
        "loso_mean": round(float(loso_sub.mean()), 4),
        "gap_mean_pp": round(float(gap_vals.mean() * 100), 2),
        "gap_std_pp": round(float(gap_vals.std(ddof=1) * 100), 2),
        "p_one_sided": round(p_one, 6),
    }

# ============================================================================
# STEP 8: SAVE COMPREHENSIVE JSON
# ============================================================================

results = {
    "feature_set": "freq-72",
    "norm_mode": "per_subject",
    "n_subjects": len(subjects),
    "loso_f1_means": {
        "SVM": round(float(svm_arr.mean()), 4),
        "RF":  round(float(rf_arr.mean()), 4),
        "CNN": round(float(cnn_arr.mean()), 4),
    },
    "wilcoxon_tests": wilcoxon_results,
    "confidence_intervals": ci_results,
    "subject_difficulty": {
        "hardest_5": hard5, "easiest_5": easy5,
        "range_pp": round(f1_range, 1),
        "correlations": {
            "pearson_RF_SVM": round(r_rf_svm, 3),
            "pearson_RF_CNN": round(r_rf_cnn, 3),
            "pearson_SVM_CNN": round(r_svm_cnn, 3),
        },
    },
    "generalization_gap_stats": gap_results,
}

out_json = OUT / "freq72_statistical_tests.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n  [saved] freq72_statistical_tests.json")

print("\n" + "=" * 60)
print("ALL DONE - freq72_analysis.py COMPLETE")
print("=" * 60)
print(f"\nOutputs in:")
print(f"  {OUT}/freq72_*.csv/png/json")
print(f"  {ERR_OUT}/")
print(f"  {DIFF_OUT}/")

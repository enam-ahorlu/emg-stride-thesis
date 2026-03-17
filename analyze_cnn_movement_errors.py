"""
analyze_cnn_movement_errors.py
==============================
CNN per-class LOSO breakdown: per-class F1, precision, recall,
top confusion pairs, and a combined 3-model (RF/SVM/CNN) F1 comparison chart.
Uses manual confusion_matrix implementation (no sklearn dependency).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent

PRED_DIR   = ROOT / "results_cnn_loso" / "predictions"
CNN_OUTDIR = ROOT / "results_cnn_loso" / "error_analysis" / "movement"
RPT_DIR    = ROOT / "report_figs"
CLS_ERR    = ROOT / "results_loso_light" / "error_analysis" / "movement"

CNN_OUTDIR.mkdir(parents=True, exist_ok=True)
RPT_DIR.mkdir(exist_ok=True)

INT_TO_STR = {0: "DNS", 1: "STDUP", 2: "UPS", 3: "WAK"}
LABELS_INT = list(range(4))
LABELS_STR = [INT_TO_STR[i] for i in LABELS_INT]


def confusion_matrix_np(y_true, y_pred, labels):
    """Pure numpy confusion matrix."""
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


# ── 1. Accumulate CNN confusion matrix across all 40 LOSO subjects ─────────
y_true_all = []
y_pred_all = []

pred_files = sorted(PRED_DIR.glob("*_Sub*_y_pred.npy"))
print(f"Found {len(pred_files)} CNN LOSO per-subject prediction files (expected 40)")

for pred_f in pred_files:
    true_f = Path(str(pred_f).replace("_y_pred.npy", "_y_true.npy"))
    if not true_f.exists():
        print(f"  [WARN] Missing y_true for {pred_f.name} — skipping")
        continue
    y_pred_all.append(np.load(pred_f))
    y_true_all.append(np.load(true_f))

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)

print(f"Total windows: {len(y_true_all)}")
print(f"True label distribution: {dict(zip(*np.unique(y_true_all, return_counts=True)))}")

# Build confusion matrix (rows=true, cols=pred)
cm_int = confusion_matrix_np(y_true_all, y_pred_all, LABELS_INT)
cm_df  = pd.DataFrame(cm_int, index=LABELS_STR, columns=LABELS_STR)
print("\nCNN LOSO Confusion Matrix (aggregate):")
print(cm_df)

# ── 2. Compute per-class metrics ───────────────────────────────────────────
support   = cm_int.sum(axis=1)
tp        = np.diag(cm_int)
fp        = cm_int.sum(axis=0) - tp
fn        = support - tp

n_cls = len(LABELS_INT)
precision = np.divide(tp, tp + fp, out=np.zeros(n_cls, dtype=float), where=(tp+fp)!=0)
recall    = np.divide(tp, tp + fn, out=np.zeros(n_cls, dtype=float), where=(tp+fn)!=0)
f1        = np.divide(2*precision*recall, precision+recall,
                      out=np.zeros(n_cls, dtype=float), where=(precision+recall)!=0)

metrics_df = pd.DataFrame({
    "label"    : LABELS_STR,
    "support"  : support.astype(int),
    "precision": precision,
    "recall"   : recall,
    "f1"       : f1,
})
metrics_df.to_csv(CNN_OUTDIR / "CNN_per_class_metrics.csv", index=False)
print(f"\nPer-class metrics:\n{metrics_df.to_string(index=False)}")
print(f"[saved] CNN_per_class_metrics.csv")

# ── 3. Per-class bar charts ────────────────────────────────────────────────
def bar_chart(labels, values, ylabel, title, outpath, color="#C44E52"):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color=color, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[saved] {outpath.name}")

bar_chart(LABELS_STR, f1,
          "F1 Score", "CNN LOSO F1 by Class",
          CNN_OUTDIR / "CNN_f1_by_class.png")
bar_chart(LABELS_STR, precision,
          "Precision", "CNN LOSO Precision by Class",
          CNN_OUTDIR / "CNN_precision_by_class.png")
bar_chart(LABELS_STR, recall,
          "Recall", "CNN LOSO Recall by Class",
          CNN_OUTDIR / "CNN_recall_by_class.png")

# ── 4. Top confusion pairs ─────────────────────────────────────────────────
rows = []
total = cm_int.sum()
for i, tlab in enumerate(LABELS_STR):
    row_sum = cm_int[i].sum()
    for j, plab in enumerate(LABELS_STR):
        if i == j:
            continue
        c = int(cm_int[i, j])
        if c <= 0:
            continue
        rows.append({
            "true"           : tlab,
            "pred"           : plab,
            "count"          : c,
            "rate_within_true": (c / row_sum) if row_sum else 0.0,
            "rate_overall"   : c / total if total else 0.0,
        })

conf_df = (pd.DataFrame(rows)
           .sort_values(["count","rate_within_true"], ascending=False)
           .head(15))
conf_df.to_csv(CNN_OUTDIR / "CNN_top_confusions.csv", index=False)
print(f"[saved] CNN_top_confusions.csv")
print("\nTop 10 confusions:")
print(conf_df.head(10).to_string(index=False))

# ── 5. Combined 3-model per-class F1 chart (only if classical data exists) ──
rf_path  = CLS_ERR / "RF_per_class_metrics.csv"
svm_path = CLS_ERR / "SVM_per_class_metrics.csv"

if rf_path.exists() and svm_path.exists():
    rf_metrics  = pd.read_csv(rf_path)
    svm_metrics = pd.read_csv(svm_path)
    cnn_metrics = metrics_df.copy()

    for df_ in [rf_metrics, svm_metrics, cnn_metrics]:
        df_.set_index("label", inplace=True)

    x     = np.arange(len(LABELS_STR))
    width = 0.26
    COLOURS = {"RF": "#4C72B0", "SVM": "#55A868", "CNN": "#C44E52"}

    fig, ax = plt.subplots(figsize=(11, 5))
    b_rf  = ax.bar(x - width,  rf_metrics.reindex(LABELS_STR)["f1"],  width, label="RF",
                   color=COLOURS["RF"],  edgecolor="black", linewidth=0.4, alpha=0.9)
    b_svm = ax.bar(x,           svm_metrics.reindex(LABELS_STR)["f1"], width, label="SVM",
                   color=COLOURS["SVM"], edgecolor="black", linewidth=0.4, alpha=0.9)
    b_cnn = ax.bar(x + width,  cnn_metrics.reindex(LABELS_STR)["f1"],  width, label="CNN",
                   color=COLOURS["CNN"], edgecolor="black", linewidth=0.4, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_STR, fontsize=9)
    ax.set_ylabel("F1 Score (LOSO)")
    ax.set_title("Per-Class LOSO F1 — RF vs SVM vs CNN", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.8)
    ax.spines[["top","right"]].set_visible(False)

    ax.axhline(rf_metrics.reindex(LABELS_STR)["f1"].mean(), color=COLOURS["RF"],
               linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(svm_metrics.reindex(LABELS_STR)["f1"].mean(), color=COLOURS["SVM"],
               linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(cnn_metrics.reindex(LABELS_STR)["f1"].mean(), color=COLOURS["CNN"],
               linestyle="--", linewidth=0.8, alpha=0.5)

    fig.text(0.02, 0.01,
             "Dashed lines = macro-average F1 for each model. "
             "LOSO evaluation: train on 39 subjects, test on 1 (n=40 folds).",
             fontsize=7, color="dimgray")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(RPT_DIR / "all_models_per_class_f1.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] report_figs/all_models_per_class_f1.png")

    combined = pd.DataFrame({
        "class"        : LABELS_STR,
        "RF_f1"        : rf_metrics.reindex(LABELS_STR)["f1"].values,
        "RF_precision" : rf_metrics.reindex(LABELS_STR)["precision"].values,
        "RF_recall"    : rf_metrics.reindex(LABELS_STR)["recall"].values,
        "SVM_f1"       : svm_metrics.reindex(LABELS_STR)["f1"].values,
        "SVM_precision": svm_metrics.reindex(LABELS_STR)["precision"].values,
        "SVM_recall"   : svm_metrics.reindex(LABELS_STR)["recall"].values,
        "CNN_f1"       : cnn_metrics.reindex(LABELS_STR)["f1"].values,
        "CNN_precision": cnn_metrics.reindex(LABELS_STR)["precision"].values,
        "CNN_recall"   : cnn_metrics.reindex(LABELS_STR)["recall"].values,
    })
    combined.to_csv(RPT_DIR / "all_models_per_class_metrics.csv", index=False)
    print(f"[saved] report_figs/all_models_per_class_metrics.csv")
else:
    print("[info] Classical per-class metrics not yet available. Skipping 3-model comparison chart.")
    print(f"  (waiting for: {rf_path} and {svm_path})")

# ── 6. Save confusion matrix CSV ─────────────────────────────────────────
cm_df.to_csv(CNN_OUTDIR / "CNN_LOSO_confusion_matrix.csv")
print(f"[saved] CNN_LOSO_confusion_matrix.csv")

print("\n=== analyze_cnn_movement_errors_nosklearn.py COMPLETE ===")

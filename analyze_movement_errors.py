"""
analyze_movement_errors.py
==========================
Unified per-class LOSO error analysis for ALL model types (SVM, RF, CNN).

For classical models (SVM/RF): loads confusion matrix CSVs directly.
For CNN: accumulates predictions from per-subject .npy files and builds CM.
Optionally generates a combined 3-model per-class F1 comparison chart.

Usage examples:
  # Classical only (SVM + RF from confusion CSVs):
  python analyze_movement_errors.py \
      --svm-cm results_loso_light/confusion_matrices/SVM_cm.csv \
      --rf-cm  results_loso_light/confusion_matrices/RF_cm.csv \
      --outdir results_loso_light/error_analysis/movement

  # CNN only (from per-subject prediction .npy files):
  python analyze_movement_errors.py \
      --cnn-pred-dir results_cnn_loso_norm_persubj/predictions \
      --outdir results_cnn_loso_norm_persubj/error_analysis/movement

  # All three models + combined chart:
  python analyze_movement_errors.py \
      --svm-cm results_loso_freq_persubj/confusion_matrices/SVM_cm.csv \
      --rf-cm  results_loso_freq_persubj/confusion_matrices/RF_cm.csv \
      --cnn-pred-dir results_cnn_loso_norm_persubj/predictions \
      --outdir report_figs/freq72_error_analysis \
      --combined-chart
"""

import sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
LABELS = ["DNS", "STDUP", "UPS", "WAK"]


# ── Shared utilities ──────────────────────────────────────────────────────

def confusion_matrix_np(y_true, y_pred, labels):
    """Pure numpy confusion matrix (no sklearn dependency)."""
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def compute_metrics_from_cm(cm: np.ndarray, labels):
    """Per-class precision, recall, F1 from a confusion matrix."""
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = support - tp

    n = len(labels)
    precision = np.divide(tp, tp + fp, out=np.zeros(n, dtype=float), where=(tp + fp) != 0)
    recall    = np.divide(tp, tp + fn, out=np.zeros(n, dtype=float), where=(tp + fn) != 0)
    f1        = np.divide(2 * precision * recall, precision + recall,
                          out=np.zeros(n, dtype=float), where=(precision + recall) != 0)

    return pd.DataFrame({
        "label": labels,
        "support": support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


def top_confusions(cm: np.ndarray, labels, top_k=15):
    """Extract top K off-diagonal confusion pairs sorted by count."""
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
                "true": tlab,
                "pred": plab,
                "count": c,
                "rate_within_true": (c / row_sum) if row_sum else 0.0,
                "rate_overall": c / total if total else 0.0,
            })
    return (pd.DataFrame(rows)
            .sort_values(["count", "rate_within_true"], ascending=False)
            .head(top_k))


def bar_chart(labels, values, ylabel, title, outpath, color="#4C72B0"):
    """Create a labelled bar chart and save to file."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color=color, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"  [saved] {outpath.name}")


# ── Per-model analysis ────────────────────────────────────────────────────

def analyze_classical_model(name, cm_path, outdir, top_k=15):
    """Load classical CM from CSV and produce per-class analysis."""
    df = pd.read_csv(cm_path, index_col=0)
    df = df.loc[LABELS, LABELS]
    cm = df.values.astype(int)

    metrics = compute_metrics_from_cm(cm, LABELS)
    metrics.to_csv(outdir / f"{name}_per_class_metrics.csv", index=False)

    bar_chart(LABELS, metrics["f1"].values, "F1 Score",
              f"{name} LOSO F1 by Class", outdir / f"{name}_f1_by_class.png",
              color="#4C72B0" if name == "RF" else "#55A868")
    bar_chart(LABELS, metrics["precision"].values, "Precision",
              f"{name} LOSO Precision by Class", outdir / f"{name}_precision_by_class.png",
              color="#4C72B0" if name == "RF" else "#55A868")
    bar_chart(LABELS, metrics["recall"].values, "Recall",
              f"{name} LOSO Recall by Class", outdir / f"{name}_recall_by_class.png",
              color="#4C72B0" if name == "RF" else "#55A868")

    conf = top_confusions(cm, LABELS, top_k=top_k)
    conf.to_csv(outdir / f"{name}_top_confusions.csv", index=False)
    print(f"  [done] {name}: metrics + charts + top confusions")
    return metrics


def analyze_cnn_model(pred_dir, outdir, top_k=15):
    """Accumulate CNN LOSO predictions from .npy files and produce analysis."""
    pred_dir = Path(pred_dir)
    pred_files = sorted(pred_dir.glob("*_Sub*_y_pred.npy"))
    print(f"  Found {len(pred_files)} CNN prediction files")

    y_true_all, y_pred_all = [], []
    for pred_f in pred_files:
        true_f = Path(str(pred_f).replace("_y_pred.npy", "_y_true.npy"))
        if not true_f.exists():
            print(f"    [WARN] Missing y_true for {pred_f.name}")
            continue
        y_pred_all.append(np.load(pred_f))
        y_true_all.append(np.load(true_f))

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    print(f"  Total windows: {len(y_true_all)}")

    cm = confusion_matrix_np(y_true_all, y_pred_all, list(range(4)))
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.to_csv(outdir / "CNN_LOSO_confusion_matrix.csv")

    metrics = compute_metrics_from_cm(cm, LABELS)
    metrics.to_csv(outdir / "CNN_per_class_metrics.csv", index=False)

    bar_chart(LABELS, metrics["f1"].values, "F1 Score",
              "CNN LOSO F1 by Class", outdir / "CNN_f1_by_class.png", color="#C44E52")
    bar_chart(LABELS, metrics["precision"].values, "Precision",
              "CNN LOSO Precision by Class", outdir / "CNN_precision_by_class.png", color="#C44E52")
    bar_chart(LABELS, metrics["recall"].values, "Recall",
              "CNN LOSO Recall by Class", outdir / "CNN_recall_by_class.png", color="#C44E52")

    conf = top_confusions(cm, LABELS, top_k=top_k)
    conf.to_csv(outdir / "CNN_top_confusions.csv", index=False)
    print(f"  [done] CNN: metrics + charts + top confusions + CM")
    return metrics


# ── Combined 3-model chart ────────────────────────────────────────────────

def generate_combined_chart(model_metrics, outdir):
    """Generate combined per-class F1 comparison across all models."""
    COLOURS = {"RF": "#4C72B0", "SVM": "#55A868", "CNN": "#C44E52"}
    models = [m for m in ["RF", "SVM", "CNN"] if m in model_metrics]
    if len(models) < 2:
        print("  [skip] Need at least 2 models for combined chart")
        return

    x = np.arange(len(LABELS))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(models):
        df = model_metrics[m].set_index("label")
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, df.reindex(LABELS)["f1"], width,
               label=m, color=COLOURS.get(m, "#888888"),
               edgecolor="black", linewidth=0.4, alpha=0.9)
        ax.axhline(df.reindex(LABELS)["f1"].mean(), color=COLOURS.get(m, "#888888"),
                   linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("F1 Score (LOSO)")
    ax.set_title(f"Per-Class LOSO F1 — {' vs '.join(models)}", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, 0.01,
             "Dashed lines = macro-average F1 for each model. "
             "LOSO evaluation: train on 39 subjects, test on 1 (n=40 folds).",
             fontsize=7, color="dimgray")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(outdir / "all_models_per_class_f1.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] all_models_per_class_f1.png")

    # Combined metrics CSV
    rows = {}
    for m in models:
        df = model_metrics[m].set_index("label").reindex(LABELS)
        rows[f"{m}_f1"] = df["f1"].values
        rows[f"{m}_precision"] = df["precision"].values
        rows[f"{m}_recall"] = df["recall"].values
    combined = pd.DataFrame({"class": LABELS, **rows})
    combined.to_csv(outdir / "all_models_per_class_metrics.csv", index=False)
    print(f"  [saved] all_models_per_class_metrics.csv")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Unified per-class LOSO error analysis for SVM, RF, and/or CNN.")
    ap.add_argument("--svm-cm", default=None, help="Path to SVM confusion CSV")
    ap.add_argument("--rf-cm", default=None, help="Path to RF confusion CSV")
    ap.add_argument("--cnn-pred-dir", default=None,
                    help="Directory with per-subject CNN prediction .npy files")
    ap.add_argument("--outdir", default="report_figs/error_analysis",
                    help="Output directory (default: report_figs/error_analysis)")
    ap.add_argument("--topk", type=int, default=15,
                    help="Top confusion pairs to report (default: 15)")
    ap.add_argument("--combined-chart", action="store_true",
                    help="Generate combined multi-model per-class F1 chart")
    args = ap.parse_args()

    if not args.svm_cm and not args.rf_cm and not args.cnn_pred_dir:
        ap.error("Provide at least one of: --svm-cm, --rf-cm, --cnn-pred-dir")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_metrics = {}

    if args.svm_cm:
        print("[SVM] Analyzing...")
        model_metrics["SVM"] = analyze_classical_model("SVM", args.svm_cm, outdir, args.topk)

    if args.rf_cm:
        print("[RF] Analyzing...")
        model_metrics["RF"] = analyze_classical_model("RF", args.rf_cm, outdir, args.topk)

    if args.cnn_pred_dir:
        print("[CNN] Analyzing...")
        model_metrics["CNN"] = analyze_cnn_model(args.cnn_pred_dir, outdir, args.topk)

    if args.combined_chart and len(model_metrics) >= 2:
        print("[Combined] Generating multi-model chart...")
        generate_combined_chart(model_metrics, outdir)

    print(f"\n=== Movement error analysis complete ({len(model_metrics)} model(s)) ===")
    print(f"    Output: {outdir}")


if __name__ == "__main__":
    main()

# analyze_movement_errors.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABELS = ["DNS", "STDUP", "UPS", "WAK"]  # movement types (alphabetical = encode order)


def compute_metrics_from_cm(cm: np.ndarray, labels):
    # cm rows = true, cols = predicted
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = support - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp, dtype=float), where=(precision + recall) != 0)

    df = pd.DataFrame({
        "label": labels,
        "support": support.astype(int),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    return df


def top_confusions(cm: np.ndarray, labels, top_k=15):
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
    out = pd.DataFrame(rows).sort_values(["count", "rate_within_true"], ascending=False).head(top_k)
    return out


def plot_bar(df, value_col, outpath: Path, title: str):
    plt.figure(figsize=(9, 4.5))
    plt.bar(df["label"].astype(str), df[value_col].astype(float))
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svm-cm", required=True, help="Path to SVM confusion CSV")
    ap.add_argument("--rf-cm", required=True, help="Path to RF confusion CSV")
    ap.add_argument("--outdir", default="results_loso_light/error_analysis/movement", help="Output directory")
    ap.add_argument("--topk", type=int, default=15, help="Top confusion pairs to report")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def load_cm(path):
        df = pd.read_csv(path, index_col=0)
        # ensure correct ordering
        df = df.loc[LABELS, LABELS]
        return df.values.astype(int)

    svm_cm = load_cm(args.svm_cm)
    rf_cm = load_cm(args.rf_cm)

    for name, cm in [("SVM", svm_cm), ("RF", rf_cm)]:
        metrics = compute_metrics_from_cm(cm, LABELS)
        metrics.to_csv(outdir / f"{name}_per_class_metrics.csv", index=False)

        plot_bar(metrics, "recall", outdir / f"{name}_recall_by_class.png", f"{name} LOSO Recall by Class")
        plot_bar(metrics, "precision", outdir / f"{name}_precision_by_class.png", f"{name} LOSO Precision by Class")
        plot_bar(metrics, "f1", outdir / f"{name}_f1_by_class.png", f"{name} LOSO F1 by Class")

        conf = top_confusions(cm, LABELS, top_k=args.topk)
        conf.to_csv(outdir / f"{name}_top_confusions.csv", index=False)

    print(f"[done] Movement-level metrics + top confusions saved to: {outdir}")


if __name__ == "__main__":
    main()
# plot_early_classical_figs.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def bar_chart(df: pd.DataFrame, value_col: str, title: str, out_png: Path) -> None:
    df = df.copy()
    df["feat_set"] = df["feat_set"].fillna("unknown")
    df["label"] = df["model"].astype(str) + " | " + df["feat_set"].astype(str)
    df = df.sort_values(["model", "feat_set"])

    plt.figure(figsize=(10, 4.8))
    plt.bar(df["label"], df[value_col].astype(float))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()
    print(f"[save] {out_png}")


def main():
    ap = argparse.ArgumentParser(description="Early bar charts from master_classical_results.csv (250ms focus).")
    ap.add_argument("--master", type=str, required=True, help="Path to master_classical_results.csv")
    ap.add_argument("--window-ms", type=int, default=250, help="Window length to filter (default: 250)")
    ap.add_argument("--out-dir", type=str, default="figures_early", help="Output directory for figures")
    args = ap.parse_args()

    master = pd.read_csv(Path(args.master))
    df = master[master["window_ms"] == int(args.window_ms)].copy()

    wanted = ["SVM_RBF_balanced_scaled", "RF_balanced", "LDA_scaled"]
    df = df[df["model"].isin(wanted)].copy()

    out_dir = Path(args.out_dir)

    if "f1_macro_mean" in df.columns:
        bar_chart(df, "f1_macro_mean", f"Macro-F1 (window={args.window_ms}ms)", out_dir / f"macro_f1_w{args.window_ms}.png")

    if "bal_acc_mean" in df.columns:
        bar_chart(df, "bal_acc_mean", f"Balanced Accuracy (window={args.window_ms}ms)", out_dir / f"bal_acc_w{args.window_ms}.png")


if __name__ == "__main__":
    main()

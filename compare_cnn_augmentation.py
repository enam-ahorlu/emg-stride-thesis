# compare_cnn_augmentation.py
"""
Comparison and plotting script for CNN LOSO data augmentation results.

Loads per-subject metrics CSVs from five conditions (no-aug baseline +
four augmentation modes) and produces:
  report_figs/cnn_aug_results.csv      -- summary table
  report_figs/cnn_aug_delta.csv        -- delta vs no-augmentation baseline
  report_figs/cnn_aug_bar.png          -- bar chart with error bars
  report_figs/cnn_aug_per_subject.png  -- boxplot per augmentation condition

Usage
-----
    python compare_cnn_augmentation.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
REPORT_DIR = ROOT / "report_figs"
REPORT_DIR.mkdir(exist_ok=True)

# CNN bar-chart colour (matches thesis colour scheme)
CNN_COLOR = "#C44E52"

# ---------------------------------------------------------------------------
# Condition registry: (csv_path, label)
# ---------------------------------------------------------------------------
CONDITIONS = [
    (
        ROOT / "results_cnn_loso_norm_persubj" / "per_subject_metrics_cnn_loso.csv",
        "No Aug",
    ),
    (
        ROOT / "results_cnn_loso_aug_gaussian" / "per_subject_metrics_cnn_loso.csv",
        "Gaussian",
    ),
    (
        ROOT / "results_cnn_loso_aug_chandrop" / "per_subject_metrics_cnn_loso.csv",
        "Chan. Drop",
    ),
    (
        ROOT / "results_cnn_loso_aug_timemask" / "per_subject_metrics_cnn_loso.csv",
        "Time Mask",
    ),
    (
        ROOT / "results_cnn_loso_aug_combined" / "per_subject_metrics_cnn_loso.csv",
        "Combined",
    ),
]


def load_condition(csv_path: Path, label: str) -> pd.DataFrame | None:
    """
    Load per-subject CNN metrics CSV.
    - For the baseline (No Aug), accept all rows (aug_mode column may be
      absent in older files, or may contain the norm_mode value).
    - For augmentation conditions, filter to aug_mode == label mapping
      if the column is present; otherwise accept all rows.
    Returns None gracefully if the file is missing.
    """
    if not csv_path.exists():
        warnings.warn(f"[skip] file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if "f1_macro" not in df.columns:
        warnings.warn(f"[skip] 'f1_macro' column missing in {csv_path}")
        return None

    # If aug_mode column present, filter for baseline to rows where aug_mode
    # indicates no augmentation (per_subject, none, or missing value).
    # For augmentation conditions we take all rows since each output dir
    # corresponds to exactly one augmentation condition.
    if "aug_mode" in df.columns and label == "No Aug":
        # Baseline dir may contain rows from per_subject norm run;
        # accept any aug_mode that is not an augmentation name.
        aug_names = {"gaussian", "chandrop", "timemask", "combined"}
        mask = ~df["aug_mode"].astype(str).str.lower().isin(aug_names)
        df = df[mask].copy()
        if df.empty:
            warnings.warn(f"[skip] no baseline rows after filtering aug_mode in {csv_path}")
            return None

    df["aug_label"] = label
    return df


def build_summary_table(loaded: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Build summary: aug_mode, f1_mean, f1_sd, bal_acc_mean, bal_acc_sd, n_subjects."""
    rows = []
    for label, df in loaded:
        row: dict = {
            "aug_mode":     label,
            "f1_mean":      float(df["f1_macro"].mean()),
            "f1_sd":        float(df["f1_macro"].std(ddof=1)),
            "n_subjects":   len(df),
        }
        if "bal_acc" in df.columns:
            row["bal_acc_mean"] = float(df["bal_acc"].mean())
            row["bal_acc_sd"]   = float(df["bal_acc"].std(ddof=1))
        else:
            row["bal_acc_mean"] = float("nan")
            row["bal_acc_sd"]   = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Delta F1 and delta bal_acc vs No-Aug baseline."""
    base_row = summary[summary["aug_mode"] == "No Aug"]
    if base_row.empty:
        warnings.warn("[delta] No-Aug baseline not found; deltas will be NaN")
        base_f1      = float("nan")
        base_balacc  = float("nan")
    else:
        base_f1     = float(base_row["f1_mean"].iloc[0])
        base_balacc = float(base_row["bal_acc_mean"].iloc[0])

    rows = []
    for _, row in summary.iterrows():
        rows.append({
            "aug_mode":        row["aug_mode"],
            "f1_mean":         row["f1_mean"],
            "f1_sd":           row["f1_sd"],
            "delta_f1":        row["f1_mean"] - base_f1,
            "bal_acc_mean":    row["bal_acc_mean"],
            "bal_acc_sd":      row["bal_acc_sd"],
            "delta_bal_acc":   row["bal_acc_mean"] - base_balacc,
            "n_subjects":      row["n_subjects"],
        })
    return pd.DataFrame(rows)


def plot_bar(summary: pd.DataFrame, out_png: Path) -> None:
    """Bar chart with error bars, one bar per augmentation condition."""
    labels = list(summary["aug_mode"])
    means  = list(summary["f1_mean"])
    sds    = list(summary["f1_sd"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Slightly lighter shade for non-baseline bars
    colors = []
    for lbl in labels:
        if lbl == "No Aug":
            colors.append(CNN_COLOR)
        else:
            colors.append("#e08a8c")  # lighter red

    bars = ax.bar(
        x,
        means,
        color=colors,
        yerr=sds,
        capsize=5,
        error_kw={"elinewidth": 1.5, "capthick": 1.5},
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Macro F1 (LOSO)", fontsize=11)
    ax.set_title(
        "CNN Data Augmentation: LOSO F1 (Per-Subject Norm)",
        fontsize=12,
    )
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    # Annotate bar tops
    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[save] {out_png}")


def plot_per_subject_boxplot(loaded: list[tuple[str, pd.DataFrame]], out_png: Path) -> None:
    """Boxplot of per-subject F1 for each augmentation condition."""
    data_list  = [df["f1_macro"].dropna().values for _, df in loaded]
    tick_labels = [lbl for lbl, _ in loaded]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = []
    for lbl in tick_labels:
        colors.append(CNN_COLOR if lbl == "No Aug" else "#e08a8c")

    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("Per-subject F1 (LOSO)", fontsize=11)
    ax.set_title(
        "CNN Data Augmentation: Per-Subject F1 Distribution",
        fontsize=12,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.05)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[save] {out_png}")


def main() -> None:
    # Load all conditions
    loaded: list[tuple[str, pd.DataFrame]] = []
    for csv_path, label in CONDITIONS:
        df = load_condition(csv_path, label)
        if df is not None:
            loaded.append((label, df))
        else:
            print(f"[warn] condition '{label}' skipped (data not available)")

    if not loaded:
        print("[ERROR] No data loaded. Run CNN augmentation experiments first.")
        sys.exit(1)

    # Summary table
    summary = build_summary_table(loaded)
    out_summary = REPORT_DIR / "cnn_aug_results.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[save] {out_summary}")
    print(summary.to_string(index=False))

    # Delta table
    delta = build_delta_table(summary)
    out_delta = REPORT_DIR / "cnn_aug_delta.csv"
    delta.to_csv(out_delta, index=False)
    print(f"[save] {out_delta}")
    print(delta.to_string(index=False))

    # Plots
    plot_bar(summary, REPORT_DIR / "cnn_aug_bar.png")
    plot_per_subject_boxplot(loaded, REPORT_DIR / "cnn_aug_per_subject.png")

    print("\nDONE -- CNN augmentation comparison complete.")


if __name__ == "__main__":
    main()

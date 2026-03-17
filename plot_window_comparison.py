# plot_window_comparison.py
# Compares 150ms vs 250ms window performance from master_classical_results.csv.
#
# For each (window_ms, feat_set, model) combination, takes the best result
# (highest f1_macro_mean) from all available runs, including hyperparameter
# sweep files. This is valid because we are asking: what is the best this
# window size can achieve, to justify the window size selection for LOSO?
#
# Outputs:
#   report_figs/window_comparison_bar.png  — grouped bar chart (3 metrics)
#   report_figs/window_comparison_table.csv — underlying numbers

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent
MASTER_CSV    = PROJECT_ROOT / "results_classical" / "master_classical_results.csv"
OUT_DIR       = PROJECT_ROOT / "report_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
MODELS_KEEP = {
    "RF_balanced":           "RF",
    "SVM_RBF_balanced_scaled": "SVM",
}

METRICS = {
    "f1_macro_mean":  ("F1 Macro",         "f1_macro_std"),
    "bal_acc_mean":   ("Balanced Accuracy", "bal_acc_std"),
    "acc_mean":       ("Accuracy",          "acc_std"),
}

WINDOW_COLOURS = {150: "#4C72B0", 250: "#DD8452"}  # blue / orange

FEAT_LABELS = {"base": "Base TD", "ext": "Extended TD"}


def load_and_filter(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)

    # Keep only the two canonical models
    df = df[df["model"].isin(MODELS_KEEP)].copy()

    # For each (window_ms, feat_set, model) take the run with highest F1.
    # This picks the best sweep result for 250ms-base where no single
    # canonical run exists, and the single result for all other combos.
    df = (
        df.sort_values("f1_macro_mean", ascending=False)
          .groupby(["window_ms", "feat_set", "model"], sort=False)
          .first()
          .reset_index()
    )

    df["model_label"] = df["model"].map(MODELS_KEEP)
    df["feat_label"]  = df["feat_set"].map(FEAT_LABELS)

    return df


def make_bar_chart(df: pd.DataFrame) -> None:
    feat_sets   = ["base", "ext"]
    model_keys  = list(MODELS_KEEP.keys())
    windows     = [150, 250]
    metric_keys = list(METRICS.keys())

    # x-axis positions: one slot per (feat_set, model) pair
    n_groups    = len(feat_sets) * len(model_keys)   # 4
    group_width = 0.8
    bar_width   = group_width / len(windows)          # 0.4
    x_base      = np.arange(n_groups)

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(15, 5), sharey=False)
    fig.suptitle(
        "Window Length Comparison: 150 ms vs 250 ms\n"
        "(Subject-Dependent Baselines, Best Result per Configuration)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Build group labels
    x_labels = []
    for fs in feat_sets:
        for mk in model_keys:
            x_labels.append(f"{FEAT_LABELS[fs]}\n{MODELS_KEEP[mk]}")

    for ax, metric_key in zip(axes, metric_keys):
        metric_label, std_key = METRICS[metric_key]

        for w_idx, win in enumerate(windows):
            heights = []
            errs    = []

            for fs in feat_sets:
                for mk in model_keys:
                    row = df[
                        (df["window_ms"] == win) &
                        (df["feat_set"]  == fs)  &
                        (df["model"]     == mk)
                    ]
                    if row.empty:
                        heights.append(0.0)
                        errs.append(0.0)
                    else:
                        heights.append(float(row[metric_key].iloc[0]))
                        errs.append(
                            float(row[std_key].iloc[0])
                            if std_key in row.columns and not row[std_key].isna().all()
                            else 0.0
                        )

            offsets = x_base + (w_idx - (len(windows) - 1) / 2) * bar_width
            bars = ax.bar(
                offsets, heights,
                width=bar_width * 0.9,
                color=WINDOW_COLOURS[win],
                label=f"{win} ms",
                capsize=4,
                zorder=3,
            )
            ax.errorbar(
                offsets, heights, yerr=errs,
                fmt="none", color="black", linewidth=1.2, capsize=4, zorder=4,
            )

            # Value labels on bars
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + 0.005,
                        f"{h:.3f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=90,
                    )

        ax.set_title(metric_label, fontsize=11)
        ax.set_xticks(x_base)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0.55, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.legend(title="Window", fontsize=9)

        # Vertical separator between feat_set groups
        ax.axvline(x=1.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    out_png = OUT_DIR / "window_comparison_bar.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print(f"[save] {out_png}")
    plt.show()


def save_table(df: pd.DataFrame) -> None:
    cols = ["window_ms", "feat_set", "model", "f1_macro_mean", "f1_macro_std",
            "bal_acc_mean", "bal_acc_std", "acc_mean", "acc_std"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].sort_values(["feat_set", "model", "window_ms"])
    out_csv = OUT_DIR / "window_comparison_table.csv"
    out.to_csv(out_csv, index=False)
    print(f"[save] {out_csv}")
    print(out.to_string(index=False))


def main() -> None:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"Master CSV not found: {MASTER_CSV}\n"
            "Run summarize_classical_results_patched.py first."
        )

    df = load_and_filter(MASTER_CSV)

    print("\n==== DATA USED FOR COMPARISON ====")
    print(df[["window_ms", "feat_set", "model", "f1_macro_mean",
              "bal_acc_mean", "acc_mean", "source_file"]].to_string(index=False))

    save_table(df)
    make_bar_chart(df)

    # Print a plain summary
    print("\n==== WINDOW SIZE SUMMARY ====")
    for metric_key, (label, _) in METRICS.items():
        print(f"\n{label}:")
        pivot = df.pivot_table(
            index=["feat_set", "model"],
            columns="window_ms",
            values=metric_key,
        ).round(4)
        pivot["Delta (250-150)"] = (pivot[250] - pivot[150]).round(4)
        print(pivot.to_string())


if __name__ == "__main__":
    main()

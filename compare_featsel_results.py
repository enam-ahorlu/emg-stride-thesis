# compare_featsel_results.py
"""
Comparison and plotting script for feature selection ablation results.

Loads per-subject LOSO subjectwise CSVs from five conditions and produces:
  report_figs/featsel_results.csv     -- summary table
  report_figs/featsel_delta.csv       -- delta vs Full-72 baseline
  report_figs/featsel_bar.png         -- grouped bar chart
  report_figs/featsel_per_subject.png -- boxplot per condition per model

Usage
-----
    python compare_featsel_results.py
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

# ---------------------------------------------------------------------------
# Condition registry: (dir_name, label, n_features, method)
# ---------------------------------------------------------------------------
CONDITIONS = [
    ("results_loso_freq_persubj",  "Full-72",  72, "none"),
    ("results_loso_freq_rfe36",    "RFE-36",   36, "rfe"),
    ("results_loso_freq_rfe27",    "RFE-27",   27, "rfe"),
    ("results_loso_freq_mi36",     "MI-36",    36, "mi"),
    ("results_loso_freq_mi27",     "MI-27",    27, "mi"),
]

MODELS = ["SVM", "RF"]

# Bar-chart colours per condition
COND_COLORS = {
    "Full-72": "#555555",   # dark grey
    "RFE-36":  "#1f77b4",   # blue
    "RFE-27":  "#aec7e8",   # light blue
    "MI-36":   "#2ca02c",   # green
    "MI-27":   "#98df8a",   # light green
}


def load_condition_model(result_dir: Path, label: str, model: str):
    """
    Load the per-subject subjectwise CSV for a given condition + model.
    Returns None gracefully if the directory or file is missing.
    """
    if not result_dir.exists():
        warnings.warn(f"[skip] directory not found: {result_dir}")
        return None

    # Search for subjectwise CSV in result dir and checkpoints subdir
    pattern = f"*{model}*subjectwise*.csv"
    matches = list(result_dir.glob(pattern))
    ckpt_dir = result_dir / "checkpoints"
    if ckpt_dir.exists():
        matches += list(ckpt_dir.glob(f"*{model}*subjectwise*.csv"))

    if not matches:
        warnings.warn(f"[skip] no subjectwise CSV for model={model} in {result_dir}")
        return None

    # Prefer non-checkpoint file
    non_ckpt = [p for p in matches if "ckpt" not in p.name]
    chosen = non_ckpt[0] if non_ckpt else matches[0]
    df = pd.read_csv(chosen)

    if "f1_macro" not in df.columns:
        warnings.warn(f"[skip] 'f1_macro' column missing in {chosen}")
        return None

    df["condition"] = label
    df["model"] = model
    return df


def build_summary_table(all_data: dict, conditions_meta: list) -> pd.DataFrame:
    """Build summary table: condition x model -> mean/sd F1, n_features, method."""
    label_to_meta = {lbl: (n_feat, method) for _, lbl, n_feat, method in conditions_meta}
    rows = []
    for label, model_dfs in all_data.items():
        n_feat, method = label_to_meta.get(label, (None, None))
        for model, df in model_dfs.items():
            if df is None:
                continue
            rows.append({
                "condition":     label,
                "model":         model,
                "f1_macro_mean": float(df["f1_macro"].mean()),
                "f1_macro_sd":   float(df["f1_macro"].std(ddof=1)),
                "n_subjects":    len(df),
                "n_features":    n_feat,
                "method":        method,
            })
    return pd.DataFrame(rows)


def build_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute delta F1 vs Full-72 baseline per model."""
    baseline = (
        summary[summary["condition"] == "Full-72"]
        .set_index("model")["f1_macro_mean"]
    )
    rows = []
    for _, row in summary.iterrows():
        model = row["model"]
        base_f1 = baseline.get(model, float("nan"))
        rows.append({
            "condition":  row["condition"],
            "model":      model,
            "f1_mean":    row["f1_macro_mean"],
            "f1_sd":      row["f1_macro_sd"],
            "delta_f1":   row["f1_macro_mean"] - base_f1,
            "n_features": row["n_features"],
            "method":     row["method"],
        })
    return pd.DataFrame(rows)


def plot_grouped_bar(summary: pd.DataFrame, out_png: Path) -> None:
    """Grouped bar chart: x=models (SVM, RF), bars=conditions, with error bars."""
    conditions = [lbl for _, lbl, _, _ in CONDITIONS]
    present = [c for c in conditions if c in summary["condition"].values]

    x = np.arange(len(MODELS))
    n_conds = len(present)
    width = 0.8 / max(n_conds, 1)
    offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, cond in enumerate(present):
        means, errs = [], []
        for model in MODELS:
            row = summary[(summary["condition"] == cond) & (summary["model"] == model)]
            if row.empty:
                means.append(float("nan"))
                errs.append(0.0)
            else:
                means.append(float(row["f1_macro_mean"].iloc[0]))
                errs.append(float(row["f1_macro_sd"].iloc[0]))

        ax.bar(
            x + offsets[i],
            means,
            width=width * 0.9,
            label=cond,
            color=COND_COLORS.get(cond, f"C{i}"),
            yerr=errs,
            capsize=4,
            error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=12)
    ax.set_ylabel("Macro F1 (LOSO)", fontsize=11)
    ax.set_title("Feature Selection: LOSO F1 (Freq-72, Per-Subject Norm)", fontsize=12)
    ax.legend(title="Feature Set", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")


def plot_per_subject_boxplot(all_data: dict, out_png: Path) -> None:
    """Side-by-side boxplot panels for SVM and RF."""
    conditions = [lbl for _, lbl, _, _ in CONDITIONS]
    present = [c for c in conditions if any(
        all_data.get(c, {}).get(m) is not None for m in MODELS
    )]

    n_models = len(MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, MODELS):
        data_list: list = []
        tick_labels: list = []
        for cond in present:
            df = all_data.get(cond, {}).get(model)
            if df is not None:
                data_list.append(df["f1_macro"].dropna().values)
                tick_labels.append(cond)
            else:
                data_list.append(np.array([]))
                tick_labels.append(f"{cond} (N/A)")

        colors = [COND_COLORS.get(lbl, "C0") for lbl in present]
        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{model}", fontsize=12)
        ax.set_ylabel("Per-subject F1 (LOSO)", fontsize=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.05)

    fig.suptitle(
        "Feature Selection: Per-Subject F1 Distribution (Per-Subject Norm)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[save] {out_png}")


def main() -> None:
    # Load all data
    all_data: dict = {}
    for dir_name, label, n_feat, method in CONDITIONS:
        result_dir = ROOT / dir_name
        model_dfs: dict = {}
        for model in MODELS:
            df = load_condition_model(result_dir, label, model)
            model_dfs[model] = df
        all_data[label] = model_dfs

    # Summary table
    summary = build_summary_table(all_data, CONDITIONS)
    if summary.empty:
        print(
            "[ERROR] No data loaded. Check that results directories exist "
            "and contain *subjectwise*.csv files."
        )
        sys.exit(1)

    out_summary = REPORT_DIR / "featsel_results.csv"
    summary.to_csv(out_summary, index=False)
    print(f"[save] {out_summary}")
    print(summary.to_string(index=False))

    # Delta table
    delta = build_delta_table(summary)
    out_delta = REPORT_DIR / "featsel_delta.csv"
    delta.to_csv(out_delta, index=False)
    print(f"[save] {out_delta}")
    print(delta.to_string(index=False))

    # Plots
    plot_grouped_bar(summary, REPORT_DIR / "featsel_bar.png")
    plot_per_subject_boxplot(all_data, REPORT_DIR / "featsel_per_subject.png")

    print("\nDONE -- feature selection comparison complete.")


if __name__ == "__main__":
    main()

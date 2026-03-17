# merge_plot_generalization.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _lineplot_by_subject(df: pd.DataFrame, value_col: str, title: str, ylabel: str, outpath: Path, dpi: int = 220):
    d = df.dropna(subset=["subject", value_col]).copy()
    if d.empty:
        print(f"[skip] empty for {outpath.name}")
        return
    d = d.sort_values("subject")
    plt.figure(figsize=(12, 4.6))
    plt.plot(d["subject"].astype(int), d[value_col].astype(float), marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Subject")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    if d["subject"].nunique() > 15:
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[save] {outpath}")


def _boxplot(groups: Dict[str, np.ndarray], title: str, ylabel: str, outpath: Path, dpi: int = 220):
    labels = list(groups.keys())
    data = [np.asarray(groups[k], dtype=float) for k in labels]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_alpha(0.25)

    # jittered points
    rng = np.random.default_rng(42)
    for i, y in enumerate(data, start=1):
        x = rng.normal(loc=i, scale=0.04, size=len(y))
        ax.scatter(x, y, s=18, alpha=0.55)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[save] {outpath}")


def _delta_bar(df: pd.DataFrame, delta_col: str, title: str, ylabel: str, outpath: Path, dpi: int = 220):
    d = df.dropna(subset=["subject", delta_col]).copy()
    if d.empty:
        print(f"[skip] empty for {outpath.name}")
        return
    d = d.sort_values("subject")
    plt.figure(figsize=(12, 4.6))
    plt.bar(d["subject"].astype(int), d[delta_col].astype(float))
    plt.title(title)
    plt.xlabel("Subject")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", alpha=0.25)
    if d["subject"].nunique() > 15:
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[save] {outpath}")


def main():
    ap = argparse.ArgumentParser(description="Make LOSO generalization plots from generalization_gap.csv.")
    ap.add_argument("--gap", default="results_loso_light/generalization_gap.csv", help="Output from compute_generalization_gap.py")
    ap.add_argument("--outdir", default="results_loso_light/generalization_plots", help="Directory for plots")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--models", default="", help="Optional comma-separated model filter (e.g., SVM,RF,LDA)")
    args = ap.parse_args()

    gap_path = Path(args.gap)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    df = pd.read_csv(gap_path)

    required = {"subject", "model", "f1_sd", "f1_loso", "delta_f1", "bal_acc_sd", "bal_acc_loso", "delta_bal_acc"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"gap CSV missing columns: {missing}. Have: {list(df.columns)}")

    if args.models.strip():
        keep = {m.strip().upper() for m in args.models.split(",") if m.strip()}
        df = df[df["model"].astype(str).str.upper().isin(keep)].copy()

    # Per-model plots
    for model, g in df.groupby("model"):
        mdir = outdir / str(model)
        _ensure_dir(mdir)

        # Subject-wise LOSO lines
        _lineplot_by_subject(
            g, "f1_loso",
            title=f"{model}: LOSO Macro-F1 by Subject",
            ylabel="Macro-F1 (LOSO)",
            outpath=mdir / f"{model}_loso_f1_by_subject.png",
            dpi=args.dpi,
        )
        _lineplot_by_subject(
            g, "bal_acc_loso",
            title=f"{model}: LOSO Balanced Accuracy by Subject",
            ylabel="Balanced Accuracy (LOSO)",
            outpath=mdir / f"{model}_loso_balacc_by_subject.png",
            dpi=args.dpi,
        )

        # SD vs LOSO boxplots
        _boxplot(
            {"SD": g["f1_sd"].to_numpy(), "LOSO": g["f1_loso"].to_numpy()},
            title=f"{model}: Macro-F1 (SD vs LOSO)",
            ylabel="Macro-F1",
            outpath=mdir / f"{model}_box_sd_vs_loso_f1.png",
            dpi=args.dpi,
        )
        _boxplot(
            {"SD": g["bal_acc_sd"].to_numpy(), "LOSO": g["bal_acc_loso"].to_numpy()},
            title=f"{model}: Balanced Accuracy (SD vs LOSO)",
            ylabel="Balanced Accuracy",
            outpath=mdir / f"{model}_box_sd_vs_loso_balacc.png",
            dpi=args.dpi,
        )

        # Δ plots (generalization gap)
        _delta_bar(
            g, "delta_f1",
            title=f"{model}: Generalization Gap ΔF1 = F1_SD − F1_LOSO (per subject)",
            ylabel="ΔF1 (SD − LOSO)",
            outpath=mdir / f"{model}_delta_f1_bar.png",
            dpi=args.dpi,
        )
        _delta_bar(
            g, "delta_bal_acc",
            title=f"{model}: Generalization Gap ΔBalAcc (per subject)",
            ylabel="ΔBalAcc (SD − LOSO)",
            outpath=mdir / f"{model}_delta_balacc_bar.png",
            dpi=args.dpi,
        )

    # Combined: mean gap per model
    mean_gap = (
        df.groupby("model")[["delta_f1", "delta_bal_acc"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("delta_f1", ascending=False)
    )
    mean_gap.to_csv(outdir / "mean_gap_by_model.csv", index=False)
    print(f"[save] {outdir / 'mean_gap_by_model.csv'}")

    # Simple combined bar for ΔF1 mean
    plt.figure(figsize=(9.5, 4.8))
    plt.bar(mean_gap["model"].astype(str), mean_gap["delta_f1"].astype(float))
    plt.xticks(rotation=25, ha="right")
    plt.title("Mean Generalization Gap (ΔF1) by Model")
    plt.ylabel("Mean ΔF1 (SD − LOSO)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(outdir / "mean_delta_f1_by_model.png", dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"[save] {outdir / 'mean_delta_f1_by_model.png'}")


if __name__ == "__main__":
    main()

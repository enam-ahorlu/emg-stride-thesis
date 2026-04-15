#!/usr/bin/env python3
"""
compare_all_optimizations.py
=============================
Phases 3, 4, 5 of the optimisation comparison:

Phase 3 — Combined Optimisation Table:
  report_figs/optimization_summary.csv

Phase 4 — Optimisation Journey Plot:
  report_figs/optimization_journey.png
  report_figs/optimization_journey_table.csv

Phase 5 — Baseline vs Optimised Gap Plots:
  report_figs/baseline_all_models_delta_f1_bar.png
  report_figs/optimized_all_models_delta_f1_bar.png
  report_figs/gap_comparison_baseline_vs_optimized.png

All values loaded from actual result files (nothing hardcoded).
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).parent
RPT_DIR = ROOT / "report_figs"
RPT_DIR.mkdir(exist_ok=True)


# ============================================================
# Loaders
# ============================================================

def load_classical_subjectwise(results_dir: str | Path, model: str) -> np.ndarray | None:
    """Load per-subject F1 from subjectwise CSV or checkpoint CSV."""
    p = ROOT / results_dir if not Path(results_dir).is_absolute() else Path(results_dir)
    if not p.exists():
        print(f"    [WARN] dir not found: {p}")
        return None
    # Try subjectwise CSV first
    sw_files = sorted(p.glob(f"*{model}*subjectwise*.csv"))
    if sw_files:
        df = pd.read_csv(sw_files[0])
        subj_col = "heldout_subject" if "heldout_subject" in df.columns else "subject"
        df = df.drop_duplicates(subset=[subj_col]).sort_values(subj_col)
        return df["f1_macro"].values
    # Try checkpoint CSV
    ckpt_dir = p / "checkpoints"
    if ckpt_dir.exists():
        ckpt_files = sorted(ckpt_dir.glob(f"*{model}*ckpt.csv"))
        if ckpt_files:
            df = pd.read_csv(ckpt_files[0])
            subj_col = "heldout_subject" if "heldout_subject" in df.columns else "subject"
            df = df.drop_duplicates(subset=[subj_col]).sort_values(subj_col)
            return df["f1_macro"].values
    print(f"    [WARN] no subjectwise CSV for {model} in {p}")
    return None


def load_cnn_subjectwise(results_dir: str | Path) -> np.ndarray | None:
    """Load per-subject F1 from CNN per_subject_metrics CSV."""
    p = ROOT / results_dir if not Path(results_dir).is_absolute() else Path(results_dir)
    csv_path = p / "per_subject_metrics_cnn_loso.csv"
    if not csv_path.exists():
        print(f"    [WARN] CNN metrics not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path).drop_duplicates(subset=["subject"]).sort_values("subject")
    return df["f1_macro"].values


def load_ensemble_per_subject(csv_name: str) -> np.ndarray | None:
    """Load per-subject F1 from an ensemble per-subject CSV."""
    p = RPT_DIR / csv_name
    if not p.exists():
        print(f"    [WARN] ensemble CSV not found: {p}")
        return None
    df = pd.read_csv(p).sort_values("subject")
    return df["f1_macro"].values


def load_ensemble_summary() -> pd.DataFrame | None:
    """Load ensemble_summary.csv with mean F1 per ensemble config."""
    p = RPT_DIR / "ensemble_summary.csv"
    if not p.exists():
        print(f"    [WARN] ensemble_summary.csv not found")
        return None
    return pd.read_csv(p)


def load_sd_classical(model_key: str) -> float | None:
    """Load mean SD F1 for SVM or RF from per_subject_metrics_freq72_sd.csv.
    model_key: 'SVM_RBF_balanced_scaled' or 'RF_balanced'
    """
    p = ROOT / "results_classical_freq72_v2" / "per_subject_metrics_freq72_sd.csv"
    if not p.exists():
        print(f"    [WARN] SD classical CSV not found: {p}")
        return None
    df = pd.read_csv(p)
    sub = df[df["model"] == model_key]
    if sub.empty:
        print(f"    [WARN] model {model_key} not found in SD CSV")
        return None
    return float(sub["f1_macro"].mean())


def load_sd_cnn() -> float | None:
    """Load mean SD F1 for CNN from cnn_subjectdep results (5-fold CV)."""
    p = ROOT / "results_cnn" / "cnn_subjectdep_w250_env_zscore_5fold.csv"
    if not p.exists():
        print(f"    [WARN] CNN SD CSV not found: {p}")
        return None
    df = pd.read_csv(p)
    return float(df["f1"].mean())


def fmt(val, decimals=3):
    if val is None:
        return "---"
    return f"{val:.{decimals}f}"


def mean_or_none(arr):
    if arr is None or len(arr) == 0:
        return None
    return float(np.mean(arr))


def sd_or_none(arr):
    if arr is None or len(arr) < 2:
        return None
    return float(np.std(arr, ddof=1))


# ============================================================
# PHASE 3 — Combined Optimisation Table
# ============================================================

def build_optimization_table():
    print("=" * 70)
    print("PHASE 3: Combined Optimisation Table")
    print("=" * 70)

    rows = []

    def add_row(stage, config, svm_arr, rf_arr, cnn_arr, notes=""):
        rows.append({
            "stage": stage,
            "config": config,
            "svm_f1": fmt(mean_or_none(svm_arr)),
            "rf_f1": fmt(mean_or_none(rf_arr)),
            "cnn_f1": fmt(mean_or_none(cnn_arr)),
            "svm_sd": fmt(sd_or_none(svm_arr)),
            "rf_sd": fmt(sd_or_none(rf_arr)),
            "cnn_sd": fmt(sd_or_none(cnn_arr)),
            "notes": notes,
        })

    # --- Baseline: Full-72, global norm ---
    print("\n  Loading Baseline (full-72, global norm)...")
    svm_base = load_classical_subjectwise("results_loso_freq", "SVM")
    rf_base  = load_classical_subjectwise("results_loso_freq", "RF")
    cnn_base = load_cnn_subjectwise("results_cnn_loso")
    add_row("Baseline", "Full-72, global norm", svm_base, rf_base, cnn_base, "Starting point")

    # --- Norm only: Full-72, per_subject norm ---
    print("  Loading Norm only (full-72, per_subject norm)...")
    svm_persubj = load_classical_subjectwise("results_loso_freq_persubj", "SVM")
    rf_persubj  = load_classical_subjectwise("results_loso_freq_persubj", "RF")
    cnn_persubj = load_cnn_subjectwise("results_cnn_loso_norm_persubj")
    add_row("Norm only", "Full-72, per_subject norm", svm_persubj, rf_persubj, cnn_persubj,
            "Best normalisation")

    # --- Feat sel only (global): RFE-36, global norm ---
    print("  Loading Feat sel only (RFE-36, global norm)...")
    svm_rfe36_glob = load_classical_subjectwise("results_loso_freq_rfe36_globnorm", "SVM")
    rf_rfe36_glob  = load_classical_subjectwise("results_loso_freq_rfe36_globnorm", "RF")
    add_row("Feat sel only (global)", "RFE-36, global norm",
            svm_rfe36_glob, rf_rfe36_glob, None, "Independent feat sel")

    # --- Feat sel only (global): MI-36, global norm ---
    print("  Loading Feat sel only (MI-36, global norm)...")
    svm_mi36_glob = load_classical_subjectwise("results_loso_freq_mi36_globnorm", "SVM")
    rf_mi36_glob  = load_classical_subjectwise("results_loso_freq_mi36_globnorm", "RF")
    add_row("Feat sel only (global)", "MI-36, global norm",
            svm_mi36_glob, rf_mi36_glob, None, "Independent feat sel")

    # --- Aug only (global): Gaussian, global norm ---
    print("  Loading Aug only (Gaussian, global norm)...")
    cnn_gauss_glob = load_cnn_subjectwise("results_cnn_loso_aug_gaussian_globnorm")
    add_row("Aug only (global)", "Gaussian, global norm", None, None, cnn_gauss_glob,
            "Independent augmentation")

    # --- Aug only (global): ChanDrop, global norm ---
    print("  Loading Aug only (ChanDrop, global norm)...")
    cnn_chandrop_glob = load_cnn_subjectwise("results_cnn_loso_aug_chandrop_globnorm")
    add_row("Aug only (global)", "ChanDrop, global norm", None, None, cnn_chandrop_glob,
            "Independent augmentation")

    # --- Aug only (global): TimeMask, global norm ---
    print("  Loading Aug only (TimeMask, global norm)...")
    cnn_timemask_glob = load_cnn_subjectwise("results_cnn_loso_aug_timemask_globnorm")
    add_row("Aug only (global)", "TimeMask, global norm", None, None, cnn_timemask_glob,
            "Independent augmentation")

    # --- Combined: RFE-36, per_subject norm ---
    print("  Loading Combined (RFE-36, per_subject norm)...")
    svm_rfe36_persubj = load_classical_subjectwise("results_loso_freq_rfe36", "SVM")
    rf_rfe36_persubj  = load_classical_subjectwise("results_loso_freq_rfe36", "RF")
    add_row("Combined", "RFE-36, per_subject norm",
            svm_rfe36_persubj, rf_rfe36_persubj, None, "Norm + feat sel")

    # --- Combined: Gaussian, per_subject norm ---
    print("  Loading Combined (Gaussian, per_subject norm)...")
    cnn_gauss_persubj = load_cnn_subjectwise("results_cnn_loso_aug_gaussian")
    add_row("Combined", "Gaussian, per_subject norm", None, None, cnn_gauss_persubj,
            "Norm + augmentation")

    # --- Ensemble: SVM+RF per_subject ---
    print("  Loading Ensemble (SVM+RF)...")
    ens_svmrf = load_ensemble_per_subject("ensemble_svm_rf_per_subject.csv")
    ens_summary = load_ensemble_summary()

    # For SVM+RF ensemble, we report as a single F1 (it's a voting ensemble)
    rows.append({
        "stage": "Ensemble (SVM+RF)",
        "config": "per_subject norm",
        "svm_f1": fmt(mean_or_none(ens_svmrf)),
        "rf_f1": "---",
        "cnn_f1": "---",
        "svm_sd": fmt(sd_or_none(ens_svmrf)),
        "rf_sd": "---",
        "cnn_sd": "---",
        "notes": "2-way voting",
    })

    # --- Ensemble: 3-way ---
    print("  Loading Ensemble (3-way)...")
    ens_3way = load_ensemble_per_subject("ensemble_3way_per_subject.csv")
    rows.append({
        "stage": "Ensemble (3-way)",
        "config": "per_subject norm",
        "svm_f1": "---",
        "rf_f1": "---",
        "cnn_f1": fmt(mean_or_none(ens_3way)),
        "svm_sd": "---",
        "rf_sd": "---",
        "cnn_sd": fmt(sd_or_none(ens_3way)),
        "notes": "SVM+RF+CNN voting",
    })

    df = pd.DataFrame(rows)
    out_csv = RPT_DIR / "optimization_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  [saved] {out_csv}")

    # Print nicely
    print(f"\n{'='*100}")
    print(f"  {'Stage':<25s} {'Config':<28s} {'SVM F1':>8s} {'RF F1':>8s} {'CNN F1':>8s}  {'Notes'}")
    print(f"  {'-'*25} {'-'*28} {'-'*8} {'-'*8} {'-'*8}  {'-'*25}")
    for _, r in df.iterrows():
        print(f"  {r['stage']:<25s} {r['config']:<28s} {r['svm_f1']:>8s} {r['rf_f1']:>8s} {r['cnn_f1']:>8s}  {r['notes']}")
    print(f"{'='*100}\n")

    # Return key arrays for other phases
    return {
        "svm_base": svm_base, "rf_base": rf_base, "cnn_base": cnn_base,
        "svm_persubj": svm_persubj, "rf_persubj": rf_persubj, "cnn_persubj": cnn_persubj,
        "svm_rfe36_persubj": svm_rfe36_persubj,
        "rf_rfe36_persubj": rf_rfe36_persubj,
        "cnn_gauss_persubj": cnn_gauss_persubj,
        "ens_3way": ens_3way,
        "ens_svmrf": ens_svmrf,
        "df": df,
    }


# ============================================================
# PHASE 4 — Optimisation Journey Plot
# ============================================================

def build_journey_plot(data: dict):
    print("=" * 70)
    print("PHASE 4: Optimisation Journey Plot")
    print("=" * 70)

    stage_labels = [
        "Baseline\n(global norm)",
        "Norm\n(per-subject)",
        "Feat Sel / Aug\n(+ per-subject)",
        "Ensemble\n(3-way)",
    ]

    # Build per-model journey arrays
    # SVM: baseline -> per_subj -> RFE-36 per_subj -> ensemble 3-way
    svm_journey = [
        mean_or_none(data["svm_base"]),
        mean_or_none(data["svm_persubj"]),
        mean_or_none(data["svm_rfe36_persubj"]),  # RFE-36 + per_subject
        mean_or_none(data["ens_3way"]),
    ]

    # RF: baseline -> per_subj -> stays at full-72 per_subj -> ensemble 3-way
    rf_journey = [
        mean_or_none(data["rf_base"]),
        mean_or_none(data["rf_persubj"]),
        mean_or_none(data["rf_persubj"]),  # stays at full-72 per_subj (RFE hurts RF)
        mean_or_none(data["ens_3way"]),
    ]

    # CNN: baseline -> per_subj -> gaussian per_subj -> ensemble 3-way
    cnn_journey = [
        mean_or_none(data["cnn_base"]),
        mean_or_none(data["cnn_persubj"]),
        mean_or_none(data["cnn_gauss_persubj"]),
        mean_or_none(data["ens_3way"]),
    ]

    print(f"  SVM journey: {[fmt(v) for v in svm_journey]}")
    print(f"  RF  journey: {[fmt(v) for v in rf_journey]}")
    print(f"  CNN journey: {[fmt(v) for v in cnn_journey]}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"SVM": "#2196F3", "RF": "#4CAF50", "CNN": "#FF9800"}
    markers = {"SVM": "o", "RF": "s", "CNN": "D"}

    for model_name, journey in [("SVM", svm_journey), ("RF", rf_journey), ("CNN", cnn_journey)]:
        x_vals, y_vals = [], []
        for i, val in enumerate(journey):
            if val is not None:
                x_vals.append(i)
                y_vals.append(val)

        if x_vals:
            # Draw the main line (stages 0-2)
            main_x = [x for x in x_vals if x < 3]
            main_y = [y for x, y in zip(x_vals, y_vals) if x < 3]
            if main_x:
                ax.plot(main_x, main_y, f"-{markers[model_name]}",
                        color=colors[model_name], label=model_name,
                        markersize=10, linewidth=2.5)

            # Draw dashed line from stage 2 to ensemble (stage 3)
            if 2 in x_vals and 3 in x_vals:
                idx2 = x_vals.index(2)
                idx3 = x_vals.index(3)
                ax.plot([x_vals[idx2], x_vals[idx3]], [y_vals[idx2], y_vals[idx3]],
                        "--", color=colors[model_name], linewidth=1.5, alpha=0.5)

            # Annotate each point (except ensemble)
            for xi, yi in zip(x_vals, y_vals):
                if xi < 3:
                    offset_y = 12
                    ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                               xytext=(0, offset_y), ha="center", fontsize=9,
                               fontweight="bold", color=colors[model_name])

    # Add ensemble star marker (convergence point)
    ens_f1 = mean_or_none(data["ens_3way"])
    if ens_f1 is not None:
        ax.plot(3, ens_f1, "*", color="#E91E63", markersize=20,
                label="3-way ensemble", markeredgecolor="black", markeredgewidth=0.5,
                zorder=10)
        ax.annotate(f"{ens_f1:.3f}", (3, ens_f1), textcoords="offset points",
                   xytext=(0, 14), ha="center", fontsize=10, fontweight="bold",
                   color="#E91E63")

    ax.set_xticks(range(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=10)
    ax.set_ylabel("LOSO Macro F1", fontsize=12)
    ax.set_title("Optimisation Journey: LOSO F1 Across All Stages\n(Freq-72, 40-Subject LOSO)",
                fontsize=13)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Dynamic y-axis limits
    all_vals = [v for v in svm_journey + rf_journey + cnn_journey if v is not None]
    if all_vals:
        y_min = min(all_vals) - 0.03
        y_max = max(all_vals) + 0.03
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    out_png = RPT_DIR / "optimization_journey.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out_png}")

    # Also save a clean thesis table
    thesis_rows = []
    for i, label in enumerate(stage_labels):
        clean_label = label.replace("\n", " ")
        thesis_rows.append({
            "stage_num": i,
            "stage_label": clean_label,
            "svm_f1": fmt(svm_journey[i]),
            "rf_f1": fmt(rf_journey[i]),
            "cnn_f1": fmt(cnn_journey[i]),
        })
    thesis_df = pd.DataFrame(thesis_rows)
    thesis_path = RPT_DIR / "optimization_journey_table.csv"
    thesis_df.to_csv(thesis_path, index=False)
    print(f"  [saved] {thesis_path}")


# ============================================================
# PHASE 5 — Baseline vs Optimised Gap Plots
# ============================================================

def build_gap_plots(data: dict):
    print("\n" + "=" * 70)
    print("PHASE 5: Baseline vs Optimised Gap Plots")
    print("=" * 70)

    # Load SD F1 values
    print("\n  Loading SD F1 values...")
    sd_svm = load_sd_classical("SVM_RBF_balanced_scaled")
    sd_rf  = load_sd_classical("RF_balanced")
    sd_cnn = load_sd_cnn()
    print(f"    SD SVM F1 = {fmt(sd_svm)}")
    print(f"    SD RF  F1 = {fmt(sd_rf)}")
    print(f"    SD CNN F1 = {fmt(sd_cnn)}")

    # LOSO baseline (global norm)
    loso_base_svm = mean_or_none(data["svm_base"])
    loso_base_rf  = mean_or_none(data["rf_base"])
    loso_base_cnn = mean_or_none(data["cnn_base"])
    print(f"\n    LOSO Baseline: SVM={fmt(loso_base_svm)}, RF={fmt(loso_base_rf)}, CNN={fmt(loso_base_cnn)}")

    # LOSO optimised (best per-model)
    # SVM: RFE-36 + per_subject
    # RF: full-72 + per_subject
    # CNN: Gaussian + per_subject
    loso_opt_svm = mean_or_none(data["svm_rfe36_persubj"])
    loso_opt_rf  = mean_or_none(data["rf_persubj"])
    loso_opt_cnn = mean_or_none(data["cnn_gauss_persubj"])
    print(f"    LOSO Optimised: SVM={fmt(loso_opt_svm)}, RF={fmt(loso_opt_rf)}, CNN={fmt(loso_opt_cnn)}")

    models = ["SVM", "RF", "CNN"]
    sd_vals = [sd_svm, sd_rf, sd_cnn]
    baseline_loso = [loso_base_svm, loso_base_rf, loso_base_cnn]
    optimised_loso = [loso_opt_svm, loso_opt_rf, loso_opt_cnn]

    # Compute gaps
    baseline_gaps = [s - l if s is not None and l is not None else None
                     for s, l in zip(sd_vals, baseline_loso)]
    optimised_gaps = [s - l if s is not None and l is not None else None
                      for s, l in zip(sd_vals, optimised_loso)]

    print(f"\n    Baseline gaps:  {[fmt(g) for g in baseline_gaps]}")
    print(f"    Optimised gaps: {[fmt(g) for g in optimised_gaps]}")

    bar_colors = {"SVM": "#2196F3", "RF": "#4CAF50", "CNN": "#FF9800"}
    model_colors = [bar_colors[m] for m in models]

    # ---- Plot 1: Baseline delta F1 bar ----
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(models))
    bars = ax1.bar(x, [g if g is not None else 0 for g in baseline_gaps],
                   color=model_colors, edgecolor="black", linewidth=0.5, width=0.5)
    for i, (bar, gap) in enumerate(zip(bars, baseline_gaps)):
        if gap is not None:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{gap:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12)
    ax1.set_ylabel("Generalization Gap (SD F1 - LOSO F1)", fontsize=11)
    ax1.set_title("Generalization Gap at Baseline\n(Full-72, Global Norm)", fontsize=13)
    ax1.set_ylim(0, max([g for g in baseline_gaps if g is not None], default=0.3) + 0.04)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out1 = RPT_DIR / "baseline_all_models_delta_f1_bar.png"
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"\n  [saved] {out1}")

    # ---- Plot 2: Optimised delta F1 bar ----
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    bars = ax2.bar(x, [g if g is not None else 0 for g in optimised_gaps],
                   color=model_colors, edgecolor="black", linewidth=0.5, width=0.5)
    for i, (bar, gap) in enumerate(zip(bars, optimised_gaps)):
        if gap is not None:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{gap:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12)
    ax2.set_ylabel("Generalization Gap (SD F1 - LOSO F1)", fontsize=11)
    ax2.set_title("Generalization Gap After Optimisation\n(Best Config per Model)", fontsize=13)
    ax2.set_ylim(0, max([g for g in optimised_gaps if g is not None], default=0.3) + 0.04)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out2 = RPT_DIR / "optimized_all_models_delta_f1_bar.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [saved] {out2}")

    # ---- Plot 3: Grouped bar — baseline vs optimised side by side ----
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    width = 0.30
    x = np.arange(len(models))

    bars_base = ax3.bar(x - width / 2,
                        [g if g is not None else 0 for g in baseline_gaps],
                        width, label="Baseline (global norm)",
                        color="#EF5350", edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_opt = ax3.bar(x + width / 2,
                       [g if g is not None else 0 for g in optimised_gaps],
                       width, label="Optimised (best config)",
                       color="#66BB6A", edgecolor="black", linewidth=0.5, alpha=0.85)

    # Value labels
    for bar, gap in zip(bars_base, baseline_gaps):
        if gap is not None:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                     f"{gap:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, gap in zip(bars_opt, optimised_gaps):
        if gap is not None:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                     f"{gap:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Reduction arrows/annotations
    for i, (gb, go) in enumerate(zip(baseline_gaps, optimised_gaps)):
        if gb is not None and go is not None:
            reduction = gb - go
            pct = 100 * reduction / gb if gb > 0 else 0
            mid_y = max(gb, go) + 0.025
            ax3.annotate(f"-{reduction:.3f}\n({pct:.0f}%)",
                        xy=(i, mid_y), ha="center", fontsize=8,
                        color="#1565C0", fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=12)
    ax3.set_ylabel("Generalization Gap (SD F1 - LOSO F1)", fontsize=11)
    ax3.set_title("Generalization Gap: Baseline vs Optimised\n(Freq-72, 40-Subject LOSO)", fontsize=13)
    ax3.legend(fontsize=10, loc="upper left")
    all_gaps = [g for g in baseline_gaps + optimised_gaps if g is not None]
    ax3.set_ylim(0, max(all_gaps, default=0.3) + 0.06)
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out3 = RPT_DIR / "gap_comparison_baseline_vs_optimized.png"
    fig3.savefig(out3, dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [saved] {out3}")


# ============================================================
# Main
# ============================================================

def main():
    print("\n" + "#" * 70)
    print("# compare_all_optimizations.py")
    print("#" * 70 + "\n")

    data = build_optimization_table()
    build_journey_plot(data)
    build_gap_plots(data)

    print("\n" + "=" * 70)
    print("=== compare_all_optimizations.py COMPLETE ===")
    print("=" * 70)


if __name__ == "__main__":
    main()

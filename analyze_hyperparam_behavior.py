# analyze_hyperparam_behavior.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing required columns: {missing}")


def _save_line(df: pd.DataFrame, x: str, y: str, title: str, out_png: Path) -> None:
    if df.empty:
        print(f"[skip] empty plot for {out_png.name}")
        return
    plt.figure(figsize=(8.5, 4.8))
    plt.plot(df[x].astype(float), df[y].astype(float), marker="o")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()
    print(f"[save] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze SVM C sweep and RF n_estimators sweep behavior from master results.")
    ap.add_argument("--master", type=str, required=True, help="Path to master_classical_results.csv")
    ap.add_argument("--window-ms", type=int, default=250, help="Filter window_ms (default: 250)")
    ap.add_argument("--feat-set", type=str, default="base", choices=["base", "ext"], help="Filter feat_set")
    ap.add_argument("--out-dir", type=str, default="hparam_analysis", help="Output directory")
    ap.add_argument("--make-plots", action="store_true", help="Also save simple line plots")
    args = ap.parse_args()

    master_path = Path(args.master)
    df = pd.read_csv(master_path)

    _ensure_cols(
        df,
        ["window_ms", "feat_set", "model", "f1_macro_mean", "bal_acc_mean", "fit_time_mean_sec", "infer_time_per_window_ms"],
        "master",
    )

    # Common filter
    df = df[(df["window_ms"] == int(args.window_ms)) & (df["feat_set"] == args.feat_set)].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # SVM: sweep over C 
    svm_df = df[df["model"].astype(str).str.startswith("SVM_")].copy()
    if not svm_df.empty and "svm_c" in svm_df.columns:
        # Keep only runs where svm_c is present
        svm_df = svm_df[svm_df["svm_c"].notna()].copy()

        # If svm_scale exists, stratify by it, else treat all as one group
        if "svm_scale" in svm_df.columns:
            group_cols = ["svm_scale", "svm_c"]
        else:
            group_cols = ["svm_c"]

        svm_sweep = (
            svm_df.groupby(group_cols, dropna=False)[
                ["f1_macro_mean", "bal_acc_mean", "fit_time_mean_sec", "infer_time_per_window_ms",
                 "sv_support_vectors_mean", "sv_frac_mean", "sv_n_iter_mean"]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(group_cols)
        )

        svm_csv = out_dir / "svm_c_sweep.csv"
        svm_sweep.to_csv(svm_csv, index=False)
        print(f"[save] {svm_csv} ({len(svm_sweep)} rows)")

        if args.make_plots:
            # If multiple svm_scale groups, plot each separately to keep plots readable
            if "svm_scale" in svm_sweep.columns:
                for scale_val, g in svm_sweep.groupby("svm_scale"):
                    tag = "scaled" if int(scale_val) == 1 else "unscaled"
                    g = g.sort_values("svm_c")
                    _save_line(g, "svm_c", "f1_macro_mean", f"SVM Macro-F1 vs C ({tag})", out_dir / f"svm_f1_vs_C_{tag}.png")
                    _save_line(g, "svm_c", "bal_acc_mean", f"SVM Balanced Acc vs C ({tag})", out_dir / f"svm_balacc_vs_C_{tag}.png")
                    _save_line(g, "svm_c", "fit_time_mean_sec", f"SVM Fit time vs C ({tag})", out_dir / f"svm_fit_vs_C_{tag}.png")
                    _save_line(g, "svm_c", "infer_time_per_window_ms", f"SVM Inference ms/window vs C ({tag})", out_dir / f"svm_infer_vs_C_{tag}.png")
                    if "sv_support_vectors_mean" in g.columns and g["sv_support_vectors_mean"].notna().any():
                        _save_line(g, "svm_c", "sv_support_vectors_mean", f"SVM #Support Vectors vs C ({tag})", out_dir / f"svm_svcount_vs_C_{tag}.png")
                        _save_line(g, "svm_c", "sv_frac_mean", f"SVM Support Vector Fraction vs C ({tag})", out_dir / f"svm_svfrac_vs_C_{tag}.png")
            else:
                g = svm_sweep.sort_values("svm_c")
                _save_line(g, "svm_c", "f1_macro_mean", "SVM Macro-F1 vs C", out_dir / "svm_f1_vs_C.png")
                _save_line(g, "svm_c", "bal_acc_mean", "SVM Balanced Acc vs C", out_dir / "svm_balacc_vs_C.png")
                _save_line(g, "svm_c", "fit_time_mean_sec", "SVM Fit time vs C", out_dir / "svm_fit_vs_C.png")
                _save_line(g, "svm_c", "infer_time_per_window_ms", "SVM Inference ms/window vs C", out_dir / "svm_infer_vs_C.png")
                if "sv_support_vectors_mean" in g.columns and g["sv_support_vectors_mean"].notna().any():
                    _save_line(g, "svm_c", "sv_support_vectors_mean", "SVM #Support Vectors vs C", out_dir / "svm_svcount_vs_C.png")
                    _save_line(g, "svm_c", "sv_frac_mean", "SVM Support Vector Fraction vs C", out_dir / "svm_svfrac_vs_C.png")
    else:
        print("[info] No SVM rows with svm_c found in this (window_ms, feat_set) slice. Skipping SVM sweep outputs.")


    # RF: sweep over n_estimators
    rf_df = df[df["model"].astype(str).str.startswith("RF_")].copy()
    if not rf_df.empty and "rf_n_estimators" in rf_df.columns:
        rf_df = rf_df[rf_df["rf_n_estimators"].notna()].copy()
        rf_sweep = (
            rf_df.groupby(["rf_n_estimators"], dropna=False)[
                ["f1_macro_mean", "bal_acc_mean", "fit_time_mean_sec", "infer_time_per_window_ms"]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["rf_n_estimators"])
        )

        rf_csv = out_dir / "rf_n_estimators_sweep.csv"
        rf_sweep.to_csv(rf_csv, index=False)
        print(f"[save] {rf_csv} ({len(rf_sweep)} rows)")

        if args.make_plots:
            _save_line(rf_sweep, "rf_n_estimators", "f1_macro_mean", "RF Macro-F1 vs #trees", out_dir / "rf_f1_vs_trees.png")
            _save_line(rf_sweep, "rf_n_estimators", "bal_acc_mean", "RF Balanced Acc vs #trees", out_dir / "rf_balacc_vs_trees.png")
            _save_line(rf_sweep, "rf_n_estimators", "fit_time_mean_sec", "RF Fit time vs #trees", out_dir / "rf_fit_vs_trees.png")
            _save_line(rf_sweep, "rf_n_estimators", "infer_time_per_window_ms", "RF Inference ms/window vs #trees", out_dir / "rf_infer_vs_trees.png")
    else:
        print("[info] No RF rows with rf_n_estimators found in this (window_ms, feat_set) slice. Skipping RF sweep outputs.")


if __name__ == "__main__":
    main()

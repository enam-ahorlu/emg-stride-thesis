# analyze_classical_compute.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def _safe_div(a: float, b: float) -> Optional[float]:
    try:
        if b == 0:
            return None
        return a / b
    except Exception:
        return None


def _save_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_png: Path) -> None:
    if df.empty:
        print(f"[skip] empty dataframe for plot: {out_png.name}")
        return

    plt.figure(figsize=(10, 4.8))
    plt.bar(df[x_col].astype(str), df[y_col].astype(float))
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(y_col)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close()
    print(f"[save] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze timing/computational cost for classical model runs.")
    ap.add_argument("--master", type=str, required=True, help="Path to master_classical_results.csv")
    ap.add_argument("--window-ms", type=int, default=None, help="Optional filter: window length (e.g., 250)")
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="Optional comma-separated model filter (e.g., 'SVM_RBF_balanced_scaled,RF_balanced,LDA_scaled')",
    )
    ap.add_argument("--out-dir", type=str, default="compute_analysis", help="Output directory")
    ap.add_argument("--make-plots", action="store_true", help="Save simple bar charts for timing/latency")
    args = ap.parse_args()

    master_path = Path(args.master)
    df = pd.read_csv(master_path)

    required = [
        "window_ms",
        "feat_set",
        "model",
        "f1_macro_mean",
        "bal_acc_mean",
        "fit_time_mean_sec",
        "pred_time_mean_sec",
        "infer_time_per_window_ms",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"master is missing required columns: {missing}")

    if args.window_ms is not None:
        df = df[df["window_ms"] == int(args.window_ms)].copy()

    if args.models:
        wanted = {m.strip() for m in args.models.split(",") if m.strip()}
        df = df[df["model"].isin(wanted)].copy()

    out = df.copy()
    out["config"] = (
        out["model"].astype(str)
        + " | "
        + out["feat_set"].astype(str)
        + " | w"
        + out["window_ms"].astype(int).astype(str)
    )

    out["f1_per_ms"] = out.apply(
        lambda r: _safe_div(float(r["f1_macro_mean"]), float(r["infer_time_per_window_ms"])), axis=1
    )
    out["balacc_per_ms"] = out.apply(
        lambda r: _safe_div(float(r["bal_acc_mean"]), float(r["infer_time_per_window_ms"])), axis=1
    )

    cols = [
        "source_file",
        "window_ms",
        "feat_set",
        "model",
        "n_samples",
        "n_features",
        "cv_splits",
        "f1_macro_mean",
        "bal_acc_mean",
        "fit_time_mean_sec",
        "pred_time_mean_sec",
        "infer_time_per_window_ms",
        "f1_per_ms",
        "balacc_per_ms",
    ]
    cols = [c for c in cols if c in out.columns]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = out_dir / "compute_summary.csv"
    out[cols].sort_values(["window_ms", "model", "feat_set"]).to_csv(summary_csv, index=False)
    print(f"[save] {summary_csv} ({len(out)} rows)")

    # ext vs base overhead within same (window_ms, model)
    overhead_rows = []
    for (w, m), g in out.groupby(["window_ms", "model"]):
        g2 = g.set_index("feat_set", drop=False)
        if "base" in g2.index and "ext" in g2.index:
            base = g2.loc["base"]
            ext = g2.loc["ext"]
            infer_base = float(base["infer_time_per_window_ms"])
            infer_ext = float(ext["infer_time_per_window_ms"])
            fit_base = float(base["fit_time_mean_sec"])
            fit_ext = float(ext["fit_time_mean_sec"])
            overhead_rows.append(
                {
                    "window_ms": int(w),
                    "model": str(m),
                    "infer_ms_base": infer_base,
                    "infer_ms_ext": infer_ext,
                    "infer_overhead_pct": (infer_ext - infer_base) / infer_base * 100.0 if infer_base != 0 else None,
                    "fit_sec_base": fit_base,
                    "fit_sec_ext": fit_ext,
                    "fit_overhead_pct": (fit_ext - fit_base) / fit_base * 100.0 if fit_base != 0 else None,
                    "f1_base": float(base["f1_macro_mean"]),
                    "f1_ext": float(ext["f1_macro_mean"]),
                    "f1_delta": float(ext["f1_macro_mean"]) - float(base["f1_macro_mean"]),
                    "bal_acc_base": float(base["bal_acc_mean"]),
                    "bal_acc_ext": float(ext["bal_acc_mean"]),
                    "bal_acc_delta": float(ext["bal_acc_mean"]) - float(base["bal_acc_mean"]),
                }
            )

    if overhead_rows:
        overhead_df = pd.DataFrame(overhead_rows).sort_values(["window_ms", "model"])
        overhead_csv = out_dir / "compute_overhead_ext_vs_base.csv"
        overhead_df.to_csv(overhead_csv, index=False)
        print(f"[save] {overhead_csv} ({len(overhead_df)} rows)")
    else:
        print("[info] No base/ext pairs found for the same (window_ms, model). Skipping overhead table.")

    if args.make_plots:
        plot_df = out.sort_values(["model", "feat_set"]).copy()
        plot_df["label"] = plot_df["model"].astype(str) + " | " + plot_df["feat_set"].astype(str)

        _save_bar(
            plot_df,
            x_col="label",
            y_col="fit_time_mean_sec",
            title=f"Fit time per fold (mean){' w'+str(args.window_ms)+'ms' if args.window_ms else ''}",
            out_png=out_dir / "fit_time_mean_sec.png",
        )

        _save_bar(
            plot_df,
            x_col="label",
            y_col="infer_time_per_window_ms",
            title=f"Inference latency per window{' w'+str(args.window_ms)+'ms' if args.window_ms else ''}",
            out_png=out_dir / "infer_time_per_window_ms.png",
        )

        if plot_df["f1_per_ms"].notna().any():
            _save_bar(
                plot_df.fillna({"f1_per_ms": 0.0}),
                x_col="label",
                y_col="f1_per_ms",
                title=f"Macro-F1 per ms latency proxy{' w'+str(args.window_ms)+'ms' if args.window_ms else ''}",
                out_png=out_dir / "f1_per_ms.png",
            )


if __name__ == "__main__":
    main()

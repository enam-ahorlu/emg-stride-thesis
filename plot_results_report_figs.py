# plot_results_report_figs.py
# Report-ready plots comparing CNN vs classical models using per-subject metrics CSVs,
# plus optional "early master" summary plots from master_classical_results.csv.
#
# Inputs (defaults match your filenames):
# - CNN per-subject CSV: columns like: subject, acc, bal_acc, f1 (train_time_s/latency_ms optional)
# - Classical per-subject CSV: columns like: subject, model, acc, bal_acc, f1_macro (or f1)
# - Master classical CSV (optional): master_classical_results.csv (for early bar plots)
#
# Outputs:
# - PNG figures in outdir
# - summary_mean_sd.csv (mean±sd per model)
# - combined_per_subject_cnn_vs_classical.csv (stacked per-subject rows)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Helpers
# -----------------------------

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from candidates (case-insensitive), else None."""
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _coerce_subject_series(s: pd.Series) -> pd.Series:
    """Best effort: keep subject values but prefer int-like when possible."""
    try:
        return s.astype(int)
    except Exception:
        return s.astype(str)


def _mean_sd(series: pd.Series) -> Dict[str, float]:
    s = series.dropna().astype(float)
    if len(s) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "n": int(len(s)),
    }


def _savefig(outpath: Path, dpi: int = 220) -> None:
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def _boxplot(
    data_by_label: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    outpath: Path,
    dpi: int,
    show_points: bool = True,
):
    labels = list(data_by_label.keys())
    data = [np.asarray(data_by_label[k], dtype=float) for k in labels]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    bp = ax.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    # No fixed colors; just add transparency so it prints nicely.
    for patch in bp["boxes"]:
        patch.set_alpha(0.25)

    if show_points:
        rng = np.random.default_rng(42)
        for i, y in enumerate(data, start=1):
            x = rng.normal(loc=i, scale=0.04, size=len(y))
            ax.scatter(x, y, s=18, alpha=0.55)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    _savefig(outpath, dpi=dpi)


def _mean_sd_bar(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    outpath: Path,
    dpi: int,
    model_order: Optional[List[str]] = None,
):
    d = df.dropna(subset=[metric]).copy()
    if len(d) == 0:
        return

    stats = (
        d.groupby("model")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Order bars
    if model_order:
        order_map = {m: i for i, m in enumerate(model_order)}
        stats["__ord"] = stats["model"].map(lambda m: order_map.get(m, 999))
        stats = stats.sort_values(["__ord", "mean"], ascending=[True, False]).drop(columns="__ord")
    else:
        stats = stats.sort_values("mean", ascending=False)

    plt.figure(figsize=(10, 4.8))
    plt.bar(stats["model"], stats["mean"].astype(float), yerr=stats["std"].astype(float))
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    _savefig(outpath, dpi=dpi)


def _lineplot_by_subject(
    subjects: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    outpath: Path,
    dpi: int,
):
    # Sort by subject id if int-like
    try:
        order = np.argsort(subjects.astype(int))
    except Exception:
        order = np.argsort(subjects.astype(str))

    s = subjects[order]
    v = values[order].astype(float)

    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(111)

    ax.plot(s, v, marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    if len(s) > 15:
        ax.tick_params(axis="x", labelrotation=45)

    _savefig(outpath, dpi=dpi)


def _bar_early_master(master: pd.DataFrame, outdir: Path, dpi: int, window_ms: int, wanted_models: List[str]) -> None:
    dfm = master.copy()

    if "window_ms" in dfm.columns:
        dfm = dfm[dfm["window_ms"] == int(window_ms)].copy()

    if "model" in dfm.columns:
        dfm = dfm[dfm["model"].isin(wanted_models)].copy()

    if len(dfm) == 0:
        print("[warn] Master filtering produced 0 rows; skipping early master plots.")
        return

    dfm["feat_set"] = dfm.get("feat_set", pd.Series(["unknown"] * len(dfm))).fillna("unknown")
    dfm["label"] = dfm["model"].astype(str) + " | " + dfm["feat_set"].astype(str)

    def _bar(df: pd.DataFrame, col: str, title: str, fname: str):
        if col not in df.columns:
            return
        d = df.sort_values(["model", "feat_set"]).copy()
        plt.figure(figsize=(10, 4.8))
        plt.bar(d["label"].astype(str), d[col].astype(float))
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(col)
        plt.title(title)
        _savefig(outdir / fname, dpi=dpi)

    _bar(dfm, "f1_macro_mean", f"Classical Macro-F1 mean (master, window={window_ms}ms)", f"early_master_f1_macro_mean_w{window_ms}.png")
    _bar(dfm, "bal_acc_mean", f"Classical Balanced Acc mean (master, window={window_ms}ms)", f"early_master_bal_acc_mean_w{window_ms}.png")
    _bar(dfm, "acc_mean", f"Classical Accuracy mean (master, window={window_ms}ms)", f"early_master_acc_mean_w{window_ms}.png")


# -----------------------------
# Loaders (schema normalization)
# -----------------------------

def load_cnn(cnn_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cnn_csv)

    subj = _find_col(df, ["subject", "Subject", "subj", "subject_id"])
    if subj is None:
        raise ValueError(f"CNN CSV missing a subject column. Columns: {list(df.columns)}")

    acc = _find_col(df, ["acc", "accuracy"])
    bal = _find_col(df, ["bal_acc", "balanced_acc", "balanced_accuracy", "bal"])
    f1 = _find_col(df, ["f1", "f1_macro", "macro_f1", "f1-macro"])

    if bal is None or f1 is None:
        raise ValueError(
            "CNN CSV must have balanced accuracy and macro-F1 columns.\n"
            f"Columns: {list(df.columns)}"
        )

    train_time = _find_col(df, ["train_time_s", "train_time", "time_s"])
    latency = _find_col(df, ["latency_ms", "inference_ms", "ms_per_window"])

    out = pd.DataFrame({
        "subject": _coerce_subject_series(df[subj]),
        "model": "CNN (env+zscore)",
        "acc": df[acc] if acc else np.nan,
        "bal_acc": df[bal],
        "f1": df[f1],
        "train_time_s": df[train_time] if train_time else np.nan,
        "latency_ms": df[latency] if latency else np.nan,
    })
    return out


def load_classical(per_subject_csv: Path, keep_models: List[str]) -> pd.DataFrame:
    df = pd.read_csv(per_subject_csv)

    subj = _find_col(df, ["subject", "Subject", "subj", "subject_id"])
    model = _find_col(df, ["model", "Model", "clf", "classifier"])
    bal = _find_col(df, ["bal_acc", "balanced_acc", "balanced_accuracy", "bal"])
    f1 = _find_col(df, ["f1", "f1_macro", "macro_f1", "f1-macro", "f1_macro_mean"])
    acc = _find_col(df, ["acc", "accuracy"])

    missing = [k for k, v in [("subject", subj), ("model", model), ("bal_acc", bal), ("f1", f1)] if v is None]
    if missing:
        raise ValueError(
            f"Classical per-subject CSV missing columns: {missing}\nColumns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "subject": _coerce_subject_series(df[subj]),
        "model": df[model].astype(str),
        "acc": df[acc] if acc else np.nan,
        "bal_acc": df[bal],
        "f1": df[f1],
    })

    # Filter models (case-insensitive)
    keep_lower = {m.lower() for m in keep_models}
    out = out[out["model"].str.lower().isin(keep_lower)].copy()

    # Optional cols
    tt = _find_col(df, ["train_time_s", "train_time", "time_s"])
    lat = _find_col(df, ["latency_ms", "inference_ms", "ms_per_window"])
    out["train_time_s"] = df[tt] if tt else np.nan
    out["latency_ms"] = df[lat] if lat else np.nan

    # Normalize display names
    rename_map = {
        "svm": "SVM",
        "svm_rbf": "SVM",
        "svm_rbf_balanced_scaled": "SVM",
        "rf": "RF",
        "rf_balanced": "RF",
        "random_forest": "RF",
        "lda": "LDA",
        "lda_scaled": "LDA",
    }
    out["model"] = out["model"].apply(lambda s: rename_map.get(s.lower(), s))
    return out


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Generate report figures: CNN vs classical comparisons (per-subject), plus optional early master bars.")
    ap.add_argument("--cnn-csv", default="results_cnn/cnn_subjectdep_w250_env_zscore.csv")
    ap.add_argument("--classical-csv", default="results_classical/per_subject_metrics_250_base.csv")
    ap.add_argument("--master", default="results_classical/master_classical_results.csv", help="Optional master file for early plots (skip if missing).")
    ap.add_argument("--window-ms", type=int, default=250, help="Window length filter for MASTER plots (if used).")
    ap.add_argument("--outdir", default="report_figs")
    ap.add_argument("--models", default="SVM_RBF_balanced_scaled,RF_balanced,LDA_scaled", help="Comma-separated classical models to include (as they appear in the CSV).")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--no-points", action="store_true", help="Disable scatter points on boxplots.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    # Load
    df_cnn = load_cnn(Path(args.cnn_csv))
    keep_models = [m.strip() for m in args.models.split(",") if m.strip()]
    df_cl = load_classical(Path(args.classical_csv), keep_models=keep_models)

    # Combined per-subject table (stacked)
    df_all = pd.concat([df_cnn, df_cl], ignore_index=True)
    df_all.to_csv(outdir / "combined_per_subject_cnn_vs_classical.csv", index=False)

    # Preferred order for plots/tables
    model_order = ["CNN (env+zscore)", "SVM", "RF", "LDA"]
    for m in df_all["model"].unique():
        if m not in model_order:
            model_order.append(m)

    # -----------------------------
    # Summary mean ± sd (across subjects)
    # -----------------------------
    rows = []
    for m in model_order:
        sub = df_all[df_all["model"] == m].copy()
        if len(sub) == 0:
            continue
        rows.append({
            "Model": m,
            "Accuracy_mean": _mean_sd(sub["acc"])["mean"],
            "Accuracy_sd": _mean_sd(sub["acc"])["sd"],
            "Accuracy_n": _mean_sd(sub["acc"])["n"],
            "BalancedAcc_mean": _mean_sd(sub["bal_acc"])["mean"],
            "BalancedAcc_sd": _mean_sd(sub["bal_acc"])["sd"],
            "BalancedAcc_n": _mean_sd(sub["bal_acc"])["n"],
            "MacroF1_mean": _mean_sd(sub["f1"])["mean"],
            "MacroF1_sd": _mean_sd(sub["f1"])["sd"],
            "MacroF1_n": _mean_sd(sub["f1"])["n"],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary_mean_sd.csv", index=False)

    # -----------------------------
    # Boxplots across subjects
    # -----------------------------
    show_points = (not args.no_points)

    def _data_by_model(metric: str) -> Dict[str, np.ndarray]:
        out = {}
        for m in model_order:
            vals = df_all.loc[df_all["model"] == m, metric].dropna().astype(float).to_numpy()
            if len(vals) > 0:
                out[m] = vals
        return out

    # Accuracy (now supported for classical too, if present)
    acc_data = _data_by_model("acc")
    if len(acc_data) > 0:
        _boxplot(
            acc_data,
            title="Accuracy across subjects (CNN vs Classical)",
            ylabel="Accuracy",
            outpath=outdir / "acc_boxplot.png",
            dpi=args.dpi,
            show_points=show_points,
        )
        _mean_sd_bar(
            df_all, "acc",
            title="Accuracy mean ± SD across subjects (CNN vs Classical)",
            ylabel="Accuracy (mean ± SD)",
            outpath=outdir / "acc_mean_sd_bar.png",
            dpi=args.dpi,
            model_order=model_order,
        )

    _boxplot(
        _data_by_model("bal_acc"),
        title="Balanced Accuracy across subjects (CNN vs Classical)",
        ylabel="Balanced Accuracy",
        outpath=outdir / "bal_acc_boxplot.png",
        dpi=args.dpi,
        show_points=show_points,
    )
    _mean_sd_bar(
        df_all, "bal_acc",
        title="Balanced Accuracy mean ± SD across subjects (CNN vs Classical)",
        ylabel="Balanced Accuracy (mean ± SD)",
        outpath=outdir / "bal_acc_mean_sd_bar.png",
        dpi=args.dpi,
        model_order=model_order,
    )

    _boxplot(
        _data_by_model("f1"),
        title="Macro-F1 across subjects (CNN vs Classical)",
        ylabel="Macro-F1",
        outpath=outdir / "macro_f1_boxplot.png",
        dpi=args.dpi,
        show_points=show_points,
    )
    _mean_sd_bar(
        df_all, "f1",
        title="Macro-F1 mean ± SD across subjects (CNN vs Classical)",
        ylabel="Macro-F1 (mean ± SD)",
        outpath=outdir / "macro_f1_mean_sd_bar.png",
        dpi=args.dpi,
        model_order=model_order,
    )

    # -----------------------------
    # CNN per-subject lines (metrics + optional runtime/latency)
    # -----------------------------
    cnn_only = df_all[df_all["model"] == "CNN (env+zscore)"].copy()
    if len(cnn_only) > 0:
        _lineplot_by_subject(
            subjects=cnn_only["subject"].to_numpy(),
            values=cnn_only["f1"].to_numpy(),
            title="CNN Macro-F1 by subject",
            ylabel="Macro-F1",
            outpath=outdir / "cnn_f1_by_subject.png",
            dpi=args.dpi,
        )
        _lineplot_by_subject(
            subjects=cnn_only["subject"].to_numpy(),
            values=cnn_only["bal_acc"].to_numpy(),
            title="CNN Balanced Accuracy by subject",
            ylabel="Balanced Accuracy",
            outpath=outdir / "cnn_bal_acc_by_subject.png",
            dpi=args.dpi,
        )
        if "acc" in cnn_only.columns and cnn_only["acc"].notna().any():
            _lineplot_by_subject(
                subjects=cnn_only["subject"].to_numpy(),
                values=cnn_only["acc"].to_numpy(),
                title="CNN Accuracy by subject",
                ylabel="Accuracy",
                outpath=outdir / "cnn_acc_by_subject.png",
                dpi=args.dpi,
            )
        if "train_time_s" in cnn_only.columns and cnn_only["train_time_s"].notna().any():
            _lineplot_by_subject(
                subjects=cnn_only["subject"].to_numpy(),
                values=cnn_only["train_time_s"].to_numpy(),
                title="CNN train time by subject (seconds)",
                ylabel="Train time (s)",
                outpath=outdir / "cnn_train_time_by_subject.png",
                dpi=args.dpi,
            )
        if "latency_ms" in cnn_only.columns and cnn_only["latency_ms"].notna().any():
            _lineplot_by_subject(
                subjects=cnn_only["subject"].to_numpy(),
                values=cnn_only["latency_ms"].to_numpy(),
                title="CNN inference latency by subject (ms/window)",
                ylabel="Latency (ms/window)",
                outpath=outdir / "cnn_latency_by_subject.png",
                dpi=args.dpi,
            )

    # -----------------------------
    # CNN vs best classical per subject (Macro-F1)
    # -----------------------------
    cl_only = df_all[df_all["model"] != "CNN (env+zscore)"].copy()
    if len(cnn_only) > 0 and len(cl_only) > 0:
        # For each subject: pick classical row with best macro-F1
        # (idxmax requires numeric)
        cl_only["f1"] = cl_only["f1"].astype(float)
        best_idx = cl_only.groupby("subject")["f1"].idxmax()
        cl_best = cl_only.loc[best_idx, ["subject", "model", "f1"]].rename(columns={"model": "best_classical_model", "f1": "best_classical_f1"})

        merged = pd.merge(
            cnn_only[["subject", "f1"]].rename(columns={"f1": "cnn_f1"}),
            cl_best,
            on="subject",
            how="inner",
        )

        subs = merged["subject"].to_numpy()
        try:
            ord_idx = np.argsort(subs.astype(int))
        except Exception:
            ord_idx = np.argsort(subs.astype(str))

        fig = plt.figure(figsize=(12, 4.8))
        ax = fig.add_subplot(111)
        ax.plot(merged["subject"].to_numpy()[ord_idx], merged["cnn_f1"].to_numpy()[ord_idx], marker="o", linewidth=1.5, label="CNN")
        ax.plot(merged["subject"].to_numpy()[ord_idx], merged["best_classical_f1"].to_numpy()[ord_idx], marker="o", linewidth=1.5, label="Best classical (by Macro-F1)")
        ax.set_title("Per-subject Macro-F1: CNN vs best classical model")
        ax.set_xlabel("Subject")
        ax.set_ylabel("Macro-F1")
        ax.grid(True, alpha=0.25)
        if len(ord_idx) > 15:
            ax.tick_params(axis="x", labelrotation=45)
        ax.legend()
        _savefig(outdir / "cnn_vs_best_classical_f1.png", dpi=args.dpi)

        merged.to_csv(outdir / "cnn_vs_best_classical_per_subject.csv", index=False)

    # -----------------------------
    # Optional: Early master plots (if master file exists)
    # -----------------------------
    master_path = Path(args.master)
    if master_path.exists():
        master = pd.read_csv(master_path)
        _bar_early_master(
            master=master,
            outdir=outdir,
            dpi=args.dpi,
            window_ms=args.window_ms,
            wanted_models=["SVM_RBF_balanced_scaled", "RF_balanced", "LDA_scaled"],
        )
    else:
        print("[info] Master file not found; skipping early master plots.")

    print(f"[OK] Saved figures + summary to: {outdir.resolve()}")
    print(f"[OK] Summary CSV: {outdir / 'summary_mean_sd.csv'}")
    print(f"[OK] Combined per-subject CSV: {outdir / 'combined_per_subject_cnn_vs_classical.csv'}")


if __name__ == "__main__":
    main()

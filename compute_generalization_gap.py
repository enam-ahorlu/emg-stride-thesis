# compute_generalization_gap.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import pandas as pd


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _require(df: pd.DataFrame, col: Optional[str], name: str, file_tag: str) -> str:
    if col is None:
        raise ValueError(f"{file_tag}: missing required column for {name}. Have columns: {list(df.columns)}")
    return col


def _norm_model(s: pd.Series) -> pd.Series:
    # normalize common naming differences so merges don't fail
    def norm_one(x: str) -> str:
        t = str(x).strip()
        low = t.lower()
        if "svm" in low:
            return "SVM"
        if low in {"rf", "random_forest", "randomforest"} or "rf" in low:
            return "RF"
        if "lda" in low:
            return "LDA"
        return t
    return s.astype(str).map(norm_one)


def main():
    ap = argparse.ArgumentParser(description="Compute SD vs LOSO generalization gap per subject (ΔF1 = F1_SD − F1_LOSO).")
    ap.add_argument("--sd", default="results_classical/per_subject_metrics_250_base.csv", help="Subject-dependent per-subject metrics CSV")
    ap.add_argument("--loso", default="results_loso_light/per_subject_metrics_250_base_loso.csv", help="LOSO per-subject metrics CSV")
    ap.add_argument("--out", default="results_loso_light/generalization_gap.csv", help="Output CSV path")
    ap.add_argument("--models", default="", help="Optional comma-separated model filter (e.g., SVM,RF,LDA)")
    args = ap.parse_args()

    sd_path = Path(args.sd)
    loso_path = Path(args.loso)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sd = pd.read_csv(sd_path)
    lo = pd.read_csv(loso_path)

    # Detect columns
    sd_subj = _require(sd, _find_col(sd, ["subject", "subject_id", "subj"]), "subject", "SD")
    sd_model = _require(sd, _find_col(sd, ["model", "classifier", "clf"]), "model", "SD")
    sd_f1 = _require(sd, _find_col(sd, ["f1_macro", "f1", "macro_f1", "f1-macro"]), "f1", "SD")
    sd_bal = _require(sd, _find_col(sd, ["bal_acc", "balanced_accuracy", "balanced_acc"]), "bal_acc", "SD")
    sd_acc = _find_col(sd, ["acc", "accuracy"])

    lo_subj = _require(lo, _find_col(lo, ["subject", "subject_id", "subj"]), "subject", "LOSO")
    lo_model = _require(lo, _find_col(lo, ["model", "classifier", "clf"]), "model", "LOSO")
    lo_f1 = _require(lo, _find_col(lo, ["f1_macro", "f1", "macro_f1", "f1-macro"]), "f1", "LOSO")
    lo_bal = _require(lo, _find_col(lo, ["bal_acc", "balanced_accuracy", "balanced_acc"]), "bal_acc", "LOSO")
    lo_acc = _find_col(lo, ["acc", "accuracy"])

    # Minimal normalized frames
    sd2 = pd.DataFrame({
        "subject": sd[sd_subj].astype(int),
        "model": _norm_model(sd[sd_model]),
        "f1_sd": sd[sd_f1].astype(float),
        "bal_acc_sd": sd[sd_bal].astype(float),
        "acc_sd": sd[sd_acc].astype(float) if sd_acc else pd.NA,
    })

    lo2 = pd.DataFrame({
        "subject": lo[lo_subj].astype(int),
        "model": _norm_model(lo[lo_model]),
        "f1_loso": lo[lo_f1].astype(float),
        "bal_acc_loso": lo[lo_bal].astype(float),
        "acc_loso": lo[lo_acc].astype(float) if lo_acc else pd.NA,
    })

    # Optional model filter
    if args.models.strip():
        keep = {m.strip().upper() for m in args.models.split(",") if m.strip()}
        sd2 = sd2[sd2["model"].str.upper().isin(keep)].copy()
        lo2 = lo2[lo2["model"].str.upper().isin(keep)].copy()

    # Merge & compute gaps
    merged = sd2.merge(lo2, on=["subject", "model"], how="inner")

    merged["delta_f1"] = merged["f1_sd"] - merged["f1_loso"]
    merged["delta_bal_acc"] = merged["bal_acc_sd"] - merged["bal_acc_loso"]

    if "acc_sd" in merged.columns and "acc_loso" in merged.columns:
        # may contain <NA>
        merged["delta_acc"] = pd.to_numeric(merged["acc_sd"], errors="coerce") - pd.to_numeric(merged["acc_loso"], errors="coerce")

    merged = merged.sort_values(["model", "subject"]).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"[save] {out_path} ({len(merged)} rows)")

    # Also save quick summary
    summary = (
        merged.groupby("model")[["f1_sd", "f1_loso", "delta_f1", "bal_acc_sd", "bal_acc_loso", "delta_bal_acc"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[save] {summary_path}")


if __name__ == "__main__":
    main()

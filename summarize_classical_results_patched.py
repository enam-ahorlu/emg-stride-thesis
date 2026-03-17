# summarize_classical_results.py
from __future__ import annotations

import re
import argparse
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

def parse_tags_from_stem(stem: str) -> dict:
    tags = {}

    m_w = re.search(r"_w(150|250)_", stem)
    tags["window_ms"] = int(m_w.group(1)) if m_w else None

    m_feat = re.search(r"_features_(base|ext)(?:_|$)", stem)
    tags["feat_set"] = m_feat.group(1) if m_feat else None

    tags["stem"] = stem
    return tags


def main():
    ap = argparse.ArgumentParser(description="Merge classical baseline CSV outputs into one master table.")
    ap.add_argument("--results-dir", type=str, default="results_classical",
                    help="Directory containing *_subjdep_cv.csv outputs.")
    ap.add_argument("--out", type=str, default="master_classical_results.csv",
                    help="Output master CSV filename (saved inside results-dir).")
    ap.add_argument("--xlsx", action="store_true", help="Also write an Excel version.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    files = sorted(results_dir.glob("*_subjdep_cv.csv"))
    if not files:
        raise FileNotFoundError(f"No *_subjdep_cv.csv files found in {results_dir}")

    frames = []
    for f in files:
        try:
            if f.stat().st_size == 0:
                print(f"[skip] empty file: {f}")
                continue
            df = pd.read_csv(f)
        except EmptyDataError:
            print(f"[skip] unreadable/empty csv: {f}")
            continue

        tags = parse_tags_from_stem(f.stem)
        df.insert(0, "source_file", f.name)
        df.insert(1, "window_ms", tags["window_ms"])
        df.insert(2, "feat_set", tags["feat_set"])
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)

    preferred = [
        "source_file", "window_ms", "feat_set", "model",
        "n_samples", "n_features", "subjects", "cv_splits",
        "acc_mean", "bal_acc_mean", "f1_macro_mean",
        "acc_std", "bal_acc_std", "f1_macro_std",
        "fit_time_mean_sec", "pred_time_mean_sec", "infer_time_per_window_ms",
    ]
    cols = [c for c in preferred if c in master.columns] + [c for c in master.columns if c not in preferred]
    master = master[cols]

    # Add 'notes' column: tag ablation runs vs canonical runs.
    # "SVM_RBF_balanced" (without _scaled suffix) = unscaled SVM = ablation.
    # All other rows = canonical experiment.
    if "model" in master.columns:
        master = master.copy()
        master["notes"] = master["model"].apply(
            lambda m: "ablation_no_scaling" if str(m).strip() == "SVM_RBF_balanced" else "canonical"
        )

    out_csv = results_dir / args.out
    master.to_csv(out_csv, index=False)
    print(f"[save] master CSV: {out_csv} ({len(master)} rows)")

    if args.xlsx:
        out_xlsx = out_csv.with_suffix(".xlsx")
        master.to_excel(out_xlsx, index=False)
        print(f"[save] master XLSX: {out_xlsx}")

    if "bal_acc_mean" in master.columns and "f1_macro_mean" in master.columns:
        best = master.sort_values(["bal_acc_mean", "f1_macro_mean"], ascending=False).head(1)
        print("\n==== BEST (by bal_acc_mean then f1_macro_mean) ====")
        print(best.to_string(index=False))


if __name__ == "__main__":
    main()

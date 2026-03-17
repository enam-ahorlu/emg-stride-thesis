# analyze_subject_metadata.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_sub_id(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Normalize subject identifiers to a common form:
    - If values look like 'Sub01' -> keep them.
    - If values look like 1..40 -> convert to 'Sub01'..'Sub40'.
    """
    s = df[col].astype(str).str.strip()

    # If already SubXX format
    if s.str.match(r"^Sub\d+$").any():
        # standardize zero-padding if needed (Sub1 -> Sub01)
        s = s.str.replace(r"^Sub(\d)$", r"Sub0\1", regex=True)
        s = s.str.replace(r"^Sub(\d{2})$", r"Sub\1", regex=True)
        return s

    # If integers like 1..40
    if s.str.match(r"^\d+$").all():
        ints = s.astype(int)
        return ints.map(lambda x: f"Sub{x:02d}")

    # Fallback: return as-is
    return s


def scatter_with_fit(df, x, y, outpath, title):

    AXIS_LABELS = {
    "height": "Height (mm)",
    "weight": "Weight (kg)",
    "age": "Age (years)"
    }
    xvals = pd.to_numeric(df[x], errors="coerce").values
    yvals = pd.to_numeric(df[y], errors="coerce").values

    plt.figure(figsize=(6.8, 4.6))
    plt.scatter(xvals, yvals)

    mask = np.isfinite(xvals) & np.isfinite(yvals)
    if mask.sum() >= 2:
        a, b = np.polyfit(xvals[mask], yvals[mask], 1)
        xs = np.linspace(xvals[mask].min(), xvals[mask].max(), 100)
        plt.plot(xs, a * xs + b)

        # correlation
        r = np.corrcoef(xvals[mask], yvals[mask])[0, 1]
        plt.title(f"{title}\nPearson r={r:.3f}")
    else:
        plt.title(title)

    plt.xlabel(AXIS_LABELS.get(x, x))
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def boxplot(df, x, y, outpath, title):
    cats = [c for c in df[x].dropna().unique().tolist()]
    data = [pd.to_numeric(df[df[x] == c][y], errors="coerce").dropna().values for c in cats]

    plt.figure(figsize=(6.8, 4.6))
    plt.boxplot(data, tick_labels=cats, showfliers=False)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loso-metrics", required=True, help="per_subject_metrics_250_base_loso.csv")
    ap.add_argument("--subject-info", required=True, help="SubjectInformation.xlsx")
    ap.add_argument("--model", default="SVM", help="SVM or RF")
    ap.add_argument("--outdir", default="results_loso_light/error_analysis/metadata", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load LOSO metrics
    loso = pd.read_csv(args.loso_metrics)
    loso = loso[loso["model"] == args.model].copy()

    # normalize subject id into SubXX
    loso["sub_id"] = ensure_sub_id(loso, "subject")

    # mean metrics per subject
    # (your file likely contains f1_macro + bal_acc, plus n_windows)
    subj = loso.groupby("sub_id", as_index=False).agg(
        f1_loso=("f1_macro", "mean"),
        bal_acc_loso=("bal_acc", "mean"),
        n_windows=("n_windows", "sum") if "n_windows" in loso.columns else ("f1_macro", "size"),
    )

    # ---- load SubjectInformation.xlsx
    info = pd.read_excel(args.subject_info)

    # normalize subject id into SubXX
    if "Subject" not in info.columns:
        raise ValueError("Expected 'Subject' column in SubjectInformation.xlsx")
    info["sub_id"] = ensure_sub_id(info, "Subject")

    # ---- merge
    df = subj.merge(info, on="sub_id", how="left")
    df.to_csv(outdir / "subject_metadata_merged.csv", index=False)

    # ---- plots (use your exact column names)
    numeric_cols = [
        "age", "weight", "height",
        "AVE FCH-ICT", "AVE KNE-FCH", "AVE ANK-KNE", "AVE PM6-FCC", "AVE FLE-FME", "AVE FAL-TAM"
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        scatter_with_fit(
            df, col, "f1_loso",
            outdir / f"f1_loso_vs_{col.replace(' ', '_')}.png",
            f"{args.model}: LOSO F1 vs {col}"
        )
        scatter_with_fit(
            df, col, "bal_acc_loso",
            outdir / f"balacc_loso_vs_{col.replace(' ', '_')}.png",
            f"{args.model}: LOSO Balanced Acc vs {col}"
        )

    # sex boxplot
    if "sex" in df.columns:
        boxplot(df, "sex", "f1_loso", outdir / "f1_loso_by_sex.png", f"{args.model}: LOSO F1 by sex")
        boxplot(df, "sex", "bal_acc_loso", outdir / "balacc_loso_by_sex.png", f"{args.model}: LOSO BalAcc by sex")

    # correlations table
    corr_cols = ["f1_loso", "bal_acc_loso"] + numeric_cols
    corr = df[corr_cols].corr(numeric_only=True)
    corr.to_csv(outdir / "correlations.csv")

    print(f"[done] Saved merged table + plots to: {outdir}")


if __name__ == "__main__":
    main()
# dataset_balance_analysis.py
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


LABELS = ["1", "2", "3", "4", "5", "A", "R"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="windows_..._meta.csv (must include subject + status_mode)")
    ap.add_argument("--outdir", default="results_loso_light/error_analysis/dataset", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta)

    # Basic checks
    if "subject" not in meta.columns:
        raise ValueError("meta CSV must contain a 'subject' column")
    if "status_mode" not in meta.columns:
        raise ValueError("meta CSV must contain a 'status_mode' column")

    meta["subject"] = meta["subject"].astype(int)
    meta["status_mode"] = meta["status_mode"].astype(str)

    # ---- windows per subject
    per_subj = meta.groupby("subject").size().reset_index(name="n_windows")
    per_subj.to_csv(outdir / "windows_per_subject.csv", index=False)

    plt.figure(figsize=(10, 4.5))
    plt.bar(per_subj["subject"].astype(str), per_subj["n_windows"].astype(int))
    plt.title("Windows per Subject")
    plt.xlabel("Subject")
    plt.ylabel("Number of windows")
    plt.tight_layout()
    plt.savefig(outdir / "windows_per_subject.png", dpi=200)
    plt.close()

    # ---- class distribution by subject (stacked proportions)
    pivot = (
        meta.pivot_table(index="subject", columns="status_mode", aggfunc="size", fill_value=0)
        .reindex(columns=LABELS, fill_value=0)
    )
    pivot.to_csv(outdir / "class_counts_by_subject.csv")

    # convert to proportions per subject for easier comparison
    prop = pivot.div(pivot.sum(axis=1), axis=0)
    prop.to_csv(outdir / "class_proportions_by_subject.csv")

    plt.figure(figsize=(10, 5))
    bottom = None
    subjects = prop.index.astype(str)

    for lab in LABELS:
        vals = prop[lab].values
        if bottom is None:
            plt.bar(subjects, vals, label=f"Class {lab}")
            bottom = vals
        else:
            plt.bar(subjects, vals, bottom=bottom, label=f"Class {lab}")
            bottom = bottom + vals

    plt.title("Class Distribution by Subject (Proportions)")
    plt.xlabel("Subject")
    plt.ylabel("Proportion of windows")
    plt.legend(title="Class", bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outdir / "class_distribution_by_subject.png", dpi=200)
    plt.close()

    print(f"[done] Dataset analysis saved to: {outdir}")


if __name__ == "__main__":
    main()
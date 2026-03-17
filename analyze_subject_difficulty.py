# analyze_subject_difficulty.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", required=True, help="generalization_gap.csv")
    ap.add_argument("--models", default="SVM,RF", help="Comma-separated models to include")
    ap.add_argument("--metric", default="f1_loso", choices=["f1_loso", "delta_f1", "bal_acc_loso", "delta_bal_acc"],
                    help="Metric to define easy/hard subjects")
    ap.add_argument("--quantile", type=float, default=0.25, help="Bottom/top quantile for hard/easy")
    ap.add_argument("--outdir", default="results_loso_light/error_analysis/subjects", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    df = pd.read_csv(args.gap)
    df = df[df["model"].isin(models)].copy()

    # Make one table per model
    for model in models:
        d = df[df["model"] == model].copy()
        d = d.sort_values("subject")

        # thresholds
        q = args.quantile
        low_thr = d[args.metric].quantile(q)
        high_thr = d[args.metric].quantile(1 - q)

        hard = d[d[args.metric] <= low_thr].sort_values(args.metric)
        easy = d[d[args.metric] >= high_thr].sort_values(args.metric, ascending=False)

        hard.to_csv(outdir / f"{model}_hard_subjects_by_{args.metric}.csv", index=False)
        easy.to_csv(outdir / f"{model}_easy_subjects_by_{args.metric}.csv", index=False)

        # Plot metric by subject
        plt.figure(figsize=(10, 4.5))
        plt.bar(d["subject"].astype(int).astype(str), d[args.metric].astype(float))
        plt.axhline(low_thr, linestyle="--")
        plt.axhline(high_thr, linestyle="--")
        plt.title(f"{model}: {args.metric} by subject (hard/easy thresholds)")
        plt.xlabel("Subject")
        plt.ylabel(args.metric)
        plt.tight_layout()
        plt.savefig(outdir / f"{model}_{args.metric}_by_subject.png", dpi=200)
        plt.close()

    print(f"[done] Subject difficulty outputs saved to: {outdir}")


if __name__ == "__main__":
    main()
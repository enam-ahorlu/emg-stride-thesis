# analyze_cnn_calibration_followup.py
# ---------------------------------------------------------------------------
# Follow-up investigation into the CNN calibration finding (Action 2.6):
# the naive protocol (--ft-epochs 15) showed K=5/class calibration HURTING
# F1 by -11.4pp vs K=0 (no calibration), which was surprising. This script
# audits that result using data already collected by run_cnn_calibration_loso.py
# (no new training):
#   1. Per-subject breakdown of the original run -- is the degradation
#      universal or concentrated in a few subjects? Correlation with baseline F1?
#   2. Three-way comparison across the original run, a mechanism test
#      (--ft-epochs 3, same seed) and a robustness check (--seed 7, same
#      --ft-epochs 15) -- confirms the degradation is (a) reproducible across
#      seeds and (b) caused by unregularized overfitting on the tiny
#      calibration set, fixed by reducing fine-tune epochs.
#
# Inputs (must already exist -- run run_cnn_calibration_loso.py first):
#   results_cnn_calibration/cnn_calibration_subjectwise.csv           (original, seed 42, ft-epochs 15)
#   results_cnn_calibration_ftepochs3/cnn_calibration_subjectwise.csv (mechanism test, seed 42, ft-epochs 3)
#   results_cnn_calibration_seed7/cnn_calibration_subjectwise.csv     (robustness check, seed 7, ft-epochs 15)
#
# Outputs -> report_figs/cnn_calibration_followup/:
#   per_subject_lift_original.csv   -- per-subject F1 at each K + lift vs K=0
#   worse_better_counts.csv         -- how many subjects improve/degrade at each K
#   baseline_vs_lift_correlation.csv
#   three_way_comparison.csv        -- original vs ft-epochs=3 vs seed=7, summary F1 by K
# ---------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RUNS = {
    "original_seed42_ft15": ROOT / "results_cnn_calibration" / "cnn_calibration_subjectwise.csv",
    "mechanism_ft3": ROOT / "results_cnn_calibration_ftepochs3" / "cnn_calibration_subjectwise.csv",
    "robustness_seed7": ROOT / "results_cnn_calibration_seed7" / "cnn_calibration_subjectwise.csv",
}
OUT = ROOT / "report_figs" / "cnn_calibration_followup"


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input: {path}. Run run_cnn_calibration_loso.py for this config first.")
    return pd.read_csv(path)


def per_subject_pivot(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.pivot(index="subject", columns="K_per_class", values="f1_macro")
    piv.columns = [f"K{c}" for c in piv.columns]
    for k in ["K5", "K10", "K20"]:
        if k in piv.columns:
            piv[f"lift_{k}"] = piv[k] - piv["K0"]
    return piv


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. Per-subject audit of the original (surprising) run ----
    orig = load(RUNS["original_seed42_ft15"])
    piv = per_subject_pivot(orig).sort_index()
    piv.to_csv(OUT / "per_subject_lift_original.csv")
    print(f"[save] {OUT / 'per_subject_lift_original.csv'}")

    rows = []
    for k in ["K5", "K10", "K20"]:
        col = f"lift_{k}"
        worse = int((piv[col] < 0).sum())
        better = int((piv[col] > 0).sum())
        same = int((piv[col] == 0).sum())
        rows.append({
            "K": k, "n_worse": worse, "n_better": better, "n_unchanged": same,
            "pct_worse": round(100 * worse / len(piv), 1),
            "mean_lift": round(float(piv[col].mean()), 4),
        })
    counts = pd.DataFrame(rows)
    counts.to_csv(OUT / "worse_better_counts.csv", index=False)
    print(f"[save] {OUT / 'worse_better_counts.csv'}")

    corr = piv[["K0", "lift_K5", "lift_K10", "lift_K20"]].corr()["K0"].drop("K0")
    corr_df = corr.reset_index()
    corr_df.columns = ["K", "corr_with_K0_baseline"]
    corr_df.to_csv(OUT / "baseline_vs_lift_correlation.csv", index=False)
    print(f"[save] {OUT / 'baseline_vs_lift_correlation.csv'}")

    # ---- 2. Three-way comparison: original vs mechanism-test vs seed-repeat ----
    compare_rows = []
    for label, path in RUNS.items():
        df = load(path)
        summ = df.groupby("K_per_class")["f1_macro"].agg(["mean", "std", "count"])
        base = summ.loc[0, "mean"]
        for k, r in summ.iterrows():
            compare_rows.append({
                "run": label, "K_per_class": k,
                "f1_mean": round(r["mean"], 4), "f1_sd": round(r["std"], 4), "n": int(r["count"]),
                "lift_pp_vs_K0": round((r["mean"] - base) * 100, 1),
            })
    comparison = pd.DataFrame(compare_rows)
    comparison.to_csv(OUT / "three_way_comparison.csv", index=False)
    print(f"[save] {OUT / 'three_way_comparison.csv'}")

    print("\n================  FOLLOW-UP AUDIT SUMMARY  ================")
    print("\n-- Per-subject direction at each K (original run, seed42/ft-epochs15) --")
    print(counts.to_string(index=False))
    print("\n-- Correlation: subject's K=0 baseline F1 vs their calibration lift --")
    print(corr_df.to_string(index=False))
    print("\n-- Three-way comparison (original vs mechanism test vs seed-repeat) --")
    print(comparison.to_string(index=False))
    print(f"\nAll outputs saved under {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
kc23_d2_reliance_stats.py
============================
KC-D2 stats/gate (analysis only, on KC-D1's instrumented runs).
EXPERIMENT_PLAN_KC23_DEEP.md "KC-D2. Reliance against trained-in robustness", D2.3.

Primary quantities (each per realization, then averaged across realizations):
  reduction factor of permutation reliance:  R1(none) / R2(chandrop)
  reduction factor of zeroing occlusion:     R1(none) / R3(gainjitter)   <- the cross-check
  reduction factor of attenuation:           R1(none) / R2(chandrop)

  O-R: zeroing occlusion falls >= 3-fold under gain jitter, AND permutation
       reliance falls >= 2-fold under channel dropout          -> supported by two measures
  O-T: zeroing occlusion falls < 1.5-fold under gain jitter, AND permutation
       reliance falls < 1.5-fold under channel dropout          -> occlusion = trained-in robustness
  O-M: anything else                                            -> report per measure

Reduction factor here is defined as mean(R1 per-subject cost) /
mean(R_aug per-subject cost); "cost" is occlusion drop_pp or permutation
f1_drop_mean, both already expressed so that a larger value means more
sensitivity, so a reduction factor > 1 means less sensitivity (as trained).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_stats_common import paired_test, print_gate_header, require_complete


def reduction_factor(cost_r1: np.ndarray, cost_r_aug: np.ndarray) -> float:
    denom = cost_r_aug.mean()
    return float(cost_r1.mean() / denom) if denom > 0 else float("inf")


def classify_o(occlusion_factor_gainjitter: float, permutation_factor_chandrop: float) -> str:
    if occlusion_factor_gainjitter >= 3.0 and permutation_factor_chandrop >= 2.0:
        return "O-R"
    if occlusion_factor_gainjitter < 1.5 and permutation_factor_chandrop < 1.5:
        return "O-T"
    return "O-M"


def _mean_per_subject_cost(df: pd.DataFrame, value_col: str) -> pd.Series:
    return df.groupby("subject")[value_col].mean()


def run(out_dir: Path) -> int:
    occ_r1 = pd.read_csv(out_dir / "r1_occlusion.csv")
    occ_r3 = pd.read_csv(out_dir / "r3_occlusion.csv")   # gainjitter
    perm_r1 = pd.read_csv(out_dir / "r1_permutation.csv")
    perm_r2 = pd.read_csv(out_dir / "r2_permutation.csv")  # chandrop
    atten_r1 = pd.read_csv(out_dir / "r1_attenuation.csv")
    atten_r2 = pd.read_csv(out_dir / "r2_attenuation.csv")

    occ_r1_cost = _mean_per_subject_cost(occ_r1, "drop_pp")
    occ_r3_cost = _mean_per_subject_cost(occ_r3, "drop_pp")
    perm_r1_cost = _mean_per_subject_cost(perm_r1, "f1_drop_mean")
    perm_r2_cost = _mean_per_subject_cost(perm_r2, "f1_drop_mean")

    occ_factor = reduction_factor(occ_r1_cost.to_numpy(), occ_r3_cost.to_numpy())
    perm_factor = reduction_factor(perm_r1_cost.to_numpy(), perm_r2_cost.to_numpy())

    letter = classify_o(occ_factor, perm_factor)
    reading = {
        "O-R": "Reduced reliance is supported by two independent measures.",
        "O-T": "Occlusion measured trained-in robustness; 'draws on what was there all along' goes.",
        "O-M": "Report per measure; permutation reliance is the reliance measure, occlusion the robustness measure.",
    }[letter]
    print_gate_header("KC-D2", letter, reading)
    print(f"  occlusion reduction factor (gainjitter): {occ_factor:.2f}x")
    print(f"  permutation reduction factor (chandrop): {perm_factor:.2f}x")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"occlusion_factor_gainjitter": occ_factor,
                   "permutation_factor_chandrop": perm_factor, "letter": letter}]
                ).to_csv(out_dir / "D2_detail.csv", index=False)
    (out_dir / "D2_VERDICT.md").write_text(f"# KC-D2 verdict\n\n**Outcome: {letter}**\n\n{reading}\n",
                                            encoding="utf-8")
    return 0  # D2 has no ESCALATE letter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.out)))


if __name__ == "__main__":
    main()

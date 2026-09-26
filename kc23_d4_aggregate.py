#!/usr/bin/env python3
"""
kc23_d4_aggregate.py
======================
KC-D4 aggregator (26 September 2026). kc23_d4_invariance_stats.py reads
d4_dose_sweep.csv and d4_gainjitter_boundary.csv; nothing produced them, so the
gate could only ever crash. This builds them from the real runs.

Inputs, seeds 42, 7 and 123 (the plan's 3 realizations), per-subject
normalization, ResNet-SE, instrumented:
  mpchandrop  SD {0.40, 0.50, 0.60, 0.80, 1.00}   results_kc23_d4_mpchandrop_sd<SD>_s<seed>
  gainjitter  SD 0.30 (R14), 0.40 (R3), 0.50 (R16)  results_kc23_d1_r14/r3/r16_s<seed>
              SD 0.80, 1.00                          results_kc23_d4_gainjitter_sd<SD>_s<seed>
  reference   R1 (none), R13 (chandrop 0.1), R2 (0.2), R17 (0.3), R15 (0.5)   results_kc23_d1_r*_s<seed>
Each run's run_config.json is checked against the arm it is supposed to be.

Outputs (in --out):
  d4_dose_sweep.csv           realization, subject, dose, f1, subject_probe_bacc, class_silhouette,
                              class_probe_bacc, permutation_sum   (mpchandrop, 5 doses)
  d4_gainjitter_boundary.csv  realization, subject, dose, f1        (gainjitter, 5 SDs)
  d4_reference_points.csv     arm, label, then the seed-averaged mean of each quantity
Any missing, mislabelled or incomplete run exits 1 and writes nothing.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

from kc23_run_loader import load_run

SEEDS = [42, 7, 123]
MPCHANDROP_DOSES = [0.40, 0.50, 0.60, 0.80, 1.00]
GAINJITTER_SOURCES = {          # SD -> (directory pattern, is D1 arm)
    0.30: "results_kc23_d1_r14_s{seed}",
    0.40: "results_kc23_d1_r3_s{seed}",
    0.50: "results_kc23_d1_r16_s{seed}",
    0.80: "results_kc23_d4_gainjitter_sd0.80_s{seed}",
    1.00: "results_kc23_d4_gainjitter_sd1.00_s{seed}",
}
REFERENCE_ARMS = [   # (arm, directory pattern, augmentation, chandrop p or None)
    ("R1", "results_kc23_d1_r1_s{seed}", "none", None),
    ("R13", "results_kc23_d1_r13_s{seed}", "chandrop", 0.1),
    ("R2", "results_kc23_d1_r2_s{seed}", "chandrop", 0.2),
    ("R17", "results_kc23_d1_r17_s{seed}", "chandrop", 0.3),
    ("R15", "results_kc23_d1_r15_s{seed}", "chandrop", 0.5),
]
QUANTITIES = ["f1", "subject_probe_bacc", "class_silhouette", "class_probe_bacc", "permutation_sum"]


def build(root: Path):
    sweep, gj, ref = [], [], []
    for seed in SEEDS:
        for dose in MPCHANDROP_DOSES:
            d = root / f"results_kc23_d4_mpchandrop_sd{dose:.2f}_s{seed}"
            r = load_run(d, augmentation="mpchandrop", seed=seed, gain_sd=dose)
            for s in r["f1"].index:
                sweep.append({"realization": seed, "subject": int(s), "dose": dose, "f1": r["f1"][s],
                              "subject_probe_bacc": r["subject_probe_bacc"][s],
                              "class_silhouette": r["class_silhouette"][s],
                              "class_probe_bacc": r["class_probe_bacc"][s],
                              "permutation_sum": r["permutation_sum"][s]})
        for dose, pattern in GAINJITTER_SOURCES.items():
            r = load_run(root / pattern.format(seed=seed), instrumented=False, augmentation="gainjitter",
                         seed=seed, gain_sd=dose)
            for s, v in r["f1"].items():
                gj.append({"realization": seed, "subject": int(s), "dose": dose, "f1": v})
        for arm, pattern, aug, p in REFERENCE_ARMS:
            r = load_run(root / pattern.format(seed=seed), augmentation=aug, seed=seed, chandrop_p=p)
            ref.append({"arm": arm, "seed": seed, **{q: float(r[q].mean()) for q in QUANTITIES}})
    ref_df = pd.DataFrame(ref).groupby("arm", sort=False)[QUANTITIES].mean().reset_index()
    ref_df["label"] = ref_df["arm"].map({a: f"{aug}" + (f" p={p}" if p else "") for a, _, aug, p in REFERENCE_ARMS})
    return pd.DataFrame(sweep), pd.DataFrame(gj), ref_df


def run(root: Path, out_dir: Path) -> int:
    try:
        sweep, gj, ref = build(root)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[D4-aggregate] FAIL, no output written: {e}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_dir / "d4_dose_sweep.csv", index=False)
    gj.to_csv(out_dir / "d4_gainjitter_boundary.csv", index=False)
    ref.to_csv(out_dir / "d4_reference_points.csv", index=False)
    print(f"[D4-aggregate] wrote d4_dose_sweep.csv ({len(sweep)} rows), d4_gainjitter_boundary.csv ({len(gj)}), "
          f"d4_reference_points.csv ({len(ref)})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.root), Path(args.out)))


if __name__ == "__main__":
    main()

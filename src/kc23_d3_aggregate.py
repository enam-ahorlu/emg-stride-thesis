#!/usr/bin/env python3
"""
src/kc23_d3_aggregate.py
======================
KC-D3 aggregator (26 September 2026). src/kc23_d3_axis_stats.py read
d3_realization_means.csv from the directory of the last X4 run, where nothing
had ever written it, so the gate could only crash. This builds it from the real
runs (plan D3.2 and D3.3, "realization means of X1 to X4 against R1 and R3 on
shared seeds").

Inputs, seeds 42, 7 and 123, ResNet-SE, per-subject normalization:
  X1  gaussian   sigma 0.10   results/kc23_d3_x1_s<seed>
  X2  gaussian   sigma 0.40   results/kc23_d3_x2_s<seed>
  X3  chanoffset SD 0.40      results/kc23_d3_x3_s<seed>
  X4  globalgain SD 0.40      results/kc23_d3_x4_s<seed>
  R1  none                    results/kc23_d1_r1_s<seed>   (comparator)
  R3  gainjitter SD 0.40      results/kc23_d1_r3_s<seed>   (comparator)
Each run's configuration is checked against its arm. Outputs:
  d3_realization_means.csv  arm, n_realizations, f1_mean (mean of the 40-fold means over the realizations),
                            f1_sd_across_realizations
  d3_per_arm_seed.csv       arm, seed, f1_mean
Any missing, mislabelled or incomplete run exits 1 and writes nothing.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

from kc23_run_loader import load_run

SEEDS = [42, 7, 123]
ARMS = {   # arm -> (directory pattern, augmentation, expected kwargs)
    "R1": ("results/kc23_d1_r1_s{seed}", "none", {}),
    "R3": ("results/kc23_d1_r3_s{seed}", "gainjitter", {"gain_sd": 0.40}),
    "X1": ("results/kc23_d3_x1_s{seed}", "gaussian", {}),
    "X2": ("results/kc23_d3_x2_s{seed}", "gaussian", {}),
    "X3": ("results/kc23_d3_x3_s{seed}", "chanoffset", {"gain_sd": 0.40}),
    "X4": ("results/kc23_d3_x4_s{seed}", "globalgain", {"gain_sd": 0.40}),
}
SIGMA = {"X1": 0.10, "X2": 0.40}


def build(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for arm, (pattern, aug, kw) in ARMS.items():
        for seed in SEEDS:
            d = root / pattern.format(seed=seed)
            r = load_run(d, instrumented=False, augmentation=aug, seed=seed, **kw)
            if arm in SIGMA:
                from kc23_run_loader import read_config
                sg = float(read_config(d).get("aug_sigma", -1))
                if abs(sg - SIGMA[arm]) > 1e-9:
                    raise ValueError(f"{d.name}: aug_sigma={sg}, expected {SIGMA[arm]}")
            rows.append({"arm": arm, "seed": seed, "f1_mean": float(r["f1"].mean())})
    per = pd.DataFrame(rows)
    g = per.groupby("arm", sort=False)["f1_mean"]
    means = pd.DataFrame({"n_realizations": g.size(), "f1_mean": g.mean(), "f1_sd_across_realizations": g.std(ddof=1)})
    return means.reset_index(), per


def run(root: Path, out_dir: Path) -> int:
    try:
        means, per = build(root)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[D3-aggregate] FAIL, no output written: {e}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    means.to_csv(out_dir / "d3_realization_means.csv", index=False)
    per.to_csv(out_dir / "d3_per_arm_seed.csv", index=False)
    print(f"[D3-aggregate] wrote d3_realization_means.csv ({len(means)} arms x {len(SEEDS)} realizations)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.root), Path(args.out)))


if __name__ == "__main__":
    main()

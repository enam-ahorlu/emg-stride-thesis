#!/usr/bin/env python3
"""
src/kc23_d2_aggregate.py
======================
KC-D2 aggregator (26 September 2026). D2 is analysis only, on KC-D1's instrumented
runs (plan D2.2). src/kc23_d2_reliance_stats.py read r1_occlusion.csv and five
siblings from its own directory, where nothing had ever written them, so the
gate could only crash. This builds its input from the real runs.

Arms (KC-D1's R1 to R5), seeds 42, 7, 123, 1001 and 2026 (the five instrumented Tier A
realizations), per-subject normalization, instrumented:
  R1 resnet_se none            R2 resnet_se chandrop 0.2      R3 resnet_se gainjitter 0.40
  R4 resnet    none            R5 resnet    chandrop 0.2

Per subject, per arm and realization, three summed drops (KC-D1's C17 definition,
"per-subject summed drop", NO clipping of negative drops; src/kc23_run_loader.py):
  occlusion_sum    sum over the channels of the F1 drop (pp) when the channel is zeroed
  attenuation_sum  the same when the channel is multiplied by alpha = 0.5
  permutation_sum  the same, in pp, when the channel is permuted across the subject's windows (mean over R = 5)

Output d2_persubject_sums.csv: arm, realization, subject, occlusion_sum, attenuation_sum, permutation_sum.
Any missing, mislabelled or incomplete run exits 1 and writes nothing.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

from kc23_run_loader import load_run

SEEDS = [42, 7, 123, 1001, 2026]
ARMS = {   # arm -> (directory pattern, arch, augmentation, expected kwargs)
    "R1": ("results/kc23_d1_r1_s{seed}", "resnet_se", "none", {}),
    "R2": ("results/kc23_d1_r2_s{seed}", "resnet_se", "chandrop", {"chandrop_p": 0.2}),
    "R3": ("results/kc23_d1_r3_s{seed}", "resnet_se", "gainjitter", {"gain_sd": 0.40}),
    "R4": ("results/kc23_d1_r4_s{seed}", "resnet", "none", {}),
    "R5": ("results/kc23_d1_r5_s{seed}", "resnet", "chandrop", {"chandrop_p": 0.2}),
}
QUANTITIES = ["occlusion_sum", "attenuation_sum", "permutation_sum"]


def build(root: Path) -> pd.DataFrame:
    rows = []
    for arm, (pattern, arch, aug, kw) in ARMS.items():
        for seed in SEEDS:
            r = load_run(root / pattern.format(seed=seed), probes=False, arch=arch, augmentation=aug, seed=seed, **kw)
            for s in r["f1"].index:
                rows.append({"arm": arm, "realization": seed, "subject": int(s),
                             **{q: float(r[q][s]) for q in QUANTITIES}})
    return pd.DataFrame(rows)


def run(root: Path, out_dir: Path) -> int:
    try:
        df = build(root)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[D2-aggregate] FAIL, no output written: {e}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "d2_persubject_sums.csv", index=False)
    print(f"[D2-aggregate] wrote d2_persubject_sums.csv ({len(df)} rows: {len(ARMS)} arms x {len(SEEDS)} realizations x 40)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.root), Path(args.out)))


if __name__ == "__main__":
    main()

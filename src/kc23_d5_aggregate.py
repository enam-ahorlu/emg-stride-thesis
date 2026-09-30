#!/usr/bin/env python3
"""
src/kc23_d5_aggregate.py
======================
KC-D5 aggregator (added 2026-09-25). src/kc23_d5_replication_stats.py reads
d5_finding_diffs.csv and d5_resnet_vs_svm.csv, and nothing produced them, so
the stats script ran on an empty directory and wrote a header-only verdict
with exit 0. This script produces them from the 15 real runs.

Inputs: results/kc23_d5_e{1,2,3}_s{42,7,123,1001,2026}, each one
src/run_cnn_arch_loso.py run on ENABL3S (10 subjects), instrumented:
  E1 ResNet-SE, no augmentation       (run_config.json augmentation "none")
  E2 ResNet-SE, channel dropout 0.2   ("chandrop")
  E3 ResNet-SE, gain jitter 0.40      ("gainjitter")
Each dir must hold cnn_arch_subjectwise.csv, instr/occlusion.csv and
instr/permutation.csv for the SAME 10 subjects, and a run_config.json whose
augmentation and seed match the directory name. Anything missing, duplicated,
mislabelled or NaN stops this script with a non-zero exit and no output file:
nothing is defaulted, and no published number is substituted.

Outputs (in --out):
  d5_finding_diffs.csv   finding, subject, diff, expected_sign
  d5_resnet_vs_svm.csv   seed, f1_resnet          (E2's 10-subject mean per seed)
  d5_magnitudes.csv      quantity, mean, sd_across_seeds, per_seed
  d5_arm_seed_means.csv  arm, seed, f1_mean        (descriptive)

Findings and directions (decision D-6c, 25 September 2026: the direction the
thesis states for each finding on SIAT-LLMD, NOT chosen from ENABL3S numbers;
both contested readings take the reading less favourable to the thesis):
  chandrop_gain              E2 - E1 F1                      expected +1
  gainjitter_vs_chandrop     E3 - E2 F1                      expected +1
      (Section 4.3.3: gain jitter is AHEAD of channel dropout)
  occlusion_reduction        E1 - E2 occlusion cost (drop_pp) expected +1
  permutation_reduction      E1 - E2 permutation cost         expected +1
      (the reliance claim is a REDUCTION under channel dropout)
All are realization-averaged per subject (mean over the 5 seeds). Occlusion
and permutation cost are the per-subject mean over channels, exactly as
kc23_d2_reliance_stats._mean_per_subject_cost defines them, and the reduction
FACTOR is that module's reduction_factor (mean R1 cost / mean augmented cost),
computed per realization and then averaged, with its across-seed SD.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kc23_d2_reliance_stats import _mean_per_subject_cost, reduction_factor

SEEDS = [42, 7, 123, 1001, 2026]
ARMS = {"e1": "none", "e2": "chandrop", "e3": "gainjitter"}
N_SUBJECTS = 10

FINDING_SIGNS = {
    "chandrop_gain": 1,
    "gainjitter_vs_chandrop": 1,
    "occlusion_reduction": 1,
    "permutation_reduction": 1,
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing input: {path}")
    return pd.read_csv(path)


def load_run(root: Path, arm: str, seed: int) -> dict:
    d = root / f"results/kc23_d5_{arm}_s{seed}"
    label = f"{arm}_s{seed}"
    cfg_path = d / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing input: {cfg_path}")
    args = json.loads(cfg_path.read_text(encoding="utf-8")).get("args", {})
    if args.get("augmentation") != ARMS[arm] or int(args.get("seed", -1)) != seed:
        raise ValueError(f"{label}: run_config says augmentation={args.get('augmentation')!r} "
                         f"seed={args.get('seed')!r}, expected {ARMS[arm]!r} / {seed}. Mislabelled run.")

    f = _read(d / "cnn_arch_subjectwise.csv")
    if set(f.columns) < {"subject", "arch", "f1_macro"}:
        raise ValueError(f"{label}: cnn_arch_subjectwise.csv lacks subject/arch/f1_macro")
    if set(f["arch"]) != {"resnet_se"}:
        raise ValueError(f"{label}: arch is {sorted(set(f['arch']))}, expected resnet_se only")
    if f["subject"].duplicated().any() or len(f) != N_SUBJECTS or f["f1_macro"].isna().any():
        raise ValueError(f"{label}: cnn_arch_subjectwise.csv must hold {N_SUBJECTS} unique non-NaN subjects, "
                         f"found {len(f)} rows")
    f1 = f.set_index("subject")["f1_macro"].astype(float).sort_index()

    occ = _read(d / "instr" / "occlusion.csv")
    perm = _read(d / "instr" / "permutation.csv")
    if not {"subject", "drop_pp"} <= set(occ.columns) or not {"subject", "f1_drop_mean"} <= set(perm.columns):
        raise ValueError(f"{label}: occlusion.csv needs subject/drop_pp, permutation.csv needs subject/f1_drop_mean")
    if occ["drop_pp"].isna().any() or perm["f1_drop_mean"].isna().any():
        raise ValueError(f"{label}: NaN in the occlusion or permutation cost")
    occ_cost = _mean_per_subject_cost(occ, "drop_pp").sort_index()
    perm_cost = _mean_per_subject_cost(perm, "f1_drop_mean").sort_index()
    subs = set(f1.index)
    if set(occ_cost.index) != subs or set(perm_cost.index) != subs:
        raise ValueError(f"{label}: occlusion/permutation subjects differ from the F1 subjects")
    return {"f1": f1, "occ": occ_cost, "perm": perm_cost * 100.0}   # permutation cost in pp, like occlusion


def load_all(root: Path) -> dict:
    runs = {(arm, seed): load_run(root, arm, seed) for arm in ARMS for seed in SEEDS}
    ref = set(next(iter(runs.values()))["f1"].index)
    for k, r in runs.items():
        if set(r["f1"].index) != ref:
            raise ValueError(f"{k}: subject set differs from the other runs")
    return runs


def _avg(runs: dict, arm: str, key: str) -> pd.Series:
    return pd.concat([runs[(arm, s)][key] for s in SEEDS], axis=1).mean(axis=1).sort_index()


def build_outputs(runs: dict) -> dict[str, pd.DataFrame]:
    f1 = {a: _avg(runs, a, "f1") for a in ARMS}
    occ = {a: _avg(runs, a, "occ") for a in ARMS}
    perm = {a: _avg(runs, a, "perm") for a in ARMS}
    diffs = {
        "chandrop_gain": (f1["e2"] - f1["e1"]) * 100.0,
        "gainjitter_vs_chandrop": (f1["e3"] - f1["e2"]) * 100.0,
        "occlusion_reduction": occ["e1"] - occ["e2"],
        "permutation_reduction": perm["e1"] - perm["e2"],
    }
    rows = [{"finding": name, "subject": int(s), "diff": float(v), "expected_sign": FINDING_SIGNS[name]}
            for name, ser in diffs.items() for s, v in ser.items()]

    per_seed_f1 = [float(runs[("e2", s)]["f1"].mean()) for s in SEEDS]
    resnet = pd.DataFrame({"seed": SEEDS, "f1_resnet": per_seed_f1})

    occ_factor = [reduction_factor(runs[("e1", s)]["occ"].to_numpy(), runs[("e2", s)]["occ"].to_numpy())
                  for s in SEEDS]
    perm_factor = [reduction_factor(runs[("e1", s)]["perm"].to_numpy(), runs[("e2", s)]["perm"].to_numpy())
                   for s in SEEDS]
    occ_gj_factor = [reduction_factor(runs[("e1", s)]["occ"].to_numpy(), runs[("e3", s)]["occ"].to_numpy())
                     for s in SEEDS]
    mags = []
    for name, vals in (("occlusion_reduction_factor_E1_over_E2", occ_factor),
                       ("permutation_reduction_factor_E1_over_E2", perm_factor),
                       ("occlusion_reduction_factor_E1_over_E3", occ_gj_factor)):
        mags.append({"quantity": name, "mean": float(np.mean(vals)),
                     "sd_across_seeds": float(np.std(vals, ddof=1)),
                     "per_seed": ";".join(f"{v:.3f}" for v in vals)})
    arm_means = pd.DataFrame([{"arm": a, "seed": s, "f1_mean": float(runs[(a, s)]["f1"].mean())}
                              for a in ARMS for s in SEEDS])
    return {"d5_finding_diffs.csv": pd.DataFrame(rows), "d5_resnet_vs_svm.csv": resnet,
            "d5_magnitudes.csv": pd.DataFrame(mags), "d5_arm_seed_means.csv": arm_means}


def run(root: Path, out_dir: Path) -> int:
    try:
        runs = load_all(root)
        outputs = build_outputs(runs)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[D5-aggregate] FAIL, no output written: {e}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(out_dir / name, index=False)
    print(f"[D5-aggregate] wrote {len(outputs)} files to {out_dir} from {len(runs)} runs")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.exit(run(Path(args.root), Path(args.out)))


if __name__ == "__main__":
    main()

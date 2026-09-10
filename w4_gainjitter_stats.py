#!/usr/bin/env python3
"""
w4_gainjitter_stats.py
======================
W-4. Is channel dropout's gain about REMOVING a channel, or about per-channel
multiplicative variation in general? Applies the pre-registered rule in
EXPERIMENT_PLAN_GAIN_JITTER.md section 4. No GPU.

  results_cd_resnet_noaug_repro     shared baseline, --arch resnet, no aug
  results_cd_resnet_nose_chandrop   channel dropout p = 0.2                (G1)
  results_w4_gainjitter             gain jitter, matched variance sd = 0.4 (W-4)

Estimators come from window_ablation_stats.py, verified by hand in W-1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from window_ablation_stats import bca_ci, cohens_d_paired


def load(d: str) -> pd.Series:
    p = ROOT / d / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    if df["subject"].duplicated().any():
        sys.exit(f"{p} has duplicated subjects; a --resume has double-appended")
    if len(df) != 40:
        print(f"WARNING: {d} has {len(df)} subjects, expected 40", file=sys.stderr)
    return df.set_index("subject")["f1_macro"].sort_index()


def line(label: str, diff: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    lo, hi = bca_ci(diff)
    w = stats.wilcoxon(a, b)
    d = cohens_d_paired(a, b)
    print(f"  {label:<40} {diff.mean()*100:+6.2f} pp  95% BCa [{lo*100:+.2f}, {hi*100:+.2f}]  "
          f"p = {w.pvalue:<10.4g} d = {d:+.2f}  "
          f"({int((diff > 0).sum())}/{int((diff < 0).sum())} +/-)")
    return {"mean_pp": diff.mean() * 100, "p": float(w.pvalue), "d": d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results_cd_resnet_noaug_repro")
    ap.add_argument("--cd", default="results_cd_resnet_nose_chandrop")
    ap.add_argument("--gain", default="results_w4_gainjitter")
    ap.add_argument("--extra-noaug", nargs="*", default=["results_g3_noaug_instr"],
                    help="further no-augmentation repeats, for the section 6 noise floor")
    args = ap.parse_args()

    base, cd, gain = load(args.base), load(args.cd), load(args.gain)
    for nm, s in [("chandrop", cd), ("gainjitter", gain)]:
        if not s.index.equals(base.index):
            sys.exit(f"subject index of {nm} does not match the baseline; gate 3 fails")

    print("arm means")
    for nm, s in [("no augmentation", base), ("channel dropout p=0.2", cd),
                  ("gain jitter sd=0.4", gain)]:
        print(f"  {nm:<24} {s.mean():.4f}  (sd {s.std(ddof=1):.4f})")

    print("\n--- gains against the shared no-augmentation baseline ---")
    d_cd = (cd - base).to_numpy()
    d_gn = (gain - base).to_numpy()
    line("channel dropout", d_cd, cd.to_numpy(), base.to_numpy())
    r_gn = line("gain jitter", d_gn, gain.to_numpy(), base.to_numpy())

    print("\n--- the contrast: removal minus variation ---")
    contrast = d_cd - d_gn                       # per subject, before testing
    lo, hi = bca_ci(contrast)
    w = stats.wilcoxon(d_cd, d_gn)
    dz = contrast.mean() / contrast.std(ddof=1)
    print(f"  {'chandrop minus gainjitter':<40} {contrast.mean()*100:+6.2f} pp  "
          f"95% BCa [{lo*100:+.2f}, {hi*100:+.2f}]  p = {w.pvalue:<10.4g} d = {dz:+.2f}  "
          f"({int((contrast > 0).sum())}/{int((contrast < 0).sum())} +/-)")
    print("  family of one; no Holm correction applies")

    g = r_gn["mean_pp"]
    sig_pos = (w.pvalue < 0.05) and (contrast.mean() > 0)
    if g >= 4.0 and not sig_pos:
        letter, meaning = "P", (
            "perturbation. Any channel-structured multiplicative perturbation does the work; "
            "zeroing is not special. Section 5.7's liftoff-and-re-siting framing is too specific "
            "and must be reworded to a claim about sensor variability.")
    elif 2.0 <= g < 4.0:
        letter, meaning = "M", (
            "mixed. Both removal and variation contribute. Section 5.7 apportions. Run the "
            "conditional second arm at --aug-gain-sd 0.2 to test gain jitter's own dose-response.")
    elif g < 2.0 and sig_pos:
        letter, meaning = "R", (
            "removal. Varying a channel is not enough; the network has to learn to proceed "
            "without one. Section 5.7 gets its strongest form and the sensor-failure framing "
            "is earned rather than asserted.")
    else:
        letter, meaning = "?", (
            f"outside the pre-registered grid: gain {g:+.2f} pp with contrast p = {w.pvalue:.4g}. "
            "Report the numbers and do not force a letter.")
    print(f"\nOUTCOME {letter}: {meaning}")

    # ---- section 6, the free noise-floor by-product --------------------------
    reps = [base] + [load(d) for d in args.extra_noaug if (ROOT / d).exists()]
    if len(reps) > 1:
        m = np.array([r.mean() for r in reps])
        print(f"\n--- section 6: no-augmentation repeats at identical settings (n = {len(m)}) ---")
        print("  means: " + ", ".join(f"{x:.4f}" for x in m))
        print(f"  SD {m.std(ddof=1)*100:.2f} pp, range {m.min():.4f} to {m.max():.4f} "
              f"(span {(m.max()-m.min())*100:.2f} pp)")
        print("  This is the run-to-run floor for --arch resnet under per-subject "
              "normalization. R-1 measured 0.47 pp on resnet_se under global normalization.")
    else:
        print("\n(section 6 skipped: only one no-augmentation run found)")

    pd.DataFrame({"noaug": base, "chandrop": cd, "gainjitter": gain,
                  "d_cd_pp": d_cd * 100, "d_gain_pp": d_gn * 100,
                  "contrast_pp": contrast * 100}).round(4).to_csv(
        ROOT / "w4_gainjitter_pairs.csv")
    print("\nper-subject values written to w4_gainjitter_pairs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

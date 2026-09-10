#!/usr/bin/env python3
"""
verify_section_4_8_1.py
=======================
Regenerates every number quoted in thesis Section 4.8.1, "The Gain Is
Architecture-Dependent, and Squeeze-and-Excitation Is Not the Reason", and in
the corresponding passage of Section 5.7. Reads only; no GPU.

Estimators are imported from window_ablation_stats.py, whose BCa interval,
paired Cohen's d and Holm routine were verified by hand during W-1.

Run this after any change to the underlying result directories, and before
believing any figure in Section 4.8.1.

  python verify_section_4_8_1.py

Expected output (2 September 2026):

  Simple +CD                    +0.30 pp  CI[-0.69,+1.45]  p=0.816     d=+0.09  (16/24)
  ResNet-SE +CD                 +5.73 pp  CI[+4.34,+7.27]  p=3.163e-09 d=+1.20  (36/4)
  ResNet noSE +CD               +6.51 pp  CI[+5.00,+7.98]  p=2.294e-09 d=+1.35  (36/4)
  SE main effect, +CD           +1.44 pp  CI[+0.65,+2.31]  p=0.002061  d=+0.53  (27/13)
  SE main effect, no aug        +2.23 pp  CI[+1.33,+3.18]  p=4.404e-05 d=+0.73  (31/9)
  family: repro vs persubj      +0.65 pp  CI[-0.23,+1.46]  p=0.1316    d=+0.24  (23/17)
  G1 interaction (SE - noSE)    -0.78 pp  CI[-2.05,+0.55]  p=0.2214    d=-0.18

Note on families. The SimpleEMGCNN arms come from train_cnn_loso.py and the
residual arms from run_cnn_arch_loso.py. Every gain reported above is measured
WITHIN one script; the last row is the cross-script check that licenses putting
the two gains side by side, and it is not significant.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from window_ablation_stats import bca_ci, cohens_d_paired


def simple(d: str) -> pd.Series:
    return (pd.read_csv(ROOT / d / "per_subject_metrics_cnn_loso.csv")
              .set_index("subject")["f1_macro"].sort_index())


def harn(d: str) -> pd.Series:
    return (pd.read_csv(ROOT / d / "cnn_arch_subjectwise.csv")
              .set_index("subject")["f1_macro"].sort_index())


def paired(name: str, a: pd.Series, b: pd.Series) -> np.ndarray:
    ix = a.index.intersection(b.index)
    if len(ix) != 40:
        print(f"WARNING: {name} pairs {len(ix)} subjects, expected 40", file=sys.stderr)
    a, b = a.loc[ix], b.loc[ix]
    d = (a - b).to_numpy()
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(a, b)
    print(f"{name:<30} {d.mean()*100:+6.2f} pp  CI[{lo*100:+.2f},{hi*100:+.2f}]  "
          f"p={w.pvalue:<11.4g} d={cohens_d_paired(a.to_numpy(), b.to_numpy()):+.2f}  "
          f"({int((d > 0).sum())}/{int((d < 0).sum())})")
    return d


def main() -> int:
    paired("Simple +CD",
           simple("results_cnn_loso_aug_chandrop"), simple("results_cnn_loso_norm_persubj"))
    d_se = paired("ResNet-SE +CD",
                  harn("results_cnn_aug_resnet_se_chandrop"), harn("results_cnn_loso_resnet_se"))
    d_r = paired("ResNet noSE +CD",
                 harn("results_cd_resnet_nose_chandrop"), harn("results_cd_resnet_noaug_repro"))
    paired("SE main effect, +CD",
           harn("results_cnn_aug_resnet_se_chandrop"), harn("results_cd_resnet_nose_chandrop"))
    paired("SE main effect, no aug",
           harn("results_cnn_loso_resnet_se"), harn("results_cd_resnet_noaug_repro"))
    paired("family: repro vs persubj",
           harn("results_cnn_loso_simple_repro"), simple("results_cnn_loso_norm_persubj"))

    inter = d_se - d_r                      # per subject, before testing
    lo, hi = bca_ci(inter)
    w = stats.wilcoxon(d_se, d_r)
    print(f"{'G1 interaction (SE - noSE)':<30} {inter.mean()*100:+6.2f} pp  "
          f"CI[{lo*100:+.2f},{hi*100:+.2f}]  p={w.pvalue:<11.4g} "
          f"d={inter.mean()/inter.std(ddof=1):+.2f}")

    # ---- W-3, the skip-connection ablation and the decomposition -------------
    nb, nc = harn("results_w3_nores_noaug"), harn("results_w3_nores_chandrop")
    print()
    paired("no-skip vs plain, no aug", nb, harn("results_cd_resnet_noaug_repro"))
    d_nr = paired("no-skip +CD", nc, nb)

    skips = d_r - d_nr
    lo, hi = bca_ci(skips); w = stats.wilcoxon(d_r, d_nr)
    print(f"{'skip contribution':<30} {skips.mean()*100:+6.2f} pp  "
          f"CI[{lo*100:+.2f},{hi*100:+.2f}]  p={w.pvalue:<11.4g} "
          f"d={skips.mean()/skips.std(ddof=1):+.2f}")

    simp = (simple("results_cnn_loso_aug_chandrop").to_numpy()
            - simple("results_cnn_loso_norm_persubj").to_numpy())
    depth = d_nr - simp
    lo, hi = bca_ci(depth); w = stats.wilcoxon(d_nr, simp)
    print(f"{'depth + capacity contribution':<30} {depth.mean()*100:+6.2f} pp  "
          f"CI[{lo*100:+.2f},{hi*100:+.2f}]  p={w.pvalue:<11.4g} "
          f"d={depth.mean()/depth.std(ddof=1):+.2f}")
    tot = depth.mean() + skips.mean()
    print(f"{'total architecture-dependence':<30} {tot*100:+6.2f} pp   "
          f"depth+capacity {100*depth.mean()/tot:.0f}%, skips {100*skips.mean()/tot:.0f}%")

    # headroom sensitivity: the skip-free baseline sits higher, which cuts against the skip term
    pb = harn("results_cd_resnet_noaug_repro").to_numpy()
    exp_nr = d_r * (1 - nb.to_numpy()) / (1 - pb)
    resid = exp_nr - d_nr
    lo, hi = bca_ci(resid); w = stats.wilcoxon(exp_nr, d_nr)
    print(f"{'  skip term, headroom-adjusted':<30} {resid.mean()*100:+6.2f} pp  "
          f"CI[{lo*100:+.2f},{hi*100:+.2f}]  p={w.pvalue:<11.4g}   "
          "(post hoc sensitivity check, not pre-registered; the BCa interval and the "
          "rank test disagree, and the thesis reports both)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

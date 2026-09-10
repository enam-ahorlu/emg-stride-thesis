#!/usr/bin/env python3
"""
g2_rate_stats.py
================
W-2 Stage G2, the channel-dropout rate sweep, reactivated 2 September 2026.

Applies the PRE-REGISTERED rule in EXPERIMENT_PLAN_CHANNEL_DROPOUT.md sections
4.1 and 4.3 without alteration. Two questions, and they are different:

  MECHANISM  does the sweep show the monotone-then-decline shape? That would
             mirror the over-alignment result of §4.13.2 and give the thesis
             "too much invariance hurts" from two unrelated mechanisms. This
             is what the stage is for.

  OPERATING  can any p displace 0.2 as the model of record? §4.3 requires ALL
  POINT      of four conditions. If they hold, STOP AND ESCALATE; do not adopt
             the new p and do not touch any downstream artifact.

Arms, all --arch resnet_se, per-subject normalization, seed 42, instrumented:
  results_cd_rate_p0.2   the era-internal comparator (NOT the published 0.8395)
  results_cd_rate_p0.1 / p0.3 / p0.5

No GPU.
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
from window_ablation_stats import bca_ci, cohens_d_paired, holm  # verified in W-1

PUBLISHED_P02 = 0.8395     # results_cnn_aug_resnet_se_chandrop, the model of record
REPRO_GATE_PP = 1.5        # R-1: ~3 sigma on a measured run-to-run SD of 0.47 pp
MIN_MARGIN_PP = 3.0        # §4.3, revised 1 September after R-1


def load(d: str) -> pd.Series:
    p = ROOT / d / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    if df["subject"].duplicated().any():
        sys.exit(f"{p} has duplicated subjects; a --resume has double-appended")
    return df.set_index("subject")["f1_macro"].sort_index()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", nargs="*", default=["0.1", "0.2", "0.3", "0.5"])
    ap.add_argument("--prefix", default="results_cd_rate_p")
    ap.add_argument("--noise-sd-pp", type=float, default=0.47,
                    help="measured run-to-run SD; R-1 gives 0.47 pp on this architecture")
    args = ap.parse_args()

    arms = {r: load(f"{args.prefix}{r}") for r in args.rates}
    if "0.2" not in arms:
        sys.exit("p = 0.2 must be among the arms; it is the comparator")
    ref = arms["0.2"]
    for r, s in arms.items():
        if not s.index.equals(ref.index):
            sys.exit(f"subject index of p={r} does not match p=0.2")

    # ---- gate: the fresh p = 0.2 must reproduce the model of record ----------
    delta = (ref.mean() - PUBLISHED_P02) * 100
    print(f"--- §4.1 reproduction gate ---")
    print(f"  fresh p=0.2 {ref.mean():.4f} vs published {PUBLISHED_P02:.4f}: {delta:+.2f} pp")
    if abs(delta) > REPRO_GATE_PP:
        print(f"  FAIL, exceeds {REPRO_GATE_PP} pp. Stop and report; the sweep is not "
              "interpretable against a comparator that does not reproduce.")
        return 2
    print(f"  PASS (gate is +/-{REPRO_GATE_PP} pp; +/-0.002 would fail ~67% of the time by chance)")

    print("\n--- arm means ---")
    for r in sorted(arms, key=float):
        print(f"  p = {r:<4} {arms[r].mean():.4f}  (sd {arms[r].std(ddof=1):.4f})")

    # ---- family 2 of the parent plan: each rate against p = 0.2 -------------
    rows = []
    for r in sorted(arms, key=float):
        if r == "0.2":
            continue
        a, b = arms[r].to_numpy(), ref.to_numpy()
        d = a - b
        lo, hi = bca_ci(d)
        rows.append({"label": f"p = {r} vs p = 0.2", "rate": float(r),
                     "mean_pp": d.mean() * 100, "lo_pp": lo * 100, "hi_pp": hi * 100,
                     "p_raw": float(stats.wilcoxon(a, b).pvalue),
                     "d": cohens_d_paired(a, b),
                     "pos": int((d > 0).sum()), "neg": int((d < 0).sum())})
    holm(rows)
    print("\n--- §4.3 family: each rate against p = 0.2, Holm corrected within ---")
    for r in rows:
        print(f"  {r['label']:<22} {r['mean_pp']:+6.2f} pp  "
              f"95% BCa [{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]  "
              f"p = {r['p_raw']:<10.4g} Holm p = {r['p_holm']:<10.4g} d = {r['d']:+.2f}  "
              f"({r['pos']}/{r['neg']} +/-)")

    # ---- the mechanism reading, which is what the stage is for ---------------
    print("\n--- the mechanism question: is the shape monotone then declining? ---")
    order = sorted(arms, key=float)
    means = [arms[r].mean() for r in order]
    peak = order[int(np.argmax(means))]
    p05 = next((r for r in rows if r["rate"] == 0.5), None)
    print(f"  peak at p = {peak}; sequence " +
          " -> ".join(f"{m:.4f}" for m in means))
    if p05 is not None and p05["mean_pp"] <= -MIN_MARGIN_PP and p05["p_holm"] < 0.05:
        print(f"  p = 0.5 is {abs(p05['mean_pp']):.2f} pp BELOW p = 0.2 and significant. "
              "The over-regularization parallel HOLDS: too much enforced channel invariance "
              "destroys class structure, mirroring §4.13.2's over-alignment result. This is "
              "the reportable finding of the stage.")
    else:
        print("  p = 0.5 is not clearly worse than p = 0.2. The over-regularization parallel "
              "is NOT supported; report the sweep as flat within noise and do not claim the "
              "shape. Say so plainly rather than reading a trend into it.")

    # ---- the operating-point question, §4.3 ---------------------------------
    thresh = max(MIN_MARGIN_PP, 2 * args.noise_sd_pp)
    print(f"\n--- §4.3 operating point: can any p displace 0.2? "
          f"(margin threshold {thresh:.2f} pp) ---")
    challengers = [r for r in rows
                   if r["p_holm"] < 0.05 and r["mean_pp"] >= thresh]
    if not challengers:
        print("  No. p = 0.2 STANDS. Every rate either fails Holm-corrected significance or "
              "falls short of the margin. The sweep is reported as a mechanism result only, "
              "and no downstream artifact is touched. This is the expected and desired case.")
    else:
        for c in challengers:
            print(f"  CHALLENGER: {c['label']} at {c['mean_pp']:+.2f} pp, Holm p = {c['p_holm']:.4g}")
        print("  Two of §4.3's four conditions are met. The remaining two are NOT tested here:\n"
              "    - the ordering must reproduce under seed 7\n"
              "    - the margin must exceed the spread between repeats of p = 0.2 against itself\n"
              "  STOP AND ESCALATE TO ENAM. Do not adopt the new p. Do not touch the ensemble, "
              "the causal chain, the external validation or the headline. See parent plan §9: "
              "eight downstream result sets consume p = 0.2.")

    pd.DataFrame({f"p{r}": arms[r] for r in order}).round(4).to_csv(ROOT / "g2_rate_pairs.csv")
    pd.DataFrame(rows).round(5).to_csv(ROOT / "g2_rate_tests.csv", index=False)
    print("\nwrote g2_rate_pairs.csv, g2_rate_tests.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

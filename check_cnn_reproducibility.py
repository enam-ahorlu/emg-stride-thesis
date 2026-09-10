#!/usr/bin/env python3
"""
check_cnn_reproducibility.py
============================
Step 1 of the CNN reproducibility check. Costs no GPU time.

Two runs claim to measure the same quantity, ResNet-SE+CD at 250 ms under
train-fold global normalization, and disagree by about 1.5 pp:

  results_adabn_chandrop/       f1_pre_adabn   = 0.7874   (the thesis's published 0.787)
  results_win250_cnn_global/    f1_macro       = 0.7723   (generated during W-1)

This script pairs them per subject and asks which of two explanations the
disagreement fits. It does not decide; it produces the evidence.

  NONDETERMINISM  differences scattered around a small mean, both signs well
                  represented, per-subject spread of several pp, and the two
                  runs still strongly correlated across subjects. The pipeline
                  is the same and cuDNN is simply not deterministic.

  CONFIG DIFF     differences with a consistent sign on most subjects, tight
                  spread, or a weak correlation. Something in the two runs is
                  genuinely different and Step 0 has missed it.

Usage:
  python check_cnn_reproducibility.py
  python check_cnn_reproducibility.py --a <dir> --a-col f1_pre_adabn --b <dir> --b-col f1_macro
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent


def load(d: Path, col: str) -> pd.Series:
    hits = sorted(d.glob("*subjectwise*.csv"))
    if not hits:
        sys.exit(f"no *subjectwise*.csv under {d}")
    df = pd.read_csv(hits[0])
    if col not in df.columns:
        sys.exit(f"{hits[0].name} has no column {col!r}; columns are {list(df.columns)}")
    sc = "heldout_subject" if "heldout_subject" in df.columns else "subject"
    df = df[[sc, col]].dropna().rename(columns={sc: "subject", col: "f1"})
    if df["subject"].duplicated().any():
        sys.exit(f"{hits[0].name} has duplicated subjects; a resume has double-appended")
    return df.set_index("subject")["f1"].sort_index()


def main() -> int:
    ap = argparse.ArgumentParser("Pair two nominally identical CNN runs, per subject.")
    ap.add_argument("--a", default="results_adabn_chandrop")
    ap.add_argument("--a-col", default="f1_pre_adabn")
    ap.add_argument("--a-label", default="AdaBN pre-swap (published 0.787)")
    ap.add_argument("--b", default="results_win250_cnn_global")
    ap.add_argument("--b-col", default="f1_macro")
    ap.add_argument("--b-label", default="W-1 fresh run (0.772)")
    args = ap.parse_args()

    a = load(ROOT / args.a, args.a_col)
    b = load(ROOT / args.b, args.b_col)
    common = a.index.intersection(b.index)
    if len(common) != 40:
        print(f"WARNING: {len(common)} subjects in common, expected 40", file=sys.stderr)
    a, b = a.loc[common], b.loc[common]
    d = (a - b).to_numpy()

    print(f"A = {args.a_label:<34} mean {a.mean():.4f}  sd {a.std(ddof=1):.4f}")
    print(f"B = {args.b_label:<34} mean {b.mean():.4f}  sd {b.std(ddof=1):.4f}")
    print(f"\nmean paired difference A - B : {d.mean()*100:+.2f} pp")
    print(f"sd of the differences        : {d.std(ddof=1)*100:.2f} pp")
    print(f"range                        : {d.min()*100:+.2f} to {d.max()*100:+.2f} pp")

    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    print(f"sign split                   : {pos} subjects A>B, {neg} A<B")

    r = stats.pearsonr(a, b)
    rho = stats.spearmanr(a, b)
    print(f"correlation across subjects  : Pearson r = {r[0]:.3f}, Spearman = {rho[0]:.3f}")

    w = stats.wilcoxon(a, b) if not np.allclose(d, 0) else None
    if w is not None:
        dz = d.mean() / d.std(ddof=1)
        print(f"paired Wilcoxon              : p = {w.pvalue:.4g},  d = {dz:.2f}")

    # --- what the pattern looks like ------------------------------------
    # Only two of these discriminate. Correlation is high under BOTH hypotheses,
    # because the same 40 subjects drive it either way, so it is a sanity check
    # on the pairing rather than evidence for one explanation over the other.
    print("\n--- reading of the pattern (indicative, not a verdict) ---")
    balanced = min(pos, neg) >= 0.30 * len(d)
    wide = d.std(ddof=1) * 100 >= 1.5
    for label, ok, why in [
        ("signs are balanced across subjects", balanced,
         f"{pos}/{neg} split; nondeterminism scatters both ways, a config change shifts one way"),
        ("per-subject spread is wide", wide,
         f"sd {d.std(ddof=1)*100:.2f} pp; run-to-run noise is wide, a systematic offset is narrow"),
    ]:
        print(f"  [{'x' if ok else ' '}] {label:<36} {why}")

    if r[0] < 0.85:
        print(f"  [!] correlation is only r = {r[0]:.3f}. Expected high under either "
              "explanation, so a low value means the pairing itself is suspect. "
              "Check the two runs really cover the same subjects and metric.")

    votes = sum([balanced, wide])
    print()
    if votes == 2:
        print("  Consistent with NONDETERMINISM. Proceed to Step 2 to measure the noise floor.")
    elif votes == 0:
        print("  Consistent with a CONFIG DIFFERENCE. Go back to Step 0 and find it "
              "before spending GPU time; Step 2 would only measure noise around the wrong baseline.")
    else:
        print("  Ambiguous, one signature each way. Step 2 settles it.")

    out = pd.DataFrame({"A": a, "B": b, "diff_pp": d * 100}).round(4)
    out.to_csv(ROOT / "cnn_reproducibility_pairs.csv")
    print(f"\nper-subject pairs written to cnn_reproducibility_pairs.csv")
    print("\nlargest disagreements:")
    print(out.reindex(out.diff_pp.abs().sort_values(ascending=False).index).head(6).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

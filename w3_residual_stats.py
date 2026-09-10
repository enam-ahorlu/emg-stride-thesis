#!/usr/bin/env python3
"""
w3_residual_stats.py
====================
W-3, the skip-connection ablation. Reads four 40-fold LOSO runs and applies the
pre-registered decision rule in EXPERIMENT_PLAN_RESIDUAL_ABLATION.md sections 4
and 5. No GPU. Run after run_w3_nores.sh completes.

The four arms, all SE-free, all per-subject normalization, seed 42:

  results_cd_resnet_noaug_repro    plain residual, no augmentation      (G1)
  results_cd_resnet_nose_chandrop  plain residual, channel dropout 0.2  (G1)
  results_w3_nores_noaug           no skip connections, no augmentation (W-3)
  results_w3_nores_chandrop        no skip connections, chandrop 0.2    (W-3)

Estimators are imported from window_ablation_stats.py so that the BCa interval,
the paired Cohen's d and the Holm routine are the ones verified by hand in W-1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from window_ablation_stats import bca_ci, cohens_d_paired  # verified in W-1

PLAIN_BASE = "results_cd_resnet_noaug_repro"
PLAIN_CD = "results_cd_resnet_nose_chandrop"
NORES_BASE = "results_w3_nores_noaug"
NORES_CD = "results_w3_nores_chandrop"

PLAIN_BASE_REF = 0.7600  # the §4 trainability yardstick, re-read below rather than trusted


def load(dirname: str) -> pd.Series:
    d = ROOT / dirname
    hits = sorted(d.glob("*subjectwise*.csv"))
    if not hits:
        sys.exit(f"no *subjectwise*.csv under {d}")
    df = pd.read_csv(hits[0])
    sc = "heldout_subject" if "heldout_subject" in df.columns else "subject"
    df = df[[sc, "f1_macro"]].dropna().rename(columns={sc: "subject"})
    if df["subject"].duplicated().any():
        sys.exit(f"{hits[0]} has duplicated subjects; a --resume has double-appended. "
                 "Delete the run directory and re-run rather than trusting it.")
    if len(df) != 40:
        print(f"WARNING: {dirname} has {len(df)} subjects, expected 40", file=sys.stderr)
    return df.set_index("subject")["f1_macro"].sort_index()


def report(label: str, diff: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    lo, hi = bca_ci(diff)
    w = stats.wilcoxon(a, b)
    d = cohens_d_paired(a, b)
    print(f"{label:<44} {diff.mean()*100:+6.2f} pp   "
          f"95% BCa [{lo*100:+.2f}, {hi*100:+.2f}]   "
          f"p = {w.pvalue:.4g}   d = {d:+.2f}   "
          f"({int((diff > 0).sum())}/{int((diff < 0).sum())} +/-)")
    return {"mean_pp": diff.mean() * 100, "lo": lo * 100, "hi": hi * 100,
            "p": w.pvalue, "d": d}


def main() -> int:
    pb, pc = load(PLAIN_BASE), load(PLAIN_CD)
    nb, nc = load(NORES_BASE), load(NORES_CD)

    ix = pb.index
    for name, s in [("plain+CD", pc), ("nores", nb), ("nores+CD", nc)]:
        if not s.index.equals(ix):
            sys.exit(f"subject index of {name} does not match {PLAIN_BASE}. "
                     "Gate 2 of section 7 fails; stop.")

    print("arm means")
    for name, s in [("plain residual, no aug", pb), ("plain residual + CD", pc),
                    ("no skip, no aug", nb), ("no skip + CD", nc)]:
        print(f"  {name:<26} {s.mean():.4f}   (sd {s.std(ddof=1):.4f})")

    # ---- section 4, the trainability check, which comes first -----------------
    gap = (nb.mean() - pb.mean()) * 100
    print(f"\n--- section 4: trainability check ---")
    print(f"no-skip no-aug baseline {nb.mean():.4f} against plain residual "
          f"{pb.mean():.4f}: {gap:+.2f} pp")
    if gap > -3.0:
        verdict4 = "both networks trained; the comparison below is clean"
        stop = False
        conditional = False
    elif gap > -8.0 and nb.mean() >= 0.68:
        verdict4 = ("partial degradation; the verdict below is CONDITIONAL, since a larger "
                    "channel-dropout gain on a weaker baseline is partly headroom")
        stop = False
        conditional = True
    else:
        verdict4 = ("the no-skip network did not train. The result says nothing about channel "
                    "dropout. Report the collapse and stop; do not read the interaction test.")
        stop = True
        conditional = True
    print(f"  -> {verdict4}")
    if stop:
        return 0

    # ---- section 5, the interaction ------------------------------------------
    print(f"\n--- section 5: channel-dropout gains and their interaction ---")
    d_res = (pc - pb).to_numpy()
    d_nores = (nc - nb).to_numpy()
    r_res = report("channel dropout on plain residual", d_res, pc.to_numpy(), pb.to_numpy())
    r_nores = report("channel dropout on no-skip", d_nores, nc.to_numpy(), nb.to_numpy())

    inter = d_res - d_nores  # per subject, before testing
    lo, hi = bca_ci(inter)
    w = stats.wilcoxon(d_res, d_nores)
    dz = inter.mean() / inter.std(ddof=1)
    print(f"{'interaction (res minus no-skip)':<44} {inter.mean()*100:+6.2f} pp   "
          f"95% BCa [{lo*100:+.2f}, {hi*100:+.2f}]   p = {w.pvalue:.4g}   d = {dz:+.2f}   "
          f"({int((inter > 0).sum())}/{int((inter < 0).sum())} +/-)")
    print("  family of one; no Holm correction applies")

    # ---- the pre-registered outcome letter -----------------------------------
    g = r_nores["mean_pp"]
    sig_pos = (w.pvalue < 0.05) and (inter.mean() > 0)
    if g < 2.0 and sig_pos:
        letter, meaning = "K", ("skip-dependent. The identity path is what lets channel dropout "
                                "work. Section 4.8.1 narrows from three candidates to one.")
    elif 2.0 <= g < 4.0:
        letter, meaning = "M", ("mixed. Skip connections contribute but do not account for the "
                                "effect. Section 4.8.1 apportions between skips and depth/capacity.")
    elif g >= 4.0 and not sig_pos:
        letter, meaning = "D", ("not the skips. Section 4.8.1's confound narrows from three "
                                "candidates to two, depth and capacity. This is a real result.")
    else:
        letter, meaning = "?", ("outside the pre-registered grid: gain "
                                f"{g:+.2f} pp with interaction p = {w.pvalue:.4g}. "
                                "Report the numbers and do not force a letter.")
    print(f"\nOUTCOME {letter}: {meaning}")
    if conditional:
        print("CONDITIONAL on the section 4 caveat above; state it in the same sentence "
              "as the outcome letter, not in a footnote.")

    out = pd.DataFrame({"plain_base": pb, "plain_cd": pc, "nores_base": nb, "nores_cd": nc,
                        "d_res_pp": d_res * 100, "d_nores_pp": d_nores * 100,
                        "interaction_pp": inter * 100}).round(4)
    out.to_csv(ROOT / "w3_residual_pairs.csv")
    print("\nper-subject values written to w3_residual_pairs.csv")
    print("\nlargest interaction disagreements:")
    print(out.reindex(out.interaction_pp.abs().sort_values(ascending=False).index)
             .head(6)[["d_res_pp", "d_nores_pp", "interaction_pp"]].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

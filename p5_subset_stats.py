#!/usr/bin/env python3
"""
p5_subset_stats.py
==================
Stage P-5 (EXPERIMENT_PLAN_CD_PARITY.md section 5A). Subset training, a
published channel-level alternative (Pereira et al., ICASSP 2024), against the
incumbent channel dropout. No GPU here; reads the 40-fold run written by
run_p5_p6.sh.

  results_p5_subset_resnet_se            subset mode, resnet_se, per-subject norm  (new)
  results_cnn_aug_resnet_se_chandrop     channel dropout p = 0.2                   (0.840)
  results_cnn_aug_resnet_se_none         no augmentation, resnet_se                (0.782)

Primary  : subset vs channel dropout.  Secondary: subset vs no augmentation.
Paired /40, BCa 95%, paired Wilcoxon, paired Cohen's d, Holm family of two.
2.0 pp floor carried from W-5. Grid M / B / W / F / X in section 5A.4.
Estimators from window_ablation_stats.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from window_ablation_stats import bca_ci, cohens_d_paired

OUT = ROOT / "results_parity"
OUT.mkdir(exist_ok=True)

SUBSET = "results_p5_subset_resnet_se"
CHANDROP = "results_cnn_aug_resnet_se_chandrop"
NOAUG = "results_cnn_aug_resnet_se_none"
FLOOR_PP = 2.0
TRAIN_X_PP = 3.0


def load(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p} (has run_p5_p6.sh finished the subset arm?)")
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    df = df[[sc, "f1_macro"]].dropna().rename(columns={sc: "subject"})
    if df["subject"].duplicated().any():
        sys.exit(f"{p} has duplicated subjects; a --resume double-appended")
    return df.set_index("subject")["f1_macro"].sort_index().rename(dirname)


def holm2(pa: float, pb: float) -> tuple[float, float]:
    order = sorted([("a", pa), ("b", pb)], key=lambda kv: kv[1])
    out, run = {}, 0.0
    for k, (nm, p) in enumerate(order):
        run = max(run, min(1.0, p * (2 - k)))
        out[nm] = run
    return out["a"], out["b"]


def contrast(a: pd.Series, b: pd.Series, label: str) -> dict:
    if not a.index.equals(b.index):
        sys.exit(f"{label}: subject indices differ")
    d = (a - b).to_numpy()
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(a.to_numpy(), b.to_numpy())
    dz = cohens_d_paired(a.to_numpy(), b.to_numpy())
    print(f"  {label}")
    print(f"    {a.name.split('_')[-1] if False else 'A'} {a.mean():.4f}  vs  B {b.mean():.4f}   "
          f"delta {d.mean()*100:+.2f} pp   95% BCa [{lo*100:+.2f}, {hi*100:+.2f}] pp   "
          f"p = {w.pvalue:.4g}   d = {dz:+.2f}   ({int((d>0).sum())}/{int((d<0).sum())} +/-)")
    return {"label": label, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta_pp": float(d.mean() * 100), "lo_pp": lo * 100, "hi_pp": hi * 100,
            "p_raw": float(w.pvalue), "d": dz, "n": len(d)}


def main() -> int:
    print("=" * 78)
    print("STAGE P-5: subset training vs channel dropout")
    print("=" * 78)

    sub, cd, na = load(SUBSET), load(CHANDROP), load(NOAUG)
    for nm, s in [("subset", sub), ("chandrop", cd), ("no aug", na)]:
        print(f"  {nm:<10} {s.name:<38} n = {len(s)}  mean F1 {s.mean():.4f}")
    ix = cd.index
    if not (sub.index.equals(ix) and na.index.equals(ix)):
        common = sub.index.intersection(cd.index).intersection(na.index)
        print(f"  WARNING: indices differ; intersecting to {len(common)} subjects")
        sub, cd, na = sub.loc[common], cd.loc[common], na.loc[common]
    n = len(sub)
    print(f"  paired on {n} subjects")

    print("\n--- Pereira subset enumeration used (P-5 arm) ---")
    print("  vocabulary: all C(9,2) = 36 ways to omit a channel pair; each subset retains 7 of 9")
    print("  channels (9 * 0.8 = 7.2 expected retained under chandrop p = 0.2). One subset drawn")
    print("  uniformly per training sample; the two omitted channels are zeroed.")

    print("\n--- contrasts ---")
    primary = contrast(sub, cd, "PRIMARY  : subset (A) vs channel dropout (B)")
    secondary = contrast(sub, na, "SECONDARY: subset (A) vs no augmentation (B)")
    ph_a, ph_b = holm2(primary["p_raw"], secondary["p_raw"])
    primary["p_holm"], secondary["p_holm"] = ph_a, ph_b
    print(f"\n  Holm family of two: primary Holm p = {ph_a:.4g}, secondary Holm p = {ph_b:.4g}")

    # uncorrected reference: subset vs no aug is 'secondary'; also chandrop vs no aug
    ref = contrast(cd, na, "REF (uncorrected): channel dropout vs no augmentation")

    # ---- pre-registered grid (section 5A.4) ----
    trainability_gap = (sub.mean() - na.mean()) * 100
    prim_sig = primary["p_holm"] < 0.05
    prim_mag = abs(primary["delta_pp"])
    sec_positive = (secondary["delta_pp"] > 0) and (secondary["p_holm"] < 0.05 or secondary["d"] >= 0.5)

    if trainability_gap < -TRAIN_X_PP:
        letter = "X"
        meaning = (f"Did not train. The subset arm sits {trainability_gap:+.2f} pp against the "
                   f"no-augmentation ResNet-SE baseline, past the {TRAIN_X_PP:.1f} pp trainability "
                   "floor. Exclude and report.")
    elif secondary["delta_pp"] <= 0:
        letter = "F"
        meaning = ("Subset fails. The secondary contrast (subset vs no augmentation) is null or "
                   "negative: this channel-level augmentation does not work on this task, which "
                   "bounds W-4's generalization usefully. Report it.")
    elif prim_sig and primary["delta_pp"] >= FLOOR_PP:
        letter = "B"
        meaning = ("Subset better. The primary contrast is Holm-significant and at least 2.0 pp in "
                   "favour of subset training. This is a finding about the class, not a problem: a "
                   "member that differs from channel dropout in a stated way (fixed subset "
                   "vocabulary rather than independent per-channel draw) does the job better, and "
                   "that difference is evidence about the operative property. Report what differs "
                   "and what it implies. Separately and for engineering reasons only, the deep "
                   "model of record does not change; keep the two apart in the write-up.")
    elif prim_sig and primary["delta_pp"] <= -FLOOR_PP:
        letter = "W"
        meaning = ("Subset worse. The primary contrast is Holm-significant and at least 2.0 pp "
                   "against subset training. Random per-sample masking beats a fixed subset "
                   "vocabulary: a positive specificity result, the first one channel dropout has. "
                   "Write it up as such.")
    elif not prim_sig and prim_mag < FLOOR_PP and sec_positive:
        letter = "M"
        meaning = ("Matches. The primary contrast is within 2.0 pp and not Holm-significant, and "
                   "the secondary is clearly positive. A third channel-structured perturbation, "
                   "this one from the literature, does the same work: W-4's generalization stops "
                   "resting only on a control the thesis invented. Section 4.8.2's claim "
                   "strengthens materially.")
    else:
        letter = "?"
        meaning = (f"Outside the pre-registered grid: primary delta {primary['delta_pp']:+.2f} pp, "
                   f"Holm p = {primary['p_holm']:.4g}; secondary delta {secondary['delta_pp']:+.2f} pp, "
                   f"d = {secondary['d']:+.2f}. Report the numbers, do not force a letter.")
    print("\n" + "=" * 78)
    print(f"P-5 OUTCOME {letter}: {meaning}")
    print("=" * 78)

    rows = pd.DataFrame({
        "subject": sub.index,
        "subset_f1": sub.values,
        "chandrop_f1": cd.reindex(sub.index).values,
        "noaug_f1": na.reindex(sub.index).values,
        "subset_minus_chandrop_pp": (sub - cd.reindex(sub.index)).values * 100,
        "subset_minus_noaug_pp": (sub - na.reindex(sub.index)).values * 100,
    }).round(5)
    rows.to_csv(OUT / "p5_subset_tests.csv", index=False)

    res = {"stage": "P-5", "n": n, "subset_mean_f1": float(sub.mean()),
           "chandrop_mean_f1": float(cd.mean()), "noaug_mean_f1": float(na.mean()),
           "primary": primary, "secondary": secondary, "chandrop_vs_noaug": ref,
           "trainability_gap_pp": float(trainability_gap),
           "outcome": letter, "outcome_meaning": meaning,
           "subset_vocabulary_size": 36}
    json.dump(res, open(OUT / "p5_outcome.json", "w"), indent=2)

    (OUT / "p5_verdict.md").write_text(
        f"# Stage P-5 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"Arm: subset mode on resnet_se, per-subject normalization, 250 ms, seed 42, every other "
        f"flag default, into results_p5_subset_resnet_se. Subset vocabulary: all C(9,2) = 36 "
        f"channel-omit pairs, each subset retaining 7 of 9 channels; one drawn uniformly per "
        f"training sample.\n\n"
        f"Arm means (n = {n}): subset {sub.mean():.4f}, channel dropout {cd.mean():.4f}, "
        f"no augmentation {na.mean():.4f}.\n\n"
        f"## Primary: subset vs channel dropout\n\n"
        f"delta {primary['delta_pp']:+.2f} pp, 95% BCa [{primary['lo_pp']:+.2f}, {primary['hi_pp']:+.2f}] pp, "
        f"raw p = {primary['p_raw']:.4g}, Holm p = {primary['p_holm']:.4g}, d = {primary['d']:+.2f}.\n\n"
        f"## Secondary: subset vs no augmentation\n\n"
        f"delta {secondary['delta_pp']:+.2f} pp, 95% BCa [{secondary['lo_pp']:+.2f}, {secondary['hi_pp']:+.2f}] pp, "
        f"raw p = {secondary['p_raw']:.4g}, Holm p = {secondary['p_holm']:.4g}, d = {secondary['d']:+.2f}.\n\n"
        f"Reference (uncorrected): channel dropout vs no augmentation delta {ref['delta_pp']:+.2f} pp, "
        f"raw p = {ref['p_raw']:.4g}, d = {ref['d']:+.2f}.\n\n"
        f"Trainability: subset arm is {trainability_gap:+.2f} pp against the no-augmentation "
        f"ResNet-SE baseline ({TRAIN_X_PP:.1f} pp floor for outcome X).\n\n"
        f"## FDR family contribution (plan section 5A.5 / 6.3)\n\n"
        f"Two new paired Wilcoxon tests: primary raw p = {primary['p_raw']:.4g}, "
        f"secondary raw p = {secondary['p_raw']:.4g}. Section 4.17 not edited.\n"
    )
    print(f"\nwrote {OUT/'p5_subset_tests.csv'}, {OUT/'p5_verdict.md'}, {OUT/'p5_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

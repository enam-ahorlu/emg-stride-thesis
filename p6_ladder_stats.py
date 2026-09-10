#!/usr/bin/env python3
"""
p6_ladder_stats.py
==================
Stage P-6 (EXPERIMENT_PLAN_CD_PARITY.md section 5B). The perturbation ladder:
decompose the channel-level perturbation by which moment of the multiplier it
acts on. No GPU here; reads the 40-fold runs written by run_p5_p6.sh.

Arms, all resnet_se, per-subject norm, 250 ms, seed 42:
  results_cnn_aug_resnet_se_none            no augmentation           mean 1.0, SD 0     (exists)
  results_cnn_aug_resnet_se_chandrop        channel dropout p = 0.2   mean 0.8, SD 0.40  (exists)
  results_p6_gainjitter_resnet_se_sd0.40    gain jitter sd = 0.40     mean 1.0, SD 0.40  (new)
  results_p6_mpchandrop_resnet_se_sd0.40    mean-preserving chandrop  mean 1.0, SD 0.40  (new)

Two contrasts, Holm family of two:
  C1  channel dropout vs mean-preserving chandrop   -> isolates the expected-activation shift
  C2  mean-preserving chandrop vs gain jitter       -> isolates the form (zeroing vs jitter)
Each arm is also reported against no augmentation, uncorrected, beside the family.
2.0 pp floor carried from W-5. Grid V / A / F / AF / R / X in section 5B.5.
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

NOAUG = "results_cnn_aug_resnet_se_none"
CHANDROP = "results_cnn_aug_resnet_se_chandrop"
GAINJIT = "results_p6_gainjitter_resnet_se_sd0.40"
MPCD = "results_p6_mpchandrop_resnet_se_sd0.40"
FLOOR_PP = 2.0
TRAIN_X_PP = 3.0


def load(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p} (has run_p5_p6.sh finished this arm?)")
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
    d = (a - b).to_numpy()
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(a.to_numpy(), b.to_numpy())
    dz = cohens_d_paired(a.to_numpy(), b.to_numpy())
    print(f"  {label:<52} {d.mean()*100:+6.2f} pp  95% BCa [{lo*100:+.2f}, {hi*100:+.2f}]  "
          f"p = {w.pvalue:.4g}  d = {dz:+.2f}  ({int((d>0).sum())}/{int((d<0).sum())} +/-)")
    return {"label": label, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta_pp": float(d.mean() * 100), "lo_pp": lo * 100, "hi_pp": hi * 100,
            "p_raw": float(w.pvalue), "d": dz, "n": len(d)}


def main() -> int:
    print("=" * 78)
    print("STAGE P-6: the perturbation ladder")
    print("=" * 78)

    na, cd, gj, mp = load(NOAUG), load(CHANDROP), load(GAINJIT), load(MPCD)
    arms = {"no aug": na, "channel dropout p=0.2": cd, "gain jitter sd=0.40": gj,
            "mean-preserving chandrop": mp}
    print("\n--- arm means (resnet_se, per-subject norm, 250 ms) ---")
    for nm, s in arms.items():
        print(f"  {nm:<26} {s.name:<40} n = {len(s)}  mean F1 {s.mean():.4f}  sd {s.std(ddof=1):.4f}")

    ix = na.index
    if not all(s.index.equals(ix) for s in (cd, gj, mp)):
        common = ix
        for s in (cd, gj, mp):
            common = common.intersection(s.index)
        print(f"  WARNING: indices differ; intersecting to {len(common)} subjects")
        na, cd, gj, mp = na.loc[common], cd.loc[common], gj.loc[common], mp.loc[common]
    n = len(na)
    print(f"  paired on {n} subjects")

    print("\n--- the two contrasts (Holm family of two) ---")
    c1 = contrast(cd, mp, "C1 channel dropout vs mean-preserving chandrop")
    c2 = contrast(mp, gj, "C2 mean-preserving chandrop vs gain jitter")
    ph1, ph2 = holm2(c1["p_raw"], c2["p_raw"])
    c1["p_holm"], c2["p_holm"] = ph1, ph2
    print(f"  Holm: C1 Holm p = {ph1:.4g}   C2 Holm p = {ph2:.4g}")
    print("  C1 isolates the expected-activation shift; C2 isolates the form (zeroing vs jitter).")

    print("\n--- each arm vs no augmentation (uncorrected, beside the family) ---")
    v_cd = contrast(cd, na, "channel dropout vs no aug")
    v_mp = contrast(mp, na, "mean-preserving chandrop vs no aug")
    v_gj = contrast(gj, na, "gain jitter vs no aug")

    # ---- pre-registered grid (section 5B.5) ----
    gaps = {"gain jitter sd=0.40": (gj.mean() - na.mean()) * 100,
            "mean-preserving chandrop": (mp.mean() - na.mean()) * 100,
            "channel dropout p=0.2": (cd.mean() - na.mean()) * 100}
    x_arms = [k for k, g in gaps.items() if g < -TRAIN_X_PP]
    r_arms = [k for k, g in gaps.items() if -TRAIN_X_PP <= g <= 0]

    c1_hit = (c1["p_holm"] < 0.05) and (abs(c1["delta_pp"]) >= FLOOR_PP)
    c2_hit = (c2["p_holm"] < 0.05) and (abs(c2["delta_pp"]) >= FLOOR_PP)

    if x_arms:
        letter = "X"
        meaning = (f"Did not train. Arm(s) {x_arms} sit more than {TRAIN_X_PP:.1f} pp below the "
                   "no-augmentation baseline. Exclude the lost cell(s) and report.")
    elif r_arms:
        letter = "R"
        meaning = (f"Reversal. Arm(s) {r_arms} fall at or below no augmentation. Something is wrong "
                   "with the arm, not the class. The P-6 multiplier gate (p6_multiplier_gate.py) "
                   "PASSED before the run (mean 1.00, SD 0.40, p' = 0.1379 derived from the "
                   "requested SD), so the multiplier itself is not the cause; investigate the arm "
                   "before reading the ladder.")
    elif c1_hit and c2_hit:
        letter = "AF"
        meaning = ("Both. The expected-activation shift (C1) and the form (C2) each move the score "
                   "by at least 2.0 pp with Holm significance. Report both and do not force a "
                   "single operative property.")
    elif c1_hit:
        letter = "A"
        meaning = (f"The activation shift matters. C1 is Holm-significant at {c1['delta_pp']:+.2f} pp: "
                   "reducing expected activation does part of the work, separately from the "
                   "variance. Channel dropout is then not merely a degenerate case of gain jitter "
                   "and W-4's conclusion narrows. The direction of the effect is stated above and "
                   "must be reported.")
    elif c2_hit:
        letter = "F"
        meaning = (f"The form matters. C2 is Holm-significant at {c2['delta_pp']:+.2f} pp: actual "
                   "zeroing differs from continuous jitter at matched moments. A specificity "
                   "result, the first one channel dropout has.")
    else:
        letter = "V"
        sub_floor_note = ""
        for tag, c in (("C1 (activation shift)", c1), ("C2 (form)", c2)):
            if c["p_holm"] < 0.05 and abs(c["delta_pp"]) < FLOOR_PP:
                sub_floor_note += (f" Note: {tag} is Holm-significant at {c['delta_pp']:+.2f} pp "
                                   f"(Holm p = {c['p_holm']:.3g}) but below the 2.0 pp materiality "
                                   "floor, so it is a detectable but immaterial effect and does "
                                   "not change the letter; report the number, not a claim.")
        meaning = ("Variance is what matters. Neither contrast reaches 2.0 pp with Holm "
                   "significance: the mean shift does nothing material and the form does nothing "
                   "material. What the class does is inject per-channel multiplicative variance. "
                   "This is the cleanest version of the finding and section 4.8.2 should state it "
                   "as the operative property." + sub_floor_note)
    print("\n" + "=" * 78)
    print(f"P-6 OUTCOME {letter}: {meaning}")
    print("=" * 78)

    # coherence check for the optional extension (section 5B.3): the class behaves
    # coherently if every arm clearly beats no augmentation and no arm reversed.
    coherent = (not x_arms and not r_arms
                and v_cd["delta_pp"] > 0 and v_mp["delta_pp"] > 0 and v_gj["delta_pp"] > 0
                and v_cd["p_raw"] < 0.05 and v_mp["p_raw"] < 0.05 and v_gj["p_raw"] < 0.05)
    print(f"\noptional extension (gain jitter sd 0.30 and 0.50): core arms coherent = {coherent}")
    print("  run run_p6_extension.sh only if this is True (plan section 5B.3).")

    rows = pd.DataFrame({
        "subject": na.index,
        "noaug_f1": na.values,
        "chandrop_f1": cd.reindex(na.index).values,
        "mpchandrop_f1": mp.reindex(na.index).values,
        "gainjitter_f1": gj.reindex(na.index).values,
        "C1_chandrop_minus_mpchandrop_pp": (cd.reindex(na.index) - mp.reindex(na.index)).values * 100,
        "C2_mpchandrop_minus_gainjitter_pp": (mp.reindex(na.index) - gj.reindex(na.index)).values * 100,
    }).round(5)
    rows.to_csv(OUT / "p6_ladder_tests.csv", index=False)

    res = {"stage": "P-6", "n": n,
           "arm_means": {nm: float(s.mean()) for nm, s in
                         [("no_aug", na), ("chandrop", cd), ("gainjitter", gj), ("mpchandrop", mp)]},
           "C1_chandrop_vs_mpchandrop": c1, "C2_mpchandrop_vs_gainjitter": c2,
           "vs_noaug": {"chandrop": v_cd, "mpchandrop": v_mp, "gainjitter": v_gj},
           "arm_gaps_vs_noaug_pp": gaps,
           "outcome": letter, "outcome_meaning": meaning,
           "extension_core_coherent": bool(coherent)}
    json.dump(res, open(OUT / "p6_outcome.json", "w"), indent=2)

    (OUT / "p6_verdict.md").write_text(
        f"# Stage P-6 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"New modes added to augment_batch (guarded, defaulted off, inertness PASS on all six "
        f"existing modes via p5p6_inertness.py): `subset` (P-5) and `mpchandrop` (P-6). "
        f"mpchandrop multiplies by Bernoulli(1 - p') / (1 - p') with "
        f"p' = SD^2 / (1 + SD^2) derived from the requested SD; at SD = 0.40, p' = 0.13793. "
        f"p6_multiplier_gate.py PASS: realized multiplier mean 1.00, SD 0.40.\n\n"
        f"## Arm means (resnet_se, per-subject norm, 250 ms, n = {n})\n\n"
        f"| arm | multiplier mean | multiplier SD | mean F1 |\n|---|---|---|---|\n"
        f"| no augmentation | 1.0 | 0 | {na.mean():.4f} |\n"
        f"| channel dropout p = 0.2 | 0.8 | 0.40 | {cd.mean():.4f} |\n"
        f"| mean-preserving chandrop | 1.0 | 0.40 | {mp.mean():.4f} |\n"
        f"| gain jitter sd = 0.40 | 1.0 | 0.40 | {gj.mean():.4f} |\n\n"
        f"## C1 channel dropout vs mean-preserving chandrop (isolates the expected-activation shift)\n\n"
        f"delta {c1['delta_pp']:+.2f} pp, 95% BCa [{c1['lo_pp']:+.2f}, {c1['hi_pp']:+.2f}] pp, "
        f"raw p = {c1['p_raw']:.4g}, Holm p = {c1['p_holm']:.4g}, d = {c1['d']:+.2f}.\n\n"
        f"## C2 mean-preserving chandrop vs gain jitter (isolates the form)\n\n"
        f"delta {c2['delta_pp']:+.2f} pp, 95% BCa [{c2['lo_pp']:+.2f}, {c2['hi_pp']:+.2f}] pp, "
        f"raw p = {c2['p_raw']:.4g}, Holm p = {c2['p_holm']:.4g}, d = {c2['d']:+.2f}.\n\n"
        f"## Each arm vs no augmentation (uncorrected)\n\n"
        f"- channel dropout: {v_cd['delta_pp']:+.2f} pp, raw p = {v_cd['p_raw']:.4g}, d = {v_cd['d']:+.2f}\n"
        f"- mean-preserving chandrop: {v_mp['delta_pp']:+.2f} pp, raw p = {v_mp['p_raw']:.4g}, d = {v_mp['d']:+.2f}\n"
        f"- gain jitter: {v_gj['delta_pp']:+.2f} pp, raw p = {v_gj['p_raw']:.4g}, d = {v_gj['d']:+.2f}\n\n"
        f"## Optional extension\n\n"
        f"Core arms coherent (every arm clearly beats no augmentation, no reversal): {coherent}. "
        f"Extension (gain jitter sd 0.30 and 0.50) {'is' if coherent else 'is NOT'} warranted.\n\n"
        f"## FDR family contribution (plan section 5B.6 / 6.3)\n\n"
        f"Two new paired Wilcoxon tests in the corrected family: C1 raw p = {c1['p_raw']:.4g}, "
        f"C2 raw p = {c2['p_raw']:.4g} (four with the extension). The arm-vs-no-aug contrasts are "
        f"reported uncorrected beside the family. Section 4.17 not edited.\n"
    )
    print(f"\nwrote {OUT/'p6_ladder_tests.csv'}, {OUT/'p6_verdict.md'}, {OUT/'p6_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

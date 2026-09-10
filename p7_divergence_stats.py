#!/usr/bin/env python3
"""
p7_divergence_stats.py
======================
Stage P-7 (EXPERIMENT_PLAN_P7_DIVERGENCE.md). What breaks channel dropout at
high dose: decompose the SD 0.50 divergence into the expected-activation shift
(D1) and the form (D2). No GPU here; reads the one 40-fold run from run_p7.sh.

Arms, all resnet_se, per-subject norm, 250 ms, seed 42:
  results_cd_rate_p0.5                     channel dropout p = 0.5   mean 0.5, SD 0.50  (0.8205)
  results_p7_mpchandrop_resnet_se_sd0.50   mean-preserving, p' = 0.2 mean 1.0, SD 0.50  (new)
  results_p6_gainjitter_resnet_se_sd0.50   gain jitter sd 0.50       mean 1.0, SD 0.50  (0.8486)
  results_cnn_aug_resnet_se_none           no augmentation                              (0.7822)

  D1 = mpchandrop(0.50) - channel dropout p0.5   isolates the activation shift at high dose
  D2 = gain jitter(0.50) - mpchandrop(0.50)      isolates the form at high dose
Paired /40, BCa 95%, paired Wilcoxon, paired Cohen's d, Holm family of two.

CRITERION (plan section 5): share of the observed +2.81 pp gap, NOT a 2.0 pp
floor. A component carries the divergence when it is Holm-significant AND holds
at least 60% of the gap. Gates 3 and 4 (completeness, additivity) are checked
here; gates 1 and 2 (multiplier, inertness) were run before the job. All four
gate results are printed in full.
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

NEW = "results_p7_mpchandrop_resnet_se_sd0.50"
CD05 = "results_cd_rate_p0.5"
GJ05 = "results_p6_gainjitter_resnet_se_sd0.50"
NOAUG = "results_cnn_aug_resnet_se_none"
# for the SD 0.40 restatement (section 9.3), era-consistent pairing = rate sweep
CD02 = "results_cd_rate_p0.2"
MP04 = "results_p6_mpchandrop_resnet_se_sd0.40"
GJ04 = "results_p6_gainjitter_resnet_se_sd0.40"
CD02_CORE = "results_cnn_aug_resnet_se_chandrop"  # P-6 core's chandrop comparator, noted only

OBS_GAP_SD50_PP = 2.81   # plan-stated observed gain jitter minus channel dropout at SD 0.50
ADDITIVITY_TOL_PP = 0.30
SHARE_THRESHOLD = 0.60
TRAIN_X_PP = 3.0


def load(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    df = df[[sc, "f1_macro"]].dropna().rename(columns={sc: "subject"})
    return df.set_index("subject")["f1_macro"].sort_index().rename(dirname)


def completeness(dirname: str) -> dict:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    return {"rows": len(df), "unique": int(df[sc].nunique()),
            "dups": bool(len(df) != df[sc].nunique()),
            "subjects": sorted(df[sc].astype(int).tolist())}


def paired(a: pd.Series, b: pd.Series, label: str) -> dict:
    d = (a - b).to_numpy()
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(a.to_numpy(), b.to_numpy())
    dz = cohens_d_paired(a.to_numpy(), b.to_numpy())
    return {"label": label, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta_pp": float(d.mean() * 100), "lo_pp": lo * 100, "hi_pp": hi * 100,
            "p_raw": float(w.pvalue), "d": dz, "n": len(d),
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}


def holm2(pa: float, pb: float) -> tuple[float, float]:
    order = sorted([("a", pa), ("b", pb)], key=lambda kv: kv[1])
    out, run = {}, 0.0
    for k, (nm, p) in enumerate(order):
        run = max(run, min(1.0, p * (2 - k)))
        out[nm] = run
    return out["a"], out["b"]


def show(r: dict) -> None:
    print(f"  {r['label']:<52} {r['delta_pp']:+6.2f} pp  95% BCa [{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]  "
          f"raw p = {r['p_raw']:.4g}  d = {r['d']:+.2f}  ({r['pos']}/{r['neg']} +/-)")


def main() -> int:
    print("=" * 78)
    print("STAGE P-7: what breaks channel dropout at high dose")
    print("=" * 78)

    mp, cd, gj, na = load(NEW), load(CD05), load(GJ05), load(NOAUG)

    # ================= GATES =================
    print("\n" + "-" * 78)
    print("SECTION 3 GATES (all four before any outcome letter)")
    print("-" * 78)
    print("Gate 1 (multiplier): run separately via p6_multiplier_gate.py 0.50 -> "
          "derived p' = 0.200000, realized multiplier mean 1.00132, SD 0.49901. PASS.")
    print("Gate 2 (inertness): p5p6_inertness.py --check p5p6_fp_before.json -> all six existing "
          "modes byte-identical, resnet/resnet_se init unchanged. PASS (regression check, no code "
          "changes in P-7).")

    c_new, c_cd, c_gj = completeness(NEW), completeness(CD05), completeness(GJ05)
    ids_match = c_new["subjects"] == c_cd["subjects"] == c_gj["subjects"]
    gate3 = (c_new["rows"] == 40 and c_new["unique"] == 40 and not c_new["dups"] and ids_match)
    print(f"\nGate 3 (completeness): {NEW} rows {c_new['rows']} / unique {c_new['unique']} / "
          f"dups {c_new['dups']}; subject id set identical to {CD05} and {GJ05}: {ids_match}  "
          f"-> {'PASS' if gate3 else 'FAIL'}")
    if not gate3:
        sys.exit("Gate 3 failed; stop and report.")

    ix = mp.index
    mp, cd, gj, na = mp.loc[ix], cd.loc[ix], gj.loc[ix], na.loc[ix]
    n = len(mp)

    d1 = paired(mp, cd, "D1 mean-preserving chandrop minus channel dropout p0.5")
    d2 = paired(gj, mp, "D2 gain jitter sd0.50 minus mean-preserving chandrop")
    obs_gap = paired(gj, cd, "observed gap: gain jitter sd0.50 minus channel dropout p0.5")
    add_sum = d1["delta_pp"] + d2["delta_pp"]
    add_resid = add_sum - obs_gap["delta_pp"]
    gate4 = abs(add_resid) <= ADDITIVITY_TOL_PP
    print(f"\nGate 4 (additivity): D1 + D2 = {d1['delta_pp']:+.2f} + {d2['delta_pp']:+.2f} = "
          f"{add_sum:+.2f} pp; observed gap (this pairing) = {obs_gap['delta_pp']:+.2f} pp "
          f"(plan-stated +{OBS_GAP_SD50_PP:.2f}); residual {add_resid:+.2f} pp, tol "
          f"+/-{ADDITIVITY_TOL_PP:.2f}  -> {'PASS' if gate4 else 'FAIL (outcome N)'}")

    # ================= trainability =================
    gain_new = (mp.mean() - na.mean()) * 100
    gain_cd = (cd.mean() - na.mean()) * 100
    train_gap = gain_new - gain_cd
    print(f"\nSection 4 trainability: mp gain vs no aug {gain_new:+.2f} pp against channel dropout "
          f"p0.5 gain {gain_cd:+.2f} pp -> {train_gap:+.2f} pp "
          f"({'X, lost cell' if train_gap < -TRAIN_X_PP else 'OK, within 3.0 pp'})")

    # ================= contrasts =================
    ph1, ph2 = holm2(d1["p_raw"], d2["p_raw"])
    d1["p_holm"], d2["p_holm"] = ph1, ph2
    G = obs_gap["delta_pp"]
    d1["share"] = d1["delta_pp"] / G if G != 0 else float("nan")
    d2["share"] = d2["delta_pp"] / G if G != 0 else float("nan")

    print("\n" + "-" * 78)
    print("D1 / D2 at SD 0.50 (Holm family of two)")
    print("-" * 78)
    for r in (d1, d2):
        show(r)
        print(f"      Holm p = {r['p_holm']:.4g}   share of the {G:+.2f} pp gap = {r['share']*100:+.1f}% "
              f"(60% threshold = {0.6*G:+.2f} pp)")
    show(obs_gap)

    print("\n--- each arm vs no augmentation (uncorrected) ---")
    for s, nm in [(cd, "channel dropout p0.5"), (mp, "mean-preserving chandrop 0.50"),
                  (gj, "gain jitter sd0.50")]:
        r = paired(s, na, f"{nm} vs no aug")
        show(r)

    # ================= SD 0.40 restatement (section 9.3) =================
    print("\n" + "-" * 78)
    print("SD 0.40 decomposition restated, era-consistent pairing (vs results_cd_rate_p0.2)")
    print("-" * 78)
    cd02, mp04, gj04 = load(CD02), load(MP04), load(GJ04)
    jx = cd02.index
    cd02, mp04, gj04 = cd02.loc[jx], mp04.loc[jx], gj04.loc[jx]
    d1_40 = paired(mp04, cd02, "D1(0.40) mp0.40 minus channel dropout p0.2 (rate sweep)")
    d2_40 = paired(gj04, mp04, "D2(0.40) gain jitter sd0.40 minus mp0.40")
    gap40 = paired(gj04, cd02, "observed gap at SD 0.40 (vs rate sweep p0.2)")
    for r in (d1_40, d2_40, gap40):
        show(r)
    d1_40["share"] = d1_40["delta_pp"] / gap40["delta_pp"]
    d2_40["share"] = d2_40["delta_pp"] / gap40["delta_pp"]
    print(f"  (P-6 core, noted only: D1 vs results_cnn_aug_resnet_se_chandrop and D2 vs mp gave "
          f"-0.45 and +1.28 pp; core used the 0.8395 arm, this table uses the 0.8376 rate-sweep arm.)")

    # ================= grid (section 5) =================
    d1_sig, d2_sig = d1["p_holm"] < 0.05, d2["p_holm"] < 0.05
    d1_carry = d1_sig and d1["share"] >= SHARE_THRESHOLD
    d2_carry = d2_sig and d2["share"] >= SHARE_THRESHOLD
    below_both = (mp.mean() < cd.mean()) and (mp.mean() < gj.mean())
    above_gj = (mp.mean() - gj.mean()) * 100 > 1.0

    if train_gap < -TRAIN_X_PP:
        letter, meaning = "X", ("Did not train. The mean-preserving arm's no-aug-relative gain is "
                                f"{train_gap:+.2f} pp against channel dropout p0.5's. Exclude, report "
                                "the lost cell; do not read D1 and D2.")
    elif below_both or above_gj:
        letter, meaning = "R", ("Reversal / not anticipated. The new arm "
                                + ("falls below both comparators" if below_both else
                                   f"sits {(mp.mean()-gj.mean())*100:+.2f} pp above gain jitter")
                                + ". Gate 1 passed (multiplier mean 1.00, SD 0.50), so the multiplier "
                                "is not the cause. Report the numbers, do not force a letter.")
    elif not gate4:
        letter, meaning = "N", ("Additivity failed. D1 + D2 does not reproduce the SD 0.50 gap "
                                f"(residual {add_resid:+.2f} pp). The decomposition does not hold at "
                                "this dose; something outside the design is producing the divergence. "
                                "A real result, reported as one.")
    elif d1_carry and not d2_carry:
        margin = d1["delta_pp"] - SHARE_THRESHOLD * G
        d2_note = ""
        if d2_sig:
            d2_note = (f" D2 is itself Holm-significant at {d2['delta_pp']:+.2f} pp "
                       f"({d2['share']*100:.0f}% of the gap): the discrete-zeroing form is a real "
                       "secondary contributor, it simply does not reach the 60% mark. Report both "
                       "shares.")
        near = ""
        if margin < 0.25:
            near = (f" This clears the 60% bar narrowly (D1 = {d1['delta_pp']:+.2f} pp against a "
                    f"{SHARE_THRESHOLD*G:+.2f} pp threshold, margin {margin:+.2f} pp); the letter is "
                    "M by the pre-registered rule but the split is close to B.")
        letter, meaning = "M", ("The mean shift carries it. What predominantly breaks channel "
                                "dropout at high dose is the collapse in expected activation rather "
                                "than the zeroing. At p = 0.5 half the signal is removed on average, "
                                "and that carries the majority of the cost. Channel dropout is a "
                                "variance injector with an unwanted side effect that grows with "
                                "rate; the transferable recommendation is the mean-preserving form."
                                + d2_note + near)
    elif d2_carry and not d1_carry:
        letter, meaning = "F", ("The form carries it. Discrete zeroing differs from continuous jitter "
                                "once enough channels are being removed, independently of the moments. "
                                "A genuine specificity result for channel dropout, the first the "
                                "thesis would have.")
    elif d1_sig and d2_sig and not d1_carry and not d2_carry:
        letter, meaning = "B", ("Both contribute. The divergence is jointly produced: D1 holds "
                                f"{d1['share']*100:+.0f}% and D2 {d2['share']*100:+.0f}% of the gap, "
                                "neither at the 60% mark. Report both shares, do not force one "
                                "property.")
    elif not d1_sig and not d2_sig:
        letter, meaning = "N", ("Neither component is Holm-significant. The decomposition does not "
                                "resolve the SD 0.50 divergence into these two properties at this "
                                "dose. A real result, reported as one.")
    else:
        letter, meaning = "?", (f"Outside the grid: D1 {d1['delta_pp']:+.2f} pp (Holm {d1['p_holm']:.3g}, "
                                f"share {d1['share']*100:+.0f}%), D2 {d2['delta_pp']:+.2f} pp "
                                f"(Holm {d2['p_holm']:.3g}, share {d2['share']*100:+.0f}%). Report the "
                                "numbers, do not force a letter.")
    print("\n" + "=" * 78)
    print(f"P-7 OUTCOME {letter}: {meaning}")
    print("=" * 78)

    # ================= outputs =================
    rows = pd.DataFrame({
        "subject": mp.index,
        "cd_p0.5_f1": cd.values, "mpchandrop_sd0.50_f1": mp.values,
        "gainjitter_sd0.50_f1": gj.values, "noaug_f1": na.values,
        "D1_mp_minus_cd_pp": (mp - cd).values * 100,
        "D2_gj_minus_mp_pp": (gj - mp).values * 100,
    }).round(5)
    rows.to_csv(OUT / "p7_divergence_tests.csv", index=False)

    res = {
        "stage": "P-7", "n": n,
        "gates": {
            "multiplier": {"derived_p_prime": 0.2, "realized_mean": 1.00132, "realized_sd": 0.49901,
                           "pass": True},
            "inertness": {"pass": True, "note": "regression check, no code changes"},
            "completeness": {"rows": c_new["rows"], "unique": c_new["unique"], "dups": c_new["dups"],
                             "ids_match_comparators": ids_match, "pass": bool(gate3)},
            "additivity": {"d1_pp": d1["delta_pp"], "d2_pp": d2["delta_pp"], "sum_pp": add_sum,
                           "observed_gap_pp": obs_gap["delta_pp"], "residual_pp": add_resid,
                           "tol_pp": ADDITIVITY_TOL_PP, "pass": bool(gate4)},
        },
        "trainability": {"gain_new_pp": gain_new, "gain_cd_pp": gain_cd, "train_gap_pp": train_gap,
                         "x": bool(train_gap < -TRAIN_X_PP)},
        "arm_means": {"cd_p0.5": float(cd.mean()), "mpchandrop_sd0.50": float(mp.mean()),
                      "gainjitter_sd0.50": float(gj.mean()), "noaug": float(na.mean())},
        "sd50": {"D1": d1, "D2": d2, "observed_gap": obs_gap, "gap_pp": G,
                 "share_threshold": SHARE_THRESHOLD},
        "sd40_restated": {"D1": d1_40, "D2": d2_40, "observed_gap": gap40},
        "outcome": letter, "outcome_meaning": meaning,
    }
    json.dump(res, open(OUT / "p7_outcome.json", "w"), indent=2)

    def row(tag, d1r, d2r, gapr):
        return (f"| {tag} | {d1r['delta_pp']:+.2f} ({d1r['share']*100:+.0f}%) | "
                f"{d1r['p_holm']:.4g} | {d2r['delta_pp']:+.2f} ({d2r['share']*100:+.0f}%) | "
                f"{d2r['p_holm'] if 'p_holm' in d2r else float('nan'):.4g} | {gapr['delta_pp']:+.2f} |")
    d1_40["p_holm"] = d1_40["p_raw"]; d2_40["p_holm"] = d2_40["p_raw"]  # SD0.40 not Holm-corrected here

    (OUT / "p7_verdict.md").write_text(
        f"# Stage P-7 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"## Section 3 gates (all four)\n\n"
        f"1. Multiplier: derived p' = 0.200000; realized multiplier mean 1.00132, SD 0.49901. PASS.\n"
        f"2. Inertness: all six existing augment modes byte-identical, resnet/resnet_se init "
        f"unchanged (regression check, no code changes in P-7). PASS.\n"
        f"3. Completeness: {c_new['rows']} rows, {c_new['unique']} unique subjects, dups "
        f"{c_new['dups']}; subject id set identical to results_cd_rate_p0.5 and "
        f"results_p6_gainjitter_resnet_se_sd0.50: {ids_match}. {'PASS' if gate3 else 'FAIL'}.\n"
        f"4. Additivity: D1 + D2 = {d1['delta_pp']:+.2f} + {d2['delta_pp']:+.2f} = {add_sum:+.2f} pp; "
        f"observed gap {obs_gap['delta_pp']:+.2f} pp; residual {add_resid:+.2f} pp against a "
        f"+/-{ADDITIVITY_TOL_PP:.2f} pp tolerance. {'PASS' if gate4 else 'FAIL, outcome N'}.\n\n"
        f"## Trainability (section 4)\n\n"
        f"mean-preserving arm gain vs no aug {gain_new:+.2f} pp; channel dropout p0.5 gain "
        f"{gain_cd:+.2f} pp; difference {train_gap:+.2f} pp "
        f"({'outcome X' if train_gap < -TRAIN_X_PP else 'within the 3.0 pp bound'}).\n\n"
        f"## Decomposition, both doses (n = 40, share of each dose's own observed gap)\n\n"
        f"| dose | D1 activation shift (mp minus channel dropout) | D1 Holm p | "
        f"D2 form (gain jitter minus mp) | D2 Holm p | observed gap |\n"
        f"|---|---|---|---|---|---|\n"
        + row("SD 0.50 (channel dropout p0.5)", d1, d2, obs_gap) + "\n"
        + row("SD 0.40 (channel dropout p0.2, rate sweep)", d1_40, d2_40, gap40) + "\n\n"
        f"D1 and D2 shares sum to the observed gap by construction (additivity gate). At SD 0.40 "
        f"the P-6 core figures against results_cnn_aug_resnet_se_chandrop (0.8395) were D1 -0.45 pp "
        f"and D2 +1.28 pp; this table pairs against the era-consistent rate-sweep arm (0.8376) so "
        f"both dose rows use the same comparator family.\n\n"
        f"## Arm means (resnet_se, per-subject norm, 250 ms)\n\n"
        f"- no augmentation {na.mean():.4f}\n- channel dropout p = 0.5 {cd.mean():.4f}\n"
        f"- mean-preserving chandrop, p' = 0.2, SD 0.50 {mp.mean():.4f}\n"
        f"- gain jitter sd = 0.50 {gj.mean():.4f}\n\n"
        f"## What this cannot say (section 6)\n\n"
        f"Two doses is not a dose-response curve for either property. Nothing here bears on section "
        f"4.8.2's primary finding (the 5.6-fold occlusion-cost reduction) or on whether channel "
        f"dropout works; it bears only on the top of the rate range. The deep model of record stays "
        f"the channel-dropout resnet_se at 0.840.\n\n"
        f"## FDR family (section 8)\n\n"
        f"Two new paired Wilcoxon tests: D1 raw p = {d1['p_raw']:.4g}, D2 raw p = {d2['p_raw']:.4g}. "
        f"Section 4.17 not edited.\n"
    )
    print(f"\nwrote {OUT/'p7_divergence_tests.csv'}, {OUT/'p7_verdict.md'}, {OUT/'p7_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

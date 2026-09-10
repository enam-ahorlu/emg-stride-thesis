#!/usr/bin/env python3
"""
p9_attenuation_stats.py
=======================
Stage P-9 (EXPERIMENT_PLAN_LOCUS.md section 3, amended 3 September 2026 to FOUR
arms on both backbones). NO GPU here; reads four re-run instrumented arms.

  A1  results_p9_atten_resnet_noaug        resnet     none      (reproduction of g3_noaug_instr)
  A2  results_p9_atten_resnet_chandrop     resnet     chandrop  (reproduction of cd_resnet_nose_chandrop)
  A3  results_p9_atten_resnet_se_noaug     resnet_se  none      (FIRST measurement, never a reproduction)
  A4  results_p9_atten_resnet_se_chandrop  resnet_se  chandrop  (FIRST measurement, the model of record)

Reported, in order:
  1. section 3.2 gate: attenuation alpha = 0 reproduces same-run occlusion, all 4 arms.
  2. section 3.5a reproduction check: A1 and A2 summed occlusion cost vs the
     published 84.2 pp and 15.0 pp, 3.0 pp gate per arm. A3/A4 are new numbers.
  3. the A1/A2 reproduction spread, which IS the yardstick, stated BEFORE any
     cross-backbone comparison.
  4. backbone question (section 3.7): letter S / D / X.
  5. transfer question (section 3.7), per backbone: letter T / N / U / X each.
  6. per-subject coupling: retired as underpowered by design (P-8: ~151 subjects
     at rho = -0.226, ~351 at rho = +0.149), reported, not run.

Section 4.8.2 keeps quoting the original 84.2 pp and 15.0 pp; P-9 does not restate
the headline. Every P-9 statistic is computed within the new arms only, except
the section 3.5a reproduction check itself.
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

OUT = ROOT / "results_locus"
OUT.mkdir(exist_ok=True)

SEED = 42
N_PERM = 10_000
N_CH = 9
ALPHAS_DESC = [1.0, 0.75, 0.5, 0.25, 0.0]

A1 = "results_p9_atten_resnet_noaug"
A2 = "results_p9_atten_resnet_chandrop"
A3 = "results_p9_atten_resnet_se_noaug"
A4 = "results_p9_atten_resnet_se_chandrop"

# section 3.5a targets, precise values from the original instrumented runs on disk
REPRO_TARGET_A1_PP = 84.211      # results_g3_noaug_instr summed occlusion cost mean
REPRO_TARGET_A2_PP = 15.029      # results_cd_resnet_nose_chandrop summed occlusion cost mean
REPRO_F1_A1 = 0.7684
REPRO_F1_A2 = 0.8251
REPRO_ORIG_REDUCTION = REPRO_TARGET_A1_PP / REPRO_TARGET_A2_PP   # 5.60x
REPRO_GATE_PP = 3.0

P8_TESTA_AGREEMENT = 0.1668
P2_RESNET_OCC_AGREEMENT = 0.3053
CENSOR_LIMIT = 0.30
DROP_PRIMARY = 2.0
DROP_FALLBACK = 1.0


# ------------------------------------------------------------------ loaders
def load_atten(dirname: str) -> pd.DataFrame:
    p = ROOT / dirname / "instr" / "attenuation.csv"
    if not p.exists():
        sys.exit(f"missing {p} (has run_p9_p10.sh finished this arm?)")
    df = pd.read_csv(p)
    if df.duplicated(["subject", "channel", "alpha"]).any():
        sys.exit(f"{p} has duplicate (subject, channel, alpha) rows; a --resume double-appended")
    if df["subject"].nunique() != 40:
        sys.exit(f"{p} covers {df['subject'].nunique()} subjects, expected 40")
    return df


def load_occ(dirname: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / dirname / "instr" / "occlusion.csv")


def alpha0_gate(dirname: str) -> tuple[bool, float]:
    a = load_atten(dirname)
    o = load_occ(dirname)
    a0 = a[a.alpha == 0.0].set_index(["subject", "channel"])["drop_pp"]
    oo = o.set_index(["subject", "channel"])["drop_pp"]
    common = a0.index.intersection(oo.index)
    return float((a0.loc[common] - oo.loc[common]).abs().max()) < 1e-6, \
        float((a0.loc[common] - oo.loc[common]).abs().max())


def summed_cost_by_subject(dirname: str) -> pd.Series:
    """alpha = 0 summed single-electrode occlusion cost, per subject."""
    a = load_atten(dirname)
    return a[a.alpha == 0.0].groupby("subject")["drop_pp"].sum().sort_index()


def f1_mean(dirname: str) -> float:
    return float(pd.read_csv(ROOT / dirname / "cnn_arch_summary.csv")["f1_macro_mean"].iloc[0])


# ------------------------------------------------------------------ threshold + transfer helpers
def reliance_threshold(df: pd.DataFrame, drop_crit: float) -> tuple[pd.DataFrame, float]:
    rows = []
    for (s, c), g in df.groupby(["subject", "channel"]):
        g = g.set_index("alpha").reindex(ALPHAS_DESC)
        d = g["drop_pp"].to_numpy()
        thr = np.nan
        for i in range(1, len(ALPHAS_DESC)):
            if d[i] >= drop_crit:
                a_hi, a_lo, d_hi, d_lo = ALPHAS_DESC[i - 1], ALPHAS_DESC[i], d[i - 1], d[i]
                thr = a_lo if d_lo == d_hi else a_hi + (a_lo - a_hi) * (drop_crit - d_hi) / (d_lo - d_hi)
                break
        rows.append({"subject": int(s), "channel": int(c), "threshold": thr})
    wide = pd.DataFrame(rows).pivot(index="subject", columns="channel", values="threshold").sort_index()
    return wide, float(wide.isna().to_numpy().mean())


def rank_rows(mat):
    return np.apply_along_axis(stats.rankdata, 1, mat)


def mean_pairwise_spearman(rank_mat):
    r = np.asarray(stats.spearmanr(rank_mat, axis=1).statistic, float)
    iu = np.triu_indices_from(r, k=1)
    return float(np.nanmean(r[iu]))


def per_subject_consistency(rank_mat):
    r = np.asarray(stats.spearmanr(rank_mat, axis=1).statistic, float)
    np.fill_diagonal(r, np.nan)
    return np.nanmean(r, axis=1)


def within_shuffle_null(mat, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    n, c = mat.shape
    return np.array([mean_pairwise_spearman(rank_rows(np.stack([mat[i, rng.permutation(c)]
                    for i in range(n)]))) for _ in range(n_perm)])


def subject_swap_p(cd_r, base_r, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs = mean_pairwise_spearman(cd_r) - mean_pairwise_spearman(base_r)
    n, hits = cd_r.shape[0], 1
    for _ in range(n_perm):
        sw = rng.random(n) < 0.5
        a = np.where(sw[:, None], base_r, cd_r)
        b = np.where(sw[:, None], cd_r, base_r)
        if abs(mean_pairwise_spearman(a) - mean_pairwise_spearman(b)) >= abs(obs) - 1e-15:
            hits += 1
    return hits / (n_perm + 1)


def shrink_control(base_wide, cd_wide, n_real=200, seed=SEED):
    b = np.nan_to_num(base_wide.to_numpy(), nan=0.0)
    c = np.nan_to_num(cd_wide.to_numpy(), nan=0.0)
    k = c.sum() / b.sum() if b.sum() > 0 else 1.0
    shrunk = b * k
    target_zero = float((c <= 1e-9).mean())
    rng = np.random.default_rng(seed)
    lo, hi = 1e-6, 5.0 * (abs(shrunk).mean() + 1e-6)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        zf = float(np.mean([((shrunk + rng.normal(0, mid, shrunk.shape)) <= 1e-9).mean() for _ in range(40)]))
        lo, hi = (mid, hi) if zf < target_zero else (lo, mid)
    sigma = 0.5 * (lo + hi)
    rng = np.random.default_rng(seed + 1)
    ag = [mean_pairwise_spearman(rank_rows(np.clip(shrunk + rng.normal(0, sigma, shrunk.shape), 0, 1)))
          for _ in range(n_real)]
    return {"shrink_factor": k, "noise_sigma": sigma,
            "sim_agreement_mean": float(np.mean(ag)), "sim_agreement_sd": float(np.std(ag, ddof=1))}


def boot_ratio_ci(num: np.ndarray, den: np.ndarray, n_boot=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(num)
    b = []
    for _ in range(n_boot):
        ii = rng.integers(0, n, n)
        b.append(num[ii].mean() / den[ii].mean())
    return float(num.mean() / den.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def transfer_within_backbone(base_dir, cd_dir, label, drop_crit):
    a_base, a_cd = load_atten(base_dir), load_atten(cd_dir)
    na_w, na_c = reliance_threshold(a_base, drop_crit)
    cd_w, cd_c = reliance_threshold(a_cd, drop_crit)
    na_f, cd_f = na_w.fillna(0.0), cd_w.fillna(0.0)
    idx = na_f.index.intersection(cd_f.index)
    na_f, cd_f = na_f.loc[idx], cd_f.loc[idx]
    na_r, cd_r = rank_rows(na_f.to_numpy()), rank_rows(cd_f.to_numpy())
    na_ag, cd_ag = mean_pairwise_spearman(na_r), mean_pairwise_spearman(cd_r)
    na_null = within_shuffle_null(na_f.to_numpy(), seed=SEED)
    cd_null = within_shuffle_null(cd_f.to_numpy(), seed=SEED + 7)
    na_p = (int(np.sum(np.abs(na_null) >= abs(na_ag) - 1e-15)) + 1) / (N_PERM + 1)
    cd_p = (int(np.sum(np.abs(cd_null) >= abs(cd_ag) - 1e-15)) + 1) / (N_PERM + 1)
    na_cons, cd_cons = per_subject_consistency(na_r), per_subject_consistency(cd_r)
    d = cd_cons - na_cons
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(cd_cons, na_cons)
    p_swap = subject_swap_p(cd_r, na_r)
    sc = shrink_control(na_f, cd_f)
    # bootstrap CI of the un-augmented arm's agreement, for the P-8 benchmark
    rng = np.random.default_rng(SEED)
    arr, n = na_f.to_numpy(), na_f.shape[0]
    boot = [mean_pairwise_spearman(rank_rows(arr[rng.integers(0, n, n)])) for _ in range(5000)]
    na_lo, na_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    censor_x = cd_c > CENSOR_LIMIT
    cd_above_null = cd_p < 0.05 and cd_ag > float(np.quantile(cd_null, 0.975))
    sig_below = (float(w.pvalue) < 0.05 or p_swap < 0.05) and cd_ag < na_ag
    if censor_x:
        letter, meaning = "U", (f"censoring on the augmented arm is {cd_c:.1%} even at the "
                                f"{drop_crit:.1f} pp criterion, above the {CENSOR_LIMIT:.0%} limit; "
                                "a third measure has failed. Do not commission a fourth.")
    elif not cd_above_null:
        letter, meaning = "U", ("the augmented arm's threshold-ordering agreement is "
                                "indistinguishable from its within-subject shuffle null; the "
                                "measure has no resolution here. A third measure has failed.")
    elif sig_below:
        letter, meaning = "N", ("above its null but significantly below the un-augmented arm's: "
                                "reliance is flatter AND more idiosyncratic under channel dropout.")
    else:
        letter, meaning = "T", ("above its null and not significantly below the un-augmented arm's: "
                                "reliance is flatter and still shared. Closes the Section 4.8.2 "
                                "transfer question two measures could not.")

    over = na_lo > P8_TESTA_AGREEMENT
    under = na_hi < P8_TESTA_AGREEMENT
    md = ("over_commits" if over else "under_structured" if under else "tracks_data")
    return {
        "label": label, "drop_crit": drop_crit,
        "censoring": {"noaug": na_c, "chandrop": cd_c},
        "agreement": {"noaug": na_ag, "chandrop": cd_ag, "noaug_boot95": [na_lo, na_hi]},
        "null_p": {"noaug": na_p, "chandrop": cd_p},
        "null_975": {"chandrop": float(np.quantile(cd_null, 0.975))},
        "paired_consistency": {"mean_delta": float(d.mean()), "bca_lo": lo, "bca_hi": hi,
                               "p_raw": float(w.pvalue), "d": cohens_d_paired(cd_cons, na_cons),
                               "pos": int((d > 0).sum()), "neg": int((d < 0).sum())},
        "subject_swap_p": p_swap, "shrink_control": sc,
        "model_vs_data": md, "letter": letter, "meaning": meaning,
        "thresholds": {"noaug": na_w, "chandrop": cd_w},
    }


# ------------------------------------------------------------------ main
def main() -> int:
    print("=" * 78)
    print("STAGE P-9: graded attenuation, FOUR arms, both backbones (amended)")
    print("=" * 78)

    # ---- 1. section 3.2 gate, all four arms ----
    print("\n--- gate (section 3.2 / 3.7 X): attenuation alpha = 0 == same-run occlusion ---")
    gates = {a: alpha0_gate(a) for a in (A1, A2, A3, A4)}
    for a, (ok, md) in gates.items():
        print(f"  {a:<38} max|diff| {md:.2e}  -> {'PASS' if ok else 'FAIL'}")
    gate_ok = all(ok for ok, _ in gates.values())
    print("  RNG-inertness assertions around instrument_fold run every fold; the runs completed, "
          "so they held.")
    if not gate_ok:
        print("\nP-9 TRANSFER OUTCOME X (both backbones): alpha = 0 does not reproduce same-run "
              "occlusion. Instrumentation problem, not science. Stop.")
        json.dump({"stage": "P-9", "transfer_outcome": "X", "backbone_outcome": "n/a",
                   "reason": "section 3.2 gate failed"}, open(OUT / "p9_outcome.json", "w"), indent=2)
        return 0

    # ---- 2. section 3.5a reproduction check (A1, A2 only) ----
    print("\n" + "-" * 78)
    print("SECTION 3.5a REPRODUCTION CHECK -- A1 and A2 only, reported before any cross-backbone")
    print("comparison. Section 4.8.2 keeps its original 84.2 pp -> 15.0 pp; P-9 does not restate it.")
    print("-" * 78)
    c1 = summed_cost_by_subject(A1)
    c2 = summed_cost_by_subject(A2)
    c3 = summed_cost_by_subject(A3)
    c4 = summed_cost_by_subject(A4)
    m1, m2, m3, m4 = c1.mean(), c2.mean(), c3.mean(), c4.mean()
    repro = []
    for tag, m, tgt, f1d, tgt_f1 in [("A1 resnet none", m1, REPRO_TARGET_A1_PP, f1_mean(A1), REPRO_F1_A1),
                                     ("A2 resnet chandrop", m2, REPRO_TARGET_A2_PP, f1_mean(A2), REPRO_F1_A2)]:
        diff = m - tgt
        within = abs(diff) <= REPRO_GATE_PP
        repro.append({"arm": tag, "p9_cost_pp": float(m), "original_pp": tgt, "diff_pp": float(diff),
                      "within_gate": bool(within), "p9_f1": f1d, "original_f1": tgt_f1,
                      "f1_diff_pp": (f1d - tgt_f1) * 100})
        print(f"  {tag}: summed occlusion cost P-9 {m:.2f} pp vs original {tgt:.2f} pp  "
              f"({diff:+.2f} pp, {'within' if within else 'OUTSIDE'} the {REPRO_GATE_PP:.1f} pp gate); "
              f"40-fold F1 {f1d:.4f} vs {tgt_f1:.4f} ({(f1d - tgt_f1)*100:+.2f} pp)")
    repro_pass = all(r["within_gate"] for r in repro)

    # ---- 3. the A1/A2 reproduction spread = the yardstick ----
    spread_pp = max(abs(m1 - REPRO_TARGET_A1_PP), abs(m2 - REPRO_TARGET_A2_PP))
    R_resnet, Rr_lo, Rr_hi = boot_ratio_ci(c1.to_numpy(), c2.to_numpy())
    R_se, Rs_lo, Rs_hi = boot_ratio_ci(c3.reindex(c4.index).to_numpy(), c4.to_numpy())
    reduction_spread = abs(R_resnet - REPRO_ORIG_REDUCTION)
    print("\n" + "-" * 78)
    print("A1/A2 REPRODUCTION SPREAD -- the yardstick, stated before the cross-backbone comparison")
    print("-" * 78)
    print(f"  summed-cost spread: max(|A1 - 84.21|, |A2 - 15.03|) = {spread_pp:.2f} pp")
    print(f"  reduction factor: original {REPRO_ORIG_REDUCTION:.2f}x  |  P-9 resnet (A1/A2) "
          f"{R_resnet:.2f}x [95% {Rr_lo:.2f}, {Rr_hi:.2f}]  ->  factor moved {reduction_spread:.2f}x "
          "on a same-config re-run")

    # ---- 4. A3/A4: first measurement on resnet_se (NOT a reproduction) ----
    print("\n--- A3 / A4: first instrumented measurement on resnet_se (new numbers, not reproductions) ---")
    print(f"  A3 resnet_se none    : summed occlusion cost {m3:.2f} pp   40-fold F1 {f1_mean(A3):.4f}")
    print(f"  A4 resnet_se chandrop: summed occlusion cost {m4:.2f} pp   40-fold F1 {f1_mean(A4):.4f}  "
          "(the model of record)")
    print(f"  resnet_se reduction (A3/A4): {R_se:.2f}x [95% {Rs_lo:.2f}, {Rs_hi:.2f}]")

    # per-subject paired backbone difference in CD-induced absolute cost reduction
    red_resnet = (c1 - c2).reindex(c1.index)
    red_se = (c3.reindex(c4.index) - c4).reindex(c4.index)
    ii = red_resnet.index.intersection(red_se.index)
    bdiff = (red_se.loc[ii] - red_resnet.loc[ii]).to_numpy()
    blo, bhi = bca_ci(bdiff)
    bw = stats.wilcoxon(red_se.loc[ii].to_numpy(), red_resnet.loc[ii].to_numpy())
    print(f"  per-subject CD-induced reduction: resnet {red_resnet.mean():.2f} pp, resnet_se "
          f"{red_se.mean():.2f} pp; paired backbone difference {bdiff.mean():+.2f} pp "
          f"95% BCa [{blo:+.2f}, {bhi:+.2f}]  p = {bw.pvalue:.4g}")

    # ---- backbone grid letter (section 3.7) ----
    if not repro_pass:
        bb_letter = "X"
        bb_meaning = ("No yardstick. A1/A2 fail the 3.0 pp reproduction gate, so the occlusion "
                      "statistic is less reproducible than assumed and no cross-backbone comparison "
                      "may be read. This is itself worth reporting and it bounds every occlusion "
                      "claim in the thesis.")
    else:
        # "matches to within the A1/A2 reproduction spread" = the two reduction
        # estimates are statistically consistent given how noisy a same-config
        # re-run is (A1/A2 moved the factor by `reduction_spread`), AND the
        # per-subject paired backbone difference is not distinguishable from zero.
        factors_consistent = not (Rs_hi < Rr_lo or Rs_lo > Rr_hi)   # bootstrap CIs overlap
        factor_within_spread = abs(R_se - R_resnet) <= max(reduction_spread, 0.30)
        paired_null = (blo <= 0.0 <= bhi) and (bw.pvalue >= 0.05)
        if factors_consistent and factor_within_spread and paired_null:
            bb_letter = "S"
            bb_meaning = ("Split vindicated. The resnet_se occlusion reduction "
                          f"({R_se:.2f}x, 95% [{Rs_lo:.2f}, {Rs_hi:.2f}]) matches the SE-free one "
                          f"({R_resnet:.2f}x, 95% [{Rr_lo:.2f}, {Rr_hi:.2f}]) to within the A1/A2 "
                          f"reproduction spread: |difference| {abs(R_se - R_resnet):.2f}x against a "
                          f"{reduction_spread:.2f}x shift on a pure re-run, overlapping bootstrap "
                          f"intervals, and a per-subject paired backbone difference of "
                          f"{bdiff.mean():+.2f} pp (95% BCa [{blo:+.2f}, {bhi:+.2f}], p = "
                          f"{bw.pvalue:.3g}) that is not distinguishable from zero. The mechanism "
                          "findings transfer across backbones, measured rather than assumed. "
                          "Section 4.8.2 may keep quoting either figure provided it names the "
                          "backbone, and the G1 warrant is retrospectively supported on a mechanism "
                          "quantity.")
        else:
            bb_letter = "D"
            bb_meaning = ("Backbone matters. The two reductions differ by more than the A1/A2 "
                          f"reproduction spread (resnet {R_resnet:.2f}x vs resnet_se {R_se:.2f}x; "
                          f"paired backbone difference in absolute reduction {bdiff.mean():+.2f} pp, "
                          f"95% BCa [{blo:+.2f}, {bhi:+.2f}]). Section 4.8.2's headline belongs to "
                          "resnet_se, the number of record changes to the A3/A4 figure, and the "
                          "SE-free result becomes the companion. The G1 null did not license the "
                          "transfer.")
    print("\n" + "=" * 78)
    print(f"P-9 BACKBONE OUTCOME {bb_letter}: {bb_meaning}")
    print("=" * 78)

    # ---- 5. transfer question, per backbone ----
    print("\n" + "-" * 78)
    print("TRANSFER QUESTION, PER BACKBONE (section 3.7)")
    print("-" * 78)
    per = {}
    for label, base, cd in [("resnet", A1, A2), ("resnet_se", A3, A4)]:
        r = transfer_within_backbone(base, cd, label, DROP_PRIMARY)
        crit = DROP_PRIMARY
        if r["censoring"]["chandrop"] > CENSOR_LIMIT:
            print(f"  [{label}] augmented censoring {r['censoring']['chandrop']:.1%} > "
                  f"{CENSOR_LIMIT:.0%}; retrying at the pre-committed {DROP_FALLBACK:.1f} pp criterion")
            r = transfer_within_backbone(base, cd, label, DROP_FALLBACK)
            crit = DROP_FALLBACK
        per[label] = r
        pc = r["paired_consistency"]
        print(f"\n  [{label}] criterion {crit:.1f} pp | censoring no-aug {r['censoring']['noaug']:.1%}, "
              f"chandrop {r['censoring']['chandrop']:.1%}")
        print(f"  [{label}] threshold-ordering agreement: no-aug {r['agreement']['noaug']:+.4f} "
              f"(95% boot [{r['agreement']['noaug_boot95'][0]:+.4f}, {r['agreement']['noaug_boot95'][1]:+.4f}], "
              f"null p {r['null_p']['noaug']:.4g}); chandrop {r['agreement']['chandrop']:+.4f} "
              f"(null p {r['null_p']['chandrop']:.4g})")
        print(f"  [{label}] paired consistency (chandrop - no-aug): {pc['mean_delta']:+.4f} "
              f"95% BCa [{pc['bca_lo']:+.4f}, {pc['bca_hi']:+.4f}] p = {pc['p_raw']:.4g} "
              f"d = {pc['d']:+.2f}; subject-swap p = {r['subject_swap_p']:.4g}")
        print(f"  [{label}] shrink control simulated agreement {r['shrink_control']['sim_agreement_mean']:+.4f}")
        bench = P2_RESNET_OCC_AGREEMENT if label == "resnet" else None
        extra = f" (P-2 occlusion-ordering on this arm family was {bench:+.3f})" if bench else ""
        print(f"  [{label}] model vs data (P-8 Test A {P8_TESTA_AGREEMENT:+.3f}): {r['model_vs_data']}{extra}")
        print(f"  [{label}] TRANSFER LETTER {r['letter']}: {r['meaning']}")

    disagree = per["resnet"]["letter"] != per["resnet_se"]["letter"]
    if disagree:
        print(f"\n  The two backbones DISAGREE on the transfer letter (resnet {per['resnet']['letter']}, "
              f"resnet_se {per['resnet_se']['letter']}). Reporting both, forcing neither; this is "
              "itself evidence on the backbone question.")

    # ---- 6. per-subject coupling: retired ----
    print("\n--- per-subject coupling (P-1b / P-1c analogue): RETIRED, underpowered by design ---")
    print("  P-8 measured this on per-channel LDA informativeness (non-saturating). At the observed "
          "effect sizes it needs about 151 subjects at rho = -0.226 and 351 at rho = +0.149. The "
          "binding constraint is cohort size, not measurement, so the threshold-change vs "
          "F1-change correlation is reported as underpowered by design and NOT run.")

    # ---- outputs ----
    long = []
    for label, r in per.items():
        for arm, w in [("noaug", r["thresholds"]["noaug"]), ("chandrop", r["thresholds"]["chandrop"])]:
            for s in w.index:
                for c in range(N_CH):
                    v = w.loc[s, c]
                    long.append({"backbone": label, "arm": arm, "subject": int(s), "channel": c,
                                 "threshold": (None if pd.isna(v) else float(v)),
                                 "censored": bool(pd.isna(v))})
    pd.DataFrame(long).to_csv(OUT / "p9_attenuation_thresholds.csv", index=False)

    res = {
        "stage": "P-9", "arms": 4, "scope": "transfer only; per-subject coupling retired (underpowered by design)",
        "section_3_2_gate": {a: {"maxdiff": md, "pass": ok} for a, (ok, md) in gates.items()},
        "reproduction_check_3_5a": {"gate_pp": REPRO_GATE_PP, "pass": repro_pass, "arms": repro},
        "a1_a2_reproduction_spread_pp": spread_pp,
        "reduction_factors": {"original": REPRO_ORIG_REDUCTION,
                              "resnet_A1_A2": {"est": R_resnet, "ci": [Rr_lo, Rr_hi]},
                              "resnet_se_A3_A4": {"est": R_se, "ci": [Rs_lo, Rs_hi]}},
        "summed_cost_pp": {"A1": float(m1), "A2": float(m2), "A3": float(m3), "A4": float(m4)},
        "f1_mean": {"A1": f1_mean(A1), "A2": f1_mean(A2), "A3": f1_mean(A3), "A4": f1_mean(A4)},
        "paired_backbone_reduction_diff_pp": {"mean": float(bdiff.mean()), "bca_lo": blo, "bca_hi": bhi,
                                              "p_raw": float(bw.pvalue)},
        "backbone_outcome": bb_letter, "backbone_meaning": bb_meaning,
        "transfer": {label: {k: v for k, v in r.items() if k != "thresholds"} for label, r in per.items()},
        "transfer_letters": {label: r["letter"] for label, r in per.items()},
        "transfer_backbones_disagree": bool(disagree),
        "coupling": "retired; underpowered by design (~151 subjects at rho -0.226, ~351 at +0.149)",
    }
    json.dump(res, open(OUT / "p9_outcome.json", "w"), indent=2)

    def tline(label, r):
        pc = r["paired_consistency"]
        return (f"### {label} - transfer letter {r['letter']}\n\n{r['meaning']}\n\n"
                f"- criterion {r['drop_crit']:.1f} pp; censoring no-aug {r['censoring']['noaug']:.1%}, "
                f"chandrop {r['censoring']['chandrop']:.1%}\n"
                f"- threshold-ordering agreement: no-aug {r['agreement']['noaug']:+.4f} "
                f"(95% boot [{r['agreement']['noaug_boot95'][0]:+.4f}, {r['agreement']['noaug_boot95'][1]:+.4f}], "
                f"null p {r['null_p']['noaug']:.4g}); chandrop {r['agreement']['chandrop']:+.4f} "
                f"(null p {r['null_p']['chandrop']:.4g})\n"
                f"- paired consistency (chandrop - no-aug): {pc['mean_delta']:+.4f}, 95% BCa "
                f"[{pc['bca_lo']:+.4f}, {pc['bca_hi']:+.4f}], raw p = {pc['p_raw']:.4g}, d = {pc['d']:+.2f}; "
                f"subject-swap p = {r['subject_swap_p']:.4g}\n"
                f"- shrink control simulated agreement {r['shrink_control']['sim_agreement_mean']:+.4f} "
                f"(sd {r['shrink_control']['sim_agreement_sd']:.4f})\n"
                f"- model vs data (P-8 Test A {P8_TESTA_AGREEMENT:+.3f}): {r['model_vs_data']}\n")

    (OUT / "p9_verdict.md").write_text(
        f"# Stage P-9 verdict (4 arms, both backbones)\n\n"
        f"**Backbone question: {bb_letter}. Transfer question: resnet {per['resnet']['letter']}, "
        f"resnet_se {per['resnet_se']['letter']}"
        + ("  (backbones disagree; both reported, neither forced)." if disagree else ".") + "**\n\n"
        f"Scope: transfer only. Per-subject coupling retired as underpowered by design (P-8: about "
        f"151 subjects at rho = -0.226, 351 at rho = +0.149).\n\n"
        f"## Section 3.2 gate\n\n"
        + "".join(f"- {a}: alpha = 0 vs same-run occlusion max|diff| {md:.2e} ({'PASS' if ok else 'FAIL'})\n"
                 for a, (ok, md) in gates.items())
        + f"\n## Section 3.5a reproduction check (A1, A2 only; Section 4.8.2 keeps 84.2 -> 15.0)\n\n"
        + "".join(f"- {r['arm']}: P-9 {r['p9_cost_pp']:.2f} pp vs original {r['original_pp']:.2f} pp "
                 f"({r['diff_pp']:+.2f} pp, {'within' if r['within_gate'] else 'OUTSIDE'} the 3.0 pp "
                 f"gate); 40-fold F1 {r['p9_f1']:.4f} vs {r['original_f1']:.4f} ({r['f1_diff_pp']:+.2f} pp)\n"
                 for r in repro)
        + f"\n## A1/A2 reproduction spread (the yardstick, stated before the cross-backbone comparison)\n\n"
        f"Summed-cost spread max(|A1 - 84.21|, |A2 - 15.03|) = {spread_pp:.2f} pp. Reduction factor: "
        f"original {REPRO_ORIG_REDUCTION:.2f}x, P-9 resnet {R_resnet:.2f}x [95% {Rr_lo:.2f}, {Rr_hi:.2f}] "
        f"(moved {reduction_spread:.2f}x on a same-config re-run).\n\n"
        f"## A3 / A4 first measurement on resnet_se (new numbers, never reproductions)\n\n"
        f"- A3 resnet_se none: summed occlusion cost {m3:.2f} pp, 40-fold F1 {f1_mean(A3):.4f}\n"
        f"- A4 resnet_se chandrop p = 0.2 (model of record): summed occlusion cost {m4:.2f} pp, "
        f"40-fold F1 {f1_mean(A4):.4f}\n"
        f"- resnet_se reduction (A3/A4): {R_se:.2f}x [95% {Rs_lo:.2f}, {Rs_hi:.2f}]\n"
        f"- per-subject CD-induced absolute reduction: resnet {red_resnet.mean():.2f} pp, resnet_se "
        f"{red_se.mean():.2f} pp; paired backbone difference {bdiff.mean():+.2f} pp, 95% BCa "
        f"[{blo:+.2f}, {bhi:+.2f}], p = {bw.pvalue:.4g}\n\n"
        f"## Backbone question: {bb_letter}\n\n{bb_meaning}\n\n"
        f"## Transfer question, per backbone\n\n" + tline("resnet", per["resnet"]) + "\n"
        + tline("resnet_se", per["resnet_se"])
        + ("\nThe two backbones disagree on the transfer letter; both are reported and neither "
           "forced, which is itself evidence bearing on the backbone question.\n" if disagree else "")
        + f"\n## FDR family\n\nP-9's transfer statistics are permutation-null correlations and "
        f"shuffle-null paired comparisons, not paired Wilcoxon tests, so they sit outside the "
        f"Benjamini-Hochberg family (as P-1 and P-2). The per-subject consistency Wilcoxon and the "
        f"paired backbone-reduction Wilcoxon are reported beside the family: resnet consistency "
        f"raw p = {per['resnet']['paired_consistency']['p_raw']:.4g}, resnet_se consistency raw p = "
        f"{per['resnet_se']['paired_consistency']['p_raw']:.4g}, backbone-reduction raw p = "
        f"{bw.pvalue:.4g}. Section 4.17 not edited.\n"
    )
    print(f"\nwrote {OUT/'p9_attenuation_thresholds.csv'}, {OUT/'p9_verdict.md'}, {OUT/'p9_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
run_deepcoral_d1.py
===================
EXPERIMENT_PLAN_DEEPCORAL.md, D1 ONLY. The paired statistics that Section 4.7 /
Table 4.10 should already carry for the CNN-side adaptation comparison, plus the
one arithmetic check on the "2.2 pp" claim. CPU, no retraining. D2 is NOT run.

Inputs (40 rows each, aligned on `subject`):
  results_cnn_aug_resnet_se_chandrop/cnn_arch_subjectwise.csv   per-subject norm   (pub mean 0.8395)
  results_deep_coral_chandrop/deep_coral_subjectwise.csv        Deep CORAL lam=1   (pub mean 0.8257)
  results_adabn_chandrop/adabn_subjectwise.csv                  AdaBN             (pub mean 0.8182)

Does not fabricate numbers, does not edit any chapter.
Output: results_deepcoral_d1/d1_paired_stats.csv
"""
from __future__ import annotations
import sys, io, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, norm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent
OUT = ROOT / "results_deepcoral_d1"; OUT.mkdir(exist_ok=True)

PUB = {"persubj": 0.8395, "coral": 0.8257, "adabn": 0.8182}
NONDET_PP = 0.5           # Section 3.16 run-to-run nondeterminism on the 40-fold mean
ADABN_PRE_PUB = 0.7874
ADABN_ACROSS_RUN_PP = 4.6
ADABN_WITHIN_RUN_PP = 3.1


def bca_ci(x, n_boot=10000):
    x = np.asarray(x, float)
    seed = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(x).tobytes(), digest_size=4).digest(), "big")
    rng = np.random.default_rng(seed)
    th = x.mean()
    bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n_boot)])
    z0 = norm.ppf(min(max((bs < th).mean(), 1e-4), 1 - 1e-4))
    jk = np.array([np.delete(x, i).mean() for i in range(len(x))]); jm = jk.mean()
    den = 6 * (((jm - jk) ** 2).sum() ** 1.5)
    a = (((jm - jk) ** 3).sum() / den) if den else 0.0
    def q(al):
        z = z0 + norm.ppf(al)
        return np.percentile(bs, 100 * norm.cdf(z0 + z / (1 - a * z)))
    return float(q(.025)), float(q(.975))


def cohen_dz(d):
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan


def holm(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    ps = pd.read_csv(ROOT / "results_cnn_aug_resnet_se_chandrop" / "cnn_arch_subjectwise.csv").sort_values("subject")
    co = pd.read_csv(ROOT / "results_deep_coral_chandrop" / "deep_coral_subjectwise.csv").sort_values("subject")
    ad = pd.read_csv(ROOT / "results_adabn_chandrop" / "adabn_subjectwise.csv").sort_values("subject")

    print("=" * 78 + "\nD1 Phase 0  reproduce the three means + one arithmetic check\n" + "=" * 78)
    for df, name in [(ps, "persubj"), (co, "coral"), (ad, "adabn")]:
        assert df["subject"].is_unique and len(df) == 40, f"{name}: not 40 unique subjects"
    same = set(ps.subject) == set(co.subject) == set(ad.subject)
    print(f"  all three cover the same 40 subjects, no duplicates: {same}")

    v = {"persubj": ps["f1_macro"].to_numpy(float),
         "coral": co["f1_macro"].to_numpy(float),
         "adabn": ad["f1_macro"].to_numpy(float)}
    gate = True
    for k in ("persubj", "coral", "adabn"):
        m = v[k].mean()
        hit = abs(round(m, 4) - PUB[k]) <= 5e-4
        gate &= hit
        print(f"  {k:8s} mean f1_macro = {m:.4f}  (published {PUB[k]:.4f})  {'OK' if hit else 'MISS'}")
    if not gate:
        print("\n  *** D1 PHASE 0 GATE FAILED -- stop. ***")
        return
    print("  GATE: PASS")

    d_coral = v["persubj"] - v["coral"]
    d_adabn = v["persubj"] - v["adabn"]
    d_ca = v["coral"] - v["adabn"]
    print(f"\n  per-subject - Deep CORAL : mean diff = {d_coral.mean()*100:.3f} pp  (Section 4.7 says 1.4)")
    print(f"  per-subject - AdaBN      : mean diff = {d_adabn.mean()*100:.3f} pp  (Section 4.7 says 2.2)")
    rounds = round(d_adabn.mean() * 100, 1)
    print(f"  -> AdaBN difference to 3 dp = {d_adabn.mean()*100:.3f} pp; rounds to {rounds:.1f} pp.  "
          f"Section 4.7's '2.2 pp' should read '{rounds:.1f} pp'." if abs(rounds - 2.2) > 1e-9
          else f"  -> AdaBN difference to 3 dp = {d_adabn.mean()*100:.3f} pp; rounds to 2.2 pp, as written.")

    print("\n" + "=" * 78 + "\nD1 Phase 1  paired tests (Holm within family of 3)\n" + "=" * 78)
    tests = [("per-subject norm vs Deep CORAL (lambda=1)", d_coral),
             ("per-subject norm vs AdaBN", d_adabn),
             ("Deep CORAL (lambda=1) vs AdaBN", d_ca)]
    recs, pvals = [], []
    for label, d in tests:
        W, p = wilcoxon(d)
        lo, hi = bca_ci(d)
        recs.append(dict(contrast=label, mean_diff_pp=d.mean() * 100, cohen_dz=cohen_dz(d),
                         bca_lo_pp=lo * 100, bca_hi_pp=hi * 100, n_improved=int((d > 0).sum()),
                         W=float(W), p=float(p)))
        pvals.append(p)
    hp = holm(pvals)
    for r, h in zip(recs, hp):
        r["p_holm"] = float(h); r["sig_holm_0.05"] = "Yes" if h < 0.05 else "No"
        print(f"  {r['contrast']:42s}  diff = {r['mean_diff_pp']:+.3f} pp   dz = {r['cohen_dz']:+.2f}   "
              f"Wilcoxon p = {r['p']:.4f}   Holm p = {r['p_holm']:.4f} ({r['sig_holm_0.05']})   "
              f"BCa95 = [{r['bca_lo_pp']:+.2f}, {r['bca_hi_pp']:+.2f}] pp   {r['n_improved']}/40 improved")
    print()

    # AdaBN within-run paired lift
    d_within = ad["f1_macro"].to_numpy(float) - ad["f1_pre_adabn"].to_numpy(float)
    W_w, p_w = wilcoxon(d_within)
    lo_w, hi_w = bca_ci(d_within)
    print(f"  AdaBN within-run paired lift (f1_macro - f1_pre_adabn):")
    print(f"    mean = {d_within.mean()*100:.3f} pp  (Section 4.7 discloses +3.1 pp within-run vs +4.6 pp across-run)")
    print(f"    pre-adaptation mean = {ad['f1_pre_adabn'].mean():.4f}  (published {ADABN_PRE_PUB})")
    print(f"    Wilcoxon p = {p_w:.2e},  BCa 95% = [{lo_w*100:.2f}, {hi_w*100:.2f}] pp")

    print("\n" + "=" * 78 + "\nD1 Phase 2  differences vs the 0.5 pp nondeterminism band (Section 3.16)\n" + "=" * 78)
    for r in recs:
        mult = r["mean_diff_pp"] / NONDET_PP
        clears = (min(abs(r["bca_lo_pp"]), abs(r["bca_hi_pp"])) > NONDET_PP) and (np.sign(r["bca_lo_pp"]) == np.sign(r["bca_hi_pp"]))
        r["diff_in_nondet_units"] = mult
        r["bca_clears_0.5pp_band"] = "Yes" if clears else "No"
        print(f"  {r['contrast']:42s}  {r['mean_diff_pp']:+.2f} pp = {mult:+.1f} x 0.5pp  "
              f"BCa [{r['bca_lo_pp']:+.2f}, {r['bca_hi_pp']:+.2f}] pp  clears band: {'Yes' if clears else 'No'}")

    pd.DataFrame(recs).to_csv(OUT / "d1_paired_stats.csv", index=False)
    print(f"\n  [save] {OUT / 'd1_paired_stats.csv'}")

    # ---- Gate D1 verdict ----
    print("\n" + "=" * 78 + "\nGate D1 verdict\n" + "=" * 78)
    r_coral = recs[0]
    sig = r_coral["sig_holm_0.05"] == "Yes"
    clears = r_coral["bca_clears_0.5pp_band"] == "Yes"
    d_ok = abs(r_coral["cohen_dz"]) >= 0.5
    if sig and d_ok and clears:
        out = ("1", "the 1.4 pp is significant (Holm), effect size moderate/large, and the BCa interval "
                    "clears the 0.5 pp nondeterminism band; the claim is sound but was unreported -- add the "
                    "statistics to Section 4.7 / Table 4.10. D2 would be warranted (but is NOT run here).")
    else:
        why = []
        if not sig: why.append("not significant after Holm")
        if not clears: why.append("BCa interval does not exclude a 0.5 pp difference")
        if not d_ok: why.append(f"effect size |dz|={abs(r_coral['cohen_dz']):.2f} < 0.5")
        out = ("2", "the 1.4 pp is " + "; ".join(why) + " -- the honest claim is a near-tie, not a win; "
                    "Section 4.7 / 5.8 / 6.2 / abstract need softening to 'within run-to-run variation, at a "
                    "fraction of the cost'. D2 becomes optional. Report and stop for a decision.")
    r_ca = recs[2]
    o3 = "indistinguishable" if r_ca["sig_holm_0.05"] == "No" else "distinguishable"
    print(f"  OUTCOME {out[0]}: {out[1]}")
    print(f"  (Outcome 3 check) Deep CORAL vs AdaBN are {o3} after Holm "
          f"(p_holm = {r_ca['p_holm']:.3f}); Section 4.7's '0.8 pp above AdaBN' ordering "
          f"{'is not meaningful' if o3 == 'indistinguishable' else 'holds'}.")
    print("\n  Appendix A.6 family: the enumerated composition lists '37 tests extending the robustness, "
          "external-validation, calibration and adaptation analyses to the headline models', and the null "
          "list names 'adaptation contrasts in which neither Deep CORAL nor AdaBN outperforms the simple "
          "baseline'. So per-subject-vs-CORAL and per-subject-vs-AdaBN appear already represented; the "
          "Deep CORAL vs AdaBN contrast is the candidate new member. Flag for the write-up wave rather "
          "than asserting a count.")


if __name__ == "__main__":
    main()

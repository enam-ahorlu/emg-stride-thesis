#!/usr/bin/env python3
"""
p1_coupling_stats.py
====================
Stage P-1 of the channel-dropout parity programme (EXPERIMENT_PLAN_CD_PARITY.md
section 2). Does the measured mechanism explain the outcome? No GPU. Reads only
occlusion.csv and cnn_arch_subjectwise.csv from runs that already exist; writes
only into results_parity/.

Three questions, one grid (plan section 2.5):

  P-1a  dose curve on the resnet_se family (4 rate arms): does single-electrode
        occlusion cost keep falling with rate while F1 plateaus / falls?
  P-1b  per-subject coupling on the resnet family (g3_noaug_instr vs
        cd_resnet_nose_chandrop): do the subjects whose reliance falls most gain
        most? Account predicts a NEGATIVE Spearman(delta cost, delta F1).
  P-1c  does baseline reliance predict benefit? Account predicts a POSITIVE
        Spearman(baseline occlusion cost, channel-dropout gain).

Holm family of four: P-1b, P-1c, P-1a Spearman(cost, rate), P-1a Page trend.
Correlations carry a 10,000-draw permutation null at seed 42 and, per plan
section 6.3, are reported separately from the paired-Wilcoxon FDR family.

Estimators (bca_ci, cohens_d_paired) are imported from window_ablation_stats.py,
verified by hand in W-1.
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
from window_ablation_stats import bca_ci, cohens_d_paired  # verified in W-1

OUT = ROOT / "results_parity"
OUT.mkdir(exist_ok=True)

SEED = 42
N_PERM = 10_000
N_CH = 9

RATE_ARMS = {
    0.1: "results_cd_rate_p0.1",
    0.2: "results_cd_rate_p0.2",
    0.3: "results_cd_rate_p0.3",
    0.5: "results_cd_rate_p0.5",
}
RESNET_BASE = "results_g3_noaug_instr"          # --arch resnet, no aug, instrumented
RESNET_CD = "results_cd_resnet_nose_chandrop"   # --arch resnet, chandrop 0.2, instrumented


# --------------------------------------------------------------------------
def load_cost(dirname: str) -> pd.Series:
    """Total single-electrode occlusion cost per subject: sum of drop_pp over the
    9 channels (raw, not clipped), matching g3_occlusion_stats.py's headline
    quantity."""
    p = ROOT / dirname / "instr" / "occlusion.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    need = {"subject", "channel", "drop_pp"}
    if not need.issubset(df.columns):
        sys.exit(f"{p} lacks {need - set(df.columns)}")
    if df.duplicated(["subject", "channel"]).any():
        sys.exit(f"{p} has duplicate (subject, channel) rows; a --resume double-appended")
    wide = df.pivot(index="subject", columns="channel", values="drop_pp").sort_index()
    if wide.shape != (40, N_CH):
        sys.exit(f"{p} pivots to {wide.shape}, expected (40, {N_CH})")
    if wide.isna().any().any():
        sys.exit(f"{p} has missing cells")
    return wide.sum(axis=1).rename(dirname)


def load_f1(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    df = df[[sc, "f1_macro"]].dropna().rename(columns={sc: "subject"})
    if df["subject"].duplicated().any():
        sys.exit(f"{p} has duplicated subjects; a --resume double-appended")
    if len(df) != 40:
        print(f"  WARNING: {dirname} has {len(df)} subjects, expected 40", file=sys.stderr)
    return df.set_index("subject")["f1_macro"].sort_index().rename(dirname)


def perm_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM,
                  seed: int = SEED) -> tuple[float, float]:
    """Two-sided Spearman with a permutation null: shuffle y's subject labels.
    Returns (rho_obs, p_perm) with p_perm = (#|rho*| >= |rho_obs| + 1)/(n_perm+1)."""
    rho_obs = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    yy = np.asarray(y, float)
    hits = 1
    for _ in range(n_perm):
        r = stats.spearmanr(x, rng.permutation(yy)).statistic
        if abs(r) >= abs(rho_obs) - 1e-15:
            hits += 1
    return float(rho_obs), hits / (n_perm + 1)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for k, (name, p) in enumerate(items):
        run = max(run, min(1.0, p * (m - k)))
        out[name] = run
    return out


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("STAGE P-1: does the mechanism explain the outcome?")
    print("=" * 78)

    # ---- input verification (plan section 7 item 1) -----------------------
    print("\n--- input files ---")
    rate_cost, rate_f1 = {}, {}
    for rate, d in RATE_ARMS.items():
        c, f = load_cost(d), load_f1(d)
        rate_cost[rate], rate_f1[rate] = c, f
        print(f"  {d:<34} occlusion 360 rows / 40 subj OK   "
              f"f1 n={len(f)}  mean cost {c.mean():7.2f} pp  mean F1 {f.mean():.4f}")
    base_cost, base_f1 = load_cost(RESNET_BASE), load_f1(RESNET_BASE)
    cd_cost, cd_f1 = load_cost(RESNET_CD), load_f1(RESNET_CD)
    for nm, s in [("g3_noaug_instr cost", base_cost), ("g3_noaug_instr f1", base_f1),
                  ("nose_chandrop cost", cd_cost), ("nose_chandrop f1", cd_f1)]:
        print(f"  {nm:<34} n={len(s)}")
    if not (base_cost.index.equals(cd_cost.index) and base_f1.index.equals(cd_f1.index)
            and base_cost.index.equals(base_f1.index)):
        sys.exit("resnet-family subject indices do not align; stop")
    n_pair = len(base_cost)
    print(f"  resnet family pairs cleanly on {n_pair} subjects")

    results: dict = {"stage": "P-1"}

    # ================================================================
    # P-1a  dose curve, resnet_se family only
    # ================================================================
    print("\n" + "-" * 78)
    print("P-1a  dose curve (resnet_se family, 4 rate arms). Primary.")
    print("-" * 78)
    rates = sorted(RATE_ARMS)
    arm_cost = np.array([rate_cost[r].mean() for r in rates])
    arm_f1 = np.array([rate_f1[r].mean() * 100 for r in rates])
    print(f"  {'rate':>6} {'mean occlusion cost pp':>24} {'mean LOSO macro-F1 %':>22}")
    for r, c, f in zip(rates, arm_cost, arm_f1):
        print(f"  {r:>6} {c:>24.3f} {f:>22.3f}")

    rho_cr = stats.spearmanr(rates, arm_cost)
    # Page trend test: does occlusion cost fall monotonically as rate rises?
    # Page needs replications x conditions with a predicted ordering. Use the 40
    # per-subject cost vectors as replications, conditions ordered by rate, and
    # test the DECREASING alternative by feeding reversed columns.
    cost_mat = np.column_stack([rate_cost[r].to_numpy() for r in rates])  # (40, 4)
    page_dec = stats.page_trend_test(cost_mat[:, ::-1])  # predicts increasing on reversed = decreasing on rate
    print(f"\n  Spearman(rate, mean cost) over 4 arm means : rho = {rho_cr.statistic:+.3f}, "
          f"p = {rho_cr.pvalue:.4g}")
    print(f"  Page trend test (decreasing cost vs rate)   : L = {page_dec.statistic:.1f}, "
          f"p = {page_dec.pvalue:.4g}  (n=40 per-subject replications)")

    # adjacent-rate per-subject paired contrasts on occlusion cost (Wilcoxon)
    print("\n  adjacent-rate per-subject paired contrasts on total occlusion cost:")
    adj_rows = []
    for a, b in zip(rates[:-1], rates[1:]):
        d = (rate_cost[b] - rate_cost[a]).to_numpy()
        lo, hi = bca_ci(d)
        w = stats.wilcoxon(rate_cost[b].to_numpy(), rate_cost[a].to_numpy())
        dz = cohens_d_paired(rate_cost[b].to_numpy(), rate_cost[a].to_numpy())
        print(f"    p={a} -> p={b:<4}  {d.mean():+7.2f} pp  95% BCa [{lo:+.2f}, {hi:+.2f}]  "
              f"p = {w.pvalue:.4g}  d = {dz:+.2f}  ({int((d>0).sum())}/{int((d<0).sum())} +/-)")
        adj_rows.append({"contrast": f"cost p{a}->p{b}", "mean_pp": d.mean(),
                         "lo_pp": lo, "hi_pp": hi, "p_raw": float(w.pvalue), "d": dz})

    f1_mono = (arm_f1[1] - arm_f1[0], arm_f1[2] - arm_f1[1], arm_f1[3] - arm_f1[2])
    cost_mono_all_neg = bool(np.all(np.diff(arm_cost) < 0))
    print(f"\n  F1 steps across rate (pp): "
          f"{f1_mono[0]:+.2f} (0.1->0.2), {f1_mono[1]:+.2f} (0.2->0.3), {f1_mono[2]:+.2f} (0.3->0.5)")
    print(f"  occlusion cost monotone decreasing across all 3 steps: {cost_mono_all_neg}")
    results["p1a"] = {
        "rates": rates, "arm_cost_pp": arm_cost.tolist(), "arm_f1_pct": arm_f1.tolist(),
        "spearman_rate_cost_rho": rho_cr.statistic, "spearman_rate_cost_p": float(rho_cr.pvalue),
        "page_L": float(page_dec.statistic), "page_p": float(page_dec.pvalue),
        "cost_monotone_decreasing": cost_mono_all_neg,
        "f1_steps_pp": list(f1_mono), "adjacent": adj_rows,
    }

    # ================================================================
    # P-1b  per-subject coupling, resnet family
    # ================================================================
    print("\n" + "-" * 78)
    print("P-1b  per-subject coupling (resnet family). Primary.")
    print("-" * 78)
    d_cost = (cd_cost - base_cost).reindex(base_cost.index)
    d_f1 = ((cd_f1 - base_f1) * 100).reindex(base_cost.index)
    rho_b, p_b = perm_spearman(d_cost.to_numpy(), d_f1.to_numpy())
    print(f"  delta occlusion cost: mean {d_cost.mean():+.2f} pp  (range {d_cost.min():+.1f} .. {d_cost.max():+.1f})")
    print(f"  delta macro-F1      : mean {d_f1.mean():+.2f} pp  (range {d_f1.min():+.1f} .. {d_f1.max():+.1f})")
    print(f"  Spearman(delta cost, delta F1) = {rho_b:+.3f}   permutation p = {p_b:.4g}  "
          f"(10,000 draws, seed {SEED})")
    print(f"  account predicts rho < 0 (reliance falls most where F1 gains most): "
          f"{'consistent' if rho_b < 0 else 'INCONSISTENT (sign)'}")
    results["p1b"] = {"spearman_rho": rho_b, "perm_p": p_b, "predicted_sign": "negative",
                      "sign_consistent": bool(rho_b < 0)}

    # ================================================================
    # P-1c  does baseline reliance predict benefit? resnet family
    # ================================================================
    print("\n" + "-" * 78)
    print("P-1c  baseline reliance predicts benefit (resnet family). Primary.")
    print("-" * 78)
    gain = ((cd_f1 - base_f1) * 100).reindex(base_cost.index)
    rho_c, p_c = perm_spearman(base_cost.to_numpy(), gain.to_numpy())
    print(f"  baseline occlusion cost: mean {base_cost.mean():.2f} pp  "
          f"(range {base_cost.min():.1f} .. {base_cost.max():.1f})")
    print(f"  channel-dropout gain   : mean {gain.mean():+.2f} pp")
    print(f"  Spearman(baseline cost, gain) = {rho_c:+.3f}   permutation p = {p_c:.4g}")
    print(f"  account predicts rho > 0 (hardest single-electrode leaners gain most): "
          f"{'consistent' if rho_c > 0 else 'INCONSISTENT (sign)'}")
    results["p1c"] = {"spearman_rho": rho_c, "perm_p": p_c, "predicted_sign": "positive",
                      "sign_consistent": bool(rho_c > 0)}

    # companion (4th family member): does baseline reliance predict the CHANGE in
    # reliance? part of the mechanism chain P-1b/P-1c share.
    rho_c2, p_c2 = perm_spearman(base_cost.to_numpy(), d_cost.to_numpy())
    print(f"\n  companion  Spearman(baseline cost, delta cost) = {rho_c2:+.3f}  "
          f"permutation p = {p_c2:.4g}  (reported for the family of four)")
    results["p1c_companion"] = {"spearman_rho": rho_c2, "perm_p": p_c2}

    # ================================================================
    # Holm family of four
    # ================================================================
    fam = {
        "P-1a Spearman(rate,cost)": float(rho_cr.pvalue),
        "P-1a Page trend": float(page_dec.pvalue),
        "P-1b coupling": p_b,
        "P-1c prediction": p_c,
    }
    hp = holm(fam)
    print("\n" + "-" * 78)
    print("Holm family of four (plan section 2.5)")
    print("-" * 78)
    for name in fam:
        print(f"  {name:<28} raw p = {fam[name]:.4g}   Holm p = {hp[name]:.4g}")
    results["holm_family_of_four"] = {k: {"raw": fam[k], "holm": hp[k]} for k in fam}

    # ================================================================
    # pre-registered grid (plan section 2.5)
    # ================================================================
    corr_sig = []
    if hp["P-1b coupling"] < 0.05 and rho_b < 0:
        corr_sig.append("P-1b")
    if hp["P-1c prediction"] < 0.05 and rho_c > 0:
        corr_sig.append("P-1c")
    reversal = ((hp["P-1b coupling"] < 0.05 and rho_b > 0) or
                (hp["P-1c prediction"] < 0.05 and rho_c < 0))
    p1a_monotone = cost_mono_all_neg and (hp["P-1a Page trend"] < 0.05 or hp["P-1a Spearman(rate,cost)"] < 0.05)
    f1_plateau_or_fall = (arm_f1[2] - arm_f1[0] < 1.0)  # G2: flat 0.1->0.3

    underpowered = n_pair < 35
    if reversal:
        letter = "R"
        meaning = ("Reversal. A correlation survives correction with the sign opposite to the "
                   "mechanism account. Report the numbers, do not force a reading, escalate.")
    elif corr_sig and p1a_monotone:
        letter = "L"
        meaning = ("Linked. The mechanism explains the benefit. Section 4.8.2 can be written as "
                   "an explanation rather than a co-occurrence.")
    elif corr_sig:
        letter = "L"
        meaning = ("Linked (correlation arm). A per-subject correlation survives correction with "
                   "the predicted sign; P-1a was not decisive on its own.")
    elif p1a_monotone and f1_plateau_or_fall:
        letter = "S"
        meaning = ("Separated at the boundary. Occlusion cost keeps falling with rate while F1 "
                   "plateaus / falls: more invariance keeps buying robustness and stops buying "
                   "accuracy. Write as the over-alignment analogue of section 4.13.2, structural "
                   "not mechanistic.")
    elif underpowered:
        letter = "U"
        meaning = ("Underpowered. Fewer than 35 subjects pair cleanly, or the null cannot resolve "
                   "the effect at issue. The measure cannot answer it.")
    else:
        letter = "C"
        meaning = ("Co-occurring only. No correlation survives correction and P-1a shows no "
                   "monotone trend. Section 4.8.2 must say robustness and accuracy co-occur, not "
                   "that one explains the other. Informative negative, written up.")
    print("\n" + "=" * 78)
    print(f"P-1 OUTCOME {letter}: {meaning}")
    print("=" * 78)
    results["outcome"] = letter
    results["outcome_meaning"] = meaning

    # ================================================================
    # outputs
    # ================================================================
    arm_summary = pd.DataFrame({
        "rate": rates,
        "mean_occlusion_cost_pp": arm_cost,
        "mean_loso_macro_f1_pct": arm_f1,
    })
    arm_summary.to_csv(OUT / "p1_arm_summary.csv", index=False)

    coupling = pd.DataFrame({
        "subject": base_cost.index,
        "base_occlusion_cost_pp": base_cost.values,
        "cd_occlusion_cost_pp": cd_cost.reindex(base_cost.index).values,
        "delta_occlusion_cost_pp": d_cost.values,
        "base_f1_macro": base_f1.reindex(base_cost.index).values,
        "cd_f1_macro": cd_f1.reindex(base_cost.index).values,
        "delta_f1_pp": d_f1.values,
    }).round(5)
    coupling.to_csv(OUT / "p1_coupling_tests.csv", index=False)

    (OUT / "p1_verdict.md").write_text(
        f"# Stage P-1 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"## P-1a dose curve (resnet_se family)\n\n"
        f"| rate | mean occlusion cost (pp) | mean LOSO macro-F1 (%) |\n|---|---|---|\n"
        + "".join(f"| {r} | {c:.3f} | {f:.3f} |\n" for r, c, f in zip(rates, arm_cost, arm_f1))
        + f"\nSpearman(rate, cost) rho = {rho_cr.statistic:.3f}, raw p = {rho_cr.pvalue:.4g}, "
        f"Holm p = {hp['P-1a Spearman(rate,cost)']:.4g}. "
        f"Page trend (decreasing cost vs rate) L = {page_dec.statistic:.1f}, raw p = {page_dec.pvalue:.4g}, "
        f"Holm p = {hp['P-1a Page trend']:.4g}. "
        f"Occlusion cost monotone decreasing across all three rate steps: {cost_mono_all_neg}. "
        f"F1 change 0.1 to 0.3: {arm_f1[2]-arm_f1[0]:+.2f} pp; 0.3 to 0.5: {arm_f1[3]-arm_f1[2]:+.2f} pp.\n\n"
        f"No instrumented no-augmentation resnet_se run exists, so this curve has no p = 0 origin. "
        f"The SE-free results_g3_noaug_instr is NOT substituted as the origin (plan section 2.2).\n\n"
        f"## P-1b per-subject coupling (resnet family, n = {n_pair})\n\n"
        f"Spearman(delta occlusion cost, delta F1) = {rho_b:.3f}, permutation p = {p_b:.4g} "
        f"(10,000 draws, seed {SEED}), Holm p = {hp['P-1b coupling']:.4g}. "
        f"Account predicts rho < 0; observed sign {'consistent' if rho_b < 0 else 'inconsistent'}.\n\n"
        f"## P-1c baseline reliance predicts benefit (resnet family, n = {n_pair})\n\n"
        f"Spearman(baseline occlusion cost, channel-dropout gain) = {rho_c:.3f}, "
        f"permutation p = {p_c:.4g}, Holm p = {hp['P-1c prediction']:.4g}. "
        f"Account predicts rho > 0; observed sign {'consistent' if rho_c > 0 else 'inconsistent'}. "
        f"Companion Spearman(baseline cost, delta cost) = {rho_c2:.3f}, permutation p = {p_c2:.4g}.\n\n"
        f"## FDR family contribution (plan section 6.3)\n\n"
        f"Correlations against a permutation null are not paired Wilcoxon tests and are reported "
        f"outside the Benjamini-Hochberg family. The three adjacent-rate per-subject paired "
        f"Wilcoxon contrasts on total occlusion cost (P-1a) are new paired tests:\n\n"
        + "".join(f"- {r['contrast']}: raw p = {r['p_raw']:.4g}\n" for r in adj_rows)
        + "\n"
    )

    json.dump(results, open(OUT / "p1_outcome.json", "w"), indent=2)
    print(f"\nwrote {OUT/'p1_arm_summary.csv'}, {OUT/'p1_coupling_tests.csv'}, "
          f"{OUT/'p1_verdict.md'}, {OUT/'p1_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

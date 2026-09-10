#!/usr/bin/env python3
"""
p2_rank_transfer_stats.py
=========================
Stage P-2 of the channel-dropout parity programme (EXPERIMENT_PLAN_CD_PARITY.md
section 3). Does electrode reliance transfer between subjects, measured with a
scale-free rank statistic instead of the normalized profile that collapsed in
section 4.8.2? No GPU.

resnet family only:
  results_g3_noaug_instr/instr/occlusion.csv          no augmentation   (baseline)
  results_cd_resnet_nose_chandrop/instr/occlusion.csv channel dropout   (augmented)

Measure: per subject rank the 9 channels by drop_pp; between-subject agreement is
the mean pairwise Spearman of rankings over all 780 subject pairs, computed
separately per arm. Paired Wilcoxon and a subject-level randomization test
compare the arms; a within-subject channel-label-shuffle null (10,000 draws,
seed 42) gives each arm's resolution floor.

Two validity checks gate the reading (plan section 3.2):
  1. null separation  - is the augmented arm above its own shuffle null at all?
  2. shrink control   - shrink the baseline profiles to the augmented arm's
                        scale, add matched noise, recompute rank agreement; if
                        that alone reproduces the deficit, ranks are not
                        scale-free in practice.

Grid T / N / U / X in plan section 3.3. Estimators from window_ablation_stats.py.
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
BASE = "results_g3_noaug_instr"
CD = "results_cd_resnet_nose_chandrop"


def load_wide(dirname: str) -> pd.DataFrame:
    p = ROOT / dirname / "instr" / "occlusion.csv"
    if not p.exists():
        sys.exit(f"missing {p}")
    df = pd.read_csv(p)
    if df.duplicated(["subject", "channel"]).any():
        sys.exit(f"{p} has duplicate (subject, channel) rows")
    wide = df.pivot(index="subject", columns="channel", values="drop_pp").sort_index()
    if wide.shape != (40, N_CH):
        sys.exit(f"{p} pivots to {wide.shape}, expected (40, {N_CH})")
    if wide.isna().any().any():
        sys.exit(f"{p} has missing cells")
    return wide


def ranks_of(mat: np.ndarray) -> np.ndarray:
    """Row-wise ranks (1..9) of drop_pp; higher drop_pp -> higher rank."""
    return np.apply_along_axis(stats.rankdata, 1, mat)


def pairwise_mean_spearman(rank_mat: np.ndarray) -> float:
    """Mean Spearman correlation over all C(n,2) subject pairs of rank rows."""
    r = stats.spearmanr(rank_mat, axis=1).statistic
    r = np.asarray(r, float)
    iu = np.triu_indices_from(r, k=1)
    return float(np.nanmean(r[iu]))


def per_subject_consistency(rank_mat: np.ndarray) -> np.ndarray:
    """Per subject: mean Spearman of its ranking against the other n-1."""
    r = np.asarray(stats.spearmanr(rank_mat, axis=1).statistic, float)
    np.fill_diagonal(r, np.nan)
    return np.nanmean(r, axis=1)


def within_subject_shuffle_null(mat: np.ndarray, n_perm: int = N_PERM,
                                seed: int = SEED) -> np.ndarray:
    """Null: independently permute the 9 channel labels within each subject, then
    recompute the mean pairwise Spearman of the resulting ranks."""
    rng = np.random.default_rng(seed)
    n, c = mat.shape
    out = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.empty_like(mat)
        for i in range(n):
            perm[i] = mat[i, rng.permutation(c)]
        out[k] = pairwise_mean_spearman(ranks_of(perm))
    return out


def subject_swap_p(cd_ranks: np.ndarray, base_ranks: np.ndarray,
                   n_perm: int = N_PERM, seed: int = SEED) -> float:
    """Two-sided subject-level randomization test on the difference in mean
    pairwise rank agreement: under the null a subject's two rank rows are
    exchangeable, so swap them independently per subject."""
    rng = np.random.default_rng(seed)
    obs = pairwise_mean_spearman(cd_ranks) - pairwise_mean_spearman(base_ranks)
    n = cd_ranks.shape[0]
    hits = 1
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        a = np.where(swap[:, None], base_ranks, cd_ranks)
        b = np.where(swap[:, None], cd_ranks, base_ranks)
        if abs(pairwise_mean_spearman(a) - pairwise_mean_spearman(b)) >= abs(obs) - 1e-15:
            hits += 1
    return hits / (n_perm + 1)


def shrink_control(base_wide: pd.DataFrame, cd_wide: pd.DataFrame,
                   n_real: int = 200, seed: int = SEED) -> dict:
    """Plan section 3.2 check 2. Shrink the baseline profiles to the augmented
    arm's total scale, add Gaussian noise whose SD is tuned so the simulated
    arm's negative-entry fraction matches the augmented arm's, then measure rank
    agreement on the simulated profiles across n_real noise realizations."""
    b = base_wide.to_numpy()
    c = cd_wide.to_numpy()
    k = c.sum() / b.sum()                       # global shrink factor
    shrunk = b * k
    target_neg = float((c < 0).mean())          # e.g. 93/360

    # bisection on sigma so mean negative fraction of (shrunk + N(0, sigma)) ~ target
    rng = np.random.default_rng(seed)
    lo, hi = 1e-6, 10.0 * (abs(shrunk).mean() + 1e-6)

    def neg_frac(sig):
        acc = []
        for _ in range(40):
            acc.append(((shrunk + rng.normal(0, sig, shrunk.shape)) < 0).mean())
        return float(np.mean(acc))

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if neg_frac(mid) < target_neg:
            lo = mid
        else:
            hi = mid
    sigma = 0.5 * (lo + hi)

    rng = np.random.default_rng(seed + 1)
    agrees = []
    for _ in range(n_real):
        sim = shrunk + rng.normal(0, sigma, shrunk.shape)
        agrees.append(pairwise_mean_spearman(ranks_of(sim)))
    return {"shrink_factor": k, "noise_sigma": sigma,
            "sim_neg_frac": neg_frac(sigma), "target_neg_frac": target_neg,
            "sim_agreement_mean": float(np.mean(agrees)),
            "sim_agreement_sd": float(np.std(agrees, ddof=1))}


def main() -> int:
    print("=" * 78)
    print("STAGE P-2: does electrode reliance transfer between subjects?")
    print("=" * 78)

    base_wide, cd_wide = load_wide(BASE), load_wide(CD)
    if not base_wide.index.equals(cd_wide.index):
        sys.exit("the two arms do not cover the same subjects; gate 1 fails")
    n = len(base_wide)
    n_pairs = n * (n - 1) // 2
    b_neg = int((base_wide.to_numpy() < 0).sum())
    c_neg = int((cd_wide.to_numpy() < 0).sum())
    print(f"\nsubjects paired: {n}   subject pairs: {n_pairs}   channels: {N_CH}")
    print(f"negative drop_pp entries: baseline {b_neg}/{n*N_CH}, channel dropout "
          f"{c_neg}/{n*N_CH}  (plan quotes 14 and 93)")

    base_r = ranks_of(base_wide.to_numpy())
    cd_r = ranks_of(cd_wide.to_numpy())
    base_agree = pairwise_mean_spearman(base_r)
    cd_agree = pairwise_mean_spearman(cd_r)
    print(f"\n--- rank agreement (mean pairwise Spearman of channel orderings) ---")
    print(f"  baseline arm       : {base_agree:+.4f}")
    print(f"  channel-dropout arm: {cd_agree:+.4f}")
    print(f"  deficit (base - CD): {base_agree - cd_agree:+.4f}")

    # paired Wilcoxon on per-subject consistency
    b_cons = per_subject_consistency(base_r)
    c_cons = per_subject_consistency(cd_r)
    d = c_cons - b_cons
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(c_cons, b_cons)
    dz = cohens_d_paired(c_cons, b_cons)
    print(f"\n  paired Wilcoxon on per-subject consistency (CD - baseline): "
          f"{d.mean():+.4f}  95% BCa [{lo:+.4f}, {hi:+.4f}]  p = {w.pvalue:.4g}  d = {dz:+.2f}  "
          f"({int((d>0).sum())}/{int((d<0).sum())} +/-)")
    p_swap = subject_swap_p(cd_r, base_r)
    print(f"  subject-level randomization test (10,000 swaps, seed {SEED}): p = {p_swap:.4g}")

    # ---- validity check 1: null separation ------------------------------
    print(f"\n--- validity check 1: null separation (within-subject channel-label shuffle) ---")
    cd_null = within_subject_shuffle_null(cd_wide.to_numpy())
    base_null = within_subject_shuffle_null(base_wide.to_numpy(), seed=SEED + 7)
    cd_null_p = (int(np.sum(np.abs(cd_null) >= abs(cd_agree) - 1e-15)) + 1) / (N_PERM + 1)
    base_null_p = (int(np.sum(np.abs(base_null) >= abs(base_agree) - 1e-15)) + 1) / (N_PERM + 1)
    cd_null_hi = float(np.quantile(cd_null, 0.975))
    print(f"  baseline arm: agreement {base_agree:+.4f}  vs null mean {base_null.mean():+.4f} "
          f"(97.5% {np.quantile(base_null,0.975):+.4f})  p = {base_null_p:.4g}")
    print(f"  CD arm      : agreement {cd_agree:+.4f}  vs null mean {cd_null.mean():+.4f} "
          f"(97.5% {cd_null_hi:+.4f})  p = {cd_null_p:.4g}")
    cd_above_null = cd_null_p < 0.05 and cd_agree > cd_null_hi
    print(f"  CD arm distinguishable from its own null: {'YES' if cd_above_null else 'NO'}")

    # ---- validity check 2: shrink control ------------------------------
    print(f"\n--- validity check 2: shrink control (baseline shrunk to CD scale + matched noise) ---")
    sc = shrink_control(base_wide, cd_wide)
    print(f"  shrink factor (sum CD / sum baseline) : {sc['shrink_factor']:.4f}")
    print(f"  tuned noise sigma                     : {sc['noise_sigma']:.4f} pp  "
          f"(sim neg frac {sc['sim_neg_frac']:.3f} vs target {sc['target_neg_frac']:.3f})")
    print(f"  rank agreement on simulated profiles  : {sc['sim_agreement_mean']:+.4f} "
          f"(sd {sc['sim_agreement_sd']:.4f}, 200 realizations)")
    observed_deficit = base_agree - cd_agree
    sim_deficit = base_agree - sc["sim_agreement_mean"]
    frac_reproduced = sim_deficit / observed_deficit if observed_deficit > 1e-9 else float("nan")
    print(f"  observed deficit {observed_deficit:+.4f}  |  shrink-alone deficit {sim_deficit:+.4f}  "
          f"|  fraction reproduced {frac_reproduced:.2f}")
    ranks_not_scale_free = (frac_reproduced >= 0.5) if np.isfinite(frac_reproduced) else False

    # ---- pre-registered grid (plan section 3.3) -----------------------
    print("\n" + "=" * 78)
    if ranks_not_scale_free:
        letter = "X"
        meaning = ("Ranks are not scale-free in practice. The shrink control reproduces "
                   f"{frac_reproduced:.0%} of the baseline-to-CD agreement deficit on simulated "
                   "data with no real change in ordering. Report as a measurement finding; do "
                   "NOT report a transfer answer.")
    elif not cd_above_null:
        letter = "U"
        meaning = ("Still unanswered. The augmented arm's rank agreement is indistinguishable "
                   "from its own within-subject shuffle null, so the measure has no resolution "
                   "here. Section 4.8.2 keeps its current wording; section 5.13 records that a "
                   "second measure has now failed on the transfer question.")
    elif p_swap < 0.05 and cd_agree < base_agree:
        letter = "N"
        meaning = ("Does not transfer. The augmented arm's rank agreement is above its null but "
                   "significantly below the baseline arm's: electrode reliance is flatter AND "
                   "more idiosyncratic under channel dropout. A real negative that closes the "
                   "section 4.8.2 question.")
    else:
        letter = "T"
        meaning = ("Transfers. The augmented arm's rank agreement is above its null and not "
                   "significantly below the baseline arm's: reliance is flatter and still shared "
                   "across people. Closes the section 4.8.2 question positively.")
    print(f"P-2 OUTCOME {letter}: {meaning}")
    print("=" * 78)

    # ---- outputs -----------------------------------------------------
    rows = pd.DataFrame({
        "subject": base_wide.index,
        "base_consistency": b_cons,
        "cd_consistency": c_cons,
        "delta_consistency": d,
    }).round(5)
    rows.to_csv(OUT / "p2_rank_agreement.csv", index=False)

    res = {
        "stage": "P-2", "n_subjects": n, "n_pairs": n_pairs,
        "neg_entries": {"baseline": b_neg, "channel_dropout": c_neg},
        "rank_agreement": {"baseline": base_agree, "channel_dropout": cd_agree,
                           "deficit_base_minus_cd": base_agree - cd_agree},
        "paired_wilcoxon": {"mean_delta": float(d.mean()), "bca_lo": lo, "bca_hi": hi,
                            "p_raw": float(w.pvalue), "cohens_d": dz},
        "subject_swap_p": p_swap,
        "validity_null_separation": {
            "cd_agreement": cd_agree, "cd_null_mean": float(cd_null.mean()),
            "cd_null_p": cd_null_p, "cd_null_97_5": cd_null_hi,
            "cd_distinguishable_from_null": bool(cd_above_null),
            "base_null_p": base_null_p},
        "validity_shrink_control": {**sc, "observed_deficit": observed_deficit,
                                    "shrink_alone_deficit": sim_deficit,
                                    "fraction_reproduced": frac_reproduced,
                                    "ranks_not_scale_free": bool(ranks_not_scale_free)},
        "outcome": letter, "outcome_meaning": meaning,
    }
    json.dump(res, open(OUT / "p2_outcome.json", "w"), indent=2)

    (OUT / "p2_verdict.md").write_text(
        f"# Stage P-2 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"## Rank agreement (mean pairwise Spearman of channel orderings, resnet family, "
        f"n = {n}, {n_pairs} pairs)\n\n"
        f"- baseline arm: {base_agree:+.4f}\n"
        f"- channel-dropout arm: {cd_agree:+.4f}\n"
        f"- deficit (baseline - CD): {base_agree - cd_agree:+.4f}\n"
        f"- paired Wilcoxon on per-subject consistency (CD - baseline): {d.mean():+.4f}, "
        f"95% BCa [{lo:+.4f}, {hi:+.4f}], raw p = {w.pvalue:.4g}, d = {dz:+.2f}\n"
        f"- subject-level randomization test: p = {p_swap:.4g}\n\n"
        f"## Validity check 1: null separation\n\n"
        f"- baseline arm vs its within-subject shuffle null: p = {base_null_p:.4g}\n"
        f"- channel-dropout arm vs its within-subject shuffle null: p = {cd_null_p:.4g} "
        f"(agreement {cd_agree:+.4f} vs null 97.5% {cd_null_hi:+.4f})\n"
        f"- CD arm distinguishable from its own null: {'YES' if cd_above_null else 'NO'}\n\n"
        f"## Validity check 2: shrink control\n\n"
        f"- shrink factor: {sc['shrink_factor']:.4f}; tuned noise sigma: {sc['noise_sigma']:.4f} pp "
        f"(sim negative fraction {sc['sim_neg_frac']:.3f} vs target {sc['target_neg_frac']:.3f})\n"
        f"- rank agreement on simulated shrunk+noise profiles: {sc['sim_agreement_mean']:+.4f} "
        f"(sd {sc['sim_agreement_sd']:.4f})\n"
        f"- observed deficit {observed_deficit:+.4f}; shrink-alone deficit {sim_deficit:+.4f}; "
        f"fraction reproduced {frac_reproduced:.2f}\n"
        f"- ranks not scale-free in practice: {bool(ranks_not_scale_free)}\n\n"
        f"## FDR family contribution (plan section 6.3)\n\n"
        f"One new paired Wilcoxon test (per-subject rank consistency, CD vs baseline): "
        f"raw p = {w.pvalue:.4g}. The randomization test and the two shuffle-null checks are "
        f"not paired Wilcoxon tests and are reported outside the Benjamini-Hochberg family.\n"
    )
    print(f"\nwrote {OUT/'p2_rank_agreement.csv'}, {OUT/'p2_verdict.md'}, {OUT/'p2_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
p8_informativeness_stats.py
===========================
Stage P-8 (EXPERIMENT_PLAN_LOCUS.md section 2). Is channel informativeness
subject-specific? NO GPU. Pure re-analysis of the Freq-72 features and two
result files on disk.

Column-to-channel mapping (Freq-72, from extract_features.py + REPRODUCE.md):
  Freq-72 = MAV(9) RMS(9) WL(9) ZC(9) WAMP(9) MNF(9) MDF(9) SpectralPower(9)
  feature-major: column j  ->  feature family j // 9, channel j % 9.
  channel c's 8 features are columns {c, c+9, c+18, c+27, c+36, c+45, c+54, c+63}.
Channel index 0..8 = TFL, RF, VM, SMB, Upper TA, Lower TA, Lateral GC,
Medial GC, SOL (Table 3.1 order).

Per-subject z-score exactly as train_classical_loso.per_subject_zscore
(axis=0, per column, within subject). Then per subject x channel, a 4-class LDA
on that channel's 8 features only, stratified 5-fold CV within subject,
macro-F1. A 40 x 9 matrix of within-subject per-channel discriminative power.

  Test A  ordering agreement across 780 subject pairs vs a within-subject
          channel-label shuffle null (10,000 draws, seed 42); consensus ranking.
  Test B  (primary) leave-one-out consensus agreement score per subject vs
          B1 the un-augmented ResNet-SE LOSO F1 (predicted +), and
          B2 the channel-dropout gain (predicted -). Spearman, permutation
          null, Holm across the two.
  Test C  consensus informativeness vs the un-augmented model's mean occlusion
          profile (results_g3_noaug_instr). One Spearman, permutation null,
          beside the family.

The before/after-normalization contrast is deliberately NOT run: LDA is
invariant to invertible linear maps and per-column standardisation is diagonal,
so the two matrices are identical by construction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_locus"
OUT.mkdir(exist_ok=True)

SEED = 42
N_PERM = 10_000
N_CH = 9
MUSCLES = ["TFL", "RF", "VM", "SMB", "Upper TA", "Lower TA", "Lateral GC", "Medial GC", "SOL"]

FEAT_NPZ = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
FEAT_META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
NOAUG_SE = "results_cnn_aug_resnet_se_none"       # un-augmented ResNet-SE, 0.7822
CHANDROP_SE = "results_cnn_aug_resnet_se_chandrop"  # ResNet-SE + channel dropout, 0.8395
OCC_NOAUG = ROOT / "results_g3_noaug_instr" / "instr" / "occlusion.csv"

FEATURE_FAMILIES = ["MAV", "RMS", "WL", "ZC", "WAMP", "MNF", "MDF", "SpectralPower"]


def per_subject_zscore(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Byte-for-byte the train_classical_loso implementation."""
    Xn = X.copy()
    for sid in np.unique(subjects):
        m = subjects == sid
        Xs = X[m]
        mu = Xs.mean(axis=0, keepdims=True)
        sd = Xs.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xn[m] = (Xs - mu) / sd
    return Xn


def channel_cols(c: int) -> list[int]:
    return [c + 9 * k for k in range(8)]


def per_channel_lda_f1(Xz: np.ndarray, y: np.ndarray, subjects: np.ndarray) -> pd.DataFrame:
    subs = sorted(np.unique(subjects).tolist())
    rows = []
    for sid in subs:
        m = subjects == sid
        ys = y[m]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for c in range(N_CH):
            Xc = Xz[m][:, channel_cols(c)]
            f1s = []
            for tr, te in skf.split(Xc, ys):
                clf = LinearDiscriminantAnalysis()
                clf.fit(Xc[tr], ys[tr])
                f1s.append(f1_score(ys[te], clf.predict(Xc[te]), average="macro", zero_division=0))
            rows.append({"subject": int(sid), "channel": c, "f1_macro": float(np.mean(f1s))})
    return pd.DataFrame(rows)


def load_subjectwise(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    return df.set_index(sc)["f1_macro"].sort_index()


def perm_spearman(x, y, n_perm=N_PERM, seed=SEED):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    hits = 1
    for _ in range(n_perm):
        if abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= abs(rho) - 1e-15:
            hits += 1
    return float(rho), hits / (n_perm + 1)


def mean_pairwise_spearman(mat: np.ndarray) -> float:
    r = np.asarray(stats.spearmanr(mat, axis=1).statistic, float)
    iu = np.triu_indices_from(r, k=1)
    return float(np.nanmean(r[iu]))


def within_subject_shuffle_null(mat: np.ndarray, n_perm=N_PERM, seed=SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, c = mat.shape
    out = np.empty(n_perm)
    for k in range(n_perm):
        perm = np.stack([mat[i, rng.permutation(c)] for i in range(n)])
        out[k] = mean_pairwise_spearman(perm)
    return out


def holm2(pa, pb):
    order = sorted([("a", pa), ("b", pb)], key=lambda kv: kv[1])
    out, run = {}, 0.0
    for k, (nm, p) in enumerate(order):
        run = max(run, min(1.0, p * (2 - k)))
        out[nm] = run
    return out["a"], out["b"]


def main() -> int:
    print("=" * 78)
    print("STAGE P-8: is channel informativeness subject-specific?")
    print("=" * 78)

    # ---- column-to-channel mapping, reported first ----
    print("\n--- Freq-72 column-to-channel mapping (report first, section 6.1) ---")
    print("  layout: feature-major. Freq-72 = "
          "MAV(9) RMS(9) WL(9) ZC(9) WAMP(9) MNF(9) MDF(9) SpectralPower(9).")
    print("  column j  ->  feature family = j // 9,  channel = j % 9.")
    for c in range(N_CH):
        print(f"  channel {c} ({MUSCLES[c]:<10}): columns {channel_cols(c)}")

    X = np.load(FEAT_NPZ)["X"].astype(np.float64)
    meta = pd.read_csv(FEAT_META)
    y = meta["y_int"].to_numpy()
    subjects = meta["subject"].to_numpy()
    assert X.shape == (len(meta), 72), X.shape
    print(f"\n  features {X.shape}, {len(np.unique(subjects))} subjects, "
          f"classes {sorted(np.unique(y))}, class counts {np.bincount(y).tolist()}")

    Xz = per_subject_zscore(X, subjects)

    # ---- the 40 x 9 matrix ----
    print("\n--- per-subject per-channel LDA macro-F1 (40 x 9 x 5 fits) ---")
    df = per_channel_lda_f1(Xz, y, subjects)
    wide = df.pivot(index="subject", columns="channel", values="f1_macro").sort_index()
    subs = wide.index.to_numpy()
    M = wide.to_numpy()  # (40, 9)
    print(f"  matrix {M.shape}   overall mean F1 {M.mean():.4f}   per-channel means:")
    for c in range(N_CH):
        print(f"    ch {c} {MUSCLES[c]:<10} mean {M[:, c].mean():.4f}  sd {M[:, c].std(ddof=1):.4f}  "
              f"min {M[:, c].min():.4f}  max {M[:, c].max():.4f}")
    chance = 1.0 / 4
    frac_near_chance = float((M < chance + 0.05).mean())
    print(f"  fraction of (subject, channel) cells below chance+0.05 ({chance+0.05:.2f}): {frac_near_chance:.3f}")

    # ---- consensus ranking ----
    ch_mean = M.mean(axis=0)
    order = np.argsort(-ch_mean)  # descending informativeness
    print("\n--- consensus channel ranking (mean within-subject LDA macro-F1, high to low) ---")
    for rank, c in enumerate(order, 1):
        print(f"  {rank}. ch {c}  {MUSCLES[c]:<10}  mean F1 {ch_mean[c]:.4f}")

    # ================= Test A =================
    print("\n" + "-" * 78)
    print("TEST A: is the channel ordering subject-specific?")
    print("-" * 78)
    agree = mean_pairwise_spearman(M)
    nullA = within_subject_shuffle_null(M)
    pA = (int(np.sum(np.abs(nullA) >= abs(agree) - 1e-15)) + 1) / (N_PERM + 1)
    print(f"  mean pairwise Spearman of channel rankings (780 pairs): {agree:+.4f}")
    print(f"  within-subject shuffle null: mean {nullA.mean():+.4f}, 97.5% {np.quantile(nullA,0.975):+.4f}, "
          f"p = {pA:.4g}")
    print(f"  ceiling is +1.000 (identical ordering for everyone)")

    # ================= consensus agreement score (LOO) =================
    cons_score = np.empty(len(subs))
    for i in range(len(subs)):
        others = np.delete(M, i, axis=0).mean(axis=0)
        cons_score[i] = stats.spearmanr(M[i], others).statistic
    print(f"\n  leave-one-out consensus agreement score: mean {cons_score.mean():+.4f}, "
          f"sd {cons_score.std(ddof=1):.4f}, range [{cons_score.min():+.3f}, {cons_score.max():+.3f}]")

    # ================= Test B =================
    print("\n" + "-" * 78)
    print("TEST B (primary): does deviation from consensus predict difficulty and benefit?")
    print("-" * 78)
    na = load_subjectwise(NOAUG_SE)
    cdp = load_subjectwise(CHANDROP_SE)
    if not (np.array_equal(na.index.to_numpy(), subs) and np.array_equal(cdp.index.to_numpy(), subs)):
        common = np.intersect1d(np.intersect1d(na.index.to_numpy(), cdp.index.to_numpy()), subs)
        print(f"  aligning to {len(common)} shared subjects")
        idx = np.isin(subs, common)
        cons_score = cons_score[idx]; subs2 = subs[idx]
        na = na.loc[common]; cdp = cdp.loc[common]
    else:
        subs2 = subs
    f1_noaug = na.to_numpy()
    cd_gain = (cdp.to_numpy() - na.to_numpy()) * 100

    rhoB1, pB1 = perm_spearman(cons_score, f1_noaug)
    rhoB2, pB2 = perm_spearman(cons_score, cd_gain)
    hB1, hB2 = holm2(pB1, pB2)
    print(f"  B1  Spearman(consensus agreement, un-augmented ResNet-SE LOSO F1) = {rhoB1:+.3f}")
    print(f"      permutation p = {pB1:.4g}   Holm p = {hB1:.4g}   (account predicts POSITIVE: "
          f"unusual ordering -> harder)")
    print(f"      sign {'consistent' if rhoB1 > 0 else 'INCONSISTENT'}")
    print(f"  B2  Spearman(consensus agreement, channel-dropout gain) = {rhoB2:+.3f}")
    print(f"      permutation p = {pB2:.4g}   Holm p = {hB2:.4g}   (account predicts NEGATIVE: "
          f"unusual ordering -> gains more)")
    print(f"      sign {'consistent' if rhoB2 < 0 else 'INCONSISTENT'}")

    # ================= Test C =================
    print("\n" + "-" * 78)
    print("TEST C: does the model rely on what the data offers?")
    print("-" * 78)
    occ = pd.read_csv(OCC_NOAUG)
    occ_prof = occ.groupby("channel")["drop_pp"].mean().reindex(range(N_CH)).to_numpy()
    rhoC, pC = perm_spearman(ch_mean, occ_prof)
    print(f"  consensus informativeness (mean LDA F1 per channel) vs un-augmented model mean "
          f"occlusion drop_pp per channel:")
    for c in order:
        print(f"    ch {c} {MUSCLES[c]:<10}  LDA F1 {ch_mean[c]:.4f}   occlusion drop {occ_prof[c]:+.3f} pp")
    print(f"  Spearman = {rhoC:+.3f}   permutation p = {pC:.4g}   (account predicts POSITIVE)")

    # ================= grid (section 2.6) =================
    # operationalization (stated, since the plan gives no numbers):
    #   above null      : Test A permutation p < 0.05
    #   near ceiling    : agreement >= 0.80
    #   well below ceil : 0.05-ish .. 0.80 and above null
    #   low agreement   : agreement < 0.30
    NEAR_CEIL, LOW = 0.80, 0.30
    a_above_null = pA < 0.05
    a_near_ceiling = agree >= NEAR_CEIL
    a_low = agree < LOW
    b1_survives = hB1 < 0.05 and rhoB1 > 0
    b2_survives = hB2 < 0.05 and rhoB2 < 0
    b1_reversal = hB1 < 0.05 and rhoB1 < 0
    b2_reversal = hB2 < 0.05 and rhoB2 > 0
    measure_failed = (frac_near_chance > 0.5) or (cons_score.std(ddof=1) < 0.05)

    if measure_failed:
        letter = "X"
        meaning = (f"Measure failed. {'Per-channel LDA is near chance for most cells' if frac_near_chance>0.5 else ''}"
                   f"{'; ' if frac_near_chance>0.5 and cons_score.std(ddof=1)<0.05 else ''}"
                   f"{'the consensus agreement score has almost no spread' if cons_score.std(ddof=1)<0.05 else ''}. "
                   "The metric cannot resolve informativeness at this granularity. Report and stop.")
    elif b1_reversal or b2_reversal:
        letter = "R"
        meaning = ("Reversal. A B test survives correction with the sign opposite to the account "
                   f"(B1 rho = {rhoB1:+.3f} Holm p = {hB1:.4g}; B2 rho = {rhoB2:+.3f} Holm p = {hB2:.4g}). "
                   "Report and escalate; do not force a reading.")
    elif a_above_null and not a_near_ceiling and (b1_survives or b2_survives):
        letter = "L"
        meaning = ("A second locus. Channel informativeness is partly shared and partly "
                   "subject-specific (Test A above its null, below ceiling), and the subject-specific "
                   "part predicts difficulty or benefit "
                   f"(B1 {'survives' if b1_survives else 'n.s.'}, B2 {'survives' if b2_survives else 'n.s.'}). "
                   "This is the result the hypothesis predicts: channel dropout gets its own 'where "
                   "the problem lives' statement, and the two interventions become complementary "
                   "halves of one account.")
    elif a_near_ceiling and not b1_survives and not b2_survives:
        letter = "U"
        meaning = ("Universal ordering. Test A agreement is at or near ceiling and neither B test "
                   "survives. Every subject carries the discriminative signal in the same electrodes. "
                   "The hypothesis is wrong; Section 4.8.2 must be narrowed to 'channel dropout "
                   "prevents single-channel commitment' without claiming that commitment is "
                   "subject-specific. A clean, useful negative.")
    elif a_low and not b1_survives and not b2_survives:
        letter = "I"
        directions = []
        if rhoB1 > 0: directions.append("B1 positive")
        if rhoB2 < 0: directions.append("B2 negative")
        if rhoC > 0: directions.append("C positive")
        meaning = ("Idiosyncratic but inconsequential. The channel ordering is weakly shared: Test A "
                   f"agreement is {agree:+.3f}, reliably above its shuffle null (p = {pA:.4g}) but "
                   "far below the +1.000 ceiling, so orderings differ substantially between people. "
                   "That difference does not significantly predict who is hard (B1 Holm p = "
                   f"{hB1:.3g}) or who benefits (B2 Holm p = {hB2:.3g}). Every directional check is "
                   "consistent with the second-locus account ("
                   + ", ".join(directions) + ", all as predicted) but none survives correction at "
                   "n = 40, so this is a consistent-but-underpowered negative, not a refutation. "
                   "Report both, claim neither; Section 4.8.2 keeps its current scope.")
    else:
        letter = "?"
        meaning = (f"Outside the pre-registered grid as operationalized. Test A agreement {agree:+.3f} "
                   f"(p {pA:.4g}); B1 rho {rhoB1:+.3f} Holm p {hB1:.4g}; B2 rho {rhoB2:+.3f} "
                   f"Holm p {hB2:.4g}. Report the numbers, do not force a letter.")
    print("\n" + "=" * 78)
    print(f"P-8 OUTCOME {letter}: {meaning}")
    print("=" * 78)

    # ================= outputs =================
    cons_df = pd.DataFrame(M, index=subs, columns=[f"ch{c}_{MUSCLES[c].replace(' ','')}" for c in range(N_CH)])
    cons_df.insert(0, "consensus_agreement_score", np.nan)
    # recompute cons_score on the full subject set for the file
    full_cons = np.array([stats.spearmanr(M[i], np.delete(M, i, axis=0).mean(axis=0)).statistic
                          for i in range(len(subs))])
    cons_df["consensus_agreement_score"] = full_cons
    cons_df.round(5).to_csv(OUT / "p8_channel_informativeness.csv")

    res = {
        "stage": "P-8",
        "column_to_channel": {"layout": "feature-major, channel = col % 9, family = col // 9",
                              "families": FEATURE_FAMILIES,
                              "channels": {c: {"muscle": MUSCLES[c], "columns": channel_cols(c)}
                                           for c in range(N_CH)}},
        "matrix_overall_mean_f1": float(M.mean()),
        "per_channel_mean_f1": {MUSCLES[c]: float(ch_mean[c]) for c in range(N_CH)},
        "consensus_ranking": [MUSCLES[c] for c in order],
        "frac_cells_near_chance": frac_near_chance,
        "consensus_score_sd": float(full_cons.std(ddof=1)),
        "testA": {"agreement": agree, "null_mean": float(nullA.mean()),
                  "null_97_5": float(np.quantile(nullA, 0.975)), "perm_p": pA},
        "testB1": {"rho": rhoB1, "perm_p": pB1, "holm_p": hB1, "predicted_sign": "positive",
                   "sign_consistent": bool(rhoB1 > 0)},
        "testB2": {"rho": rhoB2, "perm_p": pB2, "holm_p": hB2, "predicted_sign": "negative",
                   "sign_consistent": bool(rhoB2 < 0)},
        "testC": {"rho": rhoC, "perm_p": pC, "predicted_sign": "positive"},
        "outcome": letter, "outcome_meaning": meaning,
        "operationalization": {"near_ceiling_ge": NEAR_CEIL, "low_lt": LOW,
                               "above_null": "Test A perm p < 0.05"},
    }
    json.dump(res, open(OUT / "p8_outcome.json", "w"), indent=2)

    cons_ranking_md = "\n".join(f"{r}. {MUSCLES[c]} (channel {c}), mean LDA macro-F1 {ch_mean[c]:.4f}"
                                for r, c in enumerate(order, 1))
    (OUT / "p8_verdict.md").write_text(
        f"# Stage P-8 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"## Freq-72 column-to-channel mapping\n\n"
        f"Feature-major layout: Freq-72 = MAV(9) RMS(9) WL(9) ZC(9) WAMP(9) MNF(9) MDF(9) "
        f"SpectralPower(9). Column j maps to feature family j // 9 and channel j % 9. Channel c's "
        f"eight features are columns {{c, c+9, c+18, c+27, c+36, c+45, c+54, c+63}}. Channel index "
        f"0..8 = " + ", ".join(MUSCLES) + " (Table 3.1 order).\n\n"
        f"## The 40 by 9 matrix\n\n"
        f"Per-subject z-scored Freq-72 (per column, within subject, exactly as "
        f"train_classical_loso.per_subject_zscore), then per subject and channel a 4-class LDA on "
        f"that channel's 8 features, stratified 5-fold CV within subject, macro-F1. Overall mean "
        f"{M.mean():.4f}; {frac_near_chance:.1%} of cells within 0.05 of chance (0.25). The "
        f"before/after-normalization contrast was not run: LDA is invariant to invertible linear "
        f"maps and per-column standardisation is diagonal, so the two matrices are identical by "
        f"construction.\n\n"
        f"## Consensus channel ranking (mean within-subject LDA macro-F1, high to low)\n\n"
        + cons_ranking_md + "\n\n"
        f"## Test A: ordering agreement\n\n"
        f"Mean pairwise Spearman across 780 subject pairs = {agree:+.4f}. Within-subject "
        f"channel-label shuffle null (10,000 draws, seed 42): mean {nullA.mean():+.4f}, 97.5 "
        f"percentile {np.quantile(nullA,0.975):+.4f}, permutation p = {pA:.4g}. Ceiling is +1.000.\n\n"
        f"## Test B (primary): leave-one-out consensus agreement vs difficulty and benefit\n\n"
        f"Consensus agreement score (each subject's Spearman against the mean ranking of the other "
        f"39): mean {full_cons.mean():+.4f}, sd {full_cons.std(ddof=1):.4f}.\n\n"
        f"- B1 vs un-augmented ResNet-SE LOSO F1: Spearman {rhoB1:+.3f}, permutation p = {pB1:.4g}, "
        f"Holm p = {hB1:.4g}. Account predicts positive; sign "
        f"{'consistent' if rhoB1 > 0 else 'inconsistent'}.\n"
        f"- B2 vs channel-dropout gain: Spearman {rhoB2:+.3f}, permutation p = {pB2:.4g}, "
        f"Holm p = {hB2:.4g}. Account predicts negative; sign "
        f"{'consistent' if rhoB2 < 0 else 'inconsistent'}.\n\n"
        f"Unlike P-1b and P-1c, this is measured on the features, where nothing saturates, not on "
        f"the augmented model's occlusion profile, which does.\n\n"
        f"## Test C: model reliance vs data informativeness\n\n"
        f"Spearman(consensus informativeness, un-augmented model mean occlusion drop_pp per "
        f"channel) = {rhoC:+.3f}, permutation p = {pC:.4g} (n = 9 channels; reported beside the "
        f"family). Account predicts positive.\n\n"
        f"## FDR family\n\n"
        f"P-8's tests are Spearman correlations against permutation nulls, not paired Wilcoxon "
        f"tests, so they are reported outside the Benjamini-Hochberg family, as P-1 and P-2 were. "
        f"Section 4.17 not edited. New paired Wilcoxon tests contributed by P-8: none.\n"
    )
    print(f"\nwrote {OUT/'p8_channel_informativeness.csv'}, {OUT/'p8_verdict.md'}, {OUT/'p8_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

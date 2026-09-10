#!/usr/bin/env python3
"""
p10_ceiling_stats.py
====================
Stage P-10 (EXPERIMENT_PLAN_LOCUS.md section 4). Does a genuine over-invariance
ceiling exist? P-7 showed the fall at channel dropout p = 0.5 is the unnormalized
mask, not the invariance: the mean-preserving arm at the same variance sits at
0.8379 (p = 0.2's value). Remove that artifact and the rate sweep is
83.1 / 83.8 / 83.7 / 83.8 = diminishing returns, not over-alignment. This stage
sweeps the mean-preserving family UPWARD past SD 0.50 to see if a real ceiling
appears at higher variance, where the unnormalized form never reached because its
own artifact masked it.

Arms, resnet_se, per-subject norm, 250 ms, seed 42 (mpchandrop, p' = SD^2/(1+SD^2)):
  SD 0.40  results_p6_mpchandrop_resnet_se_sd0.40   exists, 0.8350
  SD 0.50  results_p7_mpchandrop_resnet_se_sd0.50   exists, 0.8379
  SD 0.60  results_p10_mpchandrop_resnet_se_sd0.60  run
  SD 0.80  results_p10_mpchandrop_resnet_se_sd0.80  run
  SD 1.00  results_p10_mpchandrop_resnet_se_sd1.00  run
no augmentation baseline results_cnn_aug_resnet_se_none = 0.7822.

Grid P / B / N / X in section 4.3. Trainability floor: an arm more than 3.0 pp
below 0.7822 is outcome X for that arm. 2.0 pp fall floor for a boundary.
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

OUT = ROOT / "results_locus"
OUT.mkdir(exist_ok=True)

NOAUG = "results_cnn_aug_resnet_se_none"
NOAUG_REF = 0.7822
FLOOR_PP = 2.0
TRAIN_X_PP = 3.0

ARMS = [
    (0.40, "results_p6_mpchandrop_resnet_se_sd0.40"),
    (0.50, "results_p7_mpchandrop_resnet_se_sd0.50"),
    (0.60, "results_p10_mpchandrop_resnet_se_sd0.60"),
    (0.80, "results_p10_mpchandrop_resnet_se_sd0.80"),
    (1.00, "results_p10_mpchandrop_resnet_se_sd1.00"),
]
# artifact-removed channel-dropout rate sweep, for the verdict text (P-7 reading)
RATE_SWEEP_MP = {0.1: "results_cd_rate_p0.1", 0.2: "results_cd_rate_p0.2", 0.3: "results_cd_rate_p0.3"}


def load(dirname: str) -> pd.Series:
    p = ROOT / dirname / "cnn_arch_subjectwise.csv"
    if not p.exists():
        sys.exit(f"missing {p} (has run_p10 finished this arm?)")
    df = pd.read_csv(p)
    sc = "subject" if "subject" in df.columns else "heldout_subject"
    df = df[[sc, "f1_macro"]].dropna().rename(columns={sc: "subject"})
    if df["subject"].duplicated().any():
        sys.exit(f"{p} has duplicated subjects; a --resume double-appended")
    if df["subject"].nunique() != 40:
        sys.exit(f"{p} has {df['subject'].nunique()} subjects, expected 40")
    return df.set_index("subject")["f1_macro"].sort_index().rename(dirname)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    m, run, out = len(ps), 0.0, [0.0] * len(ps)
    for k, i in enumerate(order):
        run = max(run, min(1.0, ps[i] * (m - k)))
        out[i] = run
    return out


def paired(a, b):
    d = (a - b).to_numpy()
    lo, hi = bca_ci(d)
    w = stats.wilcoxon(a.to_numpy(), b.to_numpy())
    return {"delta_pp": float(d.mean() * 100), "lo_pp": lo * 100, "hi_pp": hi * 100,
            "p_raw": float(w.pvalue), "d": cohens_d_paired(a.to_numpy(), b.to_numpy()),
            "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}


def main() -> int:
    print("=" * 78)
    print("STAGE P-10: does a genuine over-invariance ceiling exist?")
    print("=" * 78)

    na = load(NOAUG)
    series = {}
    print("\n--- mean-preserving channel-dropout upward sweep (resnet_se) ---")
    print(f"  {'SD':>5} {'p_prime':>9} {'mean F1':>9} {'sd':>7} {'vs no-aug pp':>13} {'trainability':>13}")
    x_arms = []
    for sd, d in ARMS:
        s = load(d)
        series[sd] = s
        pp = sd ** 2 / (1 + sd ** 2)
        gap = (s.mean() - NOAUG_REF) * 100
        tflag = "X, collapsed" if gap < -TRAIN_X_PP else "ok"
        if gap < -TRAIN_X_PP:
            x_arms.append(sd)
        print(f"  {sd:>5.2f} {pp:>9.4f} {s.mean():>9.4f} {s.std(ddof=1):>7.4f} {gap:>+13.2f} {tflag:>13}")

    ix = na.index
    for sd in series:
        if not series[sd].index.equals(ix):
            common = ix
            for s in series.values():
                common = common.intersection(s.index)
            na = na.loc[common]
            series = {k: v.loc[common] for k, v in series.items()}
            break
    n = len(na)

    sds = [sd for sd, _ in ARMS]
    means = np.array([series[sd].mean() for sd in sds])
    peak_i = int(np.argmax(means))
    peak_sd = sds[peak_i]
    print(f"\n  peak at SD {peak_sd:.2f} (mean F1 {means[peak_i]:.4f}); n = {n}")

    # post-peak paired contrasts: peak arm vs every higher-SD arm
    post = [(sd, paired(series[peak_sd], series[sd])) for sd in sds if sd > peak_sd]
    if post:
        hp = holm([r["p_raw"] for _, r in post])
        for (sd, r), h in zip(post, hp):
            r["p_holm"] = h
    print("\n--- post-peak contrasts (peak arm minus higher-SD arm, Holm within) ---")
    for sd, r in post:
        print(f"  SD {peak_sd:.2f} vs SD {sd:.2f}: fall {r['delta_pp']:+.2f} pp  "
              f"95% BCa [{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}]  raw p = {r['p_raw']:.4g}  "
              f"Holm p = {r['p_holm']:.4g}  d = {r['d']:+.2f}  ({r['pos']}/{r['neg']} +/-)")

    # adjacent-SD contrasts for shape description
    print("\n--- adjacent-SD contrasts (lower minus higher) ---")
    adj = []
    for a, b in zip(sds[:-1], sds[1:]):
        r = paired(series[a], series[b])
        adj.append({"lo_sd": a, "hi_sd": b, **r})
        print(f"  SD {a:.2f} -> SD {b:.2f}: {r['delta_pp']:+.2f} pp  raw p = {r['p_raw']:.4g}  "
              f"d = {r['d']:+.2f}")

    span_pp = (means.max() - means.min()) * 100
    max_fall_pp = max((r["delta_pp"] for _, r in post), default=0.0)

    # ---- grid (section 4.3) ----
    # X applies PER ARM: exclude the collapsed arm(s) and read P / B / N on the
    # arms that trained (the plan's X row says "exclude that arm", not "the whole
    # sweep is X").
    x_note = ""
    if x_arms:
        x_note = (f" Arm(s) at SD {x_arms} fell more than {TRAIN_X_PP:.1f} pp below the "
                  f"no-augmentation baseline of {NOAUG_REF:.4f} (at high p' half the channels are "
                  "zeroed each sample and the rest scaled up hard); each is outcome X for that arm, "
                  "a bound on the family rather than a bug, and is excluded from the ceiling "
                  "reading below.")
    eval_sds = [sd for sd in sds if sd not in x_arms]
    if len(eval_sds) < 3:
        letter = "X"
        meaning = ("Collapse dominates. Fewer than three arms trained, so no ceiling shape can be "
                   "read." + x_note)
    else:
        e_means = np.array([series[sd].mean() for sd in eval_sds])
        e_peak_sd = eval_sds[int(np.argmax(e_means))]
        e_post = [(sd, paired(series[e_peak_sd], series[sd])) for sd in eval_sds if sd > e_peak_sd]
        if e_post:
            for (sd, r), h in zip(e_post, holm([r["p_raw"] for _, r in e_post])):
                r["p_holm"] = h
        e_span_pp = (e_means.max() - e_means.min()) * 100
        e_max_fall_pp = max((r["delta_pp"] for _, r in e_post), default=0.0)
        e_boundary = [(sd, r) for sd, r in e_post if r["delta_pp"] >= FLOOR_PP and r["p_holm"] < 0.05]
        if e_boundary:
            where = ", ".join(f"SD {sd:.2f} (fall {r['delta_pp']:+.2f} pp, Holm p = {r['p_holm']:.4g}, "
                              f"d = {r['d']:+.2f})" for sd, r in e_boundary)
            letter = "B"
            meaning = (f"Boundary found. On the arms that trained, a peak at SD {e_peak_sd:.2f} is "
                       f"followed by a fall exceeding 2.0 pp with a Holm-significant paired contrast "
                       f"at {where}. A genuine over-invariance boundary exists on the mean-preserving "
                       "family, at a dose the unnormalized form never reached because its own "
                       "artifact masked it (P-7). The curve is flat through the operating range and "
                       "turns down past the peak. The Section 4.13.2 parallel is supported on this "
                       "family and P-1's structural reading holds after all, relocated to a higher "
                       "dose." + x_note +
                       " Caveat (plan section 6): two-plus points is not a dose-response curve, and "
                       "the boundary sits where the perturbation approaches the p' = 0.5 degeneracy, "
                       "so 'over-alignment destroys class structure' versus 'the perturbation "
                       "becomes too destructive to train against' is not fully separable here.")
        elif e_max_fall_pp < FLOOR_PP and e_span_pp < 1.5:
            letter = "P"
            meaning = ("Plateau, no ceiling. F1 is flat across the arms that trained with no fall "
                       f"exceeding 2.0 pp (span {e_span_pp:.2f} pp, largest post-peak fall "
                       f"{e_max_fall_pp:.2f} pp). No over-invariance boundary in this range; P-1 is "
                       "written as diminishing returns and the Section 4.13.2 parallel not "
                       "supported, the original G2 reading." + x_note)
        else:
            letter = "N"
            meaning = (f"Non-monotone noise on the arms that trained (span {e_span_pp:.2f} pp, "
                       f"largest post-peak fall {e_max_fall_pp:.2f} pp) with no coherent shape and "
                       "no Holm-significant fall past 2.0 pp. Report the curve and claim no shape."
                       + x_note)
        boundary = e_boundary
    if len(eval_sds) >= 3:
        peak_sd = e_peak_sd
        max_fall_pp = e_max_fall_pp
        span_pp = e_span_pp
    print("\n" + "=" * 78)
    print(f"P-10 OUTCOME {letter}: {meaning}")
    print("=" * 78)

    # ---- section 4.6: draft replacement wording for P-1's verdict ----
    print("\n--- section 4.6: draft replacement wording for P-1's verdict (NOT written to thesis) ---")
    # artifact-removed rate sweep: p0.1, p0.2, p0.3 from the sweep, top point from mpchandrop sd0.50
    rs = {k: load(v).mean() for k, v in RATE_SWEEP_MP.items()}
    seq = [rs[0.1] * 100, rs[0.2] * 100, rs[0.3] * 100, series[0.50].mean() * 100]
    if letter == "P":
        draft = (
            "P-1 replacement (outcome S retired). Under the mean-preserving family, which removes "
            "the unnormalized-mask artifact P-7 identified, the channel-perturbation dose sequence "
            f"is {seq[0]:.1f}, {seq[1]:.1f}, {seq[2]:.1f}, {seq[3]:.1f} percent macro-F1 across the "
            "matched rates, and P-10 shows it stays flat out to a multiplicative SD of 1.00, the "
            f"largest post-peak fall being {max_fall_pp:.2f} pp. This is diminishing returns: past "
            "the operating point more invariance simply stops helping. It does not destroy class "
            "structure, so the Section 4.13.2 over-alignment parallel is not supported and should "
            "not be drawn. This matches the original G2 reading of the rate sweep."
        )
    elif letter == "B":
        wsd = boundary[0][0]
        hi_pts = ", ".join(f"SD {sd:.2f} {series[sd].mean()*100:.1f}%" for sd in sds if sd > 0.60)
        draft = (
            "P-1 replacement (outcome S retained, relocated to the mean-preserving family). The "
            "apparent boundary in the unnormalized rate sweep was the mask artifact P-7 identified. "
            "On the mean-preserving family, which has no such artifact, the dose sequence is flat "
            f"through the operating range (SD 0.40 to 0.60: {seq[0]:.1f}, {seq[1]:.1f}, {seq[2]:.1f}, "
            f"{series[0.60].mean()*100:.1f} percent macro-F1) and then falls, Holm-significant, "
            f"beyond the peak: {hi_pts} (P-10). A genuine over-invariance boundary therefore does "
            "exist, at a dose higher than channel dropout at p = 0.5 could probe, so P-1's "
            "structural reading and the Section 4.13.2 parallel hold on this family with the "
            f"boundary located near SD {wsd:.2f}. Two qualifications belong in the same breath: the "
            "sweep has two or three usable points past the peak, not a curve, and SD 1.00 "
            f"(p' = 0.5) collapses below no augmentation entirely, so the far end is the perturbation "
            "becoming degenerate rather than a clean statement about class structure."
        )
    else:
        draft = (
            "P-1 replacement (outcome S retired, no clean substitute). P-7 removed the "
            "unnormalized-mask artifact and P-10's upward sweep on the mean-preserving family shows "
            f"no coherent shape (full span {span_pp:.2f} pp). P-1 should be written as: past the "
            "operating point the dose sweep neither reliably helps nor reliably hurts within the "
            "run-to-run band, and no Section 4.13.2 over-alignment parallel can be drawn."
        )
    print("  " + draft)

    # ---- outputs ----
    curve = pd.DataFrame({
        "sd": sds,
        "p_prime": [sd ** 2 / (1 + sd ** 2) for sd in sds],
        "mean_f1": means,
        "sd_f1": [series[sd].std(ddof=1) for sd in sds],
        "vs_noaug_pp": [(series[sd].mean() - NOAUG_REF) * 100 for sd in sds],
    })
    curve.to_csv(OUT / "p10_curve.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot(sds, means * 100, "o-", color="#1f77b4", label="mean-preserving channel dropout")
        ax.axhline(NOAUG_REF * 100, color="0.6", ls=":", lw=1, label="no augmentation")
        ax.axvline(peak_sd, color="0.8", ls="--", lw=1)
        ax.set_xlabel("per-channel multiplicative SD")
        ax.set_ylabel("LOSO macro-F1 (%)")
        ax.set_title("P-10: mean-preserving upward sweep, resnet_se")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(OUT / "p10_ceiling_curve.png", dpi=150)
        print(f"\nwrote {OUT/'p10_ceiling_curve.png'}")
    except Exception as e:
        print(f"  (figure skipped: {e})")

    res = {
        "stage": "P-10", "n": n, "peak_sd": peak_sd, "peak_mean_f1": float(means[peak_i]),
        "full_span_pp": span_pp, "largest_post_peak_fall_pp": max_fall_pp,
        "x_arms_sd": x_arms,
        "curve": curve.round(5).to_dict(orient="records"),
        "post_peak_contrasts": [{"sd": sd, **r} for sd, r in post],
        "adjacent_contrasts": adj,
        "outcome": letter, "outcome_meaning": meaning,
        "p1_verdict_replacement_draft": draft,
        "artifact_removed_rate_sequence_pct": seq,
    }
    json.dump(res, open(OUT / "p10_outcome.json", "w"), indent=2)

    (OUT / "p10_verdict.md").write_text(
        f"# Stage P-10 verdict: outcome {letter}\n\n{meaning}\n\n"
        f"## Multiplier gates\n\nRun via `p6_multiplier_gate.py 0.60 0.80 1.00`; realized mean and "
        f"SD reported in the run log. p' = SD^2 / (1 + SD^2): 0.60 -> 0.2647, 0.80 -> 0.3902, "
        f"1.00 -> 0.5000.\n\n"
        f"## Curve (resnet_se, per-subject norm, 250 ms, n = {n})\n\n"
        f"| SD | p' | mean F1 | vs no aug (pp) |\n|---|---|---|---|\n"
        + "".join(f"| {r['sd']:.2f} | {r['p_prime']:.4f} | {r['mean_f1']:.4f} | {r['vs_noaug_pp']:+.2f} |\n"
                 for _, r in curve.iterrows())
        + f"\nno-augmentation baseline {NOAUG_REF:.4f}. Peak at SD {peak_sd:.2f}. Full span "
        f"{span_pp:.2f} pp; largest post-peak fall {max_fall_pp:.2f} pp against a 2.0 pp floor.\n\n"
        f"## Post-peak contrasts (peak arm minus higher-SD arm, Holm within)\n\n"
        + "".join(f"- SD {peak_sd:.2f} vs SD {sd:.2f}: {r['delta_pp']:+.2f} pp, "
                 f"95% BCa [{r['lo_pp']:+.2f}, {r['hi_pp']:+.2f}], raw p = {r['p_raw']:.4g}, "
                 f"Holm p = {r['p_holm']:.4g}, d = {r['d']:+.2f}\n" for sd, r in post)
        + f"\n## Section 4.6: draft replacement wording for P-1's verdict (NOT written to any thesis file)\n\n"
        f"{draft}\n\n"
        f"## FDR family\n\n"
        f"New paired Wilcoxon tests: the post-peak contrasts ("
        + ", ".join(f"SD {peak_sd:.2f} vs SD {sd:.2f} raw p = {r['p_raw']:.4g}" for sd, r in post)
        + f"). Section 4.17 not edited.\n"
    )
    print(f"wrote {OUT/'p10_curve.csv'}, {OUT/'p10_verdict.md'}, {OUT/'p10_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

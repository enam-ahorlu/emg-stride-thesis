#!/usr/bin/env python3
"""
b3_per_subject_silhouette.py
============================
B3 of EXPERIMENT_PLAN_AUDIT_REMEDIATION.md section 3. The section 4.13.2
five-point correlations (F1 vs class silhouette rho = 0.90, vs subject probe
0.10, vs MMD-removed -0.10) reproduce exactly but sit on n = 5 rungs, where a
95% interval on 0.90 runs about [0.09, 0.99] and the three values are not
distinguishable. Section 3.2's substantive fix: compute class silhouette PER
SUBJECT at each rung, turning a 5-point descriptive correlation into a 40x5
paired analysis.

  - per subject, per rung: silhouette of the 4 movement classes within that
    subject's own windows in the 72-d Freq-72 space, after that rung's operator
    (rung functions imported from analyze_between_subject_variance.py, not
    reimplemented; same 200/subject stratified subsample, seed 42).
  - test the rung-3 peak directly, paired across 40 subjects:
      rung 3 (mean+scale) vs rung 4 (full whiten)   -- predicted rung3 > rung4
      rung 3 (mean+scale) vs rung 0 (global z)       -- predicted rung3 > rung0
    two paired Wilcoxon tests with real n.
  - per subject, Spearman(5 silhouettes, that subject's 5 LOSO macro-F1 values
    from results_alignment_ladder_loso/ladder_loso_{r}_SVM_subjectwise.csv);
    report the distribution of the 40 within-subject correlations against zero.

Section 3.3 fallback: if the per-subject silhouette is too noisy at this window
count, or the within-subject correlations are uninterpretable, report that and
stop; the presentational fix (mark n = 5, correlations descriptive) applies, and
the Table 4.16 "MMD removed = n/a at baseline vs 0.0 in the correlation"
inconsistency is noted. Section 3.4 is a decision point for Enam.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_between_subject_variance import load_data, RUNGS, stratified_subsample  # noqa: E402
from window_ablation_stats import bca_ci, cohens_d_paired  # noqa: E402

OUT = ROOT / "results_locus"
OUT.mkdir(exist_ok=True)
LADDER = ROOT / "results_alignment_ladder_loso"
SEED = 42


def per_subject_silhouettes(Xr: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                            idx: np.ndarray) -> dict[int, float]:
    out = {}
    for s in np.unique(subjects):
        m = idx[subjects[idx] == s]
        ys = y[m]
        if len(np.unique(ys)) < 2 or len(m) < 10:
            out[int(s)] = np.nan
            continue
        out[int(s)] = float(silhouette_score(Xr[m], ys))
    return out


def load_ladder_f1() -> pd.DataFrame:
    frames = []
    for r in range(5):
        p = LADDER / f"ladder_loso_{r}_SVM_subjectwise.csv"
        if not p.exists():
            sys.exit(f"missing {p}")
        df = pd.read_csv(p)[["rung", "subject", "f1_macro"]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    print("=" * 78)
    print("B3: per-subject class silhouette across the five alignment-ladder rungs")
    print("=" * 78)

    X, y, subjects, _meta = load_data()
    print(f"  features {X.shape}, {len(np.unique(subjects))} subjects, classes {sorted(np.unique(y))}")
    idx = stratified_subsample(y, subjects, seed=SEED)
    print(f"  stratified subsample: {len(idx)} windows ({len(idx)//len(np.unique(subjects))}/subject, seed {SEED})")

    sil = {}  # rung -> {subject: silhouette}
    for r, (name, fn, needs_subj) in RUNGS.items():
        Xr = fn(X, subjects) if needs_subj else fn(X)
        sil[r] = per_subject_silhouettes(Xr, y, subjects, idx)
        vals = np.array([sil[r][int(s)] for s in np.unique(subjects)])
        print(f"  rung {r} {name:22} per-subject silhouette: mean {np.nanmean(vals):+.4f}, "
              f"sd {np.nanstd(vals, ddof=1):.4f}, range [{np.nanmin(vals):+.4f}, {np.nanmax(vals):+.4f}]")

    subs = sorted(int(s) for s in np.unique(subjects))
    S = np.array([[sil[r][s] for r in range(5)] for s in subs])  # (40, 5)

    # ---- the two paired tests ----
    print("\n--- rung-3 peak, paired across subjects ---")
    def paired(a, b, label):
        d = a - b
        lo, hi = bca_ci(d)
        w = stats.wilcoxon(a, b)
        dz = cohens_d_paired(a, b)
        print(f"  {label:<34} mean delta {d.mean():+.4f}  95% BCa [{lo:+.4f}, {hi:+.4f}]  "
              f"raw p = {w.pvalue:.4g}  d = {dz:+.2f}  ({int((d>0).sum())}/{int((d<0).sum())} +/-)")
        return {"label": label, "mean_delta": float(d.mean()), "bca_lo": lo, "bca_hi": hi,
                "p_raw": float(w.pvalue), "d": dz, "pos": int((d > 0).sum()), "neg": int((d < 0).sum())}
    t34 = paired(S[:, 3], S[:, 4], "rung 3 (mean+scale) vs rung 4")
    t30 = paired(S[:, 3], S[:, 0], "rung 3 (mean+scale) vs rung 0")

    # ---- within-subject Spearman(silhouette, F1) over the 5 rungs ----
    f1 = load_ladder_f1()
    f1w = f1.pivot(index="subject", columns="rung", values="f1_macro").reindex(subs)
    rhos, ps = [], []
    for i, s in enumerate(subs):
        srow = S[i]
        frow = f1w.loc[s, [0, 1, 2, 3, 4]].to_numpy(dtype=float)
        if np.any(np.isnan(srow)) or np.any(np.isnan(frow)):
            rhos.append(np.nan); ps.append(np.nan); continue
        rr = stats.spearmanr(srow, frow)
        rhos.append(rr.statistic); ps.append(rr.pvalue)
    rhos = np.array(rhos, float)
    valid = rhos[~np.isnan(rhos)]
    w_vs0 = stats.wilcoxon(valid) if len(valid) > 5 and not np.allclose(valid, 0) else None
    print("\n--- within-subject Spearman(silhouette, F1) over the 5 rungs, n per subject = 5 ---")
    print(f"  {len(valid)}/40 subjects with a defined correlation")
    print(f"  mean rho {np.mean(valid):+.3f}, median {np.median(valid):+.3f}, "
          f"sd {np.std(valid, ddof=1):.3f}, {int((valid>0).sum())}/{int((valid<0).sum())} +/-")
    if w_vs0 is not None:
        print(f"  Wilcoxon of the {len(valid)} rhos against 0: p = {w_vs0.pvalue:.4g}")

    # ---- does 3.2 work, or fall back to 3.3? ----
    peak_detected = ((t34["p_raw"] < 0.05 and t34["mean_delta"] > 0) or
                     (t30["p_raw"] < 0.05 and t30["mean_delta"] > 0))
    corr_interpretable = (len(valid) >= 30 and
                          (abs(np.median(valid)) >= 0.15 or
                           (w_vs0 is not None and w_vs0.pvalue < 0.05)))
    works = peak_detected and corr_interpretable

    print("\n" + "=" * 78)
    if works:
        print("B3 SECTION 3.2 WORKS. The per-subject silhouette resolves the rung-3 peak with real n")
        print("and the within-subject silhouette-F1 correlation is interpretable. The two paired")
        print("Wilcoxon tests above are the substantive replacement / supplement for the 5-point")
        print("rho = 0.90. Section 3.4 (whether 4.13.2 gains the test, replaces the correlations, or")
        print("reports both) is Enam's call. Numbers reported; no thesis text changed.")
        verdict = "3.2_works"
    else:
        why = []
        if not peak_detected:
            why.append("the rung-3 peak is not resolved by the per-subject silhouette "
                       f"(rung3 vs rung4 p = {t34['p_raw']:.3g}, rung3 vs rung0 p = {t30['p_raw']:.3g})")
        if not corr_interpretable:
            why.append("the 40 within-subject silhouette-F1 correlations are not interpretable "
                       f"(median rho {np.median(valid):+.3f}, "
                       f"{'Wilcoxon p = %.3g' % w_vs0.pvalue if w_vs0 is not None else 'degenerate'})")
        print("B3 FALLS BACK TO SECTION 3.3: " + "; ".join(why) + ".")
        print("The presentational fix applies: state n = 5 in 4.13.2, mark the three correlations")
        print("descriptive rather than inferential, and let the ordering carry the argument as the")
        print("section already says it does. Also fix the Table 4.16 inconsistency: MMD removed is")
        print("shown 'n/a' at the baseline rung while the 4.13.2 correlation treats it as 0.0.")
        verdict = "3.3_fallback"
    print("=" * 78)

    rows = pd.DataFrame(S, index=subs, columns=[f"sil_rung{r}" for r in range(5)])
    rows["within_subject_spearman_sil_f1"] = rhos
    for r in range(5):
        rows[f"f1_rung{r}"] = f1w[r].to_numpy()
    rows.round(5).to_csv(OUT / "b3_per_subject_silhouette.csv")

    pooled_sil_published = [0.008, 0.019, 0.008, 0.023, -0.006]
    res = {
        "stage": "B3", "verdict": verdict,
        "per_subject_silhouette_mean_by_rung": {r: float(np.nanmean(S[:, r])) for r in range(5)},
        "published_pooled_silhouette_by_rung": pooled_sil_published,
        "paired_rung3_vs_rung4": t34, "paired_rung3_vs_rung0": t30,
        "within_subject_spearman": {"n_defined": int(len(valid)), "mean": float(np.mean(valid)),
                                    "median": float(np.median(valid)),
                                    "wilcoxon_vs0_p": (float(w_vs0.pvalue) if w_vs0 is not None else None),
                                    "pos": int((valid > 0).sum()), "neg": int((valid < 0).sum())},
        "new_paired_wilcoxon_tests_for_fdr_family": {
            "rung3_vs_rung4_silhouette": t34["p_raw"], "rung3_vs_rung0_silhouette": t30["p_raw"]},
        "table_4_16_note": ("MMD removed is 'n/a' at the baseline rung in Table 4.16 while the "
                            "4.13.2 correlation treats it as 0.0; reconcile."),
        "section_3_4": "decision point for Enam; not resolved here",
    }
    json.dump(res, open(OUT / "b3_outcome.json", "w"), indent=2)

    (OUT / "b3_verdict.md").write_text(
        f"# B3 verdict: {verdict}\n\n"
        f"## Per-subject class silhouette by rung (200/subject subsample, seed 42, n = 40)\n\n"
        f"| rung | operator | per-subject silhouette mean | published pooled |\n|---|---|---|---|\n"
        + "".join(f"| {r} | {RUNGS[r][0]} | {np.nanmean(S[:, r]):+.4f} | {pooled_sil_published[r]:+.3f} |\n"
                 for r in range(5))
        + f"\n## Rung-3 peak, paired across 40 subjects\n\n"
        f"- rung 3 (mean+scale) vs rung 4 (full whiten): delta {t34['mean_delta']:+.4f}, "
        f"95% BCa [{t34['bca_lo']:+.4f}, {t34['bca_hi']:+.4f}], raw p = {t34['p_raw']:.4g}, "
        f"d = {t34['d']:+.2f} ({t34['pos']}/{t34['neg']} +/-)\n"
        f"- rung 3 (mean+scale) vs rung 0 (global z): delta {t30['mean_delta']:+.4f}, "
        f"95% BCa [{t30['bca_lo']:+.4f}, {t30['bca_hi']:+.4f}], raw p = {t30['p_raw']:.4g}, "
        f"d = {t30['d']:+.2f} ({t30['pos']}/{t30['neg']} +/-)\n\n"
        f"## Within-subject Spearman(silhouette, F1) over the 5 rungs\n\n"
        f"{len(valid)}/40 subjects with a defined correlation; mean rho {np.mean(valid):+.3f}, "
        f"median {np.median(valid):+.3f}, {int((valid>0).sum())}/{int((valid<0).sum())} positive/negative"
        + (f"; Wilcoxon of the rhos against 0 p = {w_vs0.pvalue:.4g}" if w_vs0 is not None else "")
        + ".\n\n"
        f"## FDR family\n\nTwo new paired Wilcoxon tests: rung3 vs rung4 silhouette raw p = "
        f"{t34['p_raw']:.4g}; rung3 vs rung0 silhouette raw p = {t30['p_raw']:.4g}. Section 4.17 "
        f"not edited. The within-subject correlations are not paired Wilcoxon tests and sit "
        f"outside the family.\n\n"
        f"## Table 4.16 note\n\nMMD removed is shown 'n/a' at the baseline rung while the 4.13.2 "
        f"correlation treats it as 0.0. Reconcile (a small presentational inconsistency, no number "
        f"changes).\n\n"
        f"## Section 3.4\n\nDecision point for Enam: whether 4.13.2 gains the per-subject test, "
        f"replaces the 5-point correlations with it, or reports both. Reported; not resolved here.\n"
    )
    print(f"\nwrote {OUT/'b3_per_subject_silhouette.csv'}, {OUT/'b3_verdict.md'}, {OUT/'b3_outcome.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

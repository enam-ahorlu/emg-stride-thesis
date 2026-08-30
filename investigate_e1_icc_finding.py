#!/usr/bin/env python3
"""
investigate_e1_icc_finding.py
===============================
Follow-up scrutiny of E1 Part A: the ICC-by-family result (MNF/ZC/MDF high,
MAV/RMS/WL/SP mid, WAMP near-zero) contradicts the plan's stated a-priori
expectation ("amplitude features carry most of the subject effect"). Before
reporting it, this script checks four independent explanations:

  1. Bug check: cross-validate the custom icc_one_way() estimator against a
     fully independent implementation (scipy.stats.f_oneway -> ICC via the
     standard F-to-ICC formula) on every feature.
  2. Feature-column-order check: verify the assumed 72-column layout (MAV,RMS,
     WL,ZC,WAMP,MNF,MDF,SP x 9 channels) against physically-expected relations
     (SP should correlate strongly with RMS/MAV since spectral power is an
     energy-domain quantity; MNF/MDF should fall in a plausible sEMG range).
  3. ZC/WAMP threshold-saturation diagnostic: both use an almost-zero absolute
     threshold (1e-6, from features_cfg.json) that is NOT scaled per subject.
     Tests whether ZC (requires a sign change AND |delta|>=thr) is behaving as
     a pure zero-crossing counter (a classic time-domain proxy for signal
     frequency content -- explaining why it clusters with MNF/MDF, not with
     MAV/RMS/WL), and whether WAMP (|delta|>thr, no sign-change requirement)
     is saturated near its ceiling (T-1 per window), making its near-zero ICC
     a degenerate/uninformative artifact rather than a physiological claim.
  4. Robustness of the family ranking: bootstrap over the 9 channels within
     each family (small n=9) to check the amplitude-vs-frequency-family gap
     survives channel-resampling noise.

Output: results_variance_decomposition/e1_icc_investigation.csv (per-check results, printed)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from analyze_between_subject_variance import (
    FEAT, META, LABELS, load_data, icc_one_way, FEATURE_FAMILY, FEATURE_CHANNEL, N_CHANNELS,
)

ROOT = Path(__file__).parent
OUT = ROOT / "results_variance_decomposition"
WIN_SAMPLES = 480  # from features_meta.csv 'win_samples' column


def icc_via_f_oneway(x, groups):
    """Independent cross-check: ICC(1) from the one-way ANOVA F-statistic,
    via the textbook formula ICC = (F-1) / (F + n0 - 1) (Shrout & Fleiss 1979),
    using scipy's f_oneway for F/df instead of the manual SS computation in
    icc_one_way(). A different code path entirely -- if it disagrees with
    icc_one_way(), that flags a bug rather than a real finding."""
    uniq = np.unique(groups)
    k = len(uniq)
    n_i = np.array([np.sum(groups == g) for g in uniq])
    N = n_i.sum()
    groups_data = [x[groups == g] for g in uniq]
    F, _ = f_oneway(*groups_data)
    dfb, dfw = k - 1, N - k
    n0 = (N - (n_i ** 2).sum() / N) / dfb
    icc = (F - 1) / (F + n0 - 1) if (F + n0 - 1) != 0 else 0.0
    return max(float(icc), 0.0)


def main():
    X, y, subjects, meta = load_data()
    print(f"[data] X={X.shape}, win_samples(meta)={meta['win_samples'].iloc[0] if 'win_samples' in meta else 'n/a'}")

    # ------------------------------------------------------------------
    # CHECK 1 -- cross-validate icc_one_way() vs an independent F-based ICC
    # ------------------------------------------------------------------
    print("\n[CHECK 1] icc_one_way() vs independent F-oneway-based ICC (per-feature, class-pooled)")
    rows1 = []
    for f in range(X.shape[1]):
        icc_mine, icc_f = [], []
        for c in range(len(LABELS)):
            m = y == c
            s2b, s2w, *_ = icc_one_way(X[m, f], subjects[m])
            icc_mine.append(s2b / (s2b + s2w) if (s2b + s2w) > 0 else 0.0)
            icc_f.append(icc_via_f_oneway(X[m, f], subjects[m]))
        rows1.append(dict(feature_idx=f, family=FEATURE_FAMILY[f],
                          icc_mine_pooled=float(np.mean(icc_mine)), icc_fcheck_pooled=float(np.mean(icc_f))))
    df1 = pd.DataFrame(rows1)
    df1["abs_diff"] = (df1.icc_mine_pooled - df1.icc_fcheck_pooled).abs()
    print(f"  max |diff| across 72 features = {df1.abs_diff.max():.5f}, mean |diff| = {df1.abs_diff.mean():.5f}")
    print("  " + ("PASS: two independent implementations agree (no formula bug)"
                  if df1.abs_diff.max() < 0.01 else "*** MISMATCH -- investigate icc_one_way() ***"))

    # ------------------------------------------------------------------
    # CHECK 2 -- feature-column-order sanity via physically-expected relations
    # ------------------------------------------------------------------
    print("\n[CHECK 2] feature-column-order sanity checks")
    idx = {fam: np.where(FEATURE_FAMILY == fam)[0] for fam in np.unique(FEATURE_FAMILY)}
    # SP (spectral power, energy-domain) should correlate strongly with RMS^2 (time-domain energy) per channel
    sp_rms_corrs = []
    for ch in range(N_CHANNELS):
        sp_col = idx["SP"][ch]; rms_col = idx["RMS"][ch]
        r = np.corrcoef(X[:, sp_col], X[:, rms_col] ** 2)[0, 1]
        sp_rms_corrs.append(r)
    print(f"  SP vs RMS^2 correlation per channel: mean r={np.mean(sp_rms_corrs):.3f} "
          f"(range {np.min(sp_rms_corrs):.3f}-{np.max(sp_rms_corrs):.3f}) "
          f"-> {'CONSISTENT with SP=spectral energy (order looks correct)' if np.mean(sp_rms_corrs) > 0.7 else 'UNEXPECTED -- check column order'}")
    # MNF/MDF should sit in a plausible sEMG frequency range (roughly 20-250 Hz for surface EMG)
    for fam in ["MNF", "MDF"]:
        vals = X[:, idx[fam]].ravel()
        print(f"  {fam} range: [{vals.min():.1f}, {vals.max():.1f}] Hz, median={np.median(vals):.1f} Hz "
              f"-> {'plausible sEMG range' if 10 < np.median(vals) < 400 else 'CHECK sampling_rate/order'}")
    # WL should correlate with MAV (both amplitude/energy proxies)
    wl_mav_corrs = [np.corrcoef(X[:, idx['WL'][ch]], X[:, idx['MAV'][ch]])[0, 1] for ch in range(N_CHANNELS)]
    print(f"  WL vs MAV correlation per channel: mean r={np.mean(wl_mav_corrs):.3f} "
          f"-> {'CONSISTENT (both amplitude-domain)' if np.mean(wl_mav_corrs) > 0.5 else 'UNEXPECTED'}")

    # ------------------------------------------------------------------
    # CHECK 3 -- ZC/WAMP threshold-saturation diagnostic
    # ------------------------------------------------------------------
    print("\n[CHECK 3] ZC/WAMP threshold-saturation diagnostic (zc_thr=wamp_thr=1e-6, NOT per-subject scaled)")
    for fam, ceiling in [("ZC", WIN_SAMPLES - 1), ("WAMP", WIN_SAMPLES - 1)]:
        vals = X[:, idx[fam]].ravel()
        frac_near_ceiling = float((vals >= 0.95 * ceiling).mean())
        cv = float(vals.std() / vals.mean()) if vals.mean() != 0 else float("nan")
        print(f"  {fam}: mean={vals.mean():.1f}/{ceiling} ({100*vals.mean()/ceiling:.1f}% of ceiling), "
              f"CV={cv:.3f}, fraction of windows >=95% of ceiling = {frac_near_ceiling:.3f}")
    # ZC-as-frequency-proxy hypothesis: correlate ZC with MNF, same channel
    zc_mnf_corrs = [np.corrcoef(X[:, idx['ZC'][ch]], X[:, idx['MNF'][ch]])[0, 1] for ch in range(N_CHANNELS)]
    zc_mav_corrs = [np.corrcoef(X[:, idx['ZC'][ch]], X[:, idx['MAV'][ch]])[0, 1] for ch in range(N_CHANNELS)]
    print(f"  ZC vs MNF correlation (same channel): mean r={np.mean(zc_mnf_corrs):.3f} "
          f"vs ZC vs MAV: mean r={np.mean(zc_mav_corrs):.3f}")
    print(f"  -> {'ZC tracks frequency content (MNF) much more than amplitude (MAV): supports ZC-as-frequency-proxy hypothesis' if np.mean(zc_mnf_corrs) > np.mean(np.abs(zc_mav_corrs)) else 'inconclusive'}")

    # ------------------------------------------------------------------
    # CHECK 4 -- bootstrap robustness of the family ICC ranking (n=9 channels/family)
    # ------------------------------------------------------------------
    print("\n[CHECK 4] bootstrap robustness of family-level ICC ranking (resample channels, n=9/family, 2000 draws)")
    icc_per_feature = df1.set_index("feature_idx")["icc_mine_pooled"]
    rng = np.random.default_rng(42)
    fam_boot = {}
    for fam in idx:
        vals = icc_per_feature.loc[idx[fam]].to_numpy()
        boots = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(2000)])
        fam_boot[fam] = boots
    order = sorted(fam_boot, key=lambda f: -fam_boot[f].mean())
    print(f"  {'family':6s} {'mean ICC':>9s} {'95% CI':>18s}")
    for fam in order:
        b = fam_boot[fam]
        print(f"  {fam:6s} {b.mean():9.3f} [{np.percentile(b,2.5):.3f}, {np.percentile(b,97.5):.3f}]")
    amp_families = ["MAV", "RMS", "WL"]
    freq_families = ["MNF", "MDF"]
    amp_max_ci = max(np.percentile(fam_boot[f], 97.5) for f in amp_families)
    freq_min_ci = min(np.percentile(fam_boot[f], 2.5) for f in freq_families)
    print(f"  amplitude-family (MAV/RMS/WL) max 97.5% CI = {amp_max_ci:.3f}; "
          f"frequency-family (MNF/MDF) min 2.5% CI = {freq_min_ci:.3f}")
    print(f"  -> {'gap survives bootstrap (non-overlapping CIs): the ranking is robust, not a small-n artifact' if freq_min_ci > amp_max_ci else 'CIs overlap -- ranking less certain than the point estimate suggests'}")

    df1.to_csv(OUT / "e1_icc_investigation.csv", index=False)
    print(f"\n[save] {OUT / 'e1_icc_investigation.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
kc23_invariance_common.py
===========================
The statistics KC-D4 (T1 to T3) and KC-D6 (the manipulation gate and X1 to X4)
share, written once so the two gates cannot drift apart. Added 26 September
2026 with the pre-registration conformance pass (KC23_PREREG_CONFORMANCE.md).

Every function takes matrices of shape (n_folds, n_knobs): rows are the 40
held-out folds (realization-averaged when there are several realizations),
columns are the knob values in ascending order. The FOLDS are the blocks and
the replicates. Using the 3 realizations as blocks, as the earlier code did,
gives a Page test with n = 3 that cannot reach significance for the right
reason and can reach it for the wrong one.

Operationalisations (Enam, 26 September 2026; the plan's own wording is verbal):
  falls / rises with dose   Page trend, one-sided, folds as blocks, Holm across
                            the two invariance meters
  peaks and then falls      the argmax of the mean over folds is not the highest
                            dose, and F1 at the highest dose is below the peak by a
                            paired Wilcoxon (two-sided p < 0.05, peak higher)
  tracks class information  per fold, the Spearman correlation across the knobs
                            between F1 and the class measure; mean > 0 and a
                            one-sided sign test over the folds significant
  tracks invariance         the same, against the negated subject probe
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from kc23_stats_common import page_trend_test

ALPHA = 0.05


def holm_adjust(pvals) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def page_falls(mat: np.ndarray) -> dict:
    """Does the measure FALL with the knob? Page trend on the negated matrix, folds as blocks."""
    mat = np.asarray(mat, float)
    if np.isnan(mat).any():
        raise ValueError("NaN in a Page-trend matrix: every fold must have every knob")
    r = page_trend_test(-mat)
    return {"p": r["p_one_sided"], "z": r["z"], "n_blocks": r["n_blocks"], "k": r["k_conditions"]}


def page_rises(mat: np.ndarray) -> dict:
    mat = np.asarray(mat, float)
    if np.isnan(mat).any():
        raise ValueError("NaN in a Page-trend matrix: every fold must have every knob")
    r = page_trend_test(mat)
    return {"p": r["p_one_sided"], "z": r["z"], "n_blocks": r["n_blocks"], "k": r["k_conditions"]}


def per_fold_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation across the knobs, one value per fold. NaN where either row is constant."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(a.shape[0], np.nan)
    for i in range(a.shape[0]):
        if np.ptp(a[i]) == 0 or np.ptp(b[i]) == 0:
            continue
        out[i] = stats.spearmanr(a[i], b[i]).statistic
    return out


def sign_test_positive(values: np.ndarray) -> dict:
    """One-sided sign test that more folds are positive than negative. NaN and exact zeros are dropped."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    v = v[v != 0]
    n = int(len(v))
    n_pos = int((v > 0).sum())
    p = float(stats.binomtest(n_pos, n, 0.5, alternative="greater").pvalue) if n > 0 else 1.0
    return {"n_valid": n, "n_pos": n_pos, "n_neg": n - n_pos, "p": p}


def tracks(f1_mat: np.ndarray, other_mat: np.ndarray, alpha: float = ALPHA) -> dict:
    """Within-fold tracking: per-fold Spearman(F1, other) across the knobs; positive if mean > 0 and the sign
    test over the folds is significant."""
    rho = per_fold_spearman(f1_mat, other_mat)
    st = sign_test_positive(rho)
    mean_rho = float(np.nanmean(rho)) if np.isfinite(rho).any() else float("nan")
    return {"mean_rho": mean_rho, "n_valid": st["n_valid"], "n_pos": st["n_pos"], "n_neg": st["n_neg"],
            "sign_p": st["p"], "tracks": bool(mean_rho > 0 and st["p"] < alpha)}


def tracks_invariance(f1_mat: np.ndarray, subject_probe_mat: np.ndarray, alpha: float = ALPHA) -> dict:
    """Invariance = the NEGATED subject probe (a lower probe means a more invariant embedding)."""
    r = tracks(f1_mat, -np.asarray(subject_probe_mat, float), alpha)
    return {**r, "tracks_invariance": r["tracks"]}


def peak_then_fall(f1_mat: np.ndarray, alpha: float = ALPHA, min_gap_pp: float | None = None) -> dict:
    """F1 peaks and then falls: the argmax of the fold-mean F1 is not the highest knob, and F1 at the highest knob
    is below the peak by a paired Wilcoxon over the folds (two-sided p < alpha, peak higher). min_gap_pp adds the
    plan's magnitude clause where it has one (KC-D6 X1: at least 2 pts below the peak)."""
    f1 = np.asarray(f1_mat, float)
    mean = f1.mean(axis=0)
    peak = int(np.argmax(mean))
    k = f1.shape[1]
    last = f1[:, -1]
    top = f1[:, peak]
    diff = top - last
    gap_pp = float(diff.mean() * 100.0)
    if peak == k - 1 or np.allclose(diff, 0):
        p = 1.0
    else:
        p = float(stats.wilcoxon(top, last, zero_method="wilcox", alternative="two-sided").pvalue)
    falls = bool(peak != k - 1 and gap_pp > 0 and p < alpha and (min_gap_pp is None or gap_pp >= min_gap_pp))
    return {"peak_idx": peak, "peak_not_highest": bool(peak != k - 1), "gap_to_highest_pp": gap_pp,
            "wilcoxon_p": p, "falls": falls, "peak_is_lowest": bool(peak == 0), "min_gap_pp": min_gap_pp}

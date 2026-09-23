#!/usr/bin/env python3
"""
kc23_stats_common.py
======================
Shared statistical primitives for every KC23 stage stats/gate script, so the
paired-test machinery is written once and each stage script only encodes its
own outcome grid. Conventions match the codebase's existing house style
(window_ablation_stats.py: paired Wilcoxon, paired Cohen's dz, BCa bootstrap
10,000 resamples, Holm within a family).

Exit-code convention every gate script in tests_kc23/ follows:
  0  continue (letter is a clean, non-escalating outcome)
  10 report, continue (a claim-level result the queue should not halt on)
  20 ESCALATE (halt dependents; the stage's own docstring says which letters)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
N_BOOT = 10_000


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, float) - np.asarray(b, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def bca_ci(diff: np.ndarray, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = SEED):
    """Bias-corrected and accelerated bootstrap CI for the mean of `diff`."""
    diff = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan")
    theta = diff.mean()
    boot = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(n_boot)])
    prop = np.mean(boot < theta)
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = stats.norm.ppf(prop)
    jack = np.array([np.delete(diff, i).mean() for i in range(n)])
    jbar = jack.mean()
    num = np.sum((jbar - jack) ** 3)
    den = 6.0 * (np.sum((jbar - jack) ** 2) ** 1.5)
    a_hat = num / den if den != 0 else 0.0

    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a_hat * (z0 + z)))

    lo = adj(stats.norm.ppf(alpha / 2))
    hi = adj(stats.norm.ppf(1 - alpha / 2))
    return float(np.quantile(boot, lo)), float(np.quantile(boot, hi))


def paired_test(a: np.ndarray, b: np.ndarray, label: str = "", family: str = "") -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    if np.allclose(diff, 0):
        p = 1.0
        W = 0.0
    else:
        r = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        p, W = float(r.pvalue), float(r.statistic)
    lo, hi = bca_ci(diff)
    n_improved = int((diff > 0).sum())
    return {
        "family": family, "comparison": label,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta_pp": float(diff.mean() * 100), "wilcoxon_W": W, "p_raw": p,
        "cohens_dz": cohens_d_paired(a, b),
        "bca_lo_pp": lo * 100, "bca_hi_pp": hi * 100,
        "n": len(a), "n_improved": n_improved,
    }


def holm(rows: list[dict], p_key: str = "p_raw", out_key: str = "p_holm") -> list[dict]:
    """Holm-Bonferroni within one family. Adds out_key, monotonic, capped at 1."""
    order = sorted(range(len(rows)), key=lambda i: rows[i][p_key])
    m = len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * rows[i][p_key])
        running = max(running, adj)
        rows[i][out_key] = running
    return rows


def tost_equivalence(a: np.ndarray, b: np.ndarray, bound_pp: float = 1.0, alpha: float = 0.05) -> dict:
    """Two one-sided tests for equivalence within +/- bound_pp (percentage
    points) on the paired mean difference. Returns p_tost = max(p_lower,
    p_upper); equivalent=True if p_tost < alpha (both one-sided tests reject
    their respective null of a difference at least as large as the bound)."""
    diff = (np.asarray(a, float) - np.asarray(b, float)) * 100.0
    n = len(diff)
    mean_d = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    if se == 0 or np.isnan(se):
        equivalent = abs(mean_d) < bound_pp
        return {"mean_diff_pp": float(mean_d), "p_tost": 0.0 if equivalent else 1.0,
               "equivalent": bool(equivalent), "bound_pp": bound_pp, "n": n}
    df = n - 1
    t_lower = (mean_d - (-bound_pp)) / se
    t_upper = (mean_d - bound_pp) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    return {"mean_diff_pp": float(mean_d), "p_tost": float(p_tost),
           "equivalent": bool(p_tost < alpha), "bound_pp": bound_pp, "n": n}


def page_trend_test(matrix: np.ndarray) -> dict:
    """Page's L test for a monotonic (increasing) trend across k ordered
    conditions, within n blocks (subjects/realizations). matrix: (n, k),
    columns already in the hypothesized increasing order. Returns the L
    statistic and a normal-approximation p-value (one-sided, valid for the
    typical k in this programme, k in 4..7; exact tables are not bundled with
    scipy).
    """
    matrix = np.asarray(matrix, float)
    n, k = matrix.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    weights = np.arange(1, k + 1)
    L = float(np.sum(weights * rank_sums))
    mean_L = n * k * (k + 1) ** 2 / 4.0
    var_L = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144.0
    z = (L - mean_L) / np.sqrt(var_L) if var_L > 0 else 0.0
    p = float(1 - stats.norm.cdf(z))  # one-sided: increasing trend
    return {"L": L, "z": float(z), "p_one_sided": p, "n_blocks": n, "k_conditions": k}


def read_subjectwise(path_or_dir, model_token: str = "", subject_col_candidates=("subject", "heldout_subject"),
                     f1_col_candidates=("f1_macro",)) -> pd.DataFrame:
    """Load a per-subject CSV, tolerant of the column-name variation already
    present across this codebase's drivers (subject vs heldout_subject)."""
    p = Path(path_or_dir)
    candidates = [p] if p.is_file() else sorted(p.glob("*subjectwise*.csv")) + sorted(p.glob("checkpoints/*.csv"))
    if model_token:
        toked = [c for c in candidates if model_token.upper() in c.name.upper()]
        if toked:
            candidates = toked
    last_err = None
    for c in candidates:
        try:
            df = pd.read_csv(c)
        except Exception as e:
            last_err = e
            continue
        cols = {col.lower(): col for col in df.columns}
        subj = next((cols[s] for s in subject_col_candidates if s in cols), None)
        f1 = next((cols[s] for s in f1_col_candidates if s in cols), None)
        if subj and f1:
            return df.rename(columns={subj: "subject", f1: "f1_macro"})
    raise FileNotFoundError(f"no subject/f1_macro subjectwise csv found under {path_or_dir} ({last_err})")


def require_complete(df: pd.DataFrame, n_expected: int, label: str = "") -> np.ndarray:
    """Return f1_macro sorted by subject, after checking for duplicates and
    completeness -- the resume-hygiene check the standing rules require."""
    dupes = df["subject"].duplicated().sum()
    if dupes:
        raise ValueError(f"{label}: {dupes} duplicated subject rows -- deduplicate before trusting this.")
    if len(df) != n_expected:
        raise ValueError(f"{label}: expected {n_expected} rows, found {len(df)}. Incomplete.")
    return df.sort_values("subject")["f1_macro"].to_numpy(dtype=float)


def print_gate_header(stage: str, letter: str, reading: str):
    print(f"\n================ {stage} OUTCOME: {letter} ================")
    print(reading)


def exit_for_letters(fired: list[str], escalate: set, report_only: set) -> int:
    """Standard exit-code dispatch: 20 if any fired letter is in `escalate`,
    else 10 if any is in `report_only`, else 0."""
    if any(l in escalate for l in fired):
        return 20
    if any(l in report_only for l in fired):
        return 10
    return 0

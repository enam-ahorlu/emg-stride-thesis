#!/usr/bin/env python3
"""
optimization_statistical_tests.py
===================================
Wilcoxon signed-rank tests (two-sided, scipy.stats.wilcoxon)
for every meaningful optimization comparison in the thesis pipeline.

Tests cover:
  1. Normalization contribution (global -> per_subject)
  2. Feature selection independent contribution (on global norm)
  3. Feature selection marginal gain (on per_subject norm)
  4. CNN augmentation independent contribution (on global norm)
  5. CNN augmentation marginal gain (on per_subject norm)
  6. End-to-end improvement

Outputs:
  report_figs/optimization_wilcoxon_table.csv   -- full results
  report_figs/optimization_effect_sizes.csv     -- significant results with interpretation
"""
from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon as scipy_wilcoxon

ROOT    = Path(__file__).parent
RPT_DIR = ROOT / "report_figs"
RPT_DIR.mkdir(exist_ok=True)

# ============================================================
# Wilcoxon signed-rank using scipy
# ============================================================

def wilcoxon_signed_rank(x, y):
    """Two-sided Wilcoxon signed-rank test via scipy.stats.wilcoxon. Returns dict."""
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    diff = a - b
    n_eff = int(np.sum(diff != 0))
    if n_eff == 0:
        return dict(W=0.0, z=float("nan"), p=1.0, n_eff=0)
    stat, p = scipy_wilcoxon(diff, alternative="two-sided")
    return dict(W=float(stat), z=float("nan"), p=round(float(p), 6), n_eff=n_eff)


def cohen_d_paired(x, y):
    """Paired Cohen's d = mean(x-y) / SD(x-y)."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(d.mean() / sd)


def effect_size_label(d_val):
    d_abs = abs(d_val)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


# ============================================================
# Data loaders
# ============================================================

def _get_subj_col(df):
    return "heldout_subject" if "heldout_subject" in df.columns else "subject"


def load_classical(results_dir, model):
    """Load per-subject F1 dict {subject_id: f1} from classical LOSO results."""
    rdir = ROOT / results_dir

    # Try subjectwise CSV first
    files = sorted(rdir.glob(f"*{model}*subjectwise*.csv"))
    if not files:
        # Try checkpoint CSV
        ckpt_dir = rdir / "checkpoints"
        if ckpt_dir.exists():
            files = sorted(ckpt_dir.glob(f"*{model}*ckpt.csv"))
    if not files:
        raise FileNotFoundError(f"No {model} subjectwise/ckpt CSV in {rdir}")

    df = pd.read_csv(files[0])
    sc = _get_subj_col(df)
    df = df.drop_duplicates(subset=[sc]).sort_values(sc)
    return {int(r[sc]): float(r["f1_macro"]) for _, r in df.iterrows()}


def load_cnn(results_dir):
    """Load per-subject F1 dict {subject_id: f1} from CNN LOSO results."""
    path = ROOT / results_dir / "per_subject_metrics_cnn_loso.csv"
    df = pd.read_csv(path)
    sc = _get_subj_col(df)
    df = df.drop_duplicates(subset=[sc]).sort_values(sc)
    return {int(r[sc]): float(r["f1_macro"]) for _, r in df.iterrows()}


def align_paired(dict_a, dict_b):
    """Align two {subject: f1} dicts, return paired arrays."""
    shared = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if len(shared) < 30:
        print(f"    WARNING: only {len(shared)} shared subjects!")
    return (np.array([dict_a[s] for s in shared]),
            np.array([dict_b[s] for s in shared]))


# ============================================================
# Load all data
# ============================================================
print("=" * 70)
print("LOADING DATA SOURCES")
print("=" * 70)

d = {}

# LOSO baselines (global norm, full-72)
d["SVM_full72_global"]  = load_classical("results_loso_freq", "SVM")
d["RF_full72_global"]   = load_classical("results_loso_freq", "RF")
d["CNN_full72_global"]  = load_cnn("results_cnn_loso")

# LOSO per_subject norm, full-72
d["SVM_full72_persubj"] = load_classical("results_loso_freq_persubj", "SVM")
d["RF_full72_persubj"]  = load_classical("results_loso_freq_persubj", "RF")
d["CNN_full72_persubj"] = load_cnn("results_cnn_loso_norm_persubj")

# Feature selection (per_subject norm)
d["SVM_rfe36_persubj"]  = load_classical("results_loso_freq_rfe36", "SVM")
d["RF_rfe36_persubj"]   = load_classical("results_loso_freq_rfe36", "RF")
d["SVM_mi36_persubj"]   = load_classical("results_loso_freq_mi36", "SVM")
d["RF_mi36_persubj"]    = load_classical("results_loso_freq_mi36", "RF")

# Feature selection (global norm)
d["SVM_rfe36_global"]   = load_classical("results_loso_freq_rfe36_globnorm", "SVM")
d["RF_rfe36_global"]    = load_classical("results_loso_freq_rfe36_globnorm", "RF")
d["SVM_mi36_global"]    = load_classical("results_loso_freq_mi36_globnorm", "SVM")
d["RF_mi36_global"]     = load_classical("results_loso_freq_mi36_globnorm", "RF")

# CNN augmentation (per_subject norm)
d["CNN_gaussian_persubj"]  = load_cnn("results_cnn_loso_aug_gaussian")
d["CNN_chandrop_persubj"]  = load_cnn("results_cnn_loso_aug_chandrop")
d["CNN_timemask_persubj"]  = load_cnn("results_cnn_loso_aug_timemask")

# CNN augmentation (global norm)
d["CNN_gaussian_global"]   = load_cnn("results_cnn_loso_aug_gaussian_globnorm")
d["CNN_chandrop_global"]   = load_cnn("results_cnn_loso_aug_chandrop_globnorm")
d["CNN_timemask_global"]   = load_cnn("results_cnn_loso_aug_timemask_globnorm")

print(f"  Loaded {len(d)} conditions.\n")

# ============================================================
# Define comparisons: (comparison_name, model, cond_a, cond_b, key_a, key_b)
# ============================================================

tests = [
    # 1. Normalization contribution
    ("1_norm_contribution", "SVM", "full72_global", "full72_persubj"),
    ("1_norm_contribution", "RF",  "full72_global", "full72_persubj"),
    ("1_norm_contribution", "CNN", "full72_global", "full72_persubj"),

    # 2. Feature selection independent (global norm)
    ("2_featsel_indep_global", "SVM", "full72_global", "rfe36_global"),
    ("2_featsel_indep_global", "SVM", "full72_global", "mi36_global"),
    ("2_featsel_indep_global", "RF",  "full72_global", "rfe36_global"),
    ("2_featsel_indep_global", "RF",  "full72_global", "mi36_global"),

    # 3. Feature selection marginal (per_subject norm)
    ("3_featsel_marginal_persubj", "SVM", "full72_persubj", "rfe36_persubj"),
    ("3_featsel_marginal_persubj", "RF",  "full72_persubj", "rfe36_persubj"),

    # 4. CNN augmentation independent (global norm)
    ("4_cnn_aug_indep_global", "CNN", "full72_global", "gaussian_global"),
    ("4_cnn_aug_indep_global", "CNN", "full72_global", "chandrop_global"),
    ("4_cnn_aug_indep_global", "CNN", "full72_global", "timemask_global"),

    # 5. CNN augmentation marginal (per_subject norm)
    ("5_cnn_aug_marginal_persubj", "CNN", "full72_persubj", "gaussian_persubj"),
    ("5_cnn_aug_marginal_persubj", "CNN", "full72_persubj", "chandrop_persubj"),
    ("5_cnn_aug_marginal_persubj", "CNN", "full72_persubj", "timemask_persubj"),

    # 6. End-to-end
    ("6_end_to_end", "SVM", "full72_global", "rfe36_persubj"),
    ("6_end_to_end", "RF",  "full72_global", "rfe36_persubj"),
    ("6_end_to_end", "CNN", "full72_global", "gaussian_persubj"),
]

# ============================================================
# Run all tests
# ============================================================
print("=" * 70)
print("WILCOXON SIGNED-RANK TESTS  (two-sided, scipy.stats.wilcoxon)")
print("=" * 70)

rows = []
prev_cat = ""
for comp_name, model, cond_a, cond_b, in tests:
    if comp_name != prev_cat:
        print(f"\n  --- {comp_name} ---")
        prev_cat = comp_name

    key_a = f"{model}_{cond_a}"
    key_b = f"{model}_{cond_b}"
    arr_a, arr_b = align_paired(d[key_a], d[key_b])

    res = wilcoxon_signed_rank(arr_a, arr_b)
    cd = cohen_d_paired(arr_b, arr_a)   # positive = B better than A
    mean_a = float(arr_a.mean())
    mean_b = float(arr_b.mean())
    delta  = mean_b - mean_a
    sig    = res["p"] < 0.05

    row = {
        "comparison_name": comp_name,
        "model": model,
        "condition_a": cond_a,
        "condition_b": cond_b,
        "mean_a": round(mean_a, 4),
        "mean_b": round(mean_b, 4),
        "delta": round(delta, 4),
        "W_statistic": round(res["W"], 1),
        "z_score": res["z"],
        "p_value": res["p"],
        "significant": sig,
        "cohen_d": round(cd, 4),
    }
    rows.append(row)

    sig_str = "***" if res["p"] < 0.001 else ("**" if res["p"] < 0.01 else ("*" if sig else "ns"))
    print(f"  {model:4s} | {cond_a:20s} -> {cond_b:20s} | "
          f"delta={delta:+.4f} | W={res['W']:.1f} | p={res['p']:.6f} {sig_str} | d={cd:+.3f}")

# ============================================================
# Save full table
# ============================================================
full_df = pd.DataFrame(rows)
out1 = RPT_DIR / "optimization_wilcoxon_table.csv"
full_df.to_csv(out1, index=False)
print(f"\n  [saved] {out1.name}")

# ============================================================
# Save effect sizes (significant only, with interpretation)
# ============================================================
sig_df = full_df[full_df["significant"]].copy()
sig_df["effect_size_interp"] = sig_df["cohen_d"].apply(effect_size_label)
out2 = RPT_DIR / "optimization_effect_sizes.csv"
sig_df.to_csv(out2, index=False)
print(f"  [saved] {out2.name}")

# ============================================================
# Print summary table
# ============================================================
print("\n" + "=" * 120)
print("SUMMARY TABLE")
print("=" * 120)

hdr = (f"{'Comparison':<30s} {'Model':>5s} {'Condition A':>20s} {'Condition B':>20s} "
       f"{'MeanA':>7s} {'MeanB':>7s} {'Delta':>8s} {'p':>10s} {'Sig':>4s} {'d':>7s}")
print(hdr)
print("-" * 120)
for r in rows:
    sig_str = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["significant"] else "ns"))
    print(f"{r['comparison_name']:<30s} {r['model']:>5s} {r['condition_a']:>20s} {r['condition_b']:>20s} "
          f"{r['mean_a']:>7.4f} {r['mean_b']:>7.4f} {r['delta']:>+8.4f} {r['p_value']:>10.6f} {sig_str:>4s} {r['cohen_d']:>+7.3f}")

n_sig = sum(1 for r in rows if r["significant"])
print(f"\n  Total tests: {len(rows)}")
print(f"  Significant (p<0.05): {n_sig}")
print(f"  Non-significant: {len(rows) - n_sig}")

print("\n=== optimization_statistical_tests.py COMPLETE ===")

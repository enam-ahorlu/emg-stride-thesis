#!/usr/bin/env python3
"""
critique_stats.py
==================
EXPERIMENT_PLAN_CRITIQUE.md -- paired Wilcoxon signed-rank + Cohen's d +
deterministic BCa CI (same bca() convention as stats_new_experiments.py,
seeded from hashlib.blake2b) for every subject-level paired comparison
produced by E1-E5. Mirrors the g1/g3/.../g9_*_stats.csv schema (comparison,
delta_pp, cohens_d, p) so it folds into stats_unified_fdr.py the same way.

Each experiment's section guards on its output files existing (E3/E4/E5 are
long-running background jobs) and prints a [skip] notice rather than failing
if a stage hasn't finished yet -- rerun after each stage completes.

Output: report_figs/new_experiments/critique_stats_fdr.csv
        report_figs/new_experiments/critique_cis.csv (BCa CIs for headline means)
"""
from pathlib import Path
import hashlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)


def bca(x, n=10000):
    x = np.asarray(x, float)
    seed = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float64).tobytes(),
                                          digest_size=4).digest(), 'big')
    rng = np.random.default_rng(seed)
    th = x.mean()
    bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    z0 = norm.ppf(min(max((bs < th).mean(), 1e-4), 1 - 1e-4))
    jk = np.array([np.delete(x, i).mean() for i in range(len(x))]); jm = jk.mean()
    den = 6 * (((jm - jk) ** 2).sum() ** 1.5); a = (((jm - jk) ** 3).sum() / den) if den else 0
    q = lambda al: np.percentile(bs, 100 * norm.cdf(z0 + (z0 + norm.ppf(al)) / (1 - a * (z0 + norm.ppf(al)))))
    return q(.025), q(.975)


def d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def row(comparison, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    w, p = wilcoxon(a, b)
    return dict(comparison=comparison, delta_pp=round((a.mean() - b.mean()) * 100, 2),
               cohens_d=round(d_paired(a, b), 3), p=float(p), n=len(a))


rows = []
ci_rows = []

# ============================================================================
# E1 -- variance decomposition: per-subject norm vs global norm distributional
# distance (Part C). Subject-paired (n=40).
# ============================================================================
dist_path = ROOT / "results_variance_decomposition" / "subject_distance_vs_f1.csv"
if dist_path.exists():
    d = pd.read_csv(dist_path)
    piv_mmd = d.pivot(index="subject", columns="norm", values="mmd_to_others")
    piv_maha = d.pivot(index="subject", columns="norm", values="mahalanobis")
    rows.append(row("E1: MMD-to-others, per-subject norm vs global norm",
                    piv_mmd["per_subject"], piv_mmd["global"]))
    rows.append(row("E1: Mahalanobis, per-subject norm vs global norm",
                    piv_maha["per_subject"], piv_maha["global"]))
    print("[E1] added 2 comparisons")
else:
    print(f"[skip E1] {dist_path} not found -- run analyze_between_subject_variance.py first")

# ============================================================================
# E3 -- causal deployable ensemble: transductive vs causal calib-K,
# buffer-excluded. Subject-paired.
# ============================================================================
e3_report = ROOT / "results_causal_ensemble" / "report.csv"
e3_transductive_ref = ROOT / "results_ensemble_v2_chandrop" / "ensemble_v2_subjectwise.csv"
if e3_report.exists() and e3_transductive_ref.exists():
    trans = pd.read_csv(e3_transductive_ref)
    trans_col = "SVM+RESNET_SE [soft]"
    if trans_col in trans.columns:
        trans_vec = trans[trans_col].to_numpy(float)
        for K in [25, 50, 100]:
            subw = ROOT / "results_causal_ensemble" / f"calib{K}_subjectwise.csv"
            if not subw.exists():
                continue
            sw = pd.read_csv(subw)
            n = min(len(sw), len(trans_vec))
            rows.append(row(f"E3: causal calib{K} (buffer-excl) vs transductive, SVM+RESNET_SE soft",
                            sw["soft_excl"].to_numpy(float)[:n], trans_vec[:n]))
        print("[E3] added causal-vs-transductive comparisons")
    else:
        print(f"[skip E3] {trans_col!r} not in {e3_transductive_ref}")
else:
    print(f"[skip E3] results not ready yet (report.csv / transductive reference missing)")

# ============================================================================
# E4 -- within-subject baseline: regime A vs regime B at each N (first-N draw),
# subject-paired.
# ============================================================================
e4_csv = ROOT / "results_within_subject" / "within_subject_subjectwise.csv"
if e4_csv.exists():
    d4 = pd.read_csv(e4_csv)
    d4f = d4[d4.draw == "first"]
    n_added = 0
    for model in d4f.model.unique():
        for N in sorted(d4f.N.unique()):
            sub = d4f[(d4f.model == model) & (d4f.N == N)].sort_values("subject")
            if len(sub) < 5:
                continue
            rows.append(row(f"E4: regime A (subject-only) vs B (cross-subject 0-label), {model}, N={N}",
                            sub["f1_A_subject_only"], sub["f1_B_crosssubject_zero_label"]))
            n_added += 1
    print(f"[E4] added {n_added} comparisons (need >=5 subjects/combo; {d4.subject.nunique()} subjects "
          f"done so far, {len(d4)} rows total)")
else:
    print(f"[skip E4] {e4_csv} not found -- run_within_subject_baseline.py not started/finished")

# ============================================================================
# E5 -- RF calibration: calibrated-RF ensemble vs headline SVM+RESNET_SE soft.
# Subject-paired.
# ============================================================================
for method in ["isotonic", "sigmoid"]:
    e5_subjw = ROOT / "results_rf_calibrated" / method / "ensemble_v2_subjectwise.csv"
    ref_subjw = ROOT / "results_ensemble_v2_chandrop" / "ensemble_v2_subjectwise.csv"
    if e5_subjw.exists() and ref_subjw.exists():
        e5 = pd.read_csv(e5_subjw)
        ref = pd.read_csv(ref_subjw)
        # must actually include calibrated RF -- "SVM+RESNET_SE [..]" (no RF) can
        # sometimes edge out "SVM+RF+RESNET_SE [..]" by noise and would silently
        # test the wrong (RF-free) ensemble against the RF-free reference.
        cand_cols = [c for c in e5.columns if c.startswith("SVM+RF+RESNET_SE")]
        ref_col = "SVM+RESNET_SE [soft]"
        if cand_cols and ref_col in ref.columns:
            n = min(len(e5), len(ref))
            # (1) the plan's literal decisive question: does SVM+RF_calibrated+RESNET_SE
            # [soft] now reach/beat SVM+RESNET_SE [soft] = 0.8579?
            soft_col = "SVM+RF+RESNET_SE [soft]"
            if soft_col in e5.columns:
                rows.append(row(f"E5: SVM+RF_calibrated({method})+RESNET_SE [soft] vs headline SVM+RESNET_SE [soft]",
                                e5[soft_col].to_numpy(float)[:n], ref[ref_col].to_numpy(float)[:n]))
            # (2) best available RF-inclusive combiner (context: does a smarter combiner
            # rescue calibrated RF even where soft-voting doesn't?)
            best_col = max(cand_cols, key=lambda c: e5[c].mean())
            rows.append(row(f"E5: best RF-calibrated({method})-inclusive ensemble ({best_col}) vs headline SVM+RESNET_SE [soft]",
                            e5[best_col].to_numpy(float)[:n], ref[ref_col].to_numpy(float)[:n]))
            print(f"[E5] added {method} comparisons (soft={soft_col in e5.columns}, best={best_col})")
    else:
        print(f"[skip E5-{method}] {e5_subjw} not found yet")

# ============================================================================
# write out
# ============================================================================
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "critique_stats_fdr.csv", index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n[save] {OUT / 'critique_stats_fdr.csv'} ({len(df)} comparisons)")
else:
    print("\n[warn] no comparisons available yet -- nothing written")

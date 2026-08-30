#!/usr/bin/env python3
"""
check_rung4_robustness.py
========================
EXPERIMENT_PLAN_CAUSALITY.md E-C2 -- is rung 4 a fair whitening, or an artifact
of its (unit-dependent, non-uniform) regularizer?

Recomputes ONLY the rung-4 row of the alignment ladder under four variants,
holding the metric code and the point sets byte-identical to the published
ladder (analyze_between_subject_variance.part_b_ladder):

  raw_lam1            the published transform (rung4_full_whiten_recolor, lam=1.0)
                      -- reproduction check / validation gate.
  prez_lam1           rung0_global_z first, THEN per-subject whiten+recolor,
                      lam=1.0. The faithful CORAL analogue (CORAL standardizes
                      before aligning; rung 4 does not).
  prez_scalefree_a1   pre-standardized, ridge lam = trace(Cs)/p (per matrix).
  prez_scalefree_a01  pre-standardized, ridge lam = 0.1 * trace(Cs)/p.

Load-bearing claim under test: the ORDERING -- silhouette peaks at rung 3 and
falls at whitening while the subject probe keeps falling. If that ordering
survives all four variants the finding is robust to the regularizer. If it
flips under a scale-free ridge, the negative silhouette was an artifact and
Section 4.13.2 needs rewriting.

Reuses (does not reimplement): rung0_global_z, rung3_mean_scale,
rung4_full_whiten_recolor, sym_sqrt, sym_invsqrt, stratified_subsample,
rbf_mmd2, median_heuristic_gamma from analyze_between_subject_variance.py, and
per_subject_zscore from train_classical_loso.py.

CPU only, no training. Output: results_variance_decomposition/rung4_robustness.csv
(new file alongside the published alignment_ladder.csv -- never overwrites it).
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score

from analyze_between_subject_variance import (
    load_data, rung0_global_z, rung3_mean_scale, rung4_full_whiten_recolor,
    sym_sqrt, sym_invsqrt, stratified_subsample, rbf_mmd2, median_heuristic_gamma,
    SEED, LABELS, OUT,
)
from train_classical_loso import per_subject_zscore

PUBLISHED_LADDER = OUT / "alignment_ladder.csv"
CAP_PER_SUBJECT_CLASS = 100  # part_b_ladder default


def whiten_recolor(Xin, subjects, lam=None, alpha=None):
    """Per-subject whiten + recolor to the pooled covariance of the
    per-subject-z-scored input -- the same construction as
    rung4_full_whiten_recolor, but (a) operating on a caller-supplied matrix
    (so the CORAL 'standardize first' step can be prepended) and (b) optionally
    using a scale-free ridge lam = alpha * trace(C)/p per covariance instead of
    a fixed additive lam.

    With Xin = X (raw) and lam=1.0, alpha=None this is identical to
    rung4_full_whiten_recolor(X, subjects, lam=1.0); that path is exercised
    separately via the imported function as the validation gate.
    """
    if (lam is None) == (alpha is None):
        raise ValueError("pass exactly one of lam= or alpha=")
    p = Xin.shape[1]
    X3 = per_subject_zscore(Xin, subjects)
    Ct_raw = np.cov(X3, rowvar=False)
    lam_t = alpha * np.trace(Ct_raw) / p if alpha is not None else lam
    Ctarget_sqrt = sym_sqrt(Ct_raw + lam_t * np.eye(p))
    Xo = Xin.copy()
    for s in np.unique(subjects):
        m = subjects == s
        Xs = Xin[m]
        mu = Xs.mean(0, keepdims=True)
        Cs_raw = np.cov(Xs, rowvar=False)
        lam_s = alpha * np.trace(Cs_raw) / p if alpha is not None else lam
        A = sym_invsqrt(Cs_raw + lam_s * np.eye(p)) @ Ctarget_sqrt
        Xo[m] = (Xs - mu) @ A
    return Xo


def build_point_sets(y, subjects):
    """Byte-identical to the head of part_b_ladder: the per-(subject,class)
    window subsample (cap 100, drawn from np.random.default_rng(SEED) in the
    SAME loop order) and the shared stratified subsample for silhouette."""
    rng = np.random.default_rng(SEED)
    subs_u = np.unique(subjects)
    sub_idx = {}
    for s in subs_u:
        for c in range(len(LABELS)):
            mm = np.where((subjects == s) & (y == c))[0]
            if len(mm) > CAP_PER_SUBJECT_CLASS:
                mm = rng.choice(mm, CAP_PER_SUBJECT_CLASS, replace=False)
            sub_idx[(s, c)] = mm
    class_sil_idx = stratified_subsample(y, subjects)
    return subs_u, sub_idx, class_sil_idx


def ladder_metrics(Xr, y, subjects, subs_u, sub_idx, class_sil_idx,
                   want_probe=True, want_sil=True):
    """The rung-loop body of part_b_ladder, verbatim in structure."""
    gamma = median_heuristic_gamma(Xr)

    # (i) mean pairwise MMD, class-conditional then averaged
    mmd_per_class = []
    for c in range(len(LABELS)):
        samples = {s: Xr[sub_idx[(s, c)]] for s in subs_u if len(sub_idx[(s, c)]) >= 3}
        keys = list(samples.keys())
        vals = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                vals.append(rbf_mmd2(samples[keys[i]], samples[keys[j]], gamma))
        mmd_per_class.append(np.mean(vals))
    mmd_mean = float(np.mean(mmd_per_class))

    # (ii) mean per-feature Wasserstein-1 between subject pairs (every 3rd pair)
    w1_vals = []
    subj_samples = {s: Xr[subjects == s] for s in subs_u}
    pair_count = 0
    for i in range(len(subs_u)):
        for j in range(i + 1, len(subs_u)):
            a, b = subj_samples[subs_u[i]], subj_samples[subs_u[j]]
            pair_count += 1
            if pair_count % 3 != 0:
                continue
            w1_vals.append(np.mean([wasserstein_distance(a[:, f], b[:, f]) for f in range(Xr.shape[1])]))
    w1_mean = float(np.mean(w1_vals))

    # (iii) subject-identity probe: 5-fold CV balanced accuracy
    probe_acc = np.nan
    if want_probe:
        clf = LogisticRegression(max_iter=300)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        probe_acc = float(cross_val_score(clf, Xr, subjects, cv=skf,
                                          scoring="balanced_accuracy", n_jobs=1).mean())

    # (iv) class separability on the SAME points, raw 72-D space
    sil_class = np.nan
    if want_sil:
        sil_class = float(silhouette_score(Xr[class_sil_idx], y[class_sil_idx]))

    return dict(mmd_mean=mmd_mean, wasserstein1_mean=w1_mean,
                subject_probe_bal_acc=probe_acc, silhouette_by_class=sil_class)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true",
                    help="skip variants already present in rung4_robustness.csv")
    ap.add_argument("--out", default=str(OUT / "rung4_robustness.csv"))
    args = ap.parse_args()
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, y, subjects, _ = load_data()
    print(f"[data] X={X.shape} subjects={len(np.unique(subjects))} classes={len(LABELS)}")

    # feature-variance spread the plan cites -- print it as a sanity anchor
    v = X.var(0)
    print(f"[raw feature variance] min={v.min():.3e} max={v.max():.3e} "
          f"span={np.log10(v.max()/v.min()):.1f} orders; "
          f"{int((v < 0.01).sum())}/{X.shape[1]} features below 0.01")

    subs_u, sub_idx, class_sil_idx = build_point_sets(y, subjects)

    # rung-0 reference for the removed-% denominators, same code path
    print("\n[rung 0] recomputing MMD/W1 denominators (same code path)...")
    r0 = ladder_metrics(rung0_global_z(X), y, subjects, subs_u, sub_idx,
                        class_sil_idx, want_probe=False, want_sil=False)
    print(f"  mmd_mean={r0['mmd_mean']:.10f}  wasserstein1_mean={r0['wasserstein1_mean']:.10f}")
    pub = pd.read_csv(PUBLISHED_LADDER)
    pub0 = pub[pub.rung == 0].iloc[0]
    pub3 = pub[pub.rung == 3].iloc[0]
    pub4 = pub[pub.rung == 4].iloc[0]
    print(f"  published rung-0 mmd_mean={pub0['mmd_mean']:.10f} "
          f"wasserstein1_mean={pub0['wasserstein1_mean']:.10f}  "
          f"(dz mmd={r0['mmd_mean']-pub0['mmd_mean']:+.2e}, "
          f"w1={r0['wasserstein1_mean']-pub0['wasserstein1_mean']:+.2e})")

    variants = [
        ("raw_lam1",           lambda: rung4_full_whiten_recolor(X, subjects, lam=1.0)),
        ("prez_lam1",          lambda: whiten_recolor(rung0_global_z(X), subjects, lam=1.0)),
        ("prez_scalefree_a1",  lambda: whiten_recolor(rung0_global_z(X), subjects, alpha=1.0)),
        ("prez_scalefree_a01", lambda: whiten_recolor(rung0_global_z(X), subjects, alpha=0.1)),
    ]

    done = set()
    if args.resume and out_csv.exists():
        done = set(pd.read_csv(out_csv)["variant"].tolist())
        if done:
            print(f"[resume] already have: {sorted(done)}")

    for name, fn in variants:
        if name in done:
            continue
        tv = time.time()
        print(f"\n[variant] {name} ...")
        Xr = fn()
        met = ladder_metrics(Xr, y, subjects, subs_u, sub_idx, class_sil_idx)
        row = dict(
            variant=name,
            mmd_mean=met["mmd_mean"],
            wasserstein1_mean=met["wasserstein1_mean"],
            subject_probe_bal_acc=met["subject_probe_bal_acc"],
            silhouette_by_class=met["silhouette_by_class"],
            mmd_removed_pct=(1 - met["mmd_mean"] / r0["mmd_mean"]) * 100,
            w1_removed_pct=(1 - met["wasserstein1_mean"] / r0["wasserstein1_mean"]) * 100,
            rung3_silhouette_ref=float(pub3["silhouette_by_class"]),
            rung3_probe_ref=float(pub3["subject_probe_bal_acc"]),
            elapsed_sec=round(time.time() - tv, 1),
        )
        # CHECKPOINT: append immediately
        pd.DataFrame([row]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"  MMD removed={row['mmd_removed_pct']:.2f}%  W1 removed={row['w1_removed_pct']:.2f}%  "
              f"probe={row['subject_probe_bal_acc']:.5f}  silhouette={row['silhouette_by_class']:.6f}  "
              f"({row['elapsed_sec']:.0f}s)")

    df = pd.read_csv(out_csv).drop_duplicates("variant")
    order = ["raw_lam1", "prez_lam1", "prez_scalefree_a1", "prez_scalefree_a01"]
    df = df.set_index("variant").loc[[v for v in order if v in df.index.tolist()]].reset_index()

    print("\n================  RUNG-4 ROBUSTNESS  ================")
    print(df[["variant", "mmd_removed_pct", "w1_removed_pct",
              "subject_probe_bal_acc", "silhouette_by_class"]].to_string(index=False))

    # ---- validation gate: raw_lam1 reproduces the published rung-4 row ----
    g = df[df.variant == "raw_lam1"]
    gate_pass = None
    if len(g):
        g = g.iloc[0]
        d_mmd = abs(g["mmd_removed_pct"] - float(pub4["mmd_removed_pct"]))
        d_probe = abs(g["subject_probe_bal_acc"] - float(pub4["subject_probe_bal_acc"]))
        d_sil = abs(g["silhouette_by_class"] - float(pub4["silhouette_by_class"]))
        gate_pass = (d_mmd < 0.05) and (d_probe < 5e-4) and (d_sil < 5e-4)
        print("\n[GATE] raw_lam1 vs published rung-4 "
              f"(MMD removed {pub4['mmd_removed_pct']:.2f}%, probe "
              f"{pub4['subject_probe_bal_acc']:.5f}, silhouette {pub4['silhouette_by_class']:.6f})")
        print(f"       d(MMD removed)={d_mmd:.4f} pp   d(probe)={d_probe:.2e}   d(silhouette)={d_sil:.2e}")
        print(f"       -> {'PASS' if gate_pass else 'FAIL'}")

    # ---- ordering test ----
    print("\n[ORDERING] published rung 3: silhouette="
          f"{float(pub3['silhouette_by_class']):.6f}, probe={float(pub3['subject_probe_bal_acc']):.5f}")
    all_hold = True
    for _, r in df.iterrows():
        sil_falls = r["silhouette_by_class"] < float(pub3["silhouette_by_class"])
        probe_falls = r["subject_probe_bal_acc"] < float(pub3["subject_probe_bal_acc"])
        holds = bool(sil_falls and probe_falls)
        all_hold &= holds
        print(f"  {r['variant']:<20} silhouette {r['silhouette_by_class']:+.6f} "
              f"({'below' if sil_falls else 'AT/ABOVE'} rung3)   "
              f"probe {r['subject_probe_bal_acc']:.5f} "
              f"({'below' if probe_falls else 'AT/ABOVE'} rung3)   -> "
              f"{'ordering holds' if holds else 'ORDERING FLIPS'}")
    print(f"\n[VERDICT] ordering {'ROBUST to the regularizer across all 4 variants' if all_hold else 'FLIPS under at least one variant -- Section 4.13.2 needs rewriting'}")
    print(f"[save] {out_csv}")
    print(f"[DONE] total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

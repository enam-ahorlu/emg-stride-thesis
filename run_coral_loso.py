# run_coral_loso.py
# ---------------------------------------------------------------------------
# Action 2.2 — Unsupervised domain-adaptation (UDA) baseline: CORAL.
#
# Benchmarks the thesis's simple per-subject z-score against a real, published
# UDA method (CORrelation ALignment, Sun et al., 2016). CORAL aligns the
# second-order statistics (covariance) of the pooled training subjects (source)
# to the held-out subject (target). It is unsupervised — it uses only the test
# subject's *features*, never its labels — so it is directly comparable to
# per-subject z-score as a label-free adaptation strategy.
#
# Per LOSO fold:
#   1. Standardise features (StandardScaler fit on training subjects only).
#   2. Cs = cov(source) + lambda*I ; Ct = cov(target) + lambda*I.
#   3. Recolour source: Xs' = (Xs - mean_s) @ Cs^{-1/2} @ Ct^{1/2}.
#   4. Train SVM/RF on Xs'; test on (Xt - mean_t).
#
# The headline question: does the simple per-subject z-score (0.777) match or
# beat CORAL at a fraction of the complexity?
#
# Example:
#   python run_coral_loso.py \
#       --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
#       --meta     features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --models SVM,RF
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import eigh

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

from train_classical_loso import load_features_npz, encode_labels


def sym_sqrt(M, eps=1e-8):
    w, V = eigh((M + M.T) / 2.0)
    w = np.clip(w, eps, None)
    return (V * np.sqrt(w)) @ V.T


def sym_invsqrt(M, eps=1e-8):
    w, V = eigh((M + M.T) / 2.0)
    w = np.clip(w, eps, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def coral_align(Xs, Xt, lam=1.0):
    """Recolour source Xs to match target Xt covariance (unsupervised)."""
    ms = Xs.mean(0, keepdims=True)
    mt = Xt.mean(0, keepdims=True)
    d = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + lam * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + lam * np.eye(d)
    A = sym_invsqrt(Cs) @ sym_sqrt(Ct)
    Xs_aligned = (Xs - ms) @ A
    Xt_centred = Xt - mt
    return Xs_aligned, Xt_centred


def build_search(model_name, inner_cv, Xtr, ytr, gtr, seed, n_jobs=1, rf_n_jobs=1):
    # GridSearchCV's own n_jobs deadlocks under repeated calls in a detached process
    # on Windows -- keep at 1. RandomForestClassifier's own n_jobs (tree-building
    # within one fit) is a different joblib path and is safe to raise; use rf_n_jobs.
    if model_name == "SVM":
        pipe = Pipeline([("clf", SVC(kernel="rbf", class_weight="balanced", cache_size=500))])
        grid = {"clf__C": [1, 5, 10], "clf__gamma": ["scale"]}
    elif model_name == "RF":
        pipe = Pipeline([("clf", RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=rf_n_jobs))])
        grid = {"clf__n_estimators": [200, 400, 500], "clf__max_depth": [None, 10]}
    else:
        raise ValueError(model_name)
    return GridSearchCV(pipe, grid, scoring="f1_macro",
                        cv=list(inner_cv.split(Xtr, ytr, groups=gtr)),
                        n_jobs=n_jobs, refit=True)


def main():
    ap = argparse.ArgumentParser("CORAL UDA baseline under LOSO (classical).")
    ap.add_argument("--features", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--models", default="SVM,RF")
    ap.add_argument("--lam", type=float, default=1.0, help="CORAL covariance regulariser (default 1.0)")
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers for GridSearchCV itself. KEEP AT 1: deadlocks "
                         "under repeated calls in a detached process on Windows.")
    ap.add_argument("--rf-n-jobs", type=int, default=1,
                    help="Parallel workers for RandomForestClassifier's own tree-building. "
                         "Safe to raise (e.g. 3-4) for real speed.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume: skip subjects already saved in --out; append new ones as they finish.")
    ap.add_argument("--out", default="results_loso_freq_coral")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    X = load_features_npz(Path(args.features)).astype(np.float64)
    meta = pd.read_csv(args.meta)
    label_col = next((c for c in ["movement", "label", "y_str", "status_mode", "y"] if c in meta.columns), None)
    subj_col = next((c for c in ["subject", "subject_id", "subject_int", "sid"] if c in meta.columns), None)
    if label_col is None or subj_col is None:
        raise KeyError(f"meta needs a label and subject column; has {list(meta.columns)}")
    y, _ = encode_labels(meta[label_col].astype(str).to_numpy())
    subjects = meta[subj_col].astype(int).to_numpy()

    import gc
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    subjects_u = sorted(np.unique(subjects).tolist())
    csvpaths = {m: out_dir / f"coral_{m}_subjectwise.csv" for m in models}
    done = {m: set() for m in models}
    if args.resume:
        for m in models:
            if csvpaths[m].exists():
                done[m] = set(pd.read_csv(csvpaths[m])["subject"].astype(int).tolist())
                if done[m]:
                    print(f"[resume] {m}: {len(done[m])} subjects already done, skipping")

    for heldout in subjects_u:
        if all(heldout in done[m] for m in models):
            continue
        te = (subjects == heldout); tr = ~te
        # Standardise on training subjects only (leak-free)
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X[tr]); Xt = scaler.transform(X[te])
        # CORAL: align source to this target subject (uses target features only)
        Xs_al, Xt_c = coral_align(Xs, Xt, lam=args.lam)

        ytr, gtr = y[tr], subjects[tr]; yte = y[te]
        inner_cv = GroupKFold(n_splits=min(args.inner_splits, len(np.unique(gtr))))
        for m in models:
            if heldout in done[m]:
                continue
            search = build_search(m, inner_cv, Xs_al, ytr, gtr, args.seed,
                                   n_jobs=args.n_jobs, rf_n_jobs=args.rf_n_jobs)
            search.fit(Xs_al, ytr)
            yhat = search.best_estimator_.predict(Xt_c)
            row = {"subject": int(heldout),
                   "f1_macro": float(f1_score(yte, yhat, average="macro", zero_division=0)),
                   "bal_acc": float(balanced_accuracy_score(yte, yhat)),
                   "acc": float(accuracy_score(yte, yhat))}
            # CHECKPOINT: append immediately (crash-safe / resumable)
            pd.DataFrame([row]).to_csv(csvpaths[m], mode="a",
                                       header=not csvpaths[m].exists(), index=False)
            done[m].add(heldout)
        del Xs, Xt, Xs_al, Xt_c
        gc.collect()
        print(f"[fold] Sub{heldout:02d} done")

    summary_rows = []
    for m in models:
        dfm = pd.read_csv(csvpaths[m]).drop_duplicates("subject").sort_values("subject")
        mean_f1 = dfm["f1_macro"].mean(); sd_f1 = dfm["f1_macro"].std(ddof=1)
        summary_rows.append({"method": "CORAL", "model": m,
                             "f1_macro_mean": round(mean_f1, 4),
                             "f1_macro_sd": round(sd_f1, 4), "n": len(dfm)})
        print(f"[CORAL] {m}: F1 = {mean_f1:.4f} ± {sd_f1:.4f}")

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "coral_summary.csv", index=False)
    print("\n================  SUMMARY (paste this back)  ================")
    print(summ.to_string(index=False))
    print("\nCompare against (same features, same folds):")
    print("  global z-score   : SVM 0.7080, RF 0.7220")
    print("  per-subject z-score (headline): SVM 0.7767, RF 0.7732")
    print(f"[save] {out_dir/'coral_summary.csv'}")


if __name__ == "__main__":
    main()

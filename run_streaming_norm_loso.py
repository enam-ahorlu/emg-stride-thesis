# run_streaming_norm_loso.py
# ---------------------------------------------------------------------------
# Action 1.1 — Causal / streaming version of per-subject normalisation.
#
# The thesis's headline result uses *transductive* per-subject z-score: the
# held-out test subject is standardised using statistics computed over ALL of
# that subject's windows (the whole session). That is leak-free w.r.t. labels
# but NON-CAUSAL — a real prosthesis cannot use future windows to normalise the
# current one. This script re-runs the classical LOSO with deployable, CAUSAL
# normalisation of the test subject and reports how much of the +6.9 pp gain
# survives.
#
# Training subjects are always normalised transductively (they are offline
# reference data). Only the TEST subject's normalisation changes:
#   transductive : whole-session stats        (baseline; reproduces ~0.777)
#   calibK       : stats from the FIRST K windows only, then frozen
#                  (a short, one-off, label-free calibration buffer)
#   running      : expanding causal stats — window t uses windows [0..t]
#                  with a warmup of the first --warmup windows
#
# Model configuration (SVM RBF, RF) and the nested 5-fold GroupKFold tuning
# mirror train_classical_loso.py exactly, so the ONLY thing that differs from
# the headline run is the test-subject normalisation.
#
# Example:
#   python run_streaming_norm_loso.py \
#       --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
#       --meta     features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --configs transductive,calib25,calib50,calib100,running \
#       --models SVM,RF
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

# Reuse the exact loaders/encoders from the authoritative trainer
from train_classical_loso import load_features_npz, encode_labels

TIME_COL_CANDIDATES = ["t_start", "win_start", "start", "t0"]


def causal_calib_stats(Xsub_ordered: np.ndarray, k: int):
    """Mean/std from the first k windows (calibration buffer)."""
    k = max(1, min(k, Xsub_ordered.shape[0]))
    mu = Xsub_ordered[:k].mean(axis=0, keepdims=True)
    sd = Xsub_ordered[:k].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def normalise_test_subject(Xsub: np.ndarray, order: np.ndarray, mode: str,
                           k: int, warmup: int) -> np.ndarray:
    """Return a causally-normalised copy of one subject's windows.

    Xsub  : (n, F) windows for the test subject (original row order)
    order : indices that sort Xsub by acquisition time
    """
    n = Xsub.shape[0]
    Xord = Xsub[order]
    out_ord = np.empty_like(Xord)

    if mode == "transductive":
        mu = Xord.mean(axis=0, keepdims=True)
        sd = Xord.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out_ord = (Xord - mu) / sd

    elif mode == "calib":
        mu, sd = causal_calib_stats(Xord, k)
        out_ord = (Xord - mu) / sd

    elif mode == "running":
        w = max(1, min(warmup, n))
        # warmup window: frozen stats from the first w windows
        mu0 = Xord[:w].mean(axis=0, keepdims=True)
        sd0 = Xord[:w].std(axis=0, keepdims=True)
        sd0 = np.where(sd0 < 1e-8, 1.0, sd0)
        out_ord[:w] = (Xord[:w] - mu0) / sd0
        # expanding causal stats for the remainder
        csum = Xord[:w].sum(axis=0)
        csqsum = (Xord[:w] ** 2).sum(axis=0)
        cnt = w
        for i in range(w, n):
            mu = csum / cnt
            var = np.maximum(csqsum / cnt - mu ** 2, 0.0)
            sd = np.sqrt(var)
            sd = np.where(sd < 1e-8, 1.0, sd)
            out_ord[i] = (Xord[i] - mu) / sd
            csum += Xord[i]
            csqsum += Xord[i] ** 2
            cnt += 1
    else:
        raise ValueError(f"unknown test-norm mode {mode}")

    # unsort back to original order
    out = np.empty_like(out_ord)
    out[order] = out_ord
    return out


def per_subject_transductive(X, subjects, mask):
    """Transductive per-subject z-score for the subjects selected by mask."""
    Xn = X.copy()
    for sid in np.unique(subjects[mask]):
        m = (subjects == sid) & mask
        Xs = X[m]
        mu = Xs.mean(axis=0, keepdims=True)
        sd = Xs.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xn[m] = (Xs - mu) / sd
    return Xn


def build_search(model_name, inner_cv, Xtr, ytr, gtr, seed, n_jobs=1, rf_n_jobs=1):
    # GridSearchCV's own n_jobs (parallelising across the grid/CV folds) deadlocks
    # under repeated calls in a detached/background process on Windows (loky reusable
    # executor hangs on the 2nd+ .fit()) -- keep it at 1. RandomForestClassifier's
    # OWN internal n_jobs (parallelising tree-building within one fit) is a different
    # joblib call path and is safe under repeated calls (verified 2026-07-08); use
    # rf_n_jobs to restore the multi-core speed train_classical_loso.py always had.
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


def parse_config(cfg: str):
    cfg = cfg.strip().lower()
    if cfg == "transductive":
        return ("transductive", 0)
    if cfg == "running":
        return ("running", 0)
    if cfg.startswith("calib"):
        k = int(cfg.replace("calib", ""))
        return ("calib", k)
    raise ValueError(f"unknown config '{cfg}' (use transductive, calibK, running)")


def main():
    ap = argparse.ArgumentParser("Causal/streaming per-subject normalisation LOSO (classical).")
    ap.add_argument("--features", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--models", default="SVM,RF")
    ap.add_argument("--configs", default="transductive,calib25,calib50,calib100,running",
                    help="Comma list: transductive, calibK (e.g. calib50), running")
    ap.add_argument("--warmup", type=int, default=16,
                    help="Warmup windows for the 'running' estimator (default 16 ~= 2 s at 125 ms step)")
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="Parallel workers for GridSearchCV itself (across grid/CV folds). "
                         "KEEP AT 1: deadlocks under repeated calls in a detached process on Windows.")
    ap.add_argument("--rf-n-jobs", type=int, default=1,
                    help="Parallel workers for RandomForestClassifier's OWN tree-building "
                         "(within one fit). Safe to raise (e.g. 3-4); this is what "
                         "train_classical_loso.py's --n-jobs controlled for RF speed.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume: skip (config, model, subject) combos already saved in --out, "
                         "and append new ones as they finish. Safe to re-run after a crash.")
    ap.add_argument("--out", default="results_loso_freq_streaming")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    X = load_features_npz(Path(args.features)).astype(np.float64)
    meta = pd.read_csv(args.meta)
    # labels + subjects (match train_classical_loso column conventions)
    label_col = next((c for c in ["movement", "label", "y_str", "status_mode", "y"] if c in meta.columns), None)
    subj_col = next((c for c in ["subject", "subject_id", "subject_int", "sid"] if c in meta.columns), None)
    if label_col is None or subj_col is None:
        raise KeyError(f"meta needs a label and subject column; has {list(meta.columns)}")
    y, label_map = encode_labels(meta[label_col].astype(str).to_numpy())
    subjects = meta[subj_col].astype(int).to_numpy()

    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    if time_col is None:
        print("[warn] no time column found; assuming rows are already in acquisition order per subject")
        time_vals = np.arange(len(y))
    else:
        time_vals = meta[time_col].to_numpy()
        print(f"[info] ordering test-subject windows by '{time_col}'")

    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    configs = [parse_config(c) for c in args.configs.split(",") if c.strip()]
    subjects_u = sorted(np.unique(subjects).tolist())

    import gc
    summary_rows = []
    for (mode, k) in configs:
        cfg_name = mode if mode != "calib" else f"calib{k}"
        print(f"\n================  TEST-NORM = {cfg_name}  ================")
        csvpaths = {m: out_dir / f"streaming_{cfg_name}_{m}_subjectwise.csv" for m in models}
        # --- resume: which (model, subject) are already saved? ---
        done = {m: set() for m in models}
        if args.resume:
            for m in models:
                if csvpaths[m].exists():
                    done[m] = set(pd.read_csv(csvpaths[m])["subject"].astype(int).tolist())
                    if done[m]:
                        print(f"  [resume] {m}: {len(done[m])} subjects already done, skipping them")

        for heldout in subjects_u:
            if all(heldout in done[m] for m in models):
                continue
            te = (subjects == heldout); tr = ~te
            # training subjects: transductive per-subject z-score
            Xn = per_subject_transductive(X, subjects, tr)
            # test subject: causal normalisation
            order = np.argsort(time_vals[te], kind="stable")
            Xn[te] = normalise_test_subject(X[te], order, mode, k, args.warmup)
            Xtr, ytr, gtr = Xn[tr], y[tr], subjects[tr]
            Xte, yte = Xn[te], y[te]
            inner_cv = GroupKFold(n_splits=min(args.inner_splits, len(np.unique(gtr))))

            for m in models:
                if heldout in done[m]:
                    continue
                search = build_search(m, inner_cv, Xtr, ytr, gtr, args.seed,
                                       n_jobs=args.n_jobs, rf_n_jobs=args.rf_n_jobs)
                search.fit(Xtr, ytr)
                yhat = search.best_estimator_.predict(Xte)
                row = {"subject": int(heldout),
                       "f1_macro": float(f1_score(yte, yhat, average="macro", zero_division=0)),
                       "bal_acc": float(balanced_accuracy_score(yte, yhat)),
                       "acc": float(accuracy_score(yte, yhat))}
                # CHECKPOINT: append this subject immediately (crash-safe / resumable)
                pd.DataFrame([row]).to_csv(csvpaths[m], mode="a",
                                           header=not csvpaths[m].exists(), index=False)
                done[m].add(heldout)
            del Xn, Xtr, Xte
            gc.collect()
            print(f"  [fold] Sub{heldout:02d} done")

        for m in models:
            dfm = pd.read_csv(csvpaths[m]).drop_duplicates("subject").sort_values("subject")
            mean_f1 = dfm["f1_macro"].mean(); sd_f1 = dfm["f1_macro"].std(ddof=1)
            summary_rows.append({"config": cfg_name, "model": m,
                                 "f1_macro_mean": round(mean_f1, 4),
                                 "f1_macro_sd": round(sd_f1, 4),
                                 "n": len(dfm)})
            print(f"  [{cfg_name}] {m}: F1 = {mean_f1:.4f} ± {sd_f1:.4f}")

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "streaming_norm_summary.csv", index=False)
    print("\n================  SUMMARY (paste this back)  ================")
    print(summ.to_string(index=False))
    print("\nReference (transductive per-subject z-score, headline): SVM 0.7767, RF 0.7732")
    print(f"[save] {out_dir/'streaming_norm_summary.csv'}")


if __name__ == "__main__":
    main()

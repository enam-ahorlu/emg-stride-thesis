#!/usr/bin/env python3
"""
run_within_subject_baseline.py
================================
EXPERIMENT_PLAN_CRITIQUE.md E4 (triage T11) -- at matched label budget, does a
purely subject-specific model beat the label-free cross-subject pipeline?

Three regimes, scored on the SAME truncated test set (subject's session minus
the first-N-per-class labelled/calibration windows), all under per-subject
(transductive) normalisation -- consistent with the thesis's headline pipeline:

  A - subject-specific only: train on the subject's own N labelled windows/class,
      no source data. Fixed config (no GridSearchCV; N is too small to tune):
      SVM(C=1, RBF, balanced) primary, RF(n_estimators=200, balanced) secondary.
  B - cross-subject, zero labels (the thesis pipeline): the existing per-subject
      LOSO model (best_params reused, single fit per subject), RE-SCORED on the
      truncated test set (never reuses the published full-session 0.7767).
  C - cross-subject + the same N labels: source training data (39 subjects,
      transductive per-subject norm) UNION the subject's own N labelled windows
      (same norm), best_params reused (no GridSearchCV -- infeasible at this
      scale: N(5) x draws(6) x subjects(40) x models(2) refits).

Draws: deterministic first-N (time-ordered, deployment-realistic) + 5 random
draws of N per class (robustness), mean + spread across draws.

Output: results_within_subject/{within_subject_subjectwise.csv,crossover.csv}
        report_figs/new_experiments/within_subject_learning_curve.png
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from train_classical_loso import load_features_npz, encode_labels
from run_streaming_norm_loso import per_subject_transductive, TIME_COL_CANDIDATES

ROOT = Path(__file__).parent
OUT = ROOT / "results_within_subject"; OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)
FEAT = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz"
META = ROOT / "features_out" / "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv"
LABELS = ["DNS", "STDUP", "UPS", "WAK"]
SEED = 42
N_LIST = [5, 10, 20, 50, 100]
N_RANDOM_DRAWS = 5
CSV_PATH = OUT / "within_subject_subjectwise.csv"


def make_svm(C=1):
    return SVC(kernel="rbf", C=C, gamma="scale", class_weight="balanced", random_state=SEED, cache_size=500)


def make_rf(n_estimators=200, max_depth=None, n_jobs=1):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  class_weight="balanced", random_state=SEED, n_jobs=n_jobs)


def labeled_indices(class_time_idx, N, draw, rng):
    """class_time_idx: dict{class:int -> array of original row-idx, time-sorted}.
    draw: 'first' or 'randK'. Returns concatenated labelled-window indices."""
    out = []
    for c, idx_sorted in class_time_idx.items():
        avail = len(idx_sorted)
        n = min(N, avail)
        if draw == "first":
            out.append(idx_sorted[:n])
        else:
            out.append(rng.choice(idx_sorted, n, replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rf-n-jobs", type=int, default=1)
    ap.add_argument("--models", default="SVM,RF")
    args = ap.parse_args()
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]

    X = load_features_npz(FEAT).astype(np.float64)
    meta = pd.read_csv(META)
    y, _ = encode_labels(meta["movement"].astype(str).to_numpy())
    subjects = meta["subject"].astype(int).to_numpy()
    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    tvals = meta[time_col].to_numpy() if time_col else np.arange(len(y))

    bp = json.loads(open(ROOT / "_bestparams.json").read())
    bp = {m: {int(k): v for k, v in d.items()} for m, d in bp.items()}

    subs_u = sorted(np.unique(subjects).tolist())
    draws = ["first"] + [f"rand{i}" for i in range(N_RANDOM_DRAWS)]

    done = set()
    if args.resume and CSV_PATH.exists():
        prev = pd.read_csv(CSV_PATH)
        done = set(zip(prev.subject, prev.N, prev.draw, prev.model))
        print(f"[resume] {len(done)} (subject,N,draw,model) rows already done")

    for heldout in subs_u:
        t_sub0 = time.time()
        # Skip the ENTIRE subject (no data load, no regime-B refit) if every
        # (N,draw,model) combo for it is already checkpointed. Without this,
        # every restart re-paid a full regime-B SVM+RF refit for EVERY already-
        # finished subject just to reach the frontier -- fine at subject 2, but
        # by subject 30 that's ~30 wasted refits (tens of minutes) per restart,
        # right when we're relying on frequent cheap restarts as the main
        # defense against the process's own creeping memory footprint.
        expected_keys = {(int(heldout), int(N), draw, m)
                         for N in N_LIST for draw in draws for m in models}
        if expected_keys <= done:
            print(f"[fold] Sub{heldout:02d} already complete, skipped ({time.time()-t_sub0:.0f}s)", flush=True)
            continue

        te_all = (subjects == heldout); tr_all = ~te_all
        X_persubj = per_subject_transductive(X, subjects, np.ones_like(tr_all, dtype=bool))  # all 40 transductive
        Xs, ys = X_persubj[te_all], y[te_all]
        tvals_s = tvals[te_all]

        # time-sorted original-row indices per class, within this subject
        class_time_idx = {}
        orig_idx = np.where(te_all)[0]
        for c in range(len(LABELS)):
            m = ys == c
            idx_c = orig_idx[m]
            order_c = np.argsort(tvals_s[m], kind="stable")
            class_time_idx[c] = idx_c[order_c]

        # ---- regime B: fit ONCE per subject (model doesn't depend on N/draw) ----
        Xtr_b = X_persubj[tr_all]; ytr_b = y[tr_all]
        regimeB_pred = {}
        for m in models:
            if m == "SVM":
                clf = make_svm(bp["SVM"][heldout]["clf__C"])
            else:
                clf = make_rf(bp["RF"][heldout]["clf__n_estimators"], bp["RF"][heldout]["clf__max_depth"],
                              n_jobs=args.rf_n_jobs)
            clf.fit(Xtr_b, ytr_b)
            regimeB_pred[m] = clf.predict(Xs)   # full-subject predictions; subset per (N,draw) below
            del clf
        gc.collect()  # regime-B RF (up to 500 trees, unconstrained depth) no longer needed past here

        n_written_this_subject = 0
        for N in N_LIST:
            for draw in draws:
                rng = np.random.default_rng(SEED + heldout * 1000 + N * 10 + draws.index(draw))
                lab_idx = labeled_indices(class_time_idx, N, draw, rng)
                lab_mask_local = np.isin(orig_idx, lab_idx)
                test_mask_local = ~lab_mask_local
                if test_mask_local.sum() == 0:
                    continue
                Xte, yte = Xs[test_mask_local], ys[test_mask_local]

                for m in models:
                    key = (int(heldout), int(N), draw, m)
                    if key in done:
                        continue

                    # Regime A: subject-specific only
                    Xa, ya = Xs[lab_mask_local], ys[lab_mask_local]
                    f1_a = np.nan
                    if len(np.unique(ya)) >= 2 and len(ya) >= len(LABELS):
                        clf_a = make_svm() if m == "SVM" else make_rf(n_jobs=args.rf_n_jobs)
                        clf_a.fit(Xa, ya)
                        f1_a = f1_score(yte, clf_a.predict(Xte), average="macro", zero_division=0)

                    # Regime B: cross-subject zero-label, re-scored on truncated test set
                    f1_b = f1_score(yte, regimeB_pred[m][test_mask_local], average="macro", zero_division=0)

                    # Regime C: cross-subject + N labels pooled
                    Xtr_c = np.vstack([Xtr_b, Xa]); ytr_c = np.concatenate([ytr_b, ya])
                    if m == "SVM":
                        clf_c = make_svm(bp["SVM"][heldout]["clf__C"])
                    else:
                        clf_c = make_rf(bp["RF"][heldout]["clf__n_estimators"], bp["RF"][heldout]["clf__max_depth"],
                                        n_jobs=args.rf_n_jobs)
                    clf_c.fit(Xtr_c, ytr_c)
                    f1_c = f1_score(yte, clf_c.predict(Xte), average="macro", zero_division=0)
                    del Xtr_c, ytr_c, clf_c  # Xtr_c is a fresh ~14 MB vstack copy every iteration (60x/subject)
                    gc.collect()

                    # CHECKPOINT: write this row immediately (crash/kill-safe). Writing
                    # only once at the end of the subject (as before) meant a kill
                    # ANYWHERE during a subject's ~60 combos discarded ALL of that
                    # subject's progress -- on a memory-constrained machine where the
                    # guard may kill mid-subject, that turned every restart into a
                    # full redo of the subject instead of resuming a few rows in.
                    row = dict(subject=int(heldout), N=int(N), draw=draw, model=m,
                              n_labeled=int(lab_mask_local.sum()), n_test=int(test_mask_local.sum()),
                              f1_A_subject_only=round(float(f1_a), 4),
                              f1_B_crosssubject_zero_label=round(float(f1_b), 4),
                              f1_C_crosssubject_plus_labels=round(float(f1_c), 4))
                    pd.DataFrame([row]).to_csv(CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)
                    done.add(key)
                    n_written_this_subject += 1
                    if n_written_this_subject % 10 == 0:
                        print(f"  Sub{heldout:02d}: {n_written_this_subject}/60 combos done "
                              f"({time.time()-t_sub0:.0f}s elapsed)", flush=True)
        print(f"[fold] Sub{heldout:02d} done ({time.time()-t_sub0:.0f}s)", flush=True)

    # ---- summary + crossover N* ----
    df = pd.read_csv(CSV_PATH).drop_duplicates(["subject", "N", "draw", "model"])
    summ = df.groupby(["model", "N", "draw"])[["f1_A_subject_only", "f1_B_crosssubject_zero_label",
                                                "f1_C_crosssubject_plus_labels"]].mean().reset_index()
    summ.to_csv(OUT / "within_subject_summary.csv", index=False)
    print(summ.to_string(index=False))

    crossover_rows = []
    for m in models:
        first_only = summ[(summ.model == m) & (summ.draw == "first")].sort_values("N")
        nstar = None
        for _, r in first_only.iterrows():
            if r.f1_A_subject_only >= r.f1_B_crosssubject_zero_label:
                nstar = int(r.N); break
        crossover_rows.append(dict(model=m, N_star=nstar,
                                   note="smallest N (first-N draw) where regime A >= regime B" if nstar
                                        else "no crossover within tested N range (A never catches B)"))
    crossover_df = pd.DataFrame(crossover_rows)
    crossover_df.to_csv(OUT / "crossover.csv", index=False)
    print(crossover_df.to_string(index=False))

    # ---- learning curve figure ----
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 5), squeeze=False)
    for mi, m in enumerate(models):
        ax = axes[0][mi]
        sub = df[df.model == m]
        agg = sub.groupby("N").agg(
            A_mean=("f1_A_subject_only", "mean"), A_sd=("f1_A_subject_only", "std"),
            B_mean=("f1_B_crosssubject_zero_label", "mean"), B_sd=("f1_B_crosssubject_zero_label", "std"),
            C_mean=("f1_C_crosssubject_plus_labels", "mean"), C_sd=("f1_C_crosssubject_plus_labels", "std"),
        ).reset_index()
        for col, label, color in [("A", "A: subject-only", "#C44E52"),
                                   ("B", "B: cross-subject, 0 labels", "#4C72B0"),
                                   ("C", "C: cross-subject + N labels", "#55A868")]:
            ax.errorbar(agg["N"], agg[f"{col}_mean"], yerr=agg[f"{col}_sd"], marker="o", label=label, color=color)
        ax.set_xlabel("N labelled windows / class"); ax.set_ylabel("macro-F1 (truncated test set)")
        ax.set_title(f"{m}: within-subject label budget (mean +/- SD over subjects x draws)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "within_subject_learning_curve.png", dpi=150)
    plt.close(fig)
    print(f"[save] {FIGDIR / 'within_subject_learning_curve.png'}")


if __name__ == "__main__":
    main()

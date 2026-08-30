# run_lda_loso.py
# ---------------------------------------------------------------------------
# Reviewer ask (multiple critiques): carry LDA through LOSO as the "classical
# minimal" baseline, not just as the SD baseline it was dropped after.
#
# This runs Linear Discriminant Analysis under the SAME nested-LOSO protocol as
# the headline SVM/RF: per-subject z-score normalisation of the Freq feature set
# (the dominant intervention), inner 5-fold GroupKFold tuning of the shrinkage
# strength, macro-F1 scoring. It reports where LDA lands relative to the SVM
# (0.777), RF (0.773) and CNN (0.754) under identical conditions, so the
# classical-vs-deep comparison is complete.
#
# Two normalisation modes are supported so LDA also slots into the normalisation
# ablation (global vs per-subject), mirroring Table 4.7:
#   --norm-mode per_subject   (headline; transductive per-subject z-score)
#   --norm-mode global        (StandardScaler fit on training subjects only)
#
# Example:
#   python run_lda_loso.py \
#       --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
#       --meta     features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --norm-mode per_subject --out results_lda_persubj
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

from train_classical_loso import load_features_npz, encode_labels
from run_streaming_norm_loso import per_subject_transductive


def build_search(inner_cv, Xtr, ytr, gtr, n_jobs=1):
    # LDA with the lsqr solver supports Ledoit-Wolf ('auto') and manual shrinkage.
    pipe = Pipeline([("clf", LinearDiscriminantAnalysis(solver="lsqr"))])
    grid = {"clf__shrinkage": ["auto", 0.0, 0.1, 0.25, 0.5, 1.0]}
    return GridSearchCV(pipe, grid, scoring="f1_macro",
                        cv=list(inner_cv.split(Xtr, ytr, groups=gtr)),
                        n_jobs=n_jobs, refit=True)


def main():
    ap = argparse.ArgumentParser("LDA classical-minimal baseline under nested LOSO.")
    ap.add_argument("--features", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--norm-mode", default="per_subject", choices=["per_subject", "global"])
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results_lda_persubj")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lda_subjectwise.csv"

    X = load_features_npz(Path(args.features)).astype(np.float64)
    meta = pd.read_csv(args.meta)
    label_col = next((c for c in ["movement", "label", "y_str", "status_mode", "y"] if c in meta.columns), None)
    subj_col = next((c for c in ["subject", "subject_id", "subject_int", "sid"] if c in meta.columns), None)
    y, _ = encode_labels(meta[label_col].astype(str).to_numpy())
    subjects = meta[subj_col].astype(int).to_numpy()
    subjects_u = sorted(np.unique(subjects).tolist())

    done = set()
    if args.resume and csv_path.exists():
        done = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())
        print(f"[resume] {len(done)} subjects already done")

    # per-subject transductive z-score is applied to ALL subjects once, up front
    Xn_persubj = per_subject_transductive(X, subjects, np.ones(len(y), bool)) if args.norm_mode == "per_subject" else None

    for heldout in subjects_u:
        if heldout in done:
            continue
        te = (subjects == heldout); tr = ~te
        if args.norm_mode == "per_subject":
            Xtr, Xte = Xn_persubj[tr], Xn_persubj[te]
        else:  # global: StandardScaler fit on training subjects only
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        ytr, gtr, yte = y[tr], subjects[tr], y[te]
        inner_cv = GroupKFold(n_splits=min(args.inner_splits, len(np.unique(gtr))))
        search = build_search(inner_cv, Xtr, ytr, gtr, n_jobs=args.n_jobs)
        search.fit(Xtr, ytr)
        yhat = search.best_estimator_.predict(Xte)
        row = {"subject": int(heldout),
               "f1_macro": float(f1_score(yte, yhat, average="macro", zero_division=0)),
               "bal_acc": float(balanced_accuracy_score(yte, yhat)),
               "acc": float(accuracy_score(yte, yhat)),
               "best_shrinkage": str(search.best_params_["clf__shrinkage"])}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"[fold] Sub{heldout:02d} LDA f1={row['f1_macro']:.4f} (shrink={row['best_shrinkage']})", flush=True)

    df = pd.read_csv(csv_path).drop_duplicates("subject").sort_values("subject")
    mean_f1, sd_f1 = df["f1_macro"].mean(), df["f1_macro"].std(ddof=1)
    pd.DataFrame([{"method": "LDA", "norm_mode": args.norm_mode,
                   "f1_macro_mean": round(mean_f1, 4), "f1_macro_sd": round(sd_f1, 4),
                   "n": len(df)}]).to_csv(out_dir / "lda_summary.csv", index=False)
    print(f"\n[LDA/{args.norm_mode}] F1 = {mean_f1:.4f} ± {sd_f1:.4f} (n={len(df)})")
    print("Reference (same features/folds): SVM 0.7767, RF 0.7732, CNN 0.754; global-norm SVM 0.708, RF 0.722")


if __name__ == "__main__":
    main()

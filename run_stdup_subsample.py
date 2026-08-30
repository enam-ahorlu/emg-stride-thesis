# run_stdup_subsample.py
# ---------------------------------------------------------------------------
# Reviewer ask (Critiques 1 & 2): STDUP is the majority class (~56% of windows)
# and also the easiest (F1 ~0.95). Is that advantage biomechanical, or just a
# sample-size effect that macro-F1 masks rather than removes? This script tests
# it directly: it re-runs LOSO with the STDUP TRAINING windows downsampled to the
# mean count of the other three classes (test set untouched), and compares the
# per-class F1 against the imbalanced baseline.
#
# If STDUP F1 holds up under balanced training, the biomechanical-distinctiveness
# reading is supported. If it falls toward the other classes, part of the
# advantage was sample size. Fixed headline hyperparameters (SVM C=1; RF
# n_estimators=500) are used so the ONLY thing that changes between conditions is
# the STDUP training count (a clean, controlled comparison).
#
# Example:
#   python run_stdup_subsample.py \
#       --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz \
#       --meta     features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --models SVM,RF
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from train_classical_loso import load_features_npz, encode_labels
from run_streaming_norm_loso import per_subject_transductive


def fit_model(name, Xtr, ytr, seed):
    if name == "SVM":
        return SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", cache_size=500).fit(Xtr, ytr)
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=seed, n_jobs=4).fit(Xtr, ytr)


def balance_training(ytr_idx, stdup_id, rng):
    """Return row indices (into the training arrays) with STDUP downsampled to the
    mean count of the other three classes."""
    classes, counts = np.unique(ytr_idx, return_counts=True)
    other = counts[classes != stdup_id]
    target = int(round(other.mean()))
    keep = []
    for c in classes:
        idx = np.where(ytr_idx == c)[0]
        if c == stdup_id and len(idx) > target:
            idx = rng.choice(idx, size=target, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def main():
    ap = argparse.ArgumentParser("STDUP class-balance sub-sampling test under LOSO.")
    ap.add_argument("--features", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--models", default="SVM,RF")
    ap.add_argument("--conditions", default="imbalanced,balanced")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results_stdup_subsample")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stdup_subsample_subjectwise.csv"

    X = load_features_npz(Path(args.features)).astype(np.float64)
    meta = pd.read_csv(args.meta)
    subj = meta[next(c for c in ["subject","subject_id","sid"] if c in meta.columns)].astype(int).to_numpy()
    y, label_map = encode_labels(meta[next(c for c in ["movement","label","y"] if c in meta.columns)].astype(str).to_numpy())
    labels_sorted = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    stdup_id = label_map["STDUP"]
    Xn = per_subject_transductive(X, subj, np.ones(len(y), bool))
    subjects_u = sorted(np.unique(subj).tolist())
    models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    done = set()
    if args.resume and csv_path.exists():
        d = pd.read_csv(csv_path); done = set(zip(d.subject, d.model, d.condition))

    for heldout in subjects_u:
        te = (subj == heldout); tr = ~te
        Xtr_all, ytr_all = Xn[tr], y[tr]; Xte, yte = Xn[te], y[te]
        rng = np.random.default_rng(args.seed * 100 + heldout)
        for cond in conditions:
            if cond == "balanced":
                idx = balance_training(ytr_all, stdup_id, rng)
                Xtr, ytr = Xtr_all[idx], ytr_all[idx]
            else:
                Xtr, ytr = Xtr_all, ytr_all
            for m in models:
                if (heldout, m, cond) in done:
                    continue
                est = fit_model(m, Xtr, ytr, args.seed)
                yhat = est.predict(Xte)
                per_cls = f1_score(yte, yhat, labels=list(range(len(labels_sorted))), average=None, zero_division=0)
                row = {"subject": int(heldout), "model": m, "condition": cond,
                       "n_stdup_train": int((ytr == stdup_id).sum()),
                       "macro_f1": float(f1_score(yte, yhat, average="macro", zero_division=0))}
                for li, lab in enumerate(labels_sorted):
                    row[f"f1_{lab}"] = float(per_cls[li])
                pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
                print(f"[fold] Sub{heldout:02d} {m}/{cond}: macro={row['macro_f1']:.3f} "
                      f"STDUP={row['f1_STDUP']:.3f} (n_stdup_train={row['n_stdup_train']})", flush=True)

    d = pd.read_csv(csv_path).drop_duplicates(["subject", "model", "condition"])
    agg = d.groupby(["model", "condition"])[[c for c in d.columns if c.startswith("f1_") or c == "macro_f1"]].mean().round(4)
    agg.to_csv(out_dir / "stdup_subsample_summary.csv")
    print("\n=== MEAN per-class F1 by model x condition ===")
    print(agg.to_string())
    print("\nRead: if f1_STDUP stays high under 'balanced', the advantage is biomechanical, not sample-size.")


if __name__ == "__main__":
    main()

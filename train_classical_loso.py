# train_classical_loso.py
from __future__ import annotations

import argparse
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.base import BaseEstimator, TransformerMixin


# ---- helpers copied/compatible with your current conventions ----
LABEL_CANDIDATES = ["movement", "label", "status_mode", "Status", "y"]
SUBJECT_CANDIDATES = ["subject", "subject_id", "sid", "subj"]

def pick_first_existing_column(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {what} column. Tried {candidates}. Have {list(df.columns)}")

def load_features_npz(npz_path: Path) -> np.ndarray:
    obj = np.load(npz_path, allow_pickle=False)
    # your features NPZs are typically single array; pick first key
    key = obj.files[0]
    X = obj[key]
    if X.ndim != 2:
        raise ValueError(f"Expected 2D features (N,F). Got {X.shape} from key={key}")
    return X

def encode_labels(y_str: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    uniq = sorted(set([str(v) for v in y_str]))
    label_map = {lab: i for i, lab in enumerate(uniq)}
    y_int = np.array([label_map[str(v)] for v in y_str], dtype=np.int32)
    return y_int, label_map

def save_confusion_png_csv(cm: np.ndarray, labels: List[str], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"{stem}_confusion.csv")

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("LOSO Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_confusion.png", dpi=160)
    plt.close(fig)

def infer_ms_per_window(model, Xte: np.ndarray, n_sample: int = 200) -> float:
    if Xte.shape[0] == 0:
        return float("nan")
    n = min(n_sample, Xte.shape[0])
    Xs = Xte[:n]
    # warmup
    _ = model.predict(Xs)
    t0 = time.perf_counter()
    _ = model.predict(Xs)
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1000.0


@dataclass
class LosoRow:
    model: str
    heldout_subject: int
    n_test: int
    acc: float
    bal_acc: float
    f1_macro: float
    best_params: str
    fit_time_sec: float
    infer_ms_per_window: float


class ToFloat32(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser("Leakage-proof LOSO for classical models (nested tuning).")
    ap.add_argument("--features", required=True, help="Path to features .npz (N,F)")
    ap.add_argument("--meta", required=True, help="Path to meta CSV aligned to features rows (N,...)")
    ap.add_argument("--out", default="results_loso", help="Output dir")
    ap.add_argument("--models", default="SVM,RF", help="Comma-separated subset of {SVM,RF}")
    ap.add_argument("--no-scale", action="store_true", help="Disable StandardScaler (NOT recommended; kept for legacy comparison)")
    ap.add_argument("--inner-splits", type=int, default=5, help="GroupKFold splits for tuning (train subjects only)")
    ap.add_argument("--save-preds", action="store_true", help="Save y_true/y_pred aligned to full meta rows")
    ap.add_argument("--cv-scheme", default="loso", help="Tag for saved preds filenames (default: loso)")
    ap.add_argument("--resume", action="store_true", help="Resume from existing checkpoints in --out")
    ap.add_argument("--flush-preds", action="store_true", help="Save per-subject predictions each fold (recommended for long runs)")
    args = ap.parse_args()

    features_path = Path(args.features)
    meta_path = Path(args.meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)


    X = load_features_npz(features_path).astype(np.float32, copy=False)
    meta = pd.read_csv(meta_path)

    label_col = pick_first_existing_column(meta, LABEL_CANDIDATES, "label")
    subj_col = pick_first_existing_column(meta, SUBJECT_CANDIDATES, "subject")

    y_str = meta[label_col].astype(str).to_numpy()
    subjects = meta[subj_col].astype(int).to_numpy()
    y, label_map = encode_labels(y_str)

    inv = {v: k for k, v in label_map.items()}
    labels_sorted = [inv[i] for i in sorted(inv.keys())]

    wanted = [m.strip().upper() for m in args.models.split(",") if m.strip()]

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    ckpt_files = {m: ckpt_dir / f"{features_path.stem}__{m}_subjectwise_ckpt.csv" for m in wanted}
    pred_fold_dir = out_dir / "predictions_folds"
    pred_fold_dir.mkdir(exist_ok=True)

    results: Dict[str, List[LosoRow]] = {m: [] for m in wanted}

    done_subjects = {m: set() for m in wanted}
    if args.resume:
        for m in wanted:
            fp = ckpt_dir / f"{features_path.stem}__{m}_subjectwise_ckpt.csv"
            if fp.exists():
                prev = pd.read_csv(fp)
                if "heldout_subject" in prev.columns:
                    done_subjects[m] = set(prev["heldout_subject"].astype(int).tolist())
                # also preload rows into results so final save still works
                for _, r in prev.iterrows():
                    results[m].append(LosoRow(**r.to_dict()))


    # full-length prediction buffers (each sample predicted once: by its heldout fold)
    y_pred_full: Dict[str, np.ndarray] = {m: np.full_like(y, fill_value=-1) for m in wanted}

    unique_subjects = sorted(np.unique(subjects).tolist())
    print(f"[info] LOSO over {len(unique_subjects)} subjects")

    for heldout in unique_subjects:
        # Skip if ALL requested models already completed this subject
        if args.resume and all(heldout in done_subjects[m] for m in wanted):
            print(f"[resume] skipping heldout Sub{heldout:02d} (already done for all models)")
            continue

        te_mask = (subjects == heldout)
        tr_mask = ~te_mask

        Xtr, ytr, gtr = X[tr_mask], y[tr_mask], subjects[tr_mask]
        Xte, yte = X[te_mask], y[te_mask]

        # Inner CV for tuning: subject-group splits on TRAINING SUBJECTS only
        n_train_groups = len(np.unique(gtr))
        inner_splits = min(int(args.inner_splits), n_train_groups)
        inner_cv = GroupKFold(n_splits=inner_splits)

        for model_name in wanted:
            if args.resume and heldout in done_subjects.get(model_name, set()):
                print(f"[resume] skip {model_name} heldout Sub{heldout:02d}")
                continue

            # ---- StandardScaler is ALWAYS used (inside Pipeline → leak-free) ----
            # Rationale: ensures train-only normalisation within each LOSO
            # fold; particularly important for frequency-domain features
            # whose scale differs from time-domain features.
            use_scaler = not args.no_scale

            if model_name == "SVM":
                steps = []
                if use_scaler:
                    steps.append(("scaler", StandardScaler()))
                    steps.append(("to32", ToFloat32()))
                steps.append(("clf", SVC(kernel="rbf", class_weight="balanced", cache_size=500)))
                pipe = Pipeline(steps)

                param_grid = {
                    "clf__C": [1, 5, 10],
                    "clf__gamma": ["scale"],
                }

                search = GridSearchCV(
                    pipe,
                    param_grid=param_grid,
                    scoring="f1_macro",
                    cv=inner_cv.split(Xtr, ytr, groups=gtr),
                    n_jobs=-1,  # parallelise SVM grid search across cores
                    refit=True,
                    verbose=2
                )

            elif model_name == "RF":
                steps = []
                if use_scaler:
                    steps.append(("scaler", StandardScaler()))
                    steps.append(("to32", ToFloat32()))
                steps.append(("clf", RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1
                )))
                pipe = Pipeline(steps)

                param_grid = {
                    "clf__n_estimators": [200, 400, 500],
                    "clf__max_depth": [None, 10],
                }

                search = GridSearchCV(
                    pipe,
                    param_grid=param_grid,
                    scoring="f1_macro",
                    cv=inner_cv.split(Xtr, ytr, groups=gtr),
                    n_jobs=1,
                    refit=True,
                    verbose=2
                )
            else:
                continue

            t0 = time.perf_counter()
            search.fit(Xtr, ytr)
            t1 = time.perf_counter()

            best = search.best_estimator_
            yhat = best.predict(Xte)

            if args.flush_preds:
                # Save only this subject's predictions (safe, small, resume-friendly)
                np.save(pred_fold_dir / f"{features_path.stem}_{model_name}_sub{heldout:02d}_y_true.npy", yte.astype(np.int32, copy=False))
                np.save(pred_fold_dir / f"{features_path.stem}_{model_name}_sub{heldout:02d}_y_pred.npy", yhat.astype(np.int32, copy=False))

            # store fold predictions back into full vector
            y_pred_full[model_name][te_mask] = yhat.astype(np.int32, copy=False)

            acc = accuracy_score(yte, yhat)
            bal = balanced_accuracy_score(yte, yhat)
            f1 = f1_score(yte, yhat, average="macro", zero_division=0)

            infer_ms = infer_ms_per_window(best, Xte)

            results[model_name].append(LosoRow(
                model=model_name,
                heldout_subject=int(heldout),
                n_test=int(te_mask.sum()),
                acc=float(acc),
                bal_acc=float(bal),
                f1_macro=float(f1),
                best_params=str(search.best_params_),
                fit_time_sec=float(t1 - t0),
                infer_ms_per_window=float(infer_ms),
            ))

            # checkpoint append
            ckpt_path = ckpt_dir / f"{features_path.stem}__{model_name}_subjectwise_ckpt.csv"
            row_df = pd.DataFrame([results[model_name][-1].__dict__])
            if ckpt_path.exists():
                row_df.to_csv(ckpt_path, mode="a", header=False, index=False)
            else:
                row_df.to_csv(ckpt_path, index=False)

            done_subjects[model_name].add(int(heldout))

        print(f"[fold] heldout Sub{heldout:02d} done")

        # Encourage cleanup between folds (helps long runs on Windows)
        gc.collect()

    # ---- Save outputs ----
    for model_name, rows in results.items():
        df = pd.DataFrame([r.__dict__ for r in rows])
        df = df.drop_duplicates(subset=["heldout_subject"], keep="last").sort_values("heldout_subject")
        out_csv = out_dir / f"{features_path.stem}__{model_name}_nested_loso_subjectwise.csv"
        df.to_csv(out_csv, index=False)
        print(f"[save] {out_csv}")

        # Mean summary
        summary = {
            "model": model_name,
            "f1_macro_mean": float(df["f1_macro"].mean()),
            "bal_acc_mean": float(df["bal_acc"].mean()),
            "acc_mean": float(df["acc"].mean()),
            "f1_macro_sd": float(df["f1_macro"].std(ddof=1)),
            "bal_acc_sd": float(df["bal_acc"].std(ddof=1)),
        }
        pd.DataFrame([summary]).to_csv(out_dir / f"{features_path.stem}__{model_name}_nested_loso_summary.csv", index=False)

        # Global report + confusion
        yhat_full = y_pred_full[model_name]
        if (yhat_full < 0).any():
            raise RuntimeError(f"Some samples were never predicted for model={model_name}. Check LOSO loop.")
        rep = classification_report(y, yhat_full, target_names=labels_sorted, digits=4, zero_division=0)
        (out_dir / "reports").mkdir(exist_ok=True)
        with open(out_dir / "reports" / f"{features_path.stem}__{model_name}_LOSO_report.txt", "w", encoding="utf-8") as f:
            f.write(rep)

        cm = confusion_matrix(y, yhat_full)
        save_confusion_png_csv(cm, labels_sorted, out_dir / "confusion_matrices", f"{features_path.stem}__{model_name}_LOSO")

    # Save full predictions aligned to meta (so your compute_per_subject_metrics.py works)
    if args.save_preds:
        pred_dir = out_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        stem = features_path.stem
        scheme = args.cv_scheme

        for model_name in wanted:
            np.save(pred_dir / f"{stem}_{model_name}_{scheme}_y_true.npy", y.astype(np.int32, copy=False))
            np.save(pred_dir / f"{stem}_{model_name}_{scheme}_y_pred.npy", y_pred_full[model_name].astype(np.int32, copy=False))
        print(f"[save] preds -> {pred_dir}")

    print("DONE")


if __name__ == "__main__":
    main()

# train_classical.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import time  
import matplotlib.pyplot as plt  
import sys  


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler




# Config / helpers
DEFAULT_FEATURES_NPZ = "features_out/windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_base.npz"
DEFAULT_META_CSV = "features_out/windows_WAK_UPS_DNS_STDUP_v1_w150_ov50_conf60_AorR_features_meta.csv"

def derive_default_paths_from_feat_set(feat_set: str) -> Tuple[str, str]:
    if feat_set == "base":
        return DEFAULT_FEATURES_NPZ, DEFAULT_META_CSV

    # ext
    feat_ext = DEFAULT_FEATURES_NPZ.replace("_features_base.npz", "_features_ext.npz")
    meta_ext = DEFAULT_META_CSV  # meta CSV is the same regardless of base/ext
    return feat_ext, meta_ext


LABEL_CANDIDATES = ["movement", "mode_label", "label", "y_str", "y"]
SUBJECT_CANDIDATES = ["subject", "subject_id", "sid", "subj", "subj_id", "subject_int"]


def pick_first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
    what: str,
    required: bool = True,
) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(
            f"Could not find a {what} column. Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_features_npz(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"Features NPZ not found: {npz_path}")
    npz = np.load(npz_path, allow_pickle=False)
    # Expect numeric-only: a single key 'X' 
    if "X" in npz.files:
        X = npz["X"]
    else:
        # tolerate older naming if needed
        if len(npz.files) == 1:
            X = npz[npz.files[0]]
        else:
            raise KeyError(f"Expected key 'X' in {npz_path.name}. Found keys: {npz.files}")
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D (N,F). Got shape {X.shape}")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError(f"Expected numeric X. Got dtype {X.dtype}")
    return X


def load_meta_csv(meta_path: Path) -> pd.DataFrame:
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")
    df = pd.read_csv(meta_path)
    if len(df) == 0:
        raise ValueError(f"Meta CSV is empty: {meta_path}")
    return df


def encode_labels(y_str: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    # stable mapping
    uniq = sorted(pd.unique(y_str))
    label_map = {lab: i for i, lab in enumerate(uniq)}
    y_int = np.array([label_map[str(v)] for v in y_str], dtype=np.int32)
    return y_int, label_map


def build_run_tag(args) -> str:
    parts = []

    # model selection (so RF-only vs SVM-only are distinguishable)
    wanted = "-".join([m.strip().upper() for m in args.models.split(",") if m.strip()])
    if wanted:
        parts.append(wanted)

    # SVM knobs
    if any(m.strip().upper() == "SVM" for m in args.models.split(",")):
        parts.append(f"C{float(args.svm_c):g}")
        parts.append("scaled" if args.svm_scale else "unscaled")

    # RF knobs
    if any(m.strip().upper() == "RF" for m in args.models.split(",")):
        parts.append(f"trees{int(getattr(args, 'rf_n_estimators', 0))}")
        md = int(args.rf_max_depth)
        parts.append("mdNone" if md < 0 else f"md{md}")

    return "__".join(parts) if parts else "run"



def save_confusion_outputs(
    cm: np.ndarray,
    labels: List[str],
    out_dir: Path,
    stem: str,
    model_name: str,
) -> Tuple[Path, Path]:
    # Save confusion matrix as CSV + PNG.
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    out_csv = out_dir / f"{stem}_{model_name}_confusion.csv"
    cm_df.to_csv(out_csv, index=True)

    # PNG
    out_png = out_dir / f"{stem}_{model_name}_confusion.png"
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    return out_csv, out_png


@dataclass
class ModelResult:
    model: str
    n_samples: int
    n_features: int
    subjects: str
    cv_splits: int
    acc_mean: float
    bal_acc_mean: float
    f1_macro_mean: float
    acc_std: float
    bal_acc_std: float
    f1_macro_std: float
    fit_time_mean_sec: float
    pred_time_mean_sec: float
    infer_time_per_window_ms: float
    svm_c: float
    svm_scale: int
    rf_n_estimators: int
    rf_max_depth: int
    sv_support_vectors_mean: float
    sv_frac_mean: float
    sv_n_iter_mean: float




def eval_subject_dependent_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model,
    n_splits: int = 5,
    seed: int = 42,
    log_fold_dist: bool = False,  
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    # Subject-dependent baseline: stratified k-fold across windows.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    sv_counts = []
    sv_fracs = []
    sv_niters = []


    accs, bals, f1s = [], [], []

    # Store predictions aligned to the ORIGINAL row order.
    y_true_full = np.full(shape=(len(y),), fill_value=-1, dtype=np.int32)
    y_pred_full = np.full(shape=(len(y),), fill_value=-1, dtype=np.int32)
    fold_id_full = np.full(shape=(len(y),), fill_value=-1, dtype=np.int32)

    fit_times = []
    pred_times = []
    pred_windows = 0

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        if log_fold_dist:
            n_classes = int(y.max()) + 1
            tr_counts = np.bincount(y_tr, minlength=n_classes)
            te_counts = np.bincount(y_te, minlength=n_classes)
            print(f"[{model_name}] fold {fold}: train_counts={tr_counts.tolist()} test_counts={te_counts.tolist()}")


        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        # VM diagnostics: support vectors, fraction, iterations 
        try:
            svm_obj = None
            if hasattr(model, "named_steps") and "svc" in getattr(model, "named_steps", {}):
                svm_obj = model.named_steps["svc"]
            elif hasattr(model, "support_"):
                svm_obj = model

            if svm_obj is not None and hasattr(svm_obj, "support_"):
                sv_total = int(len(svm_obj.support_))
                sv_counts.append(sv_total)
                sv_fracs.append(float(sv_total) / float(len(X_tr)))

                # n_iter_ can be scalar or array depending on sklearn build
                if hasattr(svm_obj, "n_iter_"):
                    n_it = svm_obj.n_iter_
                    if isinstance(n_it, (list, tuple, np.ndarray)):
                        sv_niters.append(float(np.mean(n_it)))
                    else:
                        sv_niters.append(float(n_it))
        except Exception:
            # Don't let diagnostics break CV
            pass

        t1 = time.perf_counter()
        y_hat = model.predict(X_te)
        t2 = time.perf_counter()

        fit_times.append(t1 - t0)
        pred_times.append(t2 - t1)
        pred_windows += int(len(X_te))

        accs.append(accuracy_score(y_te, y_hat))
        bals.append(balanced_accuracy_score(y_te, y_hat))
        f1s.append(f1_score(y_te, y_hat, average="macro"))

        y_true_full[te] = y_te
        y_pred_full[te] = y_hat
        fold_id_full[te] = fold

        print(
            f"[{model_name}] fold {fold}/{n_splits}: "
            f"acc={accs[-1]:.4f}, bal_acc={bals[-1]:.4f}, f1_macro={f1s[-1]:.4f}"
        )

    # Every row must have been predicted exactly once
    if np.any(y_true_full < 0) or np.any(y_pred_full < 0):
        raise RuntimeError('Some CV rows were not assigned predictions. Check CV loop logic.')
    y_true_all = y_true_full
    y_pred_all = y_pred_full

    mean_fit = float(np.mean(fit_times)) if fit_times else 0.0
    mean_pred = float(np.mean(pred_times)) if pred_times else 0.0
    infer_per_window_ms = (sum(pred_times) / pred_windows * 1000.0) if pred_windows > 0 else 0.0

    summary = {
        "acc_mean": float(np.mean(accs)),
        "bal_acc_mean": float(np.mean(bals)),
        "f1_macro_mean": float(np.mean(f1s)),
        "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "bal_acc_std": float(np.std(bals, ddof=1)) if len(bals) > 1 else 0.0,
        "f1_macro_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "fit_time_mean_sec": mean_fit,
        "pred_time_mean_sec": mean_pred,
        "infer_time_per_window_ms": float(infer_per_window_ms),
        "sv_support_vectors_mean": float(np.mean(sv_counts)) if sv_counts else float("nan"),
        "sv_frac_mean": float(np.mean(sv_fracs)) if sv_fracs else float("nan"),
        "sv_n_iter_mean": float(np.mean(sv_niters)) if sv_niters else float("nan"),

    }
    return summary, y_true_all, y_pred_all, fold_id_full


def build_models(seed: int = 42, svm_scale: bool = False, svm_c: float = 5.0, rf_max_depth=None, rf_n_estimators: int = 300, include_lda: bool = True):
    # Balanced SVM baseline
    if svm_scale:
        svm = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(
                kernel="rbf",
                C=float(svm_c),
                gamma="scale",
                class_weight="balanced",
            )),
        ])
        svm_name = "SVM_RBF_balanced_scaled"
    else:
        svm = SVC(
            kernel="rbf",
            C=float(svm_c),
            gamma="scale",
            class_weight="balanced",
        )
        svm_name = "SVM_RBF_balanced"

    # Balanced RF baseline
    rf = RandomForestClassifier(
        n_estimators=int(rf_n_estimators),
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        max_depth=rf_max_depth,
    )


    lda = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis()),
    ])

    models = {
        svm_name: svm,
        "RF_balanced": rf,
    }
    if include_lda:
        models["LDA_scaled"] = lda
    return models




# Main
def main():
    ap = argparse.ArgumentParser(description="Subject-dependent classical ML baseline (SVM/RF) for EMG features.")
    ap.add_argument("--features", type=str, default=DEFAULT_FEATURES_NPZ, help="Path to *_features_base.npz or *_features_ext.npz")
    ap.add_argument("--meta", type=str, default=DEFAULT_META_CSV, help="Path to *_features_meta.csv")
    ap.add_argument(
        "--feat-set",
        choices=["base", "ext"],
        default=None,
        help="Convenience selector for default feature NPZ (base vs ext). Ignored if --features is explicitly provided.",
    ) 
    ap.add_argument("--subjects", type=str, default="all", help="Comma-separated subject IDs (e.g. '1,2,3') or 'all'")
    ap.add_argument("--splits", type=int, default=5, help="StratifiedKFold splits")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out", type=str, default="results_classical", help="Output directory for results")
    ap.add_argument("--svm-c", type=float, default=5.0,
                help="SVM RBF C value (manual, limited tuning).")
    ap.add_argument("--rf-max-depth", type=int, default=-1,
                    help="RF max_depth. Use -1 for None.")
    ap.add_argument("--models", type=str, default="SVM,RF,LDA",
                    help="Comma-separated subset of {SVM,RF,LDA} to run.")
    ap.add_argument("--save-preds", action="store_true",
                    help="Save CV predictions aligned to meta rows (for per-subject metrics).")
    ap.add_argument("--cv-scheme", type=str, default="subjdep",
                    help="CV scheme tag for filenames (default: subjdep).")
    ap.add_argument(
        "--log-fold-dist",
        action="store_true",
        help="Print class distribution per fold (train/test) to verify stratification.",
        )
    ap.add_argument(
        "--svm-scale",
        action="store_true",
        help="Apply StandardScaler inside CV folds for SVM (recommended)."
    )
    ap.add_argument(
        "--rf-n-estimators",
        type=int,
        default=300,
        help="RF number of trees (n_estimators). Used for sweep / convergence checks."
    )



    args = ap.parse_args()

    
    if args.feat_set is not None and "--features" not in sys.argv:
        feat_path, meta_path = derive_default_paths_from_feat_set(args.feat_set)
        args.features = feat_path

        if "--meta" not in sys.argv:
            args.meta = meta_path


    features_path = Path(args.features)
    meta_path = Path(args.meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    X = load_features_npz(features_path)
    meta_df = load_meta_csv(meta_path)

    # Pick label + subject columns
    label_col = pick_first_existing_column(meta_df, LABEL_CANDIDATES, what="label", required=True)
    subj_col = pick_first_existing_column(meta_df, SUBJECT_CANDIDATES, what="subject", required=True)

    # Filter subjects
    if args.subjects.strip().lower() == "all":
        keep_subjects = sorted(meta_df[subj_col].dropna().astype(int).unique().tolist())
    else:
        keep_subjects = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]

    mask = meta_df[subj_col].astype(int).isin(keep_subjects).to_numpy()


    if mask.sum() == 0:
        raise ValueError(f"No rows matched subjects={keep_subjects}. Check {subj_col} values in meta CSV.")

    meta_sub = meta_df.loc[mask].reset_index(drop=True)
    X_sub = X[mask]

    # Extract labels + encode
    y_str = meta_sub[label_col].astype(str).to_numpy()
    y_int, label_map = encode_labels(y_str)

    # Final sanity checks
    assert len(X_sub) == len(meta_sub) == len(y_int), "Length mismatch between X and meta/labels."
    print("\n==== DATA SUMMARY ====")
    print(f"Features file : {features_path}")
    print(f"Meta file     : {meta_path}")
    print(f"X shape       : {X_sub.shape} (N,F)")
    print(f"Label col     : {label_col}")
    print(f"Subject col   : {subj_col}")
    print(f"Subjects used : {sorted(meta_sub[subj_col].astype(int).unique().tolist())}")
    print("Label counts:")
    print(pd.Series(y_str).value_counts())

    # Save label map 
    label_map_path = out_dir / f"{features_path.stem}_label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[save] label_map: {label_map_path}")

    # Train/eval
    rf_depth = None if int(args.rf_max_depth) < 0 else int(args.rf_max_depth)
    models_all = build_models(
    seed=args.seed,
    svm_scale=args.svm_scale,
    svm_c=float(args.svm_c),
    rf_max_depth=rf_depth,
    rf_n_estimators=int(args.rf_n_estimators),
    include_lda=True
    )

    wanted = {m.strip().upper() for m in args.models.split(',') if m.strip()}
    models = {}
    for k, v in models_all.items():
        if k.startswith('SVM_'):
            tag = 'SVM'
        elif k == 'RF_balanced' or k.startswith('RF_'):
            tag = 'RF'
        elif k == 'LDA_scaled' or k.startswith('LDA_'):
            tag = 'LDA'
        else:
            tag = k.upper()
        if tag in wanted:
            models[k] = v
    if not models:
        raise ValueError(f'No models selected. --models={args.models} gave empty set.')

    results_rows = []
    for name, model in models.items():
        print(f"\n==== MODEL: {name} ====")
        summary, y_true_all, y_pred_all, fold_id_all = eval_subject_dependent_cv(
            X_sub, y_int, model_name=name, model=model,
            n_splits=args.splits, seed=args.seed,
            log_fold_dist=args.log_fold_dist
        )


        # Print report + confusion matrix
        inv_map = {v: k for k, v in label_map.items()}
        labels_sorted = [inv_map[i] for i in sorted(inv_map.keys())]

        report_txt = classification_report(y_true_all, y_pred_all, target_names=labels_sorted, digits=4)
        print("\nClassification report:")
        print(report_txt)

        cm = confusion_matrix(y_true_all, y_pred_all)
        print("Confusion matrix (rows=true, cols=pred):")
        print(cm)

        #save report + confusion outputs
        rep_path = out_dir / f"{features_path.stem}_{name}_report.txt"
        with open(rep_path, "w", encoding="utf-8") as f:
            f.write(report_txt)
        print(f"[save] report TXT: {rep_path}")

        cm_csv, cm_png = save_confusion_outputs(
            cm=cm,
            labels=labels_sorted,
            out_dir=out_dir,
            stem=features_path.stem,
            model_name=name,
        )
        print(f"[save] confusion CSV: {cm_csv}")
        print(f"[save] confusion PNG: {cm_png}")

        # save predictions aligned to meta_sub rows
        if args.save_preds:
            pred_dir = out_dir / "predictions"
            pred_dir.mkdir(parents=True, exist_ok=True)

            stem = features_path.stem
            scheme = args.cv_scheme

            ytrue_path = pred_dir / f"{stem}_{name}_{scheme}_y_true.npy"
            ypred_path = pred_dir / f"{stem}_{name}_{scheme}_y_pred.npy"
            fold_path  = pred_dir / f"{stem}_{name}_{scheme}_fold_id.npy"

            np.save(ytrue_path, y_true_all.astype(np.int32, copy=False))
            np.save(ypred_path, y_pred_all.astype(np.int32, copy=False))
            np.save(fold_path,  fold_id_all.astype(np.int32, copy=False))

            print(f"[save] y_true: {ytrue_path}")
            print(f"[save] y_pred: {ypred_path}")
            print(f"[save] fold_id: {fold_path}")


        results_rows.append(
            ModelResult(
                model=name,
                n_samples=int(X_sub.shape[0]),
                n_features=int(X_sub.shape[1]),
                subjects=",".join(map(str, keep_subjects)),
                cv_splits=int(args.splits),
                acc_mean=summary["acc_mean"],
                bal_acc_mean=summary["bal_acc_mean"],
                f1_macro_mean=summary["f1_macro_mean"],
                acc_std=summary["acc_std"],
                bal_acc_std=summary["bal_acc_std"],
                f1_macro_std=summary["f1_macro_std"],
                fit_time_mean_sec=summary["fit_time_mean_sec"],
                pred_time_mean_sec=summary["pred_time_mean_sec"],
                infer_time_per_window_ms=summary["infer_time_per_window_ms"],
                svm_c=float(args.svm_c),
                svm_scale=1 if args.svm_scale else 0,
                rf_n_estimators=int(args.rf_n_estimators),
                rf_max_depth=int(args.rf_max_depth),

                sv_support_vectors_mean=summary.get("sv_support_vectors_mean", float("nan")),
                sv_frac_mean=summary.get("sv_frac_mean", float("nan")),
                sv_n_iter_mean=summary.get("sv_n_iter_mean", float("nan")),

            ).__dict__
        )

    # Save results
    run_tag = build_run_tag(args)
    out_csv = out_dir / f"{features_path.stem}__{run_tag}_subjdep_cv.csv"

    pd.DataFrame(results_rows).to_csv(out_csv, index=False)
    print(f"\n[save] results CSV: {out_csv}")

    print("\nDONE")


if __name__ == "__main__":
    main()

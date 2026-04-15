#!/usr/bin/env python3
"""
run_ensemble_loso.py
====================
Ensemble evaluation under LOSO using existing per-fold hard predictions.

Since only hard predictions (y_pred) are saved (no probabilities), this uses
hard majority voting:
  - 2-model ensembles: where both agree, use the agreed prediction;
    ties broken by defaulting to the model with higher overall LOSO F1.
  - 3-model ensemble: true majority wins; all-disagree ties broken by
    the model with highest overall LOSO F1 (model A = SVM).

Ensembles evaluated:
  1. SVM + RF  (classical ensemble)
  2. CNN + SVM (cross-model)
  3. CNN + RF  (cross-model)
  4. SVM + RF + CNN (3-way)

All predictions use freq-72 features + per_subject normalization (best config).

Usage
-----
    python run_ensemble_loso.py
    python run_ensemble_loso.py --classical-dir results_loso_freq_persubj
    python run_ensemble_loso.py --skip-cross-model
"""
from __future__ import annotations

import argparse
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    classification_report,
)

ROOT = Path(__file__).parent
LABELS = ["DNS", "STDUP", "UPS", "WAK"]


def load_per_fold_preds(pred_dir: Path, model_key: str, n_subjects: int = 40):
    """Load per-fold y_true and y_pred for a given model.

    Handles two naming conventions:
      - predictions_folds/*_SVM_sub01_y_pred.npy  (classical)
      - predictions/*_CNN_loso_Sub01_y_pred.npy   (CNN)

    Returns dict: subject_id -> (y_true, y_pred)
    """
    preds = {}
    for s in range(1, n_subjects + 1):
        patterns = [
            f"*{model_key}*sub{s:02d}_y_pred.npy",
            f"*{model_key}*Sub{s:02d}_y_pred.npy",
        ]
        y_pred_file = None
        for pat in patterns:
            matches = list(pred_dir.glob(pat))
            if matches:
                y_pred_file = matches[0]
                break

        if y_pred_file is None:
            print(f"  [WARN] No prediction file for {model_key} sub{s:02d} in {pred_dir}")
            continue

        y_true_file = Path(str(y_pred_file).replace("_y_pred.npy", "_y_true.npy"))
        if not y_true_file.exists():
            print(f"  [WARN] No y_true file for {model_key} sub{s:02d}")
            continue

        preds[s] = (np.load(y_true_file), np.load(y_pred_file))

    return preds


def majority_vote_2(preds_a: np.ndarray, preds_b: np.ndarray) -> np.ndarray:
    """Hard majority vote between two models.

    Where both agree -> agreed prediction.
    Where they disagree -> default to model A (the model with higher overall F1,
    passed as the first argument by the caller).
    """
    result = np.copy(preds_a)  # default = model A on disagreements
    agree = preds_a == preds_b
    result[agree] = preds_a[agree]
    # Disagreements: keep preds_a (the higher-F1 model)
    return result


def majority_vote_3(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    preds_c: np.ndarray,
) -> np.ndarray:
    """Hard majority vote among three models.

    True majority (2+) wins.  All-disagree ties broken by model A
    (the model with highest overall F1, passed first by the caller).
    """
    n = len(preds_a)
    result = np.empty(n, dtype=preds_a.dtype)
    for i in range(n):
        a, b, c = preds_a[i], preds_b[i], preds_c[i]
        if a == b or a == c:
            result[i] = a
        elif b == c:
            result[i] = b
        else:
            # All different: default to model A
            result[i] = a
    return result


def validate_ytrue_match(
    preds_a: dict, preds_b: dict, name_a: str, name_b: str
) -> tuple[int, int]:
    """Validate that y_true arrays match between two models.

    Returns (n_matched, n_mismatched).
    """
    subjects = sorted(set(preds_a.keys()) & set(preds_b.keys()))
    n_match = 0
    n_mismatch = 0
    for s in subjects:
        yt_a = preds_a[s][0]
        yt_b = preds_b[s][0]
        if len(yt_a) == len(yt_b) and np.array_equal(yt_a, yt_b):
            n_match += 1
        else:
            n_mismatch += 1
            if len(yt_a) != len(yt_b):
                print(
                    f"  [MISMATCH] Sub{s:02d}: {name_a} has {len(yt_a)} samples, "
                    f"{name_b} has {len(yt_b)} samples"
                )
            else:
                n_diff = int((yt_a != yt_b).sum())
                print(
                    f"  [MISMATCH] Sub{s:02d}: {name_a} vs {name_b} differ in "
                    f"{n_diff}/{len(yt_a)} labels"
                )
    return n_match, n_mismatch


def compute_individual_f1(preds_dict: dict) -> float:
    """Compute overall macro F1 for a single model across all folds."""
    y_true_all = []
    y_pred_all = []
    for s in sorted(preds_dict.keys()):
        yt, yp = preds_dict[s]
        y_true_all.append(yt)
        y_pred_all.append(yp)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return float(f1_score(y_true_all, y_pred_all, average="macro"))


def evaluate_ensemble(
    preds_dict_a: dict,
    preds_dict_b: dict,
    name_a: str,
    name_b: str,
    preds_dict_c: dict = None,
    name_c: str = None,
) -> pd.DataFrame:
    """Evaluate ensemble across all LOSO folds.

    Model A should always be the model with highest individual F1
    (used for tie-breaking).

    If preds_dict_c is provided, do 3-way majority vote.
    """
    subjects = sorted(set(preds_dict_a.keys()) & set(preds_dict_b.keys()))
    if preds_dict_c is not None:
        subjects = sorted(set(subjects) & set(preds_dict_c.keys()))

    rows = []
    for s in subjects:
        y_true_a, y_pred_a = preds_dict_a[s]
        y_true_b, y_pred_b = preds_dict_b[s]

        # Verify y_true arrays match
        if len(y_true_a) != len(y_true_b):
            raise ValueError(
                f"Subject {s}: prediction length mismatch between "
                f"{name_a} ({len(y_true_a)}) and {name_b} ({len(y_true_b)}). "
                f"This indicates an upstream data inconsistency that must be resolved."
            )
        elif not np.array_equal(y_true_a, y_true_b):
            n_diff = int((y_true_a != y_true_b).sum())
            print(
                f"  [WARN] Subject {s}: y_true mismatch ({n_diff} samples differ) "
                f"between {name_a} and {name_b}"
            )

        if preds_dict_c is not None:
            y_true_c, y_pred_c = preds_dict_c[s]
            if len(y_pred_c) != len(y_pred_a):
                min_len = min(len(y_pred_a), len(y_pred_c))
                y_true_a = y_true_a[:min_len]
                y_pred_a = y_pred_a[:min_len]
                y_pred_b = y_pred_b[:min_len]
                y_pred_c = y_pred_c[:min_len]
            y_ens = majority_vote_3(y_pred_a, y_pred_b, y_pred_c)
        else:
            y_ens = majority_vote_2(y_pred_a, y_pred_b)

        y_true = y_true_a
        f1 = f1_score(y_true, y_ens, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_ens)
        acc = accuracy_score(y_true, y_ens)

        # Per-class F1
        per_class_f1 = f1_score(y_true, y_ens, average=None, zero_division=0)
        class_dict = {}
        for idx, label in enumerate(LABELS):
            if idx < len(per_class_f1):
                class_dict[f"f1_{label}"] = round(per_class_f1[idx], 6)

        row = {
            "subject": s,
            "n_test": len(y_true),
            "acc": round(acc, 6),
            "bal_acc": round(bal_acc, 6),
            "f1_macro": round(f1, 6),
        }
        row.update(class_dict)
        rows.append(row)

    return pd.DataFrame(rows)


def print_ensemble_results(
    df: pd.DataFrame,
    ensemble_name: str,
    individual_f1s: dict[str, float],
) -> None:
    """Print ensemble summary with comparison to individual models."""
    ens_f1 = df["f1_macro"].mean()
    ens_f1_sd = df["f1_macro"].std(ddof=1)
    ens_bal = df["bal_acc"].mean()
    ens_bal_sd = df["bal_acc"].std(ddof=1)
    ens_acc = df["acc"].mean()

    print(f"\n  {ensemble_name} Ensemble:")
    print(f"    F1 macro:    {ens_f1:.4f} +/- {ens_f1_sd:.4f}")
    print(f"    Bal acc:     {ens_bal:.4f} +/- {ens_bal_sd:.4f}")
    print(f"    Accuracy:    {ens_acc:.4f}")

    # Per-class F1
    class_cols = [c for c in df.columns if c.startswith("f1_") and c != "f1_macro"]
    if class_cols:
        parts = []
        for c in class_cols:
            label = c.replace("f1_", "")
            parts.append(f"{label}={df[c].mean():.3f}")
        print(f"    Per-class:   {', '.join(parts)}")

    # Compare to individual models
    print(f"    --- vs individual models ---")
    best_indiv_name = max(individual_f1s, key=individual_f1s.get)
    best_indiv_f1 = individual_f1s[best_indiv_name]
    for name, f1 in individual_f1s.items():
        delta = ens_f1 - f1
        sign = "+" if delta >= 0 else ""
        print(f"    vs {name:<5s} (F1={f1:.4f}):  {sign}{delta:.4f}")
    delta_best = ens_f1 - best_indiv_f1
    sign = "+" if delta_best >= 0 else ""
    print(f"    Delta vs best ({best_indiv_name}): {sign}{delta_best:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble evaluation under LOSO")
    ap.add_argument(
        "--classical-dir",
        default="results_loso_freq_persubj",
        help="Directory with classical LOSO per-fold predictions (default: per_subject norm)",
    )
    ap.add_argument(
        "--cnn-dir",
        default="results_cnn_loso_norm_persubj",
        help="Directory with CNN LOSO per-fold predictions",
    )
    ap.add_argument(
        "--skip-cross-model",
        action="store_true",
        help="Skip cross-model ensembles (CNN+SVM, CNN+RF)",
    )
    ap.add_argument(
        "--out",
        default="report_figs",
        help="Output directory for results CSVs",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    classical_pred_dir = ROOT / args.classical_dir / "predictions_folds"
    cnn_pred_dir = ROOT / args.cnn_dir / "predictions"

    # ---- Load all predictions ----
    print("Loading predictions...")
    svm_preds = load_per_fold_preds(classical_pred_dir, "SVM")
    rf_preds = load_per_fold_preds(classical_pred_dir, "RF")
    print(f"  SVM folds loaded: {len(svm_preds)}/40")
    print(f"  RF  folds loaded: {len(rf_preds)}/40")

    cnn_preds = {}
    if not args.skip_cross_model:
        cnn_preds = load_per_fold_preds(cnn_pred_dir, "CNN")
        print(f"  CNN folds loaded: {len(cnn_preds)}/40")

    # ---- Compute individual model F1s (for tie-breaking and comparison) ----
    individual_f1 = {}
    if svm_preds:
        individual_f1["SVM"] = compute_individual_f1(svm_preds)
        print(f"\n  Individual SVM F1 (overall): {individual_f1['SVM']:.4f}")
    if rf_preds:
        individual_f1["RF"] = compute_individual_f1(rf_preds)
        print(f"  Individual RF  F1 (overall): {individual_f1['RF']:.4f}")
    if cnn_preds:
        individual_f1["CNN"] = compute_individual_f1(cnn_preds)
        print(f"  Individual CNN F1 (overall): {individual_f1['CNN']:.4f}")

    # ---- Validate y_true consistency ----
    print("\nValidating y_true consistency...")

    if svm_preds and rf_preds:
        n_ok, n_bad = validate_ytrue_match(svm_preds, rf_preds, "SVM", "RF")
        print(f"  SVM vs RF:  {n_ok} matched, {n_bad} mismatched")

    if cnn_preds and svm_preds:
        n_ok, n_bad = validate_ytrue_match(cnn_preds, svm_preds, "CNN", "SVM")
        print(f"  CNN vs SVM: {n_ok} matched, {n_bad} mismatched")

    if cnn_preds and rf_preds:
        n_ok, n_bad = validate_ytrue_match(cnn_preds, rf_preds, "CNN", "RF")
        print(f"  CNN vs RF:  {n_ok} matched, {n_bad} mismatched")

    # Determine tie-break order: model A = higher F1
    def ordered_ab(name_a, name_b, preds_a, preds_b):
        """Return (higher_f1_preds, lower_f1_preds, higher_name, lower_name)."""
        f1_a = individual_f1.get(name_a, 0)
        f1_b = individual_f1.get(name_b, 0)
        if f1_a >= f1_b:
            return preds_a, preds_b, name_a, name_b
        else:
            return preds_b, preds_a, name_b, name_a

    df_svmrf = None
    df_cnnsvm = None
    df_cnnrf = None
    df_3way = None

    # ---- 1. SVM + RF ensemble ----
    print(f"\n{'='*60}")
    print("1. SVM + RF HARD MAJORITY VOTING ENSEMBLE")
    print("=" * 60)

    if svm_preds and rf_preds:
        pa, pb, na, nb = ordered_ab("SVM", "RF", svm_preds, rf_preds)
        print(f"  Tie-break priority: {na} (F1={individual_f1[na]:.4f})")

        df_svmrf = evaluate_ensemble(pa, pb, na, nb)
        df_svmrf.insert(0, "model", "SVM+RF_ensemble")

        out_path = out_dir / "ensemble_svm_rf_per_subject.csv"
        df_svmrf.to_csv(out_path, index=False)
        print_ensemble_results(df_svmrf, "SVM+RF", {"SVM": individual_f1["SVM"], "RF": individual_f1["RF"]})
        print(f"  [saved] {out_path}")
    else:
        print("  [SKIP] Not enough prediction files for SVM+RF ensemble")

    if args.skip_cross_model:
        print("\n  [SKIP] Cross-model ensembles (--skip-cross-model)")
    else:
        # ---- 2. CNN + SVM ensemble ----
        print(f"\n{'='*60}")
        print("2. CNN + SVM CROSS-MODEL ENSEMBLE")
        print("=" * 60)

        if cnn_preds and svm_preds:
            pa, pb, na, nb = ordered_ab("SVM", "CNN", svm_preds, cnn_preds)
            print(f"  Tie-break priority: {na} (F1={individual_f1[na]:.4f})")

            df_cnnsvm = evaluate_ensemble(pa, pb, na, nb)
            df_cnnsvm.insert(0, "model", "CNN+SVM_ensemble")

            out_path = out_dir / "ensemble_cnn_svm_per_subject.csv"
            df_cnnsvm.to_csv(out_path, index=False)
            print_ensemble_results(
                df_cnnsvm, "CNN+SVM",
                {"SVM": individual_f1["SVM"], "CNN": individual_f1["CNN"]},
            )
            print(f"  [saved] {out_path}")
        else:
            print("  [SKIP] Not enough prediction files for CNN+SVM ensemble")

        # ---- 3. CNN + RF ensemble ----
        print(f"\n{'='*60}")
        print("3. CNN + RF CROSS-MODEL ENSEMBLE")
        print("=" * 60)

        if cnn_preds and rf_preds:
            pa, pb, na, nb = ordered_ab("RF", "CNN", rf_preds, cnn_preds)
            print(f"  Tie-break priority: {na} (F1={individual_f1[na]:.4f})")

            df_cnnrf = evaluate_ensemble(pa, pb, na, nb)
            df_cnnrf.insert(0, "model", "CNN+RF_ensemble")

            out_path = out_dir / "ensemble_cnn_rf_per_subject.csv"
            df_cnnrf.to_csv(out_path, index=False)
            print_ensemble_results(
                df_cnnrf, "CNN+RF",
                {"RF": individual_f1["RF"], "CNN": individual_f1["CNN"]},
            )
            print(f"  [saved] {out_path}")
        else:
            print("  [SKIP] Not enough prediction files for CNN+RF ensemble")

        # ---- 4. 3-way: SVM + RF + CNN ensemble ----
        print(f"\n{'='*60}")
        print("4. SVM + RF + CNN THREE-WAY ENSEMBLE")
        print("=" * 60)

        if svm_preds and rf_preds and cnn_preds:
            # Model A = highest F1 (tie-break for all-disagree case)
            sorted_models = sorted(individual_f1.items(), key=lambda x: x[1], reverse=True)
            model_order = [m[0] for m in sorted_models]
            preds_map = {"SVM": svm_preds, "RF": rf_preds, "CNN": cnn_preds}
            print(f"  Model priority order: {' > '.join(f'{m}({individual_f1[m]:.4f})' for m in model_order)}")

            df_3way = evaluate_ensemble(
                preds_map[model_order[0]],
                preds_map[model_order[1]],
                model_order[0],
                model_order[1],
                preds_dict_c=preds_map[model_order[2]],
                name_c=model_order[2],
            )
            df_3way.insert(0, "model", "SVM+RF+CNN_ensemble")

            out_path = out_dir / "ensemble_3way_per_subject.csv"
            df_3way.to_csv(out_path, index=False)
            print_ensemble_results(
                df_3way, "SVM+RF+CNN",
                {m: individual_f1[m] for m in model_order},
            )
            print(f"  [saved] {out_path}")
        else:
            print("  [SKIP] Not enough prediction files for 3-way ensemble")

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("SUMMARY - ALL ENSEMBLES")
    print("=" * 60)

    summary_rows = []
    for name, df in [
        ("SVM+RF", df_svmrf),
        ("CNN+SVM", df_cnnsvm),
        ("CNN+RF", df_cnnrf),
        ("SVM+RF+CNN", df_3way),
    ]:
        if df is not None:
            summary_rows.append(
                {
                    "ensemble": name,
                    "f1_mean": round(df["f1_macro"].mean(), 4),
                    "f1_sd": round(df["f1_macro"].std(ddof=1), 4),
                    "bal_acc_mean": round(df["bal_acc"].mean(), 4),
                    "bal_acc_sd": round(df["bal_acc"].std(ddof=1), 4),
                    "acc_mean": round(df["acc"].mean(), 4),
                    "n_subjects": len(df),
                }
            )

    # Add individual model baselines to summary
    for name, preds in [("SVM (indiv)", svm_preds), ("RF (indiv)", rf_preds), ("CNN (indiv)", cnn_preds)]:
        if preds:
            f1s = []
            bals = []
            accs = []
            for s in sorted(preds.keys()):
                yt, yp = preds[s]
                f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
                bals.append(balanced_accuracy_score(yt, yp))
                accs.append(accuracy_score(yt, yp))
            summary_rows.append(
                {
                    "ensemble": name,
                    "f1_mean": round(np.mean(f1s), 4),
                    "f1_sd": round(np.std(f1s, ddof=1), 4),
                    "bal_acc_mean": round(np.mean(bals), 4),
                    "bal_acc_sd": round(np.std(bals, ddof=1), 4),
                    "acc_mean": round(np.mean(accs), 4),
                    "n_subjects": len(preds),
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / "ensemble_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(summary_df.to_string(index=False))
        print(f"\n  [saved] {summary_path}")

    print("\n=== ENSEMBLE EVALUATION COMPLETE ===")


if __name__ == "__main__":
    main()

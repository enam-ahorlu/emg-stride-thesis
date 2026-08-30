# run_stdup_subsample_cnn.py
# ---------------------------------------------------------------------------
# G6: extend the STDUP class-balance sub-sampling control (run_stdup_subsample.py,
# SVM/RF) to the CNN and ResNet-SE architectures. Same protocol: downsample STDUP
# TRAINING windows (pooled across the 39 non-held-out subjects) to the mean count
# of the other three classes, test set untouched, per-subject normalization,
# identical LOSO loop. If STDUP F1 holds under balanced training for the deep
# models too, the "biomechanical, not sample-size" reading extends beyond the
# classical models.
#
# Example:
#   python run_stdup_subsample_cnn.py --npz windows_..._AorR.npz --meta features_out/freq_..._features_meta.csv \
#       --arch resnet_se --conditions imbalanced,balanced --out results_stdup_subsample --resume
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from train_cnn_loso import (
    WindowsDataset, per_subject_zscore_3d, choose_val_subjects, normalize_label_to_str, LABELS,
)
from run_cnn_arch_loso import train_fold, evaluate_with_proba
from cnn_architectures import build_model
from run_stdup_subsample import balance_training

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def main():
    ap = argparse.ArgumentParser("STDUP class-balance sub-sampling test under LOSO, CNN/ResNet-SE.")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--conditions", default="imbalanced,balanced")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results_stdup_subsample")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tag = "CNN" if args.arch == "simple" else "RESNET_SE"
    csv_path = out_dir / f"stdup_subsample_{tag}_subjectwise.csv"

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    X = per_subject_zscore_3d(X, subjects)  # headline normalisation
    in_ch = X.shape[1]
    stdup_id = LABEL_TO_IDX["STDUP"]
    subjects_u = sorted(np.unique(subjects).tolist())
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print(f"[arch] {args.arch} ({tag}): device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    done = set()
    if args.resume and csv_path.exists():
        d = pd.read_csv(csv_path); done = set(zip(d.subject, d.condition))

    for heldout in subjects_u:
        te = (subjects == heldout); tr = ~te
        Xtr_full, ytr_full, subtr_full = X[tr], y[tr], subjects[tr]
        rng = np.random.default_rng(args.seed * 100 + heldout)

        for cond in conditions:
            if (heldout, cond) in done:
                continue
            if cond == "balanced":
                idx = balance_training(ytr_full, stdup_id, rng)
                Xtr, ytr, subtr = Xtr_full[idx], ytr_full[idx], subtr_full[idx]
            else:
                Xtr, ytr, subtr = Xtr_full, ytr_full, subtr_full

            n_stdup_train = int((ytr == stdup_id).sum())
            tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
            m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)

            torch.manual_seed(args.seed)
            model = build_model(args.arch, in_ch, len(LABELS)).to(device)
            model = train_fold(model, Xtr[m_tr], ytr[m_tr], Xtr[m_va], ytr[m_va],
                               device, args.epochs, args.batch, args.lr, args.patience, args.seed)

            te_dl = DataLoader(WindowsDataset(X[te], y[te]), batch_size=512, shuffle=False)
            yt, yp, _ = evaluate_with_proba(model, te_dl, device)
            per_cls = f1_score(yt, yp, labels=list(range(len(LABELS))), average=None, zero_division=0)
            row = {"subject": int(heldout), "model": tag, "condition": cond,
                   "n_stdup_train": n_stdup_train,
                   "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}
            for li, lab in enumerate(LABELS):
                row[f"f1_{lab}"] = float(per_cls[li])
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            done.add((heldout, cond))
            print(f"[fold] Sub{heldout:02d} {tag}/{cond}: macro={row['macro_f1']:.3f} "
                  f"STDUP={row['f1_STDUP']:.3f} (n_stdup_train={n_stdup_train})", flush=True)

    d = pd.read_csv(csv_path).drop_duplicates(["subject", "condition"])
    agg = d.groupby(["model", "condition"])[[c for c in d.columns if c.startswith("f1_") or c == "macro_f1"]].mean().round(4)
    agg.to_csv(out_dir / f"stdup_subsample_{tag}_summary.csv")
    print(f"\n=== MEAN per-class F1 by condition ({tag}) ===")
    print(agg.to_string())


if __name__ == "__main__":
    main()

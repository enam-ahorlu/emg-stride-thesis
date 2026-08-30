# run_cnn_arch_subjectdep.py
# ---------------------------------------------------------------------------
# Subject-dependent (SD) 5-fold CV for a chosen CNN architecture, looped over
# all subjects, with subjectwise + summary CSV output. Mirrors run_cnn_arch_loso.py's
# conventions (--arch, --resume, cnn_arch_subjectwise.csv) so ResNet-SE's SD
# result is directly comparable to Table 4.1 / Table 4.5 (generalization gap).
#
# Per-subject 5-fold StratifiedKFold CV, 15% of each training fold held out for
# early stopping (patience=5), matching train_cnn_subjectdep.py's SimpleEMGCNN
# protocol exactly except architecture is selectable via cnn_architectures.build_model.
#
# Example:
#   python run_cnn_arch_subjectdep.py --npz windows_..._AorR.npz \
#       --meta features_out/freq_..._features_meta.csv --arch resnet_se \
#       --epochs 20 --out results_sd_resnet_se --resume
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from cnn_architectures import build_model, count_params
from train_cnn_loso import normalize_label_to_str, LABELS

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


class EMGWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def zscore_fit(Xtr):
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-8
    return mu, sd


def zscore_apply(X, mu, sd):
    return (X - mu) / sd


def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward(); opt.step()
        total += loss.item() * len(X)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, yh = [], []
    for X, y in loader:
        X = X.to(device)
        pred = model(X).argmax(1).cpu().numpy()
        ys.append(y.numpy()); yh.append(pred)
    y_true = np.concatenate(ys); y_pred = np.concatenate(yh)
    return (accuracy_score(y_true, y_pred), balanced_accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


def run_subject_sd(subject, X, y, arch, epochs, batch, seed, device, n_folds=5, patience=5,
                    norm="zscore", dropout=0.25, weight_decay=0.0):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_accs, fold_bals, fold_f1s = [], [], []
    for tr_idx, te_idx in skf.split(X, y):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        val_size = max(1, int(0.15 * len(Xtr)))
        Xtr_inner, Xval = Xtr[:-val_size], Xtr[-val_size:]
        ytr_inner, yval = ytr[:-val_size], ytr[-val_size:]

        if norm == "zscore":
            mu, sd = zscore_fit(Xtr_inner)
            Xtr_inner = zscore_apply(Xtr_inner, mu, sd)
            Xval = zscore_apply(Xval, mu, sd)
            Xte_n = zscore_apply(Xte, mu, sd)
        else:
            Xte_n = Xte

        train_loader = DataLoader(EMGWindowDataset(Xtr_inner, ytr_inner), batch_size=batch, shuffle=True)
        val_loader = DataLoader(EMGWindowDataset(Xval, yval), batch_size=batch)
        test_loader = DataLoader(EMGWindowDataset(Xte_n, yte), batch_size=batch)

        torch.manual_seed(seed)
        model = build_model(arch, X.shape[1], len(LABELS)).to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)

        best_val, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            train_one_epoch(model, train_loader, loss_fn, opt, device)
            model.eval()
            vloss, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    vloss += loss_fn(model(xb), yb).item() * len(yb); vn += len(yb)
            vloss /= max(vn, 1)
            if vloss < best_val:
                best_val, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state); model.to(device)

        acc, bal, f1 = eval_model(model, test_loader, device)
        fold_accs.append(acc); fold_bals.append(bal); fold_f1s.append(f1)

    return float(np.mean(fold_accs)), float(np.mean(fold_bals)), float(np.mean(fold_f1s)), float(np.std(fold_f1s, ddof=1))


def main():
    ap = argparse.ArgumentParser("Subject-dependent 5-fold CV for a chosen CNN architecture, all subjects.")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_raw", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--norm", default="zscore", choices=["none", "zscore"])
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="Adam weight decay; small-data recipe check uses e.g. 1e-3.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--out", default="results_sd_resnet_se")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cnn_arch_subjectdep_subjectwise.csv"

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X_all = data[args.xkey].astype(np.float32)
    y_all = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    in_ch = X_all.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())

    print(f"[arch] {args.arch}: {count_params(build_model(args.arch, in_ch, len(LABELS))):,} params | device={device}", flush=True)
    done = set()
    if args.resume and csv_path.exists():
        done = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())

    for subject in subjects_u:
        if subject in done:
            continue
        mask = subjects == subject
        X, y = X_all[mask], y_all[mask]
        t0 = time.perf_counter()
        acc, bal, f1, f1_sd = run_subject_sd(subject, X, y, args.arch, args.epochs, args.batch,
                                              args.seed, device, norm=args.norm,
                                              weight_decay=args.weight_decay)
        dt = time.perf_counter() - t0
        row = {"subject": int(subject), "arch": args.arch, "n_windows": int(mask.sum()),
               "acc": acc, "bal_acc": bal, "f1_macro": f1, "f1_macro_fold_sd": f1_sd, "fit_time_sec": dt}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(subject)
        print(f"[subject] Sub{subject:02d} {args.arch} SD f1={f1:.4f} (5-fold, {dt:.1f}s)", flush=True)

    df = pd.read_csv(csv_path).drop_duplicates("subject")
    m, s = df["f1_macro"].mean(), df["f1_macro"].std(ddof=1)
    pd.DataFrame([{"arch": args.arch, "f1_macro_mean": round(m, 4), "f1_macro_sd": round(s, 4),
                   "bal_acc_mean": round(df["bal_acc"].mean(), 4), "acc_mean": round(df["acc"].mean(), 4),
                   "n": len(df)}]).to_csv(out_dir / "cnn_arch_subjectdep_summary.csv", index=False)
    print(f"\n[{args.arch}] SD F1 = {m:.4f} +/- {s:.4f} (n={len(df)}) | SimpleEMGCNN SD headline = 0.904")


if __name__ == "__main__":
    main()

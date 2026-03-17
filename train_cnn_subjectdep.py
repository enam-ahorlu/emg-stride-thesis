# train_cnn_subjectdep.py
# SimpleEMGCNN for per-subject SD evaluation (same architecture as LOSO for
# apples-to-apples generalization gap comparison).
# Adds: correct indexing, stratified split, optional input choice (raw/env),
#       optional normalization (none/zscore), class distribution + baselines,
#       optional overfit test (PATCHED properly), and clearer reporting.

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# =============================
# Dataset
# =============================

class EMGWindowDataset(Dataset):
    def __init__(self, X, y):
        # X: (N,C,T)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================
# SimpleEMGCNN (same architecture as train_cnn_loso.py for consistency)
# =============================

class SimpleEMGCNN(nn.Module):
    """
    Compact 1D CNN (channels=EMG channels, conv over time).
    Identical architecture to train_cnn_loso.py so SD vs LOSO
    comparisons are apples-to-apples.
    """
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # -> (N, 128, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),            # -> (N, 128)
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x


# =============================
# Load NPZ + Meta
# =============================

def load_npz(npz_path: Path):
    npz = np.load(npz_path, allow_pickle=False)
    keys = set(npz.files)
    return npz, keys


def pick_X(npz, keys: set[str], use: str):
    # use: raw | env
    if use == "raw":
        if "X_raw" in keys:
            return npz["X_raw"]
        if "X" in keys:
            return npz["X"]
        raise KeyError("Requested --use raw but NPZ has no X_raw or X.")
    if use == "env":
        if "X_env" in keys:
            return npz["X_env"]
        raise KeyError("Requested --use env but NPZ has no X_env.")
    raise ValueError(f"Unknown --use {use}. Choose raw or env.")


def encode_labels(y_str):
    uniq = sorted(pd.unique(y_str))
    m = {v: i for i, v in enumerate(uniq)}
    y = np.array([m[v] for v in y_str], dtype=int)
    return y, m


# =============================
# Normalization (train stats only)
# =============================

def zscore_fit(Xtr: np.ndarray, eps: float = 1e-8):
    # Xtr: (N,C,T)
    # per-channel mean/std computed across N and T
    mu = Xtr.mean(axis=(0, 2), keepdims=True)           # (1,C,1)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + eps      # (1,C,1)
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


# =============================
# Train / Eval
# =============================

def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(X)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys, yh = [], []
    for X, y in loader:
        X = X.to(device)
        logits = model(X)
        pred = logits.argmax(1).cpu().numpy()
        ys.append(y.numpy())
        yh.append(pred)

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yh)

    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return acc, bal, f1, y_true, y_pred


def _latency_ms_per_window(model, Xte_np: np.ndarray, device: str, n_sample: int = 100) -> float:
    if Xte_np.shape[0] == 0:
        return float("nan")

    n = min(n_sample, Xte_np.shape[0])
    Xsample = torch.tensor(Xte_np[:n], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        _ = model(Xsample)  # warmup
        if device == "cuda":
            torch.cuda.synchronize()

        t2 = time.perf_counter()
        _ = model(Xsample)
        if device == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

    return (t3 - t2) / n * 1000.0


def majority_baseline(y: np.ndarray):
    # predict most frequent class
    vals, counts = np.unique(y, return_counts=True)
    maj = vals[np.argmax(counts)]
    yhat = np.full_like(y, maj)
    return {
        "majority_class": int(maj),
        "acc": accuracy_score(y, yhat),
        "bal": balanced_accuracy_score(y, yhat),
        "f1": f1_score(y, yhat, average="macro", zero_division=0),
        "counts": dict(zip([int(v) for v in vals], [int(c) for c in counts])),
    }


# =============================
# Overfit helper (PATCHED)
# =============================

def make_stratified_subset(Xtr: np.ndarray, ytr: np.ndarray, n: int, seed: int):
    """
    Create a stratified random subset of size n from (Xtr, ytr), preserving class proportions as much as possible.
    This is a better sanity-check than taking Xtr[:n], which can accidentally under-sample some classes.
    """
    n = int(min(n, len(Xtr)))
    if n <= 0:
        return Xtr, ytr

    # If n is too small to include every class at least once, we still sample as best as possible.
    # Approach: sample within each class proportional to its frequency (at least 1 if possible).
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(ytr, return_counts=True)
    total = counts.sum()

    # desired per-class counts
    raw_alloc = counts / total * n
    alloc = np.floor(raw_alloc).astype(int)

    # Ensure at least 1 per class where possible
    remaining = n - alloc.sum()
    # First, bump any class with alloc==0 up to 1 if we still have room and the class exists
    for i in range(len(classes)):
        if remaining <= 0:
            break
        if alloc[i] == 0 and counts[i] > 0:
            alloc[i] = 1
            remaining -= 1

    # Distribute remaining by largest fractional parts
    if remaining > 0:
        frac = raw_alloc - np.floor(raw_alloc)
        order = np.argsort(-frac)
        for j in order:
            if remaining <= 0:
                break
            # don't exceed available samples in that class
            if alloc[j] < counts[j]:
                alloc[j] += 1
                remaining -= 1

    # Clip to available counts (safety), then fix total if needed
    alloc = np.minimum(alloc, counts)
    # If we lost some due to clipping, top up from classes with available capacity
    deficit = n - alloc.sum()
    if deficit > 0:
        # classes with remaining capacity
        cap = counts - alloc
        order = np.argsort(-cap)
        for j in order:
            if deficit <= 0:
                break
            if cap[j] > 0:
                add = min(int(cap[j]), int(deficit))
                alloc[j] += add
                deficit -= add

    # Now sample indices per class
    chosen = []
    for c, k in zip(classes, alloc):
        idxs = np.flatnonzero(ytr == c)
        if k > 0:
            chosen.extend(rng.choice(idxs, size=int(k), replace=False).tolist())

    chosen = np.array(chosen, dtype=int)
    rng.shuffle(chosen)
    return Xtr[chosen], ytr[chosen]


def accuracy_on_loader(model, loader, device, mode: str = "eval") -> float:
    """
    mode:
      - "train": model.train() during forward (BN uses batch stats)
      - "eval":  model.eval() during forward (BN uses running stats)
    """
    if mode == "train":
        model.train()
    else:
        model.eval()

    ys, yh = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            pred = logits.argmax(1).cpu().numpy()
            ys.append(y.numpy())
            yh.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(yh)
    return accuracy_score(y_true, y_pred)



def freeze_batchnorm(model: nn.Module):
    """
    Freeze BatchNorm running stats and parameters.
    Useful for overfit sanity checks on tiny datasets.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()  # stops updating running mean/var
            for p in m.parameters():
                p.requires_grad = False



# =============================
# Main
# =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use", choices=["raw", "env"], default="raw", help="Which signal representation to use from NPZ.")
    ap.add_argument("--norm", choices=["none", "zscore"], default="none", help="Input normalization (train stats only).")
    ap.add_argument("--report", action="store_true", help="Print confusion matrix + per-class report at end.")
    ap.add_argument("--overfit-n", type=int, default=0, help="If >0, run a proper overfit sanity check on N TRAIN windows.")
    ap.add_argument("--overfit-eval", choices=["train", "test"], default="train",
                    help="In overfit mode, report metrics on training subset (default) or test set.")
    ap.add_argument("--dropout", type=float, default=0.25, help="Dropout probability in classifier head (0.25 matches LOSO).")
    ap.add_argument("--overfit-disable-regularization", action="store_true",
                    help="Overfit sanity check: set dropout=0 and freeze BatchNorm stats/params.")


    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    npz_path = Path(args.npz)
    meta_path = Path(args.meta)

    npz, keys = load_npz(npz_path)
    meta = pd.read_csv(meta_path)

    X_all = pick_X(npz, keys, args.use)

    # Correct subject indexing (original indices)
    sub_mask = (meta["subject"] == args.subject)
    idx = meta.index[sub_mask].to_numpy()
    meta_sub = meta.loc[sub_mask].reset_index(drop=True)

    X = X_all[idx]
    if len(X) != len(meta_sub):
        raise RuntimeError(f"Alignment error: X has {len(X)} rows but meta_sub has {len(meta_sub)} rows.")

    y_str = meta_sub["movement"].astype(str).values
    y, label_map = encode_labels(y_str)

    print(f"Subject {args.subject}")
    print("Windows:", len(X))
    print("Classes:", label_map)
    print(f"Using: {args.use} | Norm: {args.norm}")

    # Baseline diagnostics (on full subject data)
    base = majority_baseline(y)
    print("\n=== Class counts (label id -> count) ===")
    print(base["counts"])
    print("=== Majority baseline (predict most frequent) ===")
    print(f"acc={base['acc']:.4f} bal={base['bal']:.4f} f1={base['f1']:.4f} (majority_class={base['majority_class']})")

    # Stratified split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y
    )

    # -----------------------------
    # Proper overfit test (PATCHED)
    # -----------------------------
    # Old behavior took Xtr[:n], which may accidentally under-sample some classes.
    # New behavior samples a STRATIFIED RANDOM subset of size n and (by default) evaluates on that TRAIN subset.
    overfit_mode = bool(args.overfit_n and args.overfit_n > 0)
    if overfit_mode:
        n = min(int(args.overfit_n), len(Xtr))
        Xtr, ytr = make_stratified_subset(Xtr, ytr, n=n, seed=args.seed)
        print(f"\n[OVERFIT TEST] Training on stratified random subset of {len(Xtr)} TRAIN windows.")
        print("[OVERFIT TEST] Default evaluation is on TRAIN subset (use --overfit-eval test to still check generalization).")


    # Overfit sanity settings
    dropout_p = args.dropout
    freeze_bn = False

    if overfit_mode and args.overfit_disable_regularization:
        dropout_p = 0.0
        freeze_bn = True
        print("[OVERFIT TEST] Regularization disabled: dropout=0 and BatchNorm frozen.")


    # Normalization (train stats only)
    if args.norm == "zscore":
        mu, sd = zscore_fit(Xtr)
        Xtr = zscore_apply(Xtr, mu, sd)
        Xte = zscore_apply(Xte, mu, sd)

    train_ds = EMGWindowDataset(Xtr, ytr)
    test_ds = EMGWindowDataset(Xte, yte)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = SimpleEMGCNN(in_ch=X.shape[1], n_classes=len(label_map), dropout=dropout_p).to(device)

    if freeze_bn:
        freeze_batchnorm(model)


    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        loss = train_one_epoch(model, train_loader, loss_fn, opt, device)

        if overfit_mode:
            # accuracy on the TRAIN subset, computed two ways:
            train_acc_eval = accuracy_on_loader(model, train_loader, device, mode="eval")
            train_acc_trainmode = accuracy_on_loader(model, train_loader, device, mode="train")

            # what you want to treat as "eval" during overfit debugging:
            if args.overfit_eval == "train":
                acc, bal, f1, _, _ = eval_model(model, train_loader, device)
            else:
                acc, bal, f1, _, _ = eval_model(model, test_loader, device)

            print(
                f"Epoch {ep+1:02d} | loss={loss:.4f} "
                f"train_acc(eval)={train_acc_eval:.4f} train_acc(trainmode)={train_acc_trainmode:.4f} "
                f"eval_acc={acc:.4f} bal={bal:.4f} f1={f1:.4f}"
            )

        else:
            acc, bal, f1, _, _ = eval_model(model, test_loader, device)
            print(f"Epoch {ep+1:02d} | loss={loss:.4f} acc={acc:.4f} bal={bal:.4f} f1={f1:.4f}")



    train_time = time.perf_counter() - t0

    # Final eval + latency
    if overfit_mode and args.overfit_eval == "train":
        acc, bal, f1, y_true, y_pred = eval_model(model, train_loader, device)
        # latency still measured on test set windows (or training subset if no test)
        latency = _latency_ms_per_window(model, Xte, device=device, n_sample=100) if len(Xte) else float("nan")
    else:
        acc, bal, f1, y_true, y_pred = eval_model(model, test_loader, device)
        latency = _latency_ms_per_window(model, Xte, device=device, n_sample=100)

    print("\n=== CNN RESULT ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Acc: {bal:.4f}")
    print(f"F1 macro: {f1:.4f}")
    print(f"Train time (s): {train_time:.2f}")
    print(f"Inference per window (ms): {latency:.4f}")

    if args.report:
        # Label names in id order
        inv = {v: k for k, v in label_map.items()}
        names = [inv[i] for i in range(len(inv))]

        print("\n=== Confusion matrix (rows=true, cols=pred) ===")
        print(confusion_matrix(y_true, y_pred, labels=list(range(len(names)))))

        print("\n=== Per-class report ===")
        print(classification_report(y_true, y_pred, target_names=names, zero_division=0))


if __name__ == "__main__":
    main()

# train_cnn_loso.py
import argparse
import os
from pathlib import Path
import json
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


LABELS = ["DNS", "STDUP", "UPS", "WAK"]  # movement types (alphabetical = encode order)


def augment_batch(Xb: torch.Tensor, mode: str, sigma: float, chandrop_p: float, mask_frac: float) -> torch.Tensor:
    """
    Apply data augmentation to a training batch in-place (returns augmented tensor).

    Parameters
    ----------
    Xb          : (N, C, T) float tensor on device
    mode        : one of 'none', 'gaussian', 'chandrop', 'timemask', 'combined'
    sigma       : std of Gaussian noise (relative to data scale)
    chandrop_p  : per-channel drop probability for chandrop
    mask_frac   : fraction of T to zero out for timemask
    """
    if mode == "none":
        return Xb

    N, C, T = Xb.shape

    if mode in ("gaussian", "combined"):
        noise = torch.randn_like(Xb) * sigma
        Xb = Xb + noise

    if mode in ("chandrop", "combined"):
        # For each sample independently, zero out channels where rand < chandrop_p
        # drop_mask shape: (N, C, 1) — broadcast over T
        drop_mask = (torch.rand(N, C, 1, device=Xb.device) >= chandrop_p).float()
        Xb = Xb * drop_mask

    if mode in ("timemask", "combined"):
        mask_len = max(1, int(mask_frac * T))
        for i in range(N):
            start = torch.randint(0, max(1, T - mask_len + 1), (1,)).item()
            Xb[i, :, start:start + mask_len] = 0.0

    return Xb


def normalize_label_to_str(x) -> str:
    """Coerce label values to clean strings (handles numeric '1.0' → '1')."""
    s = str(x).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


class WindowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, C, T), y: (N,)
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleEMGCNN(nn.Module):
    """
    Compact 1D CNN (channels=EMG channels, conv over time).
    Designed to be fast + stable for LOSO loops.
    """
    def __init__(self, in_ch: int, n_classes: int):
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
            nn.Dropout(0.25),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, yhat = [], []
    total_loss = 0.0
    n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(Xb)
        loss = nn.functional.cross_entropy(logits, yb)
        total_loss += float(loss.item()) * int(yb.shape[0])
        n += int(yb.shape[0])

        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(yb.cpu().numpy())
        yhat.append(pred)

    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(yhat) if yhat else np.array([], dtype=int)
    avg_loss = total_loss / max(n, 1)
    return avg_loss, y_true, y_pred


def plot_confusion(cm: np.ndarray, out_png: Path, labels=LABELS, normalize=False, title="Confusion Matrix"):
    cm_plot = cm.astype(float)
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, row_sum, out=np.zeros_like(cm_plot), where=row_sum != 0)

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            txt = f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm_plot[i, j])}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def compute_train_norm(X_train: np.ndarray):
    """
    Per-channel normalization computed on train only.
    X_train shape: (N, C, T)
    Returns mean(C,1) and std(C,1) to broadcast over (N,C,T).
    """
    # mean/std across N and T for each channel
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def per_subject_zscore_3d(X: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """
    Pre-normalize each subject's windows independently (leak-free).
    X shape: (N, C, T). Mean/std computed over each subject's own windows
    and time steps per channel (axes 0 and 2).
    """
    X_norm = X.copy()
    for sid in np.unique(subjects):
        mask = (subjects == sid)
        Xs = X[mask]                                  # (N_s, C, T)
        mu = Xs.mean(axis=(0, 2), keepdims=True)      # (1, C, 1)
        sd = Xs.std(axis=(0, 2), keepdims=True)       # (1, C, 1)
        sd = np.where(sd < 1e-8, 1.0, sd)
        X_norm[mask] = (Xs - mu) / sd
    return X_norm


def apply_norm_robust(X_tr: np.ndarray, X_va: np.ndarray,
                      X_te: np.ndarray):
    """
    RobustScaler (median/IQR) for 3D windows.
    Reshapes (N,C,T) -> (N, C*T), fits on train only,
    transforms val/test, then reshapes back to (N,C,T).
    """
    N_tr, C, T = X_tr.shape
    scaler = RobustScaler()
    X_tr_2d = scaler.fit_transform(X_tr.reshape(N_tr, C * T))
    X_va_2d = scaler.transform(X_va.reshape(X_va.shape[0], C * T))
    X_te_2d = scaler.transform(X_te.reshape(X_te.shape[0], C * T))
    # guard against zero-IQR channels producing NaN
    for arr in [X_tr_2d, X_va_2d, X_te_2d]:
        np.nan_to_num(arr, copy=False, nan=0.0)
    return (X_tr_2d.reshape(N_tr, C, T),
            X_va_2d.reshape(-1, C, T),
            X_te_2d.reshape(-1, C, T))


def choose_val_subjects(train_subjects: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(set(train_subjects.tolist())))
    n_val = max(1, int(round(val_frac * len(subs))))
    val_subs = rng.choice(subs, size=n_val, replace=False)
    train_subs = np.array([s for s in subs if s not in set(val_subs.tolist())])
    return train_subs, val_subs


def class_weights_from_y(y: np.ndarray, n_classes: int):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    # inverse-frequency weights (safe for imbalance)
    w = counts.sum() / np.maximum(counts, 1.0)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser("Leakage-proof CNN LOSO on EMG windows (NPZ + meta)")
    ap.add_argument("--npz", required=True, help="windows_*.npz containing X_env or X_raw")
    ap.add_argument("--meta", required=True, help="meta CSV aligned 1:1 with NPZ windows (must have subject + movement columns)")
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"], help="NPZ key to use")
    ap.add_argument("--label-col", default="movement", help="Target label column in meta")
    ap.add_argument("--out", default="results_cnn_loso", help="Output folder")
    ap.add_argument("--epochs", type=int, default=25, help="Max epochs per fold")
    ap.add_argument("--batch", type=int, default=512, help="Batch size (512 fits 6GB GPU)")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Fraction of training subjects used for validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="Resume: skip heldout subjects already present in metrics CSV")
    ap.add_argument("--heldout", type=int, default=None, help="Run only one heldout subject (e.g., 1..40)")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (half of CPU cores)")
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--amp", action="store_true", help="Use mixed precision (GPU only)")
    ap.add_argument("--norm-mode", default="global",
                    choices=["none", "global", "per_subject", "robust"],
                    help="global=per-channel z-score on train fold (current default); "
                         "none=skip all normalization; "
                         "per_subject=z-score each subject by own stats before LOSO loop; "
                         "robust=RobustScaler (median/IQR) fit on train fold.")
    ap.add_argument("--augment", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"],
                    help="Data augmentation applied to training batches only. "
                         "none=no augmentation, gaussian=additive Gaussian noise, "
                         "chandrop=random channel dropout, timemask=contiguous time masking, "
                         "combined=all three augmentations applied sequentially.")
    ap.add_argument("--aug-sigma", type=float, default=0.1,
                    help="Gaussian noise std relative to normalized data scale (default: 0.1)")
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2,
                    help="Per-channel drop probability for chandrop augmentation (default: 0.2)")
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15,
                    help="Fraction of T to zero out for timemask augmentation (default: 0.15)")
    args = ap.parse_args()

    # Reproducibility seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    outdir = Path(args.out)
    pred_dir = outdir / "predictions"
    cm_dir = outdir / "confusion_matrices"
    outdir.mkdir(exist_ok=True)
    pred_dir.mkdir(exist_ok=True)
    cm_dir.mkdir(exist_ok=True)

    stem = Path(args.npz).stem

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    # Load data
    meta = pd.read_csv(args.meta)
    if "subject" not in meta.columns:
        raise ValueError("meta must contain 'subject' column")
    if args.label_col not in meta.columns:
        raise ValueError(f"meta must contain label column '{args.label_col}'")

    data = np.load(args.npz)
    if args.xkey not in data.files:
        raise ValueError(f"NPZ missing key {args.xkey}. Found keys: {data.files}")
    X = data[args.xkey]  # (N,C,T)
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T). Got {X.shape}")

    y_str = meta[args.label_col].map(normalize_label_to_str).values
    # map to indices 0..6
    label_to_idx = {lab: i for i, lab in enumerate(LABELS)}
    if not set(np.unique(y_str)).issubset(set(LABELS)):
        bad = sorted(list(set(np.unique(y_str)) - set(LABELS)))
        raise ValueError(f"Found label values not in {LABELS}: {bad}")
    y = np.array([label_to_idx[s] for s in y_str], dtype=np.int64)

    subjects = meta["subject"].astype(int).values
    unique_subjects = sorted(set(subjects.tolist()))
    print(f"[data] X={X.shape} y={y.shape} subjects={len(unique_subjects)} (min={min(unique_subjects)}, max={max(unique_subjects)})")

    # Normalization mode
    norm_mode = args.norm_mode
    print(f"[norm] norm_mode = {norm_mode}")

    # Augmentation mode
    aug_mode = args.augment
    print(f"[aug] aug_mode = {aug_mode}, sigma={args.aug_sigma}, "
          f"chandrop_p={args.aug_chandrop_p}, timemask_frac={args.aug_timemask_frac}")

    # Per-subject pre-normalization: applied once before the LOSO loop.
    # Leak-free because each subject is normalized using only their own windows.
    if norm_mode == "per_subject":
        print("[norm] Applying per-subject z-score (3D) before LOSO loop...")
        X = per_subject_zscore_3d(X, subjects)

    # Resume support
    metrics_csv = outdir / "per_subject_metrics_cnn_loso.csv"
    done_subjects = set()
    if args.resume and metrics_csv.exists():
        prev = pd.read_csv(metrics_csv)
        if "subject" in prev.columns:
            done_subjects = set(prev["subject"].astype(int).tolist())
            print(f"[resume] found {len(done_subjects)} completed subjects")

    heldout_list = unique_subjects
    if args.heldout is not None:
        heldout_list = [int(args.heldout)]

    all_y_true = []
    all_y_pred = []
    rows = []

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    for heldout in heldout_list:
        if args.resume and heldout in done_subjects:
            print(f"[resume] skip heldout Sub{heldout:02d}")
            continue

        print(f"\n[fold] heldout Sub{heldout:02d}")

        test_mask = (subjects == heldout)
        train_mask = (subjects != heldout)

        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        sub_train_full = subjects[train_mask]

        # choose val subjects from training subjects only
        train_subs, val_subs = choose_val_subjects(sub_train_full, val_frac=args.val_frac, seed=args.seed + heldout)
        tr_mask2 = np.isin(sub_train_full, train_subs)
        va_mask2 = np.isin(sub_train_full, val_subs)

        X_tr = X_train_full[tr_mask2]
        y_tr = y_train_full[tr_mask2]
        X_va = X_train_full[va_mask2]
        y_va = y_train_full[va_mask2]

        X_te = X[test_mask]
        y_te = y[test_mask]

        # Per-fold normalization (conditioned on norm_mode)
        if norm_mode == "global":
            mean, std = compute_train_norm(X_tr)
            X_tr = apply_norm(X_tr, mean, std)
            X_va = apply_norm(X_va, mean, std)
            X_te = apply_norm(X_te, mean, std)
        elif norm_mode == "robust":
            X_tr, X_va, X_te = apply_norm_robust(X_tr, X_va, X_te)
        # none: skip all normalization
        # per_subject: data already normalized before loop — no per-fold step

        # datasets/loaders
        ds_tr = WindowsDataset(X_tr, y_tr)
        ds_va = WindowsDataset(X_va, y_va)
        ds_te = WindowsDataset(X_te, y_te)

        pin_mem = args.pin_memory or (device.type == "cuda")
        loader_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin_mem)
        loader_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, pin_memory=pin_mem)
        loader_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False,
                               num_workers=args.num_workers, pin_memory=pin_mem)

        # model + optim
        model = SimpleEMGCNN(in_ch=X.shape[1], n_classes=len(LABELS)).to(device)

        w = class_weights_from_y(y_tr, len(LABELS)).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)

        optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val = float("inf")
        best_state = None
        patience = 5
        bad_epochs = 0

        for ep in range(1, args.epochs + 1):
            model.train()
            running = 0.0
            n_seen = 0

            for Xb, yb in loader_tr:
                Xb = Xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                # Apply data augmentation to training batch only (not val/test)
                if aug_mode != "none":
                    Xb = augment_batch(
                        Xb,
                        mode=aug_mode,
                        sigma=args.aug_sigma,
                        chandrop_p=args.aug_chandrop_p,
                        mask_frac=args.aug_timemask_frac,
                    )

                optim.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
                    logits = model(Xb)
                    loss = criterion(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                running += float(loss.item()) * int(yb.shape[0])
                n_seen += int(yb.shape[0])

            tr_loss = running / max(n_seen, 1)
            va_loss, _, _ = evaluate(model, loader_va, device)

            print(f"[ep {ep:02d}] train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

            if va_loss + 1e-6 < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print(f"[earlystop] no improvement for {patience} epochs")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # test eval
        te_loss, y_true, y_pred = evaluate(model, loader_te, device)

        f1 = f1_score(y_true, y_pred, average="macro")
        balacc = balanced_accuracy_score(y_true, y_pred)

        print(f"[test] Sub{heldout:02d} loss={te_loss:.4f} f1_macro={f1:.4f} bal_acc={balacc:.4f} n={len(y_true)}")

        rows.append({
            "model": "CNN",
            "subject": heldout,
            "f1_macro": f1,
            "bal_acc": balacc,
            "n_windows": int(len(y_true)),
            "xkey": args.xkey,
            "norm_mode": norm_mode,
            "aug_mode": aug_mode,
        })

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        # save per-subject preds (optional but useful)
        np.save(pred_dir / f"{stem}_CNN_loso_Sub{heldout:02d}_y_true.npy", y_true)
        np.save(pred_dir / f"{stem}_CNN_loso_Sub{heldout:02d}_y_pred.npy", y_pred)

        # append metrics as we go (so resume works even if crash)
        df_rows = pd.DataFrame(rows)
        if metrics_csv.exists() and args.resume:
            # merge with existing (avoid duplicates)
            prev = pd.read_csv(metrics_csv)
            combined = pd.concat([prev, df_rows], ignore_index=True)
            combined = combined.drop_duplicates(subset=["model", "subject", "xkey", "norm_mode", "aug_mode"], keep="last")
            combined.to_csv(metrics_csv, index=False)
        else:
            df_rows.to_csv(metrics_csv, index=False)

        # free memory between folds
        del model, optim, ds_tr, ds_va, ds_te, loader_tr, loader_va, loader_te
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Global outputs (if we ran at least one fold)
    if all_y_true:
        y_true_all = np.concatenate(all_y_true)
        y_pred_all = np.concatenate(all_y_pred)

        np.save(pred_dir / f"{stem}_CNN_loso_y_true.npy", y_true_all)
        np.save(pred_dir / f"{stem}_CNN_loso_y_pred.npy", y_pred_all)

        cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(LABELS))))
        cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
        cm_csv = cm_dir / f"{stem}__CNN_LOSO_confusion.csv"
        cm_df.to_csv(cm_csv)

        plot_confusion(cm, cm_dir / f"{stem}__CNN_LOSO_confusion.png",
                       labels=LABELS, normalize=False, title="CNN LOSO Confusion (counts)")
        plot_confusion(cm, cm_dir / f"{stem}__CNN_LOSO_confusion_norm.png",
                       labels=LABELS, normalize=True, title="CNN LOSO Confusion (row-normalized)")

        # summary
        df = pd.read_csv(metrics_csv)
        summary = df.groupby(["model", "xkey", "norm_mode"], as_index=False).agg(
            mean_f1=("f1_macro", "mean"),
            std_f1=("f1_macro", "std"),
            mean_balacc=("bal_acc", "mean"),
            std_balacc=("bal_acc", "std"),
            subjects=("subject", "count")
        )
        summary.to_csv(outdir / "cnn_loso_summary.csv", index=False)
        print(f"[save] {metrics_csv}")
        print(f"[save] {cm_csv}")
        print(f"[save] {outdir / 'cnn_loso_summary.csv'}")


if __name__ == "__main__":
    main()
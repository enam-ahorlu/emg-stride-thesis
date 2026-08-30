# run_cnn_arch_loso.py
# ---------------------------------------------------------------------------
# Fair-deep-baseline experiment: run a stronger CNN architecture under the
# IDENTICAL LOSO + per-subject-normalisation protocol as the headline
# SimpleEMGCNN, so the "classical vs deep under LOSO" comparison is not confounded
# by architectural under-investment. Only the architecture changes.
#
#   --arch simple     : the original SimpleEMGCNN (reproduces 0.754; sanity check)
#   --arch resnet_se  : compact 1D ResNet + squeeze-excite attention (fairer deep baseline)
#   --arch resnet     : same without SE (ablation on the attention)
#
# Reuses every data/normalisation/training convention from train_cnn_loso.py
# (per_subject_zscore_3d, choose_val_subjects, class weights, early stopping),
# so results are directly comparable to Table 4.2 / Section 4.2.2.
#
# Example (full run, GPU):
#   python run_cnn_arch_loso.py \
#       --npz  windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
#       --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --arch resnet_se --epochs 40 --out results_cnn_loso_resnet_se --resume
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_cnn_loso import (
    WindowsDataset, per_subject_zscore_3d, choose_val_subjects,
    class_weights_from_y, evaluate, normalize_label_to_str, LABELS,
    compute_train_norm, apply_norm, augment_batch,
)
from cnn_architectures import build_model, count_params

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def evaluate_with_proba(model, loader, device):
    """Like train_cnn_loso.evaluate() but also returns per-window softmax
    probabilities, in the model's output-index order (== LABELS order, since
    y was encoded via LABEL_TO_IDX over LABELS)."""
    model.eval()
    ys, yhat, probs = [], [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device, non_blocking=True)
            logits = model(Xb)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            pred = p.argmax(1)
            ys.append(yb.numpy())
            yhat.append(pred)
            probs.append(p)
    y_true = np.concatenate(ys) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(yhat) if yhat else np.array([], dtype=int)
    proba = np.concatenate(probs) if probs else np.zeros((0, len(LABELS)))
    return y_true, y_pred, proba


def train_fold(model, X_tr, y_tr, X_va, y_va, device, epochs, batch, lr, patience, seed,
                aug_mode="none", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15):
    torch.manual_seed(seed)
    w = class_weights_from_y(y_tr, len(LABELS)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr = DataLoader(WindowsDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    va = DataLoader(WindowsDataset(X_va, y_va), batch_size=batch, shuffle=False)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(device), yb.to(device)
            if aug_mode != "none":
                # training batches only (identical to train_cnn_loso.py's SimpleEMGCNN augmentation)
                Xb = augment_batch(Xb, mode=aug_mode, sigma=aug_sigma,
                                    chandrop_p=aug_chandrop_p, mask_frac=aug_timemask_frac)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xb), yb)
            loss.backward(); opt.step()
        sched.step()
        vloss, _, _ = evaluate(model, va, device)
        if vloss + 1e-6 < best:
            best, best_state, bad = vloss, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser("Stronger-architecture CNN under LOSO (per-subject norm).")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--norm-mode", default="per_subject", choices=["per_subject", "global"],
                    help="per_subject: headline transductive z-score, applied once upfront. "
                         "global: train-fold-only z-score, recomputed per LOSO fold (leak-free).")
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"],
                    help="Data augmentation applied to training batches only (identical "
                         "transforms/params to train_cnn_loso.py's --augment).")
    ap.add_argument("--aug-sigma", type=float, default=0.1,
                    help="Gaussian noise std relative to normalized data scale (default: 0.1)")
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2,
                    help="Per-channel drop probability for chandrop augmentation (default: 0.2)")
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15,
                    help="Fraction of T to zero out for timemask augmentation (default: 0.15)")
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heldout", type=int, default=None, help="Run only this held-out subject (smoke test).")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--out", default="results_cnn_loso_resnet_se")
    ap.add_argument("--save-proba", default=None,
                    help="If set, dir to save per-window softmax probabilities to, as "
                         "{save-proba}/{model-tag}_sub{K:02d}.npz (keys: proba [n,4] in "
                         "LABELS order, y_true [n]). Requires --model-tag.")
    ap.add_argument("--model-tag", default=None,
                    help="Tag used in the saved proba filename (e.g. CNN, RESNET_SE).")
    args = ap.parse_args()
    if args.save_proba and not args.model_tag:
        ap.error("--save-proba requires --model-tag")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cnn_arch_subjectwise.csv"

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    if args.norm_mode == "per_subject":
        X = per_subject_zscore_3d(X, subjects)      # headline normalisation, leak-free (own stats only)
    in_ch = X.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())
    if args.heldout is not None:
        subjects_u = [args.heldout]

    print(f"[arch] {args.arch}: {count_params(build_model(args.arch, in_ch, len(LABELS))):,} params | device={device}", flush=True)
    print(f"[aug] augmentation={args.augmentation}, sigma={args.aug_sigma}, "
          f"chandrop_p={args.aug_chandrop_p}, timemask_frac={args.aug_timemask_frac}", flush=True)
    done = set()
    if args.resume and csv_path.exists():
        done = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())

    if args.save_proba:
        proba_dir = Path(args.save_proba); proba_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import f1_score, balanced_accuracy_score
    for heldout in subjects_u:
        if heldout in done:
            continue
        te = (subjects == heldout); tr = ~te
        if args.norm_mode == "global":
            # train-fold-only stats (leak-free); recomputed per fold since the
            # training-subject set changes each time
            mean, std = compute_train_norm(X[tr])
            Xtr_full = apply_norm(X[tr], mean, std)
            Xte_fold = apply_norm(X[te], mean, std)
        else:
            Xtr_full = X[tr]
            Xte_fold = X[te]
        ytr_full, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)
        model = build_model(args.arch, in_ch, len(LABELS)).to(device)
        model = train_fold(model, Xtr_full[m_tr], ytr_full[m_tr], Xtr_full[m_va], ytr_full[m_va],
                           device, args.epochs, args.batch, args.lr, args.patience, args.seed,
                           aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
                           aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac)
        te_dl = DataLoader(WindowsDataset(Xte_fold, y[te]), batch_size=512, shuffle=False)
        yt, yp, proba = evaluate_with_proba(model, te_dl, device)
        if args.save_proba:
            proba_npz = proba_dir / f"{args.model_tag}_sub{heldout:02d}.npz"
            np.savez(proba_npz, proba=proba.astype(np.float64), y_true=yt.astype(np.int32, copy=False))
            print(f"[save-proba] {args.model_tag} Sub{heldout:02d} -> {proba_npz}", flush=True)
        row = {"subject": int(heldout), "arch": args.arch,
               "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
               "bal_acc": float(balanced_accuracy_score(yt, yp))}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"[fold] Sub{heldout:02d} {args.arch} f1={row['f1_macro']:.4f}", flush=True)

    df = pd.read_csv(csv_path).drop_duplicates("subject")
    m, s = df["f1_macro"].mean(), df["f1_macro"].std(ddof=1)
    pd.DataFrame([{"arch": args.arch, "f1_macro_mean": round(m, 4), "f1_macro_sd": round(s, 4),
                   "n": len(df)}]).to_csv(out_dir / "cnn_arch_summary.csv", index=False)
    print(f"\n[{args.arch}] LOSO F1 = {m:.4f} ± {s:.4f} (n={len(df)}) | SimpleEMGCNN headline = 0.754")


if __name__ == "__main__":
    main()

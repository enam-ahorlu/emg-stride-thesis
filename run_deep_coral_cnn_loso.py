# run_deep_coral_cnn_loso.py
# ---------------------------------------------------------------------------
# Reviewer ask (Critiques 1, 2, 5): the classical CORAL baseline was applied to
# SVM/RF only, yet domain adaptation is most interesting for the end-to-end CNN.
# This implements Deep CORAL (Sun & Saenko, 2016) for the CNN: during training,
# a CORAL loss aligns the covariance of the penultimate features between the
# labelled source (training subjects) and the UNLABELLED target (held-out
# subject), added to the classification loss. It is the deep analogue of the
# classical CORAL run and answers "does a learned UDA on the CNN beat the simple,
# label-free per-subject normalisation?".
#
# Base normalisation is train-fold GLOBAL per-channel z-score (the CNN's global
# baseline, 0.682), so Deep CORAL is the ONLY adaptation in play, exactly as the
# classical CORAL replaced per-subject norm on the features. Comparison points:
#   CNN global (no adaptation)         0.682
#   CNN per-subject z-score (simple)   0.754   <- the thesis's label-free adaptation
#   CNN + Deep CORAL (learned UDA)     <this script>
#
# Transductive UDA eval (standard): the target subject is used unlabelled for the
# CORAL term during training, then classified. No label leakage.
#
# Example (full run, GPU):
#   python run_deep_coral_cnn_loso.py \
#       --npz  windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
#       --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --arch resnet_se --coral-lambda 1.0 --epochs 40 --out results_deep_coral_cnn_resnet_se --resume
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_cnn_loso import (
    WindowsDataset, compute_train_norm, apply_norm, choose_val_subjects,
    class_weights_from_y, evaluate, normalize_label_to_str, LABELS, augment_batch,
)
from cnn_architectures import build_model, count_params

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def logits_feat(model, x):
    """Return (logits, penultimate_features) for either architecture."""
    if hasattr(model, "features"):                 # EMGResNet1D
        return model(x, return_feat=True)
    n = model.net(x)                               # SimpleEMGCNN: (N, 128, 1)
    return model.head(n), torch.flatten(n, 1)


def coral_loss(fs, ft):
    """Deep CORAL: squared Frobenius distance between source/target covariances."""
    d = fs.shape[1]
    fs = fs - fs.mean(0, keepdim=True)
    ft = ft - ft.mean(0, keepdim=True)
    cs = (fs.t() @ fs) / (fs.shape[0] - 1)
    ct = (ft.t() @ ft) / (ft.shape[0] - 1)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def train_deep_coral(model, Xs_tr, ys_tr, Xs_va, ys_va, Xt, device,
                     epochs, batch, lr, patience, lam, seed,
                     aug_mode="none", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15):
    torch.manual_seed(seed)
    w = class_weights_from_y(ys_tr, len(LABELS)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    src = DataLoader(WindowsDataset(Xs_tr, ys_tr), batch_size=batch, shuffle=True, drop_last=True)
    tgt = DataLoader(TensorDataset(torch.from_numpy(Xt.astype(np.float32))),
                     batch_size=batch, shuffle=True, drop_last=True)
    va = DataLoader(WindowsDataset(Xs_va, ys_va), batch_size=batch, shuffle=False)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        tgt_it = iter(tgt)
        for Xb, yb in src:
            try:
                (Xtb,) = next(tgt_it)
            except StopIteration:
                tgt_it = iter(tgt); (Xtb,) = next(tgt_it)
            Xb, yb, Xtb = Xb.to(device), yb.to(device), Xtb.to(device)
            if aug_mode != "none":
                # source training batches only; the unlabelled target batch used for the
                # CORAL term is left untouched (identical convention to run_cnn_arch_loso.py)
                Xb = augment_batch(Xb, mode=aug_mode, sigma=aug_sigma,
                                    chandrop_p=aug_chandrop_p, mask_frac=aug_timemask_frac)
            opt.zero_grad(set_to_none=True)
            log_s, fs = logits_feat(model, Xb)
            _, ft = logits_feat(model, Xtb)
            loss = crit(log_s, yb) + lam * coral_loss(fs, ft)
            loss.backward(); opt.step()
        sched.step()
        vloss, _, _ = evaluate(model, va, device)     # early stop on SOURCE val (no target labels used)
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
    ap = argparse.ArgumentParser("Deep CORAL (CNN UDA) under LOSO.")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--coral-lambda", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"],
                    help="Data augmentation applied to source training batches only.")
    ap.add_argument("--aug-sigma", type=float, default=0.1)
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2)
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15)
    ap.add_argument("--heldout", type=int, default=None); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results_deep_coral_cnn_resnet_se")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "deep_coral_subjectwise.csv"

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    in_ch = X.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())
    if args.heldout is not None:
        subjects_u = [args.heldout]
    print(f"[Deep CORAL] arch={args.arch} ({count_params(build_model(args.arch, in_ch, len(LABELS))):,} params) "
          f"lambda={args.coral_lambda} device={device}", flush=True)

    done = set()
    if args.resume and csv_path.exists():
        done = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())

    from sklearn.metrics import f1_score, balanced_accuracy_score
    for heldout in subjects_u:
        if heldout in done:
            continue
        te = (subjects == heldout); tr = ~te
        # GLOBAL per-channel z-score fit on TRAIN fold only (leak-free); Deep CORAL is the adaptation
        mean, std = compute_train_norm(X[tr])
        Xtr_all = apply_norm(X[tr], mean, std); Xte = apply_norm(X[te], mean, std)
        ytr_all, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)
        model = build_model(args.arch, in_ch, len(LABELS)).to(device)
        model = train_deep_coral(model, Xtr_all[m_tr], ytr_all[m_tr], Xtr_all[m_va], ytr_all[m_va],
                                 Xte, device, args.epochs, args.batch, args.lr, args.patience,
                                 args.coral_lambda, args.seed,
                                 aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
                                 aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac)
        te_dl = DataLoader(WindowsDataset(Xte, y[te]), batch_size=512, shuffle=False)
        _, yt, yp = evaluate(model, te_dl, device)
        row = {"subject": int(heldout), "arch": args.arch, "coral_lambda": args.coral_lambda,
               "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
               "bal_acc": float(balanced_accuracy_score(yt, yp))}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"[fold] Sub{heldout:02d} DeepCORAL f1={row['f1_macro']:.4f}", flush=True)

    df = pd.read_csv(csv_path).drop_duplicates("subject")
    m, s = df["f1_macro"].mean(), df["f1_macro"].std(ddof=1)
    pd.DataFrame([{"method": "DeepCORAL", "arch": args.arch, "coral_lambda": args.coral_lambda,
                   "f1_macro_mean": round(m, 4), "f1_macro_sd": round(s, 4), "n": len(df)}]
                 ).to_csv(out_dir / "deep_coral_summary.csv", index=False)
    print(f"\n[Deep CORAL/{args.arch}] LOSO F1 = {m:.4f} ± {s:.4f} (n={len(df)})")
    print("Compare: CNN global 0.682 | CNN per-subject z-score 0.754 | (classical CORAL SVM 0.724, RF 0.747)")


if __name__ == "__main__":
    main()

# run_adabn_cnn_loso.py
# ---------------------------------------------------------------------------
# Adaptive Batch Normalization (AdaBN; Li et al., 2016/2018) for the CNN — the
# deep-network analogue of the thesis's winning classical move (per-subject
# z-score), and a cheaper, more modern UDA than Deep CORAL.
#
# Idea: train the CNN on the source subjects with train-fold GLOBAL normalisation
# (no target info). Then, at test time, REPLACE every BatchNorm layer's running
# statistics with statistics computed over the held-out subject's own UNLABELLED
# windows, and classify. This re-centres/re-scales the network's internal
# activations to the target subject without touching a single weight and without
# any target labels — parameter-free, transductive, label-free.
#
# It is the fairest, most direct deep counterpart to per-subject normalisation,
# and it slots into the same comparison as Deep CORAL:
#   CNN global (no adaptation)         0.682
#   CNN per-subject z-score (input)    0.754
#   CNN + AdaBN (this script)          ?   <- adapts internal BN stats to target
#   CNN + Deep CORAL (learned UDA)     ?   (run_deep_coral_cnn_loso.py)
#
# Example (full run, GPU):
#   python run_adabn_cnn_loso.py \
#       --npz  windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
#       --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --arch resnet_se --epochs 40 --out results_adabn_cnn_resnet_se --resume
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
    WindowsDataset, compute_train_norm, apply_norm, choose_val_subjects,
    evaluate, normalize_label_to_str, LABELS,
)
from cnn_architectures import build_model, count_params
from run_cnn_arch_loso import train_fold          # reuse the exact source-training loop

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def adabn_recompute(model, Xt, device, chunk=4096):
    """Replace every BatchNorm layer's running stats with the TARGET-domain stats,
    computed over the held-out subject's unlabelled windows (Adaptive BatchNorm)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None                       # cumulative moving average
    model.train()
    with torch.no_grad():
        n = Xt.shape[0]
        # single pass over the target; chunked only if very large (BN stats pooled via CMA)
        for i in range(0, n, chunk):
            xb = torch.from_numpy(Xt[i:i + chunk].astype(np.float32)).to(device)
            if xb.shape[0] < 2:                     # BN needs >1 sample; fold a singleton in
                xb = xb.repeat(2, 1, 1)
            model(xb)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser("Adaptive BatchNorm (AdaBN) CNN UDA under LOSO.")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"],
                    help="Data augmentation applied to source training batches only.")
    ap.add_argument("--aug-sigma", type=float, default=0.1)
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2)
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15)
    ap.add_argument("--heldout", type=int, default=None); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results_adabn_cnn_resnet_se")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "adabn_subjectwise.csv"

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    in_ch = X.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())
    if args.heldout is not None:
        subjects_u = [args.heldout]
    print(f"[AdaBN] arch={args.arch} ({count_params(build_model(args.arch, in_ch, len(LABELS))):,} params) device={device}", flush=True)

    done = set()
    if args.resume and csv_path.exists():
        done = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())

    from sklearn.metrics import f1_score, balanced_accuracy_score
    for heldout in subjects_u:
        if heldout in done:
            continue
        te = (subjects == heldout); tr = ~te
        # train-fold GLOBAL per-channel z-score (leak-free); AdaBN is the only adaptation
        mean, std = compute_train_norm(X[tr])
        Xtr_all = apply_norm(X[tr], mean, std); Xte = apply_norm(X[te], mean, std)
        ytr_all, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)
        model = build_model(args.arch, in_ch, len(LABELS)).to(device)
        model = train_fold(model, Xtr_all[m_tr], ytr_all[m_tr], Xtr_all[m_va], ytr_all[m_va],
                           device, args.epochs, args.batch, args.lr, args.patience, args.seed,
                           aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
                           aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac)

        # --- pre-AdaBN (source BN stats) and post-AdaBN (target BN stats) F1 ---
        te_dl = DataLoader(WindowsDataset(Xte, y[te]), batch_size=512, shuffle=False)
        _, yt, yp0 = evaluate(model, te_dl, device)
        f1_pre = float(f1_score(yt, yp0, average="macro", zero_division=0))
        model = adabn_recompute(model, Xte, device)
        _, yt, yp1 = evaluate(model, te_dl, device)
        f1_post = float(f1_score(yt, yp1, average="macro", zero_division=0))

        row = {"subject": int(heldout), "arch": args.arch,
               "f1_pre_adabn": f1_pre, "f1_macro": f1_post,
               "bal_acc": float(balanced_accuracy_score(yt, yp1)),
               "delta_pp": round((f1_post - f1_pre) * 100, 2)}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"[fold] Sub{heldout:02d} AdaBN pre={f1_pre:.4f} -> post={f1_post:.4f} ({row['delta_pp']:+.1f}pp)", flush=True)

    df = pd.read_csv(csv_path).drop_duplicates("subject")
    m, s = df["f1_macro"].mean(), df["f1_macro"].std(ddof=1)
    pre = df["f1_pre_adabn"].mean()
    pd.DataFrame([{"method": "AdaBN", "arch": args.arch, "f1_pre_adabn_mean": round(pre, 4),
                   "f1_macro_mean": round(m, 4), "f1_macro_sd": round(s, 4), "n": len(df)}]
                 ).to_csv(out_dir / "adabn_summary.csv", index=False)
    print(f"\n[AdaBN/{args.arch}] pre {pre:.4f} -> post {m:.4f} ± {s:.4f} (n={len(df)})")
    print("Compare: CNN global 0.682 | CNN per-subject z-score 0.754 | classical CORAL SVM 0.724 / RF 0.747")


if __name__ == "__main__":
    main()

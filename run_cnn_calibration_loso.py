# run_cnn_calibration_loso.py
# ---------------------------------------------------------------------------
# Action 2.6 — Transfer-learning / calibration proof-of-concept for the CNN.
#
# The thesis repeatedly names "fine-tune the CNN on a small calibration set
# from the new subject" as the obvious next step. This script demonstrates it:
# for each held-out subject it (a) trains the base CNN on the other 39 subjects
# (per-subject z-score, exactly as in the headline run), then (b) fine-tunes
# that model on K labelled windows PER CLASS from the held-out subject and
# evaluates on the remaining windows of that subject. K = 0 is the no-calibration
# LOSO baseline. The evaluation set is held fixed across K (the windows after the
# calibration pool), so the F1 lift is a clean apples-to-apples comparison.
#
# Reuses the model definition and helpers from train_cnn_loso.py.
#
# Example:
#   python run_cnn_calibration_loso.py \
#       --npz  windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
#       --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --calib-list 0,5,10,20 --ft-epochs 15
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_cnn_loso import (
    WindowsDataset, per_subject_zscore_3d, choose_val_subjects,
    class_weights_from_y, evaluate, normalize_label_to_str, LABELS, augment_batch,
)
from cnn_architectures import build_model

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}


def train_model(X_tr, y_tr, X_va, y_va, in_ch, device, epochs, batch, lr, patience, seed, arch="simple",
                 aug_mode="none", aug_sigma=0.1, aug_chandrop_p=0.2, aug_timemask_frac=0.15):
    torch.manual_seed(seed)
    model = build_model(arch, in_ch, len(LABELS)).to(device)
    w = class_weights_from_y(y_tr, len(LABELS)).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr = DataLoader(WindowsDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    va = DataLoader(WindowsDataset(X_va, y_va), batch_size=batch, shuffle=False)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for Xb, yb in tr:
            Xb, yb = Xb.to(device), yb.to(device)
            if aug_mode != "none":
                # base pretraining only (identical to run_cnn_arch_loso.py); fine-tune stage is untouched
                Xb = augment_batch(Xb, mode=aug_mode, sigma=aug_sigma,
                                    chandrop_p=aug_chandrop_p, mask_frac=aug_timemask_frac)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xb), yb)
            loss.backward(); opt.step()
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


def finetune(base_model, X_cal, y_cal, in_ch, device, epochs, lr, batch):
    model = copy.deepcopy(base_model).to(device)
    crit = nn.CrossEntropyLoss()  # calibration set is small; uniform weighting
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(WindowsDataset(X_cal, y_cal), batch_size=min(batch, len(y_cal)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xb), yb)
            loss.backward(); opt.step()
    return model


def macro_f1(model, X, y, device):
    from sklearn.metrics import f1_score
    dl = DataLoader(WindowsDataset(X, y), batch_size=512, shuffle=False)
    _, yt, yp = evaluate(model, dl, device)
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser("CNN calibration fine-tune PoC under LOSO.")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="simple", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--calib-list", default="0,5,10,20", help="windows/class for fine-tuning")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"],
                    help="Data augmentation applied to BASE-model training batches only "
                         "(identical transforms/params to train_cnn_loso.py's --augment); "
                         "the fine-tune stage is unaugmented.")
    ap.add_argument("--aug-sigma", type=float, default=0.1)
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2)
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15)
    ap.add_argument("--resume", action="store_true",
                    help="Resume: skip held-out subjects already saved in --out; append new ones as they finish.")
    ap.add_argument("--out", default="results_cnn_calibration")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    calib_ks = sorted(set(int(k) for k in args.calib_list.split(",")))
    kmax = max(calib_ks)

    meta = pd.read_csv(args.meta)
    data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)            # (N, C, T)
    y_str = meta[args.label_col].map(normalize_label_to_str).values
    y = np.array([LABEL_TO_IDX[s] for s in y_str], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    time_col = next((c for c in ["t_start", "win_start", "start"] if c in meta.columns), None)
    tvals = meta[time_col].to_numpy() if time_col else np.arange(len(y))

    # headline normalisation: per-subject z-score before the loop
    X = per_subject_zscore_3d(X, subjects)
    in_ch = X.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())

    csv_path = out_dir / "cnn_calibration_subjectwise.csv"
    done_subj = set()
    if args.resume and csv_path.exists():
        done_subj = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())
        if done_subj:
            print(f"[resume] {len(done_subj)} subjects already done, skipping them")

    for heldout in subjects_u:
        if heldout in done_subj:
            continue
        te = (subjects == heldout); tr = ~te
        Xtr_full, ytr_full, subtr = X[tr], y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr = np.isin(subtr, tr_subs); m_va = np.isin(subtr, va_subs)
        base = train_model(Xtr_full[m_tr], ytr_full[m_tr], Xtr_full[m_va], ytr_full[m_va],
                           in_ch, device, args.epochs, args.batch, args.lr, args.patience, args.seed,
                           arch=args.arch, aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
                           aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac)

        # test subject: order by time, build a fixed calibration pool (first kmax/class)
        Xte, yte = X[te], y[te]
        order = np.argsort(tvals[te], kind="stable")
        Xte, yte = Xte[order], yte[order]
        pool_idx = []  # indices (into ordered test) reserved for calibration
        for c in range(len(LABELS)):
            cls_idx = np.where(yte == c)[0]
            pool_idx.extend(cls_idx[:kmax].tolist())
        pool_idx = set(pool_idx)
        eval_idx = np.array([i for i in range(len(yte)) if i not in pool_idx])
        if len(eval_idx) < len(LABELS):
            print(f"  [skip] Sub{heldout:02d}: too few eval windows")
            continue
        Xev, yev = Xte[eval_idx], yte[eval_idx]

        subj_rows = []
        for K in calib_ks:
            if K == 0:
                f1 = macro_f1(base, Xev, yev, device)
            else:
                cal_idx = []
                for c in range(len(LABELS)):
                    cls_idx = np.where(yte == c)[0]
                    cls_idx = [i for i in cls_idx if i in pool_idx][:K]
                    cal_idx.extend(cls_idx)
                cal_idx = np.array(cal_idx)
                if len(cal_idx) < len(LABELS):
                    f1 = macro_f1(base, Xev, yev, device)  # not enough calib data; fall back
                else:
                    ft = finetune(base, Xte[cal_idx], yte[cal_idx], in_ch, device,
                                  args.ft_epochs, args.ft_lr, args.batch)
                    f1 = macro_f1(ft, Xev, yev, device)
            subj_rows.append({"subject": int(heldout), "K_per_class": K, "f1_macro": f1})
        # CHECKPOINT: append this subject's rows immediately (crash-safe / resumable)
        pd.DataFrame(subj_rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done_subj.add(heldout)
        print(f"[fold] Sub{heldout:02d} done: " +
              ", ".join(f"K{r['K_per_class']}={r['f1_macro']:.3f}" for r in subj_rows))

    df = pd.read_csv(csv_path).drop_duplicates(["subject", "K_per_class"])
    summ = (df.groupby("K_per_class")["f1_macro"]
              .agg(["mean", "std", "count"]).reset_index()
              .rename(columns={"mean": "f1_mean", "std": "f1_sd", "count": "n"}))
    summ["f1_mean"] = summ["f1_mean"].round(4); summ["f1_sd"] = summ["f1_sd"].round(4)
    summ.to_csv(out_dir / "cnn_calibration_summary.csv", index=False)
    print("\n================  CALIBRATION SUMMARY (paste this back)  ================")
    print(summ.to_string(index=False))
    base0 = summ.loc[summ.K_per_class == 0, "f1_mean"]
    if len(base0):
        print(f"\nLift vs K=0 (no calibration, evaluated on the held-out eval set):")
        for _, r in summ.iterrows():
            print(f"  K={int(r.K_per_class):>2}/class : F1 {r.f1_mean:.4f}  "
                  f"(+{(r.f1_mean - float(base0.iloc[0]))*100:.1f} pp)")
    print(f"[save] {out_dir/'cnn_calibration_summary.csv'}")


if __name__ == "__main__":
    main()

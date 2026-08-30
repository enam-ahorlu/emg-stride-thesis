# run_adabn_causal_loso.py
# ---------------------------------------------------------------------------
# G3: causal / streaming AdaBN for the CNN — the deployable deep analogue of the
# classical causal-normalization study (run_streaming_norm_loso.py, Section 4.14),
# which was flagged "untested" on the deep side. Full-session AdaBN
# (run_adabn_cnn_loso.py) recomputes each BatchNorm layer's running stats from
# the held-out subject's WHOLE session, which is non-causal (a real-time system
# cannot see the whole session before classifying the first window). This script
# estimates BN stats from only what a deployed system could actually have seen:
#
#   calibK   : BN stats frozen from the FIRST K time-ordered windows only (a short,
#              one-off, label-free calibration buffer collected at session start),
#              then every window (including the buffer) is classified with those
#              frozen stats — mirrors causal_calib_stats() in the classical script.
#   running  : BN stats start from a `--warmup` buffer, then update causally,
#              one window at a time, in time order: window i is classified using
#              stats accumulated from windows [0..i-1] (strictly causal), and only
#              THEN is window i folded into the running stats for future windows.
#
# Base model training is unchanged from run_adabn_cnn_loso.py: trained on source
# subjects with train-fold GLOBAL normalisation, so AdaBN (causal or full-session)
# is the only adaptation mechanism, keeping the comparison to per-subject
# normalisation and to full-session AdaBN clean.
#
# Example (full run, GPU):
#   python run_adabn_causal_loso.py \
#       --npz  windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz \
#       --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv \
#       --arch resnet_se --configs calib25,calib50,calib100,running --out results_adabn_causal --resume
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score

from train_cnn_loso import (
    WindowsDataset, compute_train_norm, apply_norm, choose_val_subjects,
    evaluate, normalize_label_to_str, LABELS,
)
from cnn_architectures import build_model, count_params
from run_cnn_arch_loso import train_fold

LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}
TIME_COL_CANDIDATES = ["t_start", "win_start", "start", "t0"]


def reset_bn(model):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None  # cumulative moving average


def bn_forward_update(model, xb, device):
    """One train()-mode forward pass folds xb into the BN running stats
    (cumulative moving average, since momentum=None); no weights change
    (no optimizer step is ever taken)."""
    model.train()
    with torch.no_grad():
        x = torch.from_numpy(xb.astype(np.float32)).to(device)
        if x.shape[0] < 2:
            x = x.repeat(2, 1, 1)
        model(x)


def bn_classify(model, xb, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(xb.astype(np.float32)).to(device)
        logits = model(x)
        return logits.argmax(1).cpu().numpy()


def adabn_calibK(model, Xte_ordered, K, device, chunk=4096):
    """Freeze BN stats from the first K time-ordered windows only, then return
    a model ready to classify (eval mode)."""
    reset_bn(model)
    K = max(2, min(K, Xte_ordered.shape[0]))
    for i in range(0, K, chunk):
        bn_forward_update(model, Xte_ordered[i:min(i + chunk, K)], device)
    model.eval()
    return model


def adabn_running(model, Xte_ordered, warmup, device):
    """Causal expanding-window BN stats: warmup buffer seeds initial stats
    (frozen for the warmup windows themselves, matching the classical script's
    convention), then each subsequent window is classified using stats from
    strictly PRIOR windows before being folded in itself."""
    n = Xte_ordered.shape[0]
    w = max(2, min(warmup, n))
    reset_bn(model)
    bn_forward_update(model, Xte_ordered[:w], device)
    preds = np.empty(n, dtype=np.int64)
    # warmup windows classified with the warmup-derived stats (frozen)
    preds[:w] = bn_classify(model, Xte_ordered[:w], device)
    for i in range(w, n):
        preds[i] = bn_classify(model, Xte_ordered[i:i + 1], device)[0]
        bn_forward_update(model, Xte_ordered[i:i + 1], device)
    return preds


def main():
    ap = argparse.ArgumentParser("Causal / streaming AdaBN for the CNN under LOSO.")
    ap.add_argument("--npz", required=True); ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="resnet_se", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--val-frac", type=float, default=0.15); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="calib25,calib50,calib100,running",
                    help="Comma list: calibK (e.g. calib50), running")
    ap.add_argument("--warmup", type=int, default=16,
                    help="Warmup windows for the 'running' estimator (default 16 ~= 2s at 125ms step)")
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"],
                    help="Data augmentation applied to source training batches only.")
    ap.add_argument("--aug-sigma", type=float, default=0.1)
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2)
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15)
    ap.add_argument("--heldout", type=int, default=None); ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results_adabn_causal")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "adabn_causal_subjectwise.csv"

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    meta = pd.read_csv(args.meta); data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y = np.array([LABEL_TO_IDX[s] for s in meta[args.label_col].map(normalize_label_to_str).values], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    time_col = next((c for c in TIME_COL_CANDIDATES if c in meta.columns), None)
    tvals = meta[time_col].to_numpy() if time_col else np.arange(len(y))
    if time_col:
        print(f"[info] ordering test-subject windows by '{time_col}'")
    in_ch = X.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())
    if args.heldout is not None:
        subjects_u = [args.heldout]
    print(f"[AdaBN-causal] arch={args.arch} ({count_params(build_model(args.arch, in_ch, len(LABELS))):,} params) "
          f"configs={configs} device={device}", flush=True)

    done = set()
    if args.resume and csv_path.exists():
        d = pd.read_csv(csv_path); done = set(zip(d.subject, d.config))

    for heldout in subjects_u:
        if all((heldout, c) in done for c in configs):
            continue
        te = (subjects == heldout); tr = ~te
        mean, std = compute_train_norm(X[tr])
        Xtr_all = apply_norm(X[tr], mean, std); Xte = apply_norm(X[te], mean, std)
        ytr_all, subtr = y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr, m_va = np.isin(subtr, tr_subs), np.isin(subtr, va_subs)
        base_model = build_model(args.arch, in_ch, len(LABELS)).to(device)
        base_model = train_fold(base_model, Xtr_all[m_tr], ytr_all[m_tr], Xtr_all[m_va], ytr_all[m_va],
                                device, args.epochs, args.batch, args.lr, args.patience, args.seed,
                                aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
                                aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac)
        base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}

        yte = y[te]
        order = np.argsort(tvals[te], kind="stable")
        Xord, yord = Xte[order], yte[order]

        for cfg in configs:
            if (heldout, cfg) in done:
                continue
            model = build_model(args.arch, in_ch, len(LABELS)).to(device)
            model.load_state_dict(base_state)

            if cfg == "running":
                preds_ord = adabn_running(model, Xord, args.warmup, device)
            elif cfg.startswith("calib"):
                K = int(cfg.replace("calib", ""))
                model = adabn_calibK(model, Xord, K, device)
                preds_ord = bn_classify(model, Xord, device)
            else:
                raise ValueError(f"unknown config {cfg}")

            f1 = float(f1_score(yord, preds_ord, average="macro", zero_division=0))
            bal = float(balanced_accuracy_score(yord, preds_ord))
            row = {"subject": int(heldout), "arch": args.arch, "config": cfg, "f1_macro": f1, "bal_acc": bal}
            pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            done.add((heldout, cfg))
            print(f"[fold] Sub{heldout:02d} {cfg}: f1={f1:.4f}", flush=True)

    df = pd.read_csv(csv_path).drop_duplicates(["subject", "config"])
    summ = (df.groupby("config")["f1_macro"].agg(["mean", "std", "count"]).reset_index()
              .rename(columns={"mean": "f1_mean", "std": "f1_sd", "count": "n"}))
    summ["f1_mean"] = summ["f1_mean"].round(4); summ["f1_sd"] = summ["f1_sd"].round(4)
    summ.to_csv(out_dir / "adabn_causal_summary.csv", index=False)
    print("\n=== CAUSAL AdaBN SUMMARY ===")
    print(summ.to_string(index=False))
    print("\nCompare: CNN global norm | full-session AdaBN | per-subject z-score (transductive upper bound)")


if __name__ == "__main__":
    main()

# run_cnn_calibration_multidraw.py
# ---------------------------------------------------------------------------
# Robustness supplement for Section 4.15 (supervised CNN calibration).
#
# The primary calibration run draws a single, deployment-realistic buffer: the
# FIRST K windows/class in acquisition order. That answers "calibrate at the
# start of a session", but it does not tell us how sensitive the +3.2 pp lift is
# to WHICH K windows happen to be sampled. This script quantifies that variance:
# for each held-out subject it trains the base CNN once (seed 42, reproducing the
# headline base), then repeats the calibration N_DRAWS times, each time sampling
# K windows/class at RANDOM (nested K=5 in 10 in 20) and evaluating on the
# remaining windows. The regularised 3-epoch fine-tune schedule (the thesis's
# headline positive) is used throughout.
#
# Reports, per K: mean +/- SD of F1 and of the within-draw lift over K=0, pooled
# across draws x subjects, plus per-draw subject-mean values so the draw-to-draw
# spread of the aggregate lift is visible.
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from run_cnn_calibration_loso import train_model, finetune, macro_f1, LABEL_TO_IDX
from train_cnn_loso import per_subject_zscore_3d, choose_val_subjects, normalize_label_to_str, LABELS


def main():
    ap = argparse.ArgumentParser("CNN calibration — multi-draw variance under LOSO.")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--xkey", default="X_env", choices=["X_env", "X_raw"])
    ap.add_argument("--label-col", default="movement")
    ap.add_argument("--arch", default="simple", choices=["simple", "resnet", "resnet_se"])
    ap.add_argument("--calib-list", default="0,5,10,20")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=3)     # regularised schedule (headline)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-draws", type=int, default=5)
    ap.add_argument("--augmentation", "--augment", dest="augmentation", default="none",
                    choices=["none", "gaussian", "chandrop", "timemask", "combined"],
                    help="Data augmentation applied to BASE-model training batches only; "
                         "the fine-tune stage is unaugmented.")
    ap.add_argument("--aug-sigma", type=float, default=0.1)
    ap.add_argument("--aug-chandrop-p", type=float, default=0.2)
    ap.add_argument("--aug-timemask-frac", type=float, default=0.15)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="results_cnn_calibration_multidraw")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    calib_ks = sorted(set(int(k) for k in args.calib_list.split(","))); kmax = max(calib_ks)

    meta = pd.read_csv(args.meta)
    data = np.load(args.npz)
    X = data[args.xkey].astype(np.float32)
    y_str = meta[args.label_col].map(normalize_label_to_str).values
    y = np.array([LABEL_TO_IDX[s] for s in y_str], dtype=np.int64)
    subjects = meta["subject"].astype(int).values
    X = per_subject_zscore_3d(X, subjects)
    in_ch = X.shape[1]
    subjects_u = sorted(np.unique(subjects).tolist())

    csv_path = out_dir / "cnn_calibration_multidraw_subjectwise.csv"
    done = set()
    if args.resume and csv_path.exists():
        dd = pd.read_csv(csv_path)
        done = set(zip(dd["subject"].astype(int), dd["draw"].astype(int)))
        print(f"[resume] {len(done)} (subject,draw) rows done", flush=True)

    for heldout in subjects_u:
        if all((heldout, d) in done for d in range(args.n_draws)):
            continue
        te = (subjects == heldout); tr = ~te
        Xtr_full, ytr_full, subtr = X[tr], y[tr], subjects[tr]
        tr_subs, va_subs = choose_val_subjects(subtr, args.val_frac, args.seed + heldout)
        m_tr = np.isin(subtr, tr_subs); m_va = np.isin(subtr, va_subs)
        base = train_model(Xtr_full[m_tr], ytr_full[m_tr], Xtr_full[m_va], ytr_full[m_va],
                           in_ch, device, args.epochs, args.batch, args.lr, args.patience, args.seed,
                           arch=args.arch, aug_mode=args.augmentation, aug_sigma=args.aug_sigma,
                           aug_chandrop_p=args.aug_chandrop_p, aug_timemask_frac=args.aug_timemask_frac)
        Xte, yte = X[te], y[te]
        cls_pool = {c: np.where(yte == c)[0] for c in range(len(LABELS))}
        for d in range(args.n_draws):
            if (heldout, d) in done:
                continue
            rng = np.random.default_rng(args.seed * 1000 + heldout * 10 + d)
            pool_by_c = []
            for c in range(len(LABELS)):
                idx = cls_pool[c]
                sel = rng.choice(idx, size=min(kmax, len(idx)), replace=False) if len(idx) else np.array([], int)
                pool_by_c.append(sel)
            pool_set = set(np.concatenate([p for p in pool_by_c if len(p)]).tolist())
            eval_idx = np.array([i for i in range(len(yte)) if i not in pool_set])
            if len(eval_idx) < len(LABELS):
                continue
            Xev, yev = Xte[eval_idx], yte[eval_idx]
            rows = []
            for K in calib_ks:
                if K == 0:
                    f1 = macro_f1(base, Xev, yev, device)
                else:
                    cal = []
                    for c in range(len(LABELS)):
                        cal.extend(pool_by_c[c][:K].tolist())
                    cal = np.array(cal)
                    if len(cal) < len(LABELS):
                        f1 = macro_f1(base, Xev, yev, device)
                    else:
                        ft = finetune(base, Xte[cal], yte[cal], in_ch, device,
                                      args.ft_epochs, args.ft_lr, args.batch)
                        f1 = macro_f1(ft, Xev, yev, device)
                rows.append({"subject": int(heldout), "draw": int(d),
                             "K_per_class": K, "f1_macro": round(float(f1), 6)})
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            print(f"[fold] Sub{heldout:02d} draw{d}: " +
                  ", ".join(f"K{r['K_per_class']}={r['f1_macro']:.3f}" for r in rows), flush=True)

    df = pd.read_csv(csv_path).drop_duplicates(["subject", "draw", "K_per_class"])
    base0 = df[df.K_per_class == 0].rename(columns={"f1_macro": "f1_0"})[["subject", "draw", "f1_0"]]
    dl = df.merge(base0, on=["subject", "draw"])
    dl["lift_pp"] = (dl["f1_macro"] - dl["f1_0"]) * 100
    g = (dl.groupby("K_per_class")
           .agg(f1_mean=("f1_macro", "mean"), f1_sd=("f1_macro", "std"),
                lift_mean=("lift_pp", "mean"), lift_sd=("lift_pp", "std"),
                n=("f1_macro", "size")).reset_index().round(4))
    g.to_csv(out_dir / "cnn_calibration_multidraw_summary.csv", index=False)
    per_draw = (dl.groupby(["draw", "K_per_class"])
                  .agg(f1_mean=("f1_macro", "mean"), lift_mean=("lift_pp", "mean"))
                  .reset_index().round(4))
    per_draw.to_csv(out_dir / "cnn_calibration_multidraw_perdraw.csv", index=False)
    print("\n===== MULTIDRAW SUMMARY (pooled over draws x subjects) =====")
    print(g.to_string(index=False))
    print("\n===== PER-DRAW subject-mean F1 / lift =====")
    print(per_draw.to_string(index=False))
    print(f"[save] {out_dir/'cnn_calibration_multidraw_summary.csv'}")


if __name__ == "__main__":
    main()

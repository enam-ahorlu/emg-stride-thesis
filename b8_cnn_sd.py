#!/usr/bin/env python3
"""
b8_cnn_sd.py
============
B8 / section 4A: the CNN subject-dependent arm, re-run with the movement-blocked
+ guard-band fold design instead of the pooled random StratifiedKFold. Reuses
SimpleEMGCNN and the training helpers from train_cnn_subjectdep.py unchanged;
only the fold assignment changes. Runs both schemes (pooled and movement-blocked)
so the delta is the protocol change alone.

  python b8_cnn_sd.py --npz windows_..._w250_..._AorR.npz --meta <its _meta.csv> \
      --use env --norm zscore --epochs 20 --out results_b8_sd

Well under one LOSO run (40 subjects x 5 folds x ~20 epochs on ~500 windows).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from train_cnn_subjectdep import (EMGWindowDataset, SimpleEMGCNN, load_npz, pick_X,
                                  encode_labels, zscore_fit, zscore_apply,
                                  train_one_epoch, eval_model)
from b8_movement_blocked_sd import movement_blocked_folds

ROOT = Path(__file__).resolve().parent
SEED = 42


def one_subject(X, y, meta_sub, scheme, n_folds, win_len, epochs, norm, device, patience=5):
    if scheme == "pooled":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        fold = np.full(len(y), -1, dtype=int)
        for f, (_, te) in enumerate(skf.split(X, y)):
            fold[te] = f
        dropped = 0
    else:
        fold, dropped, _ = movement_blocked_folds(meta_sub.reset_index(drop=True), n_folds, win_len)
    keep = fold >= 0
    f1s = []
    for f in range(n_folds):
        te = np.where(keep & (fold == f))[0]
        tr = np.where(keep & (fold != f))[0]
        if len(te) == 0 or len(tr) == 0 or len(np.unique(y[tr])) < len(np.unique(y)):
            continue
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        vs = max(1, int(0.15 * len(Xtr)))
        Xin, Xval, yin, yval = Xtr[:-vs], Xtr[-vs:], ytr[:-vs], ytr[-vs:]
        if norm == "zscore":
            mu, sd = zscore_fit(Xin)
            Xin, Xval, Xte = zscore_apply(Xin, mu, sd), zscore_apply(Xval, mu, sd), zscore_apply(Xte, mu, sd)
        tl = DataLoader(EMGWindowDataset(Xin, yin), batch_size=128, shuffle=True)
        vl = DataLoader(EMGWindowDataset(Xval, yval), batch_size=128)
        el = DataLoader(EMGWindowDataset(Xte, yte), batch_size=128)
        torch.manual_seed(SEED)
        model = SimpleEMGCNN(in_ch=X.shape[1], n_classes=len(np.unique(y)), dropout=0.25).to(device)
        loss_fn = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best, best_state, bad = float("inf"), None, 0
        for _ in range(epochs):
            train_one_epoch(model, tl, loss_fn, opt, device)
            model.eval()
            vloss, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in vl:
                    xb, yb = xb.to(device), yb.to(device)
                    vloss += loss_fn(model(xb), yb).item() * len(yb); vn += len(yb)
            vloss /= max(vn, 1)
            if vloss < best:
                best, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            model.load_state_dict(best_state)
        _, _, f1, _, _ = eval_model(model, el, device)
        f1s.append(f1)
    return f1s, dropped, len(y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--use", choices=["raw", "env"], default="env")
    ap.add_argument("--norm", choices=["none", "zscore"], default="zscore")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--window-ms", type=int, default=250)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--out", default="results_b8_sd")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    win_len = args.window_ms / 1000.0
    npz, keys = load_npz(Path(ROOT / args.npz))
    Xall = pick_X(npz, keys, args.use)
    meta = pd.read_csv(ROOT / args.meta)
    print(f"[b8 cnn SD] X {Xall.shape}, {meta['subject'].nunique()} subjects, use={args.use} "
          f"norm={args.norm} epochs={args.epochs} device={device} guard {win_len:.3f}s")

    subs = sorted(meta["subject"].unique())
    res = {"pooled": {}, "movement_blocked": {}}
    gd = gt = 0
    t0 = time.time()
    for si, s in enumerate(subs, 1):
        m = meta["subject"] == s
        idx = meta.index[m].to_numpy()
        ms = meta.loc[m].reset_index(drop=True)
        X = Xall[idx]
        y, _ = encode_labels(ms["movement"].astype(str).values)
        for scheme in ("pooled", "movement_blocked"):
            f1s, dr, tot = one_subject(X, y, ms, scheme, args.splits, win_len, args.epochs,
                                       args.norm, device)
            res[scheme][int(s)] = float(np.mean(f1s)) if f1s else np.nan
            if scheme == "movement_blocked":
                gd += dr; gt += tot
        if si % 10 == 0:
            print(f"  {si}/{len(subs)} subjects  ({time.time()-t0:.0f}s)")

    a = np.array([res["movement_blocked"][s] for s in subs])
    b = np.array([res["pooled"][s] for s in subs])
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = a[ok] - b[ok]
    w = stats.wilcoxon(a[ok], b[ok])
    dz = d.mean() / d.std(ddof=1)
    print(f"\n[b8 cnn SD] guard band dropped {gd}/{gt} ({gd/gt:.2%})")
    print(f"[b8 cnn SD] SD macro-F1  old (pooled) {b[ok].mean()*100:.2f}   "
          f"new (movement-blocked) {a[ok].mean()*100:.2f}   delta {d.mean()*100:+.2f} pp   "
          f"p = {w.pvalue:.4g}   d = {dz:+.2f}   n = {ok.sum()}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject": subs, "CNN_old": b, "CNN_new": a}).round(5).to_csv(
        out_dir / "b8_cnn_w250_subjectwise.csv", index=False)
    json.dump({"tag": "cnn_w250", "use": args.use, "norm": args.norm, "epochs": args.epochs,
               "guard_frac": gd / gt, "old_mean": float(b[ok].mean()), "new_mean": float(a[ok].mean()),
               "delta_pp": float(d.mean() * 100), "wilcoxon_p": float(w.pvalue),
               "cohens_d": float(dz), "n": int(ok.sum())},
              open(out_dir / "b8_cnn_w250_outcome.json", "w"), indent=2)
    print(f"wrote {out_dir}/b8_cnn_w250_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

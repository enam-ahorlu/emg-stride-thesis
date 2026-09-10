#!/usr/bin/env python3
"""
b8_movement_blocked_sd.py
=========================
B8 / section 4A of EXPERIMENT_PLAN_AUDIT_REMEDIATION.md. The subject-dependent
(within-subject) protocol currently splits 50%-overlapping windows at random
(pooled StratifiedKFold over all 26,347 windows), so a window and its overlapping
neighbour can land on opposite sides of a fold edge. This re-runs the classical
SD arms with the correct design from section 4A.1:

  per subject, per movement: sort that movement's windows by t_start, cut into
  n_splits contiguous chunks, assign chunk i of every movement to fold i, so each
  fold is a contiguous time block of all four movements and class-balanced by
  construction. Then drop windows within one window length of every chunk
  boundary (the guard band) so no overlapping pair straddles a fold edge.

For each config it reports the OLD (pooled random) and NEW (movement-blocked +
guard band) SD macro-F1 side by side, on identical models, so the delta is the
protocol change alone. Fixed LOSO-consistent hyperparameters (no per-fold
GridSearchCV; section 4A.2 budgets this at seconds of CPU).

  python b8_movement_blocked_sd.py --features <npz> --meta <csv> --models SVM,RF,LDA \
      --window-ms 250 --tag freq72_w250 --out results_b8_sd

No thesis file edited; Section 4.17 not touched. New paired tests reported for
the FDR recompute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
LABELS = ["DNS", "STDUP", "UPS", "WAK"]  # alphabetical == y_int encode order
LAB2I = {l: i for i, l in enumerate(LABELS)}
SEED = 42


def build_model(name: str):
    name = name.upper()
    if name == "SVM":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", SVC(C=1.0, gamma="scale", class_weight="balanced",
                                     random_state=SEED))])
    if name == "RF":
        return RandomForestClassifier(n_estimators=400, max_depth=None,
                                      class_weight="balanced", random_state=SEED, n_jobs=4)
    if name == "LDA":
        return Pipeline([("sc", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])
    sys.exit(f"unknown model {name}")


def movement_blocked_folds(meta_s: pd.DataFrame, n_splits: int, win_len: float):
    """meta_s carries a clean 0..n-1 RangeIndex. Returns (fold array of length n
    in that row order, n_dropped, n_total); fold == -1 marks a guard-band drop."""
    fold = np.full(len(meta_s), -1, dtype=int)
    dropped = 0
    for _, g in meta_s.groupby("movement", sort=False):
        gi = g.sort_values("t_start", kind="stable")
        rows = gi.index.to_numpy()               # positions into meta_s (0..n-1)
        ts = gi["t_start"].to_numpy()
        chunks = np.array_split(np.arange(len(rows)), n_splits)
        bnds = [ts[chunks[k + 1][0]] for k in range(n_splits - 1) if len(chunks[k + 1])]
        for k, ch in enumerate(chunks):
            for local in ch:
                if any(abs(ts[local] - b) < win_len for b in bnds):
                    dropped += 1                 # leave fold[rows[local]] == -1
                else:
                    fold[rows[local]] = k
    return fold, dropped, len(meta_s)


def eval_sd(X, y, meta, scheme: str, n_splits: int, win_len: float):
    """Return per-subject dict of {model:{fold_f1_list}} plus X-flags."""
    subs = sorted(meta["subject"].unique())
    out = {m: {} for m in MODELS}
    x_flags = []
    guard_dropped_total = guard_total = 0
    for s in subs:
        ms = meta[meta["subject"] == s]
        Xs = X[ms.index.to_numpy()]
        ys = y[ms.index.to_numpy()]
        if scheme == "pooled":
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            fold = np.full(len(ms), -1, dtype=int)
            for f, (_, te) in enumerate(skf.split(Xs, ys)):
                fold[te] = f
        else:  # movement_blocked
            fold, dr, tot = movement_blocked_folds(ms.reset_index(drop=True), n_splits, win_len)
            guard_dropped_total += dr; guard_total += tot
        keep = fold >= 0
        for m in MODELS:
            f1s = []
            for f in range(n_splits):
                te = keep & (fold == f)
                tr = keep & (fold != f)
                if te.sum() == 0 or tr.sum() == 0:
                    x_flags.append((int(s), m, f, "empty fold"))
                    continue
                if len(np.unique(ys[tr])) < len(LABELS):
                    x_flags.append((int(s), m, f, f"train missing class(es): "
                                    f"{set(range(4)) - set(np.unique(ys[tr]).tolist())}"))
                clf = build_model(m)
                clf.fit(Xs[tr], ys[tr])
                yp = clf.predict(Xs[te])
                f1s.append(f1_score(ys[te], yp, average="macro"))
            out[m][int(s)] = f1s
    return out, x_flags, (guard_dropped_total, guard_total)


def summarise(per_subject: dict) -> dict:
    res = {}
    for m, d in per_subject.items():
        subj_means = np.array([np.mean(v) for v in d.values() if len(v)])
        res[m] = {"subject_f1": {s: float(np.mean(v)) for s, v in d.items() if len(v)},
                  "mean": float(subj_means.mean()), "sd": float(subj_means.std(ddof=1)),
                  "n": int(len(subj_means))}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--models", default="SVM,RF,LDA")
    ap.add_argument("--window-ms", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results_b8_sd")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--time-units", choices=["seconds", "samples"], default="seconds",
                    help="units of the meta's t_start column. SIAT writes seconds; the ENABL3S "
                         "windower writes sample indices, and a guard band of 0.25 in a "
                         "sample-indexed axis silently drops nothing.")
    ap.add_argument("--min-guard-frac", type=float, default=0.0,
                    help="abort if the guard band drops less than this fraction. Set it on any "
                         "new dataset: a near-zero guard fraction means the units are wrong.")
    args = ap.parse_args()

    global MODELS
    MODELS = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    X = np.load(ROOT / args.features)["X"].astype(np.float64)
    meta = pd.read_csv(ROOT / args.meta).reset_index(drop=True)
    y = meta["movement"].map(LAB2I).to_numpy()
    assert len(X) == len(meta) == len(y), (X.shape, len(meta))

    if args.time_units == "seconds":
        win_len = args.window_ms / 1000.0
        unit = "s"
    else:
        fs_med = float(np.median(meta["fs"].to_numpy()))
        win_len = args.window_ms / 1000.0 * fs_med
        unit = f"samples (fs {fs_med:g} Hz)"
    print(f"[{args.tag}] X {X.shape}, {meta['subject'].nunique()} subjects, models {MODELS}, "
          f"window {args.window_ms} ms, t_start in {args.time_units}, "
          f"guard band {win_len:g} {unit}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled, pflags, _ = eval_sd(X, y, meta, "pooled", args.splits, win_len)
    blocked, bflags, (gd, gt) = eval_sd(X, y, meta, "movement_blocked", args.splits, win_len)
    ps, bs = summarise(pooled), summarise(blocked)

    guard_frac = gd / gt if gt else 0.0
    print(f"\n[{args.tag}] guard band dropped {gd}/{gt} windows ({guard_frac:.2%})")
    if guard_frac < args.min_guard_frac:
        print(f"[{args.tag}] ABORT: guard fraction {guard_frac:.2%} below the "
              f"{args.min_guard_frac:.2%} floor. The t_start units are almost certainly wrong, "
              f"so the blocked arm is not leak-free. Nothing written.")
        return 2
    if bflags:
        print(f"[{args.tag}] OUTCOME-X flags ({len(bflags)}): {bflags[:8]}"
              + (" ..." if len(bflags) > 8 else ""))
    else:
        print(f"[{args.tag}] no outcome-X flags (every fold has test windows and all 4 classes in train)")

    print(f"\n[{args.tag}] SD macro-F1: OLD (pooled random) vs NEW (movement-blocked + guard band)")
    rows = []
    for m in MODELS:
        a = np.array(list(bs[m]["subject_f1"].values()))
        b = np.array([ps[m]["subject_f1"][s] for s in bs[m]["subject_f1"]])
        d = a - b
        w = stats.wilcoxon(a, b) if not np.allclose(a, b) else None
        dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
        pstr = f"p = {w.pvalue:.4g}" if w is not None else "p = n/a"
        print(f"  {m:4}  old {ps[m]['mean']*100:6.2f}   new {bs[m]['mean']*100:6.2f}   "
              f"delta {d.mean()*100:+6.2f} pp   {pstr}   d = {dz:+.2f}")
        rows.append({"tag": args.tag, "model": m, "window_ms": args.window_ms,
                     "sd_old_pooled": round(ps[m]["mean"], 4), "sd_new_blocked": round(bs[m]["mean"], 4),
                     "delta_pp": round(d.mean() * 100, 3),
                     "wilcoxon_p": (float(w.pvalue) if w else None), "cohens_d": round(float(dz), 3),
                     "n": bs[m]["n"], "guard_frac": round(guard_frac, 4),
                     "x_flags": len(bflags)})

    pd.DataFrame(rows).to_csv(out_dir / f"b8_{args.tag}_compare.csv", index=False)
    sw = pd.DataFrame({"subject": sorted(bs[MODELS[0]]["subject_f1"])})
    for m in MODELS:
        sw[f"{m}_old"] = [ps[m]["subject_f1"][s] for s in sw["subject"]]
        sw[f"{m}_new"] = [bs[m]["subject_f1"][s] for s in sw["subject"]]
    sw.round(5).to_csv(out_dir / f"b8_{args.tag}_subjectwise.csv", index=False)
    json.dump({"tag": args.tag, "window_ms": args.window_ms, "guard_frac": guard_frac,
               "x_flags": bflags, "old": {m: ps[m]["mean"] for m in MODELS},
               "new": {m: bs[m]["mean"] for m in MODELS}, "rows": rows},
              open(out_dir / f"b8_{args.tag}_outcome.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/b8_{args.tag}_*.csv / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

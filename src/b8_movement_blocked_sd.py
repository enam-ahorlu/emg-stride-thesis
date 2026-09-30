#!/usr/bin/env python3
"""
src/b8_movement_blocked_sd.py
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

  python src/b8_movement_blocked_sd.py --features <npz> --meta <csv> --models SVM,RF,LDA \
      --window-ms 250 --tag freq72_w250 --out results/b8_sd

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

ROOT = Path(__file__).resolve().parents[1]
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


def movement_blocked_folds(meta_s: pd.DataFrame, n_splits: int, win_len: float, guard_windows: float = 1.0):
    """meta_s carries a clean 0..n-1 RangeIndex. Returns (fold array of length n
    in that row order, n_dropped, n_total); fold == -1 marks a guard-band drop.

    guard_windows=1.0 (the default) is BYTE-IDENTICAL to the pre-KC23 function:
    the guard band is exactly one window length, as it always was. KC-C5's
    --guard-windows generalizes this to G window-lengths (the B-g arms sweep
    G in {1,2,4,8,16})."""
    fold = np.full(len(meta_s), -1, dtype=int)
    dropped = 0
    guard = guard_windows * win_len
    for _, g in meta_s.groupby("movement", sort=False):
        gi = g.sort_values("t_start", kind="stable")
        rows = gi.index.to_numpy()               # positions into meta_s (0..n-1)
        ts = gi["t_start"].to_numpy()
        chunks = np.array_split(np.arange(len(rows)), n_splits)
        bnds = [ts[chunks[k + 1][0]] for k in range(n_splits - 1) if len(chunks[k + 1])]
        for k, ch in enumerate(chunks):
            for local in ch:
                if any(abs(ts[local] - b) < guard for b in bnds):
                    dropped += 1                 # leave fold[rows[local]] == -1
                else:
                    fold[rows[local]] = k
    return fold, dropped, len(meta_s)


def pooled_random_nonoverlap_folds(meta_s: pd.DataFrame, n_splits: int, seed: int = SEED):
    """KC-C5 P0. Within each subject-by-movement recording, keep every SECOND
    window in time order (so no two retained windows overlap at 50% overlap),
    then split what remains at random (StratifiedKFold), same construction as
    the pooled_random scheme but on the non-overlapping subset."""
    keep_mask = np.zeros(len(meta_s), dtype=bool)
    for _, g in meta_s.groupby("movement", sort=False):
        gi = g.sort_values("t_start", kind="stable")
        rows = gi.index.to_numpy()
        keep_mask[rows[0::2]] = True
    fold = np.full(len(meta_s), -1, dtype=int)
    y_local = meta_s["movement"].map(LAB2I).to_numpy()
    idx_keep = np.where(keep_mask)[0]
    n_dropped = len(meta_s) - len(idx_keep)
    if len(idx_keep) >= n_splits and len(np.unique(y_local[idx_keep])) >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f, (_, te) in enumerate(skf.split(idx_keep, y_local[idx_keep])):
            fold[idx_keep[te]] = f
    return fold, n_dropped, len(meta_s)


def interleaved_folds(meta_s: pd.DataFrame, n_chunks: int, n_splits: int, win_len: float,
                      guard_windows: float):
    """KC-C5 I-g. Cut each movement's recording into n_chunks contiguous
    chunks (time order) and assign chunk i to fold i mod n_splits -- keeps the
    blocking (each fold still gets contiguous time, not randomly scattered
    windows) while removing most of the temporal extrapolation a single
    5-chunk block carries. Guard band applied only at boundaries between
    ADJACENT chunks that land in DIFFERENT folds."""
    fold = np.full(len(meta_s), -1, dtype=int)
    dropped = 0
    guard = guard_windows * win_len
    for _, g in meta_s.groupby("movement", sort=False):
        gi = g.sort_values("t_start", kind="stable")
        rows = gi.index.to_numpy()
        ts = gi["t_start"].to_numpy()
        chunks = np.array_split(np.arange(len(rows)), n_chunks)
        chunk_fold = [k % n_splits for k in range(n_chunks)]
        bnds = [ts[chunks[k + 1][0]] for k in range(n_chunks - 1)
               if len(chunks[k]) and len(chunks[k + 1]) and chunk_fold[k] != chunk_fold[k + 1]]
        for k, ch in enumerate(chunks):
            f = chunk_fold[k]
            for local in ch:
                if any(abs(ts[local] - b) < guard for b in bnds):
                    dropped += 1
                else:
                    fold[rows[local]] = f
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


def assign_folds(scheme: str, ms: pd.DataFrame, n_splits: int, win_len: float,
                 guard_windows: float, n_chunks: int, seed: int = SEED):
    """Dispatch to the fold-assignment function for one KC-C5 scheme, on one
    subject's own windows (ms carries a clean 0..n-1 RangeIndex)."""
    if scheme == "pooled_random":
        y_local = ms["movement"].map(LAB2I).to_numpy()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold = np.full(len(ms), -1, dtype=int)
        for f, (_, te) in enumerate(skf.split(np.zeros(len(ms)), y_local)):
            fold[te] = f
        return fold, 0, len(ms)
    if scheme == "pooled_random_nonoverlap":
        return pooled_random_nonoverlap_folds(ms, n_splits, seed=seed)
    if scheme == "blocked":
        return movement_blocked_folds(ms, n_splits, win_len, guard_windows=guard_windows)
    if scheme == "interleaved":
        return interleaved_folds(ms, n_chunks, n_splits, win_len, guard_windows)
    raise ValueError(f"unknown scheme {scheme!r}")


def eval_sd_v2(X, y, meta, scheme: str, n_splits: int, win_len: float, guard_windows: float,
              n_chunks: int, cv_unit: str, seed: int = SEED):
    """KC-C5: generalized evaluator over the four schemes and the two cv-units.

    cv_unit='per_subject' (the naming-control default, W-B1): fit ONE model
    PER SUBJECT PER FOLD, exactly as eval_sd() always did -- a genuinely
    per-subject model.
    cv_unit='pooled': fit ONE model PER FOLD, across ALL subjects' windows
    assigned to that fold (each subject's own fold assignment still comes
    from its own recording via assign_folds, so the blocking/guard/interleave
    structure is unchanged; only the classifier training pools across
    subjects) -- matching the "subject-inclusive" pooled-multi-subject
    construction the M8 kill-critic item describes for the ORIGINAL
    src/train_classical_patched.py SD figures, so the P50/P0/B-g/I-g arms are
    comparable to Table 4.2 rather than to eval_sd()'s already-per-subject
    default.
    """
    subs = sorted(meta["subject"].unique())
    x_flags = []
    guard_dropped_total = guard_total = 0

    # ---- per-subject fold assignment (identical regardless of cv_unit) ----
    fold_by_subject = {}
    for s in subs:
        ms = meta[meta["subject"] == s].reset_index(drop=True)
        fold, dr, tot = assign_folds(scheme, ms, n_splits, win_len, guard_windows, n_chunks, seed=seed)
        fold_by_subject[s] = fold
        guard_dropped_total += dr; guard_total += tot

    out = {m: {} for m in MODELS}

    if cv_unit == "per_subject":
        for s in subs:
            ms = meta[meta["subject"] == s]
            idx = ms.index.to_numpy()
            Xs, ys = X[idx], y[idx]
            fold = fold_by_subject[s]
            keep = fold >= 0
            for m in MODELS:
                f1s = []
                for f in range(n_splits):
                    te = keep & (fold == f); tr = keep & (fold != f)
                    if te.sum() == 0 or tr.sum() == 0:
                        x_flags.append((int(s), m, f, "empty fold")); continue
                    if len(np.unique(ys[tr])) < len(LABELS):
                        x_flags.append((int(s), m, f, f"train missing class(es): "
                                        f"{set(range(4)) - set(np.unique(ys[tr]).tolist())}"))
                    clf = build_model(m)
                    clf.fit(Xs[tr], ys[tr])
                    yp = clf.predict(Xs[te])
                    f1s.append(f1_score(ys[te], yp, average="macro"))
                out[m][int(s)] = f1s
    else:  # pooled: one model per fold, across all subjects
        # global row index -> (subject, local fold) for every row across all subjects
        subj_col = meta["subject"].to_numpy()
        fold_global = np.full(len(meta), -1, dtype=int)
        for s in subs:
            idx = meta.index[meta["subject"] == s].to_numpy()
            fold_global[idx] = fold_by_subject[s]
        keep_global = fold_global >= 0
        for m in MODELS:
            per_subject_f1s = {int(s): [] for s in subs}
            for f in range(n_splits):
                te = keep_global & (fold_global == f)
                tr = keep_global & (fold_global != f)
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                if len(np.unique(y[tr])) < len(LABELS):
                    x_flags.append(("ALL", m, f, f"pooled train missing class(es): "
                                    f"{set(range(4)) - set(np.unique(y[tr]).tolist())}"))
                clf = build_model(m)
                clf.fit(X[tr], y[tr])
                yp_all = clf.predict(X[te])
                te_subjects = subj_col[te]
                for s in np.unique(te_subjects):
                    m_s = te_subjects == s
                    if m_s.sum() == 0:
                        continue
                    per_subject_f1s[int(s)].append(
                        f1_score(y[te][m_s], yp_all[m_s], average="macro", zero_division=0))
            out[m] = per_subject_f1s

    return out, x_flags, (guard_dropped_total, guard_total)


def summarise(per_subject: dict) -> dict:
    res = {}
    for m, d in per_subject.items():
        subj_means = np.array([np.mean(v) for v in d.values() if len(v)])
        res[m] = {"subject_f1": {s: float(np.mean(v)) for s, v in d.items() if len(v)},
                  "mean": float(subj_means.mean()), "sd": float(subj_means.std(ddof=1)),
                  "n": int(len(subj_means))}
    return res


def run_single_scheme(X, y, meta, args, out_dir: Path, win_len: float) -> int:
    """KC-C5 new-scheme path (--scheme given). Computes ONE scheme's per-
    subject F1 (guard-windows / n-chunks / cv-unit respected) and writes it
    to files named by scheme + guard + cv-unit, distinct from the legacy
    b8_{tag}_* files so this path can never collide with or overwrite them."""
    out, flags, (gd, gt) = eval_sd_v2(X, y, meta, args.scheme, args.splits, win_len,
                                      args.guard_windows, args.n_chunks, args.cv_unit)
    summ = summarise(out)

    guard_frac = gd / gt if gt else 0.0
    print(f"\n[{args.tag}] scheme={args.scheme} cv_unit={args.cv_unit} "
          f"guard_windows={args.guard_windows} n_chunks={args.n_chunks}")
    print(f"[{args.tag}] guard band dropped {gd}/{gt} windows ({guard_frac:.2%})")
    if args.scheme in ("blocked", "interleaved") and guard_frac < args.min_guard_frac:
        print(f"[{args.tag}] ABORT: guard fraction {guard_frac:.2%} below the "
              f"{args.min_guard_frac:.2%} floor. Nothing written.")
        return 2
    if flags:
        print(f"[{args.tag}] OUTCOME-X flags ({len(flags)}): {flags[:8]}" + (" ..." if len(flags) > 8 else ""))
    else:
        print(f"[{args.tag}] no outcome-X flags")

    rows = []
    for m in MODELS:
        print(f"  {m:4}  mean {summ[m]['mean']*100:6.2f}  sd {summ[m]['sd']*100:6.2f}  n={summ[m]['n']}")
        rows.append({"tag": args.tag, "model": m, "window_ms": args.window_ms,
                     "scheme": args.scheme, "cv_unit": args.cv_unit,
                     "guard_windows": args.guard_windows, "n_chunks": args.n_chunks,
                     "sd_mean": round(summ[m]["mean"], 4), "sd_sd": round(summ[m]["sd"], 4),
                     "n": summ[m]["n"], "guard_frac": round(guard_frac, 4), "x_flags": len(flags)})

    stem = f"b8_{args.tag}_{args.scheme}_g{args.guard_windows:g}_{args.cv_unit}"
    pd.DataFrame(rows).to_csv(out_dir / f"{stem}_compare.csv", index=False)
    sw = pd.DataFrame({"subject": sorted(summ[MODELS[0]]["subject_f1"])})
    for m in MODELS:
        sw[m] = [summ[m]["subject_f1"].get(s, float("nan")) for s in sw["subject"]]
    sw.round(5).to_csv(out_dir / f"{stem}_subjectwise.csv", index=False)
    json.dump({"tag": args.tag, "scheme": args.scheme, "cv_unit": args.cv_unit,
               "guard_windows": args.guard_windows, "n_chunks": args.n_chunks,
               "window_ms": args.window_ms, "guard_frac": guard_frac, "x_flags": flags,
               "mean": {m: summ[m]["mean"] for m in MODELS}, "rows": rows},
              open(out_dir / f"{stem}_outcome.json", "w"), indent=2)
    print(f"\nwrote {out_dir}/{stem}_*.csv / .json")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--models", default="SVM,RF,LDA")
    ap.add_argument("--window-ms", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/b8_sd")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--time-units", choices=["seconds", "samples"], default="seconds",
                    help="units of the meta's t_start column. SIAT writes seconds; the ENABL3S "
                         "windower writes sample indices, and a guard band of 0.25 in a "
                         "sample-indexed axis silently drops nothing.")
    ap.add_argument("--min-guard-frac", type=float, default=0.0,
                    help="abort if the guard band drops less than this fraction. Set it on any "
                         "new dataset: a near-zero guard fraction means the units are wrong.")
    ap.add_argument("--scheme", default=None,
                    choices=["pooled_random", "pooled_random_nonoverlap", "blocked", "interleaved"],
                    help="KC-C5. Absent (default) reproduces the pre-KC23 output exactly: the "
                         "original pooled-vs-movement_blocked comparison, guard=1 window, "
                         "cv-unit=per_subject (this is the inertness gate). When set, runs ONE "
                         "scheme only (guard-windows/n-chunks/cv-unit apply), writing to new, "
                         "distinctly-named output files -- never the legacy b8_{tag}_* files.")
    ap.add_argument("--guard-windows", type=float, default=1.0,
                    help="KC-C5. Guard band in window-lengths, for --scheme blocked/interleaved "
                         "(default 1.0, matching the pre-KC23 fixed one-window guard).")
    ap.add_argument("--n-chunks", type=int, default=20,
                    help="KC-C5. Contiguous chunks per movement's recording, for "
                         "--scheme interleaved (default 20, per the plan).")
    ap.add_argument("--cv-unit", default="per_subject", choices=["pooled", "per_subject"],
                    help="KC-C5. per_subject (default) fits one model per subject per fold, "
                         "exactly as eval_sd() always did (the W-B1 naming-control arm). "
                         "pooled fits one model per fold across ALL subjects' windows assigned "
                         "to it, matching the ORIGINAL src/train_classical_patched.py SD figures' "
                         "subject-inclusive construction (M8).")
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

    if args.scheme is not None:
        return run_single_scheme(X, y, meta, args, out_dir, win_len)

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

#!/usr/bin/env python3
"""
kc23_c6_geometry.py
=====================
KC-C6 geometry producer (26 September 2026). kc23_c6_ladder_stats.py read
ladder_geometry.csv from the ladder job's directory; nothing wrote it (the ladder
runner's alignment_ladder_full.csv joins the PUBLISHED SIAT geometry, which says
nothing about ENABL3S). EXPERIMENT_PLAN_KC23_CLASSICAL.md KC-C6 and C2.3: "Run the
full ladder (rungs 0 to 4, plus 4lw and 4o) on the ENABL3S Freq features, using the
KC-C2 environment override. Record SVM LOSO F1 (n = 10), the geometry measures and
the probes. Chance for the subject probe is 1 in 10."

The SVM LOSO F1 per rung comes from run_alignment_ladder_loso.py. This script adds the
geometry rows, computed with the PUBLISHED metric code imported unchanged
(analyze_between_subject_variance: the rung operators, rbf_mmd2, median_heuristic_gamma,
stratified_subsample, load_data; run_alignment_ladder_loso.build_rungs_ext for 4lw and
4o; run_nonlinear_probe_ladder.PROBES/fit_probe; run_classcond_probe_ladder_v2.
size_matched_subsample). The feature files are pointed at ENABL3S through the same
LADDER_FEAT / LADDER_META environment override the KC-C2 rows use (set here from
--feat / --meta before those modules are imported).

Per rung (one row each, ladder_geometry.csv):
  mmd_mean, mmd_removed_pct          class-conditional pairwise RBF MMD, and the fraction removed against rung 0
  wasserstein1_mean, w1_removed_pct  per-feature Wasserstein-1 between subject pairs (every third pair), likewise
  subject_probe_linear / _forest / _mlp
                                     class-POOLED subject-identity probes, 5-fold balanced accuracy
  wm_probe_<probe>                   WITHIN-MOVEMENT probes (subject identity inside one class), mean over the classes
  sm_probe_<probe>                   the SIZE-MATCHED control: the same probe on a class-blind subsample of the same size
  silhouette_by_class                pooled class silhouette on the published stratified subsample
  silhouette_within_subject          class silhouette computed inside each subject, averaged over subjects
  chance_floor                       1 / number of subjects (0.1 for ENABL3S)
  oracle                             True on rung 4o (diagnostic only, never deployable)
The probes use a fixed cap of windows per subject and class (--cap) so the cost stays near the plan's hour on
CPU; the same subsample is used for every rung, so rungs are compared on identical points.
Exit 1 and no file on any input problem; --resume skips rungs already in the file.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUNGS = "0,1,2,3,4,4lw,4o"
PROBE_NAMES = ("linear", "forest", "mlp")


def _key(rid: str):
    return int(rid) if rid.isdigit() else rid


def fixed_subsample(y, subjects, cap, seed):
    rng = np.random.default_rng(seed)
    out = []
    for s in np.unique(subjects):
        for c in np.unique(y):
            m = np.where((subjects == s) & (y == c))[0]
            if len(m) > cap:
                m = rng.choice(m, cap, replace=False)
            out.append(m)
    return np.sort(np.concatenate(out))


def geometry_row(rid, name, Xr, y, subjects, probe_idx, sil_idx, fns, cap_mmd, n_draws, seed):
    stratified_subsample, rbf_mmd2, median_heuristic_gamma, LABELS, PROBES_fit, size_matched = fns
    subs_u = np.unique(subjects)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    gamma = median_heuristic_gamma(Xr)
    sub_idx = {}
    for s in subs_u:
        for c in range(len(LABELS)):
            m = np.where((subjects == s) & (y == c))[0]
            sub_idx[(s, c)] = rng.choice(m, cap_mmd, replace=False) if len(m) > cap_mmd else m
    mmd_per_class = []
    for c in range(len(LABELS)):
        samples = {s: Xr[sub_idx[(s, c)]] for s in subs_u if len(sub_idx[(s, c)]) >= 3}
        keys = list(samples)
        vals = [rbf_mmd2(samples[keys[i]], samples[keys[j]], gamma)
                for i in range(len(keys)) for j in range(i + 1, len(keys))]
        if vals:
            mmd_per_class.append(np.mean(vals))
    if not mmd_per_class:
        raise ValueError("no class has two subjects with enough windows for an MMD")
    mmd = float(np.mean(mmd_per_class))

    from scipy.stats import wasserstein_distance
    w1_vals, pair = [], 0
    by_s = {s: Xr[subjects == s] for s in subs_u}
    for i in range(len(subs_u)):
        for j in range(i + 1, len(subs_u)):
            pair += 1
            if pair % 3 != 0:
                continue
            a, b = by_s[subs_u[i]], by_s[subs_u[j]]
            w1_vals.append(np.mean([wasserstein_distance(a[:, f], b[:, f]) for f in range(Xr.shape[1])]))
    w1 = float(np.mean(w1_vals)) if w1_vals else float("nan")

    Xp, yp, sp = Xr[probe_idx], y[probe_idx], subjects[probe_idx]
    row = {"rung": rid, "name": name, "oracle": rid == "4o", "mmd_mean": mmd, "wasserstein1_mean": w1,
           "chance_floor": 1.0 / len(subs_u)}
    for p in PROBE_NAMES:
        row[f"subject_probe_{p}"] = float(PROBES_fit(p, Xp, sp).mean())
        wm, sm = [], []
        for c in np.unique(yp):
            mc = np.where(yp == c)[0]
            if len(np.unique(sp[mc])) < 2 or len(mc) < 10:
                continue
            wm.append(float(PROBES_fit(p, Xp[mc], sp[mc]).mean()))
            draws = []
            for d in range(n_draws):
                sel = size_matched(sp, len(mc), seed + 100 * d + int(c))
                draws.append(float(PROBES_fit(p, Xp[sel], sp[sel]).mean()))
            sm.append(float(np.mean(draws)))
        row[f"wm_probe_{p}"] = float(np.mean(wm)) if wm else float("nan")
        row[f"sm_probe_{p}"] = float(np.mean(sm)) if sm else float("nan")

    from sklearn.metrics import silhouette_score
    row["silhouette_by_class"] = float(silhouette_score(Xr[sil_idx], y[sil_idx]))
    sil_s = []
    for s in subs_u:
        ms = np.where(subjects == s)[0]
        if len(ms) > 600:
            ms = np.random.default_rng(seed + int(s)).choice(ms, 600, replace=False)
        if len(np.unique(y[ms])) >= 2:
            sil_s.append(float(silhouette_score(Xr[ms], y[ms])))
    row["silhouette_within_subject"] = float(np.mean(sil_s)) if sil_s else float("nan")
    row["elapsed_sec"] = round(time.time() - t0, 1)
    return row


def run(out_dir: Path, feat: str, meta: str, rungs: list[str], cap: int, cap_mmd: int, n_draws: int, resume: bool) -> int:
    os.environ["LADDER_FEAT"], os.environ["LADDER_META"] = str(feat), str(meta)   # BEFORE the published modules import
    try:
        for p in (feat, meta):
            if not Path(p).exists():
                raise FileNotFoundError(f"missing input: {p}")
        from analyze_between_subject_variance import (load_data, SEED, LABELS, stratified_subsample, rbf_mmd2,
                                                     median_heuristic_gamma)
        from run_alignment_ladder_loso import build_rungs_ext
        from run_nonlinear_probe_ladder import fit_probe
        from run_classcond_probe_ladder_v2 import size_matched_subsample
        X, y, subjects, _ = load_data()
        if len(np.unique(subjects)) < 3:
            raise ValueError("fewer than 3 subjects")
        rungs_dict = build_rungs_ext(y)
        unknown = [r for r in rungs if _key(r) not in rungs_dict]
        if unknown:
            raise ValueError(f"unknown rung ids {unknown}")
    except (FileNotFoundError, ValueError, KeyError, ImportError) as e:
        print(f"[C6-geometry] FAIL, no output written: {e}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ladder_geometry.csv"
    done = set()
    if resume and path.exists():
        done = set(pd.read_csv(path)["rung"].astype(str))
    elif path.exists():
        path.unlink()
    probe_idx = fixed_subsample(y, subjects, cap, SEED)
    sil_idx = stratified_subsample(y, subjects)
    fns = (stratified_subsample, rbf_mmd2, median_heuristic_gamma, LABELS, fit_probe, size_matched_subsample)
    try:
        for rid in rungs:
            if rid in done:
                print(f"[C6-geometry] rung {rid}: already in {path.name}, skipped (--resume)")
                continue
            name, fn, needs_subjects = rungs_dict[_key(rid)]
            Xr = fn(X, subjects) if needs_subjects else fn(X)
            row = geometry_row(rid, name, np.asarray(Xr, float), y, subjects, probe_idx, sil_idx, fns, cap_mmd,
                               n_draws, SEED)
            pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
            print(f"[C6-geometry] rung {rid} ({name}): linear probe {row['subject_probe_linear']:.3f}, "
                  f"MMD {row['mmd_mean']:.4f}, silhouette {row['silhouette_by_class']:.3f} ({row['elapsed_sec']}s)", flush=True)
    except (ValueError, FloatingPointError) as e:
        print(f"[C6-geometry] FAIL: {e}", file=sys.stderr)
        path.unlink(missing_ok=True)
        return 1

    g = pd.read_csv(path)
    g["rung"] = g["rung"].astype(str)
    if "0" not in set(g["rung"]):
        print("[C6-geometry] FAIL: rung 0 is needed for the removed-percent columns", file=sys.stderr)
        path.unlink(missing_ok=True)
        return 1
    r0 = g[g["rung"] == "0"].iloc[0]
    g["mmd_removed_pct"] = (1 - g["mmd_mean"] / r0["mmd_mean"]) * 100.0
    g["w1_removed_pct"] = (1 - g["wasserstein1_mean"] / r0["wasserstein1_mean"]) * 100.0
    if set(g["rung"]) != set(rungs) or len(g) != len(rungs) or g.isna().any().any():
        print(f"[C6-geometry] FAIL: rows {sorted(set(g['rung']))} differ from the requested {rungs}, or hold NaN",
              file=sys.stderr)
        path.unlink(missing_ok=True)
        return 1
    g.to_csv(path, index=False)
    print(f"[C6-geometry] wrote {path} ({len(g)} rungs)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feat", required=True, help="the ENABL3S Freq-72 features npz (the LADDER_FEAT override)")
    ap.add_argument("--meta", required=True, help="the matching meta csv (the LADDER_META override)")
    ap.add_argument("--rungs", default=DEFAULT_RUNGS)
    ap.add_argument("--cap", type=int, default=300, help="windows per subject and class for the probes")
    ap.add_argument("--cap-mmd", type=int, default=100)
    ap.add_argument("--n-draws", type=int, default=3, help="size-matched control draws")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    sys.exit(run(Path(args.out), args.feat, args.meta, [r.strip() for r in args.rungs.split(",") if r.strip()],
                 args.cap, args.cap_mmd, args.n_draws, args.resume))


if __name__ == "__main__":
    main()

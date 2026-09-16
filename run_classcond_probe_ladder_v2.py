#!/usr/bin/env python3
"""
run_classcond_probe_ladder_v2.py
=================================
Same computation as run_classcond_probe_ladder.py (EXPERIMENT_PLAN_PROBE_CLASSCOND.md),
rewritten for visibility and speed while the first run is already in flight:
  - unbuffered/line-buffered, timestamped logging (the original's prints were fully
    buffered and invisible until exit)
  - per-cell checkpointing: every (rung, probe, class[, draw]) result is appended to
    a checkpoint CSV the moment it's computed, and a restart skips whatever's already
    there (--resume-style, matching train_classical_loso.py's own convention)
  - the outer (rung, probe, class, draw) grid -- 60 Phase-1 cells + 300 Phase-2 cells,
    360 total -- is embarrassingly parallel and is farmed out across worker processes
    (ProcessPoolExecutor) instead of one process working the grid serially, since the
    CPU-bound estimator fits (LogisticRegression/RandomForest/MLPClassifier) release
    the GIL during their own C/Cython work but Python itself never overlaps two fits
    in one process.

Imports RUNGS/load_data/SEED/LABELS from analyze_between_subject_variance.py and
PROBES/fit_probe from run_nonlinear_probe_ladder.py UNCHANGED -- identical transforms
and identical estimator constructors to the original run, so the two are directly
comparable (and the original, once it finishes, is a free cross-check on this one).

Outputs (distinct filenames so this never collides with the original, currently-
running v1 process):
  results_nonlinear_probe/classcond_v2_phase1_ckpt.csv   (per-cell, resumable)
  results_nonlinear_probe/classcond_v2_phase2_ckpt.csv   (per-cell, resumable)
  results_nonlinear_probe/classcond_probe_ladder_v2.csv  (final, pooled)
  results_nonlinear_probe/classcond_size_control_v2.csv  (final, pooled)
  results_nonlinear_probe/subject_identifiability_vs_f1_v2.csv
  report_figs/new_experiments/classcond_probe_ladder.png (same target as v1's; whichever
                                                            finishes last writes it)
"""
from __future__ import annotations
import sys, os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from analyze_between_subject_variance import RUNGS, load_data, SEED, LABELS
from run_nonlinear_probe_ladder import PROBES, fit_probe

ROOT = Path(__file__).parent
OUT = ROOT / "results_nonlinear_probe"; OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)
PUBLISHED = OUT / "nonlinear_probe_ladder.csv"
DIST_CORR = ROOT / "results_variance_decomposition" / "distance_vs_f1_correlations.csv"

TOL = 0.005
N_DRAWS = 5
CORR_RUNGS = [0, 3]
MIN_CELL_WARN = 30
CKPT1 = OUT / "classcond_v2_phase1_ckpt.csv"
CKPT2 = OUT / "classcond_v2_phase2_ckpt.csv"

# 3 outer worker processes: RF's own estimator (imported unchanged from
# run_nonlinear_probe_ladder.PROBES) already uses n_jobs=4 internally during a
# forest fit, so 3 workers x up to 4 RF threads = up to 12 cores at a peak
# overlap, leaving headroom against this machine's 16 logical cores alongside
# whatever the original (v1) run is still using. Increase if you want more.
N_PARALLEL = int(os.environ.get("CLASSCOND_N_PARALLEL", "2"))
MAX_TASK_RETRIES = 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def size_matched_subsample(subjects, n_total, seed):
    rng = np.random.default_rng(seed)
    subs_u = np.unique(subjects)
    n_subj = len(subs_u)
    base = n_total // n_subj
    rem = n_total - base * n_subj
    order = rng.permutation(subs_u)
    idx_out = []
    for i, s in enumerate(order):
        take = base + (1 if i < rem else 0)
        pool = np.where(subjects == s)[0]
        take = min(take, len(pool))
        if take > 0:
            idx_out.append(rng.choice(pool, take, replace=False))
    return np.concatenate(idx_out)


# ----------------------------------------------------------------------------
# worker-process state: each worker loads the data and derives all 5 rungs'
# Xr ONCE at startup (via the pool's `initializer=`), instead of every task
# re-pickling a ~15 MB array across the process boundary.
_W = {}


def _worker_init():
    X, y, subjects, meta = load_data()
    rung_Xr = {}
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        rung_Xr[rung_id] = fn(X, subjects) if needs_subjects else fn(X)
    _W["y"] = y
    _W["subjects"] = subjects
    _W["rung_Xr"] = rung_Xr


def _task_phase0(rung_id, probe):
    Xr, subjects = _W["rung_Xr"][rung_id], _W["subjects"]
    scores = fit_probe(probe, Xr, subjects)
    return dict(rung=rung_id, probe=probe, value=float(scores.mean()))


def _task_phase1(rung_id, probe, class_idx):
    Xr, y, subjects = _W["rung_Xr"][rung_id], _W["y"], _W["subjects"]
    mask = y == class_idx
    try:
        scores = fit_probe(probe, Xr[mask], subjects[mask])
        v = float(scores.mean())
    except ValueError as e:
        v = float("nan")
    return dict(rung=rung_id, probe=probe, class_idx=class_idx, value=v)


def _task_phase2(rung_id, probe, class_idx, draw, n_target):
    Xr, subjects = _W["rung_Xr"][rung_id], _W["subjects"]
    idx = size_matched_subsample(subjects, n_target, seed=SEED + draw)
    try:
        scores = fit_probe(probe, Xr[idx], subjects[idx])
        v = float(scores.mean())
    except ValueError:
        v = float("nan")
    return dict(rung=rung_id, probe=probe, class_idx=class_idx, draw=draw, value=v)


def load_ckpt(path, key_cols):
    if path.exists():
        df = pd.read_csv(path)
        done = set(tuple(r) for r in df[key_cols].itertuples(index=False, name=None))
        return df, done
    return pd.DataFrame(), set()


def append_row(path, row, header_cols):
    is_new = not path.exists()
    pd.DataFrame([row])[header_cols].to_csv(path, mode="a", header=is_new, index=False)


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    log(f"N_PARALLEL={N_PARALLEL} (set CLASSCOND_N_PARALLEL env var to change)")
    X, y, subjects, meta = load_data()
    log(f"[data] X={X.shape} subjects={len(np.unique(subjects))} classes={len(LABELS)}")

    rung_Xr = {}
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        rung_Xr[rung_id] = fn(X, subjects) if needs_subjects else fn(X)

    ct = pd.crosstab(subjects, y)
    min_cell = int(ct.values.min())
    log(f"min subject-by-class window count = {min_cell} (warn threshold {MIN_CELL_WARN})")
    class_sizes = {c: int((y == c).sum()) for c in range(len(LABELS))}
    log(f"class sizes: { {LABELS[c]: n for c, n in class_sizes.items()} }")

    # ---------------- Phase 0: reproduction gate (parallel, 15 cells) ----------------
    pub = pd.read_csv(PUBLISHED)
    p0_items = [(r, p) for r in RUNGS for p in ("linear", "forest", "mlp")]
    p0_results = {}
    log(f"PHASE 0: {len(p0_items)} cells (reproduction gate)")
    with ProcessPoolExecutor(max_workers=N_PARALLEL, initializer=_worker_init) as ex:
        futs = {ex.submit(_task_phase0, r, p): (r, p) for r, p in p0_items}
        for fut in as_completed(futs):
            r, p = futs[fut]
            try:
                res = fut.result()
                p0_results[(res["rung"], res["probe"])] = res["value"]
                log(f"  phase0 rung={res['rung']} probe={res['probe']}: {res['value']:.4f}")
            except Exception as e:
                log(f"  *** phase0 rung={r} probe={p} FAILED: {type(e).__name__}: {e} ***")
    if len(p0_results) < len(p0_items):
        log(f"*** PHASE 0 incomplete ({len(p0_results)}/{len(p0_items)} cells) after worker failures -- "
            f"re-run the script to retry the missing cells (Phase 0 has no checkpoint, it's cheap). ***")
        return

    ok = True
    for rung_id, probe in p0_items:
        mean = p0_results[(rung_id, probe)]
        pv = float(pub[(pub.rung == rung_id) & (pub.probe == probe)].bal_acc_mean.values[0])
        hit = abs(mean - pv) <= TOL
        ok &= hit
        if not hit:
            log(f"  *** MISS rung={rung_id} probe={probe}: reproduced={mean:.4f} published={pv:.4f} ***")
    if not ok:
        log("*** PHASE 0 GATE FAILED -- stopping. ***")
        return
    log("PHASE 0 GATE: PASS (all 15 values within +-0.005).")

    # ---------------- Phase 1 + Phase 2 work items (independent, dispatched together) ----------------
    p1_done_df, p1_done = load_ckpt(CKPT1, ["rung", "probe", "class_idx"])
    p2_done_df, p2_done = load_ckpt(CKPT2, ["rung", "probe", "class_idx", "draw"])
    if len(p1_done) or len(p2_done):
        log(f"RESUME: {len(p1_done)} Phase-1 cells and {len(p2_done)} Phase-2 cells already checkpointed, skipping.")

    p1_items = [(r, p, c) for r in RUNGS for p in ("linear", "forest", "mlp") for c in range(len(LABELS))
                if (r, p, c) not in p1_done]
    p2_items = [(r, p, c, d) for r in RUNGS for p in ("linear", "forest", "mlp") for c in range(len(LABELS))
                for d in range(N_DRAWS) if (r, p, c, d) not in p2_done]
    log(f"PHASE 1+2: {len(p1_items)} Phase-1 cells + {len(p2_items)} Phase-2 cells remaining "
        f"(of 60 + 300 total), dispatching across {N_PARALLEL} workers")

    with ProcessPoolExecutor(max_workers=N_PARALLEL, initializer=_worker_init) as ex:
        futs = {}
        for r, p, c in p1_items:
            futs[ex.submit(_task_phase1, r, p, c)] = ("p1", r, p, c, None)
        for r, p, c, d in p2_items:
            futs[ex.submit(_task_phase2, r, p, c, d, class_sizes[c])] = ("p2", r, p, c, d)

        n_done = 0
        n_failed = 0
        n_total = len(futs)
        failed_items = []
        for fut in as_completed(futs):
            kind, r, p, c, d = futs[fut]
            n_done += 1
            try:
                res = fut.result()
            except Exception as e:
                n_failed += 1
                failed_items.append((kind, r, p, c, d))
                log(f"  [{n_done}/{n_total}] *** {kind} rung={r} probe={p} class={LABELS[c]} "
                    f"{'draw='+str(d)+' ' if d is not None else ''}FAILED: {type(e).__name__}: {e} "
                    f"(not checkpointed -- a re-run of this script will retry it) ***")
                continue
            if kind == "p1":
                row = dict(rung=r, name=RUNGS[r][0], probe=p, class_idx=c, class_name=LABELS[c], value=res["value"])
                append_row(CKPT1, row, ["rung", "name", "probe", "class_idx", "class_name", "value"])
                log(f"  [{n_done}/{n_total}] phase1 rung={r} probe={p:7s} class={LABELS[c]:6s}: {res['value']:.4f}")
            else:
                row = dict(rung=r, name=RUNGS[r][0], probe=p, class_idx=c, class_name=LABELS[c],
                          draw=d, value=res["value"])
                append_row(CKPT2, row, ["rung", "name", "probe", "class_idx", "class_name", "draw", "value"])
                log(f"  [{n_done}/{n_total}] phase2 rung={r} probe={p:7s} class={LABELS[c]:6s} draw={d}: {res['value']:.4f}")

    log(f"Phase 1+2 grid pass complete: {n_done - n_failed}/{n_total} succeeded, {n_failed} failed "
        f"(uncheckpointed, retryable by re-running this script).")
    if n_failed:
        log(f"  failed items: {failed_items}")

    # ---------------- pool and merge ----------------
    p1 = pd.read_csv(CKPT1)
    p2 = pd.read_csv(CKPT2)
    p1_pooled = p1.groupby(["rung", "name", "probe"])["value"].mean().reset_index().rename(columns={"value": "classcond_pooled"})
    p1_wide = p1.pivot_table(index=["rung", "name", "probe"], columns="class_name", values="value").reset_index()
    p1_wide.columns = [f"classcond_{c}" if c in LABELS else c for c in p1_wide.columns]
    p1_final = p1_pooled.merge(p1_wide, on=["rung", "name", "probe"])

    p2_class_mean = p2.groupby(["rung", "name", "probe", "class_name"])["value"].mean().reset_index()
    p2_pooled = p2_class_mean.groupby(["rung", "name", "probe"])["value"].mean().reset_index().rename(columns={"value": "control_pooled"})
    p2_wide = p2_class_mean.pivot_table(index=["rung", "name", "probe"], columns="class_name", values="value").reset_index()
    p2_wide.columns = [f"control_{c}" if c in LABELS else c for c in p2_wide.columns]
    p2_final = p2_pooled.merge(p2_wide, on=["rung", "name", "probe"])

    merged = p1_final.merge(p2_final, on=["rung", "name", "probe"])
    merged["classcond_minus_control_pp"] = (merged["classcond_pooled"] - merged["control_pooled"]) * 100
    merged.to_csv(OUT / "classcond_probe_ladder_v2.csv", index=False)
    p2_final.to_csv(OUT / "classcond_size_control_v2.csv", index=False)
    log(f"[save] {OUT / 'classcond_probe_ladder_v2.csv'}")
    log(f"[save] {OUT / 'classcond_size_control_v2.csv'}")
    print(merged.to_string(index=False))

    # ---------------- Phase 3: subject recall vs LOSO F1 (cheap, sequential) ----------------
    log("PHASE 3: subject-identifiability vs LOSO F1")
    svm_f1 = pd.read_csv(ROOT / "results_loso_freq_persubj" /
                          "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__SVM_nested_loso_subjectwise.csv")
    rf_f1 = pd.read_csv(ROOT / "results_loso_freq_persubj" /
                         "freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__RF_nested_loso_subjectwise.csv")
    f1_map = {"SVM": dict(zip(svm_f1.heldout_subject, svm_f1.f1_macro)),
              "RF": dict(zip(rf_f1.heldout_subject, rf_f1.f1_macro))}
    cd_path = ROOT / "results_cnn_aug_resnet_se_chandrop_proba" / "cnn_arch_subjectwise.csv"
    if cd_path.exists():
        cd_f1 = pd.read_csv(cd_path)
        f1_map["ResNet_SE_CD"] = dict(zip(cd_f1.subject, cd_f1.f1_macro))

    subs_u = np.unique(subjects)
    rows = []
    for rung_id in CORR_RUNGS:
        Xr = rung_Xr[rung_id]
        clf = PROBES["forest"]()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        y_pred = cross_val_predict(clf, Xr, subjects, cv=skf, n_jobs=1)
        recall = {int(s): float(np.mean(y_pred[subjects == s] == s)) for s in subs_u}
        for model, m in f1_map.items():
            xs, ys = [], []
            for s in subs_u:
                if int(s) in recall and s in m:
                    xs.append(recall[int(s)]); ys.append(m[s])
            if len(xs) < 3:
                continue
            rho, p = spearmanr(xs, ys)
            rows.append(dict(rung=rung_id, name=RUNGS[rung_id][0], model=model, probe="forest",
                             spearman_rho=round(float(rho), 4), p=float(p), n=len(xs)))
            log(f"  rung {rung_id} forest-recall vs {model} LOSO F1: rho={rho:+.3f} p={p:.3f} n={len(xs)}")
    pd.DataFrame(rows).to_csv(OUT / "subject_identifiability_vs_f1_v2.csv", index=False)
    log(f"[save] {OUT / 'subject_identifiability_vs_f1_v2.csv'}")
    if DIST_CORR.exists():
        print(pd.read_csv(DIST_CORR).to_string(index=False))

    log("PHASE 4: class-conditional Wasserstein -- SKIPPED (same rationale as v1: Phase 1 already ran "
        "long, Phase 4 'changes no conclusion' per the plan text).")

    # ---------------- Phase 5: figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})
    rungs_sorted = sorted(RUNGS.keys())
    names = [RUNGS[r][0] for r in rungs_sorted]
    colors = {"linear": "#4C72B0", "forest": "#55A868", "mlp": "#c44e52"}
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(rungs_sorted))
    for probe in ("linear", "forest", "mlp"):
        sub = merged[merged.probe == probe].set_index("rung").reindex(rungs_sorted)
        ax.plot(x, sub["classcond_pooled"], marker="o", ms=7, lw=2, color=colors[probe],
               label=f"{probe} class-conditional")
        ax.plot(x, sub["control_pooled"], marker="s", ms=6, lw=1.6, ls="--", color=colors[probe],
               alpha=0.75, label=f"{probe} size-matched control")
    ax.axhline(0.025, ls=":", lw=1.3, color="#555", alpha=0.8, label="chance floor (1/40)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("subject-identity probe balanced accuracy")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    ax.set_title("Class-conditional subject probe vs a size-matched class-pooled control", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "classcond_probe_ladder.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    log(f"[save] {FIGDIR / 'classcond_probe_ladder.png'}")

    # ---------------- verdict ----------------
    row = merged[(merged.probe == "forest") & (merged.rung == 3)].iloc[0]
    cc, ctrl, delta_pp = row.classcond_pooled, row.control_pooled, row.classcond_minus_control_pp
    log(f"forest @ rung3: class-conditional={cc:.4f}  size-matched control={ctrl:.4f}  delta={delta_pp:+.2f} pp")
    if cc >= 0.9:
        verdict, text = "A", "class-conditional forest stays high (>=~0.9) at rung 3 -- qualification closes with a number."
    elif abs(delta_pp) <= 5.0:
        verdict, text = "B", f"drop ({cc:.4f}) matched by the size-matched control ({ctrl:.4f}) -- sample-size artifact, conclusion unchanged."
    else:
        verdict, text = "C", f"class-conditional ({cc:.4f}) sits well below control ({ctrl:.4f}) -- class composition carried real weight."
    log(f"OUTCOME {verdict}: {text}")
    log("New BH family members added: 0 (descriptive intervals only, no per-subject paired tests here).")
    log(f"[DONE] total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

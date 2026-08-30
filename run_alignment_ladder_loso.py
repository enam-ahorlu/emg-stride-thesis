#!/usr/bin/env python3
"""
run_alignment_ladder_loso.py
============================
EXPERIMENT_PLAN_CAUSALITY.md E-C1 -- downstream LOSO F1 for every rung of the
alignment ladder, on the SAME manipulated features the published subject-probe
and class-silhouette were computed on.

Section 4.13.2 currently compares a probe/silhouette measured on the rung
operators against an F1 (72.4%) that is actually CORAL's LOSO F1 from a
different script. This makes the alignment ladder a single-axis intervention:
alignment strength, subject identity, class separability and downstream F1 are
all read off the same variable, with a measured dose-response.

Procedure (mirrors train_classical_loso.py's classical LOSO SVM):
  1. Import the rung functions from analyze_between_subject_variance.py
     (rung0_global_z .. rung4_full_whiten_recolor). Not reimplemented.
  2. Per rung: Xr = rung(X, subjects) once over the full 26347x72 matrix.
  3. Standard LOSO over 40 subjects on Xr, FULL nested GridSearchCV:
     inner 5-fold GroupKFold on training subjects, grid clf__C in {1,5,10},
     clf__gamma='scale', SVC(kernel='rbf', class_weight='balanced'),
     scoring f1_macro, refit=True, n_jobs=1 (Windows loky deadlock).
  4. Re-tune per rung. _bestparams.json is NOT reused (those C values were
     selected on rung-3 geometry).
  5. Per-subject F1 -> mean, sd per rung.
  6. Paired Wilcoxon + Cohen's d: rung 4 vs rung 3, rung 3 vs rung 0.

No scaler in the pipeline: the rung transform IS the normalization (this is
what makes rung 3 == the published --norm-mode per_subject run, gate 0.7767).

Validation gates (both mandatory; stop and report if either fails):
  - rung 3 (per_subject_zscore) must reproduce the published per-subject
    z-score SVM F1, 0.7767, to within 0.002.
  - rung 0 (global_z) must land near the published global baseline, 0.708.
    Will not match exactly: the published baseline fits StandardScaler on the
    39 training subjects inside the Pipeline; rung0_global_z standardizes over
    all 40 pooled. Report the value and the gap; a gap under ~0.005 is
    expected.

Falsifiable expected shape: F1 rises rung 0 -> rung 3, then FALLS at rung 4
(tracking the silhouette, not the monotonically-falling probe). If rung-4 F1
comes out at or above rung 3, the over-alignment account as written is wrong
and the thesis must change -- report it and stop, do not tune.

Checkpoint per subject, append immediately, --resume supported.
Outputs (results_alignment_ladder_loso/):
  ladder_loso_{rung}_SVM_subjectwise.csv   per rung
  alignment_ladder_loso_summary.csv        rung, name, f1_mean, f1_sd, n
  alignment_ladder_loso_stats.csv          paired Wilcoxon + Cohen's d
  alignment_ladder_full.csv                joined with the published ladder
                                           (new Table 4.16)
"""
from __future__ import annotations
import argparse
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score

from analyze_between_subject_variance import load_data, RUNGS

ROOT = Path(__file__).parent
PUBLISHED_LADDER = ROOT / "results_variance_decomposition" / "alignment_ladder.csv"

GATE_RUNG3_F1 = 0.7767
GATE_RUNG3_TOL = 0.002
GATE_RUNG0_F1 = 0.708


def cohens_d_paired(a, b):
    """Paired Cohen's d = mean(diff) / sd(diff, ddof=1)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def run_rung(rung_id, X, y, subjects, out_dir, seed, inner_splits, verbose, subj_filter=None):
    name, fn, needs_subjects = RUNGS[rung_id]
    csv_path = out_dir / f"ladder_loso_{rung_id}_SVM_subjectwise.csv"

    done = set()
    if csv_path.exists():
        done = set(pd.read_csv(csv_path)["subject"].astype(int).tolist())
        if done:
            print(f"[resume] rung {rung_id} ({name}): {len(done)} subjects already done")

    subjects_u = sorted(np.unique(subjects).tolist())
    if subj_filter is not None:
        subjects_u = [s for s in subjects_u if s in subj_filter]
        print(f"[rung {rung_id}] --subjects filter active: {subjects_u}")
    if all(s in done for s in subjects_u):
        print(f"[rung {rung_id}] all 40 subjects present, skipping compute")
        return csv_path

    t_rung = time.time()
    print(f"\n[rung {rung_id}] building Xr = {name}(X, subjects) over full matrix ...")
    Xr = fn(X, subjects) if needs_subjects else fn(X)
    Xr = np.ascontiguousarray(Xr, dtype=np.float64)
    print(f"[rung {rung_id}] Xr={Xr.shape}  pooled per-feature var: "
          f"min={Xr.var(0).min():.3e} med={np.median(Xr.var(0)):.3e} max={Xr.var(0).max():.3e}")

    for heldout in subjects_u:
        if heldout in done:
            continue
        te = subjects == heldout
        tr = ~te
        Xtr, ytr, gtr = Xr[tr], y[tr], subjects[tr]
        Xte, yte = Xr[te], y[te]

        n_train_groups = len(np.unique(gtr))
        inner_cv = GroupKFold(n_splits=min(inner_splits, n_train_groups))

        pipe = Pipeline([("clf", SVC(kernel="rbf", class_weight="balanced", cache_size=500))])
        grid = {"clf__C": [1, 5, 10], "clf__gamma": ["scale"]}
        search = GridSearchCV(
            pipe, param_grid=grid, scoring="f1_macro",
            cv=list(inner_cv.split(Xtr, ytr, groups=gtr)),
            n_jobs=1, refit=True, verbose=verbose,
        )
        t0 = time.perf_counter()
        search.fit(Xtr, ytr)
        t1 = time.perf_counter()
        yhat = search.best_estimator_.predict(Xte)

        row = {
            "rung": rung_id,
            "name": name,
            "subject": int(heldout),
            "n_test": int(te.sum()),
            "f1_macro": float(f1_score(yte, yhat, average="macro", zero_division=0)),
            "bal_acc": float(balanced_accuracy_score(yte, yhat)),
            "acc": float(accuracy_score(yte, yhat)),
            "best_params": str(search.best_params_),
            "fit_time_sec": round(t1 - t0, 2),
        }
        # CHECKPOINT: append immediately (crash-safe / resumable)
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        done.add(heldout)
        print(f"  [rung {rung_id}] Sub{heldout:02d}  f1={row['f1_macro']:.4f}  "
              f"C={ast.literal_eval(row['best_params'])['clf__C']}  ({row['fit_time_sec']:.0f}s)")

    print(f"[rung {rung_id}] done in {time.time()-t_rung:.0f}s")
    return csv_path


def summarise(rung_ids, out_dir):
    rows, per_subject = [], {}
    for rid in rung_ids:
        p = out_dir / f"ladder_loso_{rid}_SVM_subjectwise.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p).drop_duplicates("subject").sort_values("subject")
        rows.append({
            "rung": rid, "name": RUNGS[rid][0],
            "f1_mean": round(float(d["f1_macro"].mean()), 6),
            "f1_sd": round(float(d["f1_macro"].std(ddof=1)), 6),
            "n": int(len(d)),
        })
        per_subject[rid] = d.set_index("subject")["f1_macro"]
    summ = pd.DataFrame(rows).sort_values("rung")
    summ.to_csv(out_dir / "alignment_ladder_loso_summary.csv", index=False)
    print("\n================  E-C1 LADDER LOSO SUMMARY  ================")
    print(summ.to_string(index=False))

    # paired stats
    stat_rows = []
    for hi, lo in [(4, 3), (3, 0)]:
        if hi in per_subject and lo in per_subject:
            j = pd.concat([per_subject[hi].rename("hi"), per_subject[lo].rename("lo")], axis=1).dropna()
            if len(j) >= 2 and np.any(j["hi"].values != j["lo"].values):
                w = wilcoxon(j["hi"].values, j["lo"].values)
                stat_rows.append({
                    "comparison": f"rung{hi}_vs_rung{lo}",
                    "n": int(len(j)),
                    "mean_diff": round(float((j["hi"] - j["lo"]).mean()), 6),
                    "wilcoxon_W": float(w.statistic),
                    "wilcoxon_p": float(w.pvalue),
                    "cohens_d_paired": round(cohens_d_paired(j["hi"].values, j["lo"].values), 4),
                })
    if stat_rows:
        st = pd.DataFrame(stat_rows)
        st.to_csv(out_dir / "alignment_ladder_loso_stats.csv", index=False)
        print("\n--- paired tests ---")
        print(st.to_string(index=False))
    return summ


def join_published(summ, out_dir):
    if not PUBLISHED_LADDER.exists():
        print(f"[warn] {PUBLISHED_LADDER} missing; skipping alignment_ladder_full.csv")
        return
    pub = pd.read_csv(PUBLISHED_LADDER)
    keep = pub[["rung", "name", "mmd_removed_pct", "w1_removed_pct",
                "subject_probe_bal_acc", "silhouette_by_class"]].copy()
    merged = keep.merge(summ[["rung", "f1_mean"]].rename(columns={"f1_mean": "f1_macro_mean"}),
                        on="rung", how="left")
    merged.to_csv(out_dir / "alignment_ladder_full.csv", index=False)
    print("\n================  alignment_ladder_full.csv (new Table 4.16)  ================")
    print(merged.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results_alignment_ladder_loso")
    ap.add_argument("--rungs", default="3,0,1,2,4",
                    help="comma list; gate rungs (3 then 0) first so a failure stops early")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inner-splits", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--subjects", default=None,
                    help="restrict to a subject subset, e.g. '1-20' or '3,7,9'. "
                         "Lets one rung be split across processes; summary/gates only "
                         "run when --subjects is omitted (full 40).")
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rung_ids = [int(r) for r in args.rungs.split(",") if r.strip() != ""]
    subj_filter = None
    if args.subjects:
        subj_filter = set()
        for tok in args.subjects.split(","):
            tok = tok.strip()
            if "-" in tok:
                a, b = tok.split("-")
                subj_filter.update(range(int(a), int(b) + 1))
            elif tok:
                subj_filter.add(int(tok))

    t0 = time.time()
    X, y, subjects, _ = load_data()
    print(f"[data] X={X.shape} subjects={len(np.unique(subjects))} "
          f"classes={len(np.unique(y))} labels(encode order)=DNS,STDUP,UPS,WAK")

    for rid in rung_ids:
        run_rung(rid, X, y, subjects, out_dir, args.seed, args.inner_splits, args.verbose,
                 subj_filter=subj_filter)

    if subj_filter is not None:
        print(f"\n[partial run: --subjects {args.subjects}] skipping summary/gates; "
              f"re-run without --subjects once all 40 are present per rung.")
        print(f"[DONE] total elapsed {time.time()-t0:.0f}s")
        return

    summ = summarise(sorted(rung_ids), out_dir)

    # ---- validation gates ----
    print("\n================  VALIDATION GATES  ================")
    sm = summ.set_index("rung")
    gate_ok = True
    if 3 in sm.index:
        f3 = float(sm.loc[3, "f1_mean"])
        d3 = abs(f3 - GATE_RUNG3_F1)
        ok3 = d3 <= GATE_RUNG3_TOL
        gate_ok &= ok3
        print(f"[GATE rung 3] f1_mean={f3:.4f}  vs published {GATE_RUNG3_F1}  "
              f"|diff|={d3:.4f}  tol={GATE_RUNG3_TOL}  -> {'PASS' if ok3 else 'FAIL'}")
    else:
        print("[GATE rung 3] NOT RUN")
    if 0 in sm.index:
        f0 = float(sm.loc[0, "f1_mean"])
        gap = f0 - GATE_RUNG0_F1
        print(f"[GATE rung 0] f1_mean={f0:.4f}  vs published global baseline {GATE_RUNG0_F1}  "
              f"gap={gap:+.4f}  (rung0_global_z pools all 40; published fits StandardScaler on 39 "
              f"train subjects inside the Pipeline -- gap under ~0.005 expected)")
        if abs(gap) > 0.01:
            print("[GATE rung 0] WARNING: gap exceeds 0.01 -- larger than the definitional "
                  "difference should explain; investigate before trusting the ladder.")
    else:
        print("[GATE rung 0] NOT RUN")

    # ---- falsification check ----
    if 3 in sm.index and 4 in sm.index:
        f3, f4 = float(sm.loc[3, "f1_mean"]), float(sm.loc[4, "f1_mean"])
        print("\n================  OVER-ALIGNMENT CHECK  ================")
        if f4 >= f3:
            print(f"*** rung-4 F1 ({f4:.4f}) >= rung-3 F1 ({f3:.4f}). ***")
            print("*** The over-alignment account as written in Section 4.13.2 is CONTRADICTED. ***")
            print("*** Per the plan: report this plainly and STOP. Do not tune. ***")
        else:
            print(f"rung-4 F1 ({f4:.4f}) < rung-3 F1 ({f3:.4f}) by {f3-f4:.4f}: "
                  f"F1 falls at full whitening, consistent with the over-alignment account.")

    join_published(summ, out_dir)
    print(f"\n[GATES {'PASS' if gate_ok else 'FAIL'}]  total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

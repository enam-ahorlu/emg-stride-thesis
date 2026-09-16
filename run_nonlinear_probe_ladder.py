#!/usr/bin/env python3
"""
run_nonlinear_probe_ladder.py
==============================
EXPERIMENT_PLAN_PROBE.md -- does per-subject z-scoring remove the subject
structure a NONLINEAR probe sees, not just what a linear one sees?

Section 4.7.1's claim rests on a LogisticRegression probe going from 0.777
(rung 0) to 0.024 (rung 3, per-subject mean+scale) against a 1/40 = 0.025
chance floor. That only shows the structure is no longer LINEARLY decodable.
This script refits the identical five-rung ladder with two nonlinear probes
(RandomForest, MLP) alongside the linear one, and measures a permutation
floor at rungs 0 and 3 so the numbers are interpretable rather than assumed.

Diagnostic only -- no model retrained, no headline number can move. Imports
the rung transforms (RUNGS) and load_data() from analyze_between_subject_variance.py
UNCHANGED so the transforms are identical by construction. Does NOT edit that
file, does NOT touch results_variance_decomposition/alignment_ladder.csv, does
NOT regenerate Table 4.11, does NOT call part_b_ladder().

Run from 06_Code/ in the project .venv. CPU only, no GPU, no training.

Outputs:
  results_nonlinear_probe/nonlinear_probe_ladder.csv
  results_nonlinear_probe/permutation_null.csv
  report_figs/new_experiments/nonlinear_probe_ladder.png
"""
from __future__ import annotations
import sys, io, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Import UNCHANGED from the published script -- transforms identical by construction.
from analyze_between_subject_variance import RUNGS, load_data, SEED, LABELS

ROOT = Path(__file__).parent
OUT = ROOT / "results_nonlinear_probe"; OUT.mkdir(exist_ok=True)
FIGDIR = ROOT / "report_figs" / "new_experiments"; FIGDIR.mkdir(parents=True, exist_ok=True)
PUBLISHED = ROOT / "results_variance_decomposition" / "alignment_ladder.csv"

TOL = 0.005
N_PERM = 20
PERM_RUNGS = [0, 3]
RF_N_JOBS = 4

PROBES = {
    "linear": lambda: LogisticRegression(max_iter=300),
    "forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=RF_N_JOBS),
    "mlp":    lambda: MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True,
                                     random_state=SEED, max_iter=400),
}


def fit_probe(probe_name, Xr, subjects):
    """Same protocol as analyze_between_subject_variance.part_b_ladder's probe:
    StratifiedKFold(5, shuffle=True, random_state=SEED), balanced_accuracy."""
    clf = PROBES[probe_name]()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, Xr, subjects, cv=skf, scoring="balanced_accuracy", n_jobs=1)
    return scores


def permute_subjects_within_class(subjects, y, seed):
    """Shuffle subject labels WITHIN each movement class so class composition
    cannot leak into the permuted target."""
    rng = np.random.default_rng(seed)
    perm = subjects.copy()
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        perm[idx] = rng.permutation(subjects[idx])
    return perm


# ----------------------------------------------------------------------------
def phase0(X, y, subjects):
    print("\n" + "=" * 78 + "\nPHASE 0  reproduction gate (linear probe) + class-conditional check\n" + "=" * 78)
    pub = pd.read_csv(PUBLISHED).set_index("rung")["subject_probe_bal_acc"]

    rung_Xr = {}
    reproduced = {}
    ok = True
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        Xr = fn(X, subjects) if needs_subjects else fn(X)
        rung_Xr[rung_id] = Xr
        scores = fit_probe("linear", Xr, subjects)
        mean = float(scores.mean())
        reproduced[rung_id] = mean
        pv = float(pub.loc[rung_id])
        diff = mean - pv
        hit = abs(diff) <= TOL
        ok &= hit
        print(f"  rung {rung_id} ({name}): reproduced={mean:.4f}  published={pv:.4f}  "
              f"diff={diff:+.4f}  {'OK' if hit else 'MISS'}")

    if not ok:
        print("\n  *** PHASE 0 GATE FAILED *** -- stopping. Do not proceed to Phase 1 on an "
              "unreproduced base.")
        return None, None, False

    print("\n  PHASE 0 GATE: PASS (all five rungs within +-0.005 absolute).")

    print("\n  Phase 0.4 -- is each discrepancy measure actually class-conditional?")
    print("  (analyze_between_subject_variance.py, part_b_ladder, as read on disk)")
    print("    MMD (lines 289-299): YES, class-conditional then averaged. The loop at line 290")
    print("      ('for c in range(len(LABELS))') computes pairwise MMD within each movement class")
    print("      separately (samples keyed by (s, c) via sub_idx), then averages across the 4")
    print("      classes at line 299 ('mmd_mean = float(np.mean(mmd_per_class))'). Matches SS3.12.4.")
    print("    Wasserstein-1 (lines 301-313): NO, pooled across classes, not class-conditional.")
    print("      The code's own comment at lines 301-302 says so explicitly: '(ii) mean per-feature")
    print("      Wasserstein-1 between subject pairs (pooled across classes for tractability; MMD")
    print("      above is the class-conditional metric)'. subj_samples at line 304 is built from")
    print("      'Xr[subjects == s]' with no class mask -- all four classes mixed per subject before")
    print("      the per-feature Wasserstein distance is computed. This contradicts SS3.12.4's claim")
    print("      that Wasserstein-1 'was computed within movement class and then pooled'.")
    print("    Subject-identity probe (lines 315-318): NO, pooled across classes, not")
    print("      class-conditional. 'clf = LogisticRegression(...)' at 316 and the cross_val_score")
    print("      call at 318 are fit on the full 'Xr' (all rows, all four classes together) against")
    print("      the full 'subjects' vector -- there is no per-class loop or class mask anywhere in")
    print("      this block. This also contradicts SS3.12.4's claim for the probe.")
    print("    Net: of the three discrepancy measures SS3.12.4 describes as class-conditional-then-")
    print("    pooled, only MMD actually is. Wasserstein-1 and the subject-identity probe are both")
    print("    computed on class-pooled data. This is a documentation question about the existing,")
    print("    unedited protocol -- reported here, not changed.")

    return rung_Xr, reproduced, True


# ----------------------------------------------------------------------------
def phase1(rung_Xr, subjects):
    print("\n" + "=" * 78 + "\nPHASE 1  nonlinear probes at all five rungs\n" + "=" * 78)
    rows = []
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        Xr = rung_Xr[rung_id]
        for probe in ("linear", "forest", "mlp"):
            t0 = time.time()
            scores = fit_probe(probe, Xr, subjects)
            elapsed = time.time() - t0
            rows.append(dict(rung=rung_id, name=name, probe=probe,
                             bal_acc_mean=float(scores.mean()), bal_acc_sd=float(scores.std(ddof=1)),
                             chance_floor=1.0 / len(np.unique(subjects)), elapsed_sec=round(elapsed, 1)))
            print(f"  rung {rung_id} ({name:20s}) {probe:7s}: bal_acc={scores.mean():.4f} "
                  f"+-{scores.std(ddof=1):.4f}  ({elapsed:.1f}s)")
    df = pd.DataFrame(rows)

    print("\n  Ratio reading -- distance from each probe's OWN rung-0 value:")
    for probe in ("linear", "forest", "mlp"):
        r0 = df[(df.probe == probe) & (df.rung == 0)]["bal_acc_mean"].values[0]
        print(f"    {probe:7s} rung0={r0:.4f}")
        for rung_id in RUNGS:
            v = df[(df.probe == probe) & (df.rung == rung_id)]["bal_acc_mean"].values[0]
            print(f"      rung {rung_id} ({RUNGS[rung_id][0]:20s}): {v:.4f}   delta from own rung0 = {v - r0:+.4f}")
    return df


# ----------------------------------------------------------------------------
def phase2(rung_Xr, subjects, y, phase1_df):
    print("\n" + "=" * 78 + "\nPHASE 2  permutation null (rungs 0 and 3 only, 20 perms, seeds 0-19)\n" + "=" * 78)
    rows = []
    for rung_id in PERM_RUNGS:
        Xr = rung_Xr[rung_id]
        for probe in ("linear", "forest", "mlp"):
            observed = phase1_df[(phase1_df.probe == probe) & (phase1_df.rung == rung_id)]["bal_acc_mean"].values[0]
            null_vals = []
            t0 = time.time()
            for seed in range(N_PERM):
                perm_subj = permute_subjects_within_class(subjects, y, seed)
                scores = fit_probe(probe, Xr, perm_subj)
                null_vals.append(float(scores.mean()))
            null_vals = np.array(null_vals)
            elapsed = time.time() - t0
            p_val = (np.sum(null_vals >= observed) + 1) / (N_PERM + 1)  # one-sided
            rows.append(dict(rung=rung_id, name=RUNGS[rung_id][0], probe=probe,
                             observed=float(observed), null_mean=float(null_vals.mean()),
                             null_p95=float(np.percentile(null_vals, 95)),
                             null_min=float(null_vals.min()), null_max=float(null_vals.max()),
                             p_value=float(p_val), n_perm=N_PERM, elapsed_sec=round(elapsed, 1)))
            print(f"  rung {rung_id} {probe:7s}: observed={observed:.4f}  "
                  f"null_mean={null_vals.mean():.4f}  null_p95={np.percentile(null_vals,95):.4f}  "
                  f"p={p_val:.3f}  ({elapsed:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "permutation_null.csv", index=False)
    print(f"\n  [save] {OUT / 'permutation_null.csv'}")

    print("\n  Distance above the permutation floor:")
    for _, r in df.iterrows():
        above = r["observed"] - r["null_p95"]
        print(f"    rung {int(r.rung)} {r.probe:7s}: observed {r.observed:.4f} vs null_p95 {r.null_p95:.4f}"
              f"  -> {above:+.4f} above the 95th-percentile floor")
    return df


# ----------------------------------------------------------------------------
def phase3(phase1_df, perm_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11})

    rungs_sorted = sorted(RUNGS.keys())
    names = [RUNGS[r][0] for r in rungs_sorted]
    colors = {"linear": "#4C72B0", "forest": "#55A868", "mlp": "#c44e52"}
    chance = 1.0 / phase1_df["chance_floor"].apply(lambda v: round(1 / v)).iloc[0] if False else \
        float(phase1_df["chance_floor"].iloc[0])

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(rungs_sorted))
    for probe in ("linear", "forest", "mlp"):
        y_vals = [phase1_df[(phase1_df.probe == probe) & (phase1_df.rung == r)]["bal_acc_mean"].values[0]
                  for r in rungs_sorted]
        y_sd = [phase1_df[(phase1_df.probe == probe) & (phase1_df.rung == r)]["bal_acc_sd"].values[0]
                for r in rungs_sorted]
        ax.errorbar(x, y_vals, yerr=y_sd, marker="o", ms=7, lw=2, capsize=3,
                    color=colors[probe], label=probe)
        # permutation band where measured (rungs 0 and 3 only)
        for r in PERM_RUNGS:
            row = perm_df[(perm_df.probe == probe) & (perm_df.rung == r)]
            if len(row):
                xi = rungs_sorted.index(r)
                lo, hi = float(row.null_mean.values[0]), float(row.null_p95.values[0])
                ax.add_patch(plt.Rectangle((xi - 0.12, lo), 0.24, max(hi - lo, 1e-4),
                                           color=colors[probe], alpha=0.25, lw=0,
                                           label=f"{probe} permutation null (mean-p95)" if r == PERM_RUNGS[0] else None))

    ax.axhline(chance, ls="--", lw=1.3, color="#555", alpha=0.8, label=f"chance floor (1/40 = {chance:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("subject-identity probe balanced accuracy")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            h2.append(h); l2.append(l); seen.add(l)
    ax.legend(h2, l2, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    ax.set_title("Nonlinear probes on the alignment ladder: what falls to chance, and what does not",
                fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "nonlinear_probe_ladder.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [save] {FIGDIR / 'nonlinear_probe_ladder.png'}")


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    X, y, subjects, meta = load_data()
    print(f"[data] X={X.shape} subjects={len(np.unique(subjects))} classes={len(LABELS)}")

    rung_Xr, reproduced, ok = phase0(X, y, subjects)
    if not ok:
        print("\nSTOP after Phase 0.")
        return

    phase1_df = phase1(rung_Xr, subjects)
    phase1_df.to_csv(OUT / "nonlinear_probe_ladder.csv", index=False)
    print(f"\n  [save] {OUT / 'nonlinear_probe_ladder.csv'}")

    perm_df = phase2(rung_Xr, subjects, y, phase1_df)
    phase3(phase1_df, perm_df)

    # ---- outcome verdict ----
    print("\n" + "=" * 78 + "\nOutcome verdict (rung 3, operating point)\n" + "=" * 78)
    for probe in ("forest", "mlp"):
        obs = phase1_df[(phase1_df.probe == probe) & (phase1_df.rung == 3)]["bal_acc_mean"].values[0]
        r0 = phase1_df[(phase1_df.probe == probe) & (phase1_df.rung == 0)]["bal_acc_mean"].values[0]
        prow = perm_df[(perm_df.probe == probe) & (perm_df.rung == 3)]
        p95 = float(prow.null_p95.values[0]); pval = float(prow.p_value.values[0])
        print(f"  {probe:7s}: rung3={obs:.4f}  (rung0={r0:.4f}, delta={obs-r0:+.4f})  "
              f"permutation p95 at rung3={p95:.4f}  above_floor={obs-p95:+.4f}  p={pval:.3f}")

    lin3 = phase1_df[(phase1_df.probe == "linear") & (phase1_df.rung == 3)]["bal_acc_mean"].values[0]
    forest3 = phase1_df[(phase1_df.probe == "forest") & (phase1_df.rung == 3)]["bal_acc_mean"].values[0]
    mlp3 = phase1_df[(phase1_df.probe == "mlp") & (phase1_df.rung == 3)]["bal_acc_mean"].values[0]
    forest_p95 = float(perm_df[(perm_df.probe == "forest") & (perm_df.rung == 3)].null_p95.values[0])
    mlp_p95 = float(perm_df[(perm_df.probe == "mlp") & (perm_df.rung == 3)].null_p95.values[0])
    lin_p95 = float(perm_df[(perm_df.probe == "linear") & (perm_df.rung == 3)].null_p95.values[0])

    forest_above = forest3 - forest_p95 > 0.01  # >1pp above its own permutation floor
    mlp_above = mlp3 - mlp_p95 > 0.01
    lin_above = lin3 - lin_p95 > 0.01
    # Outcome C check: is the null itself uninformative even at rung 0?
    forest0_p95 = float(perm_df[(perm_df.probe == "forest") & (perm_df.rung == 0)].null_p95.values[0])
    mlp0_p95 = float(perm_df[(perm_df.probe == "mlp") & (perm_df.rung == 0)].null_p95.values[0])
    forest0 = phase1_df[(phase1_df.probe == "forest") & (phase1_df.rung == 0)]["bal_acc_mean"].values[0]
    mlp0 = phase1_df[(phase1_df.probe == "mlp") & (phase1_df.rung == 0)]["bal_acc_mean"].values[0]
    floor_uninformative = (forest0 - forest0_p95 < 0.01) and (mlp0 - mlp0_p95 < 0.01)

    if floor_uninformative:
        verdict = "C"
        text = ("the permutation null itself sits close to the observed rung-0 values for the "
               "nonlinear probes, so the null is not distinguishing signal from floor even at the "
               "reference rung -- the nonlinear probe is not an informative instrument at this "
               "cohort size for this question.")
    elif not forest_above and not mlp_above:
        verdict = "A"
        text = ("both nonlinear probes fall to within about a point of their own permutation floor "
               "at rung 3, matching the linear probe -- the SS4.7.1 claim generalises: no probe "
               "tried, linear or otherwise, recovers subject identity after per-subject "
               "standardisation.")
    else:
        verdict = "B"
        which = [p for p, above in [("forest", forest_above), ("mlp", mlp_above)] if above]
        text = (f"{' and '.join(which)} stay clearly above its own permutation floor at rung 3 "
               f"while the linear probe does not -- SS4.7.1's sentence must be scoped to linear "
               f"decodability; the residual is real nonlinear structure that per-subject "
               f"standardisation does not remove.")
    print(f"\n  OUTCOME {verdict}: {text}")
    print(f"\n[DONE] total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

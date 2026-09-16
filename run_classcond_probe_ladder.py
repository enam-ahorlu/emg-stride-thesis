#!/usr/bin/env python3
"""
run_classcond_probe_ladder.py
==============================
EXPERIMENT_PLAN_PROBE_CLASSCOND.md -- is the nonlinear subject-identity probe
reading physiology, or leaking class composition? Phase 0 of
run_nonlinear_probe_ladder.py confirmed the probe and the Wasserstein-1 column
are fit class-pooled, not class-conditionally (only MMD is). This script fits
the probe WITHIN each movement class and compares it to a size-matched
class-pooled control (same window budget, not class-restricted), so a drop
from class-restriction can be told apart from a drop from sample size.

Imports RUNGS/load_data/SEED/LABELS from analyze_between_subject_variance.py
and PROBES/fit_probe from run_nonlinear_probe_ladder.py UNCHANGED -- neither
file is edited, neither reproduction gate is touched.

Run from 06_Code/ in the project .venv. CPU only, no GPU, no training.

Outputs:
  results_nonlinear_probe/classcond_probe_ladder.csv
  results_nonlinear_probe/classcond_size_control.csv
  results_nonlinear_probe/subject_identifiability_vs_f1.csv
  report_figs/new_experiments/classcond_probe_ladder.png
"""
from __future__ import annotations
import sys, io, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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


def size_matched_subsample(subjects, n_total, seed):
    """Stratified-by-subject subsample of n_total rows total (>= base per
    subject so StratifiedKFold(5) on `subjects` stays valid), drawn from ALL
    classes -- the class-pooled control at the class-conditional probe's own
    window budget."""
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
def phase0(X, y, subjects):
    print("\n" + "=" * 78 + "\nPHASE 0  reproduction gate (class-pooled, all 15 values)\n" + "=" * 78)
    pub = pd.read_csv(PUBLISHED)
    rung_Xr = {}
    ok = True
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        Xr = fn(X, subjects) if needs_subjects else fn(X)
        rung_Xr[rung_id] = Xr
        for probe in ("linear", "forest", "mlp"):
            scores = fit_probe(probe, Xr, subjects)
            mean = float(scores.mean())
            row = pub[(pub.rung == rung_id) & (pub.probe == probe)]
            pv = float(row.bal_acc_mean.values[0])
            diff = mean - pv
            hit = abs(diff) <= TOL
            ok &= hit
            print(f"  rung {rung_id} ({name:20s}) {probe:7s}: reproduced={mean:.4f} published={pv:.4f} "
                  f"diff={diff:+.4f}  {'OK' if hit else 'MISS'}")
    if not ok:
        print("\n  *** PHASE 0 GATE FAILED *** -- stopping.")
        return None, None, False
    print("\n  PHASE 0 GATE: PASS (all 15 values within +-0.005).")

    print("\n  Window count per subject-by-class cell (min across 40 subjects x 4 classes):")
    ct = pd.crosstab(subjects, y)
    min_cell = int(ct.values.min())
    print(f"    min cell = {min_cell}  (warn threshold ~{MIN_CELL_WARN})")
    if min_cell < MIN_CELL_WARN:
        print(f"    *** at least one subject-by-class cell has fewer than {MIN_CELL_WARN} windows; "
              f"the class-conditional probe for that class fits on very little. Reported, not hidden.")
    else:
        print(f"    all cells >= {MIN_CELL_WARN}.")
    class_sizes = {c: int((y == c).sum()) for c in range(len(LABELS))}
    print(f"    class sizes (pooled across all 40 subjects): "
          f"{ {LABELS[c]: n for c, n in class_sizes.items()} }")
    return rung_Xr, class_sizes, True


# ----------------------------------------------------------------------------
def phase1(rung_Xr, y, subjects):
    print("\n" + "=" * 78 + "\nPHASE 1  class-conditional probe\n" + "=" * 78)
    rows = []
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        Xr = rung_Xr[rung_id]
        for probe in ("linear", "forest", "mlp"):
            per_class = []
            t0 = time.time()
            for c in range(len(LABELS)):
                mask = y == c
                try:
                    scores = fit_probe(probe, Xr[mask], subjects[mask])
                    v = float(scores.mean())
                except ValueError as e:
                    print(f"    [warn] rung {rung_id} {probe} class {LABELS[c]}: CV failed ({e}); recording NaN")
                    v = np.nan
                per_class.append(v)
            pooled = float(np.nanmean(per_class))
            elapsed = time.time() - t0
            rows.append(dict(rung=rung_id, name=name, probe=probe,
                             classcond_pooled=pooled,
                             **{f"classcond_{LABELS[c]}": per_class[c] for c in range(len(LABELS))},
                             elapsed_sec=round(elapsed, 1)))
            print(f"  rung {rung_id} ({name:20s}) {probe:7s}: pooled={pooled:.4f}  "
                  f"per-class={[round(v,3) for v in per_class]}  ({elapsed:.0f}s)")
    df = pd.DataFrame(rows)
    return df


# ----------------------------------------------------------------------------
def phase2(rung_Xr, subjects, class_sizes, phase1_df):
    print("\n" + "=" * 78 + "\nPHASE 2  size-matched class-pooled control (5 draws)\n" + "=" * 78)
    rows = []
    for rung_id, (name, fn, needs_subjects) in RUNGS.items():
        Xr = rung_Xr[rung_id]
        for probe in ("linear", "forest", "mlp"):
            per_class_ctrl = []
            per_class_draws = {}
            for c in range(len(LABELS)):
                n_c = class_sizes[c]
                draws = []
                for d in range(N_DRAWS):
                    idx = size_matched_subsample(subjects, n_c, seed=SEED + d)
                    try:
                        scores = fit_probe(probe, Xr[idx], subjects[idx])
                        draws.append(float(scores.mean()))
                    except ValueError:
                        draws.append(np.nan)
                per_class_draws[c] = draws
                per_class_ctrl.append(float(np.nanmean(draws)))
            pooled_ctrl = float(np.nanmean(per_class_ctrl))
            rows.append(dict(rung=rung_id, name=name, probe=probe, control_pooled=pooled_ctrl,
                             **{f"control_{LABELS[c]}": per_class_ctrl[c] for c in range(len(LABELS))},
                             **{f"control_{LABELS[c]}_sd": float(np.nanstd(per_class_draws[c], ddof=1))
                                for c in range(len(LABELS))}))
            print(f"  rung {rung_id} ({name:20s}) {probe:7s}: control_pooled={pooled_ctrl:.4f}  "
                  f"per-class={[round(v,3) for v in per_class_ctrl]}")
    df = pd.DataFrame(rows)
    merged = phase1_df.merge(df, on=["rung", "name", "probe"])
    merged["classcond_minus_control_pp"] = (merged["classcond_pooled"] - merged["control_pooled"]) * 100
    merged.to_csv(OUT / "classcond_probe_ladder.csv", index=False)
    df.to_csv(OUT / "classcond_size_control.csv", index=False)
    print(f"\n  [save] {OUT / 'classcond_probe_ladder.csv'}")
    print(f"  [save] {OUT / 'classcond_size_control.csv'}")

    print("\n  Class-conditional vs size-matched control (NOT vs the full-data pooled figure):")
    for _, r in merged.iterrows():
        print(f"    rung {int(r.rung)} {r.probe:7s}: class-cond={r.classcond_pooled:.4f}  "
              f"size-matched-control={r.control_pooled:.4f}  delta={r.classcond_minus_control_pp:+.2f} pp")
    return merged


# ----------------------------------------------------------------------------
def phase3(rung_Xr, X, y, subjects):
    print("\n" + "=" * 78 + "\nPHASE 3  does residual identifiability predict LOSO difficulty?\n" + "=" * 78)
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
    else:
        print(f"    [note] {cd_path} not found -- CNN arm skipped for this correlation.")

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
            print(f"  rung {rung_id} forest-recall vs {model} LOSO F1: rho={rho:+.3f} p={p:.3f} n={len(xs)}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "subject_identifiability_vs_f1.csv", index=False)
    print(f"\n  [save] {OUT / 'subject_identifiability_vs_f1.csv'}")

    if DIST_CORR.exists():
        existing = pd.read_csv(DIST_CORR)
        print("\n  Existing MMD / Mahalanobis distance-vs-F1 correlations (Section 4.5, for comparison):")
        print(existing.to_string(index=False))
    else:
        print(f"\n  [note] {DIST_CORR} not found -- cannot show the existing MMD/Mahalanobis comparison inline.")
    return df


# ----------------------------------------------------------------------------
def phase5_fig(merged):
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
    ax.set_xlabel("alignment ladder rung")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax.set_title("Class-conditional subject probe vs a size-matched class-pooled control",
                fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "classcond_probe_ladder.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [save] {FIGDIR / 'classcond_probe_ladder.png'}")


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    X, y, subjects, meta = load_data()
    print(f"[data] X={X.shape} subjects={len(np.unique(subjects))} classes={len(LABELS)}")

    rung_Xr, class_sizes, ok = phase0(X, y, subjects)
    if not ok:
        print("\nSTOP after Phase 0.")
        return

    phase1_df = phase1(rung_Xr, y, subjects)
    merged = phase2(rung_Xr, subjects, class_sizes, phase1_df)
    phase3(rung_Xr, X, y, subjects)

    print("\n" + "=" * 78 + "\nPHASE 4  class-conditional Wasserstein -- SKIPPED\n" + "=" * 78)
    print("  Not run: Phase 1 (5 rungs x 3 probes x 4 classes, including forest/MLP refits) already ran")
    print("  long by the plan's own definition, and Phase 4 'changes no conclusion' per the plan text.")
    print("  Recorded here as the explicit decision the plan authorizes, not a silent omission.")

    phase5_fig(merged)

    print("\n" + "=" * 78 + "\nOutcome verdict (rung 3, forest -- the operating point)\n" + "=" * 78)
    row = merged[(merged.probe == "forest") & (merged.rung == 3)].iloc[0]
    cc, ctrl = row.classcond_pooled, row.control_pooled
    delta_pp = row.classcond_minus_control_pp
    print(f"  forest @ rung3: class-conditional={cc:.4f}  size-matched control={ctrl:.4f}  delta={delta_pp:+.2f} pp")
    if cc >= 0.9:
        verdict = "A"
        text = (f"the class-conditional forest stays high at rung 3 ({cc:.4f}, roughly 0.9 or above) -- "
               f"the qualification in Section 4.7.1 closes with a number: subject identity survives "
               f"per-subject standardization intact, class composition was not doing the work.")
    elif abs(delta_pp) <= 5.0:
        verdict = "B"
        text = (f"class-conditional ({cc:.4f}) drops from the pooled figure, but the size-matched "
               f"control ({ctrl:.4f}) drops by a comparable amount (delta {delta_pp:+.2f} pp) -- "
               f"the drop is a sample-size artifact, not a class-composition effect; the conclusion "
               f"is unchanged.")
    else:
        verdict = "C"
        text = (f"class-conditional ({cc:.4f}) sits well below the size-matched control ({ctrl:.4f}, "
               f"delta {delta_pp:+.2f} pp) -- class composition was carrying real weight; Section "
               f"4.7.1's nonlinear figures must be restated at the class-conditional level.")
    print(f"\n  OUTCOME {verdict}: {text}")
    print(f"\n  New BH family members added: 0 (Phase 5 explicitly reports descriptive intervals on the "
          f"4-class contrasts and the 5-draw control comparison, not formal per-subject paired tests; "
          f"no Wilcoxon/BCa test of the kind that feeds the whole-thesis family is run here).")
    print(f"\n[DONE] total elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

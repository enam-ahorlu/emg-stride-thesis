# d2c_analysis.py
# ---------------------------------------------------------------------------
# Analysis for D2c (EXPERIMENT_PLAN_DEEPCORAL.md, "D2c AMENDMENT, 19 September 2026").
# Reads the per-fold CSVs written by run_deep_coral_align_loso.py and applies the decision
# rules fixed in the amendment BEFORE the runs. Reads *_subjectwise.csv only, never the
# rounded summaries.
#
#   python d2c_analysis.py                      # Stage A: lambda 100 vs 0.1
#   python d2c_analysis.py --with-lam0          # also Stage B: lambda 0.1 vs 0, if run
# ---------------------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, bootstrap

DIRS = {0.1: "results_deep_coral_align_lam0p1", 100.0: "results_deep_coral_align_lam100",
        0.0: "results_deep_coral_align_lam0"}
GATES = {0.1: 0.832075, 100.0: 0.834530, 0.0: 0.772}      # D2 arms; lambda 0 = global-norm+CD base
GATE_PP = 1.5
# (metric, threshold in pp for "moved", direction that counts as more invariance / less class info)
PRIMARY = [("domain_probe_bacc", 2.0), ("class_probe_tgt_bacc", 1.0)]
CHECKS = [("coral_rel", None), ("f1_macro", 0.5)]


def load(lam):
    d = Path(DIRS[lam])
    f = pd.read_csv(d / "deep_coral_subjectwise.csv").drop_duplicates("subject", keep="last")
    a = pd.read_csv(d / "alignment_subjectwise.csv").drop_duplicates("subject", keep="last")
    m = f[["subject", "f1_macro"]].merge(a, on="subject")
    if len(m) != 40:
        raise SystemExit(f"{d}: {len(m)} complete folds, expected 40. Report the arm as unfinished.")
    return m.sort_values("subject").reset_index(drop=True)


def paired(a, b, name, scale):
    d = (b - a) * scale
    p = wilcoxon(b, a).pvalue if np.any(d != 0) else 1.0
    ci = bootstrap((d,), np.mean, method="BCa", n_resamples=10000, random_state=42).confidence_interval
    return {"metric": name, "mean_a": a.mean(), "mean_b": b.mean(), "delta": d.mean(),
            "dz": d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
            "bca_lo": ci.low, "bca_hi": ci.high, "n_b_lower": int((d < 0).sum()), "p": p}


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def contrast(lo, hi):
    A, B = load(lo), load(hi)
    assert (A.subject.values == B.subject.values).all()
    rows = []
    for met, _ in PRIMARY + CHECKS:
        sc = 100.0 if met in ("domain_probe_bacc", "class_probe_tgt_bacc", "f1_macro") else 1.0
        rows.append(paired(A[met].values, B[met].values, met, sc))
    df = pd.DataFrame(rows); df["p_holm"] = holm(df["p"].values)
    df.insert(0, "contrast", f"lambda {hi:g} minus lambda {lo:g}")
    for c in ["feat_norm_src", "coral_scaled", "mmd2_rbf", "mean_gap_rel", "class_probe_src_bacc"]:
        df.loc[len(df)] = {"contrast": df.contrast[0], "metric": c + " (descriptive)",
                           "mean_a": A[c].mean(), "mean_b": B[c].mean(), "delta": (B[c] - A[c]).mean()}
    return df


def verdict(df):
    r = df.set_index("metric")
    sig = lambda m: r.loc[m, "p_holm"] < 0.05
    moved = sig("domain_probe_bacc") and r.loc["domain_probe_bacc", "delta"] <= -2.0
    reverse = sig("domain_probe_bacc") and r.loc["domain_probe_bacc", "delta"] >= 2.0
    cls_fell = sig("class_probe_tgt_bacc") and r.loc["class_probe_tgt_bacc", "delta"] <= -1.0
    f1_up = sig("f1_macro") and r.loc["f1_macro", "delta"] > 0.5
    if reverse:
        return "Outcome 5 (reverse): the higher weight left the embedding MORE subject-separable. Stop and escalate."
    if not moved:
        return "Outcome 1 (inert): the weight did not change subject separability. Stage B (lambda 0) runs."
    if f1_up:
        return "Outcome 4 (counter-example): more invariance AND better accuracy. Hard stop; escalate to Enam."
    if cls_fell:
        return "Outcome 2 (instance): invariance rose, class information fell, accuracy did not rise. Escalate (framing)."
    return "Outcome 3 (not better, no measured cost): invariance rose, class information held, accuracy did not rise. Escalate (framing)."


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--with-lam0", action="store_true")
    ap.add_argument("--out", default="results_deep_coral_align_d2c"); args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    lams = [0.1, 100.0] + ([0.0] if args.with_lam0 else [])
    print("Reproduction gates (macro-F1 mean vs published arm, must be within 1.5 pp):")
    for L in lams:
        m = load(L)["f1_macro"].mean()
        ok = abs(m - GATES[L]) * 100 <= GATE_PP
        print(f"  lambda {L:g}: {m:.6f} vs {GATES[L]:.6f}  offset {100*(m-GATES[L]):+.3f} pp  {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("Gate failed: stop and report. The harness has drifted.")
    res = contrast(0.1, 100.0)
    res.to_csv(out / "d2c_contrasts_stageA.csv", index=False)
    print(res.to_string(index=False)); print("\nSTAGE A:", verdict(res))
    if args.with_lam0:
        r0 = contrast(0.0, 0.1); r0.to_csv(out / "d2c_contrasts_stageB.csv", index=False)
        print(r0.to_string(index=False))
        # Stage B is a manipulation check, not a test of the shape: going from no alignment to some
        # alignment is the rising limb the through-line expects (per-subject normalization is itself
        # such a step), so an F1 gain here is not a counter-example. Only "did CORAL move invariance
        # at all" is read.
        r = r0.set_index("metric").loc["domain_probe_bacc"]
        moved = r["p_holm"] < 0.05 and r["delta"] <= -2.0
        print("\nSTAGE B (lambda 0.1 vs 0, manipulation check):",
              "the CORAL term moves subject separability at the smallest weight, and further weight adds nothing"
              if moved else "the CORAL term does not move subject separability at any weight tested")


if __name__ == "__main__":
    main()

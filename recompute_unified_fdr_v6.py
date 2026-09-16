"""Whole-thesis Benjamini-Hochberg family, v6 (14 September 2026).

v5 built the family to 189 tests. v6 folds in the September remediation programme:
the causal-smoothing sweep, the active-only ensemble arm, the four-way feature-set
comparison under LOSO, and the Deep CORAL D1 paired tests.

Scope rule, unchanged since v3: a test enters the family when the thesis reports the
claim it backs. BH is recomputed over the union, never patched.

Three decisions in this version are worth stating, because a reader checking the count
would otherwise have to infer them.

1. TWO ROWS ARE CORRECTED RATHER THAN ADDED. The family already carried
   "ResNet-SE+CD per-subject vs Deep CORAL (CD backbone, SIAT)" and
   "... vs AdaBN post (CD backbone, SIAT)" from the G9 promotion programme, at raw p
   0.001771 and 0.000994. The Deep CORAL D1 run recomputed the same two contrasts and
   got 0.009782 and 0.008264. The difference is not an error in either: G9 drew its
   per-subject arm from `results_cnn_aug_resnet_se_chandrop_proba` (mean 0.841566)
   while D1 drew it from `results_cnn_aug_resnet_se_chandrop` (mean 0.839490). The
   CORAL and AdaBN arms are byte-identical between the two. The 0.21 pp gap sits well
   inside the 0.5 pp run-to-run band of Section 3.16, so both runs are valid, but only
   0.839490 rounds to the 0.840 that Table 4.21 and Figure A.6 disclose. Section 4.7
   reports D1's figures, so the family must carry D1's p-values or the appendix would
   correct a test the body does not report. The rows are therefore updated in place and
   re-homed to the D-1 family, not duplicated.

2. THE PROBE CONTRIBUTES NOTHING. The nonlinear and permutation probe results are
   descriptive diagnostics computed on the class-pooled matrix, with no subject-paired
   test anywhere, so under the operator-family rule that already excludes P-8 they do
   not enter.

3. THE SMOOTHING SWEEP ENTERS IN FULL. Section 5.7.1 reports the five-window result
   explicitly and the shape of the curve across k implicitly, and the experiment plan
   pre-registered all six probability-averaged calib-100 contrasts. Including all six
   makes the correction strictly more conservative than including only the two the text
   names, so any claim that survives here would also survive the narrower family.
"""
import os
import numpy as np, pandas as pd

# Path resolution: v3 through v5 hard-coded the device-VM mount. Resolve relative to
# this file instead, so the script runs wherever the project is checked out.
CODE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CODE, "report_figs", "new_experiments")

# --- rebuild the v5 union from its own outputs ------------------------------------
# v5 wrote both scopes with the group/family/comparison/p/reported/bears columns
# intact, so the union of the two files reconstructs the family exactly, without
# re-executing the v3-v4-v5 chain and its hard-coded mount path.
a = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v5_A_all_reported.csv"))
b = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v5_B_bearing.csv"))
KEEP = ["group", "family", "comparison", "p", "reported", "bears"]
u = pd.concat([a[KEEP], b[KEEP]]).drop_duplicates("comparison").reset_index(drop=True)
rows = [dict(r) for _, r in u.iterrows()]
n_v5 = len(rows)
assert n_v5 == 189, n_v5

by_cmp = {r["comparison"]: r for r in rows}

def add(group, family, comparison, p, reported=True, bears=True):
    assert comparison not in by_cmp, "duplicate: " + comparison
    r = dict(group=group, family=family, comparison=comparison,
             p=float(p), reported=bool(reported), bears=bool(bears))
    rows.append(r); by_cmp[comparison] = r

def correct(comparison, p, family=None):
    r = by_cmp[comparison]
    old = r["p"]
    r["p"] = float(p)
    if family:
        r["family"] = family
    print("  corrected %-62s %.6g -> %.6g" % (comparison[:62], old, p))

# ---- D-1 CNN-side adaptation paired tests (4.7)
D1 = pd.read_csv(os.path.join(CODE, "results_deepcoral_d1", "d1_paired_stats.csv"))
d1p = dict(zip(D1.contrast, D1.p))
D1FAM = "D-1 CNN-side adaptation (4.7)"
print("D-1 corrections:")
correct("ResNet-SE+CD per-subject vs Deep CORAL (CD backbone, SIAT)",
        d1p["per-subject norm vs Deep CORAL (lambda=1)"], D1FAM)
correct("ResNet-SE+CD per-subject vs AdaBN post (CD backbone, SIAT)",
        d1p["per-subject norm vs AdaBN"], D1FAM)
add("mechanism", D1FAM, "D-1 Deep CORAL vs AdaBN (CD backbone, SIAT)",
    d1p["Deep CORAL (lambda=1) vs AdaBN"])

# ---- W-2 causal smoothing (3.2.4, 5.7.1): probability-averaged, calib-100
sm = pd.read_csv(os.path.join(CODE, "results_causal_smoothing", "smoothing_wilcoxon.csv"))
sm = sm[sm.variant == "probavg"]
MET = {"f1": "macro-F1", "crit_err": "DNS to WAK critical-error rate"}
for _, r in sm.iterrows():
    add("audit", "W-2 causal smoothing (3.2.4, 5.7.1)",
        "W-2 causal vote %s, %s [calib-100, probability-averaged]"
        % (r["contrast"].replace("_vs_", " vs "), MET[r["metric"]]), r["p"])

# ---- S-1 active-only STDUP (4.3.1): the ensemble arm the original control omitted
ae = pd.read_csv(os.path.join(CODE, "results_aonly_ensemble", "aonly_ensemble_wilcoxon.csv"))
for _, r in ae.iterrows():
    add("audit", "S-1 active-only STDUP (4.3.1)", "S-1 " + r["contrast"], r["p"])

# ---- W-3 feature-set comparison under LOSO (4.1.1)
fs = pd.read_csv(os.path.join(CODE, "results_featureset_loso", "featureset_loso_wilcoxon.csv"))
for _, r in fs.iterrows():
    add("audit", "W-3 feature-set LOSO comparison (4.1.1)",
        "W-3 %s [%s], LOSO" % (r["contrast"], r["model"]), r["p"])

df = pd.DataFrame(rows)
print("\nv5 rows: %d   v6 additions: %d   total: %d" % (n_v5, len(df) - n_v5, len(df)))
dups = df.comparison[df.comparison.duplicated()].tolist()
assert not dups, dups

def bh(p):
    p = np.asarray(p, float); m = len(p); out = np.empty(m); prev = 1.0
    for rank, i in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[i] * m / (m - rank)); out[i] = prev
    return out

def holm(p):
    p = np.asarray(p, float); m = len(p); out = np.empty(m); run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (m - rank) * p[i])); out[i] = run
    return out

NEWFAM = ("D-1 CNN-side adaptation (4.7)", "W-2 causal smoothing (3.2.4, 5.7.1)",
          "W-3 feature-set LOSO comparison (4.1.1)")

for name, mask in (("A_all_reported", df.reported), ("B_bearing", df.bears)):
    prev = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v5_%s.csv" % name))
    prev["sig_old"] = prev.p_BH < 0.05
    sub = df[mask].reset_index(drop=True).copy()
    sub["p_BH"] = bh(sub.p); sub["p_holm"] = holm(sub.p); sub["sig"] = sub.p_BH < 0.05
    sub.to_csv(os.path.join(OUT, "unified_fdr_family_v6_%s.csv" % name), index=False)
    print("\n=== SCOPE %s: m = %d (was %d), survive BH = %d, fail = %d"
          % (name, len(sub), len(prev), int(sub.sig.sum()), int((~sub.sig).sum())))
    print("    largest surviving raw p = %.6g   smallest failing raw p = %.6g"
          % (sub[sub.sig].p.max(), sub[~sub.sig].p.min()))
    newsig = dict(zip(sub.comparison, sub.sig))
    fell = [(r["comparison"], r["p_BH"]) for _, r in prev.iterrows()
            if r["sig_old"] and r["comparison"] in newsig and not newsig[r["comparison"]]]
    rose = [(r["comparison"], r["p_BH"]) for _, r in prev.iterrows()
            if (not r["sig_old"]) and r["comparison"] in newsig and newsig[r["comparison"]]]
    print("  previously surviving that now FAIL: %d" % len(fell))
    for c, pb in fell: print("     FELL %-74s v5BH=%.4f" % (c[:74], pb))
    print("  previously failing that now SURVIVE: %d" % len(rose))
    for c, pb in rose: print("     ROSE %-74s v5BH=%.4f" % (c[:74], pb))
    edge = sub[(sub.p_BH > 0.030) & (sub.p_BH < 0.080)].sort_values("p_BH")
    print("  within BH 0.030-0.080: %d" % len(edge))
    for _, r in edge.iterrows():
        print("     EDGE %-70s BH=%.4f %s" % (str(r["comparison"])[:70], r["p_BH"],
                                              "sig" if r["sig"] else "NS"))
    if name == "A_all_reported":
        nr = sub[sub.family.isin(NEWFAM) | sub.comparison.str.startswith(("W-2 ", "S-1 aonly"))]
        print("\n  the %d new or corrected tests:" % len(nr))
        for _, r in nr.sort_values("p").iterrows():
            print("     %-70s raw=%.4g BH=%.4g %s"
                  % (str(r["comparison"])[:70], r["p"], r["p_BH"], "sig" if r["sig"] else "NS"))
        print("\n  non-survivors by family:")
        ns = sub[~sub.sig]
        for fam, n in ns.family.value_counts().items():
            print("     %-48s %d" % (str(fam)[:48], n))

"""Whole-thesis Benjamini-Hochberg family, v8 (16 September 2026).

v7 built the family to 218 tests, 153 survivors (A_all_reported scope). v8 adds the D2
Deep CORAL lambda sweep of the 15-Sep AMENDMENT to EXPERIMENT_PLAN_DEEPCORAL.md.

Scope rule, unchanged since v3: a test enters the family when the thesis reports the
claim it backs. BH is recomputed over the union, never patched.

ELEVEN NEW TESTS, per the amendment's own admission rule (D2.5): admit every
lambda-against-per-subject contrast and every lambda-against-lambda-1 contrast for the
five NEW lambdas run in this sweep (0.1, 3, 10, 30, 100 -- lambda=1 is the reproduction/
comparator arm, not a new lambda), one member each (5 + 5 = 10), plus the best lambda
against AdaBN (1) = 11. Do NOT admit only the best lambda's contrast against per-subject
normalization: picking the arm after seeing the result and correcting as though one test
had been planned is exactly the selection this family exists to price.

Every arm in the sweep is a distinct GPU training run (six independent LOSO passes, seed
42, no repeats) -- none of the "genuine tests only" exclusions from v7 (bit-identical
arms, exact duplicates) apply here, but the check is applied rather than assumed: all six
subjectwise CSVs (`results_deep_coral_lam{0p1,1p0_repro,3,10,30,100}/deep_coral_subjectwise.csv`)
were confirmed to differ from one another (they are not byte-identical the way the FILTER
chain's envelope arms turned out to be), so all 11 contrasts drawn from them are genuine.

The three ORIGINAL D1 contrasts (per-subject vs Deep CORAL lambda=1, per-subject vs
AdaBN, Deep CORAL lambda=1 vs AdaBN) are already present in v7 under family "D-1 CNN-side
adaptation (4.7)" and are NOT re-added here -- confirmed by inspection before running this
script (raw p 0.009782, 0.008264, 0.451734 respectively, matching D1's own report exactly).
"""
import os
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CODE, "report_figs", "new_experiments")

a = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v7_A_all_reported.csv"))
b = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v7_B_bearing.csv"))
KEEP = ["group", "family", "comparison", "p", "reported", "bears"]
u = pd.concat([a[KEEP], b[KEEP]]).drop_duplicates("comparison").reset_index(drop=True)
rows = [dict(r) for _, r in u.iterrows()]
n_v7 = len(rows)
assert n_v7 == 218, n_v7
seen = set(r["comparison"] for r in rows)

def add(group, family, comparison, p, reported=True, bears=True):
    assert comparison not in seen, "duplicate: " + comparison
    rows.append(dict(group=group, family=family, comparison=comparison,
                     p=float(p), reported=bool(reported), bears=bool(bears)))
    seen.add(comparison)

# ---- D2 Deep CORAL lambda sweep (4.7): 11 genuine new contrasts
pc = pd.read_csv(os.path.join(CODE, "results_deep_coral_d2", "paired_contrasts.csv"))
assert len(pc) == 11, len(pc)
FAM = "D2 Deep CORAL lambda sweep (4.7)"
for _, r in pc.iterrows():
    add("audit", FAM, "D2 %s" % r["contrast"], r["p"])
print("D2 sweep rows available: %d   admitted: %d" % (len(pc), len(pc)))

df = pd.DataFrame(rows)
print("v7 rows: %d   v8 additions: %d   total: %d" % (n_v7, len(df) - n_v7, len(df)))
assert not df.comparison.duplicated().any()

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

for name, mask in (("A_all_reported", df.reported), ("B_bearing", df.bears)):
    prev = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v7_%s.csv" % name))
    prev["sig_old"] = prev.p_BH < 0.05
    sub = df[mask].reset_index(drop=True).copy()
    sub["p_BH"] = bh(sub.p); sub["p_holm"] = holm(sub.p); sub["sig"] = sub.p_BH < 0.05
    sub.to_csv(os.path.join(OUT, "unified_fdr_family_v8_%s.csv" % name), index=False)
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
    for c, pb in fell: print("     FELL %-74s v8BH=%.4f" % (c[:74], pb))
    print("  previously failing that now SURVIVE: %d" % len(rose))
    for c, pb in rose: print("     ROSE %-74s v8BH=%.4f" % (c[:74], pb))
    # explicit check on the two boundary contrasts the amendment names
    for tag in ("ENABL3S", "AdaBN post vs pre"):
        m2 = sub[sub.comparison.str.contains(tag, case=False, na=False)]
        for _, r in m2.iterrows():
            print("     BOUNDARY-WATCH %-60s p=%.4g p_BH=%.4g %s"
                  % (str(r["comparison"])[:60], r["p"], r["p_BH"], "sig" if r["sig"] else "NS"))
    if name == "A_all_reported":
        nr = sub[sub.family == FAM]
        print("\n  the %d new tests:" % len(nr))
        for _, r in nr.sort_values("p").iterrows():
            print("     %-70s raw=%.4g BH=%.4g %s"
                  % (str(r["comparison"])[:70], r["p"], r["p_BH"], "sig" if r["sig"] else "NS"))

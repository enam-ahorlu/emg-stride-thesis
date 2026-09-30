"""Whole-thesis Benjamini-Hochberg family, v7 (15 September 2026).

v6 built the family to 210 tests. v7 adds the causal filter chain of Section 4.13.

Scope rule, unchanged since v3: a test enters the family when the thesis reports the
claim it backs. BH is recomputed over the union, never patched.

ONE DECISION IN THIS VERSION NEEDS STATING, because the experiment's own outcome file
offers 32 rows and only 8 of them enter.

`results_filter_causal/causal_filter_wilcoxon.csv` holds 32 rows: four contrasts for
each combination of two models, two scoring modes and two metrics. The four arms were
designed as a 2 by 2, zero-phase against single-pass bandpass crossed with centred
against trailing envelope. The envelope factor turned out to be a null operation on
these features: Freq-72 is extracted from the bandpass output, which preprocessing
stores separately from the envelope, so arm C is bit-identical to arm A and arm D to
arm B at full LOSO scale. That collapses the four contrasts in every cell to one:

  B vs A                 the genuine filter contrast, the only one that varies anything
  C vs A                 mean delta exactly 0.0, p = 1.0, no Wilcoxon statistic at all
  D vs A                 every column identical to B vs A, because D is B
  D vs best of B and C   either 0.0, or again identical to B vs A

Only `B vs A` enters, eight times. Admitting the other 24 would put 12 comparisons of
a pair of identical arms and 12 exact duplicates of a test already in the family into
a correction whose whole purpose is to price the number of genuine tests. The zero
rows would inflate m with no information and make every real test harder to survive;
the duplicates would additionally double-count the same evidence. Section 4.13 reports
eight contrasts for the same reason, and Appendix A.6 records the exclusion.
"""
import os
import numpy as np, pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CODE, "report_figs", "new_experiments")

a = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v6_A_all_reported.csv"))
b = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v6_B_bearing.csv"))
KEEP = ["group", "family", "comparison", "p", "reported", "bears"]
u = pd.concat([a[KEEP], b[KEEP]]).drop_duplicates("comparison").reset_index(drop=True)
rows = [dict(r) for _, r in u.iterrows()]
n_v6 = len(rows)
assert n_v6 == 210, n_v6
seen = set(r["comparison"] for r in rows)

def add(group, family, comparison, p, reported=True, bears=True):
    assert comparison not in seen, "duplicate: " + comparison
    rows.append(dict(group=group, family=family, comparison=comparison,
                     p=float(p), reported=bool(reported), bears=bool(bears)))
    seen.add(comparison)

# ---- W-4 causal filter chain (4.13): the eight genuine filter contrasts
fc = pd.read_csv(os.path.join(CODE, "results_filter_causal", "causal_filter_wilcoxon.csv"))
gen = fc[fc.contrast == "B vs A"]
assert len(gen) == 8, len(gen)
MET = {"f1": "macro-F1", "crit_err": "DNS to WAK critical-error rate"}
for _, r in gen.iterrows():
    add("audit", "W-4 causal filter chain (4.13)",
        "W-4 single-pass vs zero-phase bandpass [%s, %s], %s"
        % (r["model"], r["mode"], MET[r["metric"]]), r["p"])
dropped = len(fc) - len(gen)
print("filter rows available: %d   admitted: %d   excluded as duplicate or null: %d"
      % (len(fc), len(gen), dropped))

df = pd.DataFrame(rows)
print("v6 rows: %d   v7 additions: %d   total: %d" % (n_v6, len(df) - n_v6, len(df)))
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
    prev = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v6_%s.csv" % name))
    prev["sig_old"] = prev.p_BH < 0.05
    sub = df[mask].reset_index(drop=True).copy()
    sub["p_BH"] = bh(sub.p); sub["p_holm"] = holm(sub.p); sub["sig"] = sub.p_BH < 0.05
    sub.to_csv(os.path.join(OUT, "unified_fdr_family_v7_%s.csv" % name), index=False)
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
    for c, pb in fell: print("     FELL %-74s v6BH=%.4f" % (c[:74], pb))
    print("  previously failing that now SURVIVE: %d" % len(rose))
    for c, pb in rose: print("     ROSE %-74s v6BH=%.4f" % (c[:74], pb))
    edge = sub[(sub.p_BH > 0.030) & (sub.p_BH < 0.080)].sort_values("p_BH")
    print("  within BH 0.030-0.080: %d" % len(edge))
    for _, r in edge.iterrows():
        print("     EDGE %-70s BH=%.4f %s" % (str(r["comparison"])[:70], r["p_BH"],
                                              "sig" if r["sig"] else "NS"))
    if name == "A_all_reported":
        nr = sub[sub.family == "W-4 causal filter chain (4.13)"]
        print("\n  the %d new tests:" % len(nr))
        for _, r in nr.sort_values("p").iterrows():
            print("     %-70s raw=%.4g BH=%.4g %s"
                  % (str(r["comparison"])[:70], r["p"], r["p_BH"], "sig" if r["sig"] else "NS"))

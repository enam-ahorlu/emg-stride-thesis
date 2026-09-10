# -*- coding: utf-8 -*-
"""Whole-thesis Benjamini-Hochberg family, v5 (8 September 2026).

v4 built the family to 187 tests. v5 adds the two paired tests from the alignment
ladder re-run at a 400 ms window (W-1 Stage 4), which Section 4.15 now reports as a
replication of the Section 4.7 mechanism at a second window.

Scope rule, unchanged from v4: a test enters the family when the thesis reports the
claim it backs. Only the two contrasts the ladder's own stats file computes are added;
the discrepancy, subject-probe and silhouette columns of that run were carried over
from the 250 ms ladder rather than recomputed, so nothing is drawn from them.
"""
import os
import numpy as np, pandas as pd

CODE = os.path.join(os.environ["HOME"], "mnt", "FInal Project", "06_Code")
OUT  = os.path.join(CODE, "report_figs", "new_experiments")

# --- rebuild the v4 union by executing v4's construction, then extend -------------
src = open(os.path.join(CODE, "recompute_unified_fdr_v4.py"), encoding="utf-8").read()
head = src.split("\ndf = pd.DataFrame(rows)")[0]  # leading newline: v4 itself contains the bare string
ns = {"__name__": "_v4"}
exec(compile(head, "v4_head", "exec"), ns)
rows, add = ns["rows"], ns["add"]
n_v4 = len(rows)

# ---- W-1 Stage 4: alignment ladder re-run at 400 ms (4.15)
lad = pd.read_csv(os.path.join(CODE, "results_win400_ladder",
                               "alignment_ladder_loso_stats.csv"))
LABEL = {"rung4_vs_rung3": "W-1 ladder @ 400 ms: full whitening vs per-subject z",
         "rung3_vs_rung0": "W-1 ladder @ 400 ms: per-subject z vs global z"}
for _, r in lad.iterrows():
    add("mechanism", "W-1 alignment ladder at 400 ms (4.15)",
        LABEL[r["comparison"]], float(r["wilcoxon_p"]))

df = pd.DataFrame(rows)
print("v4 rows: %d   v5 additions: %d   total: %d" % (n_v4, len(df) - n_v4, len(df)))
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

prev = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v4_A_all_reported.csv"))
prev["sig_old"] = prev.p_BH < 0.05
print("previous family (v4 scope A): m = %d, survivors = %d"
      % (len(prev), int(prev.sig_old.sum())))

for name, mask in (("A_all_reported", df.reported), ("B_bearing", df.bears)):
    sub = df[mask].reset_index(drop=True).copy()
    sub["p_BH"] = bh(sub.p); sub["p_holm"] = holm(sub.p); sub["sig"] = sub.p_BH < 0.05
    sub.to_csv(os.path.join(OUT, "unified_fdr_family_v5_%s.csv" % name), index=False)
    print("\n=== SCOPE %s: m = %d, survive BH = %d, fail = %d"
          % (name, len(sub), int(sub.sig.sum()), int((~sub.sig).sum())))
    newsig = dict(zip(sub.comparison, sub.sig))
    fell = [(r["comparison"], r["p_BH"]) for _, r in prev.iterrows()
            if r["sig_old"] and r["comparison"] in newsig and not newsig[r["comparison"]]]
    rose = [(r["comparison"], r["p_BH"]) for _, r in prev.iterrows()
            if (not r["sig_old"]) and r["comparison"] in newsig and newsig[r["comparison"]]]
    print("  previously surviving that now FAIL: %d" % len(fell))
    for c, pb in fell: print("     FELL %-78s oldBH=%.4f" % (c[:78], pb))
    print("  previously failing that now SURVIVE: %d" % len(rose))
    for c, pb in rose: print("     ROSE %-78s oldBH=%.4f" % (c[:78], pb))
    edge = sub[(sub.p_BH > 0.030) & (sub.p_BH < 0.080)].sort_values("p_BH")
    print("  within BH 0.030-0.080: %d" % len(edge))
    for _, r in edge.iterrows():
        print("     EDGE %-72s BH=%.4f %s" % (str(r["comparison"])[:72], r["p_BH"],
                                              "sig" if r["sig"] else "NS"))
    if name == "A_all_reported":
        nr = sub[sub.comparison.str.startswith("W-1 ladder")]
        print("\n  the %d new tests:" % len(nr))
        for _, r in nr.sort_values("p").iterrows():
            print("     %-70s raw=%.3g BH=%.4g %s" % (str(r["comparison"])[:70], r["p"],
                                                      r["p_BH"], "sig" if r["sig"] else "NS"))

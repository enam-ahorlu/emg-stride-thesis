# -*- coding: utf-8 -*-
"""v9 of the whole-thesis Benjamini-Hochberg family.
v8 held 229 reported tests (226 bearing on a finding). v9 admits the four subject-paired Wilcoxon contrasts of the Deep CORAL
alignment programme, under v7's genuine-tests-only rule:
  D2c Stage A, lambda 100 against lambda 0.1 (results_deep_coral_align_lam100 / _lam0p1), the two pre-registered primaries
    - domain_probe_bacc  p = 0.010628069954691455
    - class_probe_tgt_bacc p = 0.09449897358172166
  D2e, macro-F1 on arms the family does not otherwise carry (E3 is the plain script re-run at batch 256)
    - E1, lambda 0 with no target forward pass, against lambda 0 in train mode (results_deep_coral_align_lam0_notgt / _lam0)
    - E2, lambda 30 with the target pass in eval mode, against E1 (results_deep_coral_align_lam30_tgteval / _lam0_notgt)
Not admitted, as the D2c plan fixed in advance: D2c's coral_rel manipulation check and its macro-F1 contrast, which re-tests
arms the family already carries through the D2 sweep. Those are reported under the experiment's own Holm correction.
p-values for the two D2e rows are recomputed here from the subjectwise files, never copied.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
CODE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CODE, "report_figs", "new_experiments")
def f1(d): return pd.read_csv(os.path.join(CODE, d, "deep_coral_subjectwise.csv")).sort_values("subject").set_index("subject").f1_macro
def paired(a, b): return float(wilcoxon(f1(a), f1(b)).pvalue)
def paired_arch(a, b):
    x = pd.read_csv(os.path.join(CODE, a, "cnn_arch_subjectwise.csv")).sort_values("subject").set_index("subject").f1_macro
    return float(wilcoxon(x, f1(b)).pvalue)
NEW = [
    ("audit", "D2c Deep CORAL alignment follow-up (4.2.2)", "D2c domain probe, lambda 100 vs lambda 0.1", 0.010628069954691455),
    ("audit", "D2c Deep CORAL alignment follow-up (4.2.2)", "D2c target class probe, lambda 100 vs lambda 0.1", 0.09449897358172166),
    ("audit", "D2e Deep CORAL ablations (4.2.2)", "D2e E1 (lambda 0, no target pass) vs lambda 0 in train mode",
     paired("results_deep_coral_align_lam0_notgt", "results_deep_coral_align_lam0")),
    ("audit", "D2e Deep CORAL ablations (4.2.2)", "D2e E2 (lambda 30, target pass in eval) vs E1",
     paired("results_deep_coral_align_lam30_tgteval", "results_deep_coral_align_lam0_notgt")),
    ("audit", "D2e Deep CORAL ablations (4.2.2)", "D2e E3 (plain script at batch 256) vs E1",
     paired_arch("results_repro_250global_b256", "results_deep_coral_align_lam0_notgt")),
]
def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(len(p)); out[o] = np.minimum(q, 1.0); return out
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); q = p[o] * (len(p) - np.arange(len(p)))
    q = np.maximum.accumulate(q); out = np.empty(len(p)); out[o] = np.minimum(q, 1.0); return out
for name in ("A_all_reported", "B_bearing"):
    prev = pd.read_csv(os.path.join(OUT, "unified_fdr_family_v8_%s.csv" % name))
    add = pd.DataFrame([{"group": g, "family": f, "comparison": c, "p": p,
                         "reported": True, "bears": True} for g, f, c, p in NEW])
    df = pd.concat([prev[["group", "family", "comparison", "p", "reported", "bears"]], add], ignore_index=True)
    df["p_BH"] = bh(df.p); df["p_holm"] = holm(df.p); df["sig"] = df.p_BH <= 0.05
    df.to_csv(os.path.join(OUT, "unified_fdr_family_v9_%s.csv" % name), index=False)
    before = prev.assign(sig8=bh(prev.p) <= 0.05).sig8.values
    moved = int((before != df.sig.values[:len(prev)]).sum())
    print("%s: %d tests, %d survive, %d do not; existing comparisons changing side: %d"
          % (name, len(df), int(df.sig.sum()), int((~df.sig).sum()), moved))
    if name == "A_all_reported":
        for _, r in df.tail(5).iterrows():
            print("   %-62s p=%.3g BH=%.3g %s" % (r.comparison[:62], r.p, r.p_BH, "survives" if r.sig else "does not survive"))

# -*- coding: utf-8 -*-
import os, re, sys
import numpy as np, pandas as pd

CODE = os.path.join(os.environ["HOME"], "mnt", "FInal Project", "06_Code")
WORK = os.path.join(os.environ["HOME"], "work")

base = pd.read_csv(os.path.join(CODE, "report_figs", "new_experiments",
                                "unified_fdr_all_experiments.csv"))
assert len(base) == 121, len(base)
rows = [dict(group="established", family=r["family"], comparison=r["comparison"],
             p=float(r["p"]), reported=True, bears=True)
        for _, r in base.iterrows()]

w1 = pd.read_csv(os.path.join(CODE, "results_window_ablation", "window_ablation_tests.csv"))
assert len(w1) == 12, len(w1)
W1_UNREPORTED = {"SVM: 150 ms vs 400 ms (per-subject)",
                 "ResNet-SE+CD: 150 ms vs 400 ms (per-subject)"}
W1_DUPLICATE = {"SVM @ 250 ms: per-subject vs global"}
for _, r in w1.iterrows():
    c = r["comparison"]
    if c in W1_DUPLICATE: continue
    rows.append(dict(group="W-1", family="W-1 window ablation (4.16)", comparison=c,
                     p=float(r["p_raw"]),
                     reported=c not in W1_UNREPORTED, bears=c not in W1_UNREPORTED))

pat = re.compile(r"^\s*(?P<label>.+?)\s{2,}(?P<delta>[+-][\d.]+) pp\s+CI\[[^\]]*\]\s+p=(?P<p>[\d.eE+-]+)")
v = {}
for line in open(os.path.join(WORK, "v481.txt"), encoding="utf-8"):
    m = pat.match(line)
    if m: v[m.group("label").strip()] = float(m.group("p"))
assert len(v) == 12, sorted(v)
V_DUP = {"Simple +CD", "ResNet-SE +CD"}
V_NOT_BEARING = {"family: repro vs persubj", "no-skip vs plain, no aug",
                 "skip term, headroom-adjusted"}
for label, p in v.items():
    if label in V_DUP: continue
    rows.append(dict(group="mechanism", family="CD mechanism (4.8.1)", comparison=label,
                     p=p, reported=True, bears=label not in V_NOT_BEARING))

def grab(path, needle):
    for line in open(path, encoding="utf-8"):
        if needle in line:
            m = re.search(r"p = ([\d.eE+-]+)", line)
            if m: return float(m.group(1))
    raise SystemExit("MISS parse: %s in %s" % (needle, path))

g3p = os.path.join(WORK, "g3.txt"); w4p = os.path.join(WORK, "w4.txt")
mech2 = [
 ("G3 single-electrode occlusion cost, no-aug vs channel dropout", grab(g3p, "paired change"), True,  True),
 ("G3 within-subject concentration, normalized",  grab(g3p, "concentration, normalized"),      False, False),
 ("G3 between-subject consistency, Spearman",     grab(g3p, "consistency, Spearman"),          False, False),
 ("G3 between-subject consistency, Pearson",      grab(g3p, "consistency, Pearson"),           False, False),
 ("G3 concentration, raw pp (scale-confounded)",  grab(g3p, "concentration, raw pp"),          False, False),
 ("W-4 gain jitter vs no augmentation",           grab(w4p, "gain jitter  "),                  False, True),
 ("W-4 channel dropout minus gain jitter",        grab(w4p, "chandrop minus gainjitter"),      True,  True),
]
for label, p, rep, bears in mech2:
    rows.append(dict(group="mechanism", family="CD mechanism (4.8.2)", comparison=label,
                     p=p, reported=rep, bears=bears))

g2 = pd.read_csv(os.path.join(CODE, "g2_rate_tests.csv"))
assert len(g2) == 3, len(g2)
for _, r in g2.iterrows():
    rows.append(dict(group="mechanism", family="CD mechanism (4.8.2)",
                     comparison="G2 dropout rate " + str(r["label"]),
                     p=float(r["p_raw"]), reported=True, bears=True))

df = pd.DataFrame(rows)
print("candidate rows assembled:", len(df))

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

old = base.copy(); old["sig_old"] = old.p_BH < 0.05
print("established family: m=121 survivors=%d" % int(old.sig_old.sum()))

for name, mask in (("A_all_reported", df.reported), ("B_bearing", df.bears)):
    sub = df[mask].reset_index(drop=True).copy()
    sub["p_BH"] = bh(sub.p); sub["p_holm"] = holm(sub.p); sub["sig"] = sub.p_BH < 0.05
    sub.to_csv(os.path.join(WORK, "family_%s.csv" % name), index=False)
    print("\n=== SCOPE %s: m = %d, survive BH = %d, fail = %d"
          % (name, len(sub), int(sub.sig.sum()), int((~sub.sig).sum())))
    newsig = dict(zip(sub.comparison, sub.sig))
    fell = [(r["comparison"], r["p_BH"]) for _, r in old.iterrows()
            if r["sig_old"] and r["comparison"] in newsig and not newsig[r["comparison"]]]
    rose = [(r["comparison"], r["p_BH"]) for _, r in old.iterrows()
            if (not r["sig_old"]) and r["comparison"] in newsig and newsig[r["comparison"]]]
    print("  previously surviving that now FAIL: %d" % len(fell))
    for c, pb in fell: print("     FELL %-84s oldBH=%.4f" % (c[:84], pb))
    print("  previously failing that now SURVIVE: %d" % len(rose))
    for c, pb in rose: print("     ROSE %-84s oldBH=%.4f" % (c[:84], pb))
    edge = sub[(sub.p_BH > 0.035) & (sub.p_BH < 0.070)].sort_values("p_BH")
    print("  within BH 0.035-0.070: %d" % len(edge))
    for _, r in edge.iterrows(): print("     EDGE %-80s BH=%.4f %s" % (str(r["comparison"])[:80], r["p_BH"], "sig" if r["sig"] else "NS"))

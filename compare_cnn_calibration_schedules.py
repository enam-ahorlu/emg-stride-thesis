#!/usr/bin/env python3
"""
compare_cnn_calibration_schedules.py
====================================
CNN supervised subject-calibration: naive (15-epoch) vs regularised (3-epoch)
fine-tuning schedules, as a function of labelled windows/class K, on SIAT-LLMD
LOSO. Produces Results Section 4.15.

Expects results in:
  results_cnn_calibration/cnn_calibration_summary.csv            naive, 15 epochs
  results_cnn_calibration_ftepochs3/cnn_calibration_summary.csv  regularised, 3 epochs
  results_cnn_calibration_ftepochs3/cnn_calibration_subjectwise.csv (for CI band + tests)

Outputs (report_figs/new_experiments/):
  calibration_f1_vs_k.png       Figure 4.19 (F1 vs K, both schedules, CI band)
  cnn_calibration_table.csv     Table 4.15 (macro-F1 by K, both schedules)
  cnn_calibration_significance.csv  K-vs-K0 Wilcoxon for both schedules (naive n.s., regularised sig)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hashlib
from scipy.stats import wilcoxon, norm

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)
C_NAIVE, C_REG = "#b2182b", "#08519c"

def summ(p):
    d = pd.read_csv(ROOT/p); col=[c for c in d.columns if c.lower() in ("f1_mean","f1_macro_mean","f1_macro")][0]
    return {int(k): float(v) for k,v in zip(d["K_per_class"], d[col])}
naive = summ("results_cnn_calibration/cnn_calibration_summary.csv")
reg   = summ("results_cnn_calibration_ftepochs3/cnn_calibration_summary.csv")
Ks = sorted(naive)

sw = pd.read_csv(ROOT/"results_cnn_calibration_ftepochs3"/"cnn_calibration_subjectwise.csv")
def kvec(K): return sw[sw.K_per_class==K].sort_values("subject")["f1_macro"].to_numpy(float)
def bca(x, n=10000):
    x = np.asarray(x, float)
    seed = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float64).tobytes(), digest_size=4).digest(), 'big')
    rng = np.random.default_rng(seed)
    th = x.mean()
    bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    z0 = norm.ppf(min(max((bs < th).mean(), 1e-4), 1-1e-4))
    jk = np.array([np.delete(x, i).mean() for i in range(len(x))]); jm = jk.mean()
    den = 6*(((jm-jk)**2).sum()**1.5); a = (((jm-jk)**3).sum()/den) if den else 0
    q = lambda al: np.percentile(bs, 100*norm.cdf(z0 + (z0+norm.ppf(al))/(1-a*(z0+norm.ppf(al)))))
    return q(.025), q(.975)

t = pd.DataFrame([{"schedule":"naive_15ep", **{f"K{K}":round(naive[K],4) for K in Ks}},
                  {"schedule":"regularised_3ep", **{f"K{K}":round(reg[K],4) for K in Ks}}])
t.to_csv(OUT/"cnn_calibration_table.csv", index=False)
print("Table 4.15 (calibration):"); print(t.to_string(index=False))
# --- K-vs-K0 significance for BOTH schedules (naive shows the overfitting is n.s.) ---
sw_naive = pd.read_csv(ROOT/"results_cnn_calibration"/"cnn_calibration_subjectwise.csv")
def kvec_naive(K): return sw_naive[sw_naive.K_per_class==K].sort_values("subject")["f1_macro"].to_numpy(float)
sig_rows=[]
for sched, means, kfn in [("naive_15ep", naive, kvec_naive), ("regularised_3ep", reg, kvec)]:
    k0v = kfn(0)
    for K in [5,10,20]:
        _,p = wilcoxon(kfn(K), k0v)
        sig_rows.append(dict(schedule=sched, K=K, delta_pp=round((means[K]-means[0])*100,1),
                             wilcoxon_p=round(p,4), sig=("yes" if p<0.05 else "no")))
        print(f"  {sched} K{K} vs K0: {(means[K]-means[0])*100:+.1f}pp p={p:.4f} {'(sig)' if p<0.05 else '(n.s.)'}")
pd.DataFrame(sig_rows).to_csv(OUT/"cnn_calibration_significance.csv", index=False)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(Ks, [naive[K] for K in Ks], "s--", color=C_NAIVE, label="Naive (15 epochs)", lw=2)
ax.plot(Ks, [reg[K] for K in Ks], "o-", color=C_REG, label="Regularised (3 epochs)", lw=2)
band_K = [K for K in Ks if K!=5]; lo=[]; hi=[]
for K in band_K: a,b=bca(kvec(K)); lo.append(a); hi.append(b)
ax.fill_between(band_K, lo, hi, color=C_REG, alpha=0.15)
ax.axhline(reg[0], color="gray", ls=":", alpha=0.6, label="No calibration (K=0)")
ax.set_xticks(Ks); ax.set_xlabel("Labelled windows per class (K)"); ax.set_ylabel("LOSO macro-F1"); ax.grid(alpha=0.3)
ax.legend(fontsize=9, loc="lower right"); ax.set_title("CNN subject calibration: regularised fine-tuning gives a monotonic lift")
fig.tight_layout(); fig.savefig(OUT/"calibration_f1_vs_k.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved calibration_f1_vs_k.png + cnn_calibration_table.csv")

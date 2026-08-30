#!/usr/bin/env python3
"""
stats_new_experiments.py
========================
Single source of truth for the statistics of the four new experiments
(external validation, CORAL, causal normalisation, CNN calibration).
Computes BCa 95% CIs for every reported mean, paired Wilcoxon + Cohen's d
for every headline comparison, and Holm-Bonferroni + Benjamini-Hochberg
(FDR) correction across the whole 11-comparison family. Produces the data
behind Results Tables 4.16 (cross-dataset synthesis) and 4.17 (FDR family).

Expects results in (subjectwise CSVs):
  results_loso_freq/ , results_loso_freq_persubj/          SIAT SVM/RF (global, per-subject)
  results_cnn_loso/ , results_cnn_loso_norm_persubj/       SIAT CNN     (global, per-subject)
  results_ext_persubj/ , results_ext_global/               ENABL3S SVM/RF
  results_ext_cnn_persubj/ , results_ext_cnn_global/       ENABL3S CNN
  results_loso_freq_coral/                                  CORAL
  results_loso_freq_streaming/                              causal (transductive/calib100)
  results_cnn_calibration_ftepochs3/                        calibration (3-epoch)

Outputs (report_figs/new_experiments/):
  new_experiments_stats_fdr.csv   Table 4.17 (comparison, delta, d, p, Holm, BH, sig)
  new_experiments_cis.csv         BCa 95% CIs for every reported mean
  cross_dataset_synthesis.csv     Table 4.16 (SIAT vs ENABL3S, per-subject vs global)
"""
from pathlib import Path
import numpy as np, pandas as pd
import hashlib
from scipy.stats import wilcoxon, norm

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

def f1_of(df):
    c=[x for x in df.columns if x.lower()=="f1_macro"][0]
    key=[x for x in df.columns if x.lower() in ("heldout_subject","subject","subj")]
    return (df.sort_values(key[0]) if key else df)[c].to_numpy(float)
def vec(g):
    p=sorted(ROOT.glob(g)); assert p, f"no file: {g}"; return f1_of(pd.read_csv(p[0]))
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
def dp(a,b): d=np.asarray(a,float)-np.asarray(b,float); return d.mean()/d.std(ddof=1)

# ---- load vectors ----
V = {
 "SIAT_glob_SVM":vec("results_loso_freq/*SVM_nested_loso_subjectwise.csv"),
 "SIAT_glob_RF": vec("results_loso_freq/*RF_nested_loso_subjectwise.csv"),
 "SIAT_ps_SVM":  vec("results_loso_freq_persubj/*SVM_nested_loso_subjectwise.csv"),
 "SIAT_ps_RF":   vec("results_loso_freq_persubj/*RF_nested_loso_subjectwise.csv"),
 "SIAT_glob_CNN":vec("results_cnn_loso/per_subject_metrics_cnn_loso.csv"),
 "SIAT_ps_CNN":  vec("results_cnn_loso_norm_persubj/per_subject_metrics_cnn_loso.csv"),
 "EXT_ps_SVM":vec("results_ext_persubj/*SVM_nested_loso_subjectwise.csv"),
 "EXT_gl_SVM":vec("results_ext_global/*SVM_nested_loso_subjectwise.csv"),
 "EXT_ps_RF": vec("results_ext_persubj/*RF_nested_loso_subjectwise.csv"),
 "EXT_gl_RF": vec("results_ext_global/*RF_nested_loso_subjectwise.csv"),
 "EXT_ps_CNN":vec("results_ext_cnn_persubj/*subject*.csv"),
 "EXT_gl_CNN":vec("results_ext_cnn_global/*subject*.csv"),
 "CORAL_SVM":vec("results_loso_freq_coral/coral_SVM_subjectwise.csv"),
 "CORAL_RF": vec("results_loso_freq_coral/coral_RF_subjectwise.csv"),
 "c100_SVM":vec("results_loso_freq_streaming/streaming_calib100_SVM_subjectwise.csv"),
 "c100_RF": vec("results_loso_freq_streaming/streaming_calib100_RF_subjectwise.csv"),
 "tr_SVM":  vec("results_loso_freq_streaming/streaming_transductive_SVM_subjectwise.csv"),
 "tr_RF":   vec("results_loso_freq_streaming/streaming_transductive_RF_subjectwise.csv"),
}
cal = pd.read_csv(ROOT/"results_cnn_calibration_ftepochs3"/"cnn_calibration_subjectwise.csv")
kv = lambda K: cal[cal.K_per_class==K].sort_values("subject")["f1_macro"].to_numpy(float)

# ---- BCa CIs for every reported mean ----
ci_rows=[]
for name,v in list(V.items())+[("calib3ep_K0",kv(0)),("calib3ep_K10",kv(10)),("calib3ep_K20",kv(20))]:
    lo,hi=bca(v); ci_rows.append(dict(quantity=name, mean=round(v.mean(),4), ci_lo=round(lo,3), ci_hi=round(hi,3), n=len(v)))
pd.DataFrame(ci_rows).to_csv(OUT/"new_experiments_cis.csv", index=False)

# ---- 11-comparison family ----
tests=[
 ("SVM external per-subj vs global (n=10)", V["EXT_ps_SVM"], V["EXT_gl_SVM"]),
 ("RF external per-subj vs global (n=10)",  V["EXT_ps_RF"],  V["EXT_gl_RF"]),
 ("CNN external per-subj vs global (n=10)", V["EXT_ps_CNN"], V["EXT_gl_CNN"]),
 ("Per-subj vs CORAL, SVM",  V["SIAT_ps_SVM"], V["CORAL_SVM"]),
 ("Per-subj vs CORAL, RF",   V["SIAT_ps_RF"],  V["CORAL_RF"]),
 ("Causal calib-100 vs global, SVM", V["c100_SVM"], V["SIAT_glob_SVM"]),
 ("Causal calib-100 vs global, RF",  V["c100_RF"],  V["SIAT_glob_RF"]),
 ("Causal calib-100 vs transductive, SVM", V["c100_SVM"], V["tr_SVM"]),
 ("Causal calib-100 vs transductive, RF",  V["c100_RF"],  V["tr_RF"]),
 ("Calibration (3 ep) K=10 vs K=0", kv(10), kv(0)),
 ("Calibration (3 ep) K=20 vs K=0", kv(20), kv(0)),
]
lab=[t[0] for t in tests]; delt=[]; ds=[]; ps=[]
for _,a,b in tests:
    w,p=wilcoxon(a,b); delt.append((a.mean()-b.mean())*100); ds.append(dp(a,b)); ps.append(p)
ps=np.array(ps); mfam=len(ps)
holm=np.empty(mfam); hm=0
for r,i in enumerate(np.argsort(ps)): hm=max(hm,min(1,(mfam-r)*ps[i])); holm[i]=hm
bh=np.empty(mfam); prev=1
for r,i in enumerate(np.argsort(ps)[::-1]): prev=min(prev,ps[i]*mfam/(mfam-r)); bh[i]=prev
fdr=pd.DataFrame(dict(comparison=lab, delta_pp=np.round(delt,1), cohens_d=np.round(ds,2),
                      p=np.round(ps,4), p_holm=np.round(holm,4), p_BH=np.round(bh,4),
                      sig_BH=np.where(bh<0.05,"Yes","No")))
fdr.to_csv(OUT/"new_experiments_stats_fdr.csv", index=False)
print("Table 4.17 (FDR family, m=%d):"%mfam); print(fdr.to_string(index=False))

# ---- Table 4.16 cross-dataset synthesis ----
syn=[]
for mdl in ["SVM","RF","CNN"]:
    sg,sp=V[f"SIAT_glob_{mdl}"].mean(),V[f"SIAT_ps_{mdl}"].mean()
    eg,ep=V[f"EXT_gl_{mdl}"].mean(),V[f"EXT_ps_{mdl}"].mean()
    syn.append(dict(model=mdl, SIAT_global=round(sg,3), SIAT_persubj=round(sp,3), SIAT_delta_pp=round((sp-sg)*100,1),
                    ENABL3S_global=round(eg,3), ENABL3S_persubj=round(ep,3), ENABL3S_delta_pp=round((ep-eg)*100,1)))
pd.DataFrame(syn).to_csv(OUT/"cross_dataset_synthesis.csv", index=False)
print("\nTable 4.16 (synthesis):"); print(pd.DataFrame(syn).to_string(index=False))
print("\nSaved new_experiments_stats_fdr.csv, new_experiments_cis.csv, cross_dataset_synthesis.csv")

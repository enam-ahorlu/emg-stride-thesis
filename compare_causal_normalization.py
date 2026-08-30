#!/usr/bin/env python3
"""
compare_causal_normalization.py
===============================
Quantifies how much of the offline (transductive) per-subject normalisation
gain survives under causal / streaming estimators (calibration buffer + running
mean-var), on SIAT-LLMD LOSO. Produces Results Section 4.14.

Expects results in:
  results_loso_freq_streaming/streaming_norm_summary_FULL.csv   (config,model,f1_macro_mean)
  results_loso_freq_streaming/streaming_{config}_{model}_subjectwise.csv
  results_loso_freq/*{SVM,RF}*subjectwise.csv                   (global baseline)

Outputs (report_figs/new_experiments/):
  causal_retention.png          Figure 4.18 (retention line plot vs global + transductive)
  causal_normalization_table.csv Table 4.14 (all configs, SVM + RF)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)
C_SVM, C_RF, C_HL = "#2c7fb8", "#d95f0e", "#08519c"

def f1_of(df):
    c=[x for x in df.columns if x.lower()=="f1_macro"][0]
    key=[x for x in df.columns if x.lower() in ("heldout_subject","subject","subj")]
    return (df.sort_values(key[0]) if key else df)[c].to_numpy(float)
def vec(g):
    p=sorted(ROOT.glob(g)); assert p, f"no file: {g}"; return f1_of(pd.read_csv(p[0]))

summ = pd.read_csv(ROOT/"results_loso_freq_streaming"/"streaming_norm_summary_FULL.csv")
def m(cfg, model): return float(summ[(summ.config==cfg)&(summ.model==model)]["f1_macro_mean"].iloc[0])
glob_svm, glob_rf = vec("results_loso_freq/*SVM_nested_loso_subjectwise.csv").mean(), vec("results_loso_freq/*RF_nested_loso_subjectwise.csv").mean()

order = ["global","calib25","calib50","calib100","running","transductive"]
label = {"global":"global","calib25":"calib-25","calib50":"calib-50","calib100":"calib-100","running":"running","transductive":"transductive"}
rows=[]
for cfg in order:
    svm = glob_svm if cfg=="global" else m(cfg,"SVM")
    rf  = glob_rf  if cfg=="global" else m(cfg,"RF")
    rows.append(dict(normaliser=label[cfg], SVM_f1=round(svm,4), RF_f1=round(rf,4)))
t = pd.DataFrame(rows); t.to_csv(OUT/"causal_normalization_table.csv", index=False)
print("Table 4.14 (causal):"); print(t.to_string(index=False))
# paired significance: calib100 vs global and vs transductive
for model,gl in [("SVM",glob_svm),("RF",glob_rf)]:
    c100=vec(f"results_loso_freq_streaming/streaming_calib100_{model}_subjectwise.csv")
    tr  =vec(f"results_loso_freq_streaming/streaming_transductive_{model}_subjectwise.csv")
    glv =vec(f"results_loso_freq/*{model}_nested_loso_subjectwise.csv")
    _,p_g=wilcoxon(c100,glv); _,p_t=wilcoxon(c100,tr)
    print(f"  {model}: calib100 vs global p={p_g:.4f} ({(c100.mean()-glv.mean())*100:+.1f}pp) | vs transductive p={p_t:.4f} ({(c100.mean()-tr.mean())*100:+.1f}pp)")

fig, ax = plt.subplots(figsize=(7, 4.4))
xs = np.arange(len(order))
svm = [rows[i]["SVM_f1"] for i in range(len(order))]; rf = [rows[i]["RF_f1"] for i in range(len(order))]
ax.plot(xs, svm, "o-", color=C_SVM, label="SVM", lw=2); ax.plot(xs, rf, "s--", color=C_RF, label="RF", lw=2)
ax.axhline(glob_svm, color=C_SVM, ls=":", alpha=0.5); ax.axhline(glob_rf, color=C_RF, ls=":", alpha=0.5)
ax.set_xticks(xs); ax.set_xticklabels([label[c] for c in order], rotation=20); ax.set_ylabel("LOSO macro-F1"); ax.grid(alpha=0.3)
ax.legend(fontsize=9); ax.set_title("Deployability: how much of the gain survives causal normalisation")
fig.tight_layout(); fig.savefig(OUT/"causal_retention.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved causal_retention.png + causal_normalization_table.csv")

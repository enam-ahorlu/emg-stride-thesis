#!/usr/bin/env python3
"""
compare_coral_baseline.py
=========================
Compares label-free per-subject z-score normalisation against the CORAL
unsupervised domain-adaptation baseline (Sun, Feng & Saenko 2016) and the
global-norm baseline, on SIAT-LLMD LOSO. Produces Results Section 4.13.

Expects results in:
  results_loso_freq/           SIAT SVM/RF LOSO, global-norm baseline (subjectwise)
  results_loso_freq_persubj/   SIAT SVM/RF LOSO, per-subject norm (subjectwise)
  results_loso_freq_coral/     CORAL LOSO (coral_{SVM,RF}_subjectwise.csv)

Outputs (report_figs/new_experiments/):
  coral_comparison.png          Figure 4.17 (grouped bar: global / CORAL / per-subject)
  coral_comparison_table.csv    Table 4.13 (with per-subject-vs-CORAL paired test)
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)
C_PS, C_GL, C_CORAL = "#2c7fb8", "#9e9e9e", "#41ab5d"

def f1_of(df):
    c = [x for x in df.columns if x.lower() == "f1_macro"][0]
    key = [x for x in df.columns if x.lower() in ("heldout_subject","subject","subj")]
    return (df.sort_values(key[0]) if key else df)[c].to_numpy(float)
def vec(g):
    p = sorted(ROOT.glob(g)); assert p, f"no file: {g}"; return f1_of(pd.read_csv(p[0]))
def d_paired(a, b):
    d = np.asarray(a,float)-np.asarray(b,float); return d.mean()/d.std(ddof=1)

D = {
 "SVM": dict(glob=vec("results_loso_freq/*SVM_nested_loso_subjectwise.csv"),
             coral=vec("results_loso_freq_coral/coral_SVM_subjectwise.csv"),
             ps=vec("results_loso_freq_persubj/*SVM_nested_loso_subjectwise.csv")),
 "RF":  dict(glob=vec("results_loso_freq/*RF_nested_loso_subjectwise.csv"),
             coral=vec("results_loso_freq_coral/coral_RF_subjectwise.csv"),
             ps=vec("results_loso_freq_persubj/*RF_nested_loso_subjectwise.csv")),
}
rows = []
for m, d in D.items():
    w, p = wilcoxon(d["ps"], d["coral"])
    rows.append(dict(model=m, global_f1=round(d["glob"].mean(),4), coral_f1=round(d["coral"].mean(),4),
                     per_subject_f1=round(d["ps"].mean(),4),
                     persubj_minus_coral_pp=round((d["ps"].mean()-d["coral"].mean())*100,1),
                     cohens_d=round(d_paired(d["ps"],d["coral"]),2), wilcoxon_p=round(p,6), n=len(d["ps"])))
t = pd.DataFrame(rows); t.to_csv(OUT/"coral_comparison_table.csv", index=False)
print("Table 4.13 (CORAL):"); print(t.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4.2))
x = np.arange(2); w = 0.26; models = ["SVM","RF"]
ax.bar(x-w, [D[m]["glob"].mean() for m in models], w, label="Global", color=C_GL)
ax.bar(x,   [D[m]["coral"].mean() for m in models], w, label="CORAL (UDA)", color=C_CORAL)
ax.bar(x+w, [D[m]["ps"].mean() for m in models], w, label="Per-subject", color=C_PS)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0.6, 0.82); ax.set_ylabel("LOSO macro-F1")
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="upper right")
ax.set_title("Label-free per-subject normalisation beats CORAL (SIAT)")
fig.tight_layout(); fig.savefig(OUT/"coral_comparison.png", dpi=150, bbox_inches="tight"); plt.close()
print("Saved coral_comparison.png + coral_comparison_table.csv")

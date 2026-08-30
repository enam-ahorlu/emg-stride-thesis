#!/usr/bin/env python3
"""
compare_external_validation.py
==============================
External-validation replication on the independent ENABL3S dataset
(Hu, Rouse & Hargrove 2018) vs the SIAT-LLMD headline, plus the ENABL3S
per-class / confusion analysis. Produces the Results Section 4.12 figures
and tables. All numbers are computed from the LOSO result CSVs.

Expects results in:
  results_loso_freq/              SIAT SVM/RF LOSO, global-norm baseline
  results_loso_freq_persubj/      SIAT SVM/RF LOSO, per-subject norm
  results_cnn_loso/               SIAT CNN LOSO, global-norm baseline
  results_cnn_loso_norm_persubj/  SIAT CNN LOSO, per-subject norm
  results_ext_persubj/  results_ext_global/           ENABL3S SVM/RF LOSO
  results_ext_cnn_persubj/  results_ext_cnn_global/    ENABL3S CNN LOSO
  results_ext_persubj/confusion_matrices/*SVM*LOSO_confusion.csv

Outputs (report_figs/new_experiments/):
  external_persubj_vs_global.png     Figure 4.15 (grouped bar, SIAT + ENABL3S)
  enabl3s_confusion.png              Figure 4.16 (row-normalised confusion)
  external_validation_table.csv      Table 4.12 (per-subj vs global, ENABL3S)
  enabl3s_per_class_f1.csv           per-class F1 (ENABL3S, per-subject SVM)
  enabl3s_confusion_matrix.csv       row-normalised confusion (%)
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
C_PS, C_GL, C_HL = "#2c7fb8", "#9e9e9e", "#08519c"

def f1_of(df):
    c = [x for x in df.columns if x.lower() == "f1_macro"][0]
    key = [x for x in df.columns if x.lower() in ("heldout_subject","subject","subj")]
    return (df.sort_values(key[0]) if key else df)[c].to_numpy(float)

def vec(path_glob):
    p = sorted(ROOT.glob(path_glob))
    assert p, f"no file matches {path_glob}"
    return f1_of(pd.read_csv(p[0]))

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

def d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float); return d.mean()/d.std(ddof=1)

SIAT = {
 "SVM": (vec("results_loso_freq_persubj/*SVM_nested_loso_subjectwise.csv"), vec("results_loso_freq/*SVM_nested_loso_subjectwise.csv")),
 "RF":  (vec("results_loso_freq_persubj/*RF_nested_loso_subjectwise.csv"),  vec("results_loso_freq/*RF_nested_loso_subjectwise.csv")),
 "CNN": (vec("results_cnn_loso_norm_persubj/per_subject_metrics_cnn_loso.csv"), vec("results_cnn_loso/per_subject_metrics_cnn_loso.csv")),
}
EXT = {
 "SVM": (vec("results_ext_persubj/*SVM_nested_loso_subjectwise.csv"), vec("results_ext_global/*SVM_nested_loso_subjectwise.csv")),
 "RF":  (vec("results_ext_persubj/*RF_nested_loso_subjectwise.csv"),  vec("results_ext_global/*RF_nested_loso_subjectwise.csv")),
 "CNN": (vec("results_ext_cnn_persubj/*subject*.csv"), vec("results_ext_cnn_global/*subject*.csv")),
}

rows = []
for m in ["SVM","RF","CNN"]:
    ps, gl = EXT[m]; lo, hi = bca(ps); w, p = wilcoxon(ps, gl)
    rows.append(dict(model=m, per_subject_f1=round(ps.mean(),4), ci_lo=round(lo,3), ci_hi=round(hi,3),
                     global_f1=round(gl.mean(),4), delta_pp=round((ps.mean()-gl.mean())*100,1),
                     cohens_d=round(d_paired(ps,gl),2), wilcoxon_p=round(p,4), n=len(ps)))
t412 = pd.DataFrame(rows); t412.to_csv(OUT/"external_validation_table.csv", index=False)
print("Table 4.12 (ENABL3S per-subject vs global):"); print(t412.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, (name, D) in zip(axes, [("SIAT (n=40)", SIAT), ("ENABL3S (n=10)", EXT)]):
    x = np.arange(3); w = 0.38; models = ["SVM","RF","CNN"]
    ps = [D[m][0].mean() for m in models]; gl = [D[m][1].mean() for m in models]
    ax.bar(x-w/2, gl, w, label="Global", color=C_GL)
    ax.bar(x+w/2, ps, w, label="Per-subject", color=C_PS)
    for i in range(3):
        ax.text(x[i]+w/2, ps[i]+0.015, f"+{(ps[i]-gl[i])*100:.0f}", ha="center", fontsize=9, color=C_HL, fontweight="bold")
    ax.set_title(name); ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0, 0.95); ax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("LOSO macro-F1"); axes[0].legend(loc="upper right", fontsize=9)
fig.suptitle("Per-subject vs global normalisation replicates across datasets", fontsize=12)
fig.tight_layout(); fig.savefig(OUT/"external_persubj_vs_global.png", dpi=150, bbox_inches="tight"); plt.close()

cmf = sorted(ROOT.glob("results_ext_persubj/confusion_matrices/*SVM*LOSO_confusion.csv"))[0]
cm = pd.read_csv(cmf, index_col=0); labels = list(cm.index)
rn = (cm.div(cm.sum(1), axis=0) * 100).round(0).astype(int); rn.to_csv(OUT/"enabl3s_confusion_matrix.csv")
M = cm.to_numpy(float); pc = {}
for i, lab in enumerate(labels):
    tp = M[i,i]; fn = M[i].sum()-tp; fp = M[:,i].sum()-tp
    pr = tp/(tp+fp) if tp+fp else 0; rc = tp/(tp+fn) if tp+fn else 0
    pc[lab] = round(2*pr*rc/(pr+rc) if pr+rc else 0, 3)
pd.DataFrame([{"class":k,"f1":v} for k,v in pc.items()]).to_csv(OUT/"enabl3s_per_class_f1.csv", index=False)
print("\nENABL3S per-class F1 (per-subject SVM):", pc)

fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(rn.to_numpy(), cmap="Blues", vmin=0, vmax=100)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(len(labels)):
    for j in range(len(labels)):
        v = rn.to_numpy()[i,j]; ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v>50 else "black", fontsize=11)
dns_wak = rn.to_numpy()[labels.index("DNS"), labels.index("WAK")]
ax.set_title(f"ENABL3S confusion (%, per-subject SVM)\nDNS->WAK = {dns_wak}% mirrors SIAT")
fig.colorbar(im, fraction=0.046, pad=0.04); fig.tight_layout()
fig.savefig(OUT/"enabl3s_confusion.png", dpi=150, bbox_inches="tight"); plt.close()
print("\nSaved 5 outputs to report_figs/new_experiments/")

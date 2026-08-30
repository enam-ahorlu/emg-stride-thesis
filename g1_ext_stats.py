#!/usr/bin/env python3
"""
g1_ext_stats.py
===============
Paired Wilcoxon + Cohen's d for the G1 ENABL3S external-validation comparisons
(EXPERIMENT_PLAN_GAPS.md), mirroring the g3/g4/g6/g7_*_stats.csv schema so they
fold into stats_unified_fdr.py the same way.

Comparisons:
  1. ResNet-SE per-subject vs global norm (ENABL3S)            -- core normalisation replication
  2. Deep CORAL (resnet_se) vs per-subject headline (ENABL3S)  -- deployable-adaptation check
  3. AdaBN post vs pre (resnet_se, ENABL3S)                     -- AdaBN lift is real
  4. AdaBN post vs per-subject headline (ENABL3S)                -- deployable-adaptation check
  5. Best ensemble-v2 (weighted_soft) vs hard vote, SVM+RF+RESNET_SE (ENABL3S)
  6. Best ensemble-v2 (weighted_soft) vs best single model SVM (ENABL3S)

Output: report_figs/new_experiments/g1_ext_stats.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)
SUBS = [156, 185, 186, 188, 189, 190, 191, 192, 193, 194]

def d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d.mean() / d.std(ddof=1)

def row(comparison, a, b):
    w, p = wilcoxon(a, b)
    return dict(comparison=comparison,
                delta_pp=round((np.mean(a) - np.mean(b)) * 100, 2),
                cohens_d=round(d_paired(a, b), 3),
                p=p)

persubj = pd.read_csv(ROOT / "results_ext_resnet_se_persubj" / "cnn_arch_subjectwise.csv").sort_values("subject")["f1_macro"].to_numpy(float)
glob    = pd.read_csv(ROOT / "results_ext_resnet_se_global"  / "cnn_arch_subjectwise.csv").sort_values("subject")["f1_macro"].to_numpy(float)
coral   = pd.read_csv(ROOT / "results_ext_deepcoral" / "deep_coral_subjectwise.csv").sort_values("subject")["f1_macro"].to_numpy(float)
adabn   = pd.read_csv(ROOT / "results_ext_adabn" / "adabn_subjectwise.csv").sort_values("subject")
adabn_post = adabn["f1_macro"].to_numpy(float)
adabn_pre  = adabn["f1_pre_adabn"].to_numpy(float)

ens = pd.read_csv(ROOT / "results_ext_ensemble_v2" / "ensemble_v2_subjectwise.csv")
ens_weighted = ens["SVM+RF+RESNET_SE [weighted_soft]"].to_numpy(float)
ens_hard     = ens["SVM+RF+RESNET_SE [hard]"].to_numpy(float)

PROBA = ROOT / "results_ext_ensemble_v2" / "proba"
svm_f1 = np.array([
    f1_score(np.load(PROBA / f"SVM_sub{s}.npz")["y_true"].astype(int),
              np.load(PROBA / f"SVM_sub{s}.npz")["proba"].argmax(1), average="macro")
    for s in SUBS
])

rows = [
    row("ResNet-SE ENABL3S per-subject vs global norm", persubj, glob),
    row("Deep CORAL (resnet_se, ENABL3S) vs per-subject headline (0.565)", coral, persubj),
    row("AdaBN post vs pre (resnet_se, ENABL3S)", adabn_post, adabn_pre),
    row("AdaBN post (resnet_se, ENABL3S) vs per-subject headline (0.565)", adabn_post, persubj),
    row("Best ensemble-v2 (SVM+RF+RESNET_SE weighted_soft) vs hard vote (ENABL3S)", ens_weighted, ens_hard),
    row("Best ensemble-v2 (SVM+RF+RESNET_SE weighted_soft) vs best single model SVM (ENABL3S)", ens_weighted, svm_f1),
]
df = pd.DataFrame(rows)
df.to_csv(OUT / "g1_ext_stats.csv", index=False)
print(df.to_string(index=False))
print("\nwrote", OUT / "g1_ext_stats.csv")

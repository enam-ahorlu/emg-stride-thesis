#!/usr/bin/env python3
"""
stats_unified_fdr.py
====================
Most conservative multiple-comparison check: pool EVERY paired comparison in the
thesis into a single family and apply Holm-Bonferroni + Benjamini-Hochberg (FDR).
This answers the "cherry-picking" objection across all experiments at once.

Family = the SIAT optimisation-ablation tests (Section 4.11) + the 11 new-experiment
tests (Section 4.16) + the 9 July-2026 second-pass tests (LDA, resnet_se/Deep
CORAL/AdaBN vs the CNN headline, STDUP class-balance control) + the 1 ensemble-v2
combiner test (best soft/weighted-soft ensemble vs the original hard-vote ensemble)
+ the EXPERIMENT_PLAN_GAPS.md (G1-G7) tests extending the robustness/deployability
analyses to the headline models (ResNet-SE, soft ensemble): G1 ENABL3S external
validation, G3 causal AdaBN, G4 ResNet-SE calibration (single- and multi-draw), G6
STDUP class-balance control on deep models, G7 ResNet-SE SD-vs-LOSO gap.

Expects:
  report_figs/optimization_wilcoxon_table.csv                (SIAT ablation tests)
  report_figs/new_experiments/new_experiments_stats_fdr.csv  (new-experiment tests)
  report_figs/new_experiments/july2_stats_fdr.csv            (July-2026 second-pass tests)
  report_figs/new_experiments/ensemble_v2_stats_fdr.csv      (ensemble-v2 combiner test)
  report_figs/new_experiments/g1_ext_stats.csv                        (G1 ENABL3S)
  report_figs/new_experiments/g3_adabn_causal_stats.csv                (G3 causal AdaBN)
  report_figs/new_experiments/g4_resnet_se_calibration_stats.csv       (G4 calibration, single-draw)
  report_figs/new_experiments/g4_resnet_se_calibration_multidraw_stats.csv (G4 calibration, multi-draw)
  report_figs/new_experiments/g6_stdup_cnn_stats.csv                   (G6 STDUP balance, deep models)
  report_figs/new_experiments/g7_resnet_se_gap_stats.csv               (G7 SD-vs-LOSO gap)
  report_figs/new_experiments/g8_resnet_se_augmentation_stats.csv      (G8 ResNet-SE augmentation ablation)
  report_figs/new_experiments/g9_chandrop_stats.csv                    (G9 ResNet-SE+CD promotion: ext/calib/DA)
  report_figs/new_experiments/critique_stats_fdr.csv                   (EXPERIMENT_PLAN_CRITIQUE.md E1-E5)

Outputs (report_figs/new_experiments/):
  unified_fdr_all_experiments.csv   every test with Holm + BH across the whole family
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)

rows = []
# SIAT optimisation-ablation tests
opt = pd.read_csv(ROOT/"report_figs"/"optimization_wilcoxon_table.csv")
for _, r in opt.iterrows():
    rows.append(dict(family="SIAT-ablation (§4.11)",
                     comparison=f"{r['comparison_name']} [{r['model']}] {r['condition_a']}→{r['condition_b']}",
                     delta=round(float(r['delta'])*100,1), d=round(float(r['cohen_d']),2), p=float(r['p_value'])))
# new-experiment tests
new = pd.read_csv(OUT/"new_experiments_stats_fdr.csv")
for _, r in new.iterrows():
    rows.append(dict(family="new-experiment (§4.16)", comparison=r['comparison'],
                     delta=float(r['delta_pp']), d=float(r['cohens_d']), p=float(r['p'])))
# July-2026 second-pass tests (LDA, resnet_se/Deep CORAL/AdaBN, STDUP control)
july2 = pd.read_csv(OUT/"july2_stats_fdr.csv")
for _, r in july2.iterrows():
    rows.append(dict(family="July-2026 second pass", comparison=r['comparison'],
                     delta=float(r['delta_pp']), d=float(r['cohens_d']), p=float(r['p'])))
# ensemble-v2 combiner test (best ensemble vs original hard vote)
ens = pd.read_csv(OUT/"ensemble_v2_stats_fdr.csv")
for _, r in ens.iterrows():
    rows.append(dict(family="ensemble-v2 combiner", comparison=r['comparison'],
                     delta=float(r['delta_pp']), d=float(r['cohens_d']), p=float(r['p'])))
# EXPERIMENT_PLAN_GAPS.md (G1-G7) tests, headline-model extensions
for fname, family in [("g1_ext_stats.csv", "G1 external validation (ENABL3S)"),
                       ("g3_adabn_causal_stats.csv", "G3 causal AdaBN (resnet_se)"),
                       ("g4_resnet_se_calibration_stats.csv", "G4 ResNet-SE calibration"),
                       ("g4_resnet_se_calibration_multidraw_stats.csv", "G4 ResNet-SE calibration"),
                       ("g6_stdup_cnn_stats.csv", "G6 STDUP balance (deep models)"),
                       ("g7_resnet_se_gap_stats.csv", "G7 ResNet-SE SD-vs-LOSO gap"),
                       ("g8_resnet_se_augmentation_stats.csv", "G8 ResNet-SE augmentation ablation"),
                       ("g9_chandrop_stats.csv", "G9 ResNet-SE+CD promotion (Phases 3-6)")]:
    g = pd.read_csv(OUT/fname)
    for _, r in g.iterrows():
        rows.append(dict(family=family, comparison=r['comparison'],
                         delta=float(r['delta_pp']), d=float(r['cohens_d']), p=float(r['p'])))

# EXPERIMENT_PLAN_CRITIQUE.md (E1-E5): external-critique response experiments
# (variance decomposition, causal deployable ensemble, within-subject label
# budget, RF calibration). Note: E1's two distance-metric comparisons carry
# "delta_pp" units that are NOT percentage points of F1 (MMD/Mahalanobis are
# unbounded distances) -- kept in this column only for CSV-schema consistency
# with every other family; read those two rows' delta as raw distance-metric
# reduction, not pp.
critique_path = OUT / "critique_stats_fdr.csv"
if critique_path.exists():
    crit = pd.read_csv(critique_path)
    for _, r in crit.iterrows():
        rows.append(dict(family="EXPERIMENT_PLAN_CRITIQUE (E1-E5)", comparison=r['comparison'],
                         delta=float(r['delta_pp']), d=float(r['cohens_d']), p=float(r['p'])))

df = pd.DataFrame(rows)
p = df["p"].to_numpy(float); m = len(p)
# Holm
holm = np.empty(m); hm = 0.0
for rank, i in enumerate(np.argsort(p)): hm = max(hm, min(1, (m-rank)*p[i])); holm[i] = hm
# BH
bh = np.empty(m); prev = 1.0
for rank, i in enumerate(np.argsort(p)[::-1]): prev = min(prev, p[i]*m/(m-rank)); bh[i] = prev
df["p_holm"] = np.round(holm, 4); df["p_BH"] = np.round(bh, 4)
df["sig_BH"] = np.where(bh < 0.05, "Yes", "No")
df = df.sort_values("p").reset_index(drop=True)
df.to_csv(OUT/"unified_fdr_all_experiments.csv", index=False)

n_sig = int((df.sig_BH == "Yes").sum())
print(f"Unified family: m = {m} paired comparisons across all experiments.")
print(f"Survive BH-FDR (<0.05): {n_sig}/{m};  fail: {m-n_sig}/{m}")
print("\nComparisons that do NOT survive BH-FDR (expected: the already-marginal ones):")
print(df[df.sig_BH=="No"][["comparison","delta","d","p","p_BH"]].to_string(index=False))
print("\nAll headline effects (normalisation, end-to-end, external x3, CORAL x2, calibration) BH-adjusted p:")
for key in ["1_norm","6_end_to_end","external","CORAL","Calibration (3 ep) K=20"]:
    sub=df[df.comparison.str.contains(key, case=False, regex=False)]
    for _,r in sub.iterrows(): print(f"   {r['comparison'][:52]:52s} BH={r['p_BH']:.4f} {r['sig_BH']}")

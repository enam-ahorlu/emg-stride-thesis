#!/usr/bin/env python3
"""
stats_july2_experiments.py
===========================
Statistics for the second July-2026 experiment batch (external-review pass):
LDA-under-LOSO, the resnet_se/resnet CNN architecture comparison, Deep CORAL
for the CNN, AdaBN for the CNN, and the STDUP class-balance control. Mirrors
stats_new_experiments.py exactly (same BCa/Wilcoxon/Cohen's d machinery) so the
two families combine cleanly in stats_unified_fdr.py.

Expects (subjectwise CSVs, all already on disk from this pass):
  results_lda_persubj/lda_subjectwise.csv , results_lda_global/lda_subjectwise.csv
  results_cnn_loso_simple_repro/cnn_arch_subjectwise.csv (arch=simple)
  results_cnn_loso_resnet_se/cnn_arch_subjectwise.csv    (arch=resnet_se)
  results_cnn_loso_resnet/cnn_arch_subjectwise.csv       (arch=resnet)
  results_deep_coral_cnn_resnet_se/deep_coral_subjectwise.csv
  results_adabn_cnn_resnet_se/adabn_subjectwise.csv (f1_pre_adabn, f1_macro=post)
  results_stdup_subsample/stdup_subsample_subjectwise.csv (model, condition, macro_f1, f1_STDUP)
  results_loso_freq_persubj/*_{SVM,RF}_nested_loso_subjectwise.csv (headline SVM/RF)
  results_cnn_loso_norm_persubj/per_subject_metrics_cnn_loso.csv   (headline CNN, 0.754)

Outputs (report_figs/new_experiments/):
  july2_stats_fdr.csv   comparison, delta, d, p, Holm, BH, sig (this family alone)
  july2_cis.csv         BCa 95% CIs for every reported mean
"""
from pathlib import Path
import numpy as np, pandas as pd
import hashlib
from scipy.stats import wilcoxon, norm

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)


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


def dp(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d.mean() / d.std(ddof=1)


def load(path, subj_col="subject", val_col="f1_macro", **filters):
    df = pd.read_csv(ROOT / path)
    for k, v in filters.items():
        df = df[df[k] == v]
    df = df.sort_values(subj_col)
    return df[val_col].to_numpy(float)


# ---- load vectors (all sorted by subject id, 1..40) ----
V = {
    "LDA_ps": load("results_lda_persubj/lda_subjectwise.csv"),
    "LDA_gl": load("results_lda_global/lda_subjectwise.csv"),
    "SVM_ps": load("results_loso_freq_persubj/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__SVM_nested_loso_subjectwise.csv", subj_col="heldout_subject"),
    "RF_ps":  load("results_loso_freq_persubj/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext__RF_nested_loso_subjectwise.csv", subj_col="heldout_subject"),
    "CNN_ps": load("results_cnn_loso_norm_persubj/per_subject_metrics_cnn_loso.csv"),
    "simple_repro": load("results_cnn_loso_simple_repro/cnn_arch_subjectwise.csv", arch="simple"),
    "resnet":      load("results_cnn_loso_resnet/cnn_arch_subjectwise.csv", arch="resnet"),
    "resnet_se":   load("results_cnn_loso_resnet_se/cnn_arch_subjectwise.csv", arch="resnet_se"),
    "DeepCORAL":   load("results_deep_coral_cnn_resnet_se/deep_coral_subjectwise.csv"),
    "AdaBN_pre":   load("results_adabn_cnn_resnet_se/adabn_subjectwise.csv", val_col="f1_pre_adabn"),
    "AdaBN_post":  load("results_adabn_cnn_resnet_se/adabn_subjectwise.csv"),
}
stdup = pd.read_csv(ROOT / "results_stdup_subsample/stdup_subsample_subjectwise.csv")
sd = lambda model, cond, col: stdup[(stdup.model == model) & (stdup.condition == cond)].sort_values("subject")[col].to_numpy(float)
V.update({
    "STDUP_SVM_imb": sd("SVM", "imbalanced", "f1_STDUP"),
    "STDUP_SVM_bal": sd("SVM", "balanced", "f1_STDUP"),
    "STDUP_RF_imb":  sd("RF", "imbalanced", "f1_STDUP"),
    "STDUP_RF_bal":  sd("RF", "balanced", "f1_STDUP"),
    "macroF1_SVM_imb": sd("SVM", "imbalanced", "macro_f1"),
    "macroF1_SVM_bal": sd("SVM", "balanced", "macro_f1"),
    "macroF1_RF_imb":  sd("RF", "imbalanced", "macro_f1"),
    "macroF1_RF_bal":  sd("RF", "balanced", "macro_f1"),
})

for k, v in V.items():
    assert len(v) == 40, f"{k} has n={len(v)}, expected 40"

# ---- BCa CIs for every reported mean ----
ci_rows = []
for name, v in V.items():
    lo, hi = bca(v)
    ci_rows.append(dict(quantity=name, mean=round(v.mean(), 4), ci_lo=round(lo, 3), ci_hi=round(hi, 3), n=len(v)))
pd.DataFrame(ci_rows).to_csv(OUT / "july2_cis.csv", index=False)

# ---- comparison family ----
tests = [
    ("LDA per-subj vs SVM per-subj",  V["LDA_ps"], V["SVM_ps"]),
    ("LDA per-subj vs RF per-subj",   V["LDA_ps"], V["RF_ps"]),
    ("LDA per-subj vs CNN per-subj",  V["LDA_ps"], V["CNN_ps"]),
    ("LDA per-subj vs LDA global",    V["LDA_ps"], V["LDA_gl"]),
    ("resnet_se vs SimpleEMGCNN headline (0.754)", V["resnet_se"], V["CNN_ps"]),
    ("Deep CORAL (resnet_se) vs SimpleEMGCNN headline (0.754)", V["DeepCORAL"], V["CNN_ps"]),
    ("AdaBN post (resnet_se) vs SimpleEMGCNN headline (0.754)", V["AdaBN_post"], V["CNN_ps"]),
    ("STDUP-class F1 balanced vs imbalanced, SVM", V["STDUP_SVM_bal"], V["STDUP_SVM_imb"]),
    ("STDUP-class F1 balanced vs imbalanced, RF",  V["STDUP_RF_bal"],  V["STDUP_RF_imb"]),
]
lab = [t[0] for t in tests]; delt = []; ds = []; ps = []
for _, a, b in tests:
    w, p = wilcoxon(a, b); delt.append((a.mean() - b.mean()) * 100); ds.append(dp(a, b)); ps.append(p)
ps = np.array(ps); mfam = len(ps)
holm = np.empty(mfam); hm = 0
for r, i in enumerate(np.argsort(ps)): hm = max(hm, min(1, (mfam - r) * ps[i])); holm[i] = hm
bh = np.empty(mfam); prev = 1
for r, i in enumerate(np.argsort(ps)[::-1]): prev = min(prev, ps[i] * mfam / (mfam - r)); bh[i] = prev
fdr = pd.DataFrame(dict(comparison=lab, delta_pp=np.round(delt, 2), cohens_d=np.round(ds, 2),
                        p=np.round(ps, 4), p_holm=np.round(holm, 4), p_BH=np.round(bh, 4),
                        sig_BH=np.where(bh < 0.05, "Yes", "No")))
fdr.to_csv(OUT / "july2_stats_fdr.csv", index=False)
print(f"July-2 family (m={mfam}):")
print(fdr.to_string(index=False))

# ---- supplementary, architecture-isolating comparisons (descriptive; NOT pooled
#      into the formal FDR family, to avoid re-testing highly correlated contrasts) ----
supp = [
    ("resnet_se vs resnet (SE-attention ablation)", V["resnet_se"], V["resnet"]),
    ("resnet vs simple_repro (residual-only ablation)", V["resnet"], V["simple_repro"]),
    ("resnet_se+persubj vs Deep CORAL (same arch, isolates adaptation method)", V["resnet_se"], V["DeepCORAL"]),
    ("resnet_se+persubj vs AdaBN post (same arch, isolates adaptation method)", V["resnet_se"], V["AdaBN_post"]),
    ("AdaBN post vs pre (within-method adaptation effect)", V["AdaBN_post"], V["AdaBN_pre"]),
    ("STDUP macro-F1 balanced vs imbalanced, SVM", V["macroF1_SVM_bal"], V["macroF1_SVM_imb"]),
    ("STDUP macro-F1 balanced vs imbalanced, RF",  V["macroF1_RF_bal"],  V["macroF1_RF_imb"]),
]
srows = []
for name, a, b in supp:
    w, p = wilcoxon(a, b)
    srows.append(dict(comparison=name, delta_pp=round((a.mean() - b.mean()) * 100, 2), cohens_d=round(dp(a, b), 2), p=round(p, 4)))
supp_df = pd.DataFrame(srows)
supp_df.to_csv(OUT / "july2_supplementary_comparisons.csv", index=False)
print("\nSupplementary (descriptive, not in FDR family):")
print(supp_df.to_string(index=False))
print("\nSaved july2_stats_fdr.csv, july2_cis.csv, july2_supplementary_comparisons.csv")

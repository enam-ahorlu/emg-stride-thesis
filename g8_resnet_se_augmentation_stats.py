#!/usr/bin/env python3
"""
g8_resnet_se_augmentation_stats.py
===================================
Data-augmentation ablation on ResNet-SE under LOSO (per-subject norm),
mirroring the CNN augmentation experiment (Section 4.8 / Table 4.10) so the
augmentation analysis is no longer confined to SimpleEMGCNN.

For each augmentation vs the ResNet-SE 'none' baseline: paired Wilcoxon,
Cohen's d (paired), and a BCa 95% CI on the mean F1 (same bca()/d_paired()
formulas as compare_external_validation.py, for consistency with the rest
of the family).

Output: report_figs/new_experiments/g8_resnet_se_augmentation_stats.csv
        (schema: comparison, delta_pp, cohens_d, p -- matches g1/g3/g4/g6/g7)
        report_figs/new_experiments/g8_resnet_se_augmentation_table.csv
        (schema: augmentation, f1_mean, ci_lo, ci_hi, delta_pp, cohens_d, p -- full table for the thesis)
"""
from pathlib import Path
import hashlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, norm

ROOT = Path(__file__).parent
OUT = ROOT / "report_figs" / "new_experiments"; OUT.mkdir(parents=True, exist_ok=True)
AUGS = ["none", "gaussian", "chandrop", "timemask", "combined"]
LABELS = {"none": "No Augmentation", "gaussian": "Gaussian Noise", "chandrop": "Channel Dropout",
          "timemask": "Time Masking", "combined": "Combined"}

def vec(aug):
    df = pd.read_csv(ROOT / f"results_cnn_aug_resnet_se_{aug}" / "cnn_arch_subjectwise.csv")
    return df.sort_values("subject")["f1_macro"].to_numpy(float)

def d_paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return d.mean() / d.std(ddof=1)

def bca(x, n=10000):
    x = np.asarray(x, float)
    seed = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float64).tobytes(), digest_size=4).digest(), 'big')
    rng = np.random.default_rng(seed)
    th = x.mean()
    bs = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    z0 = norm.ppf(min(max((bs < th).mean(), 1e-4), 1-1e-4))
    jk = np.array([np.delete(x, i).mean() for i in range(len(x))]); jm = jk.mean()
    den = 6*(((jm-jk)**2).sum()**1.5); a_ = (((jm-jk)**3).sum()/den) if den else 0
    q = lambda al: np.percentile(bs, 100*norm.cdf(z0 + (z0+norm.ppf(al))/(1-a_*(z0+norm.ppf(al)))))
    return q(.025), q(.975)

V = {a: vec(a) for a in AUGS}
base = V["none"]

rows_table, rows_stats = [], []
for a in AUGS:
    x = V[a]
    lo, hi = bca(x)
    if a == "none":
        delta, d, p = 0.0, 0.0, 1.0
    else:
        w, p = wilcoxon(x, base)
        delta = (x.mean() - base.mean()) * 100
        d = d_paired(x, base)
        rows_stats.append(dict(comparison=f"ResNet-SE augmentation: {LABELS[a]} vs No Augmentation",
                               delta_pp=round(delta, 2), cohens_d=round(d, 3), p=p))
    rows_table.append(dict(augmentation=LABELS[a], f1_mean=round(x.mean(), 4),
                           ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                           delta_pp=round(delta, 2), cohens_d=round(d, 3),
                           p=(round(p, 6) if a != "none" else "-")))

table = pd.DataFrame(rows_table)
table.to_csv(OUT / "g8_resnet_se_augmentation_table.csv", index=False)
stats = pd.DataFrame(rows_stats)
stats.to_csv(OUT / "g8_resnet_se_augmentation_stats.csv", index=False)

print(table.to_string(index=False))
print("\nwrote", OUT / "g8_resnet_se_augmentation_table.csv")
print("wrote", OUT / "g8_resnet_se_augmentation_stats.csv")

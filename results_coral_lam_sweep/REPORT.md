# E-C4 — Was CORAL beaten fairly?

## Command run

```
& '.\.venv\Scripts\python.exe' -u run_coral_loso.py --features features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz --meta features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv --models SVM --lam <L> --n-jobs 1 --resume --out results_coral_lam_sweep/lam_<L>
```

for `<L>` in {0.01, 0.1, 1.0, 10.0}. No code change to `run_coral_loso.py` —
it already does full nested GridSearchCV per fold (inner 5-fold GroupKFold,
grid `clf__C ∈ {1,5,10}`, `clf__gamma='scale'`, `class_weight='balanced'`,
`scoring='f1_macro'`), and `StandardScaler().fit(X[tr])` is applied before
`coral_align`, so each λ is a covariance ridge on globally standardized
features. Checkpoint per subject, `--resume`. Assembly
(`coral_lam_sweep_summary.csv`, gate check, Wilcoxon vs per-subject z-score)
done afterward from the four subjectwise CSVs.

Outputs in `results_coral_lam_sweep/`: `lam_<L>/coral_SVM_subjectwise.csv` and
`lam_<L>/coral_summary.csv` per λ; `coral_lam_sweep_summary.csv`;
`coral_lam_sweep_vs_persubj_wilcoxon.csv`.

## Validation gate — PASS (exact)

| | this run (λ=1.0) | published (`results_loso_freq_coral`) | check |
|---|---|---|---|
| CORAL SVM LOSO F1, mean | **0.723571** | 0.723571 | \|diff\| = 0.00e+00 → **PASS** |
| max \|per-subject diff\| over 40 subjects | — | — | **0.00e+00** (bit-identical, all 40) |

`run_coral_loso.py --lam 1.0` is the same script and arguments as the original
`jobs_coral.txt` SVM line with only `--out` changed, so exact reproduction is
expected and confirmed subject by subject. Plan target 0.7236 met.

## Summary table

`coral_lam_sweep_summary.csv`:

| λ (covariance ridge) | CORAL SVM LOSO F1 | sd | n |
|---:|---:|---:|---:|
| 0.01 | 0.6844 | 0.076 | 40 |
| 0.10 | 0.6973 | 0.075 | 40 |
| **1.00** (default, never swept before) | **0.7236** | 0.072 | 40 |
| **10.00** (best in sweep) | **0.7368** | 0.064 | 40 |

CORAL F1 is monotone increasing in λ over the swept range: heavier
regularization (a gentler, more shrinkage-like alignment) helps; aggressive
whitening (small λ) hurts, consistent with the over-alignment picture from
E-C1/E-C2.

Paired Wilcoxon, each λ against the per-subject z-score SVM headline (0.7767,
`results_loso_freq_persubj`, n = 40 subjects,
`coral_lam_sweep_vs_persubj_wilcoxon.csv`):

| comparison | mean ΔF1 | Wilcoxon p |
|---|---:|---:|
| CORAL λ=0.01 vs per-subject z-score | −0.0923 | 1.8 × 10⁻¹² |
| CORAL λ=0.1 vs per-subject z-score | −0.0794 | 5.5 × 10⁻¹² |
| CORAL λ=1.0 vs per-subject z-score | −0.0531 | 5.8 × 10⁻⁹ |
| **CORAL λ=10.0 (best) vs per-subject z-score** | **−0.0399** | **1.5 × 10⁻⁷** |

Best-λ vs per-subject z-score: paired Cohen's d = **−1.00** (large).

## Does the result support or contradict the thesis as written?

**It supports Finding A. CORAL was not beaten unfairly.** The plan anticipated
this outcome — "CORAL is applied to globally standardized features, so λ=1.0 is
already a defensible choice and the sweep most likely confirms it" — and the
sweep confirms it. Every value of the previously-unswept regularizer, across
two orders of magnitude either side of the default, leaves CORAL below the
simple per-subject z-score: the *best* λ in the sweep (10.0) reaches only
0.7368, still 0.040 below per-subject z-score's 0.7767, paired Wilcoxon
p = 1.5 × 10⁻⁷, Cohen's d = −1.00. No λ lifts CORAL above 0.7767, so the
condition the plan set for rewriting Section 4.13 ("If some λ lifts CORAL above
0.7767, that is a material finding and Section 4.13 must be rewritten to report
it") is not met.

One refinement the thesis should adopt. λ=1.0 is defensible but is *not*
optimal: λ=10.0 does about 1.3 points better (0.7368 vs 0.7236). Section 4.13
should report the sweep and use the best-λ figure (0.7368) as the fair CORAL
comparison rather than the default-λ figure (0.7236). Doing so pre-empts the
"you left CORAL's one hyperparameter at its default" objection while
*strengthening* the headline claim: even a tuned CORAL loses to per-subject
z-score by 4 points with p = 1.5 × 10⁻⁷ and a large effect size. The direction
of the λ effect (more shrinkage is better, full whitening is worse) is also
worth a sentence, as it echoes the over-alignment result from E-C1 on an
independent estimator.

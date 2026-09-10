# B3 verdict: 3.2_works

## Per-subject class silhouette by rung (200/subject subsample, seed 42, n = 40)

| rung | operator | per-subject silhouette mean | published pooled |
|---|---|---|---|
| 0 | global_z | +0.0550 | +0.008 |
| 1 | mean_center | +0.0550 | +0.019 |
| 2 | scale_only | +0.0620 | +0.008 |
| 3 | mean_scale | +0.0620 | +0.023 |
| 4 | full_whiten_recolor | +0.0240 | -0.006 |

## Rung-3 peak, paired across 40 subjects

- rung 3 (mean+scale) vs rung 4 (full whiten): delta +0.0380, 95% BCa [+0.0322, +0.0444], raw p = 1.819e-12, d = +1.91 (40/0 +/-)
- rung 3 (mean+scale) vs rung 0 (global z): delta +0.0070, 95% BCa [+0.0046, +0.0101], raw p = 9.597e-06, d = +0.78 (32/8 +/-)

## Within-subject Spearman(silhouette, F1) over the 5 rungs

40/40 subjects with a defined correlation; mean rho +0.437, median +0.462, 35/4 positive/negative; Wilcoxon of the rhos against 0 p = 1.254e-06.

## FDR family

Two new paired Wilcoxon tests: rung3 vs rung4 silhouette raw p = 1.819e-12; rung3 vs rung0 silhouette raw p = 9.597e-06. Section 4.17 not edited. The within-subject correlations are not paired Wilcoxon tests and sit outside the family.

## Table 4.16 note

MMD removed is shown 'n/a' at the baseline rung while the 4.13.2 correlation treats it as 0.0. Reconcile (a small presentational inconsistency, no number changes).

## Section 3.4

Decision point for Enam: whether 4.13.2 gains the per-subject test, replaces the 5-point correlations with it, or reports both. Reported; not resolved here.

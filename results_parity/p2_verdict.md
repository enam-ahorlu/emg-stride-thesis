# Stage P-2 verdict: outcome X

Ranks are not scale-free in practice. The shrink control reproduces 142% of the baseline-to-CD agreement deficit on simulated data with no real change in ordering. Report as a measurement finding; do NOT report a transfer answer.

## Rank agreement (mean pairwise Spearman of channel orderings, resnet family, n = 40, 780 pairs)

- baseline arm: +0.3053
- channel-dropout arm: +0.1416
- deficit (baseline - CD): +0.1637
- paired Wilcoxon on per-subject consistency (CD - baseline): -0.1637, 95% BCa [-0.2164, -0.0962], raw p = 1.859e-06, d = -0.85
- subject-level randomization test: p = 0.0009999

## Validity check 1: null separation

- baseline arm vs its within-subject shuffle null: p = 9.999e-05
- channel-dropout arm vs its within-subject shuffle null: p = 9.999e-05 (agreement +0.1416 vs null 97.5% +0.0292)
- CD arm distinguishable from its own null: YES

## Validity check 2: shrink control

- shrink factor: 0.1785; tuned noise sigma: 2.1314 pp (sim negative fraction 0.254 vs target 0.258)
- rank agreement on simulated shrunk+noise profiles: +0.0724 (sd 0.0280)
- observed deficit +0.1637; shrink-alone deficit +0.2329; fraction reproduced 1.42
- ranks not scale-free in practice: True

## FDR family contribution (plan section 6.3)

One new paired Wilcoxon test (per-subject rank consistency, CD vs baseline): raw p = 1.859e-06. The randomization test and the two shuffle-null checks are not paired Wilcoxon tests and are reported outside the Benjamini-Hochberg family.

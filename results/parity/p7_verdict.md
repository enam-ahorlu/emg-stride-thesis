# Stage P-7 verdict: outcome M

The mean shift carries it. What predominantly breaks channel dropout at high dose is the collapse in expected activation rather than the zeroing. At p = 0.5 half the signal is removed on average, and that carries the majority of the cost. Channel dropout is a variance injector with an unwanted side effect that grows with rate; the transferable recommendation is the mean-preserving form. D2 is itself Holm-significant at +1.08 pp (38% of the gap): the discrete-zeroing form is a real secondary contributor, it simply does not reach the 60% mark. Report both shares. This clears the 60% bar narrowly (D1 = +1.73 pp against a +1.69 pp threshold, margin +0.05 pp); the letter is M by the pre-registered rule but the split is close to B.

## Section 3 gates (all four)

1. Multiplier: derived p' = 0.200000; realized multiplier mean 1.00132, SD 0.49901. PASS.
2. Inertness: all six existing augment modes byte-identical, resnet/resnet_se init unchanged (regression check, no code changes in P-7). PASS.
3. Completeness: 40 rows, 40 unique subjects, dups False; subject id set identical to results_cd_rate_p0.5 and results_p6_gainjitter_resnet_se_sd0.50: True. PASS.
4. Additivity: D1 + D2 = +1.73 + +1.08 = +2.81 pp; observed gap +2.81 pp; residual +0.00 pp against a +/-0.30 pp tolerance. PASS.

## Trainability (section 4)

mean-preserving arm gain vs no aug +5.57 pp; channel dropout p0.5 gain +3.83 pp; difference +1.73 pp (within the 3.0 pp bound).

## Decomposition, both doses (n = 40, share of each dose's own observed gap)

| dose | D1 activation shift (mp minus channel dropout) | D1 Holm p | D2 form (gain jitter minus mp) | D2 Holm p | observed gap |
|---|---|---|---|---|---|
| SD 0.50 (channel dropout p0.5) | +1.73 (+62%) | 8.198e-05 | +1.08 (+38%) | 0.002639 | +2.81 |
| SD 0.40 (channel dropout p0.2, rate sweep) | -0.26 (-26%) | 0.7446 | +1.28 (+126%) | 0.01303 | +1.01 |

D1 and D2 shares sum to the observed gap by construction (additivity gate). At SD 0.40 the P-6 core figures against results_cnn_aug_resnet_se_chandrop (0.8395) were D1 -0.45 pp and D2 +1.28 pp; this table pairs against the era-consistent rate-sweep arm (0.8376) so both dose rows use the same comparator family.

## Arm means (resnet_se, per-subject norm, 250 ms)

- no augmentation 0.7822
- channel dropout p = 0.5 0.8205
- mean-preserving chandrop, p' = 0.2, SD 0.50 0.8379
- gain jitter sd = 0.50 0.8486

## What this cannot say (section 6)

Two doses is not a dose-response curve for either property. Nothing here bears on section 4.8.2's primary finding (the 5.6-fold occlusion-cost reduction) or on whether channel dropout works; it bears only on the top of the rate range. The deep model of record stays the channel-dropout resnet_se at 0.840.

## FDR family (section 8)

Two new paired Wilcoxon tests: D1 raw p = 4.099e-05, D2 raw p = 0.002639. Section 4.17 not edited.

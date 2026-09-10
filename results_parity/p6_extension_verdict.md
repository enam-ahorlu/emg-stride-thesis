# Stage P-6 extension: DIVERGE

DIVERGE. The curves separate materially at SD 0.50 (+2.81 pp). Where they diverge says how the form matters; report the location and direction, do not average it away.

## Curves (resnet_se, per-subject norm, 250 ms), mean LOSO macro-F1

| multiplicative SD | channel dropout (Bernoulli) | gain jitter (uniform) |
|---|---|---|
| 0.300 | 0.8311 | 0.8370 |
| 0.400 | 0.8376 | 0.8477 |
| 0.458 | 0.8373 | n/a |
| 0.500 | 0.8205 | 0.8486 |

no-augmentation baseline 0.7822.

## Matched-variance paired contrasts (gain jitter minus channel dropout, Holm across 3)

- SD 0.30: +0.59 pp, 95% BCa [-0.66, +1.91] pp, raw p = 0.4049, Holm p = 0.4049, d = +0.14
- SD 0.40: +1.01 pp, 95% BCa [+0.15, +1.87] pp, raw p = 0.02885, Holm p = 0.0577, d = +0.36
- SD 0.50: +2.81 pp, 95% BCa [+1.97, +3.82] pp, raw p = 2.275e-07, Holm p = 6.825e-07, d = +0.95

Largest gap 2.81 pp against a 2.0 pp materiality floor.

## FDR family contribution

Three new paired Wilcoxon tests (matched-variance contrasts): SD 0.30 raw p = 0.4049, SD 0.40 raw p = 0.02885, SD 0.50 raw p = 2.275e-07. Section 4.17 not edited.

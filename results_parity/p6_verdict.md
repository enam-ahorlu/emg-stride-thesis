# Stage P-6 verdict: outcome V

Variance is what matters. Neither contrast reaches 2.0 pp with Holm significance: the mean shift does nothing material and the form does nothing material. What the class does is inject per-channel multiplicative variance. This is the cleanest version of the finding and section 4.8.2 should state it as the operative property. Note: C2 (form) is Holm-significant at -1.28 pp (Holm p = 0.0261) but below the 2.0 pp materiality floor, so it is a detectable but immaterial effect and does not change the letter; report the number, not a claim.

New modes added to augment_batch (guarded, defaulted off, inertness PASS on all six existing modes via p5p6_inertness.py): `subset` (P-5) and `mpchandrop` (P-6). mpchandrop multiplies by Bernoulli(1 - p') / (1 - p') with p' = SD^2 / (1 + SD^2) derived from the requested SD; at SD = 0.40, p' = 0.13793. p6_multiplier_gate.py PASS: realized multiplier mean 1.00, SD 0.40.

## Arm means (resnet_se, per-subject norm, 250 ms, n = 40)

| arm | multiplier mean | multiplier SD | mean F1 |
|---|---|---|---|
| no augmentation | 1.0 | 0 | 0.7822 |
| channel dropout p = 0.2 | 0.8 | 0.40 | 0.8395 |
| mean-preserving chandrop | 1.0 | 0.40 | 0.8350 |
| gain jitter sd = 0.40 | 1.0 | 0.40 | 0.8477 |

## C1 channel dropout vs mean-preserving chandrop (isolates the expected-activation shift)

delta +0.45 pp, 95% BCa [-0.24, +1.36] pp, raw p = 0.3011, Holm p = 0.3011, d = +0.18.

## C2 mean-preserving chandrop vs gain jitter (isolates the form)

delta -1.28 pp, 95% BCa [-2.18, -0.36] pp, raw p = 0.01303, Holm p = 0.02605, d = -0.43.

## Each arm vs no augmentation (uncorrected)

- channel dropout: +5.73 pp, raw p = 3.163e-09, d = +1.20
- mean-preserving chandrop: +5.28 pp, raw p = 6.748e-10, d = +1.25
- gain jitter: +6.55 pp, raw p = 3.456e-11, d = +1.45

## Optional extension

Core arms coherent (every arm clearly beats no augmentation, no reversal): True. Extension (gain jitter sd 0.30 and 0.50) is warranted.

## FDR family contribution (plan section 5B.6 / 6.3)

Two new paired Wilcoxon tests in the corrected family: C1 raw p = 0.3011, C2 raw p = 0.01303 (four with the extension). The arm-vs-no-aug contrasts are reported uncorrected beside the family. Section 4.17 not edited.

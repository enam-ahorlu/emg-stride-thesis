# Stage P-9 verdict (4 arms, both backbones)

**Backbone question: S. Transfer question: resnet U, resnet_se U.**

Scope: transfer only. Per-subject coupling retired as underpowered by design (P-8: about 151 subjects at rho = -0.226, 351 at rho = +0.149).

## Section 3.2 gate

- results_p9_atten_resnet_noaug: alpha = 0 vs same-run occlusion max|diff| 0.00e+00 (PASS)
- results_p9_atten_resnet_chandrop: alpha = 0 vs same-run occlusion max|diff| 0.00e+00 (PASS)
- results_p9_atten_resnet_se_noaug: alpha = 0 vs same-run occlusion max|diff| 0.00e+00 (PASS)
- results_p9_atten_resnet_se_chandrop: alpha = 0 vs same-run occlusion max|diff| 0.00e+00 (PASS)

## Section 3.5a reproduction check (A1, A2 only; Section 4.8.2 keeps 84.2 -> 15.0)

- A1 resnet none: P-9 86.88 pp vs original 84.21 pp (+2.67 pp, within the 3.0 pp gate); 40-fold F1 0.7611 vs 0.7684 (-0.73 pp)
- A2 resnet chandrop: P-9 13.40 pp vs original 15.03 pp (-1.63 pp, within the 3.0 pp gate); 40-fold F1 0.8247 vs 0.8251 (-0.04 pp)

## A1/A2 reproduction spread (the yardstick, stated before the cross-backbone comparison)

Summed-cost spread max(|A1 - 84.21|, |A2 - 15.03|) = 2.67 pp. Reduction factor: original 5.60x, P-9 resnet 6.48x [95% 5.11, 8.54] (moved 0.88x on a same-config re-run).

## A3 / A4 first measurement on resnet_se (new numbers, never reproductions)

- A3 resnet_se none: summed occlusion cost 81.77 pp, 40-fold F1 0.7746
- A4 resnet_se chandrop p = 0.2 (model of record): summed occlusion cost 13.36 pp, 40-fold F1 0.8352
- resnet_se reduction (A3/A4): 6.12x [95% 4.94, 7.85]
- per-subject CD-induced absolute reduction: resnet 73.48 pp, resnet_se 68.41 pp; paired backbone difference -5.07 pp, 95% BCa [-13.76, +2.62], p = 0.3472

## Backbone question: S

Split vindicated. The resnet_se occlusion reduction (6.12x, 95% [4.94, 7.85]) matches the SE-free one (6.48x, 95% [5.11, 8.54]) to within the A1/A2 reproduction spread: |difference| 0.36x against a 0.88x shift on a pure re-run, overlapping bootstrap intervals, and a per-subject paired backbone difference of -5.07 pp (95% BCa [-13.76, +2.62], p = 0.347) that is not distinguishable from zero. The mechanism findings transfer across backbones, measured rather than assumed. Section 4.8.2 may keep quoting either figure provided it names the backbone, and the G1 warrant is retrospectively supported on a mechanism quantity.

## Transfer question, per backbone

### resnet - transfer letter U

censoring on the augmented arm is 36.9% even at the 1.0 pp criterion, above the 30% limit; a third measure has failed. Do not commission a fourth.

- criterion 1.0 pp; censoring no-aug 8.9%, chandrop 36.9%
- threshold-ordering agreement: no-aug +0.1497 (95% boot [+0.1093, +0.2399], null p 9.999e-05); chandrop +0.0377 (null p 0.009799)
- paired consistency (chandrop - no-aug): -0.1120, 95% BCa [-0.1474, -0.0717], raw p = 8.848e-06, d = -0.89; subject-swap p = 0.007399
- shrink control simulated agreement +0.0024 (sd 0.0123)
- model vs data (P-8 Test A +0.167): tracks_data

### resnet_se - transfer letter U

censoring on the augmented arm is 37.5% even at the 1.0 pp criterion, above the 30% limit; a third measure has failed. Do not commission a fourth.

- criterion 1.0 pp; censoring no-aug 7.2%, chandrop 37.5%
- threshold-ordering agreement: no-aug +0.0990 (95% boot [+0.0614, +0.1935], null p 9.999e-05); chandrop +0.0629 (null p 0.0006999)
- paired consistency (chandrop - no-aug): -0.0361, 95% BCa [-0.0746, +0.0015], raw p = 0.07933, d = -0.29; subject-swap p = 0.2289
- shrink control simulated agreement +0.0010 (sd 0.0131)
- model vs data (P-8 Test A +0.167): tracks_data

## FDR family

P-9's transfer statistics are permutation-null correlations and shuffle-null paired comparisons, not paired Wilcoxon tests, so they sit outside the Benjamini-Hochberg family (as P-1 and P-2). The per-subject consistency Wilcoxon and the paired backbone-reduction Wilcoxon are reported beside the family: resnet consistency raw p = 8.848e-06, resnet_se consistency raw p = 0.07933, backbone-reduction raw p = 0.3472. Section 4.17 not edited.

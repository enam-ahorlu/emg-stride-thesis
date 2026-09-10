# Stage P-1 verdict: outcome S

Separated at the boundary. Occlusion cost keeps falling with rate while F1 plateaus / falls: more invariance keeps buying robustness and stops buying accuracy. Write as the over-alignment analogue of section 4.13.2, structural not mechanistic.

## P-1a dose curve (resnet_se family)

| rate | mean occlusion cost (pp) | mean LOSO macro-F1 (%) |
|---|---|---|
| 0.1 | 21.373 | 83.109 |
| 0.2 | 12.095 | 83.756 |
| 0.3 | 10.262 | 83.729 |
| 0.5 | 7.420 | 82.051 |

Spearman(rate, cost) rho = -1.000, raw p = 0, Holm p = 0. Page trend (decreasing cost vs rate) L = 1113.0, raw p = 3.022e-10, Holm p = 9.067e-10. Occlusion cost monotone decreasing across all three rate steps: True. F1 change 0.1 to 0.3: +0.62 pp; 0.3 to 0.5: -1.68 pp.

No instrumented no-augmentation resnet_se run exists, so this curve has no p = 0 origin. The SE-free results_g3_noaug_instr is NOT substituted as the origin (plan section 2.2).

## P-1b per-subject coupling (resnet family, n = 40)

Spearman(delta occlusion cost, delta F1) = 0.305, permutation p = 0.05579 (10,000 draws, seed 42), Holm p = 0.1116. Account predicts rho < 0; observed sign inconsistent.

## P-1c baseline reliance predicts benefit (resnet family, n = 40)

Spearman(baseline occlusion cost, channel-dropout gain) = -0.282, permutation p = 0.08009, Holm p = 0.1116. Account predicts rho > 0; observed sign inconsistent. Companion Spearman(baseline cost, delta cost) = -0.889, permutation p = 9.999e-05.

## FDR family contribution (plan section 6.3)

Correlations against a permutation null are not paired Wilcoxon tests and are reported outside the Benjamini-Hochberg family. The three adjacent-rate per-subject paired Wilcoxon contrasts on total occlusion cost (P-1a) are new paired tests:

- cost p0.1->p0.2: raw p = 4.73e-05
- cost p0.2->p0.3: raw p = 0.1029
- cost p0.3->p0.5: raw p = 0.06415


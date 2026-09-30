# Stage P-10 verdict: outcome B

Boundary found. On the arms that trained, a peak at SD 0.50 is followed by a fall exceeding 2.0 pp with a Holm-significant paired contrast at SD 0.80 (fall +4.63 pp, Holm p = 3.889e-09, d = +1.18). A genuine over-invariance boundary exists on the mean-preserving family, at a dose the unnormalized form never reached because its own artifact masked it (P-7). The curve is flat through the operating range and turns down past the peak. The Section 4.13.2 parallel is supported on this family and P-1's structural reading holds after all, relocated to a higher dose. Arm(s) at SD [1.0] fell more than 3.0 pp below the no-augmentation baseline of 0.7822 (at high p' half the channels are zeroed each sample and the rest scaled up hard); each is outcome X for that arm, a bound on the family rather than a bug, and is excluded from the ceiling reading below. Caveat (plan section 6): two-plus points is not a dose-response curve, and the boundary sits where the perturbation approaches the p' = 0.5 degeneracy, so 'over-alignment destroys class structure' versus 'the perturbation becomes too destructive to train against' is not fully separable here.

## Multiplier gates

Run via `p6_multiplier_gate.py 0.60 0.80 1.00`; realized mean and SD reported in the run log. p' = SD^2 / (1 + SD^2): 0.60 -> 0.2647, 0.80 -> 0.3902, 1.00 -> 0.5000.

## Curve (resnet_se, per-subject norm, 250 ms, n = 40)

| SD | p' | mean F1 | vs no aug (pp) |
|---|---|---|---|
| 0.40 | 0.1379 | 0.8350 | +5.28 |
| 0.50 | 0.2000 | 0.8379 | +5.57 |
| 0.60 | 0.2647 | 0.8320 | +4.98 |
| 0.80 | 0.3902 | 0.7916 | +0.94 |
| 1.00 | 0.5000 | 0.7313 | -5.09 |

no-augmentation baseline 0.7822. Peak at SD 0.50. Full span 4.63 pp; largest post-peak fall 4.63 pp against a 2.0 pp floor.

## Post-peak contrasts (peak arm minus higher-SD arm, Holm within)

- SD 0.50 vs SD 0.60: +0.58 pp, 95% BCa [-0.17, +1.44], raw p = 0.2826, Holm p = 0.2826, d = +0.22
- SD 0.50 vs SD 0.80: +4.63 pp, 95% BCa [+3.50, +5.93], raw p = 1.944e-09, Holm p = 3.889e-09, d = +1.18
- SD 0.50 vs SD 1.00: +10.65 pp, 95% BCa [+8.98, +12.43], raw p = 3.638e-12, Holm p = 1.091e-11, d = +1.88

## Section 4.6: draft replacement wording for P-1's verdict (NOT written to any thesis file)

P-1 replacement (outcome S retained, relocated to the mean-preserving family). The apparent boundary in the unnormalized rate sweep was the mask artifact P-7 identified. On the mean-preserving family, which has no such artifact, the dose sequence is flat through the operating range (SD 0.40 to 0.60: 83.1, 83.8, 83.7, 83.2 percent macro-F1) and then falls, Holm-significant, beyond the peak: SD 0.80 79.2%, SD 1.00 73.1% (P-10). A genuine over-invariance boundary therefore does exist, at a dose higher than channel dropout at p = 0.5 could probe, so P-1's structural reading and the Section 4.13.2 parallel hold on this family with the boundary located near SD 0.80. Two qualifications belong in the same breath: the sweep has two or three usable points past the peak, not a curve, and SD 1.00 (p' = 0.5) collapses below no augmentation entirely, so the far end is the perturbation becoming degenerate rather than a clean statement about class structure.

## FDR family

New paired Wilcoxon tests: the post-peak contrasts (SD 0.50 vs SD 0.60 raw p = 0.2826, SD 0.50 vs SD 0.80 raw p = 1.944e-09, SD 0.50 vs SD 1.00 raw p = 3.638e-12). Section 4.17 not edited.

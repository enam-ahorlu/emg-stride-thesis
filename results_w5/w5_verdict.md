# W-5 depth vs capacity: verdict

## Outcome N

Neither. The +4.24 pp step is not attributable to depth or capacity within this range; kernel width and stem remain untested. Informative negative.

## Per-arm channel-dropout gain (chandrop - noaug, paired, n=40)
- BASE gain (chandrop - noaug): +4.55 pp (p = 5.064e-08, d = 1.04, 95% BCa [+3.29, +5.95] pp)
- SHALLOW-MATCHED gain: +4.93 pp (p = 4.766e-07, d = 0.98, 95% BCa [+3.33, +6.41] pp)
- NARROW-MATCHED gain: +4.33 pp (p = 2.692e-08, d = 1.02, 95% BCa [+3.16, +5.77] pp)

## Interaction contrasts (Holm across the family of two)
- interaction: BASE - SHALLOW: -0.38 pp (raw p = 0.6179, Holm p = 1, d = -0.07, 95% BCa [-2.16, +1.41] pp) -> does not lose the effect
- interaction: BASE - NARROW: +0.22 pp (raw p = 0.7346, Holm p = 1, d = 0.05, 95% BCa [-1.08, +1.48] pp) -> does not lose the effect

## No-augmentation baselines (trainability, §5 X-check)
- BASE 0.7718 (ref 0.7718)
- SHALLOW-MATCHED 0.7521  (-1.97 pp vs BASE)
- NARROW-MATCHED 0.7714  (-0.04 pp vs BASE)
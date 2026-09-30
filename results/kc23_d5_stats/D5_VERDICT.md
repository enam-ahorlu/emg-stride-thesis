# KC-D5 verdict

- **chandrop_gain: E-R** (mean diff +5.364, 10 of 10 subjects agree with the expected sign +)
- **gainjitter_vs_chandrop: E-N** (mean diff +0.547, 4 of 10 subjects agree with the expected sign +)
- **occlusion_reduction: E-R** (mean diff +2.667, 9 of 10 subjects agree with the expected sign +)
- **permutation_reduction: E-N** (mean diff -0.225, 4 of 10 subjects agree with the expected sign +)

Occlusion magnitude beside its letter: reduction factor 1.68x (SD 0.12 across the 5 realizations) on ENABL3S, against about 6x on SIAT-LLMD. The thesis may say the direction replicates; it must not say 'six-fold' for ENABL3S.

ResNet-SE+CD 0.6487 +/- 0.0082 (SD across 5 realizations) against the ENABL3S SVM 0.6570: -0.83 pp. Reported as a number, not a letter.

Directions are those the thesis states on SIAT-LLMD (decision D-6c), encoded in the script: gain jitter ahead of channel dropout (positive) and a permutation-reliance reduction under channel dropout (positive). Both take the reading less favourable to the thesis.

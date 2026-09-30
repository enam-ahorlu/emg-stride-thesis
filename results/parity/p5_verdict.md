# Stage P-5 verdict: outcome M

Matches. The primary contrast is within 2.0 pp and not Holm-significant, and the secondary is clearly positive. A third channel-structured perturbation, this one from the literature, does the same work: W-4's generalization stops resting only on a control the thesis invented. Section 4.8.2's claim strengthens materially.

Arm: subset mode on resnet_se, per-subject normalization, 250 ms, seed 42, every other flag default, into results_p5_subset_resnet_se. Subset vocabulary: all C(9,2) = 36 channel-omit pairs, each subset retaining 7 of 9 channels; one drawn uniformly per training sample.

Arm means (n = 40): subset 0.8360, channel dropout 0.8395, no augmentation 0.7822.

## Primary: subset vs channel dropout

delta -0.35 pp, 95% BCa [-1.13, +0.55] pp, raw p = 0.3403, Holm p = 0.3403, d = -0.13.

## Secondary: subset vs no augmentation

delta +5.38 pp, 95% BCa [+4.07, +6.98] pp, raw p = 1e-10, Holm p = 2.001e-10, d = +1.15.

Reference (uncorrected): channel dropout vs no augmentation delta +5.73 pp, raw p = 3.163e-09, d = +1.20.

Trainability: subset arm is +5.38 pp against the no-augmentation ResNet-SE baseline (3.0 pp floor for outcome X).

## FDR family contribution (plan section 5A.5 / 6.3)

Two new paired Wilcoxon tests: primary raw p = 0.3403, secondary raw p = 1e-10. Section 4.17 not edited.

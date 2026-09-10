# Stage P-8 verdict: outcome I

Idiosyncratic but inconsequential. The channel ordering is weakly shared: Test A agreement is +0.167, reliably above its shuffle null (p = 9.999e-05) but far below the +1.000 ceiling, so orderings differ substantially between people. That difference does not significantly predict who is hard (B1 Holm p = 0.355) or who benefits (B2 Holm p = 0.329). Every directional check is consistent with the second-locus account (B1 positive, B2 negative, C positive, all as predicted) but none survives correction at n = 40, so this is a consistent-but-underpowered negative, not a refutation. Report both, claim neither; Section 4.8.2 keeps its current scope.

## Freq-72 column-to-channel mapping

Feature-major layout: Freq-72 = MAV(9) RMS(9) WL(9) ZC(9) WAMP(9) MNF(9) MDF(9) SpectralPower(9). Column j maps to feature family j // 9 and channel j % 9. Channel c's eight features are columns {c, c+9, c+18, c+27, c+36, c+45, c+54, c+63}. Channel index 0..8 = TFL, RF, VM, SMB, Upper TA, Lower TA, Lateral GC, Medial GC, SOL (Table 3.1 order).

## The 40 by 9 matrix

Per-subject z-scored Freq-72 (per column, within subject, exactly as train_classical_loso.per_subject_zscore), then per subject and channel a 4-class LDA on that channel's 8 features, stratified 5-fold CV within subject, macro-F1. Overall mean 0.5150; 0.0% of cells within 0.05 of chance (0.25). The before/after-normalization contrast was not run: LDA is invariant to invertible linear maps and per-column standardisation is diagonal, so the two matrices are identical by construction.

## Consensus channel ranking (mean within-subject LDA macro-F1, high to low)

1. Lower TA (channel 5), mean LDA macro-F1 0.5698
2. Upper TA (channel 4), mean LDA macro-F1 0.5477
3. Lateral GC (channel 6), mean LDA macro-F1 0.5253
4. SMB (channel 3), mean LDA macro-F1 0.5219
5. SOL (channel 8), mean LDA macro-F1 0.5191
6. TFL (channel 0), mean LDA macro-F1 0.5038
7. Medial GC (channel 7), mean LDA macro-F1 0.5013
8. VM (channel 2), mean LDA macro-F1 0.4766
9. RF (channel 1), mean LDA macro-F1 0.4694

## Test A: ordering agreement

Mean pairwise Spearman across 780 subject pairs = +0.1668. Within-subject channel-label shuffle null (10,000 draws, seed 42): mean +0.0003, 97.5 percentile +0.0299, permutation p = 9.999e-05. Ceiling is +1.000.

## Test B (primary): leave-one-out consensus agreement vs difficulty and benefit

Consensus agreement score (each subject's Spearman against the mean ranking of the other 39): mean +0.4079, sd 0.3733.

- B1 vs un-augmented ResNet-SE LOSO F1: Spearman +0.149, permutation p = 0.355, Holm p = 0.355. Account predicts positive; sign consistent.
- B2 vs channel-dropout gain: Spearman -0.226, permutation p = 0.1643, Holm p = 0.3286. Account predicts negative; sign consistent.

Unlike P-1b and P-1c, this is measured on the features, where nothing saturates, not on the augmented model's occlusion profile, which does.

## Test C: model reliance vs data informativeness

Spearman(consensus informativeness, un-augmented model mean occlusion drop_pp per channel) = +0.250, permutation p = 0.5118 (n = 9 channels; reported beside the family). Account predicts positive.

## FDR family

P-8's tests are Spearman correlations against permutation nulls, not paired Wilcoxon tests, so they are reported outside the Benjamini-Hochberg family, as P-1 and P-2 were. Section 4.17 not edited. New paired Wilcoxon tests contributed by P-8: none.

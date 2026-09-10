# Stage P-3 verdict: outcome R

Replicates. Primary contrast positive, Holm p < 0.05, d >= 0.5. The augmentation gain is not SIAT-specific; goes into section 4.12 beside the normalization replication.

Subjects paired: 10 (ids [156, 185, 186, 188, 189, 190, 191, 192, 193, 194]); subject id sets identical across all four arms.

## Primary: channel dropout vs no augmentation, per-subject normalization

- no aug 0.5650 to channel dropout 0.6367
- paired delta +7.17 pp, 95% BCa [+5.04, +10.11] pp
- Wilcoxon raw p = 0.001953, Holm p = 0.003906, Cohen's d = +1.72 (10/0 +/-)

## Secondary: channel dropout vs no augmentation, global normalization

- no aug 0.4756 to channel dropout 0.5044
- paired delta +2.88 pp, 95% BCa [+0.58, +6.57] pp
- Wilcoxon raw p = 0.2754, Holm p = 0.2754, Cohen's d = +0.58 (6/4 +/-)

At n = 10 the effect size is the primary evidence and the p-value secondary (the section 4.12 convention for the RF replication).

## FDR family contribution (plan section 6.3)

Two new paired Wilcoxon tests: primary raw p = 0.001953, secondary raw p = 0.2754.

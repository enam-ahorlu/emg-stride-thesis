# B8 verdict: subject-dependent protocol, movement-blocked re-run

**Grid outcome(s): RANKING FLIPS, CONTAINED (marginal).**

## What changed

The SD protocol split 50%-overlapping windows at random (pooled StratifiedKFold over all windows). The corrected design (section 4A.1) blocks within each movement's own clock into 5 contiguous chunks, assigns chunk i of every movement to fold i, and drops windows within one window length of every chunk boundary (guard band: 8.2% of windows at 250 ms, 5.6% at 150 ms - higher than the plan's ~1% estimate, which is what the design costs at 50% overlap). No outcome-X flags on any config: every fold keeps test windows and all four classes in training.

## Result: the overlap leak was worth 3.7 to 6.1 pp of SD macro-F1

| config | model | SD old (pooled) | SD new (blocked) | delta |
|---|---|---|---|---|
| freq72_w250 | SVM | 93.67 | 88.83 | -4.83 |
| freq72_w250 | RF | 91.98 | 87.99 | -3.98 |
| freq72_w250 | LDA | 89.85 | 84.57 | -5.27 |
| base_w250 | SVM | 92.26 | 88.01 | -4.25 |
| base_w250 | RF | 91.04 | 87.39 | -3.65 |
| ext_w250 | SVM | 91.61 | 85.93 | -5.67 |
| ext_w250 | RF | 91.59 | 87.95 | -3.65 |
| freq72_w150 | SVM | 92.44 | 86.54 | -5.90 |
| freq72_w150 | RF | 91.44 | 87.20 | -4.24 |
| freq72_w150 | LDA | 86.27 | 81.40 | -4.86 |
| base_w150 | SVM | 90.62 | 85.93 | -4.69 |
| base_w150 | RF | 90.88 | 87.05 | -3.83 |
| ext_w150 | SVM | 90.11 | 84.02 | -6.09 |
| ext_w150 | RF | 91.30 | 87.35 | -3.95 |
| cnn_w250 | CNN | 89.87 | 85.91 | -3.96 |

Every config and model, p < 1e-9, paired across 40 subjects.

## Ranking flips under a common protocol

Freq-72 / SimpleEMGCNN, 250 ms, all on the movement-blocked protocol: SVM 88.8, RF 88.0, **CNN 85.9**, LDA 84.6. The CNN no longer leads; SVM and RF are both above it. Table 4.1 currently has CNN first (90.4) beside pooled classical numbers, which is the unfair comparison this stage corrects. Section 4.1's 'the CNN leading narrowly under SD' does not hold on a common protocol.

## Gaps compress to the lower edge of 10 to 25 pp

- SVM: new SD 88.8 minus LOSO per-subject 77.7 = +11.1 pp
- RF: new SD 88.0 minus LOSO per-subject 77.3 = +10.7 pp
- CNN: new SD 85.9 minus LOSO per-subject 75.4 = +10.5 pp

All three land near 10 to 11 pp (was 15 to 22 pp pooled). Inside the section 6.1 range but at its floor; 'squarely within' should become 'at the lower edge of'.

## Reproduction caveat

My pooled-random classical SD reproduces higher than Table 4.1 (pooled SVM 93.7 vs Table 4.1's 87.4), most likely because Table 4.1 used per-fold nested GridSearchCV and a possibly different weighting/feature config. The **pooled-vs-blocked delta** and the **common-protocol ordering** are controlled comparisons and are the robust claims; the absolute classical SD levels carry this caveat.

## Not affected (section 4A.4)

LOSO is untouched (its outer split is by subject; no overlapping window crosses it). Finding A, the alignment ladder, the 85.8% headline, the 81.7% causal figure and the external replication all stand. The gap-reduction argument is untouched: the SD number is held constant across the baseline and optimized rows, so the change in the gap equals the LOSO improvement exactly (16.6 - 9.7 = 6.9 = 77.7 - 70.8).

## FDR family

Each config contributes a paired Wilcoxon (pooled vs blocked SD F1); all raw p < 1e-9. Reported for the recompute; Section 4.17 not touched.

# KC-S1 verdict

**Outcome: D-S**

**Reproduction gate: PASS**

ESCALATE. 'Without labeled calibration' holds offline only.

## Primary endpoint (K=25, realization-averaged over seeds [42, 7, 123])

Best supervised arm S-ens1 85.32% against L0 81.60%: +3.72 pt, Wilcoxon p=3.95e-08, dz=1.06, BCa 95% [+2.74, +4.88] pt, 36 of 40 subjects improved.

## Secondary 1: K curve (supervised ensemble against L0)

| K | arm | mean_arm | mean_l0 | delta_pp | p_raw | n_improved | reaches_l0 | significantly_ahead |
|---|---|---|---|---|---|---|---|---|
| 5 | S-ens1 | 78.87 | 78.19 | +0.68 | 0.115 | 25 | True | False |
| 5 | S-ens2 | 78.69 | 78.19 | +0.51 | 0.221 | 24 | True | False |
| 10 | S-ens1 | 82.27 | 80.61 | +1.67 | 0.00558 | 29 | True | True |
| 10 | S-ens2 | 81.85 | 80.61 | +1.25 | 0.0259 | 29 | True | True |
| 25 | S-ens1 | 85.32 | 81.60 | +3.72 | 3.95e-08 | 36 | True | True |
| 25 | S-ens2 | 84.23 | 81.60 | +2.63 | 2.04e-06 | 33 | True | True |

Smallest K at which each arm reaches L0 (mean difference >= 0) and at which it is significantly ahead (p < 0.05): S-ens1: reaches at K=5, significantly ahead at K=10; S-ens2: reaches at K=5, significantly ahead at K=10. The K grid is [5, 10, 25], so K=5 is the lowest value tested: a result at K=5 means 'at or below 5', not exactly 5.

## Secondary 2: S-ft against L1 (pure fine-tune gain, causal normalization)

| K | mean_s_ft | mean_l1 | delta_pp | p_raw | cohens_dz | n_improved | per_seed_delta_pp | n_seeds_positive |
|---|---|---|---|---|---|---|---|---|
| 5 | 74.32 | 75.24 | -0.92 | 0.444 | -0.14 | 18 | -0.79;-1.37;-0.60 | 0 |
| 10 | 79.75 | 78.55 | +1.19 | 0.221 | 0.20 | 23 | +1.26;+1.47;+0.86 | 3 |
| 25 | 82.92 | 79.66 | +3.26 | 0.000941 | 0.45 | 28 | +3.32;+3.04;+3.43 | 3 |

L1 is the RESNET_SE arm, the same trained base network S-ft starts from, so each seed's pairing sits inside one training realization.

## Secondary 3: S-pool and S-only are seed-invariant classical fits

| arm | K | max_abs_diff_across_seeds | seed_invariant | is_control |
|---|---|---|---|---|
| S-pool | 5 | 0 | True | False |
| S-pool | 10 | 0 | True | False |
| S-pool | 25 | 0 | True | False |
| S-only | 5 | 0 | True | False |
| S-only | 10 | 0 | True | False |
| S-only | 25 | 0 | True | False |
| RESNET_SE | 5 | 0.173 | False | True |
| RESNET_SE | 10 | 0.138 | False | True |
| RESNET_SE | 25 | 0.147 | False | True |
| S-ft | 5 | 0.138 | False | True |
| S-ft | 10 | 0.152 | False | True |
| S-ft | 25 | 0.0837 | False | True |
| L0 | 5 | 0.0877 | False | True |
| L0 | 10 | 0.0693 | False | True |
| L0 | 25 | 0.073 | False | True |

S-pool and S-only are fixed SVC fits (`random_state=42`, not the run seed) on buffers that do not depend on the seed, so their per-subject F1 is identical in every seed by construction; the rows above confirm it in the data. The controls (RESNET_SE, S-ft, L0, which depend on the trained network) are not seed-invariant, so the check can tell the difference. Their across-seed SD is therefore zero by construction and is not evidence of stability across training runs.

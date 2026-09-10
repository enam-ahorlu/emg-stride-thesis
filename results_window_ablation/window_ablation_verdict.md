# Window-length ablation: verdict

## Outcome B

Material but tolerable. The 250 ms window stands, reported with the trade-off stated honestly. Flag to Enam in the summary, but the plan does not stop here.

## Primary endpoint: does the normalization finding survive?

- SVM @ 150 ms: per-subject vs global: +6.58 pp (Holm p = 1.091e-10, d = 1.53, 95% CI [+5.33, +7.94] pp)
- SVM @ 250 ms: per-subject vs global: +6.87 pp (Holm p = 1.537e-09, d = 1.48, 95% CI [+5.49, +8.34] pp)
- SVM @ 400 ms: per-subject vs global: +6.72 pp (Holm p = 3.131e-08, d = 1.13, 95% CI [+5.00, +8.64] pp)
- ResNet-SE+CD @ 150 ms: per-subject vs global: +6.38 pp (Holm p = 3.131e-08, d = 1.04, 95% CI [+4.70, +8.48] pp)
- ResNet-SE+CD @ 250 ms: per-subject vs global: +6.72 pp (Holm p = 3.252e-09, d = 1.05, 95% CI [+4.98, +8.99] pp)
- ResNet-SE+CD @ 400 ms: per-subject vs global: +5.58 pp (Holm p = 2.037e-06, d = 0.84, 95% CI [+3.65, +7.82] pp)

The gap is positive and significant at every window and for both models. The thesis's central claim is robust to window length.

## Secondary endpoint: window comparisons under per-subject normalization

- SVM: 150 ms vs 250 ms (per-subject): -2.92 pp (Holm p = 1.091e-11, d = -1.82, 95% CI [-3.45, -2.47] pp)
- SVM: 400 ms vs 250 ms (per-subject): +2.47 pp (Holm p = 3.087e-06, d = 0.89, 95% CI [+1.66, +3.35] pp)
- SVM: 150 ms vs 400 ms (per-subject): -5.39 pp (Holm p = 1.382e-10, d = -1.58, 95% CI [-6.42, -4.36] pp)
- ResNet-SE+CD: 150 ms vs 250 ms (per-subject): -5.61 pp (Holm p = 1.819e-11, d = -2.02, 95% CI [-6.54, -4.83] pp)
- ResNet-SE+CD: 400 ms vs 250 ms (per-subject): +2.04 pp (Holm p = 0.0004266, d = 0.57, 95% CI [+0.82, +3.03] pp)
- ResNet-SE+CD: 150 ms vs 400 ms (per-subject): -7.65 pp (Holm p = 1.801e-10, d = -1.97, 95% CI [-8.72, -6.36] pp)

## Decision rule applied

Per EXPERIMENT_PLAN_WINDOW_LENGTH.md section 1.3, fixed before the runs:

- SVM: 400 ms vs 250 ms (per-subject) -> outcome B
- ResNet-SE+CD: 400 ms vs 250 ms (per-subject) -> outcome B
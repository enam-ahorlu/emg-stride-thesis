# D2 lambda sweep — pre-registered prediction

Written: 2026-09-15T12:54:27Z, before the first run starts (reproduction gate, Phase D2.0).

## Why D2 runs at all

D1 resolved to Outcome 2: per-subject normalization beats Deep CORAL at lambda = 1 by 1.378 pp
(BCa [0.15, 2.40]), which does not clear the 0.5 pp §3.16 nondeterminism band — a near-tie, not a
win. The original plan's Gate D1 called D2 optional on that basis. The 15-Sep amendment reverses
that call: a near-tie is a *worse* defence against an untuned competitor than a clear win would have
been, since §4.7 swept the classical CORAL regularizer across four orders of magnitude and built an
argument from the shape of that sweep, while the deep side got one setting. Enam's decision,
15 September: run it.

## The prediction (interior maximum, not monotone)

`coral_lambda` scales the alignment term: larger lambda = more alignment. At lambda = 0 the method
degenerates to the global-normalization baseline (~0.772). At lambda = 1 it scores 0.825712. So
macro-F1 **rises** from lambda 0 to lambda 1. The over-alignment account of §4.7.2 (already
demonstrated on the classical CORAL sweep and on the alignment ladder) predicts it must **fall**
again once alignment is pushed far enough, because alignment eventually removes class-discriminative
covariance along with subject variance.

**The prediction is an interior maximum, not a monotone curve.** The open question is where the peak
sits relative to lambda = 1:
- If the peak is at or below lambda = 1, the thesis's fixed-lambda comparison was fair by luck, and
  the over-alignment story gains a third independent instance (after the classical sweep and the
  alignment ladder).
- If the peak sits above lambda = 1, Deep CORAL has room the thesis never gave it, and the size of
  that room is what this experiment measures.

## The three outcome rules, fixed before running (threshold = 0.5 pp nondeterminism band, not the
## original plan's stale 1.5 pp — the amendment replaces that threshold because the margin is now a
## near-tie, not a believed 1.4 pp win)

**Outcome A.** Best lambda stays below per-subject normalization (0.839490) by more than 0.5 pp, with
a paired test surviving Holm. Expected case. Claim strengthens to "beats Deep CORAL at its best
setting among those tested" — strictly stronger than the thesis currently says. No claim inverts.

**Outcome B.** Best lambda closes the gap to inside the 0.5 pp band. Per-subject normalization and a
tuned Deep CORAL become indistinguishable on this backbone. The abstract's "edges its deep variant by
about a point" and the corresponding §4.7/§5.8/§6.2 sentences would need to come out. **Report and
stop. Do not edit any chapter.**

**Outcome C.** Best lambda overtakes per-subject normalization by more than 0.5 pp with a paired test
surviving Holm. The CNN-side comparison inverts. **Stop immediately, report the lambda/margin/
interval/test, do not rewrite, do not soften, do not run further arms.** Enam's decision, not the
runner's.

**In all three cases:** the normalization finding itself does not depend on this backbone alone — it
rests on consistency across three model families, the classical CORAL sweep, the alignment ladder,
ENABL3S, and the active-only control. Outcome C would narrow one sentence about one backbone, not
touch the spine of the thesis.

## Comparator, verified before this file was written

`results_cnn_aug_resnet_se_chandrop/cnn_arch_subjectwise.csv`, mean f1_macro = **0.839490** to six
decimals, 40 unique subjects, no duplicates. Confirmed NOT sourced from
`results_cnn_aug_resnet_se_chandrop_proba` (mean 0.841566, a different repeat run of the same arm).

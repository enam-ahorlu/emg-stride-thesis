# W-2 Stage G1 verdict

## Outcome R

Residual-dependent. SE is NOT the mechanism; §4.8's claim is wrong and must be corrected. Gate CLOSED. Stop and report.

ResNet+CD gain (fresh baseline): +6.51 pp (Holm p = 6.881e-09, d = 1.35, 95% CI [+5.00, +7.98] pp)
Interaction contrast (fresh):    -0.78 pp (Holm p = 0.2214, d = -0.18, 95% CI [-2.05, +0.55] pp), not significant

## Arm means (LOSO macro-F1, n=40)
- ResNet-SE+CD : 0.8395  (old, reused)
- ResNet-SE    : 0.7822  (old, reused)
- ResNet+CD    : 0.8251  (fresh, G1)
- ResNet       : 0.7600  (fresh baseline)  |  old 0.7563

## Parameter counts: resnet 546,020  |  resnet_se 557,276  (SE adds 11,256; comparison is not capacity-matched)

## Sensitivity: same contrasts on the OLD baseline
- interaction contrast [base=old]: -1.15 pp (Holm p = 0.1387, d = -0.20)
- ResNet+CD gain (CD - base) [base=old]: +6.88 pp (Holm p = 3.001e-10, d = 1.51)
- ResNet-SE+CD gain (SE+CD - SE) [base=old]: +5.73 pp (Holm p = 6.326e-09, d = 1.20)
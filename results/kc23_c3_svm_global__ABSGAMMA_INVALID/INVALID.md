# INVALID: SVM-X grid did not match the plan

Set aside on 26 September 2026 (pre-registration conformance pass, KC23_PREREG_CONFORMANCE.md). This run used the
absolute-gamma grid {0.01, 0.1, 0.3, 1, 3, 10, 'scale'} with C in {0.01 ... 30} (56 cells). EXPERIMENT_PLAN_KC23_CLASSICAL.md
C3.3 specifies gamma in {0.01, 0.1, 0.3, 1, 3, 10} x the fitted `scale` value (48 cells). For 72 standardized features
`scale` is about 0.0139, so the plan's multiples span 1.4e-4 to 0.14 while the absolute values 0.01 to 10 are one to
three orders of magnitude higher: a different search.

Kept, not deleted. It is not evidence for or against any C3 endpoint. The corrected runs write to the original
directory names (results_kc23_c3_svm_per_subject and results_kc23_c3_svm_global) with train_classical_loso.py's
multiplicative grid, and record the chosen multiplier in svm_extended_gamma.csv.

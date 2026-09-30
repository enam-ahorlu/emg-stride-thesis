# INVALID: 400 ms features were extracted with a different configuration from the published ENABL3S features

Set aside on 26 September 2026 (pre-registration conformance pass, KC23_PREREG_CONFORMANCE.md). The builder row ran
`extract_features.py --use env --freq` with no `--fs` and no `--no-wavelet`. The published ENABL3S 250 ms Freq features
(REPRODUCE.md, docs/EXPERIMENTS_README.md) were built with `--use raw --freq --fs 1000 --no-wavelet`. This directory's cfg
records use_wavelet true and sampling_rate 2000.0 (ENABL3S is recorded at 1000 Hz), and it was computed on the envelope, not
the raw window; it has 63 features per window against the published 56. It is not the same feature set at a different
window length, so it cannot serve the S2b window trade ("the locked ... SVM at 400 ms").

Kept, not deleted. The corrected row (s2b_extract_features) writes to the original directory name with the published flags.
The sibling results_kc23_s2b_svm_400__WRONGCFG_INVALID was trained on these features and is set aside for the same reason.

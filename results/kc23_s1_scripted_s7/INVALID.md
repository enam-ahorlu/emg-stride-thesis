# INVALID

This run's L0 SVM arm normalised the held-out subject from the first 25
chronological windows (run_streaming_norm_loso.normalise_test_subject's
"calib" buffer), not from 25 windows of each movement (the balanced25
buffer S1.2 specifies). Its l0_svm_f1 (0.6914279471200013) exactly matches
the published calib25 SVM figure (results_causal_ensemble/report.csv:
config=calib25, model=SVM, f1_excl_mean=0.6914), confirming the wrong
buffer, not noise -- the reproduction gate correctly FAILed against the
real balanced25 target (0.7472).

Reason: wrong buffer (chronological calib25, not per-movement balanced25).
Superseded by a re-run after run_scripted_supervised.py's L0 arm was fixed
to use balanced_buffer_indices(). Left in place, not deleted.

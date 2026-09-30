# KC-S2 verdict

Descriptive only (no halting letters, per S2.5). The readings lead with the transductive and balanced25 conditions; the causal-100 collapse is reported separately below. Measures per condition and model, windows scored against the ground-truth transition table (rows in that order).

| condition | model | steady_error | zone_error | dns_as_wak_rate_zone | dns_as_wak_rate_steady | delay_all_median_s | vote_delta_steady_error | vote_delta_zone_error | vote_added_delay_paired_median_s |
|---|---|---|---|---|---|---|---|---|---|
| transductive | SVM | 0.1665 | 0.3455 | 0.5920 | 0.2500 | 0.3250 | -0.0369 | -0.0260 | 0.0000 |
| transductive | RESNET_SE_CD | 0.2576 | 0.3522 | 0.2389 | 0.0000 | 0.4050 | -0.0833 | -0.0652 | 0.1250 |
| transductive | soft | 0.1507 | 0.2976 | 0.4172 | 0.0312 | 0.3400 | -0.0431 | -0.0396 | 0.1250 |
| causal_balanced25 | SVM | 0.2188 | 0.3807 | 0.6748 | 0.9375 | 0.3000 | -0.0301 | -0.0248 | 0.0000 |
| causal_balanced25 | RESNET_SE_CD | 0.3031 | 0.3935 | 0.1976 | 0.0938 | 0.4400 | -0.0768 | -0.0615 | 0.1250 |
| causal_balanced25 | soft | 0.2039 | 0.3314 | 0.4282 | 0.6875 | 0.3700 | -0.0403 | -0.0406 | 0.1250 |
| causal100 | SVM | 0.3365 | 0.4949 | 0.9789 | 0.9062 | 0.0700 | -0.0230 | -0.0135 | 0.0000 |
| causal100 | RESNET_SE_CD | 0.7688 | 0.7433 | 0.0793 | 0.5938 | 0.6300 | -0.0033 | -0.0033 | 0.0000 |
| causal100 | soft | 0.5890 | 0.6410 | 0.3837 | 0.6875 | 0.6825 | -0.0309 | -0.0236 | 0.0000 |

## Readings

Reading 1 (soft vote; the lead conditions):

- transductive: steady-state error change -4.31 pt, transition-zone error change -3.96 pt, paired median added delay +125 ms (881 transitions)
- causal_balanced25: steady-state error change -4.03 pt, transition-zone error change -4.06 pt, paired median added delay +125 ms (876 transitions)

So: the five-window vote cuts steady-state error and adds under 250 ms median delay under both lead conditions, so Section 4.4.3's smoothing paragraph can be stated for real transitions, with these numbers.

## Finding, reported on its own: the causal 100-window buffer collapses

A buffer taken from the start of a real continuous session does not work as a normalization reference: SVM: steady-state error 0.166 transductive, 0.336 causal-100; RESNET_SE_CD: steady-state error 0.258 transductive, 0.769 causal-100; soft: steady-state error 0.151 transductive, 0.589 causal-100. This is not a reading of S2.5; it is the measurement that supports the need for a scripted commissioning step (decision D-6b). The causal-100 rows stay in the table and in s2_measures.csv, and Reading 1 is not taken from them.

Reading 2 (S2b, 400 ms against 250 ms): NOT computed here; it needs the 400 ms per-window predictions with circuit and time (kc23_s2b_window_trade.py).

Full tables: s2_measures.csv, s2_decision_delay.csv, s2_zone_labels.csv.

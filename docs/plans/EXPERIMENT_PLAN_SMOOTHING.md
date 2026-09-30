# EXPERIMENT_PLAN_SMOOTHING.md: does a causal majority vote change the deployable figure, and is the critical-error rate really an upper bound?

**Why.** Two sentences in the thesis assert something that was never measured. §3.2.4 says predictions
are scored one window at a time with no smoothing, "which is a deliberate scope decision and makes
every per-window figure reported in this thesis a conservative estimate of what a smoothed controller
would achieve". §5.7.1 says "the per-window critical-error rate is an upper bound on the
per-transition risk rather than the risk itself" because a real controller would debounce. Both are
plausible and both are assumptions. For a thesis whose fourth contribution is deployability, the
cheapest available measurement is the one that turns them into results.

A majority vote over the current window and the previous k-1 windows uses only past data, so it is
legal under the causal constraint fixed in §3.13.1 and can be applied to the causal ensemble without
weakening any causality claim.

**This experiment needs no retraining and no GPU.** Everything it needs is already on disk.

**Verified on disk.** `results_causal_ensemble/proba_calib{25,50,100}/` each contain
`SVM_sub{01..40}.npz` and `RESNET_SE_sub{01..40}.npz`, and every one of those files carries three
arrays: `proba` of shape (n_windows, 4), `y_true` of shape (n_windows,), and **`is_buffer`** of shape
(n_windows,). `results_causal_ensemble/report.csv` and the three `calib{K}_subjectwise.csv` files hold
the published per-subject figures. The class order is the `LABELS` order used throughout,
`[DNS, STDUP, UPS, WAK]`.

The 250 ms meta CSV
`features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv`
carries `subject, movement, status_mode, confidence, fs, t_start, t_end, win_samples, n_channels,
y_str, y_int, subject_str, subject_int`. `t_start` is per movement trial, so ordering within a trial
is by `t_start` and trials are separate clocks. This matters: a majority vote must never span a trial
boundary.

Run from `06_Code/` in the project `.venv`. CPU only.
**New script:** `run_causal_smoothing.py`
**Outputs:** `results_causal_smoothing/smoothing_by_k.csv`,
`results_causal_smoothing/smoothing_subjectwise.csv`,
`results_causal_smoothing/smoothing_wilcoxon.csv`,
`report_figs/new_experiments/causal_smoothing.png`

Do NOT fabricate numbers. Do NOT edit `results_causal_ensemble/`. Do NOT edit any thesis chapter.

---

## Phase 0. Establish the row-order mapping, and gate on it [RUN, CPU]

This is the only real risk in the experiment. The npz files carry no window index, so the row order
has to be tied to the meta rows before any temporal operation is meaningful. Do not assume it.

1. For each subject, take that subject's rows from the meta CSV in file order. Compare the count
   against `y_true.shape[0]` in `proba_calib100/RESNET_SE_sub{K}.npz`, and compare `y_int` against
   `y_true` element by element.
2. **GATE:** counts must match for all 40 subjects and the label vectors must be identical. If they
   match, the npz row order is the subject's meta row order and `t_start`, `t_end` and `movement` can
   be attached directly.
3. If they do **not** match, stop. Report the first mismatching subject, both counts, and the first
   ten disagreeing labels. Do not try to realign by sorting or by guessing: an incorrect mapping
   would silently produce a plausible smoothing result that means nothing. In that case the honest
   fallback is to re-emit the causal probabilities with a window index, which is a separate run.
4. Also confirm that `SVM_sub{K}.npz` and `RESNET_SE_sub{K}.npz` agree on `y_true` and `is_buffer`
   for every subject. They must, since both members score the same windows.

## Phase 1. Reproduce the published causal ensemble, unsmoothed [RUN, CPU]

Combine the two members by the same soft vote used offline, a plain mean of the two probability
matrices, excluding rows where `is_buffer` is true from the score.

**GATE:** the calib-100 soft vote must reproduce **0.817** macro-F1 as the 40-fold mean, and calib-50
and calib-25 must reproduce **0.811** and **0.786**, each to within 0.003. The per-member figures
should come out at 0.732 for the SVM and 0.800 for ResNet-SE+CD at calib-100. If any of these misses,
stop and report which, with the reproduced value beside the published one. Everything downstream is a
delta against these numbers and is worthless if the base does not reproduce.

## Phase 2. Causal majority vote at k = 1, 3, 5, 7 [RUN, CPU]

For each buffer length, each subject and each k:

1. Sort that subject's windows by `movement` then `t_start`, so that the vote runs along each trial's
   own clock and **never crosses a movement-trial boundary**. State in the report how many windows sit
   within k-1 of a trial start, since those are the windows that get a shorter effective vote.
2. Take the ensemble's per-window argmax as the raw prediction. For window i, the smoothed prediction
   is the modal raw prediction over windows i-k+1 to i within the same trial, using only windows
   already observed. At a trial start, vote over however many windows are available. Break ties toward
   the most recent window's raw prediction, and say so in the output.
3. Score macro-F1 on non-buffer windows only, exactly as Phase 1 does.
4. Also compute the **DNS to WAK critical-error rate** at every k, defined as it is in §4.3 and
   §5.7.1, since that is the quantity §5.7.1's upper-bound claim is actually about. The macro-F1
   number is interesting; this one is the point of the experiment.

Two further arms, both cheap, both worth having:

- **Probability-averaged smoothing** as well as vote-of-argmax: mean the probability rows over the
  same causal window, then argmax. It usually beats a hard vote and it is equally deployable. Report
  both, and label which is which.
- **Transductive reference at k = 5.** Apply the same smoothing to the offline transductive ensemble
  so the 85.8% figure has a smoothed counterpart. Without it a reader cannot tell whether smoothing
  closes the causal gap or lifts both ends equally, which is the same distinction §4.15 draws for
  window length.

## Phase 3. Statistics [RUN, CPU]

Per-subject paired contrasts across the 40 subjects, for the calib-100 configuration: k = 3 against
k = 1, k = 5 against k = 1, and k = 5 against k = 3. Paired Wilcoxon signed-rank, Cohen's d, BCa 95%
intervals, Holm correction within this family. Same again for the critical-error rate.

These **do** add members to the whole-thesis Benjamini-Hochberg family. Record which and how many in
the report, but do **not** run `recompute_unified_fdr_v5.py` here: the family is rebuilt once, in the
write-up wave, after every Phase 2 experiment has landed.

## Phase 4. Figure [RUN, CPU]

`report_figs/new_experiments/causal_smoothing.png`. Two panels sharing an x axis of k: macro-F1 on the
left, DNS to WAK critical-error rate on the right, one line per buffer length, with the transductive
smoothed reference as a dashed horizontal line on the left panel. Error bars are one standard error
across the 40 subjects. Match the existing `report_figs/new_experiments` style.

---

## What each outcome means, and what it does to the next experiment

**This experiment is Gate A in `EXPERIMENT_RUN_ORDER.md`.** Its result decides how the causal-filter
experiment is designed, so the report has to answer the gate question explicitly rather than leaving
it to be read off the table.

**Outcome A, smoothing lifts the causal figure materially (2 pp or more at k = 5).** Then the number a
reader should quote is the smoothed one, §5.7.1's assumption is confirmed with a magnitude attached,
and the causal-filter experiment must score both smoothed and unsmoothed. Say so in the report.

**Outcome B, smoothing barely moves it (under 1 pp).** Then §5.7.1's claim stands as written, the
"conservative estimate" sentence in §3.2.4 gets a number that happens to be small, and the filter
experiment scores unsmoothed only. This is the cheaper branch and it is a perfectly good result: it
says the per-window figures were not hiding much.

**Outcome C, macro-F1 rises but the critical-error rate does not fall, or rises.** This is the one
worth watching for, and it would be the most interesting result in the set. It would mean smoothing
buys average accuracy while doing nothing for the error that actually matters, because DNS to WAK
confusions are not isolated flickers but sustained stretches. §5.7.1's upper-bound claim would then be
wrong in the direction that matters, and it would need rewriting rather than confirming. Do not soften
this if it happens.

**Outcome D, the Phase 0 gate fails.** Report it and stop. The experiment is still cheap once the
probabilities carry a window index; it is just not free.

## After the runs: hand back for the write-up cascade

Leave the CSVs and the PNG. Report:

1. Whether Phase 0's mapping gate passed, and whether SVM and ResNet-SE agreed on `y_true` and
   `is_buffer` for all 40 subjects.
2. The Phase 1 reproduction, three buffer lengths, beside the published 0.786 / 0.811 / 0.817.
3. The full k table for both smoothing variants, macro-F1 and critical-error rate, with the
   transductive smoothed reference.
4. How many windows sat within k-1 of a trial start at k = 5, as a fraction of the scored set.
5. Which of Outcome A, B, C or D occurred, in one sentence, with the numbers that decide it, and
   therefore what the causal-filter experiment should be scored on.
6. How many new BH family members this adds.

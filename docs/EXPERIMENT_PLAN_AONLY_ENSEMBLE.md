# EXPERIMENT_PLAN_AONLY_ENSEMBLE.md: does the 85.8% headline survive the active-only class control?

**Why.** §4.3.1 is one of the strongest passages in the thesis. It removes sit-to-stand's quiescent
windows, the class-construction objection an EMG examiner will raise first, and shows the normalization
finding gets **larger** rather than smaller: +9.54 pp for the SVM and +6.66 pp for the RF against
published margins of 6.90 and 5.10. It carried the SVM, the Random Forest and the ResNet-SE model of
record (82.3%) through the control.

It did not carry the **soft-vote ensemble**. So 85.8%, the single most quotable number in the thesis,
is the only major result that has not been through the control designed to test the objection against
it. That is a thirty-second find for an examiner and a cheap fix for us.

The abstract now states, as of the P1.5 edit, that restricting sit-to-stand to its active windows
costs that class about twelve points of F1 and that the normalization finding is larger on the
restricted task. That sentence is true and measured for the members. This experiment gives the headline
the same treatment.

**Verified on disk.**

- `results_aonly_resnet_se_cd_persubj/proba/RESNET_SE_AONLY_sub{01..40}.npz` already exists, so the
  deep member is done. `cnn_arch_summary.csv` there should hold the 82.3% figure §4.3.1 reports.
- `results_aonly_persubj/` holds the classical active-only LOSO run, but its `run_config.json` shows
  `save_proba: false`, so only `y_pred` and `y_true` were saved. Soft voting needs calibrated
  probabilities, so the **SVM must be re-run once** on the active-only features with probability
  output enabled.
- That same `run_config.json` shows `reuse_params_dir: results_loso_freq_persubj`, so the re-run
  reuses the already-tuned hyperparameters and skips the nested grid search. This is the reason the
  experiment is cheap rather than a full classical LOSO.
- Active-only features:
  `features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_ext.npz`
  with meta `..._Aonly_features_meta.csv`.

Run from `06_Code/` in the project `.venv`. CPU. Keep `--n-jobs 1`, use `--rf-n-jobs 4` if RF is run
at all. Do NOT fabricate numbers. Do NOT edit any thesis chapter.

**Outputs:** `results_aonly_ensemble/aonly_ensemble_summary.csv`,
`results_aonly_ensemble/aonly_ensemble_subjectwise.csv`,
`report_figs/new_experiments/aonly_ensemble.png`

---

## Phase 0. Reproduce the published active-only figures before adding anything [RUN, CPU]

1. Read `results_aonly_persubj/*SVM_nested_loso_summary.csv` and `*RF_nested_loso_summary.csv`, and
   `results_aonly_resnet_se_cd_persubj/cnn_arch_summary.csv`.
2. **GATE:** these must reproduce what §4.3.1 reports: per-class STDUP F1 of 0.84 for the SVM, 0.83
   for the RF and 0.88 for the ResNet-SE model of record, with the ResNet-SE mean macro-F1 at 82.3%.
   The per-subject margins over `results_aonly_global` must come out at +9.54 pp for the SVM and
   +6.66 pp for the RF. If any of these does not reproduce, stop and report which, because the whole
   experiment is a comparison against them.
3. Confirm the active-only window count is 13,643 with 1,982 STDUP windows, or 14.5%, as §4.3.1
   states, and that no subject-by-class cell falls below 32 windows.

## Phase 1. Re-run the active-only SVM with probabilities [RUN, CPU]

One command. Reuse the tuned parameters so this is a scoring pass rather than a search.

```
python train_classical_loso.py \
    --features features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_ext.npz \
    --meta     features_out/freq_fs1920_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_Aonly_features_meta.csv \
    --out results_aonly_persubj_proba --models SVM --norm-mode per_subject \
    --cv-scheme loso --inner-splits 5 --n-jobs 1 --seed 42 \
    --reuse-params-dir results_loso_freq_persubj \
    --save-proba --proba-out results_aonly_persubj_proba/proba \
    --save-preds --flush-preds --resume
```

**SANITY, and this one has a known trap.** §4.13.1 records that enabling Platt scaling routes the
SVM through pairwise coupling rather than the decision function, and that the two disagree on four to
eight per cent of windows. On the causal ensemble that showed up as 73.2% against 74.8% for the same
configuration. Expect the same kind of small shift here between the probability SVM and the published
active-only SVM, and **report it explicitly rather than treating either as an error**. If the shift
exceeds about 2 pp, stop and report, because that would be larger than the routing effect measured
elsewhere.

## Phase 2. Combine, and score the ensemble on the active-only class set [RUN, CPU]

1. Align the new SVM probabilities with `RESNET_SE_AONLY_sub{K}.npz` per subject. Gate on `y_true`
   matching element by element for all 40 subjects before combining anything. Class order is
   `[DNS, STDUP, UPS, WAK]` in both.
2. Soft vote, the plain mean of the two probability matrices, exactly as the headline uses offline.
3. Report, per subject and as the 40-fold mean: macro-F1, per-class F1, and the DNS to WAK
   critical-error rate.
4. Report the same for each member alone on this class set, so the ensemble's gain over its members
   can be read on the harder task as well as on the published one.
5. For completeness, also report the hard vote, since Appendix A.2 documents the earlier hard-vote
   generation and a reader may ask whether the soft-vote advantage survives the control.

## Phase 3. Statistics and the comparison that matters [RUN, CPU]

Per-subject paired contrasts across the 40 subjects, Wilcoxon signed-rank, Cohen's d, BCa 95%
intervals, Holm within this family:

- active-only ensemble against the active-only ResNet-SE+CD alone,
- active-only ensemble against the active-only SVM alone.

Do **not** run a paired test of the active-only ensemble against the published 85.8%. The two are
scored on different window sets, so a paired test across subjects would be comparing quantities that
do not correspond window for window. Report the difference as a level difference with both sample
sizes stated, the way §4.3.1 reports STDUP falling from 0.96 to 0.84.

These paired contrasts add members to the whole-thesis Benjamini-Hochberg family. Record how many. Do
not rebuild the family here.

## Phase 4. Figure [RUN, CPU]

Extend the existing Figure 4.6 layout rather than inventing a new one: per-class LOSO F1 under the
published class definition against the active-only definition, with the soft-vote ensemble added as a
fourth model beside the SVM, the RF and the ResNet-SE model of record. Save as
`report_figs/new_experiments/aonly_ensemble.png`. Do not overwrite the existing Figure 4.6 asset.

---

## What each outcome means

**Outcome A, the ensemble lands near 82 to 83%.** Then it behaves like its deep member, the control
costs the headline roughly what it cost every other model, and §4.3.1 gains one row while the
abstract's new class-set sentence gains a number. This is the expected case.

**Outcome B, the ensemble falls further than its members did.** Then the soft vote was drawing part of
its advantage from the quiescent class, which would be a real finding and would need saying in §4.3.1,
§5.7 and the abstract. It would not touch the normalization finding, which §4.3.1 already shows gets
stronger on the restricted task.

**Outcome C, the ensemble holds up better than its members.** Also worth stating plainly, and it would
strengthen §5.7's argument that the two members make different errors.

In every case the honest framing is the one §4.3.1 already uses: the active-only task is the harder
version of the problem, the published headline is measured on the published class definition, and the
two numbers are reported side by side rather than one replacing the other.

## After the runs: hand back for the write-up cascade

Leave the CSVs and the PNG. Do not edit chapters, and do not touch `results_aonly_persubj/`. Report:

1. Whether the Phase 0 gate reproduced §4.3.1's figures.
2. The size of the Platt-routing shift on the active-only SVM, with both values.
3. The active-only ensemble macro-F1, per-class F1 and critical-error rate, beside its two members.
4. The published 85.8% and the active-only ensemble figure side by side, with both window counts.
5. Which of Outcome A, B or C occurred, in one sentence.
6. How many new BH family members this adds.

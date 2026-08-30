# EXPERIMENT_PLAN_CAUSALITY.md: the four runs that close the PRC 2026 causality critique

**Written 29 August 2026.** Companion to `EXPERIMENT_PLAN_CRITIQUE.md`. That plan answered five external LLM reviews; this one answers a single line of criticism delivered in person at the UG DCS postgraduate conference, 27 to 29 August 2026.

**The criticism.** Every methodological choice must be stated and defended. The reviewers zeroed in on the causal estimator: what makes it causal rather than associative, given that using an estimator you call causal does not by itself establish causality.

**The diagnosis, from an audit of the text, the code and the result CSVs.** The thesis uses "causal" only in the signal-processing sense, an estimator whose output at window t depends on windows already observed and nothing later. That usage is correct and the code implements it faithfully. The word is never defined, which is what invited the question. The critique nevertheless lands somewhere real, and it is not Section 4.14. It is Sections 4.13.1 and 4.13.2, the over-alignment result, which is the one place the thesis claims a mechanism. Three defects were found there and in the causal protocol, and this plan is the four runs that close them. Everything else the audit turned up is handled in writing.

**Conventions, unchanged from the other plans.** Run from `06_Code/` with the venv Python (`& '.\.venv\Scripts\python.exe' ...`). `--resume` everywhere, checkpoint per subject, append immediately, crash-safe. Seed 42. `LABELS = ["DNS","STDUP","UPS","WAK"]`, alphabetical, which is encode order. GridSearchCV `n_jobs` stays 1 because of the Windows loky deadlock on repeated `.fit()`; RF parallelism goes through `--rf-n-jobs`. Every experiment writes `*_subjectwise.csv` plus `*_summary.csv` into its own `results_<experiment>/`.

**Do not fabricate numbers. Every one of these four is designed so that either outcome is usable. If a result contradicts the thesis, report it plainly and stop rather than tuning until it agrees.**

**Shared inputs.**
```
FEAT = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_ext.npz   # key X, (26347, 72)
META = features_out/freq_windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR_features_meta.csv
NPZ  = windows_WAK_UPS_DNS_STDUP_v1_w250_ov50_conf60_AorR.npz                                   # key X_env, (N, 9, 500)
```

**Existing artifacts you will reuse, all verified on disk 29 August 2026.**
- `analyze_between_subject_variance.py`. Defines the five alignment rungs (`rung0_global_z` through `rung4_full_whiten_recolor`), the RBF MMD with median-heuristic bandwidth, the cross-validated subject probe, and `stratified_subsample`. Reuse these functions; do not reimplement them.
- `results_variance_decomposition/alignment_ladder.csv`. The published ladder. Rung 3 silhouette 0.02296, rung 4 silhouette -0.00606, rung 4 MMD removed 72.96%.
- `run_coral_loso.py` and `results_loso_freq_coral/`. CORAL LOSO, SVM 0.7236, RF 0.7466, at the default `--lam 1.0` which was never swept.
- `run_causal_ensemble.py`. The causal ensemble harness. Its train-once-per-subject-then-evaluate-many-configs structure is the pattern E-C3 extends.
- `_bestparams.json`. Per-subject best parameters from the authoritative per-subject LOSO run.
- `rescore_streaming_buffer_v2.py`. Demonstrates that fitting directly with cached best parameters reproduces the GridSearchCV numbers exactly.

---

## E-C1. Downstream F1 for every rung of the alignment ladder. [RUN, CPU, the important one]

**The question.** Section 4.13.2 currently reads: full covariance whitening removes the most subject structure "and yet the classifier built on it performs worse, at 72.4% against 77.7% for the SVM." No classifier was ever built on rung-4 features. The 72.4% is CORAL's LOSO F1 from `results_loso_freq_coral/coral_SVM_subjectwise.csv` (verified: 0.7236), produced by a different transform in a different script. `analyze_between_subject_variance.py` line 341 says so explicitly in a comment. So the mechanism argument measures the probe and the silhouette on one operator and the F1 on another, and argues the two are analogous. That is precisely the associative-not-causal shape the reviewers objected to.

**What closes it.** Train the SVM under the identical LOSO protocol on the features produced by each of the five rungs, so that alignment strength, subject-identity, class separability and downstream accuracy are all measured on the same manipulated variable. The ladder then becomes a single-axis intervention with a measured dose-response instead of a cross-operator inference.

**New script:** `run_alignment_ladder_loso.py`. Output dir `results_alignment_ladder_loso/`.

**Procedure.**
1. Import the rung functions from `analyze_between_subject_variance.py`. Do not copy them.
2. For each rung in 0, 1, 2, 3, 4: build `Xr = rung(X, subjects)` once over the full matrix, exactly as the ladder analysis does, so the F1 is computed on the same features the published probe and silhouette were computed on.
3. Run standard LOSO over the 40 subjects on `Xr` with **full nested GridSearchCV**, mirroring `train_classical_loso.py`: inner 5-fold GroupKFold, grid `C` in {1, 5, 10} with `gamma='scale'`, `class_weight='balanced'`, scoring `f1_macro`.
4. **Re-tune per rung. Do not reuse `_bestparams.json` here.** Those parameters were selected on rung-3 features. Carrying them to rungs 0, 1, 2 and 4 would let a critic say whitening was handicapped by a hyperparameter chosen for a different geometry, which would destroy the point of the experiment.
5. Report per-subject F1, then mean and standard deviation, per rung.
6. Paired Wilcoxon plus Cohen's d for rung 4 against rung 3, and rung 3 against rung 0.

**Validation gates, both mandatory. Stop and report if either fails.**
- Rung 3 must reproduce the published per-subject z-score SVM figure, **0.7767**, to within 0.002. Rung 3 is `per_subject_zscore`, so this is the same computation by a different route.
- Rung 0 must land near the published global baseline, **0.708**. Note it will not match exactly: the published baseline fits `StandardScaler` on the 39 training subjects inside the Pipeline, whereas `rung0_global_z` standardizes over all 40 pooled. Report both the value and the gap, and say plainly which definition each uses. A gap under about 0.005 is expected and is itself worth a sentence in the thesis.

**Expected shape, stated in advance so it can be falsified.** Subject-probe accuracy falls monotonically across the ladder (0.777, 0.043, 0.909, 0.024, 0.012, already measured). If the over-alignment account is right, F1 should rise from rung 0 to rung 3 and then fall at rung 4, tracking the silhouette rather than the probe. **If rung-4 F1 comes out at or above rung 3, the over-alignment account as written is wrong and the thesis must be changed, not the experiment.** That is the whole point of running it.

**Cost.** Five rungs at roughly the cost of one original classical LOSO SVM run each. Rung 3 is insurance and is cheap relative to being wrong.

**Outputs.** `ladder_loso_{rung}_SVM_subjectwise.csv` per rung, plus `alignment_ladder_loso_summary.csv` with columns rung, name, f1_mean, f1_sd, n. Then join against `results_variance_decomposition/alignment_ladder.csv` into `alignment_ladder_full.csv` carrying rung, name, mmd_removed_pct, w1_removed_pct, subject_probe_bal_acc, silhouette_by_class, f1_macro_mean. That joined file is the new Table 4.16.

---

## E-C2. Is rung 4 a fair whitening, or an artifact of its regularizer? [RUN, CPU, minutes]

**The question.** `rung4_full_whiten_recolor` adds `lam=1.0 * I` to each subject's covariance computed on **raw** features. Measured on this matrix, feature variances span 8.2 orders of magnitude (min 3.4e-5, max 5.8e3). About a quarter of features have variance below 0.01, so for them the ridge is more than a hundred times their own variance and almost no whitening happens, while for the largest features λ=1 is negligible and whitening is essentially complete. Rung 4 is therefore a strongly non-uniform, unit-dependent shrinkage rather than a full whitening.

**This matters more than it first appears.** `run_coral_loso.py` applies `StandardScaler().fit(X[tr])` **before** `coral_align`, so CORAL whitens features that are already on a common scale and its λ=1.0 is a sensible shrinkage equal to unit variance. Rung 4 has no such step. So rung 4, whose docstring calls it "the per-subject analogue of CORAL's full second-order alignment", differs from CORAL in exactly the way that makes its regularizer meaningful. An examiner who opens the two functions side by side will find this in about a minute.

**New script:** `check_rung4_robustness.py`. Output dir `results_variance_decomposition/` (append, do not overwrite the published ladder).

**Procedure.** Recompute the rung-4 row only, under four variants, holding the metric code and the point set identical to the published ladder:
- `raw_lam1`. The published transform, as a reproduction check.
- `prez_lam1`. Apply `rung0_global_z` first, then per-subject whiten and recolor with λ=1.0. This is the faithful CORAL analogue.
- `prez_scalefree_a1`. Pre-standardized, λ = trace(Cs)/p.
- `prez_scalefree_a01`. Pre-standardized, λ = 0.1 × trace(Cs)/p.

For each, report MMD removed, Wasserstein-1 removed, cross-validated subject probe and class silhouette, using the same `stratified_subsample` seed and cap so the numbers are comparable to the published rungs.

**Validation gate.** `raw_lam1` must reproduce the published rung-4 row: MMD removed 72.96%, probe 0.01187, silhouette -0.00606.

**What we are testing.** The load-bearing claim is the *ordering*: silhouette peaks at rung 3 and falls at whitening, while the probe keeps falling. If that ordering survives all four variants, the finding is robust to the regularizer and a footnote settles it. **If the ordering flips under a scale-free ridge, then the negative silhouette was an artifact and Section 4.13.2 needs rewriting.** Report which it is.

**Optional, only if E-C1 is already finished and the ordering held:** run the best-behaved variant through E-C1's LOSO harness as a sixth rung, so the F1 column covers it too.

---

## E-C3. Does the calibration buffer's class composition flatter the deployable figure? [RUN, GPU for the ensemble arm]

**The finding that prompted this.** `t_start` in META restarts at zero for each of the four movement recordings. Confirmed: for every subject, `min(t_start)` is 0.0 for WAK, UPS and DNS and 1e-9 for STDUP, and the four recordings run 10 to 45 seconds each. So sorting a subject's windows by `t_start`, which is what `normalise_test_subject` does, interleaves the four movements. The "first K time-ordered windows of the held-out subject's session" is in fact a near-perfectly class-balanced sample drawn from the opening seconds of all four recordings. Measured across all 40 subjects, the first 100 windows average 22.0 WAK, 25.4 UPS, 25.9 DNS and 26.7 STDUP, and **all four classes are present for all 40 of 40 subjects**.

That is not a bug, and the result is not invalid. What it is, is a different protocol from the one the thesis describes, and a more favourable one. A genuinely contiguous session-start buffer would be dominated by whatever the user did first, giving a biased mean and standard deviation. The honest reading is that the code implements a **short scripted unlabeled calibration set**, about twelve seconds at K=100 covering each movement, which is a reasonable and arguably standard way to commission a clinical device. The thesis must say that instead, and it should bound how much the balance is worth.

**The question.** How much of the 81.7% depends on the buffer containing all four movements?

**New script:** `run_buffer_composition.py`. Output dir `results_buffer_composition/`.

**Procedure.** Add buffer-selection modes to the existing machinery rather than forking it. Extend `normalise_test_subject` in `run_streaming_norm_loso.py` with a `buffer_idx` argument, or add a sibling function, so the buffer is an explicit index set rather than implicitly the first K of the time sort. Modes at K=100:
- `mixed100`. The published behaviour, first 100 by `t_start`. Reproduction check.
- `single_WAK`, `single_UPS`, `single_DNS`, `single_STDUP`. First 100 windows of that movement's recording only, which is the true contiguous single-activity buffer.
- `balanced25`. 25 from the start of each movement, the explicitly scripted protocol, stated as such.

In every mode the buffer windows are excluded from the F1, using the same `is_buffer` mask logic as `run_causal_ensemble.buffer_mask`. Scoring is over all remaining windows of all four classes in every mode, so the modes differ only in which windows fit the normalizer.

**Efficiency, which decides whether this is cheap or expensive.**
- **SVM arm:** the model is fitted on training-subject features only, and the buffer mode changes nothing about training. So fit **once per subject** and evaluate all six modes from that one fitted model. 40 fits total. Use cached `_bestparams.json` here, which is legitimate because the model is identical across modes and the comparison is within-model.
- **Ensemble arm:** same logic. Train the ResNet-SE+CD **once per held-out subject**, exactly as `run_causal_ensemble.stage_cnn` already does, then evaluate all six buffer modes by re-normalizing the test subject and re-running inference. Do not retrain per mode.

**Validation gate.** `mixed100` must reproduce the published values: SVM buffer-excluded 0.7476, ResNet-SE+CD 0.7995, soft ensemble 0.8168.

**Outputs.** `buffer_composition_subjectwise.csv` with columns subject, model, mode, f1_incl, f1_excl; and `buffer_composition_summary.csv`. Paired Wilcoxon of each single-movement mode against `mixed100`.

**What the thesis does with it.** If the single-movement modes sit close to `mixed100`, the deployable figure is robust to buffer composition and that is a strong sentence. If they fall materially, Section 4.14.1 reports a range bounded by the worst single-movement buffer and the scripted one, which is more informative than the single point it currently quotes. Either way the description gets corrected.

---

## E-C4. Was CORAL beaten fairly? [RUN, CPU]

**The question.** Finding A rests in part on per-subject z-scoring outperforming CORAL. `run_coral_loso.py` exposes `--lam` with a default of 1.0, and `jobs_coral.txt` shows both the SVM and RF runs were launched without it. So CORAL's only hyperparameter was never swept. An examiner asking why every choice was not defended has an easy shot here: you beat a baseline whose one tunable was left at its default.

**Procedure.** Re-run `run_coral_loso.py --models SVM` at `--lam` in {0.01, 0.1, 1.0, 10.0}, into `results_coral_lam_sweep/lam_<value>/`. The script already does full nested GridSearchCV per fold, so no code change is needed beyond the output directory. Add RF at the best and worst λ only, if time allows.

**Validation gate.** `lam=1.0` must reproduce **0.7236** for the SVM.

**Outputs.** `coral_lam_sweep_summary.csv` with lam, model, f1_mean, f1_sd, n. Paired Wilcoxon of the best λ against per-subject z-score (0.7767).

**What we expect and what would change the thesis.** CORAL is applied to globally standardized features, so λ=1.0 is already a defensible choice and the sweep most likely confirms it. **If some λ lifts CORAL above 0.7767, that is a material finding and Section 4.13 must be rewritten to report it.** Run it precisely because the answer is not certain.

---

## Traps, all previously paid for

- **`device_bash` background processes die with the parent shell.** Launch and wait in the same call, or run these from a normal terminal on the machine. These are long jobs; prefer a real terminal.
- **GridSearchCV `n_jobs` must stay 1.** Loky's reusable executor hangs on the second and later `.fit()` in a detached process on Windows. RF parallelism goes through `--rf-n-jobs`.
- **`subject` is authoritative, not `subject_int`.** `subject` is 1 to 40 and matches `_bestparams.json` keys and `heldout_subject` in `results_loso_freq_persubj`. `subject_int` is 0-indexed and is used internally by some scripts only.
- **Checkpoint per subject and append immediately.** Every existing script in this folder does; match it so a crash costs one fold.
- **Do not overwrite `results_variance_decomposition/alignment_ladder.csv`.** It is the published ladder and Table 4.16 depends on it. Write new files alongside.
- **Order metrics on an identical point set.** The published silhouette is computed on a fixed `stratified_subsample`. Any new silhouette must use the same seed and cap or it is not comparable.

---

## What to hand back

A short report per experiment: the command run, the validation gate and whether it passed, the summary table, and one paragraph on whether the result supports or contradicts the thesis as currently written. Write the report into `results_<experiment>/REPORT.md` and leave every CSV in place. Do not edit any file under `01_Thesis/`.

# RUN_QUEUE consolidated report

Executor: Claude Code, on Enam's machine. Written 4 September 2026, on completion of the last queue item (S-1).
Covers every item in `RUN_QUEUE.md`, each against its own pre-registered grid. Nothing in `01_Thesis/` was
edited; Section 4.17 and the unified Benjamini-Hochberg family were not touched; the deep model of record
(`resnet_se` + channel dropout, 0.840) was not changed; every run wrote a new `results_*` folder and its command
line (via `run_config.json` from B1 section 1.5, plus a `run_*.sh` per item).

**Three decision points are left open for Enam, as instructed: B2 section 2.5, B3 section 3.4, B4 section 4.5.**
They are stated below, not resolved.

**Escalation triggers: none fired.** S-2 matched to the bit; S-1's normalization delta stayed within its 3.0 pp
tolerance and in the favourable direction.

---

## Block 1. Mechanism programme close (GPU)

### P-9: occlusion-reduction transfer across backbones, four arms (`EXPERIMENT_PLAN_LOCUS.md` section 3)

Arms A1 resnet/none, A2 resnet/chandrop, A3 resnet_se/none, A4 resnet_se/chandrop. All four `attenuation.csv`
alpha=0 slices reproduce their occlusion columns to `maxdiff = 0.0` (section 3.2 gate). Reproduction trap
(section 3.5a) settled first: A1 summed occlusion cost 86.88 pp vs published 84.21 (diff +2.67 pp), A2 13.40 vs
15.03 (diff -1.63 pp), both inside the 3.0 pp gate; f1 within 0.7 pp. A1/A2 spread of 2.67 pp is the yardstick.

**Backbone grid: outcome S (split vindicated).** resnet_se occlusion reduction 6.12x (95% [4.94, 7.85]) vs
SE-free 6.48x (95% [5.11, 8.54]); |difference| 0.36x against the 0.88x shift a pure re-run already produces;
overlapping bootstrap intervals; per-subject paired backbone difference -5.07 pp (95% BCa [-13.76, +2.62],
p = 0.35), not distinguishable from zero. Section 4.8.2 may keep either figure provided it names the backbone.

**Transfer grid: outcome U (underpowered), same letter both backbones.** Censoring on the augmented arm is 36.9%
(resnet) / 37.5% (resnet_se) even at the 1.0 pp criterion, above the 30% ceiling; the reliance-agreement measure
fails for the third time. Per the plan: do not commission a fourth. Un-augmented reliance agreement is +0.150
(resnet) / +0.099 (resnet_se) against the P-8 Test-A benchmark of +0.167; both models track the data rather than
over-committing (`model_vs_data = tracks_data`). Per-subject coupling half retired as underpowered by design
(~151 subjects needed at rho -0.226, ~351 at +0.149).

### P-10: mean-preserving upward dose sweep, 3 runs (`EXPERIMENT_PLAN_LOCUS.md` section 4)

SD 0.40 / 0.50 / 0.60 / 0.80 / 1.00, p' derived from SD (0.138 / 0.20 / 0.265 / 0.390 / 0.50).

**Grid: outcome B (boundary found).** Peak at SD 0.50 (mean F1 0.8379), flat through the operating range
(0.40-0.60: 83.1 / 83.8 / 83.7 / 83.2), then a Holm-significant fall past the peak: SD 0.80 -4.63 pp
(Holm p = 3.9e-9, d = 1.18), SD 1.00 -10.65 pp (Holm p = 1.1e-11, d = 1.88). SD 1.00 (p' = 0.5) collapses
5.09 pp below the no-augmentation baseline: outcome X for that arm, a bound on the family, excluded from the
ceiling read. A genuine over-invariance boundary exists on the mean-preserving family near SD 0.80, at a dose the
unnormalized rate sweep never reached because its own mask artifact (P-7) masked it. P-1's structural reading
holds, relocated to a higher dose; a `p1_verdict_replacement_draft` is in `results_locus/p10_outcome.json`.
Caveat retained: two-to-three points past the peak is not a dose-response curve, and the boundary sits near the
p' = 0.5 degeneracy, so "over-alignment destroys structure" vs "the perturbation becomes untrainable" is not
fully separable.

Mechanism programme is closed with these two items.

---

## Block 2. Cheap audit items (CPU, no model runs)

### B1: run manifests and the canonical-numbers cross-check (`EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` section 1)

`build_run_manifest.py` built `RUN_MANIFEST.csv` over **143** `results_*` directories, per-field provenance scored
in the three states of section 1.3 (verified / documented / unknown, never guessed).

- **All six core fields verified: 0 directories. At least one core field UNKNOWN: 142.** No run in the repository
  ever wrote a configuration file. That is the B1 finding, not a script limitation.
- **Cross-check (section 1.4): one machine-verifiable contradiction, the known one.** `results_g3_noaug_instr`
  and `results_cd_resnet_nose_chandrop`, which back the "84.2 pp to 15.0 pp, 5.6x" occlusion result, both
  **verify as `--arch resnet`** (SE-free) from `cnn_arch_summary.csv`, while sections 4.8.2 / 5.7 / 6.2 state the
  number without naming the backbone and 5.7's adjacency implies `resnet_se`. P-9 settled the substance
  (outcome S); the number need not move, the three passages need the backbone named. That wording change is in
  the write-up queue, not here.
- No other canonical number sits on a directory whose verifiable config contradicts the thesis. Several canonical
  dirs (85.8% ensemble, 81.7% causal, variance decomposition, W-1, calibration) record no `arch` anywhere and are
  unverifiable-but-consistent, not contradictions.
- **Section 1.5 (recurrence):** `run_config_dump.py` added; `run_cnn_arch_loso.py`, `train_cnn_loso.py` and
  `train_classical_loso.py` now write `run_config.json` (argv, resolved paths, git commit, library versions) with
  an RNG-inertness assertion. Used by every Block 1 and Block 3 run in this queue.

Report: `B1_MANIFEST_REPORT.md`, `RUN_MANIFEST.csv`.

### B2: the 70 to 85 percent literature band (`EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` section 2)

Inclusion rule fixed before collection: true LOSO (not k-fold, not leave-one-session-out), lower limb, cohort and
class count near 40 subjects / 4 classes. 13 candidate studies enumerated with author, year, dataset, n, classes,
limb, protocol, metric, value; exclusions recorded.

**Grid: the "band changes" outcome, specifically "fails to populate".** No enumerated study meets all four
criteria. The 83-85 numbers carrying a LOSO label are HD-EMG (256-electrode) hand-gesture accuracies or few-shot
results that use target-subject data. What is defensible: the **lower anchor ~70 to 78 percent macro-F1** for
unadapted cross-subject classical pipelines (matches the thesis's own 70.8 global / 77.7 per-subject, and the one
genuine large-cohort inter-subject macro-F1 in the sources, NinaPro DB2 ~0.77, though upper limb). The **upper
bound of 85 has no true-LOSO lower-limb source.**

**Metric question (section 2.4):** the enumerated comparators report **accuracy**; the thesis reports
**macro-F1**. On SIAT-LLMD with STDUP at 55.7% of windows the two differ materially and macro-F1 is the harder
metric, so the comparison is **conservative, not flattering** - and this currently goes unstated in sections
2.3.2, 5.1.1 and 6.2.

**>> Decision point, section 2.5 (for Enam):** insert the enumeration into 2.3.2 as the band's evidence; or
narrow the positioning claim to the anchored 70-78 lower bound plus a note that true-LOSO lower-limb macro-F1
comparators are scarce; or restate the comparison on a single named metric. Each option touches 2.3.2, 5.1.1/5.11
and 6.2/6.3 including the Conclusion. **Not resolved.** Table: `04_Reviews_and_QA/LITERATURE_BAND_TABLE.md`.

### B3: the five-point correlations in Section 4.13.2 (`EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` section 3)

Section 3.2 substantive fix **works** - no fallback to 3.3 needed. Per-subject class silhouette computed at each
of the five ladder rungs in the 72-d feature space, turning a 5-point descriptive correlation into a 40x5 paired
analysis:

- rung 3 (mean+scale) vs rung 4 (full whiten): +0.038 silhouette, BCa [0.032, 0.044], p = 1.8e-12, d = 1.91,
  40/40 subjects positive.
- rung 3 vs rung 0 (global-z): +0.007, BCa [0.0046, 0.0101], p = 9.6e-6, d = 0.78, 32/40 positive.
- Within-subject Spearman of the five silhouettes against that subject's five macro-F1 values: mean rho +0.44,
  median +0.46, Wilcoxon vs 0 p = 1.3e-6, 35/40 positive.

The rung-3 peak and the silhouette->F1 ordering hold **within subjects, tested**, not just across five pooled
points. Also noted: Table 4.16 shows "MMD removed" as n/a at the baseline rung while the 4.13.2 correlation
treats it as 0.0 - reconcile.

**>> Decision point, section 3.4 (for Enam):** whether 4.13.2 gains the per-subject test, replaces the five-point
correlations with it, or reports both. Two new paired Wilcoxon raw p-values recorded for the family
(1.8e-12, 9.6e-6); **Section 4.17 not touched. Not resolved.** Output: `results_locus/b3_verdict.md`,
`b3_outcome.json`, `b3_per_subject_silhouette.csv`.

### B4 / B5: incomplete result files and hygiene (`EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` section 4)

- **B4.1 alignment ladder:** `alignment_ladder_full.csv`, `..._summary.csv`, `..._stats.csv` regenerated complete
  for all five rungs from the per-subject files; regenerated means 0.7094 / 0.7482 / 0.7186 / 0.7767 / 0.6752
  **verified against Table 4.16 before overwrite**; originals backed up to
  `_ARCHIVE/prebackups/b4_ladder_20260903/`.
- **B4.3 / B5 em dashes:** 59 replaced without meaning change - `README.md` 29, `REPRODUCE.md` 24,
  `_STRUCTURE.md` 6.
- **B4.4 code robustness:** `ensemble_v2_combine.py` `subject_probs` `min()` truncation turned into an assertion
  on matching `y_true` (verified it never fired); `ensemble_v2_subjectwise.csv` now written with a `subject`
  column. `preprocess_emg.py` `build_full_dataset` bare `except Exception` now raises `RuntimeError` and records
  missing subject-by-movement cells to a printed summary rather than swallowing them.
- **B4.2 the 400 ms ladder (`results_win400_ladder/alignment_ladder_full.csv`):** still carries the 250 ms
  geometry columns verbatim (`mmd_removed_pct`, `w1_removed_pct`, `subject_probe_bal_acc`,
  `silhouette_by_class`) beside genuinely-400 ms `f1_macro_mean`. Nothing in the thesis quotes it.

**>> Decision point, section 4.5 (for Enam):** recompute the 400 ms geometry vs delete the four stale columns and
keep the F1 - different costs. **Recommendation: delete the four columns.** No thesis number depends on the 400 ms
geometry, recompute is a multi-hour LOSO-geometry job for a table nothing cites, and a deleted column cannot be
misread whereas a wrong one can. If Enam wants the 400 ms geometry as a supplementary robustness point later it
can be computed then. **Not actioned pending Enam.** Everything else in B4/B5 is done (no reported number changes).

---

## Block 3. The re-runs

### B8: subject-dependent protocol, movement-blocked re-run (`EXPERIMENT_PLAN_AUDIT_REMEDIATION.md` section 4A)

Per-movement time-blocking with a guard band (section 4A.1 trap avoided: `t_start` restarts per movement, so
contiguous blocking on raw `t_start` order collapses folds onto one class). Guard band drops 8.2% of windows at
250 ms, 5.6% at 150 ms - higher than the plan's ~1% estimate, which is what 50% overlap costs. Pooled random vs
movement-blocked, same models, 6 configs x 4 models plus a CNN arm. **No outcome-X on any config** (every fold
keeps test windows and all four classes in training).

**Grid: RANKING FLIPS + CONTAINED (marginal).**

- The overlap leak was worth **3.7 to 6.1 pp of SD macro-F1** (mean -4.59 pp, every config/model p < 1e-9,
  paired across 40 subjects).
- Under a common movement-blocked protocol at 250 ms: SVM 88.83, RF 87.99, **CNN 85.91**, LDA 84.57. The CNN no
  longer leads; Table 4.1's "CNN first at 90.4" beside pooled classical numbers is the unfair comparison this
  stage corrects, and section 4.1's "the CNN leading narrowly under SD" does not hold on a common protocol.
- Corrected SD -> LOSO gaps: SVM 11.13, RF 10.69, CNN 10.51 pp (were 15-22 pp pooled). Inside the section 6.1
  10-25 pp range but at its floor, so "squarely within" should become "at the lower edge of".
- **Not affected:** LOSO is untouched (outer split is by subject; no overlapping window crosses it). Finding A,
  the alignment ladder, the 85.8% ensemble headline, the 81.7% causal figure, the external replication and the
  gap-reduction argument all stand.
- Reproduction caveat: pooled-random classical SD reproduces higher than Table 4.1 (pooled SVM 93.7 vs 87.4),
  most likely a per-fold nested-GridSearchCV / weighting difference; the pooled-vs-blocked delta and the
  common-protocol ordering are the controlled claims, absolute classical SD levels carry the caveat.

Output: `results_b8_sd/b8_verdict.md`, `b8_outcome.json`, `b8_all_configs.csv`.

### S-2: sampling-rate invariance (`EXPERIMENT_PLAN_PREPROC_AUDIT.md` S-2)

Features re-extracted at `--fs 1920` to a new tag. Column check: MNF and MDF equal the `--fs 2000` columns times
0.96 (= 1 / 1.041667); MAV / RMS / WL / ZC / WAMP byte-identical; spectral power ignores `--fs`.

**SVM LOSO per-subject on the new features vs `results_loso_freq_persubj/*SVM*subjectwise.csv`: all 40 subject
macro-F1 values matched; largest absolute difference 0.000e+00 (byte-identical, not merely FP noise).**
Pre-registered expectation met. **No escalation.** B10.2's invariance argument is confirmed empirically. Output:
`results_s2_fs1920_svm_persubj/`, `results_locus/s2_outcome.json`.

### S-1: active-only STDUP re-run (`EXPERIMENT_PLAN_PREPROC_AUDIT.md` S-1)

Windows rebuilt with active-only STDUP: 13,643 total (DNS 4069 / STDUP 1982 = 14.5% / UPS 4550 / WAK 3042);
min subject-by-class cell 32, no cell < 30, so no count-check outcome-X. Features at `--fs 1920`. Four classical
LOSO runs (SVM + RF x per-subject / global norm) plus the optional deep arm.

**Grid: outcome H (hierarchy holds).** The plan expected C (given the AUC-0.47 active-only separability).
Per-class F1, per-subject norm, active-only (published Freq-72 LOSO in parentheses):

| class | SVM active (pub) | RF active (pub) | deep arm (resnet_se+CD) |
|---|---|---|---|
| DNS | 0.717 (0.677) | 0.718 (0.662) | 0.804 |
| STDUP | **0.844** (0.961) | **0.831** (0.960) | **0.875** |
| UPS | 0.762 (0.763) | 0.768 (0.754) | 0.810 |
| WAK | 0.733 (0.702) | 0.732 (0.709) | 0.782 |

STDUP stays top for all three models. Classical models keep the exact published ordering
STDUP > UPS > WAK > DNS; the deep model has STDUP > UPS > DNS > WAK (tail order differs, not part of the claim).
STDUP per-class F1 drops ~0.12 for the classical models and its lead over UPS compresses from ~0.20 to ~0.08, but
it stays clearly top and the other three classes each improve slightly. The deep model of record is barely
affected (STDUP 0.875; mean macro-F1 0.823, above both classical models and the SimpleEMGCNN headline 0.754,
close to the full-set 0.840). **The biomechanical reading is not a rest-window artefact; the thesis's
"STDUP >>" should soften to "STDUP >", reported as prominently as outcome C would have been.**

**Normalization delta (independent escalation gate): reported, no escalation.** Per-subject still beats global,
strongly: SVM +9.54 pp (BCa [+7.73, +11.43], p = 3.5e-11, d = 1.56), RF +6.66 pp (BCa [+5.25, +8.53],
p = 2.5e-10, d = 1.26). Published +6.90 / +5.10; drift +2.64 / +1.56 pp, both inside the 3.0 pp tolerance, both
positive (per-subject calibration matters at least as much on the harder problem). **Finding A is not
class-definition-dependent.**

**Autonomous judgement call recorded:** the plan lists the deep arm as optional and droppable ("the classical arm
answers the question on its own"). I ran it, and re-ran it once with `--save-proba` so it could speak to the
hierarchy (per-class F1) rather than only macro-F1. Rationale: H is the plan's thesis-favourable outcome to be
featured, and a confirmatory row on the exact model-of-record family strengthens it at the cost of one ~40 min
GPU LOSO. It did not change the outcome letter.

Output: `results_aonly_persubj/`, `results_aonly_global/`, `results_aonly_resnet_se_cd_persubj/`,
`results_locus/s1_outcome.json`.

**Operational note:** the S-1 classical run was babysat by `s1_memory_watchdog.sh` after an executor error left
two `train_classical_loso.py` pipelines contending on a 16 GB machine (a bash wrapper survived a `Stop-Process`
that only killed its Python child, and marched on to the next norm pass). The watchdog does a proactive
kill + `--resume` when free RAM drops below a threshold or sustained hard page-faults appear; ~39 proactive
restarts, per-subject checkpointing meant no lost work, all four runs finished `exit 0` at 40/40. `--rf-n-jobs`
was also dialled from 8 to 4 to lower the per-subject memory peak.

---

## New paired tests generated by this queue (for the eventual Section 4.17 recompute, not applied here)

| source | test | raw p |
|---|---|---|
| B3 | per-subject silhouette rung 3 vs rung 4 | 1.8e-12 |
| B3 | per-subject silhouette rung 3 vs rung 0 | 9.6e-6 |
| B3 | within-subject silhouette->F1 Spearman vs 0 (Wilcoxon) | 1.3e-6 |
| B8 | pooled vs movement-blocked SD F1, per config (6 configs + CNN) | all < 1e-9 |
| P-10 | post-peak paired contrasts SD 0.80 / 1.00 (Holm within P-10) | 3.9e-9 / 1.1e-11 |
| S-1 | per-subject vs global macro-F1, active-only, SVM / RF | 3.5e-11 / 2.5e-10 |

S-2's comparison is exact (no test). The family is left for a single recompute pass once every experiment lands.

---

## Standing constraints - confirmation

- No file in `01_Thesis/` edited.
- Section 4.17 and the unified BH family not touched; new tests listed above for a later single recompute.
- Deep model of record (`resnet_se` + channel dropout, 0.840) not changed.
- P-4 not started.
- Every run wrote a new `results_*` folder; none overwritten (B4.1 backed up originals before regen).
- Every run logged its command line (`run_config.json` via B1 section 1.5, plus `run_*.sh` per item).

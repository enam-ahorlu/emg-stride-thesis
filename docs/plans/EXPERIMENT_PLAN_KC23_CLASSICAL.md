# Plan: kill-critic 23 September, classical and verification stages (KC-C1 to KC-C7)

**Phase 0/1 status header (23 September 2026, this session; plan text below unedited):**
- **KC-C7:** done. `KC23_VERIFICATIONS.md`. V1 computed, V2/V3/V4/V6/V7 confirmed/determined. V5 does not exist in this plan (numbering gap, flagged not invented).
- **KC-C1:** done. Overall letter **E** (ESCALATE). `results_kc23_c1_nested_selection/C1_VERDICT.md`. Optimism is ~0 pt (even slightly negative) at every level; the escalation fires on "published configuration chosen in fewer than 20 of 40 folds" -- the nested procedure always picks a stacking combiner over the published soft vote, not because 85.8% is inflated.
- **KC-C2/C3/C5 code changes (Section 3 items 4/C-a/C-b/C-d) and the S2 adapter (deployment plan):** code changes complete, inertness proven (see `KC23_PHASE1_REPORT.md`). The stages themselves (rungs/grids/schemes actually run to a letter) are QUEUED, not executed, per this session's scope.

**Status:** ready to run. Written 23 September 2026 from `04_Reviews_and_QA/KILL_CRITIC_2026-09-23.md`.
**Execution:** local, on Enam's machine, through Claude Code. Dispatcher: `RUN_ORDER_KC23.md`. Master checklist: `KC23_TRACKLIST.md` (project root).
**Cost:** CPU only. About 45 to 60 CPU hours in total, most of it KC-C3. Runs alongside the GPU queue of `EXPERIMENT_PLAN_KC23_DEEP.md`, with worker counts capped (Section 0.5).
**Owner decision points:** listed per stage. Do not resolve any of them yourself.

Each stage closes one kill-critic item:

| Stage | Kill-critic item | Question |
|---|---|---|
| KC-C1 | M4 | How much of 85.8% is selection on the reporting folds? |
| KC-C2 | M5, M6 | Does the whitening penalty survive a properly scaled regularizer, and is the class-covariance mechanism real? |
| KC-C3 | M7 | Does the 6.3-point deep lead survive fully tuned classical baselines, and does Finding A hold on new families? |
| KC-C4 | M7 | Does "normalization matters more than features" hold on established richer feature sets? |
| KC-C5 | M8 | How much of the 3.7 to 6.1-point "leak" is overlap, how much autocorrelation, how much within-session drift? |
| KC-C6 | M4 | Does the alignment ladder, over-alignment included, replicate on ENABL3S? |
| KC-C7 | several | Cheap verifications that settle text items without training anything |

---

## 0. Read this first

1. **The interpreter is `06_Code/.venv/Scripts/python.exe`.** Every `jobs_*.txt` is stale.
2. **Do not edit any file in `01_Thesis/`.** Report numbers. Wording is handled in the text blocks of `KC23_TRACKLIST.md`.
3. **Do not touch the v9 statistical family** (`recompute_unified_fdr_v9.py` and its outputs). Every new test is recorded with its raw p-value and folded in at KC-F1 (`RUN_ORDER_KC23.md`).
4. **Every run writes to a NEW `results_kc23_*` directory**, logs its exact command line, and writes `run_config.json` (the B1 dump). Never overwrite a published directory.
5. **CPU budget while the GPU queue runs:** cap `--n-jobs` and `--rf-n-jobs` so that two logical cores stay free for the GPU job's data loading. Record the cap used.
6. **Any code change to a shared script is additive and proven inert first.** The house rule is byte-identical outputs on the existing paths, asserted, before any new arm runs. For classical scripts this is exact: SVM, RF and LDA are deterministic at fixed seed, so a reproduction must match the published per-subject F1 and best parameters digit for digit. If an inertness check fails, **stop**: every prior result becomes incomparable.
7. **Report the pre-registered outcome letter first**, then the gates, then the numbers. A null is a result.
8. **Decision rules live in code.** Each stage's stats script encodes its outcome grid and prints the letter, following the `window_ablation_stats.py` precedent, so the rule cannot drift after the numbers are seen.
9. **Escalation protocol.** If a stage lands on an outcome marked ESCALATE, write `06_Code/KC23_HALT.md` with the stage, the letter, the numbers and the affected thesis passages. Finish any other running stage, but start no dependent stage.

---

## KC-C1. Nested selection audit (M4). CPU, minutes.

### C1.1 Problem

The model of record, its augmentation and the ensemble (the best of 24 configurations, Table A.9) were chosen on aggregate LOSO F1 over the same 40 subjects that report them (Section 3.4.1). So 85.8% is a maximum over configurations, not a held-out estimate.

### C1.2 Design: selection nested inside the outer loop, over the existing predictions

For each outer subject *s*:

1. Among the candidate configurations, choose the one with the highest mean per-subject F1 over the **other 39** subjects.
2. Record *s*'s own F1 under that chosen configuration.

The nested estimate is the mean over *s*. The **optimism** is the published maximum minus the nested estimate. Also record how often the published configuration is chosen, out of 40.

Run it at three levels:

- **Level E:** the 24 ensemble configurations of Table A.9.
- **Level D:** deep single-model choice, over every deep candidate that competed for model of record on SIAT-LLMD. That means ResNet-SE with none, gaussian, timemask, combined and chandrop, the residual variants, and SimpleEMGCNN variants. Build the candidate list from `RUN_MANIFEST.csv` plus the plan files, and list it in the report. Where a candidate's backbone or augmentation is `UNKNOWN` in the manifest, exclude it and say so.
- **Level J:** joint. Nest the deep choice first, then the ensemble choice built on it.

Report a subject-bootstrap 95% interval on the optimism (10,000 resamples of the outer subjects, rerunning the selection in each resample).

**Recorded caveat, not a defect to fix.** The other folds' models were trained on data that includes *s*, so this nests the selection but not the training. Fully nested retraining would need 40 by 39 trainings per candidate and is out of scope. State this in the report in one sentence.

### C1.3 Outcomes

| Letter | Condition | Reading |
|---|---|---|
| **N** | Optimism < 0.3 pt at Level J, and the published configuration chosen in at least 35 of 40 folds | The headline stands; the nested figure goes beside it in one clause |
| **O** | Optimism 0.3 to 1.0 pt | Report the nested figure alongside 85.8% wherever the headline is stated |
| **E** | Optimism ≥ 1.0 pt, or the published configuration chosen in fewer than 20 of 40 folds | **ESCALATE.** Headline framing is Enam's decision |

**Output:** `results_kc23_c1_nested_selection/` with `nested_selection_{E,D,J}.csv`, `optimism_bootstrap.csv` and `C1_VERDICT.md`. Script: `kc23_c1_nested_selection.py`.

---

## KC-C2. Whitening under principled regularizers, plus an oracle mechanism test (M5, M6). CPU, about 8 hours.

### C2.1 Problem

Appendix B.6 says the published whitening rung regularizes with a fixed ridge on the raw feature scale, and that ridge exceeds the variance of 18 of the 72 features. `check_rung4_robustness.py` re-computed only the geometry (silhouette and probes) under better regularizers. The SVM LOSO F1 behind "10.1 points worse, d = -2.34", "12.1 points at 400 ms" and O3's answer all come from the mis-scaled operator.

Separately, the mechanism claim (whitening removes between-class covariance because it is computed over pooled classes) has never been tested directly.

### C2.2 Code change, additive

Add new rung IDs to `run_alignment_ladder_loso.py`, reusing `check_rung4_robustness.whiten_recolor` and the helpers in `analyze_between_subject_variance.py`. Do not reimplement anything that exists.

| Rung | Operator | Deployable? |
|---|---|---|
| `4b` | Global z first, then per-subject whiten and recolor, ridge λ = 1 (the CORAL analogue) | yes |
| `4c` | Pre-standardized, ridge λ = trace(C_s)/p per matrix | yes |
| `4d` | Pre-standardized, ridge λ = 0.1 × trace(C_s)/p | yes |
| `4lw` | Pre-standardized, Ledoit-Wolf shrinkage covariance (`sklearn.covariance.LedoitWolf`) | yes |
| `4o` | **Oracle, diagnostic only.** Per-subject whitening by that subject's pooled *within-class* covariance, which needs the subject's labels, the held-out subject's included. Recolor as in `4c`. Never deployable. Every output row carries `oracle=True`. | no |

**Path trap, from the window plan.** `analyze_between_subject_variance.py` hardcodes the 250 ms feature paths as module constants (lines 42 to 43). Add an environment-variable override that falls back to the literal. Then prove the patch inert by re-running rung 3 and rung 4 at 250 ms for subjects 1, 2 and 3, and matching the published per-subject F1 exactly.

**Inertness gate.** Rungs 0 to 4 must reproduce the published per-subject SVM F1 digit for digit on those three subjects, and the published `alignment_ladder_full.csv` means must be unchanged. Fail means stop.

### C2.3 Runs

At 250 ms and at 400 ms, run SVM nested LOSO on every new rung, at the published SVM grid so the result is comparable with Table 4.7. For each rung, also compute the full geometry row: MMD removed, W1 removed, linear, forest and MLP subject probes (class-pooled and within-movement with size-matched controls), and class silhouette pooled and within-subject. Use the published metric code.

### C2.4 Primary endpoints and outcomes

**Endpoint 1:** F1(rung 3) minus F1(each deployable whitening variant), paired over 40 subjects, with Wilcoxon, dz, BCa and subjects counted. The pre-registered headline variant for any rewritten Table 4.7 is **`4lw`**, chosen before the runs because it is the standard principled estimator and has no tunable constant.

| Letter | Condition on `4lw` (and reported for 4b to 4d) | Reading |
|---|---|---|
| **W1** | Penalty ≥ 5 pts, significant | The finding stands with a corrected size. Table 4.7 reports `4lw`, and the published rung moves to Appendix B.6 |
| **W2** | Penalty 1 to 5 pts, significant | The ordering claim holds, the size claim is revised; Section 4.2.4, O3 and Section 4.6 wording changes |
| **W3** | Penalty < 1 pt or not significant | **ESCALATE.** Over-alignment on the alignment axis becomes a regularization artifact. Abstract, O3, Sections 4.2.4, 4.6 and 5.5 are affected |

**Endpoint 2, the mechanism test:** F1(`4o`) against F1(rung 3), and against F1(`4lw`).

| Letter | Condition | Reading |
|---|---|---|
| **M1** | `4o` ≥ F1(rung 3) − 1 pt, and `4o` − `4lw` ≥ 2 pts | Mechanism supported: the damage comes from removing class-pooled covariance |
| **M2** | `4o` falls about as far as `4lw` (difference < 1 pt) | Mechanism unsupported: the damage is estimation or something else. Report; the Section 4.2.4 mechanism paragraph changes (Enam's text block) |
| **M3** | In between | Partial support; report the numbers |

Also report the subject probe under `4o`, because an oracle that keeps class structure and still removes subject identity would be the cleanest possible illustration for Section 4.6.

**Output:** `results_kc23_c2_whitening_w250/`, `results_kc23_c2_whitening_w400/` and `C2_VERDICT.md`. Stats script: `kc23_c2_whitening_stats.py`.

---

## KC-C3. Classical tuning parity and two new families (M7). CPU, about 30 to 40 hours.

### C3.1 Problem

The SVM's RBF width is never tuned (it stays at `scale`), and C sits at the lower edge of {1, 5, 10} in all 40 folds (Appendix B.2). The RF grid is two by two. There are no boosted trees and no nearest neighbours, which Section 5.3 lists as a limitation. The deep side meanwhile explored four augmentations, five architectures and two dose sweeps.

The 6.3-point lead of ResNet-SE+CD over the SVM (Section 4.3.1), and the "every model tried" scope of Finding A, both rest on that asymmetry.

### C3.2 Code change, additive

Extend `train_classical_loso.py` with:

- `--grid {default,extended}`. `default` must be byte-identical to today.
- `--search {grid,random}` with `--n-iter`.
- Two new model keys, `HGB` (`HistGradientBoostingClassifier`) and `KNN`.
- `--save-proba` for every model (already present for SVM).

Balanced sample weights where the estimator supports them. kNN does not, so record that.

**Inertness gate:** `--grid default` reproduces the published SVM and RF per-subject F1 and best parameters exactly on subjects 1, 2 and 3, under both normalizations.

### C3.3 Search spaces, fixed now

| Model | Search | Space |
|---|---|---|
| SVM-X | full grid, nested GroupKFold(5) | C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}; gamma ∈ {0.01, 0.1, 0.3, 1, 3, 10} × `scale` value (48 cells) |
| RF-X | random, 30 draws, seeded | n_estimators {200, 500, 1000}; max_depth {None, 10, 20, 40}; min_samples_leaf {1, 2, 5, 10}; max_features {sqrt, log2, 0.25, 0.5} |
| HGB | random, 30 draws, seeded | learning_rate {0.03, 0.1, 0.3}; max_leaf_nodes {15, 31, 63}; min_samples_leaf {20, 50, 100}; l2_regularization {0, 1, 10}; max_iter 500 with early stopping on an internal 10% split |
| KNN | full grid | k {5, 11, 21, 41, 81}; weights {uniform, distance} |

**Edge rule, pre-registered.** If the selected SVM C or gamma sits on a grid edge in more than 10 of 40 folds, extend that axis by two steps beyond the edge, **once**, and rerun. Report both runs.

### C3.4 Runs

| Arm | Normalization | Notes |
|---|---|---|
| SVM-X | per-subject and global | `--save-proba` on per-subject |
| RF-X | per-subject; global if time allows (optional, marked) | `--save-proba` on per-subject |
| HGB | per-subject and global | `--save-proba` on per-subject |
| KNN | per-subject and global | |

Then recompute the soft vote with ResNet-SE+CD (published probabilities, `results_cnn_aug_resnet_se_chandrop_proba`) using the SVM-X probabilities, and separately using the best new classical member. Use `ensemble_v2_combine.py`, whose row-alignment assertion must hold.

### C3.5 Endpoints and outcomes

**Endpoint 1, the deep lead:** ResNet-SE+CD minus the best tuned classical model (per-subject normalization), paired.

| Letter | Condition | Reading |
|---|---|---|
| **P1** | Best tuned classical within 1 pt of the published SVM (77.7%) | The 6.3-point lead stands; add one sentence and an appendix table |
| **P2** | Best tuned classical gains 1 to 3 pts | The lead narrows; report the revised lead as a sensitivity. **ESCALATE the framing choice:** keep the locked SVM (recommended, because the locked pipeline already ran on ENABL3S) or re-baseline |
| **P3** | Best tuned classical within 1 pt of ResNet-SE+CD, or above it | **ESCALATE.** Finding C's "strongest single model" changes |

**Endpoint 2, Finding A on the new families:** the per-subject minus global gain for HGB and KNN (and RF-X, SVM-X).

| Letter | Condition | Reading |
|---|---|---|
| **N1** | Positive and significant on every family | "Every model tried" extends to six classical families |
| **N2** | Any family with a gain ≤ 0 | **ESCALATE.** Finding A's scope wording changes |

**Endpoint 3, the ensemble:** the soft vote with SVM-X probabilities against 85.8%.

| Letter | Condition | Reading |
|---|---|---|
| **E1** | \|Δ\| < 0.5 pt | Report |
| **E2** | \|Δ\| ≥ 0.5 pt | **ESCALATE**, because it touches the headline |

**Output:** `results_kc23_c3_{svmx,rfx,hgb,knn}_{persubj,global}/`, `results_kc23_c3_ensemble/` and `C3_VERDICT.md`, which includes an edge-hit table and fit-time totals.

---

## KC-C4. Richer established feature sets (M7). CPU, about 6 to 10 hours.

### C4.1 Problem

The claim that "normalization matters more than the choice of features" (Section 4.2.1, Appendix A.2) rests on four nearly identical sets.

### C4.2 Sets

Both are computed on the filtered signal (`X_raw`) of the same 250 ms windows. Use the 1920 Hz rate for anything frequency-dependent, following the `freq_fs1920_*` precedent.

- **TDPSD-54:** the time-domain power-spectral descriptors of Al-Timemy, Khushaba and colleagues, six per channel. They were designed for robustness to force and limb-position variation, which makes them the most relevant established robust set. Implement from the original equations. Record the paper and equation numbers in the script header, and unit-test against a synthetic sinusoid with known moments.
- **Rich-126:** Freq-72, plus fourth-order autoregressive coefficients per channel (36), plus Hjorth mobility and complexity per channel (18).

New script: `kc23_c4_extract_rich.py`, writing to `features_out/` under new names. **Never overwrite an existing features file.**

### C4.3 Runs

SVM-X (the KC-C3 grid) and LDA, under both normalizations, nested LOSO, on each new set.

### C4.4 Outcomes

| Letter | Condition | Reading |
|---|---|---|
| **F-A** | Both sets within 1 pt of Freq-72 under per-subject normalization, and the normalization gain present on both | The claim extends to established richer sets |
| **F-B** | One set gains 1 to 2 pts | Report; the text notes it |
| **F-C** | One set gains > 2 pts | **ESCALATE.** It could change the classical member |

**Output:** `results_kc23_c4_*` and `C4_VERDICT.md`.

---

## KC-C5. Decomposing the subject-dependent "leak" (M8). CPU (classical) plus about 1 hour GPU (SimpleEMGCNN).

### C5.1 Problem

The movement-blocked split tests on contiguous later or earlier stretches of each recording. The 3.7 to 6.1-point drop therefore mixes three things: overlap leakage; temporal autocorrelation beyond the overlap (one gait cycle spans several windows); and extrapolation across time within the session (drift). The one-window guard band removes only the first.

Also, the classical "subject-dependent" figures pool all 40 subjects in one cross-validation, so they are subject-inclusive, not per-subject.

### C5.2 Code change, additive

Extend `b8_movement_blocked_sd.py` with:

- `--scheme {pooled_random, pooled_random_nonoverlap, blocked, interleaved}`
- `--guard-windows G`
- `--n-chunks M` (for interleaved)
- `--cv-unit {pooled, per_subject}`

The existing default must reproduce `results_b8_sd` exactly (inertness gate).

- **`pooled_random_nonoverlap`:** within each subject-by-movement recording, keep every second window in time order, so no two retained windows overlap. Then split at random.
- **`interleaved`:** cut each movement's recording into M = 20 contiguous chunks and assign chunk *i* to fold *i* mod 5, with guard G. This keeps the blocking but removes most of the temporal extrapolation.

### C5.3 Arms

SVM, RF and LDA, Freq-72, 250 ms, on SIAT-LLMD and on ENABL3S (`--time-units samples`, as in the published ENABL3S control).

| Arm | Scheme | Guard |
|---|---|---|
| P50 | pooled_random (published) | none |
| P0 | pooled_random_nonoverlap | none |
| B-g | blocked, 5 chunks | g ∈ {1, 2, 4, 8, 16} windows |
| I-g | interleaved, 20 chunks | g ∈ {1, 4, 16} |
| W-B1 | blocked, g = 1, `--cv-unit per_subject` | naming control: a true per-subject model |

**SimpleEMGCNN** (already per-subject, `b8_cnn_sd.py`): P50, P0, B at the plateau guard, and I at the plateau guard.

### C5.4 Decomposition and outcomes

- **Plateau guard g\*:** the smallest g at which B-g changes by less than 0.5 pt from B-(2g). If there is no plateau by 16, that is outcome L3.
- Δ_overlap = P50 − P0
- Δ_autocorr = P0 − I-g\*
- Δ_drift = I-g\* − B-g\*

| Letter | Condition | Reading |
|---|---|---|
| **L1** | Δ_overlap ≥ 60% of P50 − B-g\* | "Overlap leak" stands as named, with its size |
| **L2** | Δ_drift ≥ 40% of the total | Section 4.1.2 reports the difference as a split-protocol effect with components; the word "leak" is limited to Δ_overlap |
| **L3** | No plateau by g = 16, or g\* ≠ 1 and B-g\* differs from the published blocked SD by > 1 pt | **ESCALATE.** Table 4.2 and the 16-to-22 gap in the abstract change |

Also report W-B1 against the pooled blocked figure, which settles the "subject-inclusive" naming.

**Output:** `results_kc23_c5_leak_{siat,enabl3s}/` and `C5_VERDICT.md`.

---

## KC-C6. The alignment ladder on ENABL3S (M4). CPU, about 1 hour.

Run the full ladder (rungs 0 to 4, plus `4lw` and `4o` from KC-C2) on the ENABL3S Freq features, using the KC-C2 environment override. Record SVM LOSO F1 (n = 10), the geometry measures and the probes. Chance for the subject probe is 1 in 10.

| Letter | Condition | Reading |
|---|---|---|
| **R1** | z-scoring beats `4lw` in F1 for at least 7 of 10 subjects, and centering does most of the linear-probe work | Over-alignment replicates in direction; the abstract may list it |
| **R2** | Otherwise | Report non-replication. The abstract does not list it |

With n = 10, no significance claim unless it survives the KC-F1 family.

**Output:** `results_kc23_c6_ladder_enabl3s/` and `C6_VERDICT.md`.

---

## KC-C7. Verifications, no training. CPU, under 1 hour.

| ID | Check | Deliverable |
|---|---|---|
| V1 | **Table 4.11 route.** Compute the transductive SVM F1 through the probability route (argmax of the Platt probabilities in `results_ensemble_v2/proba`), so the SVM's causal cost can be stated like for like (causal 73.2 against transductive probability-route). | one number with its source, plus the like-for-like Δ |
| V2 | **Contiguous labels.** Confirm from `run_within_subject_baseline.py` (lines 71 to 76 and 102) that "first" means the first N windows of each movement's own recording in time order, which is gait initiation for the locomotion classes. Record line numbers. | a one-paragraph note |
| V3 | **ENABL3S validation subjects.** Confirm `choose_val_subjects` gives round(0.15 × 9) = 1 validation subject on ENABL3S for every deep ENABL3S run, from code and logs. | a note with the run directories |
| V4 | **Effect size.** Confirm every stats script's "d" is the paired dz (mean difference over SD of differences). List any exception. | a table |
| V6 | **The 81.8% K = 0 run.** Determine the epochs and patience used for `results_cnn_calibration_chandrop_resnet_se_multidraw` from logs, file timestamps against command history, or the `_run_logs` folder. If undeterminable, write `UNKNOWN`. This settles whether the kill critic's 2.1-point row is run noise or a configuration difference. | a note |
| V7 | **The AdaBN pre-pass (0.787) against the three-run global mean (0.772).** Confirm both used epochs 40, patience 7, batch 512, the same arch and the same channel-dropout rate. | a note |

**Output:** `KC23_VERIFICATIONS.md` in `06_Code/`.

---

## Outputs, all stages

A `C*_VERDICT.md` per stage, each opening with its outcome letter. Add a status header to the top of this file recording each stage's outcome, leaving the plan below unedited. Every new paired test goes into `kc23_new_tests.csv` (columns: stage, contrast, n, statistic, raw_p, dz, BCa_lo, BCa_hi, subjects_improved, source_dir) for KC-F1.

## What to report back

1. The outcome letter of every stage, then its gates, then its numbers.
2. Every ESCALATE and what it touches.
3. The inertness evidence for every code change.
4. Confirmation that no thesis file was edited and the v9 family was not touched.

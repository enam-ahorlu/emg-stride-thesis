# KC23 pre-registration conformance table

Written 26 September 2026. Governing rule (Enam, 26 September): the pre-registration is the 23 September plan text
(`EXPERIMENT_PLAN_KC23_{DEEP,CLASSICAL,DEPLOYMENT}.md`), not commit `0ba3575`. "Thresholds unchanged since 0ba3575" only shows
the code agrees with itself. Code conforms to the plan text, and every conformance change is committed before its stage's
data exists. Stages that already had data and a verdict are not rewritten: the non-conformance is reported here with both
readings, and the plan governs.

Column key. **Status**: `FIXED` = code corrected to the plan before the stage's verdict or data existed; `HAD-DATA` = the
stage already had data and a verdict, code left as it was, both readings given; `OK` = audited, conformant; `RULED` = the plan
left the operationalisation open and Enam ruled on it on 26 September ("Enam ruling, 26 Sep" column; code aligned where the
ruling changed it); `OPEN` = not yet built or decided. **Data existed?**
answers, for each row, whether that stage's data (results directories the gate reads) existed at the moment of the fix.

## Summary counts

83 clauses audited (rows below; a row is one plan clause or one input the gate needs, for C2 to C6, D1 to D6 and S1 to S3, plus
C1, the D1 reproduction and headline gates). 80 were found on 26 September; three more rows record the design corrections and
builds of the same day (C5-10, D6-8, D6-9).

| Result | Rows |
|---|---|
| Conformant (OK) | 12 |
| Conformant under an operationalisation the plan left open, ruled on by Enam on 26 September (RULED) | 7 |
| **Non-conformant** | **64** |
| of which FIXED before their stage's verdict or data existed | 59 |
| of which HAD-DATA (verdict already existed; reported with both readings, not changed) | 5 |
| Cross-cutting item still open (row N-1 below) | 1 |

The 5 HAD-DATA rows are C1-1, C1-2, C1-3, S1-2 and S2-1 (S2-1 is Enam's instruction to reorder the S2 reading, applied to
the text only; `s2_measures.csv` is byte-identical). The 7 RULED rows are S1-3, D5-2, S2-2, C4-4, C5-9, D1-2 and D1-7. The four
plan items that were OPEN in the first version (C3-7 edge-rule rerun generator, D1-5 MixedLM secondary model, D6-7 ADV-C, CDAN,
mechanism, secondary contrasts and divergence retry, S3-4 SVM-X and HGB cells) are now FIXED. Four FLAG rows changed code on
Enam's ruling (C5-7, D1-3, D6-4, D6-5); the flagged operationalisations Enam confirmed without a code change are RULED. Of the 59
FIXED rows, five families of results directories already existed on disk (SVM-X per-subject and global, the 400 ms S2b features
and SVM, the S3 LDA runs); none was read by a gate or produced a verdict, and each was set aside with an `INVALID.md` or
`SUPERSEDED.md` note rather than reused.

## Stages that already had data and a verdict

### KC-C1 (verdict E, produced before this pass)

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| C1-1 | "The optimism is the published maximum minus the nested estimate." | `published_f1` is the headline soft vote 0.8580, which is NOT the row maximum of the 24 configurations (`SVM+RESNET_SE [stacking]` 0.8604). Optimism reported as -0.017 pt | unchanged | Yes, verdict E exists | HAD-DATA | Confirmed. Both readings recorded: the nested estimate is 85.82%; optimism is -0.02 pt against the published soft-vote headline 85.80% and +0.22 pt against the table maximum 86.04% (stacking) |
| C1-2 | N is "optimism < 0.3 pt **at Level J**"; the letter is read at Level J | Overall letter = worst of E, D, J (`max` over the three levels) | unchanged | Yes | HAD-DATA | Confirmed (E under both readings) |
| C1-3 | The grid has no cell for optimism < 0.3 with the published configuration chosen in 20 to 34 folds | `verdict_letter` falls through to "O" | unchanged | Yes | HAD-DATA | Confirmed; not triggered |
| C1-4 | Levels E, D, J, bootstrap of 10,000 resamples, caveat sentence, candidate list from the manifest | As planned | as planned | Yes | OK | - |

Both readings of C1. Plan rule (Level J, optimism against the published maximum 0.8604): optimism = 0.8604 - 0.8582 = +0.22 pt
(under 0.3), the joint configuration chosen in 0 of 40 folds, which is under 20, so the E clause fires: **E**. Code rule (worst
of three levels, optimism against the headline 0.8580): optimism -0.017 pt, Level E chosen 0 of 40, Level J 0 of 40, Level D
40 of 40 (N): **E**. The letter is E under both readings. What changes is one number in the verdict text: the optimism is
+0.22 pt against the true maximum, not -0.02 pt, and the escalation rests entirely on the "chosen in fewer than 20 of 40"
clause (the nested procedure always picks a stacking combiner, never the published soft vote), not on optimism size.
C1-3 is not triggered by the data.

### KC-D1 reproduction gate (D1.4; verdict PASS)

| ID | Plan clause | Code | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|
| D1R-1 | R2 within 1.5 pt of 0.8395; R1 within 1.5 pt of 0.782 | `reproduction_gate`: `PUBLISHED_R2 = 0.8395`, `PUBLISHED_R1 = 0.782`, `REPRO_TOL = 0.015` | Yes | OK | - |
| D1R-2 | R10 pre-adaptation within 1.5 pt of 0.787 or 0.772 | `PUBLISHED_R10_PRE = (0.787, 0.772)`, either accepted; measured on `f1_pre_adabn` | Yes | OK | - |

Measured: R1 0.7706 (-1.14 pt), R2 0.8344 (-0.51 pt), R10 pre 0.7830 (-0.40 pt), all inside the band under both readings.
PASS stands.

### KC-S1 (verdict D-S, accepted as decision D-6b)

| ID | Plan clause | Code | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|
| S1-1 | D-S: supervised ahead by 1.0 pt or more, significant, on 25 or more of 40 subjects; D-T within +-1.0; D-L label-free ahead by 1.0 or more | `classify_d` encodes all three | Yes | OK | - |
| S1-2 | A supervised lead of 1.0 pt or more that is not significant, or on fewer than 25 subjects, has no letter | `classify_d` returns "D-T" for it | Yes | HAD-DATA | - |
| S1-3 | Reproduction gate: L0 at K = 25 exact for the SVM part, "within the KC-D1 run band" for the ensemble | Exact to 5e-5 for the SVM; ensemble tolerance a placeholder 0.015 "until KC-D1's own realization SD is measured" | unchanged | Yes | RULED | Confirmed for now. When the D1 realization SD is measured, re-evaluate the S1 check against that band and record the result (0.8141 against 0.8152 passes any plausible band) |

Both readings. Plan rule: the current D-S stands (best supervised arm S-ens1 +3.72 pt, p 3.95e-08, 36 of 40 subjects), and S1-2 is not
triggered. Reproduction: the ensemble differs from the published `balanced25` by 0.11 pt (0.8141 against 0.8152), inside 1.5 pt
and inside any plausible KC-D1 realization band, so PASS under both. The D-S outcome is unchanged.

### KC-D5 (E-R, E-N, E-R, E-N; decision D-6c)

| ID | Plan clause | Code | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|
| D5-1 | E-R: direction holds in the realization average with 7 or more of 10 subjects agreeing; E-N otherwise | `classify_er` | Yes | OK | - |
| D5-2 | "A finding's direction" is not tabulated for two findings | Directions encoded from the thesis on SIAT-LLMD, the two contested ones (gain jitter ahead of channel dropout; permutation reduction) both positive, the reading less favourable to the thesis (D-6c) | unchanged | Yes | RULED | Decision D-6c (25 Sep), unchanged |
| D5-3 | ResNet-SE+CD against the SVM (65.7) as a number with its across-seed SD | 0.6487 +- 0.0082 against 0.657 | Yes | OK | - |

No reading changes.

### KC-S2 (descriptive, verdict existed)

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| S2-1 | S2.5 Reading 1: the vote "cuts steady-state error and adds under 250 ms median delay" | Taken from the causal-100 condition, the one that collapsed | Taken from the transductive and balanced25 conditions (soft vote); causal-100 reported as a separate finding (a buffer from the start of a real continuous session does not work; supports the scripted commissioning step, D-6b) | Yes, data unchanged; `s2_measures.csv` byte-identical, verdict text regenerated | HAD-DATA (text reordered by Enam's instruction) | - |
| S2-2 | F0 letters F-OK (5 or more of each type for 8 of 10 subjects), F-MARGINAL, F-X | `classify_f0`; F-X when any type is absent for every subject | unchanged | Yes | RULED | Confirmed |
| S2-3 | S2b: 250 against 400 ms, accuracy and decision delay, Reading 2 (400 ms adds 100 ms or more for under 2 pt) | Not computed; the two S2b rows produced LOSO F1 with no window times | `kc23_s2b_window_trade.py`; rows replaced by one 400 ms predictions row plus the trade row | No 400 ms predictions existed | FIXED | - |
| S2-4 | S2b "the locked ... SVM at 400 ms": the same feature set at a new window length | 400 ms features extracted with `--use env`, wavelet on, fs 2000 (63 features per window) against the published `--use raw --fs 1000 --no-wavelet` (56) | Row corrected; the old directories set aside with `INVALID.md` | Wrong-config features and an SVM run existed, never read by any gate | FIXED | - |

Reading 1 does not flip: the vote cuts steady-state error by 4.31 pt (transductive) and 4.03 pt (balanced25) and adds a paired
median of 125 ms in both, so the smoothing statement holds on the lead conditions as it did on causal-100.

## Stages whose data did not exist (code corrected exactly to the plan)

### KC-C2

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| C2-1 | W1/W2/W3, M1/M2/M3 thresholds | `classify_w`, `classify_m` | unchanged | No verdict; w250 ladder in progress, not read | OK | - |
| C2-2 | Endpoint 1 is "F1(rung 3) minus F1(each deployable whitening variant)", paired, Wilcoxon, dz, BCa | Only 4lw | 4b, 4c, 4d, 4lw in `C2_variant_penalties.csv`; the letter still conditions on 4lw | No verdict | FIXED | - |
| C2-3 | The subject probe under 4o is reported | Never computed | `c2_geometry_w400` row (`kc23_c6_geometry.py`) writes `ladder_geometry.csv`; the verdict quotes the 4o probe, or says it is not reported | No | FIXED | - |
| C2-4 | Window label of the w400 verdict | Hard-wired "w250" | Taken from the directory name | No | FIXED | - |

### KC-C3

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| C3-1 | SVM-X grid: C in {0.01 ... 30}, gamma in {0.01, 0.1, 0.3, 1, 3, 10} times the fitted `scale` value (48 cells) | Six ABSOLUTE gammas plus 'scale' (56 cells); for 72 standardized features the plan's multiples span 1.4e-4 to 0.14, the absolute ones 0.01 to 10 | `svm_extended_grid` builds multiples of `scale`; the chosen multiplier goes to `svm_extended_gamma.csv`; default path proven unchanged against git HEAD | SVM-X runs existed on the wrong grid; no verdict. Set aside as `__ABSGAMMA_INVALID` | FIXED | - |
| C3-2 | The tuning run tunes | The SVM-X per-subject row also carried `--save-proba`, train_classical_loso.py's cheap refit path (no search, params from the published run): all 40 folds used C=1, gamma='scale', mean F1 0.7767, the published SVM | Tuning row has no `--save-proba`; a separate `..._proba` refit row uses the SELECTED params (`--reuse-params-dir`), for SVM, RF and HGB | Same directory | FIXED | - |
| C3-3 | Ensemble: the tuned SVM with the published members | Merged `results_ensemble_v2/proba` (un-augmented ResNet-SE, soft 0.8134) | `proba_aug_chandrop` (the 0.8579 headline's probabilities) | No | FIXED | - |
| C3-4 | P1/P2/P3: P1 within 1 pt of 77.7; P2 gains 1 to 3; P3 within 1 pt of ResNet-SE+CD or above | Any other case fell into P3 (gain of 3 or more with the lead still over 1) or P1 (more than 1 pt BELOW the SVM) | Those are "P-OUT", exit 10 | No | FIXED | - |
| C3-5 | N1: positive and significant on every family; N2: any family with a gain of 0 or less | Positive but not all significant fell into N2 | "N-OUT", exit 10 | No | FIXED | - |
| C3-6 | `C3_VERDICT.md` includes an edge-hit table and fit-time totals | Neither | Both, plus `C3_edge_hits.csv`, `C3_fit_times.csv` | No | FIXED | - |
| C3-7 | Edge rule: if the selected C or gamma is on the grid edge in more than 10 of 40 folds, extend that axis by two steps once and rerun, report both | Not implemented | The gate detects and states it and reports BOTH runs when the rerun exists, recomputing the letters with the rerun in place of the base run (either reading escalating escalates). `kc23_c3_edge_job_gen.py` appends the rerun rows (C two steps: 100, 300 up or 0.003, 0.001 down; gamma multiplier: 30, 100 up or 0.003, 0.001 down; never re-extends an extended axis; idempotent) | No | FIXED | Build ordered (item B) |
| C3-8 | The ensemble "with SVM-X probabilities"; best new classical member | Only the SVM member | Soft vote with the best of SVM-X, RF-X, HGB (by per-subject mean F1) reported beside Endpoint 3 | No | FIXED | - |
| C3-9 | RF-X global is optional, "marked" | Required | Optional, its absence stated | No | FIXED | - |
| C3-10 | E1/E2 | `classify_e` | unchanged | No | OK | - |
| C3-11 | Verdict must be readable by the queue | `**Outcome(s): ['P1', ...]**` (a list repr, no letter for the queue's regex) | `**Outcomes: P1, N1, E1**` | No | FIXED | - |

### KC-C4

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| C4-1 | "SVM-X (the KC-C3 grid) and LDA" on each set | Default-grid SVM rows; comparator the published default-grid SVM (`results_loso_freq_persubj`) | Rows run `--grid extended --search grid`; comparator SVM-X on Freq-72 (`results_kc23_c3_svm_per_subject`) | No | FIXED | - |
| C4-2 | LDA on each set | LDA rows existed but were not read by the gate | LDA reading against the published Freq-72 LDA, reported beside the letter; either reading escalating escalates | No | FIXED | - |
| C4-3 | F-A, F-B, F-C | A set that loses more than 1 pt, or is within 1 pt with no normalization gain, fell into F-B | "F-OUT", exit 10 | No | FIXED | - |
| C4-4 | "the normalization gain present on both" | Gain simply positive | Positive and Holm-significant across the two sets | No | RULED | Confirmed |

### KC-C5

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| C5-1 | Arms "SVM, RF and LDA" | SVM only | Decomposition and letter per model; exit escalates if any model's letter escalates | No | FIXED | - |
| C5-2 | P50, P0, B-g, I-g pooled; W-B1 the one `--cv-unit per_subject` arm | Every arm `per_subject`, so W-B1 was identical to B-1 | SUPERSEDED by C5-10 (the plan's premise was wrong): every arm is `per_subject`; W-B1 and the pooled arms are dropped | No | FIXED | See C5-10 |
| C5-3 | The decomposition needs every arm | A missing B-g or I-g file was silently dropped | Every arm required; a missing one exits 20 with no letter | No | FIXED | - |
| C5-4 | I runs at g in {1, 4, 16}; a plateau at g = 2 or 8 has no I-g* | Blanket "L3" | "L-OUT" (Delta_autocorr and Delta_drift undefined on the registered arms), exit 10 | No | FIXED | - |
| C5-5 | L1, L2, L3 | Nothing fired (or total <= 0) gave an empty list the queue could not read | "L-NONE" / "L-OUT", exit 10 | No | FIXED | - |
| C5-6 | "Also report W-B1 against the pooled blocked figure" | Not reported | Reported per model | No | FIXED | - |
| C5-7 | L3 clause "B-g* differs from the published blocked SD by more than 1 pt" | `results_b8_sd sd_new_blocked` (a PER-SUBJECT-model figure) against B-g* (pooled); no LDA row exists there | Per-subject B-g* against the published per-subject blocked SD (like for like); LDA and ENABL3S stated "not evaluated"; B-1 against the published figure reported as a reproduction check | No | FIXED | Ruling 4: the published Table 4.2 SD figures are per-subject models (`eval_sd` loops over subjects); thesis caption error is text item TA10 |
| C5-8 | Gate placement | On the W-B1 row, whose own output is a b8 csv, so a gate that wrote no letter was never noticed | `c5_<dataset>_verdict` light pseudo-row checked for a letter | No | FIXED | - |
| C5-9 | SimpleEMGCNN B and I "at the plateau guard" | `kc23_c5_cnn_job_gen.py` reads row 0 of `C5_decomposition.csv` | Row 0 is the SVM; that choice of g* for the CNN is stated | No | RULED | Confirmed: the SVM's per-subject g* is used for the CNN |
| C5-10 | DESIGN CORRECTION (Enam, pre-data): the plan premised that the published Table 4.2 classical SD figures pool all 40 subjects | The arms were pooled (P50, P0, B-g, I-g) with W-B1 the per-subject naming control | per_subject is the PRIMARY cv-unit for every arm on SIAT and ENABL3S. Arm list after the change: P50, P0, B-1, B-2, B-4, B-8, B-16, I-1, I-4, I-16 (10 arms per dataset, 20 in all, SVM, RF and LDA in each); W-B1 and the pooled arms are gone. The one pooled C5 output on disk (`results_kc23_c5_scheme_smoke3`, a unit smoke test) carries `SUPERSEDED.md` | No | FIXED | Ruling 4 |

### KC-C6

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| C6-1 | "Record SVM LOSO F1 (n = 10), the geometry measures and the probes" | Nothing wrote `ladder_geometry.csv`, which the gate reads | `kc23_c6_geometry.py` (MMD, W1, three probes pooled and within-movement, size-matched control, silhouette; published metric code imported unchanged; ENABL3S through the C2 environment override; rungs incl. 4lw and 4o) | No | FIXED | - |
| C6-2 | R1 needs "centering does most of the linear-probe work" | (probe) rung0 - rung1 over rung0 - rung3 | Fixed by Enam: (probe rung0 - rung1) / (probe rung0 - rung3) >= 0.5; `PROBE_COLUMN = subject_probe_linear` | No | FIXED | - |
| C6-3 | R1 z-scoring beats 4lw in at least 7 of 10 subjects; R2 otherwise | `classify_r` | unchanged | No | OK | - |

### KC-D1 (contrasts and headline; the reproduction gate is above)

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| D1-1 | 17 registered contrasts; ESTABLISHED / AMBIGUOUS / NOT ESTABLISHED; BH within the registered family | C10, C11, C13, C13b, C15, C16, C17 and the ensemble and global headline arms were not produced | `kc23_d1_aggregate.py` builds all of them; `kc23_d1_ensemble.py` builds the per-seed soft vote and stacking from the saved probabilities; each run's `run_config.json` is checked against its arm | Tier A seed-42 runs and some later seeds existed; no contrast had been computed from them | FIXED | - |
| D1-2 | "BH within the registered KC-D1 family" | Family unspecified in code | Fixed list of 17 tests: C1 to C14, C16c, C17a, C17b (nulls C15, C16a, C16b excluded and tested by TOST) | No verdict | RULED | Confirmed |
| D1-3 | "4 of 5 realizations (Tier A) or 3 of 3 (Tier B)" | `need_agree` 4 / 3; the published seed-42 run was used as the fifth realization where every arm had one | Tier A seeds 42, 7, 123, 1001 AND 2026 for every arm R1 to R12 (12 new GPU rows, queued after the current Tier A seeds, plus the 2026 ensemble row), so every contrast has 5 realizations from one code era and "4 of 5" applies uniformly; Tier B stays at 3. The published runs are a sensitivity analysis only, never the fifth realization: every letter, the escalation rule and the headline gate use the seeds alone and the with-published letters are reported beside them. D2 uses the same 5 seeds | No (seed 2026 had not run) | FIXED | Ruling 5 |
| D1-4 | Nulls: TOST +-1.0 pt on realization-averaged differences (C15, C16 plateau pairs) | Present | Applied to C15, C16a, C16b | No | OK | - |
| D1-5 | Secondary model `F1 ~ arm + (1 \| subject) + (1 \| seed)` via statsmodels MixedLM | Not computed | Fitted by `kc23_d1_mixedlm.py` in a SEPARATE environment, `06_Code/.venv_stats` (statsmodels 0.15.0, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, patsy 1.0.3, Python 3.14.0; frozen in `.venv_stats_requirements.txt`, git-ignored); `06_Code/.venv` stays frozen and has no statsmodels (tested). The replicate stats call it through a CSV round trip; a failure is stated in the verdict, never silent, and no letter depends on it | No | FIXED | Ruling C: do not install into `.venv`; separate `.venv_stats` used only for this model |
| D1-6 | Headline gate H1/H2 for R2, the per-seed ensemble, R12 (0.860) and "global" (0.772) | Ensemble and global arms absent | "global" = R10's `f1_pre_adabn` per seed; 400 ms = R12; ensemble = the per-seed soft vote | No | FIXED | - |
| D1-7 | Plan Section 0 rule 3: `run_config.json` for every arm | R10 runs (and R11 runs launched before 26 Sept) write none | R10 has no config check (its runner was not touched); identified by directory name; their exact commands are the first lines of `_run_logs/kc23/<job>.log` | No | RULED | Accepted; the queue log holds their exact commands |
| D1-8 | Gate placement | Contrast gate hung on `d1_r2_s1001`, before its 46 siblings existed | `d1_full_aggregate` pseudo-row after Tier B and the four ensemble rows | No | FIXED | - |

### KC-D2

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| D2-1 | Factors "computed per realization, with the across-realization mean and SD, plus the paired Wilcoxon on realization-averaged per-subject sums" | One pooled factor from the mean cost; no SD, no Wilcoxon; inputs `r1_occlusion.csv` etc. that nothing produced | `kc23_d2_aggregate.py` builds per-subject summed drops (D1's C17 definition, negatives kept, no clipping) for every arm and realization; per-realization factor = mean(R1 sum) / mean(arm sum), mean and SD; paired Wilcoxon | Instrumented runs existed for some seeds; no D2 quantity had been computed | FIXED | - |
| D2-2 | Three measures: zeroing occlusion, attenuation at 0.5, permutation (R = 5) | Attenuation factor never computed | All three | No | FIXED | - |
| D2-3 | O-R, O-T, O-M thresholds (3-fold and 2-fold; both below 1.5) | `classify_o` | unchanged | No | OK | - |
| D2-4 | Residual caveat "to record in the verdict" | Absent | In the verdict | No | FIXED | - |

### KC-D3

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| D3-1 | A1 to A4 (X2 <= R1 + 1; X3 and X4 < R3 - 1.5; X >= R3 - 1) | `classify_a` | unchanged | No | OK | - |
| D3-2 | Realization-averaged means for R1, R3, X1 to X4 from the same seeds | Input `d3_realization_means.csv` that nothing produced; the gate hung on `d3_x4_s123` | `kc23_d3_aggregate.py` (checks each run's config); `d3_stats` pseudo-row after all runs | Partial runs; nothing computed | FIXED | - |
| D3-3 | "Report every letter that fires" | List-repr verdict line | `**Outcomes: ...**` and a "none fired" statement | No | FIXED | - |

### KC-D4

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| D4-1 | (a) Subject probe falls with dose: Page trend, 40 folds as blocks, Holm across the two meters | Page trend over 3 REALIZATIONS as blocks | 40 folds as blocks (realization-averaged), Holm over subject probe and permutation reliance, subject-probe p used (fixed by Enam) | No | FIXED | - |
| D4-2 | (b) F1 peaks then falls | argmax vs the top dose, a raw 2 pt gap with no test | argmax of the realization-mean F1 is not the highest dose AND F1 at the top dose is below the peak by a paired Wilcoxon p < 0.05 | No | FIXED | - |
| D4-3 | (c) F1 tracks class silhouette | Spearman of the 5 realization-mean points > 0 | Per-fold Spearman across the 5 doses, one-sided sign test over 40 folds, positive, p < 0.05 | No | FIXED | - |
| D4-4 | (d) not tracking invariance | Spearman > 0.3 threshold | Same sign test on the NEGATED subject probe, not significantly positive | No | FIXED | - |
| D4-5 | T1 = a, b, c, d; T2 = not a; T3 = a and b but not c | Any other combination defaulted to T3 | Any other combination is "T-OUT (outside the pre-registered grid)", exit 10, never defaulted | No | FIXED | - |
| D4-6 | Gain-jitter boundary over SD {0.30, 0.40, 0.50, 0.80, 1.00} | argmax only | Argmax not at 1.00 AND paired Wilcoxon p < 0.05 (fixed by Enam) | No | FIXED | - |
| D4-7 | Inputs | `d4_dose_sweep.csv`, `d4_gainjitter_boundary.csv` that nothing produced | `kc23_d4_aggregate.py` (seeds 42, 7, 123; R14/R3/R16 plus the d4 rows; R1, R13, R2, R17, R15 as references); `d4_stats` pseudo-row | Partial runs | FIXED | - |

### KC-D6

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| D6-1 | The unseen-subject probe (D0.3 `embed_probes.csv`) is the invariance meter | The aggregator fed `class_probe_tgt_bacc`, a CLASS probe, as the "subject probe"; the D6 runners never wrote the D0.3 file | Runners take `--instrument` and `--probe-cap` (inert without the flag; proven); the aggregator reads `embed_probes.csv` | Stage 1 partly run; no gate had fired | FIXED | - |
| D6-2 | Page tests with 40 folds as blocks, Holm across the two meters, for the manipulation gate and X1; X1's falling limb "2 pt or more below the peak AND paired Wilcoxon p < 0.05" | The SUBJECT as the "realization" | As fixed by Enam | No | FIXED | - |
| D6-3 | Divergence: best-epoch validation loss above twice the lambda = 0 arm of the same seed, or NaN | Not defined in code | Implemented (ratio clause for `adv_marginal` only, the family with a zero arm); a diverged arm is flagged, never dropped | No | FIXED | - |
| D6-4 | G-PASS, G-WEAK, G-FAIL; X1 to X4 | G-FAIL whenever the domain-probe fall was under 2 pt | G-FAIL only when BOTH meters fail: domain-probe fall < 2 pt AND no significant Page trend on the unseen-subject probe; a domain fall < 2 pt with a significant unseen-subject trend is G-WEAK | No | FIXED | Ruling 1: the plan's G-WEAK explicitly covers "only one of the two meters moves" |
| D6-5 | SFC "embedding norm flat" | Undefined | Stop condition is the normalisation itself: the L2-normalised embedding the CORAL term sees must be unit norm, max abs(norm - 1) < 1e-5 over every fold, weight and batch (measured in the training loop, no RNG draw, no effect on the loss; new columns `coral_embed_normdev` and `coral_embed_normdev_max`, l2 path only). Failing it is the broken-normalisation stop. The raw penultimate norm across weights is reported descriptively, not a stop condition; the 1.10 ratio is dropped | No | FIXED | Ruling 2 |
| D6-6 | Stage-2 rows and the outcome gate | Rows lacked `--instrument`/`expected_outputs`; no outcome gate | Generator emits instrumented rows and a `d6_outcome_check` light pseudo-row | No | FIXED | - |
| D6-7 | ADV-C, CDAN, mechanism and secondary contrasts; the divergence retry with `--grad-clip 5.0` | Not implemented | Built: `kc23_d6_advc_job_gen.py` (collapse point = smallest lambda_max ABOVE the ADV peak whose realization-mean F1 is 2 pt or more below it; ADV-C rows at that value and the next up, seeds 42, 7, 123, oracle target labels, oracle=True checked on every output row; no collapse: ADV-C not run and reported), `kc23_d6_cdan_job_gen.py` (only after C-M1), `kc23_d6_retry_job_gen.py` (one retry with `--grad-clip 5.0` into `<arm>__retry`; aggregate refuses to run while a diverged arm has no retry, substitutes a completed retry, an arm that diverges again stays diverged and flagged), `--require mechanism` and `--require secondary` in the aggregator, mechanism letters C-M1/C-M2/C-M3/C-NOT-RUN and the secondary table in `kc23_d6_stats.py`, the combined through-line matrix in the outcome verdict, `--within-class-probe` (ADV runner) and `--grad-clip` (Deep CORAL runner), both inert when absent | No | FIXED | Build ordered (item B) |
| D6-8 | The mechanism test needs "ADV-C's within-class subject invariance" | No within-class measure existed | `subject_probe_within_class_bacc` (the subject probe inside each movement class, averaged over classes) written by `--within-class-probe` on the D0.3 probes; invariance = 1 - that probe; the ADV Stage 1 and Stage 2 rows carry the flag | No | FIXED | Build ordered (item B) |
| D6-9 | Plan 6.7 output: "a combined through-line matrix: axis x {invariance measured, shape shown, mechanism measured, replicated on ENABL3S where run}" | Not produced | The outcome verdict ends with the matrix; each cell lists the letters the stage that bears on it produced (C2, C6, D4, D5, D6) or says "not available"; it makes no judgment of its own | No | FIXED | Build ordered (item B) |

### KC-S3

| ID | Plan clause | Code before | Code after | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|---|
| S3-1 | "Inventory ... run only the missing cells" | Inventory saw only SVM and RF global; `results_aonly_persubj`, `results_aonly_resnet_se_cd_persubj` and `results_aonly_ensemble` already existed, and the builder would have rerun ResNet-SE+CD per-subject (hours of GPU) | Inventory reads the existing directories; the duplicate ResNet-SE+CD per-subject row is removed. `s3_svm_per_subject` (a duplicate of `results_aonly_persubj`) had already run and stays as a reproduction check | No verdict | FIXED | - |
| S3-2 | "Include the DNS to WAK critical-error rate for each cell" | Not produced; LDA runner saved no predictions; CNN rows saved no probabilities | `run_lda_loso.py --save-preds` (inert, proven identical), CNN rows `--save-proba`, LDA rows rerun into their original names (old directories kept as `__NOPREDS`) | Old LDA results existed, no verdict | FIXED | - |
| S3-3 | "A single benchmark table `benchmark_active_only.csv` and `S3_VERDICT.md`, reporting any change in the class hierarchy" | Neither existed | `kc23_s3_benchmark.py` (fail-closed on any missing cell) and an `s3_benchmark` light row | No | FIXED | - |
| S3-4 | "plus SVM-X and HGB if KC-C3 lands P2 or P3" | Not wired | `kc23_s3_extra_job_gen.py` appends the four SVM-X and HGB cells (global and per-subject, `--flush-preds`) only on P2 or P3 and makes the benchmark wait for them; P-OUT: not required, and the S3 verdict says so | No | FIXED | Ruling B: if C3 lands P-OUT the extra cells are not required |

## Cross-cutting

| ID | Plan clause | Code | Data existed? | Status | Enam ruling, 26 Sep |
|---|---|---|---|---|---|
| N-1 | Every stage: "New paired tests go to `kc23_new_tests.csv`" (the plan's report-back sections) | No gate or aggregator writes that file; every stage writes its own tables | Yes for some stages | OPEN | Not yet ruled: reported to Enam |

## Rulings of 26 September 2026 (Enam), as recorded

1. D6 G-FAIL against G-WEAK: G-FAIL only when both meters fail (domain-probe fall < 2 pt AND no significant Page trend on the
   unseen-subject probe); a domain fall < 2 pt with a significant unseen-subject trend is G-WEAK. Code changed (D6-4).
2. SFC embedding norm: the unit norm of the L2-normalised embedding the CORAL term sees is the check (max abs(norm - 1) < 1e-5);
   failing it is the broken-normalisation stop; the raw penultimate norm is reported descriptively; the 1.10 ratio is dropped.
   Code changed (D6-5).
3. C4 "normalization gain present": positive and Holm-significant across the two sets. Confirmed (C4-4).
4. C5: the plan's premise was wrong (`eval_sd` loops over subjects, so the published Table 4.2 classical SD figures are per-subject
   models; the thesis caption is text item TA10). per_subject is primary for every arm; L3 like for like; the SVM's per-subject g*
   for the CNN; W-B1 and the pooled arms dropped; pooled C5 output on disk marked SUPERSEDED. Pre-data design correction
   (C5-2, C5-7, C5-9, C5-10). Arm list: P50, P0, B-1, B-2, B-4, B-8, B-16, I-1, I-4, I-16.
5. D1 realizations: seed 2026 added to every Tier A arm R1 to R12 (12 GPU rows, queued after the current Tier A seeds); Tier B stays
   at 3; the published runs are a sensitivity analysis only (D1-3).
6. D1 BH family of 17 tests: confirmed (D1-2).
7. S1 ensemble tolerance: confirmed for now; to be re-evaluated against the D1 realization SD band once measured (S1-3).
8. S2 F-X definition: confirmed (S2-2).
9. C1: confirmed; both readings recorded (C1-1).
10. D1-7, no `run_config.json` for R10 and R11: accepted; the queue log holds their exact commands (D1-7).

Other instructions of the same message: B, the four OPEN items built (C3-7, D6-7 with D6-8 and D6-9, S3-4; D1-5 by ruling C);
C, statsmodels only in `06_Code/.venv_stats` (D1-5); D, inertness of scripts whose new flags were not re-run against the pre-change
version (below); E, the runner kill (below).

## Inertness of flags added after their outputs were produced (ruling D)

Each check runs the pre-change script (commit `d613a06`, the last commit before any of these flags) and the current script on the
same input with the flag absent, and compares the output files byte for byte.

| Script and flag | Check | Result |
|---|---|---|
| `train_classical_loso.py` `--svm-c-grid`, `--svm-gamma-mult-grid`, the multiplicative gamma grid | default path against git HEAD on a tiny synthetic fold (`test_kc23_c3_grid.py`) | per-subject F1 and best parameters identical, no sidecar written |
| `run_deep_coral_align_loso.py` `--instrument`, `--probe-cap`, `--grad-clip`, the unit-norm check | default path against `d613a06` (`test_kc23_d6_runners_instrument.py`, `test_kc23_d6_runners_mechanism_flags.py`) | subjectwise, alignment, training log and summary byte-identical; the l2 path gains only the new columns, F1 and every alignment measure unchanged |
| `run_adv_align_loso.py` `--instrument`, `--within-class-probe` | flag on against flag off, same run (`test_kc23_d6_runners_mechanism_flags.py`) | subjectwise, alignment, training log, occlusion, attenuation and permutation byte-identical; embed_probes gains two columns only |
| `run_lda_loso.py` `--save-preds` | flag on against flag off, same run (`test_kc23_s3_benchmark.py`) | `lda_subjectwise.csv` identical |
| `run_cnn_arch_loso.py` `instrument_fold(within_class_probe=...)` | default False; every D1, D3, D4, D5 row calls it without the flag | column absent when off (asserted through the ADV runner test) |
| `kc23_s2_predictions.py` `--window-ms`, `SPEC_250` / `SPEC_400` | 2 subjects (156, 185), default path, run against the existing 136,575-row output, and against the pre-change script | INTERRUPTED (laptop hibernation, 26 Sept). Subject 156, default path: metadata columns and every SVM prediction are identical to the existing output in all three conditions (0 of 14,031 rows differ); the ResNet-SE+CD rows differ (629, 1,036 and 714 rows in the transductive, causal-100 and balanced25 files) and so do the soft-vote rows that inherit them. The script diff against d613a06 touches only path plumbing (SPEC dicts and load_common arguments), so this is read as GPU training nondeterminism, not as an effect of the flags; the control (the pre-change script on subject 156) that would confirm it was interrupted at about 90 min and is NOT done. Not byte-identical, and not proven to be nondeterminism |

## The runner kill (0xC000013A) and its hardening (ruling E)

**What happened.** At about 07:47 on 26 September the runner (pid 27592, started 00:35) exited with 0xC000013A
(STATUS_CONTROL_C_EXIT) and its running GPU job (`d1_r2_s1001`) died with it; `c2_ladder_w250`, started under the earlier runner,
survived. The queue re-queued and restarted `d1_r2_s1001` (it resumes from its saved folds).

**Cause, from the evidence available.** The Task Scheduler operational history is disabled and there are no System, Application,
Winlogon, session-manager or Kernel-Power events between 07:30 and 07:56, so there is no logged record of the event. What the
process list does show: when the scheduled task started the replacement runner at 07:52:39, Windows Terminal started with it
(`WindowsTerminal.exe`, `OpenConsole.exe` and `conhost.exe` all created at 07:52:39, the runner's `cmd.exe` and `python.exe` at
07:52:39 and 07:52:40). The task ran the runner under `cmd.exe /c`, Task Scheduler gave it a console, and the default terminal
application hosts that console in a Windows Terminal window. 0xC000013A is the exit code a console process gets when its console
window is closed (or on Ctrl+C), and every job the runner launched shared that console and its process group, which is why the
running job died too. The likeliest event is a closure of that terminal window; I cannot show who or what closed it. The
runner that is running now was in the same position until the hand-off below.

**Hardening.**
1. Jobs start with `spawn_detached` (`kc23_queue.py`): `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`, no stdin. A job has its own
   hidden console and process group, so nothing aimed at the runner's console can reach it.
2. The scheduled task runs `.venv\Scripts\pythonw.exe -u kc23_queue.py --redirect-output` (`kc23_queue_task.ps1`): no console, so no
   window to close and no Windows Terminal; output goes to `_run_logs/kc23/queue_{stdout,stderr}.log`; Ctrl+C and Ctrl+Break are
   ignored. A deliberate stop is a process kill, which the job hand-off survives (`running.json` adoption).
3. Tested (`test_kc23_queue_hardening.py`, real console windows): closing the console window of a runner whose job shares it kills both
   (the control, reproducing the failure); closing the console window of a runner started with the hardened launcher kills the
   runner but leaves the job running; a console-less `pythonw` runner and its job survive closing a terminal that shows the log.
   Ctrl+C and Ctrl+Break are ignored under `--redirect-output` and still stop a runner started by hand without it.


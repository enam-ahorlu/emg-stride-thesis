# Claude Code prompts for the kill-critic programme of 23 September 2026

Paste these into Claude Code, run from `C:\Users\enama\OneDrive\Desktop\Documents\MSc CS\FInal Project`, **one at a time, in order.**

- Prompts 1, 1b, 2 and 5 are one-off. Prompt 1b ends by carrying out Prompt 2 if everything passes.
- Prompt 3 is the re-entry prompt: use it whenever a session ends, the machine sleeps, or you want a status check.
- Prompt 4 is the template for answering an escalation.

Each prompt is self-contained, because each Claude Code session starts with no memory of the last.

---

## Prompt 1. Orientation, the free stages, and all code changes (Phases 0 and 1)

```
You are running the kill-critic programme of 23 September 2026 for my MSc thesis. Work from the project root.

Read these files fully, in this order, before doing anything:
1. 06_Code/RUN_ORDER_KC23.md (the dispatcher and the standing rules; they bind every step)
2. 06_Code/EXPERIMENT_PLAN_KC23_CLASSICAL.md
3. 06_Code/EXPERIMENT_PLAN_KC23_DEEP.md
4. 06_Code/EXPERIMENT_PLAN_KC23_DEPLOYMENT.md
5. 04_Reviews_and_QA/KILL_CRITIC_2026-09-23.md (why each stage exists)
6. CLAUDE.md, 06_Code/_STRUCTURE.md and 06_Code/REPRODUCE.md (environment, the flat layout that must not be reorganised, and the exact published commands)

Then do Phases 0 and 1 of RUN_ORDER_KC23.md Section 3, items 1 to 5:
- KC-C7 verifications V1 to V7, written to 06_Code/KC23_VERIFICATIONS.md.
- KC-C1 nested selection audit, with its verdict letter.
- KC-D0: capture the "before" state FIRST, then make the two augmentation modes and the instrumentation additions, then run both inertness assertions. Include D0.5 for KC-D6: write run_adv_align_loso.py (plan Section 6.3) and add --coral-normalize to run_deep_coral_align_loso.py with its byte-identity assertion on the none path.
- The classical code changes for KC-C2, KC-C3, KC-C5 and the KC-S2 adapter extension, each with its own inertness proof as its plan specifies.
- Build kc23_queue.py per RUN_ORDER_KC23.md Section 2, write kc23_jobs_gpu.csv and kc23_jobs_cpu.csv covering items 6 to 15 (including 11b and 11c for KC-D6, with 11c's jobs generated only after the D6 manipulation gate decides which families pass) and C-a to C-e in the dispatcher's order, with depends_on and gate_script filled, and smoke-test the runner on two --heldout 1 jobs.

Rules that matter most here:
- Use 06_Code/.venv/Scripts/python.exe.
- Never edit anything in 01_Thesis/. Never touch recompute_unified_fdr_v9.py or its outputs. Never change the deep model of record.
- Every code change is additive, and it is proven byte-identical on existing paths before anything new runs. If any inertness check fails, STOP and report. Do not try to "fix forward".
- Encode each stage's outcome grid in its stats script so the rule cannot drift.
- Git commits are authored by me alone. No Co-Authored-By or Claude-Session trailers. Commit the code changes and the verification files in logical units. Do not push.
- No em dashes in any file you write.

Do NOT launch the GPU queue in this session. Stop after Phase 1 and report, in this order:
(1) the C1 letter and optimism figure;
(2) V1 to V7 findings;
(3) every inertness result with its evidence;
(4) the queue files, with job counts and estimated hours;
(5) anything in the plans you found ambiguous or wrong in practice, with the resolution you propose. Do not silently change a design.
Then tick the Phase 0 and 1 items in KC23_TRACKLIST.md Part E.
```

---

## Prompt 1b. Close Phase 1, record the C1 decision, fix every rule in code before any data exists, then launch

Written 23 September 2026 after Prompt 1's report. Replace the bracketed decision if you choose differently.

```
Continue the kill-critic programme of 23 September 2026. Read 06_Code/RUN_ORDER_KC23.md, 06_Code/KC23_PHASE1_REPORT.md, 06_Code/KC23_VERIFICATIONS.md, 06_Code/results_kc23_c1_nested_selection/C1_VERDICT.md and the three EXPERIMENT_PLAN_KC23_*.md files before acting. Same standing rules as before; in particular cap CPU workers at (logical cores minus 2) and run at most one CPU-heavy job at a time. Last session broke this; do not repeat it.

1. Record the KC-C1 escalation and my decision. The halt protocol was not followed last time: no KC23_HALT.md was written for letter E. Create it now with the C1 entry (letter, numbers, affected passages: 3.4.1, 4.4.1, 4.8, 5.3, Table 4.10, Table A.9) and record my decision under it, dated today:
   [DECISION D-6a: Keep the soft vote as the headline combiner. The nested reselection gives 85.82% against the published 85.80%, so selection adds no optimism; the escalation fired because the max-F1 rule picks stacking in 40/40 folds, where the thesis kept the soft vote as the simpler rule within 0.2 pt. Add stacking (SVM + ResNet-SE+CD, logistic-regression meta-learner fit on the other 39 subjects, as in the published stacking) as a DESCRIPTIVE row in KC-D1's per-seed ensemble computation beside C13, so its edge over the soft vote is read against run variance. No other change.]
   Add the stacking row to the D1 stats script, and note the owner decision in the status header of EXPERIMENT_PLAN_KC23_DEEP.md (the plan text stays unedited).

2. Append to KC23_VERIFICATIONS.md, under V1: the 74.8% of thesis Appendix B.7 and B.8 is the causal DECISION-route SVM with the buffer excluded (0.7476; see the run_buffer_composition.py docstring, which gates against rescore_streaming_buffer_v2.py at 0.7476). Confirm that from the files. So the like-for-like causal costs are -2.9 (decision route) and -4.49 (probability route). Also note that the V5 gap was a numbering slip in the plan: no V5 was intended.

3. Finish the KC-C3 inertness: subjects 2 and 3 under per-subject normalization, and subjects 1 to 3 under global normalization, --grid default, exact match to results_loso_freq_persubj and the published global run (f1_macro and best_params). Commit the evidence. If any mismatch, STOP.

4. Approved design addition: extend b8_cnn_sd.py with the same --scheme / --guard-windows / --n-chunks flags as b8_movement_blocked_sd.py, legacy call shape byte-identical. Prove inertness on CPU (GPU is cross-process nondeterministic here): run the legacy path before and after the edit on 2 subjects in the same CPU setting and match every output digit. Commit.

5. Write EVERY missing script now, before any KC23 result exists, so each decision rule is fixed before data:
   - Stage stats/gate scripts: C2, C3, C4, C5, C6, D1 (with the reproduction gate and the headline gate), D2, D3, D4, D5, D6 (sanity gate, per-family manipulation gate, outcome letters, the combined through-line matrix), S1, S2 (the F0 gate plus the analysis), S3 (the inventory).
   - Runners: run_scripted_supervised.py, with its reproduction gate against the published balanced25 arm; kc23_c4_extract_rich.py, with a unit test on a synthetic sinusoid of known moments.
   Each gate script prints its letter and exits 0 (continue), 10 (report, continue) or 20 (ESCALATE), exactly per its plan's outcome table. For every gate script, write tests (tests_kc23/) on SYNTHETIC inputs constructed to hit every letter in its table, and run them all green. Then commit all scripts and tests in one commit whose message says "written and tested before any KC23 result exists", and record that commit hash in KC23_STATUS.md as the pre-registration record.

6. Fix the halt protocol in kc23_queue.py: any gate exit 20 must itself append the escalation to KC23_HALT.md and KC23_STATUS.md and hold dependent jobs. Test with a synthetic gate that returns 20 on a smoke job, then restore.

7. Regenerate kc23_jobs_gpu.csv and kc23_jobs_cpu.csv, and confirm programmatically that every script, input file and depends_on reference now exists.

8. If items 1 to 7 all pass, carry out Prompt 2 of 06_Code/KC23_CLAUDE_CODE_PROMPTS.md exactly (launch the queue detached, confirm the first GPU and CPU jobs are running, report, stop). If anything fails, STOP and report instead. Do not launch on a partial pass.

Report in this order: any STOP; the C1 decision as recorded; the 74.8 confirmation; C3 inertness; b8_cnn_sd inertness; the list of scripts written with their test counts and the pre-registration commit hash; the halt-protocol fix; the queue launch state.
```

---

## Prompt 1c. KC-S1 gate failure: fix the runner, and audit every runner for stubs (24 September 2026)

```
Decision on the KC-S1 reproduction-gate failure: yes, fix it, and widen the fix. Do not interrupt the jobs now running (D1 seed 7, C3).

1. Diagnosis, confirmed from the files. Your 0.6914 equals the published calib25 SVM exactly (results_causal_ensemble/report.csv: calib25, SVM, f1_excl_mean 0.6914). stage_l0_svm normalised from the first 25 chronological windows, not from 25 windows of each movement. The correct gate targets are the published balanced25 rows of results_buffer_composition/buffer_composition_summary.csv (f1_excl_mean): SVM (decision route) 0.7472, SVM_PROBA 0.7286, soft ensemble 0.8152. Per-subject values are in buffer_composition_subjectwise.csv. Mark the three failed s1_base runs invalid with an INVALID.md inside each directory (reason: wrong buffer). Do not delete anything.

2. run_scripted_supervised.py is a stub. Its "full" stage prints "not implemented" and exits 0, and the gate compares the published ensemble figure with itself when the ensemble file is missing. Implement it fully, per EXPERIMENT_PLAN_KC23_DEPLOYMENT.md S1.2 and S1.3:
   - B_K = the first K windows of each movement's own recording, for K in {5, 10, 25}, via balanced_buffer_indices. The same B_K both normalises the held-out subject (causal) and is excluded from scoring. Every K is scored on the windows outside B_25.
   - Arms: L0, L1, L2, S-pool, S-only, S-ft, S-ens1, S-ens2. Train the ResNet-SE+CD base ONCE per fold per seed, and score L1 and every S-ft from that same base model.
   - Import from run_buffer_composition.py (balanced25 construction, the SVM and SVM_PROBA fits, the CNN training), run_within_subject_baseline.py (regime-C pooling) and run_cnn_calibration_multidraw.py (the 3-epoch fine-tune, ft-lr 5e-4). Never copy code.
   - --seed controls the deep base training. Write per-subject, per-arm, per-K CSVs and probabilities.

3. The gate becomes three checks, and a missing value is a FAIL, never a fallback:
   - SVM decision route = 0.7472, and SVM_PROBA = 0.7286, both exact to 4 dp (deterministic);
   - soft ensemble within ±1.5 pt of 0.8152, or within the KC-D1 measured band if D1 has finished.

   Route the gate through the queue's gate_script, so that a FAIL exits 20, writes KC23_HALT.md and holds S1's dependents. Apply the same rule to every stage that has a reproduction gate: the check must live in a gate_script, not only in the job's own exit code.

   Before queueing, run an integration test of the gate path on 2 subjects and match the published per-subject values from buffer_composition_subjectwise.csv.

4. Audit every KC23 runner and every job row for the same failure. The rule: no code path may exit 0 without writing its declared outputs, and no gate may compare a value with itself or fall back to a published number. Specifically:
   - c5_simplecnn_sd is still a "NOT YET RUNNABLE" comment row, although b8_cnn_sd.py gained --scheme in commit 7e1be9a. Regenerate the row.
   - c3_ensemble is a "NEEDS A DATA-PREP STEP" comment row. Implement the step.
   - kc23_d6_stage2_job_gen.py is referenced but missing. Write it now, so the Stage 2 generation logic is fixed before Stage 1 data exists, and test it on synthetic gate outputs.
   - KC-S2 has no jobs that produce its inputs. The plan (S2.3, S2.4) needs ENABL3S LOSO per-window predictions with circuit and time, for the locked SVM, ResNet-SE+CD and soft ensemble, under transductive and causal (100-window and balanced25) normalisation, plus the S2b 400 ms ENABL3S windows and the 400 ms SVM and ResNet-SE+CD training. Add the scripts and job rows.
   - KC-S3 has only 4 GPU rows. Confirm from the kc23_s3_inventory output that every other cell already exists; add rows for any that do not.
   - Then report every other placeholder, comment row or unimplemented branch you find, and fix each one the plan specifies.

5. Integrity. These fixes change runners and job rows, never the decision rules. The thresholds in every kc23_*_stats.py and gate script stay exactly as committed in 0ba3575. If any fix seems to need a threshold change, STOP and ask me. Commit the fixes as "post-pre-registration implementation fixes; decision rules unchanged since 0ba3575", and record the hash in KC23_STATUS.md.

6. Re-queue KC-S1 (3 seeds, now GPU jobs). Regenerate and validate both job CSVs. Then report:
   - the S1 gate integration-test result;
   - the audit list, with each item fixed or flagged;
   - any threshold question;
   - the queue state.
```

---

## Prompt 1d. The four flagged items: fix all four now (24 September 2026)

```
On the four items you flagged: fix all four now. Do not wait until their stages come up. Plumbing fixed before data arrives is part of the pre-registration record; plumbing fixed after is not. Do not interrupt the running jobs.

1. KC-D6. Write kc23_d6_aggregate.py now, and correct the output-directory paths in the D6 sanity and manipulation gates. Test both against the REAL output layout, using results_kc23_d6_smoke_marginal, _classcond and _cdan, plus the D0.5 SFC smoke directories, as fixtures. Also test against synthetic variants built to hit every gate letter. The gates must fail closed (exit 20) on any missing file.

2. kc23_c3_tuning_stats.py. Fix the expected filenames to match what train_classical_loso.py and the C3 ensemble step actually write. Test against the real files in results_kc23_c3_before_persubj, _before_global and _after_persubj. Missing files must fail closed, never fall back.

3. LDA.
   - Every LDA row (the four c4_*_lda_* rows and the two S3 LDA rows) must call run_lda_loso.py, the script that produced the published LDA figures (68.7 per-subject, 62.8 global, Table 4.1). Do not add an LDA branch to train_classical_loso.py.
   - First prove run_lda_loso.py reproduces results_lda_persubj and results_lda_global exactly on subjects 1 to 3 (f1_macro and best params). Then confirm it accepts the new feature files C4 produces.
   - Separately, make train_classical_loso.py fail closed on any model name it does not implement: raise, never "else: continue". Prove the default path inert on subject 1.
   - Check whether any LDA row already "completed" with no output. If so, add INVALID.md to that directory and requeue the row.

4. KC-S2 transitions. Do not use a time-gap heuristic on the windowed data. Use the raw per-sample Mode signal, which adapt_external_dataset.py already reads (_build_labels), with the circuit id and per-circuit t_start the F0 extension added.
   - Write kc23_s2_transition_table.py. For each circuit CSV, find every sample where Mode changes. Keep a transition only when both sides are retained modes (LevelWalking 1, StairAscent 4, StairDescent 5) and the change is direct (sample i in mode a, sample i+1 in mode b), so any change through a dropped mode (ramps 2 and 3, standing 6, sitting 0) is excluded, per S2.2.
   - Write subject, circuit, t_change_s, from and to.
   - Validate: every published ENABL3S window lying wholly inside one segment of the table must carry that segment's class. Report any disagreement.
   - Feed the counts to the F0 gate. Then implement the S2.3 and S2.4 producing jobs and the S2b 400 ms arm, as in my previous message.

5. Close this class of bug in the runner. Add an expected_outputs column to both job CSVs (the summary file and the expected subject count, 40 or 10). kc23_queue.py marks a job FAILED if its declared outputs are missing or its subject count is wrong, whatever its exit code. Re-validate every row.

6. Two follow-ups on your audit fixes, which I accept.
   - The deferred C5 CNN generator for the B and I arms at the plateau guard: write it now, not later. The plateau rule (C5.4) is deterministic from the classical results, so commit the generator and test it on synthetic classical outputs before any C5 data exists.
   - Confirm that the rewritten run_scripted_supervised.py writes run_config.json recording:
     - the base training: epochs 40, patience 7, batch 512, chandrop 0.2 (not the calibration scripts' 25/5 defaults, which V6 showed produced the 81.8% run);
     - the fine-tune: 3 epochs, lr 5e-4;
     - K and seed.

7. Same integrity rule as before. The decision thresholds in every kc23_*_stats.py stay as committed in 0ba3575; if a fix seems to need a threshold change, STOP and ask me. Commit as "post-pre-registration implementation fixes (2); decision rules unchanged since 0ba3575", and record the hash in KC23_STATUS.md.

Report:
- the D6 and C3 fixture-test results;
- the LDA reproduction result, and any INVALID rows;
- the S2 transition counts per subject and type, the window-agreement check, and the F0 letter;
- the expected_outputs validation;
- the queue state.
```

---

## Prompt 1e. Make the fixes live safely, and run the D1 reproduction gate for real (24 September 2026)

```
Good work on 1d. Four things, in this order.

1. The KC-D1 reproduction gate never actually ran. The d1_reproduction_check job was marked "skipped(complete)" because its directory existed, and results_kc23_d1_repro_check/D1_VERDICT.md contains only its header: no letter. The inputs it wrote (R1 0.7706, R2 0.8344, R10 pre 0.7830) look like a pass against the ±1.5 pt bands, but a pass must come from the committed gate, not from my arithmetic.
   - Run the gate now, write its letter into D1_VERDICT.md, and record the letter in KC23_STATUS.md.
   - If it is anything but a pass, follow the halt protocol, because seed 7 is already running on top of it.
   - Then list every other row marked "skipped(complete)" and re-verify each against its real outputs. For each, report whether it is complete, and if not, what is missing.

2. Populate expected_outputs for EVERY runnable row, not only the ten done so far: the summary file plus the subject count (40 or 10). For gate, stats and aggregate rows, the expected output is the verdict or summary file containing a printed outcome letter. The skip rule must use expected_outputs, never the existence of a directory; that is exactly how the D1 gate got skipped. Re-validate every row.

3. Make the fixes live without killing a running job.
   - First inspect the process tree: are the runner's child jobs independent of PID 56768?
   - If they are, stop only the runner and start the new one. The new runner must recognise the in-flight job (from its log or PID) and must not relaunch it.
   - If they are not, wait for the current GPU job to finish, then stop the runner before it launches the next one, and restart.
   - Before restarting, check every job that has already completed since the first fix commit: did any use a script that was changed after that job started? List any that did, and mark them for rerun.

4. KC-S2: implement the producing jobs now, before any S2 data exists, to the same fail-closed standard:
   - S2.3: ENABL3S LOSO per-window predictions, with circuit and time, for the locked SVM, ResNet-SE+CD and soft ensemble, under transductive, causal 100-window and balanced25 normalisation;
   - S2.4: the analysis over the transition table you built;
   - S2b: the 400 ms ENABL3S windows, plus the 400 ms SVM and ResNet-SE+CD training.
   Their position in the queue is unchanged.

KC-D5 looks complete across its seeds. Once all 15 rows are verified by expected_outputs, run its stats and give me the letters.

Same integrity rule: thresholds unchanged since 0ba3575, and STOP and ask if any fix seems to need a threshold change. Commit as "post-pre-registration implementation fixes (3); decision rules unchanged since 0ba3575" and record the hash.

Report:
- the D1 gate letter;
- the re-verified skipped rows;
- the expected_outputs coverage (rows populated / runnable rows);
- the restart method and the new PID;
- any jobs marked for rerun;
- the S2 implementation;
- the D5 letters.
```

---

## Prompt 1f. Decisions on S1 and D5, fix the three defects, sweep for fail-open scripts, and make the queue survive restarts (25 September 2026)

```
Decisions and approvals, in order.

1. KC-S1, letter D-S. Record this under the S1 entry in KC23_HALT.md, dated today:
   [DECISION D-6b: Accept D-S. The label-free claim is restated at its true scope: offline (transductive) the pipeline needs no labels, but the causal deployment needs a commissioning buffer that covers all four movements, and if that buffer is scripted, its labels come at no extra cost to the user. Used with labels (fine-tuned ResNet-SE+CD plus pooled SVM, S-ens1), the same 25 windows per movement raise causal macro-F1 from 81.6% to 85.3% (3 seeds, +3.72 pt, 36/40 subjects). The thesis reports both configurations; the label-free figure is not dropped. No queue change.]
   Then complete the S1 verdict as the plan specifies:
   - the secondary K curve: the smallest K at which the best supervised arm reaches L0;
   - S-ft against L1 at each K;
   - confirmation that S-pool and S-only are deterministic classical fits, which would explain why they are identical across seeds.
   Write it all into results_kc23_s1_gate/S1_VERDICT.md, starting with the letter.

2. KC-D5 directions, decisions A and B. The plan's E-R rule says "a finding's direction", which means the direction the thesis states for that finding on SIAT-LLMD. Do not choose a sign from the ENABL3S numbers.
   [DECISION D-6c:
   - A: gain jitter against channel dropout on ResNet-SE. The thesis's stated direction (Section 4.3.3) is that gain jitter is AHEAD (positive), so encode positive. On your descriptive numbers (+0.55 pt, 4 of 10 subjects) that is E-N.
   - B: permutation reliance. The direction of the reliance claim is a REDUCTION under channel dropout (positive), so encode reduction as positive, without waiting for D2. On your numbers (-0.23 pp, 4 of 10) that is E-N.
   Both choices take the reading that is less favourable to the thesis. Record that in the commit message.]
   - For occlusion, report the magnitude beside the letter: 1.67x on ENABL3S against about 6x on SIAT. The thesis may say the direction replicates, never "six-fold" on ENABL3S.
   - Report ResNet-SE+CD against the SVM on ENABL3S (0.6487 ± 0.0082 against 0.657) as a number with its across-seed SD, as the plan says.

3. Yes to the three defect fixes, in one commit with the usual wording:
   - wire the S1 gate as the gate_script of the s1 rows;
   - write a fail-closed D5 aggregator and stats script, reusing KC-D1's exact C17 occlusion definition and the D-6c directions;
   - give s2_transitions real expected_outputs, and make its placeholder branch exit non-zero.
   Then rerun D5 stats and s2_transitions through the queue, not by hand, so the halt protocol is exercised.

4. The fail-open pattern has now appeared in S1, D5 and S2, so sweep for it. For EVERY kc23_*_stats.py, gate, aggregate and analysis script, add a test that runs it with each required input missing, and asserts a non-zero exit and no verdict file containing a letter. Fix every script that fails its test. Report the list: script, fail-open (yes or no), fixed.

5. Yes to Task Scheduler.
   - Register the runner so it restarts after an app restart, a crash or a reboot, with a lock file so two runners can never run at once.
   - The runner resumes by expected_outputs, as in 1e.
   - Also stop the machine sleeping while the queue has work: a powercfg setting you can restore afterwards. Record the exact settings you changed, so I can revert them when the programme ends.
   - Do the switch at a safe boundary, without killing the running GPU job.

6. Commit the uncommitted KC23_HALT.md entry and results_kc23_s1_gate outputs.

Integrity as always: thresholds unchanged since 0ba3575. The D-6c directions make explicit what the plan text already fixed; they do not change a threshold.

Report:
- the S1 verdict (letter, K curve, S-ft against L1);
- the D5 letters, with the occlusion magnitude and ResNet-SE+CD against the SVM;
- the fail-open sweep table;
- the Task Scheduler and powercfg changes;
- the queue state.
```

---

## Prompt 1g. Build every missing aggregator, and bring every gate into line with the plan text before its data exists (26 September 2026)

```
Yes: build all of them now, before their data exists. But first read this, because I checked kc23_d4_invariance_stats.py and found a deeper problem than missing aggregators.

The real pre-registration is the plan text of 23 September (EXPERIMENT_PLAN_KC23_*.md), not commit 0ba3575. "Thresholds unchanged since 0ba3575" only proves the code has not drifted from itself. It does not prove the code matches the plan. D4's classify_t does not match it:
- The plan's T1 needs "within-fold F1 tracks class silhouette (mean Spearman > 0, sign test)". The code uses one Spearman over the 5 dose means (n = 5, the exact weakness the B3 audit removed from the thesis) and has no sign test.
- "Not the subject probe" is coded as an invented cutoff (rho > 0.3), which is not in the plan.
- "F1 peaks and then falls" is coded as a 2.0 pt gap with no test. The 2.0 pt figure belongs to D6's X1, not D4. has_interior_peak is computed and never used, and it counts a peak at the lowest dose as interior.
- The Page trend uses realizations (n = 3) as blocks. The plan measures per fold, so the blocks are the 40 folds, realization-averaged. And the plan says Holm < 0.05; the code uses a raw 0.05.
- Any case outside the grid silently defaults to T3.
D6 has the same pattern: the Page blocks are not folds, there is no Holm correction, and X1's falling limb lacks the plan's "paired, significant".

So, in this order:

1. PLAN CONFORMANCE AUDIT. For every gate and classify function (C2 to C6, D1 to D6, S1 to S3, and the D1 reproduction and headline gates), compare the code clause by clause against its plan's outcome table. Write 06_Code/KC23_PREREG_CONFORMANCE.md with one row per clause: plan text, code before, code after, and whether that stage's data already existed when the fix was made.
   - Where a stage's data does NOT exist yet (D3, D4, D6, C3 to C6, S3, and D2's analysis layer), correct the code to match the plan exactly. Where the plan's wording is verbal ("peaks and then falls", "does most of"), use the operationalisations in item 2, which I am fixing now, before any of that data exists.
   - Where data and a verdict ALREADY exist (C1, D1 reproduction, S1, D5), do not rewrite history. Report any non-conformance and give both readings, plan-rule and code-rule. The plan governs.
   - This supersedes "thresholds unchanged since 0ba3575". The new rule is: code conforms to the 23 September plan text, and every conformance change is committed before its stage's data exists.

2. Operationalisations, fixed now:
   - D4 T1:
     - (a) The unseen-subject probe falls with mpchandrop dose: a Page trend with the 40 folds as blocks (realization-averaged) and the 5 doses as treatments, Holm-adjusted across the two invariance meters (subject probe and permutation reliance), using the subject-probe p.
     - (b) F1 peaks and then falls: the argmax of realization-mean F1 is not the highest dose, and F1 at the highest dose is below the peak by a paired Wilcoxon over the 40 realization-averaged subjects, p < 0.05.
     - (c) F1 tracks class information: per fold, Spearman across the 5 doses between realization-averaged F1 and class silhouette; sign test over the 40 folds, positive, p < 0.05.
     - (d) F1 does not track invariance: the same per-fold Spearman between F1 and the negated subject probe is NOT significantly positive by the sign test.
     - T1 = a, b, c and d. T2 = not a. T3 = a and b, but not c.
     - Any other combination gets the new label "T-OUT (outside the pre-registered grid)", exits 10, and is reported. Never default it into T3.
     - Gain-jitter boundary: argmax of realization-mean F1 over GJ SD {0.30, 0.40, 0.50, 0.80, 1.00} not at 1.00, and F1(1.00) below that peak by a paired Wilcoxon p < 0.05.
   - D6: Page tests use the 40 folds as blocks, with Holm across the two meters, for both the manipulation gate and X1. X1's falling limb is ">= 2 pts below the peak AND a paired Wilcoxon p < 0.05", as the plan says.
   - C6 R1, "centering does most of the linear-probe work": (probe at rung 0 minus probe at rung 1) divided by (probe at rung 0 minus probe at rung 3) >= 0.5, on ENABL3S.

3. Then build the missing aggregators, each with real-format and synthetic fixture tests, each failing closed:
   - D2, per plan D2.2. Reuse KC-D1's exact C17 per-subject summed-drop definition, with no clipping of negative drops. Compute the reduction factor per realization as mean(R1 sum) / mean(arm sum), giving mean and SD across realizations, plus a paired Wilcoxon on realization-averaged per-subject sums. Three measures: zeroing occlusion, attenuation at alpha 0.5, and permutation (mean over R = 5).
   - D3, per plan D3.3: realization means of X1 to X4 against R1 and R3 on shared seeds.
   - D4, per item 2, reading embed_probes.csv, permutation.csv and the F1 summaries.
   - C6: the geometry rows (MMD, W1, the three probes pooled and within-movement, silhouette) come from analyze_between_subject_variance.py on the ENABL3S features through the C2 environment override, including rungs 4lw and 4o, plus the SVM LOSO F1 per rung.
   - D1's unproduced contrasts: C10 and C11 from R11 and R10; C13 and C13b from the per-seed soft vote and stacking over the saved R2 probabilities and the SVM probabilities; C15 from R8 and R9; C16 from the Tier B arms; C17 from the instrumented occlusion. For the headline gate, "global" = R10's f1_pre_adabn per seed, and 400 ms = R12.
   - S2b: 250 against 400 ms accuracy and decision delay, per plan S2.4.

4. KC-S2 reading. On ENABL3S a real session starts in the opening activity, so a chronological first-100 buffer is not class-covering. Unlike SIAT, where each movement's own clock interleaves them, it collapses. Lead the S2 verdict with the transductive and balanced25 conditions. Report the mixed100 collapse separately, as a finding in its own right: a buffer taken from the start of a real continuous session does not work, which supports the need for a scripted commissioning step (decision D-6b).

5. Commit the conformance table and the corrected gates first, as "pre-registration conformance: gates aligned to the 23 September plan text before their data exists". Then commit the aggregators. Record both hashes in KC23_STATUS.md.

Report:
- the conformance table summary: how many clauses were non-conformant, split into fixed-before-data and already-had-data;
- any already-existing verdict whose reading changes under the plan rule;
- the aggregator test counts;
- the queue state.
```

---

## Prompt 1h. Rulings on the FLAG items, a premise correction for C5, and the remaining open builds (26 September 2026)

```
Good work on the conformance audit. Rulings on the FLAG items, one correction to a plan premise, and the remaining builds. Record every ruling in KC23_PREREG_CONFORMANCE.md (new column: "Enam ruling, 26 Sep") and commit before the affected data exists.

A. Rulings.
1. D6 G-FAIL vs G-WEAK overlap: the plan's G-WEAK explicitly covers "only one of the two meters moves". So G-FAIL only when BOTH meters fail: domain-probe fall < 2 pt AND no significant Page trend on the unseen-subject probe. If the domain probe falls < 2 pt but the unseen-subject trend is significant, that is G-WEAK.
2. SFC "embedding norm flat": "flat by construction" refers to the embedding the CORAL term sees, the L2-normalised one. Check that it is unit norm (max |norm - 1| < 1e-5); failing that is the broken-normalisation stop. Report the raw penultimate norm across weights descriptively. It is not a stop condition, because once the loss is scale-free, raw scale no longer lowers it. Drop the 1.10 ratio.
3. C4 "normalization gain present": confirmed as positive and Holm-significant across the two sets.
4. C5: the plan's premise was wrong, and your FLAG found it. I checked b8_movement_blocked_sd.py: eval_sd loops over subjects, so the published Table 4.2 classical SD figures are PER-SUBJECT models under both the random and the blocked split. The thesis caption is what is wrong; that is now text item TA10. So:
   - make per_subject the PRIMARY cv-unit for every C5 arm (P50, P0, B-g for g in {1, 2, 4, 8, 16}, I-g), on SIAT and ENABL3S;
   - evaluate the L3 clause like for like: per-subject B-g* against the published per-subject blocked SD;
   - use the SVM's per-subject g* for the CNN;
   - drop W-B1 and the pooled arms (their only purpose was a naming question that is now settled by reading the code);
   - mark any pooled C5 output that already exists as SUPERSEDED.md.
   C5 has no data yet, so this is a pre-data design correction. Record it as such.
5. D1 realizations: add seed 2026 to every Tier A arm (R1 to R12, about 8 GPU hours), so every contrast has 5 realizations from the same code era and "4 of 5" applies uniformly. The published runs become a sensitivity analysis only, never the fifth realization. Tier B stays at 3. Queue the seed-2026 rows after the current Tier A seeds.
6. D1 BH family of 17 tests: confirmed.
7. S1 ensemble tolerance: confirmed for now. When the D1 realization SD is measured, re-evaluate the S1 check against that band and record the result (0.8141 against 0.8152 will pass any plausible band).
8. S2 F-X definition: confirmed.
9. C1: confirmed. Record both readings: nested 85.82% against the published soft-vote headline 85.80%, and against the table maximum 86.04% (stacking).
10. D1-7, no run_config.json for R10 and R11: accepted. The queue log holds their exact commands; state that.

B. Build the four OPEN items now, before their data exists, to the usual standard (fixture tests, fail-closed):
   - the C3 edge-rule rerun generator;
   - D6 ADV-C, CDAN (conditional on C-M1), the mechanism and secondary contrasts, and the divergence retry with --grad-clip 5.0, before D6 Stage 1 finishes;
   - the S3 generator for the SVM-X and HGB cells, conditional on C3 landing P2 or P3. If C3 lands P-OUT, the extra cells are not required; state that in the S3 verdict.

C. statsmodels: do NOT install it into 06_Code/.venv, which stays frozen for reproduction. Create a separate 06_Code/.venv_stats with statsmodels plus the minimum it needs, used only by the D1 MixedLM secondary. Record the versions in KC23_STATUS_NOTES.md.

D. Inertness you have not yet proven: kc23_s2_predictions.py gained flags after its 136,575-row output was produced. Re-run its default path on 2 subjects and show the output is byte-identical to the existing file. The same applies to any other script whose new flags were not re-run against HEAD.

E. The runner kill (0xC000013A, a console close or Ctrl+C) needs a cause and a fix:
   - Look for the cause in the Task Scheduler history and the Windows event log around the exit time, and report what you find.
   - Harden the runner whatever the cause. It must run with no console (pythonw.exe or CREATE_NO_WINDOW), and its children must launch in their own process group, detached, so a console close cannot signal them. Test by closing a terminal that shows the runner's log: the runner and its jobs must survive.

F. Commit, authored by me, no trailers. Report:
   - the rulings as recorded;
   - the C5 arm list after the change;
   - the seed-2026 rows added and the new GPU total;
   - the four builds with test counts;
   - the .venv_stats versions;
   - the S2 inertness result;
   - the kill cause and the hardening test;
   - the queue state.
```

---

## Prompt 2. Launch and supervise the queue (Phases 2 to 5)

```
Continue the kill-critic programme of 23 September 2026. Read 06_Code/RUN_ORDER_KC23.md, 06_Code/KC23_STATUS.md (if present), 06_Code/KC23_VERIFICATIONS.md, and the three EXPERIMENT_PLAN_KC23_*.md files before acting.

Phase 1 is complete and I have reviewed its report. Launch kc23_queue.py detached (PowerShell Start-Process with -PassThru, PID written to 06_Code/_run_logs/kc23/queue.pid) so it survives this session ending. Then supervise:

- After the KC-D1 seed-42 re-run finishes, run the D1 reproduction gate yourself as well as through the runner. If it fails, confirm the queue halted the deep stages, write KC23_HALT.md, and report. Do not diagnose by changing code unless the diagnosis is a flag mismatch against REPRODUCE.md, which you may correct once, recording it.
- As each stage completes, run its stats script, write its *_VERDICT.md starting with the outcome letter, append its paired tests to kc23_new_tests.csv, and add the status header to its plan file, leaving the plan text unedited.
- On any ESCALATE letter: follow the halt protocol exactly, let independent stages continue, and put the escalation at the top of your report.
- CPU stages C-a to C-e run beside the GPU with worker counts capped per the standing rules.
- Record the measured wall-clock of every run in KC23_STATUS.md.

You cannot stay in this session for the days the queue needs. When you have launched it, confirmed the first GPU job is running, and confirmed the first CPU job is running, report:
(1) what is running;
(2) the queue order and the estimated finish per phase;
(3) how I can check progress (KC23_STATUS.md).
Then stop. I will re-enter with Prompt 3.
```

---

## Prompt 3. Re-entry and status (reusable)

```
Re-entering the kill-critic programme of 23 September 2026. Read 06_Code/RUN_ORDER_KC23.md (standing rules), 06_Code/KC23_STATUS.md, 06_Code/KC23_HALT.md if it exists, and every *_VERDICT.md written since the last status entry.

1. Check whether kc23_queue.py is still running (the PID in 06_Code/_run_logs/kc23/queue.pid). If it is not, and the queue is not finished, find out why from the logs. Restart it only if the cause was external (sleep, reboot, closed terminal). If a job failed on its own, report the error and do not restart that job.
2. For every job finished since the last entry: confirm completeness (40 or 10 subjects, no duplicate rows after any resume), run the stage's stats script if the stage is now complete, and write the verdict.
3. Report in this order:
   - new ESCALATEs;
   - new outcome letters;
   - gates passed or failed;
   - the jobs finished, running and remaining, with measured against estimated hours;
   - anything odd.
4. Tick the finished items in KC23_TRACKLIST.md Part E.

Standing rules unchanged: no thesis edits, v9 and the model of record frozen, additive code only with inertness, commits authored by me alone with no trailers, no push, no em dashes.
```

---

## Prompt 4. Answering an escalation (template: fill the brackets)

```
Kill-critic programme of 23 September 2026: decision on escalation [STAGE, LETTER] from 06_Code/KC23_HALT.md.

My decision: [e.g. "keep the locked SVM as the classical member and report the extended-grid SVM as a sensitivity in a new appendix table" / "keep the version-of-record headline and add the realization mean" / "proceed with KC-S2 despite F-MARGINAL"].

Record this decision, with today's date, in KC23_HALT.md under the escalation and in KC23_STATUS.md. Re-enable any dependent jobs the decision allows, with no other change to the queue. Then continue exactly as in Prompt 3.
```

---

## Prompt 5. Consolidation (KC-F1 to KC-F4)

```
All stages of the kill-critic programme of 23 September 2026 are complete (confirm from 06_Code/KC23_STATUS.md before starting; stop and report if any are not). Read 06_Code/RUN_ORDER_KC23.md Section 4 and do KC-F1 to KC-F4 exactly as specified:

- recompute_unified_fdr_v10.py with both v10-add and v10-replace, the mapping table, and the list of every status change. Leave v9 untouched.
- KC23_NUMBERS_DELTA.md: every changed number, and every passage restating it, found by searching the pdftotext -layout of 01_Thesis/MSc Project Final Draft Restructured..pdf in all printed forms (fraction, percent, one and two decimals).
- Re-run build_run_manifest.py (every results_kc23_* dir must be verified on its core fields), update REPRODUCE.md and README.md, and draft RELEASE_NOTES_v1.3.0_DRAFT.md. Do not tag, push or mint anything.
- KC23_PROGRAMME_REPORT.md with the letter table, escalations, the measured run-variance band, the delta summary, and the stronger / narrowed / withdrawn claim lists.

Commit in logical units, authored by me alone, with no trailers. Then report the F1 family comparison with your recommendation, and stop.
```
